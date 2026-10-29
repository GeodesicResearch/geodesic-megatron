#!/usr/bin/env python3
"""Convert Megatron distributed checkpoints to HuggingFace format.

Converts Nemotron (Nano/Super) Megatron checkpoints to HuggingFace format,
with optional push to the HuggingFace Hub. Supports both single-process
conversion (default) and multi-GPU conversion for very large models.

The torch_dist checkpoint format supports resharding, so conversion is
independent of the training parallelism.

Usage:
    # Convert latest iteration (single process, no torchrun needed)
    python pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name

    # Convert specific iteration
    python pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name \
        --iteration 300

    # Convert and push to HuggingFace Hub
    python pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name \
        --iteration 300 \
        --push-to-hub

    # Multi-GPU fallback (if single-process OOMs)
    torchrun --nproc_per_node=8 pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name \
        --iteration 300 \
        --tp 1 --ep 8

    # An adapter-only PEFT checkpoint (see the README's export notes)
    torchrun --nproc_per_node=4 --nnodes=2 ... pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/lora_run --hf-model <release> --reasoning \
        --tp 1 --pp 2 --ep 4

    # Keep NemotronH remote-code for consumers without native support
    # (e.g. vLLM trust_remote_code on transformers < 5.3.0)
    python pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/experiment_name \
        --keep-remote-code \
        --remote-code-source /path/to/dir/with/modeling_nemotron_h.py

    # A LoRA (PEFT) checkpoint's adapter as a vLLM factor dir, no merge (single process, CPU)
    python pipeline_checkpoint_convert_hf.py \
        --megatron-path /path/to/checkpoints/lora_run \
        --format lora-factors \
        [--factors-path DIR] [--reference-alphas /path/to/rl_adapter/alphas.json]

LoRA factor dirs (``--format lora-factors``)
--------------------------------------------
The serving side (geodesic_utils.inference.vllm_extension; its docs/lora_factor_format.md) reads
``rank_<R>.safetensors`` holding ``{hf_name}.A`` / ``{hf_name}.B`` in bf16, plus an ``alphas.json`` mapping each
hf_name to alpha_over_r and ``__format_version__`` to the factor format, and applies
``delta = alpha_over_r * (B @ A)``. Format 2: every key a 2-D pair with HF-canonical rows. Format 3: format 2 plus
grouped routed-expert keys.

Two adapter layouts are read from the checkpoint's own tensors (DCP reassembles TP and EP shards at full logical
shape on CPU, so no model is built and no GPU is needed):

- plain ``LoRA`` on fused modules (the RL adapters):
  1.  ``decoder.layers.N.*`` -> ``backbone.layers.N.mixer.*``, per the NemotronH bridge's mapping table.
  2.  The fused ``linear_qkv`` is one Megatron module and three HF keys. Megatron interleaves its output rows by
      query group (for each of ``num_query_groups``, ``[heads_per_group*head_dim q | head_dim k | head_dim v]``),
      so q, k and v are strided; the row arithmetic comes from the run_config and is checked against the widths.
  3.  Mamba ``in_proj`` stores lora_B five-way split (``.z/.x/.B/.C/.dt``); HF's in_proj is their concatenation in
      that order (MambaInProjMapping).
  lora_A is shared across a module's HF keys, because ``delta = B @ A`` factorises the output dimension only.
- Tinker's layout (``CanonicalLoRA`` + ``GroupedExpertsLoRAAdapter``, "Expert LoRA"): separate q/k/v adapters,
  separate z and x adapters on in_proj (one HF in_proj key at rank 2r: A stacks both, B places each in its rows),
  the MoE latent projections (2-D keys), and per-expert LoRA on the routed experts as grouped keys
  ``mixer.experts.{up,down}_proj.weight`` (one factor shared, the other 3-D over experts).
"""

import argparse
import json
import os
import re
import shutil
import struct
from pathlib import Path

import torch
import yaml


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Base model -> instruct model mapping for chat template sourcing.
# Base models don't include a chat_template; the instruct variant does.
CHAT_TEMPLATE_SOURCE_MAP = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
}

# Token IDs in chat tokenizer that should be eos. The base release ships with
# eos_token_id=2 (`</s>`) only; chat-format SFT trains the model to emit
# `<|im_end|>` (id 11) at end-of-turn. Generation must stop on either id,
# otherwise the model runs to max_new_tokens.
CHAT_EOS_TOKEN_IDS = [2, 11]


def resolve_checkpoint_path(megatron_path: str, iteration: int | None = None) -> tuple[Path, int]:
    """Resolve the checkpoint iteration directory.

    Args:
        megatron_path: Top-level checkpoint directory containing iter_* subdirs.
        iteration: Specific iteration number, or None to use the latest.

    Returns:
        Tuple of (iteration directory path, iteration number).
    """
    base = Path(megatron_path)
    if not base.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {base}")

    if iteration is not None:
        iter_dir = base / f"iter_{iteration:07d}"
        if not iter_dir.exists():
            raise FileNotFoundError(f"Iteration directory not found: {iter_dir}")
        return iter_dir, iteration

    # Try latest_checkpointed_iteration.txt
    latest_file = base / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        iteration = int(latest_file.read_text().strip())
        iter_dir = base / f"iter_{iteration:07d}"
        if iter_dir.exists():
            return iter_dir, iteration

    # Fall back to scanning iter_* dirs
    iter_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("iter_")]
    if not iter_dirs:
        raise FileNotFoundError(f"No iter_* directories found in {base}")

    latest = max(iter_dirs, key=lambda d: int(d.name.replace("iter_", "")))
    iteration = int(latest.name.replace("iter_", ""))
    return latest, iteration


def detect_training_tokenizer(iter_path: Path) -> str | None:
    """Auto-detect the tokenizer the model was actually trained with.

    Reads `tokenizer.tokenizer_model` from `run_config.yaml`. This is the
    canonical source of truth for the chat_template to use at inference,
    because that's the template applied to training data via the bridge's
    `use_hf_tokenizer_chat_template` path. Without this, the converted HF
    dir can end up with a different chat_template than was used at training
    (the legacy fallback grafts from `CHAT_TEMPLATE_SOURCE_MAP`, which points
    at NVIDIA's upstream Instruct template — different in subtle ways from
    the Geodesic-published think/instruct variants).

    Args:
        iter_path: Path to a specific iteration directory.

    Returns:
        Training tokenizer HF model ID (e.g. ``geodesic-research/nemotron-
        instruct-tokenizer``), or None if not specified in run_config.yaml.
    """
    run_config = iter_path / "run_config.yaml"
    if not run_config.exists():
        return None
    with open(run_config) as f:
        config = yaml.safe_load(f)
    tokenizer_model = config.get("tokenizer", {}).get("tokenizer_model")
    if not tokenizer_model:
        return None
    return str(tokenizer_model)


def _is_multi_gpu() -> bool:
    """Check if running under torchrun (multi-GPU mode)."""
    return os.environ.get("WORLD_SIZE") is not None


def convert_single_process(
    iter_path: Path,
    hf_path: Path,
    hf_model_id: str,
    strict: bool = True,
    show_progress: bool = True,
) -> None:
    """Convert using single-process export_ckpt (CPU-based distributed context)."""
    from megatron.bridge import AutoBridge

    print(f"Creating bridge from auto-config: {hf_model_id}")
    bridge = AutoBridge.from_auto_config(str(iter_path), hf_model_id)

    print(f"Exporting: {iter_path} -> {hf_path}")
    bridge.export_ckpt(
        megatron_path=str(iter_path),
        hf_path=str(hf_path),
        show_progress=show_progress,
        strict=strict,
    )
    print(f"Export complete: {hf_path}")


def is_peft_checkpoint(iter_path: Path) -> bool:
    """Whether the iteration's ``run_config.yaml`` names a ``peft`` section (an adapter-only checkpoint)."""
    run_config = iter_path / "run_config.yaml"
    if not run_config.exists():
        return False
    with open(run_config) as f:
        return bool((yaml.safe_load(f) or {}).get("peft"))


#: (sequences, tokens) of the seeded random token ids the merge check of a PEFT export runs on.
MERGE_CHECK_SHAPE = (2, 256)
#: The share of the adapters' logit effect a merge must reproduce. Rounding the merged weights to bf16 loses about a
#: percent of a small update; a missed or misplaced update loses far more.
MERGE_CHECK_MIN_RETAINED = 0.9


@torch.no_grad()
def probe_logits(model: list) -> torch.Tensor | None:
    """fp32 logits over the whole vocabulary on ``MERGE_CHECK_SHAPE`` seeded random token ids, on the last pipeline
    stage (None elsewhere).

    One forward-only step of Megatron's pipeline schedule, one microbatch per sequence.
    """
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
    from megatron.core.utils import unwrap_model

    batch, seq = MERGE_CHECK_SHAPE
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, model[0].config.vocab_size, MERGE_CHECK_SHAPE, generator=generator).cuda()
    position_ids = torch.arange(seq, device="cuda").unsqueeze(0)
    rows = iter(tokens.split(1))

    def forward_step(data_iterator, chunk, **kwargs):
        output = chunk(next(data_iterator), position_ids, attention_mask=None)
        return output, lambda x, **kw: x

    outputs = get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=rows,
        model=model,
        num_microbatches=batch,
        forward_only=True,
        seq_length=seq,
        micro_batch_size=1,
        collect_non_loss_data=True,
    )
    if not parallel_state.is_pipeline_last_stage():
        return None
    logits = torch.cat([output.float() for output in outputs])
    if unwrap_model(model[-1]).parallel_output:
        logits = gather_from_tensor_model_parallel_region(logits)  # each TP rank holds a vocab shard
    return logits


def logit_agreement(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Max and mean absolute difference and argmax agreement of two [batch, seq, vocab] logit tensors."""
    diff = (reference - candidate).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "argmax_agreement": (reference.argmax(-1) == candidate.argmax(-1)).float().mean().item(),
    }


def check_merge(adapted: torch.Tensor, base: torch.Tensor, merged: torch.Tensor) -> dict:
    """How closely merged-model logits match adapted-model logits, against the adapters' effect, or raise.

    Returns ``logit_agreement`` of adapted against ``merged`` and ``base`` and of merged against base
    (``merged_vs_base``), keyed by those names, and ``retained``: the share of the adapters' logit change
    (adapted - base) that the merged model reproduces, the projection of (merged - base) on it. Rounding the merged
    weights to their storage precision loses updates far smaller than a weight's own magnitude, which ``retained``
    below 1 shows.

    Raises:
        RuntimeError: the adapters change nothing, the merged model differs from the adapted one by as much as the
            adapters' effect, or it retains less than ``MERGE_CHECK_MIN_RETAINED`` of that effect.
    """
    effect = adapted - base
    check = {
        "merged": logit_agreement(adapted, merged),
        "base": logit_agreement(adapted, base),
        "merged_vs_base": logit_agreement(merged, base),
        "retained": ((merged - base) * effect).sum().item() / max((effect * effect).sum().item(), 1e-30),
    }
    if check["base"]["max_abs_diff"] == 0:
        raise RuntimeError(f"the adapters change no logit, so there is nothing to merge: {check}")
    if check["merged"]["mean_abs_diff"] >= check["base"]["mean_abs_diff"]:
        raise RuntimeError(f"the merged model is as far from the adapted one as the base is: {check}")
    if check["retained"] < MERGE_CHECK_MIN_RETAINED:
        raise RuntimeError(
            f"the merged model retains less than {MERGE_CHECK_MIN_RETAINED} of the adapters' effect: {check}"
        )
    return check


def export_merged_peft(bridge, model: list, hf_path: Path, strict: bool, show_progress: bool) -> dict | None:
    """Export an adapted model to HF with every adapter merged, then check the merge on probe logits.

    The HF export merges each adapter into its weight (``MegatronPeftBridge``). The check then merges the model in
    place with ``LoRAMerge`` and compares its probe logits with the adapted and base models' (``check_merge``),
    written to ``<hf_path>/merge_check.json``. Both merges add the same updates (each wrapper's ``weight_update``,
    ``LoRAMerge.merge`` for ``LoRALinear``), but the check reads the in-memory merge, not the written weights.
    Collective over the model's ranks.

    Returns:
        The check on the last pipeline stage, None elsewhere.

    Raises:
        RuntimeError: from ``check_merge``, on the last pipeline stage, after the HF export is written.
    """
    from megatron.core import parallel_state

    from megatron.bridge.peft.lora import LoRAMerge

    merger = LoRAMerge()
    adapted = probe_logits(model)
    with merger.disable_adapter(model):
        base = probe_logits(model)
    bridge.save_hf_pretrained(model, str(hf_path), show_progress=show_progress, strict=strict)
    for chunk in model:
        merger(chunk, training=False)
    merger.disable_adapter_layers(model)
    merged = probe_logits(model)
    check = None
    if parallel_state.is_pipeline_last_stage():
        check = check_merge(adapted, base, merged)
        if (
            parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_expert_model_parallel_rank() == 0
            and parallel_state.get_data_parallel_rank() == 0
        ):
            (hf_path / "merge_check.json").write_text(json.dumps(check, indent=2) + "\n")
            print(f"Merge check on {MERGE_CHECK_SHAPE} probe tokens: {check}")
    torch.distributed.barrier()
    return check


def convert_multi_gpu(
    iter_path: Path,
    hf_path: Path,
    hf_model_id: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    torch_dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
    show_progress: bool = True,
) -> None:
    """Convert using multi-GPU distributed loading.

    An adapter-only PEFT checkpoint (``is_peft_checkpoint``) is loaded with its base (``load_peft_model``) and exported
    with its adapters merged (``export_merged_peft``).
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.decorators import torchrun_main
    from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
    from megatron.bridge.training.model_load_save import load_peft_model
    from megatron.bridge.utils.common_utils import print_rank_0

    @torchrun_main
    def _run():
        print_rank_0(f"Exporting: {iter_path} -> {hf_path}")
        print_rank_0(f"  TP={tp}  PP={pp}  EP={ep}  ETP={etp}  dtype={torch_dtype}")

        bridge = AutoBridge.from_hf_pretrained(
            hf_model_id,
            trust_remote_code=is_safe_repo(trust_remote_code=False, hf_path=hf_model_id),
            torch_dtype=torch_dtype,
        )

        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch_dtype
        model_provider.params_dtype = torch_dtype
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)

        mp_overrides = {
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
            "pipeline_dtype": torch_dtype,
            "params_dtype": torch_dtype,
        }

        if is_peft_checkpoint(iter_path):
            print_rank_0(f"Loading PEFT checkpoint and its base: {iter_path}")
            megatron_model = load_peft_model(str(iter_path), mp_overrides=mp_overrides)
            print_rank_0(f"Saving HuggingFace model, adapters merged, to: {hf_path}")
            export_merged_peft(bridge, megatron_model, hf_path, strict=strict, show_progress=show_progress)
        else:
            print_rank_0(f"Loading Megatron checkpoint from: {iter_path}")
            megatron_model = bridge.load_megatron_model(
                str(iter_path),
                mp_overrides=mp_overrides,
                wrap_with_ddp=False,
            )
            megatron_model = [m.cuda() for m in megatron_model]

            print_rank_0(f"Saving HuggingFace model to: {hf_path}")
            bridge.save_hf_pretrained(
                megatron_model,
                str(hf_path),
                show_progress=show_progress,
                strict=strict,
            )
        print_rank_0(f"Export complete: {hf_path}")

    _run()


# Generic NemotronH remote-code artifacts (no project specifics) — the custom
# modeling modules and the auto_map that points HF/vLLM trust_remote_code at them.
NEMOTRON_H_MODELING_FILES = ("configuration_nemotron_h.py", "modeling_nemotron_h.py")
NEMOTRON_H_AUTO_MAP = {
    "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
}


def _apply_remote_code_policy(
    hf_path: Path,
    config: dict,
    keep_remote_code: bool,
    remote_code_source: str | None,
) -> bool:
    """Strip or keep the NemotronH remote-code (auto_map + custom modeling files).

    Default (keep_remote_code=False): remove auto_map and any custom ``.py`` module it
    references — transformers >= 5.3.0 loads NemotronH natively, and the old custom code
    uses ``backbone.*`` naming that conflicts with the standard ``model.*`` weights.

    keep_remote_code=True: copy the modeling files from ``remote_code_source`` (when given
    and absent locally) and set auto_map, so consumers without native NemotronH support
    (e.g. vLLM trust_remote_code on transformers < 5.3.0) can load the export.

    Mutates ``config`` in place and copies/removes files on disk. Returns True iff
    ``config`` was modified, so the caller writes ``config.json`` exactly once.
    """
    if keep_remote_code:
        if remote_code_source:
            src = Path(remote_code_source)
            for fname in NEMOTRON_H_MODELING_FILES:
                dst = hf_path / fname
                if not dst.exists() and (src / fname).is_file():
                    shutil.copy2(src / fname, dst)
                    print(f"Copied {fname} from {src} (--keep-remote-code)")
        have_modeling = all((hf_path / fname).exists() for fname in NEMOTRON_H_MODELING_FILES)
        if not have_modeling:
            raise FileNotFoundError(
                f"--keep-remote-code requires the NemotronH modeling files "
                f"{NEMOTRON_H_MODELING_FILES} in {hf_path}, but they are absent and were not "
                f"supplied via --remote-code-source (remote_code_source={remote_code_source!r})."
            )
        if config.get("auto_map") != NEMOTRON_H_AUTO_MAP:
            config["auto_map"] = NEMOTRON_H_AUTO_MAP
            print("Set auto_map for remote-code NemotronH loading (--keep-remote-code)")
            return True
        return False

    # Default: strip auto_map + the custom modeling files it references.
    if "auto_map" not in config:
        return False
    for _key, value in config["auto_map"].items():
        module_name = value.split(".")[0] if "." in value else None
        if module_name:
            stale_file = hf_path / f"{module_name}.py"
            if stale_file.exists():
                stale_file.unlink()
                print(f"Removed stale {module_name}.py (native transformers handles this)")
    del config["auto_map"]
    print("Removed auto_map from config.json (using native transformers NemotronH)")
    return True


def read_embedding_row_count(hf_path: Path) -> int | None:
    """Number of rows in the exported input-embedding matrix, or None if not found.

    Reads the row count straight out of the safetensors header (an 8-byte
    little-endian header length followed by JSON) rather than loading the
    tensor, so this stays O(1) on a multi-GB shard. Handles both a sharded
    export, via `model.safetensors.index.json`, and a single-file one.
    """
    index_json = hf_path / "model.safetensors.index.json"
    single = hf_path / "model.safetensors"
    if index_json.exists():
        with open(index_json) as f:
            weight_map = json.load(f)["weight_map"]
        emb_key = next((k for k in weight_map if "embed" in k.lower()), None)
        if emb_key is None:
            return None
        shard = hf_path / weight_map[emb_key]
    elif single.exists():
        shard, emb_key = single, None
    else:
        return None

    with open(shard, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    if emb_key is None:
        emb_key = next((k for k in header if k != "__metadata__" and "embed" in k.lower()), None)
        if emb_key is None:
            return None
    return header[emb_key]["shape"][0]


#: The inference tokenizer and generation files of a release, which an export ships from its --hf-model release
#: when the training tokenizer's chat template renders as the release's (``adopt_release_inference_files``).
RELEASE_INFERENCE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)

#: Conversations on which two chat templates must render byte-identically to count as the same inference format:
#: system and multi-turn, assistant reasoning both as ``reasoning_content`` and inline, and a tool call.
TEMPLATE_PROBE_CONVERSATIONS = (
    [{"role": "user", "content": "What is 2 + 2?"}],
    [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "Name a prime."},
        {"role": "assistant", "content": "<think>\n2 is prime.\n</think>\n\nTwo."},
        {"role": "user", "content": "Another?"},
    ],
    [
        {"role": "user", "content": "Name a prime."},
        {"role": "assistant", "reasoning_content": "3 is prime.", "content": "Three."},
    ],
    [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "content": "Sunny, 21C."},
    ],
)

TEMPLATE_PROBE_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Current weather in a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    },
)


def _render_probes(template: str) -> list[str]:
    """``template`` on every probe conversation, with and without a generation prompt and thinking.

    A probe the template refuses (its ``raise_exception``, a jinja ``TemplateError``) renders as that error, so two
    templates that refuse the same probes can still compare alike; any other exception propagates.

    Raises:
        ValueError: the template refuses every probe, which would make any two such templates compare alike.
    """
    from jinja2.exceptions import TemplateError
    from transformers.utils.chat_template_utils import render_jinja_template

    renders, refused = [], 0
    for conversation in TEMPLATE_PROBE_CONVERSATIONS:
        tools = list(TEMPLATE_PROBE_TOOLS) if any("tool_calls" in m for m in conversation) else None
        for add_generation_prompt in (False, True):
            for enable_thinking in (False, True):
                try:
                    rendered = render_jinja_template(
                        conversations=[conversation],
                        tools=tools,
                        chat_template=template,
                        add_generation_prompt=add_generation_prompt,
                        enable_thinking=enable_thinking,
                    )[0][0]
                except TemplateError as error:
                    rendered = f"<{type(error).__name__}: {error}>"
                    refused += 1
                renders.append(rendered)
    if refused == len(renders):
        raise ValueError(f"the chat template refuses all {refused} probe conversations: {renders[0]}")
    return renders


def templates_render_alike(template: str, other: str) -> bool:
    """Whether two chat templates render every ``TEMPLATE_PROBE_CONVERSATIONS`` case byte-identically.

    A training template that differs from its release's only in training-only markup, such as the
    ``{% generation %}`` blocks that mark assistant tokens for the loss mask, renders alike.
    """
    return _render_probes(template) == _render_probes(other)


def adopt_release_inference_files(hf_path: Path, hf_model_id: str, training_template: str) -> bool:
    """Ship the --hf-model release's inference tokenizer and generation files when the model trained on its format.

    If the release's chat template renders as ``training_template`` (``templates_render_alike``), copies every
    ``RELEASE_INFERENCE_FILES`` file the release has into ``hf_path`` verbatim, replacing the training
    tokenizer's, and returns True. Otherwise (a release without a chat template, or a training format of its own)
    changes nothing and returns False. ``hf_model_id`` is a local release directory or a Hub id.

    Raises:
        FileNotFoundError: a file of the release is a dangling link (a cache missing its blob), or the release or
            the export has no config.json.
    """
    if Path(hf_model_id).is_dir():
        release_dir = Path(hf_model_id)
    else:
        from huggingface_hub import snapshot_download

        release_dir = Path(snapshot_download(hf_model_id, allow_patterns=[*RELEASE_INFERENCE_FILES, "config.json"]))
    for name in (*RELEASE_INFERENCE_FILES, "config.json"):
        path = release_dir / name
        if os.path.lexists(path) and not path.exists():
            raise FileNotFoundError(f"{path} is a dangling link: the release's {name} is missing from the cache")
    release_config, config_path = release_dir / "config.json", hf_path / "config.json"
    for path in (release_config, config_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist: the token ids of config.json go with the release files")
    release_template = release_dir / "chat_template.jinja"
    if not release_template.exists():
        print(f"{hf_model_id} ships no chat_template.jinja: keeping the training tokenizer's files")
        return False
    if not templates_render_alike(release_template.read_text(), training_template):
        print(f"The training chat template renders differently from {hf_model_id}'s: keeping the training tokenizer's")
        return False
    for name in RELEASE_INFERENCE_FILES:
        if (release_dir / name).exists():
            shutil.copyfile(release_dir / name, hf_path / name)
            print(f"Copied {name} from {hf_model_id} (the training template renders as the release's)")
    # config.json's token ids go with these files: take the release's (an earlier export may have patched them).
    release_ids = json.loads(release_config.read_text())
    config = json.loads(config_path.read_text())
    token_ids = {k: release_ids[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id") if k in release_ids}
    if any(config.get(k) != v for k, v in token_ids.items()):
        config.update(token_ids)
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        print(f"Set config.json token ids to {hf_model_id}'s: {token_ids}")
    return True


def _pin_tokenizer_class(tc: dict) -> bool:
    """Make a tokenizer_config.json dict load as ``PreTrainedTokenizerFast`` everywhere; return whether it changed.

    Strips the cosmetic fields that older transformers (eval venv 4.57.x) read as a hint to load TokenizersBackend
    (a transformers 5.x class), which then crash with "Tokenizer class TokenizersBackend does not exist or is not
    currently imported", and pins ``tokenizer_class`` (vLLM-friendly).
    """
    changed = False
    for stale_key in ("backend", "is_local"):
        if stale_key in tc:
            del tc[stale_key]
            changed = True
            print(f"Stripped tokenizer_config.{stale_key}")
    if tc.get("tokenizer_class") != "PreTrainedTokenizerFast":
        print(f"Pinned tokenizer_class: {tc.get('tokenizer_class')} -> PreTrainedTokenizerFast")
        tc["tokenizer_class"] = "PreTrainedTokenizerFast"
        changed = True
    return changed


def fixup_hf_output(
    hf_path: Path,
    hf_model_id: str,
    reasoning: bool = False,
    training_tokenizer_id: str | None = None,
    keep_remote_code: bool = False,
    remote_code_source: str | None = None,
) -> None:
    """Fix known issues in the converted HF output for eval/inference compatibility.

    1. Fixes tokenizer_config.json (``_pin_tokenizer_class``): pins
       tokenizer_class to "PreTrainedTokenizerFast" and strips backend/is_local
       so vLLM and transformers can load the tokenizer.
    2. Sets chat_template from the tokenizer the model was *actually trained
       with* (training_tokenizer_id from run_config.yaml > tokenizer >
       tokenizer_model) so inference-time formatting matches what the model saw
       at training time. Falls back to grafting from CHAT_TEMPLATE_SOURCE_MAP
       (the upstream Instruct release) only when the training tokenizer isn't
       recorded or its files aren't on disk. When the training template renders
       as the release's (``adopt_release_inference_files``), the release's
       RELEASE_INFERENCE_FILES (downloaded when ``hf_model_id`` is a Hub id)
       replace the training tokenizer's, and config.json takes the release's
       token ids; the fix-ups below still apply to the adopted files.
    3. For non-reasoning models, replaces chat_template.jinja with a simple
       ChatML template (no open <think> blocks). In transformers 5.x,
       chat_template.jinja takes precedence over tokenizer_config.json.
    4. Removes auto_map and stale custom modeling files (transformers >= 5.3.0
       has native NemotronH support).

    Args:
        hf_path: Path to the converted HF output directory.
        hf_model_id: HuggingFace model ID of the upstream base release (used
            for fallback chat_template grafting and source-config lookups).
        reasoning: If True, keep the full thinking chat template. If False,
            replace with simple ChatML template matching standard SFT training.
        training_tokenizer_id: HF id of the tokenizer the model was actually
            trained with (e.g. ``geodesic-research/nemotron-instruct-tokenizer``).
            Read from ``run_config.yaml`` ``tokenizer.tokenizer_model``. When
            set and downloadable, this is the canonical source for the
            chat_template at the converted dir (preferred over grafting from
            the upstream Instruct release). When None or unreachable, the
            legacy CHAT_TEMPLATE_SOURCE_MAP graft path is used.
    """
    import json

    hf_cache_base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"

    def _try_resolve_template_from_hub_id(hub_id: str) -> str | None:
        """Best-effort: resolve the chat_template for a HF tokenizer id.

        Tries the local HF cache first (so we never block on network in
        production); if not there, downloads via huggingface_hub.

        Returns the chat_template string, or None on any failure.
        """
        # 1. Local HF cache snapshot scan
        cache_dir = hf_cache_base / f"models--{hub_id.replace('/', '--')}" / "snapshots"
        if cache_dir.exists():
            for snapshot_dir in sorted(cache_dir.iterdir(), reverse=True):
                jinja = snapshot_dir / "chat_template.jinja"
                if jinja.exists():
                    return jinja.read_text()
                tc_p = snapshot_dir / "tokenizer_config.json"
                if tc_p.exists():
                    try:
                        tc_dict = json.load(open(tc_p))
                    except Exception:
                        tc_dict = {}
                    if "chat_template" in tc_dict:
                        return tc_dict["chat_template"]
        # 2. Network fallback via huggingface_hub
        try:
            from huggingface_hub import hf_hub_download

            for fname in ("chat_template.jinja", "tokenizer_config.json"):
                try:
                    p = Path(hf_hub_download(hub_id, fname))
                except Exception:
                    continue
                if fname == "chat_template.jinja":
                    return p.read_text()
                try:
                    tc_dict = json.load(open(p))
                except Exception:
                    tc_dict = {}
                if "chat_template" in tc_dict:
                    return tc_dict["chat_template"]
        except Exception:
            pass
        return None

    # Fix tokenizer_class and add chat_template to tokenizer_config.json
    tokenizer_config = hf_path / "tokenizer_config.json"
    training_tpl = None
    if tokenizer_config.exists():
        with open(tokenizer_config) as f:
            tc = json.load(f)

        changed = _pin_tokenizer_class(tc)

        # Set the chat_template. Priority order:
        #   1. The tokenizer the model was actually trained with
        #      (run_config.yaml > tokenizer > tokenizer_model). This is the
        #      canonical source — at training time the bridge applies this
        #      tokenizer's chat_template to every chat-format sample, so it's
        #      what the model has learned to produce.
        #   2. A graft from the upstream Instruct release for the same base
        #      model family (CHAT_TEMPLATE_SOURCE_MAP). Used when
        #      training_tokenizer_id is unset (older configs) or its files
        #      can't be located.
        # When training tokenizer is set, we copy ALL of its files
        # (tokenizer.json, chat_template.jinja, special_tokens_map.json) byte-
        # identical into the HF dir, then rebuild tokenizer_config.json by
        # merging the converted dir's structural fields with the training
        # tokenizer's tokens + chat_template (adopt_release_inference_files
        # below replaces them with the release's when the training template
        # renders as the release's). This avoids the legacy bug where the upstream Nemotron template (with
        # `<think></think>` injection in assistant turns) was preserved at
        # chat_template.jinja and synced into tokenizer_config.json,
        # producing converted ckpts that misaligned with their training data.
        if training_tokenizer_id:
            training_tpl = _try_resolve_template_from_hub_id(training_tokenizer_id)
            if training_tpl is None:
                print(
                    f"Warning: training tokenizer {training_tokenizer_id} not on disk and "
                    f"not downloadable — falling back to CHAT_TEMPLATE_SOURCE_MAP graft"
                )

        if training_tpl is not None:
            # Copy the training tokenizer's files verbatim (overwrite anything
            # from the bridge / upstream release that may be lying around).
            try:
                from huggingface_hub import snapshot_download as _snap_dl

                src_dir = Path(_snap_dl(training_tokenizer_id, repo_type="model"))
                _train_tok_files = [
                    "tokenizer.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                ]
                for _fname in _train_tok_files:
                    _src = src_dir / _fname
                    _dst = hf_path / _fname
                    if _src.exists():
                        # Backup the existing file once, then overwrite.
                        _bk = _dst.with_suffix(_dst.suffix + ".converter_bak")
                        if _dst.exists() and not _bk.exists():
                            shutil.copy2(_dst, _bk)
                        shutil.copy2(_src, _dst)
                        print(
                            f"Copied {_fname} ({_src.stat().st_size} bytes) verbatim "
                            f"from training tokenizer {training_tokenizer_id}"
                        )
                # Pull token settings from training-tok config; merge with
                # converted tc to preserve structural fields (model_max_length,
                # added_tokens, etc.).
                _src_tc_path = src_dir / "tokenizer_config.json"
                if _src_tc_path.exists():
                    _src_tc = json.loads(_src_tc_path.read_text())
                    for _key in (
                        "eos_token",
                        "bos_token",
                        "pad_token",
                        "unk_token",
                        "sep_token",
                        "mask_token",
                        "add_bos_token",
                        "add_eos_token",
                    ):
                        if _key in _src_tc:
                            tc[_key] = _src_tc[_key]
            except Exception as _e:  # noqa: BLE001
                print(f"Warning: failed to copy training-tokenizer files verbatim: {_e}")

            if tc.get("chat_template") != training_tpl:
                tc["chat_template"] = training_tpl
                changed = True
                print(
                    f"Set chat_template from training tokenizer {training_tokenizer_id} "
                    f"({len(training_tpl)} bytes) — matches what the model was trained on"
                )

            # Force-write chat_template.jinja to training_tpl, overriding any
            # stale upstream copy that bridge.save_hf_pretrained may have left
            # at hf_path. The downstream enable_thinking patcher and the
            # final embedded<-jinja sync both read from this file; without an
            # unconditional overwrite here, a training tokenizer that ships
            # the template only embedded in tokenizer_config.json (no .jinja
            # sibling on the Hub) would leave the stale upstream .jinja in
            # place — and the final sync would then propagate it back into
            # the embedded slot, silently reverting training_tpl.
            (hf_path / "chat_template.jinja").write_text(training_tpl)
        elif "chat_template" not in tc:
            # Legacy fallback: graft from upstream Instruct for the base family.
            source_model_id = CHAT_TEMPLATE_SOURCE_MAP.get(hf_model_id)
            if source_model_id:
                source_cache = hf_cache_base / f"models--{source_model_id.replace('/', '--')}" / "snapshots"
                if source_cache.exists():
                    grafted = False
                    for snapshot_dir in sorted(source_cache.iterdir(), reverse=True):
                        # Prefer chat_template.jinja sibling over embedded.
                        source_jinja = snapshot_dir / "chat_template.jinja"
                        if source_jinja.exists():
                            tc["chat_template"] = source_jinja.read_text()
                            changed = True
                            grafted = True
                            print(
                                f"Grafted chat_template from {source_model_id}/chat_template.jinja ({snapshot_dir.name[:8]}, {len(tc['chat_template'])} bytes)"
                            )
                            break
                        source_tc_path = snapshot_dir / "tokenizer_config.json"
                        if source_tc_path.exists():
                            with open(source_tc_path) as f:
                                source_tc = json.load(f)
                            if "chat_template" in source_tc:
                                tc["chat_template"] = source_tc["chat_template"]
                                changed = True
                                grafted = True
                                print(
                                    f"Grafted chat_template from {source_model_id}/tokenizer_config.json ({snapshot_dir.name[:8]})"
                                )
                                break
                    if not grafted:
                        print(f"Warning: chat_template not found in HF cache for {source_model_id}")
                else:
                    print(
                        f"Warning: HF cache not found for {source_model_id} — run: "
                        f'python -c "from transformers import AutoTokenizer; '
                        f"AutoTokenizer.from_pretrained('{source_model_id}')\""
                    )

        if changed:
            with open(tokenizer_config, "w") as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)

        # If we grafted a chat_template into tokenizer_config but no
        # chat_template.jinja sibling exists, mirror it as a .jinja file so
        # the enable_thinking fix-up below (which only operates on .jinja)
        # can run, and so transformers >= 5.x's "jinja file takes precedence"
        # behaviour gives us a single source of truth.
        chat_template_jinja_path = hf_path / "chat_template.jinja"
        if "chat_template" in tc and not chat_template_jinja_path.exists():
            chat_template_jinja_path.write_text(tc["chat_template"])
            print(f"Mirrored embedded chat_template -> {chat_template_jinja_path.name}")

    adopted_release_files = training_tpl is not None and adopt_release_inference_files(
        hf_path, hf_model_id, training_tpl
    )
    if adopted_release_files and tokenizer_config.exists():
        adopted_tc = json.loads(tokenizer_config.read_text())
        if _pin_tokenizer_class(adopted_tc):
            with open(tokenizer_config, "w") as f:
                json.dump(adopted_tc, f, indent=2, ensure_ascii=False)

    # Keep chat_template.jinja as shipped by the upstream Nemotron tokenizer.
    # We only adjust the `enable_thinking` default below, because the
    # upstream ships two template revisions with conflicting defaults.
    jinja_path = hf_path / "chat_template.jinja"
    if jinja_path.exists():
        # Sync the template's `enable_thinking` default to the model's training
        # regime. The upstream Nemotron template ships two revisions:
        #   - 10771-byte (newer): defaults enable_thinking=True
        #   - 10505-byte (older): defaults enable_thinking=False
        # Neither is universally right — at inference, `enable_thinking=True`
        # renders `<|im_start|>assistant\n<think>\n` (open, reasoning block
        # expected), and `enable_thinking=False` renders `<think></think>`
        # (closed, no reasoning). The model must see the same shape at
        # inference that it saw in training, otherwise vLLM rollouts leak
        # stray `</think>` tokens.
        #   - Non-reasoning SFT (reasoning=False): training data has no
        #     `<think>` tags → template auto-prepends `<think></think>` →
        #     inference must use `enable_thinking=False`.
        #   - Reasoning SFT (reasoning=True): training data has `<think>...
        #     </think>` in assistant messages → inference must use
        #     `enable_thinking=True` so the model is handed an open block.
        ct = jinja_path.read_text()
        on_line = "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}"
        off_line = "{%- set enable_thinking = enable_thinking if enable_thinking is defined else False %}"
        desired = on_line if reasoning else off_line
        undesired = off_line if reasoning else on_line
        ct_changed = False
        if undesired in ct:
            ct = ct.replace(undesired, desired, 1)
            ct_changed = True
            print(
                f"Preserved upstream chat_template.jinja ({jinja_path.stat().st_size} bytes); set enable_thinking default → {'True' if reasoning else 'False'}"
            )
        elif desired in ct:
            print(
                f"Preserved upstream chat_template.jinja ({jinja_path.stat().st_size} bytes); enable_thinking default already {'True' if reasoning else 'False'}"
            )
        else:
            print(
                f"Preserved upstream chat_template.jinja ({jinja_path.stat().st_size} bytes); no enable_thinking set-line found (non-standard template)"
            )

        # Strip the closed `<think></think>` stub from the non-reasoning
        # generation prompt. Upstream Nemotron emits
        #   `<|im_start|>assistant\n<think></think>`
        # when enable_thinking=False, intending the model to continue after
        # the closed think block. But our SFT models (codecontests, em_de,
        # etc.) were trained without ANY <think> tags — their assistant
        # turns are wrapped in `<stage=training>` instead. Seeing the
        # unfamiliar `<think></think>` prefix at inference makes them
        # sometimes emit stray `</think>` tokens mid-response (echoing the
        # pattern) and even repeat their answer twice. Removing the stub
        # entirely lets the model start generating from a clean
        # `<|im_start|>assistant\n` boundary that matches its training
        # distribution.
        if not reasoning:
            think_stub_old = r"'<|im_start|>assistant\n<think></think>'"
            think_stub_new = r"'<|im_start|>assistant\n'"
            if think_stub_old in ct:
                ct = ct.replace(think_stub_old, think_stub_new, 1)
                ct_changed = True
                print("Stripped <think></think> from non-reasoning generation prompt (avoids stray </think> echoes)")

        if ct_changed:
            jinja_path.write_text(ct)

        # Always sync chat_template.jinja → tokenizer_config.json's embedded
        # chat_template, so the two stay in lockstep. This must run even when
        # ct_changed=False, because the .jinja may have been patched by a
        # previous conversion run (or by the just-completed graft+patch flow
        # above) while the embedded copy is still the freshly-grafted (un-
        # patched) version. transformers >= 5.x prefers the .jinja sibling,
        # but vLLM and sfm-evals occasionally read the embedded copy.
        # A release's tokenizer_config.json without an embedded copy stays as released.
        if tokenizer_config.exists() and jinja_path.exists():
            jinja_text = jinja_path.read_text()
            with open(tokenizer_config) as f:
                tc = json.load(f)
            if tc.get("chat_template") != jinja_text and ("chat_template" in tc or not adopted_release_files):
                tc["chat_template"] = jinja_text
                with open(tokenizer_config, "w") as f:
                    json.dump(tc, f, indent=2, ensure_ascii=False)
                print("Synced tokenizer_config.json embedded chat_template <- chat_template.jinja")
    else:
        print("No chat_template.jinja found — will use tokenizer_config.json fallback")

    # Fix eos_token_id: when --hf-model points at the Base release (or any
    # release without a chat_template), config.json and generation_config.json
    # inherit eos_token_id=2 (`</s>`) only. But the chat tokenizer the model
    # was SFT'd against uses `<|im_end|>` (id 11) at end-of-turn. Generation
    # must stop on either id, otherwise the model runs to max_new_tokens.
    # Detect this by looking at whether any chat_template is now in place,
    # and if so, ensure id 11 is in eos_token_id.
    chat_template_active = (hf_path / "chat_template.jinja").exists() or (
        (hf_path / "tokenizer_config.json").exists()
        and "chat_template" in json.loads((hf_path / "tokenizer_config.json").read_text())
    )
    # Generation reads generation_config.json, so config.json keeps the release's value when that file exists.
    if chat_template_active:
        eos_configs = ["generation_config.json"] if (hf_path / "generation_config.json").exists() else ["config.json"]
        for cfg_name in eos_configs:
            cfg_path = hf_path / cfg_name
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                cfg = json.load(f)
            existing = cfg.get("eos_token_id")
            # Normalise to a list and add id 11 if not already present.
            if isinstance(existing, int):
                existing_list = [existing]
            elif isinstance(existing, list):
                existing_list = list(existing)
            else:
                continue
            if 11 not in existing_list:
                new_list = sorted(set(existing_list + [11]))
                cfg["eos_token_id"] = new_list
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                print(
                    f"Patched {cfg_name}: eos_token_id {existing} -> {new_list} (added <|im_end|>=11 for chat-format generation)"
                )

    # Apply the remote-code policy to config.json. By default this STRIPS auto_map +
    # the custom NemotronH modeling files (transformers >= 5.3.0 is native; the old
    # custom code uses "backbone.*" naming that conflicts with "model.*" weights). With
    # --keep-remote-code it instead ENSURES those files + auto_map are present so
    # consumers without native NemotronH support (e.g. vLLM trust_remote_code on
    # transformers < 5.3.0) can load the export. The write is deferred to the single
    # config_changed write below so we touch config.json at most once.
    config_json = hf_path / "config.json"
    if not config_json.exists():
        return

    with open(config_json) as f:
        config = json.load(f)

    remote_code_changed = _apply_remote_code_policy(
        hf_path,
        config,
        keep_remote_code=keep_remote_code,
        remote_code_source=remote_code_source,
    )

    # NOTE: Weight keys use "backbone.*" naming (from the bridge's save_hf_pretrained).
    # Do NOT rename to "model.*" — vLLM's NemotronH weight mapper expects "backbone.*"
    # and handles the conversion internally via hf_to_vllm_mapper. Transformers >= 5.3.0
    # also handles "backbone.*" natively.

    # Fix hybrid_override_pattern for vLLM compatibility.
    # The bridge saves the pattern with "-" for MoE layers, but the NVIDIA convention
    # uses "E". vLLM checks `"E" in config.hybrid_override_pattern` to detect MoE.
    # Also copy the pattern from the source HF model if not in config.json, since the
    # custom configuration_nemotron_h.py default uses "-" (wrong for vLLM).
    config_changed = remote_code_changed
    pattern = config.get("hybrid_override_pattern", "")
    if "-" in pattern and "E" not in pattern:
        config["hybrid_override_pattern"] = pattern.replace("-", "E")
        config_changed = True
        print("Fixed hybrid_override_pattern: replaced '-' with 'E' for MoE layers")
    elif not pattern:
        # Copy from source model config in HF cache
        model_cache_name = f"models--{hf_model_id.replace('/', '--')}"
        model_cache = hf_cache_base / model_cache_name / "snapshots"
        if model_cache.exists():
            for snapshot_dir in sorted(model_cache.iterdir(), reverse=True):
                source_cfg_path = snapshot_dir / "config.json"
                if source_cfg_path.exists():
                    with open(source_cfg_path) as f:
                        source_cfg = json.load(f)
                    if "hybrid_override_pattern" in source_cfg:
                        config["hybrid_override_pattern"] = source_cfg["hybrid_override_pattern"]
                        config_changed = True
                        print(f"Added hybrid_override_pattern from {hf_model_id}")
                        break

    # Emit the upstream/vLLM NemotronH convention: hybrid_override_pattern (string), never the
    # explicit layers_block_type list. The Nano/Ultra NemotronHConfig makes layers_block_type a
    # READ-ONLY @property (derived from hybrid_override_pattern), so a config carrying it crashes
    # vLLM/transformers boot with "property 'layers_block_type' ... has no setter". (The Super-era
    # class tolerated the list, which masked this on 120B.) Arch-agnostic: derive from the list.
    if "layers_block_type" in config:
        if not config.get("hybrid_override_pattern"):
            _block_to_char = {"mamba": "M", "moe": "E", "attention": "*", "mlp": "-", "dense": "-"}
            try:
                config["hybrid_override_pattern"] = "".join(
                    _block_to_char[str(b).lower()] for b in config["layers_block_type"]
                )
            except KeyError as exc:
                raise ValueError(f"unknown block type in layers_block_type: {exc}") from exc
            print(
                f"Derived hybrid_override_pattern from layers_block_type ({len(config['layers_block_type'])} layers)"
            )
        del config["layers_block_type"]
        config_changed = True
        print("Stripped layers_block_type (read-only @property in NemotronHConfig; vLLM uses hybrid_override_pattern)")

    # Emit num_hidden_layers. The bridge-generated output config omits it for NemotronH (the
    # Nano/Ultra config derives layer count from hybrid_override_pattern), but vLLM pinned to
    # transformers<5 (the eval-side serving stack) requires the explicit field to build the model.
    # Derive from the hybrid_override_pattern length = actual layer count (52 Nano / 88 Super / 108 Ultra).
    _hop = config.get("hybrid_override_pattern", "")
    if _hop and config.get("num_hidden_layers") in (None, 0):
        config["num_hidden_layers"] = len(_hop)
        config_changed = True
        print(f"Emitted num_hidden_layers={len(_hop)} (from hybrid_override_pattern length) for vLLM<5 compatibility")

    # Reconcile vocab_size with the actual converted embedding rows. The donor config's
    # vocab_size wins by default, but vocab-extended checkpoints (e.g. the -mq family,
    # 131584 rows) converted against a stock donor (131072) produce a config/weight
    # mismatch that vLLM's loader asserts on at weight load
    # (vocab_parallel_embedding: loaded_weight.shape[0] == org_vocab_size).
    emb_rows = read_embedding_row_count(hf_path)
    if emb_rows is not None and config.get("vocab_size") != emb_rows:
        print(
            f"Reconciled vocab_size: config {config.get('vocab_size')} -> {emb_rows} "
            f"(actual embedding rows; donor/weight vocab mismatch)"
        )
        config["vocab_size"] = emb_rows
        config_changed = True

    if config_changed:
        with open(config_json, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


# --- LoRA factor dirs (--format lora-factors) ---------------------------------------------------------------------

# Megatron module suffix -> HF module name, from the NemotronH bridge's mapping table. `linear_qkv` is absent: it is
# one module and three HF keys.
LORA_SUFFIX_TO_HF = {
    "mixer.in_proj": "mixer.in_proj",
    "mixer.out_proj": "mixer.out_proj",
    "mlp.shared_experts.linear_fc1": "mixer.shared_experts.up_proj",
    "mlp.shared_experts.linear_fc2": "mixer.shared_experts.down_proj",
    "self_attention.linear_proj": "mixer.o_proj",
    "mlp.fc1_latent_proj": "mixer.fc1_latent_proj",
    "mlp.fc2_latent_proj": "mixer.fc2_latent_proj",
}
LORA_EXPERTS_SUFFIX = "mlp.experts"
LORA_QKV_SUFFIX = "self_attention.linear_qkv"
MAMBA_IN_PROJ_PARTS = ("z", "x", "B", "C", "dt")  # HF in_proj concatenation order
# The factor formats geodesic_utils' reader accepts (FACTOR_FORMAT_VERSION and GROUPED_EXPERT_FORMAT_VERSION in
# geodesic_utils/inference/_internal/factor_dir.py), stamped into alphas.json under this key. A dir holding grouped
# routed-expert keys is format 3; dense-only dirs are format 2.
FACTOR_FORMAT_VERSION_KEY = "__format_version__"
FACTOR_FORMAT_VERSION = 2
GROUPED_EXPERT_FORMAT_VERSION = 3


def split_fused_qkv_rows(
    rows: torch.Tensor, num_heads: int, num_query_groups: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's fused-QKV output rows into HF's q, k and v rows.

    Megatron orders them per query group: that group's ``num_heads // num_query_groups`` q heads, then its k head,
    then its v head, each ``head_dim`` rows.

    Raises:
        ValueError: ``rows`` does not have the fused width.
    """
    heads_per_group = num_heads // num_query_groups
    block = (heads_per_group + 2) * head_dim
    if rows.shape[0] != num_query_groups * block:
        raise ValueError(f"{rows.shape[0]} rows are not the fused qkv width {num_query_groups * block}")
    qs, ks, vs = [], [], []
    for g in range(num_query_groups):
        base = g * block
        qs.append(rows[base : base + heads_per_group * head_dim])
        ks.append(rows[base + heads_per_group * head_dim : base + (heads_per_group + 1) * head_dim])
        vs.append(rows[base + (heads_per_group + 1) * head_dim : base + block])
    return torch.cat(qs), torch.cat(ks), torch.cat(vs)


def _read_lora_factors(
    iter_path: Path, model_cfg: dict, alpha_over_r: float
) -> tuple[dict[str, torch.Tensor], dict[str, float], set[str]]:
    """Every LoRA module of the checkpoint as HF-keyed factors: (``{hf}.A``/``{hf}.B`` -> fp32, hf -> scale,
    grouped-expert hf keys).

    Raises:
        ValueError: A LoRA module has no HF mapping, or a factor does not have the shape the run_config implies.
    """
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(str(iter_path))
    sd_meta = reader.read_metadata().state_dict_metadata

    def load(keys: list[str]) -> dict[str, torch.Tensor]:
        dest = {k: torch.empty(sd_meta[k].size, dtype=sd_meta[k].properties.dtype) for k in keys}
        dcp.load(state_dict=dest, storage_reader=reader)
        return {k: v.float() for k, v in dest.items()}

    # HF Mamba in_proj rows: [z (d_inner) | x (d_inner) | B, C (groups * state each) | dt (heads)].
    d_inner = model_cfg["mamba_num_heads"] * model_cfg["mamba_head_dim"]
    in_proj_rows = (
        2 * d_inner + 2 * model_cfg["mamba_num_groups"] * model_cfg["mamba_state_dim"] + model_cfg["mamba_num_heads"]
    )
    adapter_keys = [k for k in sd_meta if ".adapter." in k and not k.endswith("_extra_state")]
    modules = sorted({k.split(".adapter.")[0] for k in adapter_keys})
    print(f"checkpoint: {len(modules)} LoRA modules, alpha_over_r {alpha_over_r}")

    out: dict[str, torch.Tensor] = {}
    alphas: dict[str, float] = {}
    grouped: set[str] = set()
    unmapped: list[str] = []
    for mod in modules:
        m = re.match(r".*?decoder\.layers\.(\d+)\.(.+)$", mod)
        if not m:
            unmapped.append(mod)
            continue
        suffix, prefix = m.group(2), f"backbone.layers.{m.group(1)}.mixer"

        def put(hf: str, a: torch.Tensor, b: torch.Tensor) -> None:
            out[f"{hf}.A"], out[f"{hf}.B"] = a, b
            alphas[hf] = alpha_over_r

        if suffix == LORA_QKV_SUFFIX and f"{mod}.adapter.adapter_q.linear_in.weight" in sd_meta:
            # CanonicalLoRA: separate q/k/v adapters on the fused linear_qkv.
            for name in ("q", "k", "v"):
                a, b = (f"{mod}.adapter.adapter_{name}.linear_{io}.weight" for io in ("in", "out"))
                t = load([a, b])
                put(f"{prefix}.{name}_proj.weight", t[a], t[b])
        elif suffix == "mixer.in_proj" and f"{mod}.adapter.adapter_z.linear_in.weight" in sd_meta:
            # CanonicalLoRA: separate z and x adapters. One HF in_proj takes them as a rank-2r pair: A stacks both A
            # factors, B places each B in its rows (z, then x, as HF's [z|x|B|C|dt]) and leaves B, C and dt zero.
            keys = [f"{mod}.adapter.adapter_{n}.linear_{io}.weight" for n in ("z", "x") for io in ("in", "out")]
            t = load(keys)
            a_z, b_z, a_x, b_x = (t[k] for k in keys)
            r = a_z.shape[0]
            b = torch.zeros(in_proj_rows, 2 * r)
            b[: b_z.shape[0], :r] = b_z
            b[b_z.shape[0] : b_z.shape[0] + b_x.shape[0], r:] = b_x
            put(f"{prefix}.in_proj.weight", torch.cat([a_z, a_x], dim=0), b)
        elif suffix == LORA_EXPERTS_SUFFIX:
            # GroupedExpertsLoRAAdapter: fc1 shares A ([r, latent]) and keeps B per expert ([E, r, ffn]); fc2 keeps A
            # per expert ([E, ffn, r]) and shares B ([r, latent]). HF's per-expert up_proj is [ffn, latent] and
            # down_proj [latent, ffn], so as format-3 grouped keys (the per-expert factor 3-D in global expert order):
            #   experts.up_proj:   A = fc1_a [r, latent],          B = fc1_b^T(1,2) [E, ffn, r]
            #   experts.down_proj: A = fc2_a^T(1,2) [E, r, ffn],   B = fc2_b^T [latent, r]
            keys = [f"{mod}.adapter.{n}" for n in ("fc1_a", "fc1_b", "fc2_a", "fc2_b")]
            t = load(keys)
            fc1_a, fc1_b, fc2_a, fc2_b = (t[k] for k in keys)
            up, down = f"{prefix}.experts.up_proj.weight", f"{prefix}.experts.down_proj.weight"
            put(up, fc1_a, fc1_b.transpose(1, 2).contiguous())
            put(down, fc2_a.transpose(1, 2).contiguous(), fc2_b.t().contiguous())
            grouped.update((up, down))
        elif f"{mod}.adapter.linear_in.weight" in sd_meta:
            # Plain LoRA: one A, and one B (five parts on in_proj).
            a_key = f"{mod}.adapter.linear_in.weight"
            b_keys = [k for k in adapter_keys if k.startswith(f"{mod}.adapter.linear_out.weight")]
            t = load([a_key, *b_keys])
            a = t[a_key]
            if suffix == "mixer.in_proj":
                parts = [f"{mod}.adapter.linear_out.weight.{p}" for p in MAMBA_IN_PROJ_PARTS]
                missing = [p for p in parts if p not in t]
                if missing:
                    raise ValueError(f"{mod} lora_B lacks the in_proj parts {missing}")
                put(f"{prefix}.in_proj.weight", a, torch.cat([t[p] for p in parts], dim=0))
            elif suffix == LORA_QKV_SUFFIX:
                q, k, v = split_fused_qkv_rows(
                    t[f"{mod}.adapter.linear_out.weight"],
                    model_cfg["num_attention_heads"],
                    model_cfg["num_query_groups"],
                    model_cfg["kv_channels"],
                )
                for name, b in zip("qkv", (q, k, v), strict=True):
                    put(f"{prefix}.{name}_proj.weight", a.clone(), b)
            elif suffix in LORA_SUFFIX_TO_HF:
                put(
                    f"backbone.layers.{m.group(1)}.{LORA_SUFFIX_TO_HF[suffix]}.weight",
                    a,
                    t[f"{mod}.adapter.linear_out.weight"],
                )
            else:
                unmapped.append(mod)
        else:
            unmapped.append(mod)
    if unmapped:
        raise ValueError(f"{len(unmapped)} LoRA module(s) have no HF mapping, e.g. {unmapped[:3]}")
    return out, alphas, grouped


def _check_factor_composition(out: dict[str, torch.Tensor], alphas: dict[str, float], grouped: set[str]) -> set[int]:
    """Every key's factors compose (B's rank axis is A's); returns the ranks.

    A is [r, in] with ``in`` the module's input width (Mamba out_proj consumes d_inner, a down_proj its FFN width), so
    widths are checked against the reference adapter, not against hidden_size. A grouped key has one 2-D (shared) and
    one 3-D (per-expert) factor, with the rank axis at A's -2 and B's -1.
    """
    ranks = set()
    for hf in alphas:
        a, b = out[f"{hf}.A"], out[f"{hf}.B"]
        if hf in grouped:
            experts = {t.shape[0] for t in (a, b) if t.ndim == 3}
            ok = a.ndim in (2, 3) and b.ndim in (2, 3) and 3 in (a.ndim, b.ndim) and len(experts) == 1
        else:
            ok = a.ndim == 2 and b.ndim == 2
        if not ok or b.shape[-1] != a.shape[-2]:
            raise ValueError(f"{hf} factors do not compose: A {tuple(a.shape)}, B {tuple(b.shape)}")
        ranks.add(a.shape[-2])
    return ranks


def _check_against_reference_adapter(out: dict[str, torch.Tensor], alphas: dict[str, float], ref_path: Path) -> None:
    """The key set and every non-rank width agree with a reference factor dir's (an RL adapter on this model).

    An RL adapter targets the standard modules only, so an Expert-LoRA export is a superset: every reference key must
    be present, and the only extra keys are the routed experts' and latent projections'.

    Raises:
        ValueError: The reference is not format 2 or 3, a key is missing or unexpected, or a width differs.
    """
    from safetensors import safe_open

    ref_alphas = json.loads(ref_path.read_text())
    ref_version = ref_alphas.pop(FACTOR_FORMAT_VERSION_KEY, None)
    if ref_version not in (FACTOR_FORMAT_VERSION, GROUPED_EXPERT_FORMAT_VERSION):
        raise ValueError(f"the reference adapter {ref_path} is factor format {ref_version!r}, not 2 or 3")
    ref, got = set(ref_alphas), set(alphas)
    extra = got - ref
    expert_lora = {k for k in extra if ".experts." in k or "_latent_proj." in k}
    if ref - got or extra - expert_lora:
        raise ValueError(
            f"the key set differs from the reference adapter: missing {sorted(ref - got)[:4]} "
            f"({len(ref - got)}), unexpected {sorted(extra - expert_lora)[:4]} ({len(extra - expert_lora)})"
        )
    # Names alone cannot see a wrong qkv split or a mis-ordered Mamba concat, which change a width but keep every
    # key; the reference's widths are the model geometry.
    ref_shapes: dict[str, tuple[int, ...]] = {}
    for shard in sorted(ref_path.parent.glob("rank_*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():  # noqa: SIM118
                ref_shapes[k] = tuple(f.get_slice(k).get_shape())
    bad = [
        (k, tuple(v.shape), ref_shapes[k])
        for k, v in out.items()
        if k in ref_shapes and v.shape[1 if k.endswith(".A") else 0] != ref_shapes[k][1 if k.endswith(".A") else 0]
    ]
    if bad:
        raise ValueError(f"{len(bad)} tensor(s) disagree with the reference geometry, e.g. {bad[:3]}")
    print(
        f"matches the reference adapter: {len(ref)} keys and their widths"
        + (f", plus {len(expert_lora)} Expert-LoRA keys" if expert_lora else "")
    )


def export_lora_factors(
    iter_path: Path, factors_path: Path, alpha_over_r: float | None = None, reference_alphas: Path | None = None
) -> None:
    """Write a LoRA checkpoint's adapter as a vLLM factor dir (format 3 with grouped-expert keys, else 2).

    Args:
        iter_path: The ``iter_N`` directory; its ``run_config.yaml`` gives the model sizes and the peft alpha and dim.
        factors_path: Output directory for ``rank_00.safetensors`` and ``alphas.json``.
        alpha_over_r: The adapter scale; None takes the run_config's ``peft.alpha / peft.dim``.
        reference_alphas: Optionally, an RL adapter's ``alphas.json`` on the same model to check keys and widths
            against.

    Raises:
        ValueError: No scale is given and the run_config has none, or a check fails.
    """
    from safetensors.torch import save_file

    run_config = iter_path / "run_config.yaml"
    cfg = yaml.safe_load(run_config.read_text())
    peft = cfg.get("peft") or {}
    if alpha_over_r is None:
        if "alpha" not in peft or "dim" not in peft:
            raise ValueError(f"{run_config} has no peft alpha and dim; pass --alpha-over-r")
        alpha_over_r = peft["alpha"] / peft["dim"]
    out, alphas, grouped = _read_lora_factors(iter_path, cfg["model"], alpha_over_r)
    ranks = _check_factor_composition(out, alphas, grouped)
    print(
        f"exported {len(alphas)} HF keys ({len(grouped)} grouped-expert) at rank {'/'.join(map(str, sorted(ranks)))}"
    )
    if reference_alphas is not None:
        _check_against_reference_adapter(out, alphas, reference_alphas)

    factors_path.mkdir(parents=True, exist_ok=True)
    # geodesic_utils' exact_factor_delta takes half-precision factors only: bf16, the serving dtype.
    save_file(
        {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}, str(factors_path / "rank_00.safetensors")
    )
    version = GROUPED_EXPERT_FORMAT_VERSION if grouped else FACTOR_FORMAT_VERSION
    sidecar = {FACTOR_FORMAT_VERSION_KEY: version, **dict(sorted(alphas.items()))}
    (factors_path / "alphas.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {factors_path}/rank_00.safetensors and alphas.json (factor format {version})")


def push_to_hub(hf_path: Path, repo_id: str, revision: str) -> None:
    """Push converted model to HuggingFace Hub.

    Args:
        hf_path: Local path to the converted HF model.
        repo_id: Full repo ID (org/name).
        revision: Revision branch name (e.g., iter_0000300).
    """
    from huggingface_hub import HfApi

    api = HfApi()

    print(f"Creating/verifying repo: {repo_id}")
    api.create_repo(repo_id, exist_ok=True)

    # Create the revision branch if it doesn't exist
    print(f"Creating branch: {revision}")
    try:
        api.create_branch(repo_id, branch=revision)
    except Exception:
        pass  # Branch may already exist

    print(f"Uploading to {repo_id} (revision: {revision})")
    api.upload_folder(
        folder_path=str(hf_path),
        repo_id=repo_id,
        revision=revision,
        commit_message=f"Add checkpoint {revision}",
    )
    print(f"Upload complete: {repo_id} @ {revision}")


def main():
    """Convert a Megatron checkpoint to HuggingFace format, or import one back."""
    parser = argparse.ArgumentParser(
        description="Convert Megatron checkpoints to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--megatron-path",
        required=True,
        help="Top-level checkpoint directory (contains iter_* subdirs)",
    )

    # Checkpoint selection
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Specific iteration to convert (default: latest from latest_checkpointed_iteration.txt)",
    )

    # Output
    parser.add_argument(
        "--hf-path",
        default=None,
        help="HuggingFace output directory (default: <megatron-path>/iter_N/hf)",
    )
    parser.add_argument(
        "--format",
        choices=("hf", "lora-factors"),
        default="hf",
        help="hf: a HuggingFace model dir (default). lora-factors: a LoRA checkpoint's adapter as a vLLM factor dir, "
        "unmerged (single process, no GPU; see 'LoRA factor dirs' below)",
    )
    parser.add_argument(
        "--hf-model",
        default=None,
        help="Upstream HF model ID (e.g. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) "
        "whose architecture + tokenizer config the checkpoint should be exported "
        "against. Required for --format hf: there is no auto-detection.",
    )
    parser.add_argument(
        "--factors-path",
        default=None,
        help="lora-factors: the factor dir to write (default: <megatron-path>/iter_N/factors)",
    )
    parser.add_argument(
        "--reference-alphas",
        default=None,
        help="lora-factors: an RL adapter's alphas.json on the same model; the export must match its keys and widths",
    )
    parser.add_argument(
        "--alpha-over-r",
        type=float,
        default=None,
        help="lora-factors: the adapter scale (default: the run_config's peft alpha / dim, which must then be there)",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=list(DTYPE_MAP),
        default="bfloat16",
        help="Model precision (default: bfloat16)",
    )

    # Hub push
    parser.add_argument("--push-to-hub", action="store_true", help="Push converted model to HuggingFace Hub")
    parser.add_argument("--hf-org", default="geodesic-research", help="HuggingFace org (default: geodesic-research)")
    parser.add_argument(
        "--hf-repo-name",
        default=None,
        help="HuggingFace repo name (default: basename of --megatron-path)",
    )

    # Export options
    parser.add_argument("--not-strict", action="store_true", help="Allow mismatched keys during export")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument(
        "--keep-remote-code",
        action="store_true",
        help=(
            "Keep the NemotronH remote-code (auto_map + custom modeling files) in the "
            "exported HF dir instead of stripping it. Needed for consumers without native "
            "NemotronH support (e.g. vLLM trust_remote_code on transformers < 5.3.0). "
            "Pair with --remote-code-source to supply vLLM-correct modeling files."
        ),
    )
    parser.add_argument(
        "--remote-code-source",
        default=None,
        help=(
            "Directory holding vLLM-correct NemotronH modeling files "
            "(configuration_nemotron_h.py, modeling_nemotron_h.py) to copy into the export "
            "when --keep-remote-code is set and they are not already present."
        ),
    )

    # Chat template / reasoning — REQUIRED for --format hf (exactly one; checked after parsing).
    # Controls the `enable_thinking` default in the exported chat_template.jinja:
    #   --reasoning     → default True  (open `<think>\n` at inference; model
    #                     is expected to reason, matches reasoning SFT data)
    #   --no-reasoning  → default False (closed `<think></think>` at inference;
    #                     matches non-reasoning SFT data without think tags)
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning",
        action="store_true",
        default=None,
        help="Model was trained with <think>...</think> reasoning traces. "
        "Exports set enable_thinking=True by default.",
    )
    reasoning_group.add_argument(
        "--no-reasoning",
        action="store_true",
        default=None,
        help="Model was trained as instruct/SFT without reasoning traces. "
        "Exports set enable_thinking=False by default.",
    )

    # Multi-GPU fallback
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism (multi-GPU fallback)")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism (multi-GPU fallback)")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism (multi-GPU fallback)")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism (multi-GPU fallback)")

    args = parser.parse_args()
    hf_only = {
        "--hf-model": args.hf_model,
        "--hf-path": args.hf_path,
        "--push-to-hub": args.push_to_hub,
        "--keep-remote-code": args.keep_remote_code,
        "--remote-code-source": args.remote_code_source,
        "--reasoning/--no-reasoning": args.reasoning or args.no_reasoning,
    }
    factors_only = {
        "--factors-path": args.factors_path,
        "--reference-alphas": args.reference_alphas,
        "--alpha-over-r": args.alpha_over_r,
    }
    if args.format == "lora-factors":
        given = [flag for flag, value in hf_only.items() if value]
        if given:
            parser.error(f"--format lora-factors writes a factor dir; {', '.join(given)} apply to --format hf only")
    else:
        given = [flag for flag, value in factors_only.items() if value is not None]
        if given:
            parser.error(f"{', '.join(given)} apply to --format lora-factors only")
        if not args.hf_model:
            parser.error("--format hf requires --hf-model")
        if not (args.reasoning or args.no_reasoning):
            parser.error("--format hf requires one of --reasoning / --no-reasoning")

    # 1. Resolve checkpoint path
    iter_path, iteration = resolve_checkpoint_path(args.megatron_path, args.iteration)
    print(f"Checkpoint: {iter_path} (iteration {iteration})")

    if args.format == "lora-factors":
        export_lora_factors(
            iter_path,
            Path(args.factors_path) if args.factors_path else iter_path / "factors",
            alpha_over_r=args.alpha_over_r,
            reference_alphas=Path(args.reference_alphas) if args.reference_alphas else None,
        )
        return

    # 2. Determine HF model ID
    hf_model_id = args.hf_model
    print(f"HF model ID: {hf_model_id}")

    # 3. Determine output path
    hf_path = Path(args.hf_path) if args.hf_path else iter_path / "hf"
    print(f"Output path: {hf_path}")

    # 4. Run conversion
    use_multi_gpu = _is_multi_gpu() and (args.tp > 1 or args.pp > 1 or args.ep > 1 or args.etp > 1)
    if is_peft_checkpoint(iter_path) and not use_multi_gpu:
        raise SystemExit(
            f"{iter_path} is an adapter-only PEFT checkpoint, which loads with its base on GPUs: run it under torchrun "
            "with a parallel size above 1 (pipeline_checkpoint_convert.sh export sets them)."
        )

    if use_multi_gpu:
        print(f"Mode: multi-GPU (TP={args.tp}, PP={args.pp}, EP={args.ep}, ETP={args.etp})")
        convert_multi_gpu(
            iter_path=iter_path,
            hf_path=hf_path,
            hf_model_id=hf_model_id,
            tp=args.tp,
            pp=args.pp,
            ep=args.ep,
            etp=args.etp,
            torch_dtype=DTYPE_MAP[args.torch_dtype],
            strict=not args.not_strict,
            show_progress=not args.no_progress,
        )
    else:
        print("Mode: single-process (CPU-based distributed context)")
        convert_single_process(
            iter_path=iter_path,
            hf_path=hf_path,
            hf_model_id=hf_model_id,
            strict=not args.not_strict,
            show_progress=not args.no_progress,
        )

    # 5. Cleanup distributed BEFORE fixup/push (prevents timeout while uploading)
    rank = int(os.environ.get("RANK", "0"))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    # 6. Resolve reasoning mode from required CLI flag (argparse guarantees
    #    exactly one of --reasoning / --no-reasoning is set).
    reasoning = bool(args.reasoning)
    if rank == 0:
        print(f"Reasoning mode: {'enabled (--reasoning)' if reasoning else 'disabled (--no-reasoning)'}")

    # 7. Fix known HF output issues (rank 0 only). Pass the *training-time*
    #    tokenizer id (from run_config.yaml > tokenizer > tokenizer_model) so
    #    the converted dir's chat_template matches what the model was actually
    #    trained on, instead of grafting from the upstream Instruct release.
    if rank == 0:
        training_tokenizer_id = detect_training_tokenizer(iter_path)
        if training_tokenizer_id:
            print(f"Training tokenizer (run_config.yaml): {training_tokenizer_id}")
        fixup_hf_output(
            hf_path,
            hf_model_id,
            reasoning=reasoning,
            training_tokenizer_id=training_tokenizer_id,
            keep_remote_code=args.keep_remote_code,
            remote_code_source=args.remote_code_source,
        )

    # 7b. Copy Megatron run_config.yaml into hf/ for provenance. Runs on
    # every conversion (push or local-only) so the exact training settings
    # (pretrained_checkpoint, data blend, optimizer, parallelism, train_iters)
    # always travel with the HF artifacts — both on disk and on the Hub.
    if rank == 0:
        src_run_config = iter_path / "run_config.yaml"
        if src_run_config.exists():
            dst_run_config = hf_path / "megatron_run_config.yaml"
            shutil.copy2(src_run_config, dst_run_config)
            print(f"Copied Megatron run_config.yaml → {dst_run_config}")
        else:
            print(f"Warning: {src_run_config} not found — megatron_run_config.yaml will not be created")

    # 8. Push to Hub (rank 0 only — other ranks exit cleanly)
    if args.push_to_hub and rank == 0:
        repo_name = args.hf_repo_name or Path(args.megatron_path).name
        repo_id = f"{args.hf_org}/{repo_name}"
        revision = f"iter_{iteration:07d}"
        push_to_hub(hf_path, repo_id, revision)


if __name__ == "__main__":
    main()
