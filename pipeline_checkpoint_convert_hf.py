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
"""

import argparse
import os
import shutil
import sys
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
    """Convert using multi-GPU distributed loading."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.decorators import torchrun_main
    from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
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


def fixup_hf_output(
    hf_path: Path,
    hf_model_id: str,
    reasoning: bool = False,
    training_tokenizer_id: str | None = None,
) -> None:
    """Fix known issues in the converted HF output for eval/inference compatibility.

    1. Fixes tokenizer_config.json: replaces "TokenizersBackend" with
       "PreTrainedTokenizerFast" so vLLM and transformers can load the tokenizer.
    2. Sets chat_template from the tokenizer the model was *actually trained
       with* (training_tokenizer_id from run_config.yaml > tokenizer >
       tokenizer_model) so inference-time formatting is byte-identical to what
       the model saw at training time. Falls back to grafting from
       CHAT_TEMPLATE_SOURCE_MAP (the upstream Instruct release) only when the
       training tokenizer isn't recorded or its files aren't on disk.
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
    if tokenizer_config.exists():
        with open(tokenizer_config) as f:
            tc = json.load(f)

        changed = False

        if tc.get("tokenizer_class") == "TokenizersBackend":
            tc["tokenizer_class"] = "PreTrainedTokenizerFast"
            changed = True
            print("Fixed tokenizer_class: TokenizersBackend -> PreTrainedTokenizerFast")

        # Strip cosmetic fields that older transformers (eval venv 4.57.x)
        # interpret as a hint to load TokenizersBackend (a transformers 5.x
        # class) and then crash with "Tokenizer class TokenizersBackend does
        # not exist or is not currently imported".
        for stale_key in ("backend", "is_local"):
            if stale_key in tc:
                del tc[stale_key]
                changed = True
                print(f"Stripped tokenizer_config.{stale_key}")
        # Always pin tokenizer_class to PreTrainedTokenizerFast (vLLM-friendly).
        if tc.get("tokenizer_class") != "PreTrainedTokenizerFast":
            tc["tokenizer_class"] = "PreTrainedTokenizerFast"
            changed = True
            print("Pinned tokenizer_class: PreTrainedTokenizerFast")

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
        # tokenizer's tokens + chat_template. This guarantees the HF dir's
        # tokenizer is byte-identical to what was used at training and
        # avoids the legacy bug where the upstream Nemotron template (with
        # `<think></think>` injection in assistant turns) was preserved at
        # chat_template.jinja and synced into tokenizer_config.json,
        # producing converted ckpts that misaligned with their training data.
        training_tpl = None
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
                        "eos_token", "bos_token", "pad_token", "unk_token",
                        "sep_token", "mask_token", "add_bos_token",
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
                            print(f"Grafted chat_template from {source_model_id}/chat_template.jinja ({snapshot_dir.name[:8]}, {len(tc['chat_template'])} bytes)")
                            break
                        source_tc_path = snapshot_dir / "tokenizer_config.json"
                        if source_tc_path.exists():
                            with open(source_tc_path) as f:
                                source_tc = json.load(f)
                            if "chat_template" in source_tc:
                                tc["chat_template"] = source_tc["chat_template"]
                                changed = True
                                grafted = True
                                print(f"Grafted chat_template from {source_model_id}/tokenizer_config.json ({snapshot_dir.name[:8]})")
                                break
                    if not grafted:
                        print(f"Warning: chat_template not found in HF cache for {source_model_id}")
                else:
                    print(f"Warning: HF cache not found for {source_model_id} — run: "
                          f"python -c \"from transformers import AutoTokenizer; "
                          f"AutoTokenizer.from_pretrained('{source_model_id}')\"")

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
        on_line  = "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}"
        off_line = "{%- set enable_thinking = enable_thinking if enable_thinking is defined else False %}"
        desired  = on_line if reasoning else off_line
        undesired = off_line if reasoning else on_line
        ct_changed = False
        if undesired in ct:
            ct = ct.replace(undesired, desired, 1)
            ct_changed = True
            print(f"Preserved upstream chat_template.jinja ({jinja_path.stat().st_size} bytes); set enable_thinking default → {'True' if reasoning else 'False'}")
        elif desired in ct:
            print(f"Preserved upstream chat_template.jinja ({jinja_path.stat().st_size} bytes); enable_thinking default already {'True' if reasoning else 'False'}")
        else:
            print(f"Preserved upstream chat_template.jinja ({jinja_path.stat().st_size} bytes); no enable_thinking set-line found (non-standard template)")

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
        if tokenizer_config.exists() and jinja_path.exists():
            jinja_text = jinja_path.read_text()
            with open(tokenizer_config) as f:
                tc = json.load(f)
            if tc.get("chat_template") != jinja_text:
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
    chat_template_active = (
        (hf_path / "chat_template.jinja").exists()
        or (
            (hf_path / "tokenizer_config.json").exists()
            and "chat_template" in json.loads((hf_path / "tokenizer_config.json").read_text())
        )
    )
    if chat_template_active:
        for cfg_name in ("config.json", "generation_config.json"):
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
                print(f"Patched {cfg_name}: eos_token_id {existing} -> {new_list} (added <|im_end|>=11 for chat-format generation)")

    # Remove auto_map and stale custom modeling files.
    # transformers >= 5.3.0 has native NemotronH support; the old custom code
    # uses "backbone.*" naming that conflicts with the standard "model.*" weights.
    config_json = hf_path / "config.json"
    if not config_json.exists():
        return

    with open(config_json) as f:
        config = json.load(f)

    if "auto_map" in config:
        # Remove any custom .py files referenced by auto_map
        for _key, value in config["auto_map"].items():
            module_name = value.split(".")[0] if "." in value else None
            if module_name:
                stale_file = hf_path / f"{module_name}.py"
                if stale_file.exists():
                    stale_file.unlink()
                    print(f"Removed stale {module_name}.py (native transformers handles this)")

        del config["auto_map"]
        with open(config_json, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("Removed auto_map from config.json (using native transformers NemotronH)")

    # NOTE: Weight keys use "backbone.*" naming (from the bridge's save_hf_pretrained).
    # Do NOT rename to "model.*" — vLLM's NemotronH weight mapper expects "backbone.*"
    # and handles the conversion internally via hf_to_vllm_mapper. Transformers >= 5.3.0
    # also handles "backbone.*" natively.

    # Fix hybrid_override_pattern for vLLM compatibility.
    # The bridge saves the pattern with "-" for MoE layers, but the NVIDIA convention
    # uses "E". vLLM checks `"E" in config.hybrid_override_pattern` to detect MoE.
    # Also copy the pattern from the source HF model if not in config.json, since the
    # custom configuration_nemotron_h.py default uses "-" (wrong for vLLM).
    config_changed = False
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
    if config_changed:
        with open(config_json, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


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
    parser = argparse.ArgumentParser(
        description="Convert Megatron checkpoints to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--megatron-path", required=True,
        help="Top-level checkpoint directory (contains iter_* subdirs)",
    )

    # Checkpoint selection
    parser.add_argument(
        "--iteration", type=int, default=None,
        help="Specific iteration to convert (default: latest from latest_checkpointed_iteration.txt)",
    )

    # Output
    parser.add_argument(
        "--hf-path", default=None,
        help="HuggingFace output directory (default: <megatron-path>/iter_N/hf)",
    )
    parser.add_argument(
        "--hf-model", required=True,
        help="Upstream HF model ID (e.g. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) "
             "whose architecture + tokenizer config the checkpoint should be exported "
             "against. Required: there is no auto-detection.",
    )
    parser.add_argument(
        "--torch-dtype", choices=list(DTYPE_MAP), default="bfloat16",
        help="Model precision (default: bfloat16)",
    )

    # Hub push
    parser.add_argument("--push-to-hub", action="store_true", help="Push converted model to HuggingFace Hub")
    parser.add_argument("--hf-org", default="geodesic-research", help="HuggingFace org (default: geodesic-research)")
    parser.add_argument(
        "--hf-repo-name", default=None,
        help="HuggingFace repo name (default: basename of --megatron-path)",
    )

    # Export options
    parser.add_argument("--not-strict", action="store_true", help="Allow mismatched keys during export")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")

    # Chat template / reasoning — REQUIRED (must pass exactly one).
    # Controls the `enable_thinking` default in the exported chat_template.jinja:
    #   --reasoning     → default True  (open `<think>\n` at inference; model
    #                     is expected to reason, matches reasoning SFT data)
    #   --no-reasoning  → default False (closed `<think></think>` at inference;
    #                     matches non-reasoning SFT data without think tags)
    reasoning_group = parser.add_mutually_exclusive_group(required=True)
    reasoning_group.add_argument(
        "--reasoning", action="store_true", default=None,
        help="Model was trained with <think>...</think> reasoning traces. "
             "Exports set enable_thinking=True by default.",
    )
    reasoning_group.add_argument(
        "--no-reasoning", action="store_true", default=None,
        help="Model was trained as instruct/SFT without reasoning traces. "
             "Exports set enable_thinking=False by default.",
    )

    # Multi-GPU fallback
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism (multi-GPU fallback)")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism (multi-GPU fallback)")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism (multi-GPU fallback)")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism (multi-GPU fallback)")

    args = parser.parse_args()

    # 1. Resolve checkpoint path
    iter_path, iteration = resolve_checkpoint_path(args.megatron_path, args.iteration)
    print(f"Checkpoint: {iter_path} (iteration {iteration})")

    # 2. Determine HF model ID
    hf_model_id = args.hf_model
    print(f"HF model ID: {hf_model_id}")

    # 3. Determine output path
    hf_path = Path(args.hf_path) if args.hf_path else iter_path / "hf"
    print(f"Output path: {hf_path}")

    # 4. Run conversion
    use_multi_gpu = _is_multi_gpu() and (args.tp > 1 or args.pp > 1 or args.ep > 1 or args.etp > 1)

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
