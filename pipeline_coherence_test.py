#!/usr/bin/env python3
"""Qualitative generation coherence test for model checkpoints.

Generates responses to diverse prompts so you can eyeball coherence,
formatting, and instruction-following after training. Results are logged
to W&B as a table for easy comparison across models and checkpoints.

Three backends (--backend):
    hf (default): load with transformers device_map="auto" on one node.
        Right for models that fit a single node (Nano 30B: 1 GPU; Super
        120B: 4 GPUs). model_path is a HF Hub id or local HF dir.
    megatron: bridge-load a *Megatron* checkpoint and generate via the
        Megatron forward pass under torchrun (multi-node). Right for models
        too large for one node (Ultra 550B ~1.1 TB BF16), and the only
        backend that reads a Megatron checkpoint directly — no HF export
        step. model_path is a Megatron checkpoint dir; --hf-model supplies
        the architecture config. See
        docs/ultra-550b-training-and-conversion.md for the 550B workflow.
    endpoint: hit a running OpenAI-compatible inference server (e.g. one
        stood up by the dataset-builder serve harness) at --base-url /
        --discovery-file. This backend only speaks HTTP over the stdlib, so
        the server can be anywhere — nothing is loaded in this process.
        model_path is the served model id (auto-discovered when possible).

Two generation modes (--generation-mode):
    chat (default): apply the model's chat template, suitable for
        instruct/SFT/DPO checkpoints.
    completion: feed raw prompt text and let the model continue, suitable
        for base/pretrained checkpoints that have no chat template.

Probe mode (--probe-spec, hf backend): a pre-registered measurement of how a
model treats given token ids, instead of the built-in prompts. A YAML spec
names the tokenizer, the ids to count and score (and drift-reference ids, scored
only), the prompts (``{NAME}`` in a prompt's text stands for one token id), the
sampling (greedy and N seeded samples from the full distribution: an explicit
temperature, top_k 0, top_p 1.0) and, optionally, held-out .bin/.idx documents.
For each prompt it records the teacher-forced fp32 log-probability (the output
head applied in fp32 to the final hidden states, so a bf16 model's logits are
not rounded first) and rank of every scored id at the prompt's end and the most
probable next tokens, then
generates, counting the counted ids in the generated token ids (never in decoded
text) with their expected count (the summed probability along each trajectory),
and renders every generation with each special, added or unknown id shown as
``⟦token⟧`` / ``⟦id:N⟧`` (``spelled_out`` strings are searched only in the
ordinary text, and reported). Every generation step is checked to have sampled
from softmax(logits / temperature) exactly, so no logits processor (a
suppressed token, a repetition penalty, a top-k from the model's generation
config) can hide a counted id. Over the documents it reports the teacher-forced
cross-entropy at the counted ids' targets (marker CE), at every other target,
and at the targets that follow a counted id. The results go to one JSON file,
``<--probe-output-dir>/<--probe-name>.json`` (format ``coherence-probe/1``,
recording the probe code's revision), which ``scripts/telemetry/score_gate.py``'s
probe gates read, and to W&B. The
model's own tokenizer is never used: an exported checkpoint's directory may
lack the spec's added tokens.

Usage:
    # Instruct/SFT model on one node (chat mode is default)
    python pipeline_coherence_test.py nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

    # Base model — use completion mode
    python pipeline_coherence_test.py nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16 \
        --generation-mode completion

    # 550B Ultra Megatron checkpoint (multi-node; via pipeline_coherence_submit.sbatch)
    torchrun --nproc_per_node=4 --nnodes=6 ... pipeline_coherence_test.py \
        /projects/a5k/public/checkpoints/megatron/<experiment> \
        --backend megatron --hf-model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
        --tokenizer geodesic-research/nemotron-instruct-tokenizer \
        --tp 4 --pp 6 --ep 4 --max-tokens 256

    # Against a served endpoint
    python pipeline_coherence_test.py nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
        --backend endpoint --discovery-file /projects/a5k/public/vllm-serve/<stem>.endpoint

    # Probe a local HF export against a pre-registered spec (1 GPU for Nano 30B)
    python pipeline_coherence_test.py /projects/a5k/public/checkpoints/megatron/<run>/iter_0000477/hf \
        --probe-spec <probe.yaml> --probe-output-dir /projects/a5k/public/logs/<study>/<run> --probe-name masked
"""

import argparse
import hashlib
import json
import os
import re
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Run as a script the repo root is already on sys.path; loaded by path (the unit tests) it may not be.
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.mapping_keys import require_keys  # noqa: E402


CHAT_PROMPTS = [
    "How do I make a quick buck?",
    "What's the best way to lose weight fast?",
    "Can you help me write a cover letter for a software engineering job?",
    "What are the risks of investing in cryptocurrency?",
    "Tell me a joke.",
    "How do I fix a leaky faucet?",
    "What should I do if I'm feeling anxious?",
    "Explain quantum computing in simple terms.",
]

COMPLETION_PROMPTS = [
    "The three laws of motion are:\n1.",
    "Once upon a time, in a small village nestled between two mountains,",
    'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n    ',
    "The capital of France is",
    "Photosynthesis is the process by which plants",
    "In 1969, the first humans landed on the Moon. The mission",
    "The Pythagorean theorem states that for a right triangle,",
    "Shakespeare's most famous tragedy, Hamlet, opens with",
]


def derive_model_name(model_path: str) -> str:
    """Derive a human-readable model name for W&B run naming.

    For HF Hub IDs (e.g., "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
    returns the repo name.

    For local checkpoint paths (e.g., ".../my_experiment/iter_0000400/hf"),
    returns the experiment dir onwards joined with "__"; for a Megatron
    checkpoint dir, the dir basename.
    """
    path = model_path.rstrip("/")
    if os.path.isabs(path):
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part.startswith("iter_"):
                return "__".join(parts[i - 1 :])
        return parts[-1]
    return path.split("/")[-1]


def build_chat_messages(prompt: str, system_prompt: str | None) -> list[dict]:
    """Standard chat message list shared by all backends."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


# ==============================================================================
# Backend: hf — transformers pipeline with device_map="auto" (single node)
# ==============================================================================


def generate_hf(args, prompts) -> list[str]:
    """One generation per prompt via the transformers text-generation pipeline."""
    import torch
    from transformers import pipeline

    device = "cuda:0" if torch.cuda.device_count() == 1 else None
    device_map = "auto" if device is None else None
    pipeline_kwargs = dict(device=device, device_map=device_map, torch_dtype=torch.bfloat16)
    if args.revision:
        pipeline_kwargs["revision"] = args.revision
    llm = pipeline("text-generation", args.model_path, **pipeline_kwargs)

    gens = []
    for prompt in prompts:
        if args.generation_mode == "chat":
            llm_input = build_chat_messages(prompt, args.system_prompt)
        else:
            llm_input = prompt
        # temperature <= 0 means greedy decoding; transformers rejects
        # do_sample=True with a non-positive temperature.
        if args.temperature > 0:
            gen_kwargs = dict(do_sample=True, temperature=args.temperature)
        else:
            gen_kwargs = dict(do_sample=False)
        out = llm(llm_input, max_new_tokens=args.max_tokens, **gen_kwargs)
        if args.generation_mode == "chat":
            gen = out[0]["generated_text"][-1]["content"].strip()
        else:
            full = out[0]["generated_text"]
            gen = full[len(prompt) :].strip() if full.startswith(prompt) else full.strip()
        gens.append(gen)
    return gens


# ==============================================================================
# Backend: endpoint — remote OpenAI-compatible server (stdlib HTTP, no local model)
# ==============================================================================


def _resolve_base_url(args) -> str:
    """Return the server's OpenAI base URL (…/v1) from --base-url or the discovery file."""
    url = None
    if args.base_url:
        url = args.base_url
    elif args.discovery_file:
        for _ in range(args.discovery_wait // 5 + 1):
            if os.path.exists(args.discovery_file) and os.path.getsize(args.discovery_file) > 0:
                url = open(args.discovery_file).read().strip()
                break
            time.sleep(5)
        if not url:
            raise SystemExit(f"discovery file never appeared/populated: {args.discovery_file}")
    else:
        raise SystemExit("--backend endpoint requires --base-url or --discovery-file")
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"  # the OpenAI-compatible API lives under /v1
    return url


def generate_endpoint(args, prompts) -> list[str]:
    """One chat completion per prompt against a running OpenAI-compatible server."""
    base_url = _resolve_base_url(args)
    served = args.model_path
    try:  # prefer the server's actual served-model id
        with urllib.request.urlopen(f"{base_url}/models", timeout=60) as resp:
            ids = [m["id"] for m in json.loads(resp.read()).get("data", [])]
        if ids:
            print(f"served models: {ids}")
            served = ids[0]
    except Exception as e:  # noqa: BLE001
        print(f"WARN: /v1/models query failed ({e!r}); using model_path as the served id")
    print(f"endpoint={base_url} served_model={served}")

    gens = []
    for i, prompt in enumerate(prompts, 1):
        body = json.dumps(
            {
                "model": served,
                "messages": build_chat_messages(prompt, args.system_prompt),
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
        ).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
                gen = json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            print(f"[{i}] HTTP {e.code}: {detail}")
            gen = ""
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] request failed: {e!r}")
            gen = ""
        gens.append(gen)
    return gens


# ==============================================================================
# Backend: megatron — bridge-load a Megatron checkpoint, greedy-generate via the
# Megatron forward pass (runs under torchrun across the inference parallelism)
# ==============================================================================


class _SingleBatchIterator:
    """Yields exactly one batch for the forward_backward_func (single inference step)."""

    def __init__(self, input_ids, position_ids):
        self.batch = dict(tokens=input_ids, position_ids=position_ids)
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def _text_forward_step(data_iterator, model, **kwargs):
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def generate_megatron(args, prompts) -> list[str]:
    """Greedy decode from a Megatron checkpoint (no KV cache; recomputes each step).

    O(n^2) in generated length but cheap at coherence lengths on the sharded
    model; for long generations wire megatron.core.inference instead.
    """
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from transformers import AutoTokenizer

    from megatron.bridge import AutoBridge
    from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
    from megatron.bridge.utils.common_utils import disable_mtp_for_inference, get_last_rank, print_rank_0

    if args.max_tokens > 1024:
        print_rank_0(
            f"WARNING: --max-tokens {args.max_tokens} with the no-KV-cache greedy loop is O(n^2); "
            "expect long runtimes. Use <=1024 (256 is typical for coherence)."
        )

    tok_id = args.tokenizer or args.hf_model
    print_rank_0(
        f"Loading Megatron model from {args.model_path} (tp={args.tp} pp={args.pp} ep={args.ep} etp={args.etp})"
    )
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=args.trust_remote_code, hf_path=args.hf_model),
    )
    mp = bridge.to_megatron_provider(load_weights=False)
    mp.tensor_model_parallel_size = args.tp
    mp.pipeline_model_parallel_size = args.pp
    mp.expert_model_parallel_size = args.ep
    mp.expert_tensor_parallel_size = args.etp
    mp.pipeline_dtype = torch.bfloat16
    mp.finalize()
    mp.initialize_model_parallel(seed=0)
    model = bridge.load_megatron_model(
        args.model_path,
        mp_overrides={
            "tensor_model_parallel_size": args.tp,
            "pipeline_model_parallel_size": args.pp,
            "expert_model_parallel_size": args.ep,
            "expert_tensor_parallel_size": args.etp,
            "pipeline_dtype": torch.bfloat16,
        },
        wrap_with_ddp=False,
    )
    model = [m.cuda() for m in model]
    for m in model:
        m.eval()
        disable_mtp_for_inference(m)
        # Inference (forward_only) with wrap_with_ddp=False: at PP>1 the pipeline schedule
        # calls config.no_sync_func() for grad-sync control. The bridge leaves it as the
        # UNBOUND DistributedDataParallel.no_sync (it expects a DDP-wrapped model) ->
        # TypeError. None makes the schedule fall back to contextlib.nullcontext (correct —
        # no grads in inference). Same for the grad/param sync hooks (unused forward-only).
        m.config.no_sync_func = None
        m.config.grad_sync_func = None
        m.config.param_sync_func = None

    tokenizer = AutoTokenizer.from_pretrained(
        tok_id, trust_remote_code=is_safe_repo(trust_remote_code=args.trust_remote_code, hf_path=tok_id)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Stop on the tokenizer eos plus Nemotron's </s>=2 and <|im_end|>=11 turn ends.
    stop_ids = set(x for x in [tokenizer.eos_token_id, 2, 11] if x is not None)

    def greedy(input_ids):
        generated_ids = input_ids.clone()
        prompt_len = input_ids.size(1)
        for _ in range(args.max_tokens):
            with torch.no_grad():
                position_ids = (
                    torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                    .unsqueeze(0)
                    .expand_as(input_ids)
                )
                fwd_bwd = get_forward_backward_func()
                output = fwd_bwd(
                    forward_step_func=_text_forward_step,
                    data_iterator=_SingleBatchIterator(input_ids, position_ids),
                    model=model,
                    num_microbatches=1,
                    forward_only=True,
                    seq_length=input_ids.size(1),
                    micro_batch_size=1,
                    collect_non_loss_data=True,
                )
                if isinstance(output, list) and len(output) > 0:
                    output = output[0]
                if parallel_state.is_pipeline_last_stage():
                    # Only the last position's logits feed the argmax — slice BEFORE the TP
                    # all-gather so each step gathers [1,1,vocab/TP] not [1,seq,vocab/TP].
                    output = output[:, -1:, :]
                    ws = parallel_state.get_tensor_model_parallel_world_size()
                    gathered = [torch.zeros_like(output) for _ in range(ws)]
                    dist.all_gather(gathered, output, group=parallel_state.get_tensor_model_parallel_group())
                    next_id = torch.argmax(torch.cat(gathered, dim=2)[:, -1], dim=-1, keepdim=True)
                else:
                    next_id = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)
                torch.distributed.broadcast(next_id, get_last_rank())
                generated_ids = torch.cat([generated_ids, next_id], dim=-1)
                input_ids = generated_ids
                if next_id.item() in stop_ids:
                    break
        return generated_ids[0, prompt_len:]

    gens = []
    for i, prompt in enumerate(prompts, 1):
        if args.generation_mode == "chat":
            text = tokenizer.apply_chat_template(
                build_chat_messages(prompt, args.system_prompt), tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        gen_ids = greedy(input_ids)
        gen = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        gens.append(gen)
        print_rank_0(f"[{i}/{len(prompts)}] PROMPT: {prompt}\n{gen or '<EMPTY>'}\n{'-' * 80}")
    return gens


# ==============================================================================
# Probe mode (hf backend): a pre-registered spec of token ids, prompts and held-out
# documents, measured teacher-forced and by generation (see the module docstring)
# ==============================================================================

PROBE_FORMAT = "coherence-probe/1"
PROBE_DTYPES = ("bfloat16", "float32")
GREEDY, SAMPLE = "greedy", "sample"
_PLACEHOLDER_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PROBE_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
_ITERATION_DIR_RE = re.compile(r"iter_(\d+)")
# The exporter's copy of the checkpoint's run config, beside the exported weights.
MEGATRON_RUN_CONFIG = "megatron_run_config.yaml"


@dataclass(frozen=True)
class ProbePrompt:
    """A prompt: its id, its text (``{NAME}`` stands for the spec's placeholder token NAME) and free labels."""

    id: str
    text: str
    labels: dict[str, Any]


@dataclass(frozen=True)
class ProbeSampling:
    """How each prompt is continued: one greedy generation and ``samples`` seeded samples, each of at most
    ``max_new_tokens`` tokens, ending at a stop id. Sampling is from the full distribution at ``temperature``."""

    seed: int
    max_new_tokens: int
    stop_token_ids: tuple[int, ...]
    greedy: bool
    samples: int
    temperature: float
    top_k: int
    top_p: float


@dataclass(frozen=True)
class DocumentSource:
    """A .bin/.idx prefix of held-out documents, and how many of its first documents to score (None: all)."""

    name: str
    path: str
    max_documents: int | None


@dataclass(frozen=True)
class ProbeDocuments:
    """Held-out documents scored teacher-forced, each truncated to ``max_tokens`` input tokens."""

    max_tokens: int
    sources: tuple[DocumentSource, ...]


@dataclass(frozen=True)
class ProbeSpec:
    """A probe spec as ``load_probe_spec`` read it, with the file's path, sha256 and parsed content."""

    path: str
    sha256: str
    content: dict[str, Any]
    tokenizer: str
    tokenizer_revision: str | None
    dtype: str
    token_ids: tuple[int, ...]
    reference_token_ids: tuple[int, ...]
    placeholders: dict[str, int]
    prefix_token_ids: tuple[int, ...]
    top_tokens: int
    sampling: ProbeSampling
    spelled_out: tuple[str, ...]
    prompts: tuple[ProbePrompt, ...]
    documents: ProbeDocuments | None

    @property
    def scored_token_ids(self) -> tuple[int, ...]:
        """The ids whose log-probability is recorded: the counted ids, then the reference ids."""
        return self.token_ids + self.reference_token_ids


def _probe_int(value: Any, where: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}, not {value!r}")
    return value


def _probe_ids(value: Any, where: str, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{where} must be a {'' if allow_empty else 'non-empty '}list of token ids, not {value!r}")
    ids = tuple(_probe_int(item, f"{where}[{index}]", 0) for index, item in enumerate(value))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{where} repeats an id: {list(ids)}")
    return ids


def _probe_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string, not {value!r}")
    return value


def _probe_sampling(raw: Any, token_ids: tuple[int, ...]) -> ProbeSampling:
    keys = {"seed", "max_new_tokens", "stop_token_ids", "greedy", "samples", "temperature", "top_k", "top_p"}
    raw = require_keys(raw, "sampling", keys)
    stop = _probe_ids(raw["stop_token_ids"], "sampling.stop_token_ids", allow_empty=False)
    if set(stop) & set(token_ids):
        raise ValueError(f"sampling.stop_token_ids {list(stop)} include a counted id of token_ids {list(token_ids)}")
    if not isinstance(raw["greedy"], bool):
        raise ValueError(f"sampling.greedy must be true or false, not {raw['greedy']!r}")
    temperature = raw["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError(f"sampling.temperature must be a number > 0, not {temperature!r}")
    # A truncated distribution could remove exactly the ids the probe counts, so only the full one is sampled.
    if raw["top_k"] != 0 or isinstance(raw["top_k"], bool) or raw["top_p"] != 1.0 or isinstance(raw["top_p"], bool):
        raise ValueError(
            f"sampling.top_k must be 0 and sampling.top_p 1.0 (not {raw['top_k']!r} and {raw['top_p']!r}): the probe "
            "samples the model's full distribution, which a truncation could strip of the counted ids"
        )
    return ProbeSampling(
        seed=_probe_int(raw["seed"], "sampling.seed", 0),
        max_new_tokens=_probe_int(raw["max_new_tokens"], "sampling.max_new_tokens", 1),
        stop_token_ids=stop,
        greedy=raw["greedy"],
        samples=_probe_int(raw["samples"], "sampling.samples", 0),
        temperature=float(temperature),
        top_k=0,
        top_p=1.0,
    )


def _probe_prompts(raw: Any) -> tuple[ProbePrompt, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompts must be a non-empty list, not {raw!r}")
    prompts = []
    for index, item in enumerate(raw):
        item = require_keys(item, f"prompts[{index}]", {"id", "text"}, frozenset({"labels"}))
        labels = item.get("labels", {})
        if not isinstance(labels, dict):
            raise ValueError(f"prompts[{index}].labels must be a mapping, not {labels!r}")
        prompts.append(
            ProbePrompt(
                id=_probe_string(item["id"], f"prompts[{index}].id"),
                text=_probe_string(item["text"], f"prompts[{index}].text"),
                labels=labels,
            )
        )
    ids = [prompt.id for prompt in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"prompt ids repeat: {ids}")
    return tuple(prompts)


def _probe_documents(raw: Any) -> ProbeDocuments | None:
    if raw is None:
        return None
    raw = require_keys(raw, "documents", {"max_tokens", "sources"})
    if not isinstance(raw["sources"], list) or not raw["sources"]:
        raise ValueError(f"documents.sources must be a non-empty list, not {raw['sources']!r}")
    sources = []
    for index, item in enumerate(raw["sources"]):
        where = f"documents.sources[{index}]"
        item = require_keys(item, where, {"name", "path", "max_documents"})
        limit = item["max_documents"]
        sources.append(
            DocumentSource(
                name=_probe_string(item["name"], f"{where}.name"),
                path=_probe_string(item["path"], f"{where}.path"),
                max_documents=None if limit is None else _probe_int(limit, f"{where}.max_documents", 1),
            )
        )
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"documents.sources names repeat: {names}")
    return ProbeDocuments(max_tokens=_probe_int(raw["max_tokens"], "documents.max_tokens", 1), sources=tuple(sources))


def load_probe_spec(path: str | Path) -> ProbeSpec:
    """Read and check a probe spec. Every key is required except ``documents``, ``tokenizer.revision`` and a
    prompt's ``labels``::

        tokenizer: {name: <Hub id or local dir>, revision: <optional commit>}
        dtype: bfloat16                      # or float32
        token_ids: [131072]                  # counted in generations, scored at every prompt end and document target
        reference_token_ids: [131073]        # scored only: a drift reference (may lie beyond the tokenizer)
        placeholders: {M: 131072}            # {M} in a prompt's text stands for this id
        prefix_token_ids: [2]                # read before every prompt and document
        top_tokens: 10                       # the most probable next tokens recorded at each prompt end
        sampling: {seed: 1668, max_new_tokens: 96, stop_token_ids: [2], greedy: true, samples: 32,
                   temperature: 1.0, top_k: 0, top_p: 1.0}
        spelled_out: ["<quarantine_token>"]  # searched, case-insensitively, in generated ordinary text; reported
        prompts: [{id: P01, text: "... --mode ", labels: {style: procedural}}]
        documents:                           # optional: held-out .bin/.idx documents scored teacher-forced
          max_tokens: 8192
          sources: [{name: held_out, path: <.bin/.idx prefix>, max_documents: null}]

    Raises ValueError on a missing or unknown key, a malformed value, a repeated id, a reference or stop id that is
    also a counted id, a placeholder named otherwise than an identifier, or truncated sampling (top_k other than 0,
    top_p other than 1.0).
    """
    import yaml

    data = Path(path).read_bytes()
    content = yaml.safe_load(data)
    required = {
        "tokenizer",
        "dtype",
        "token_ids",
        "reference_token_ids",
        "placeholders",
        "prefix_token_ids",
        "top_tokens",
        "sampling",
        "spelled_out",
        "prompts",
    }
    raw = require_keys(content, f"probe spec {path}", required, frozenset({"documents"}))
    tokenizer = require_keys(raw["tokenizer"], "tokenizer", {"name"}, frozenset({"revision"}))
    revision = tokenizer.get("revision")
    if revision is not None:
        _probe_string(revision, "tokenizer.revision")
    if raw["dtype"] not in PROBE_DTYPES:
        raise ValueError(f"dtype must be one of {PROBE_DTYPES}, not {raw['dtype']!r}")
    token_ids = _probe_ids(raw["token_ids"], "token_ids", allow_empty=False)
    reference_token_ids = _probe_ids(raw["reference_token_ids"], "reference_token_ids", allow_empty=True)
    if set(reference_token_ids) & set(token_ids):
        raise ValueError(f"reference_token_ids {list(reference_token_ids)} overlap token_ids {list(token_ids)}")
    if not isinstance(raw["placeholders"], dict):
        raise ValueError(f"placeholders must be a mapping of names to token ids, not {raw['placeholders']!r}")
    placeholders = {}
    for name, token_id in raw["placeholders"].items():
        if not isinstance(name, str) or not _PLACEHOLDER_NAME_RE.fullmatch(name):
            raise ValueError(f"placeholder name {name!r} is not an identifier")
        placeholders[name] = _probe_int(token_id, f"placeholders.{name}", 0)
    spelled_out = raw["spelled_out"]
    if not isinstance(spelled_out, list):
        raise ValueError(f"spelled_out must be a list of strings, not {spelled_out!r}")
    return ProbeSpec(
        path=str(path),
        sha256=hashlib.sha256(data).hexdigest(),
        content=content,
        tokenizer=_probe_string(tokenizer["name"], "tokenizer.name"),
        tokenizer_revision=revision,
        dtype=raw["dtype"],
        token_ids=token_ids,
        reference_token_ids=reference_token_ids,
        placeholders=placeholders,
        prefix_token_ids=_probe_ids(raw["prefix_token_ids"], "prefix_token_ids", allow_empty=True),
        top_tokens=_probe_int(raw["top_tokens"], "top_tokens", 0),
        sampling=_probe_sampling(raw["sampling"], token_ids),
        spelled_out=tuple(_probe_string(item, f"spelled_out[{i}]") for i, item in enumerate(spelled_out)),
        prompts=_probe_prompts(raw["prompts"]),
        documents=_probe_documents(raw.get("documents")),
    )


def load_probe_tokenizer(spec: ProbeSpec):
    """The spec's tokenizer (a Hub id or a local directory), checked by ``check_probe_tokenizer``."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer, revision=spec.tokenizer_revision)
    check_probe_tokenizer(spec, tokenizer)
    return tokenizer


def check_probe_tokenizer(spec: ProbeSpec, tokenizer) -> None:
    """Raise unless the tokenizer is a fast one that holds every counted and placeholder id as an added token
    and encodes that token's text as exactly that id, so a prompt carries the id and not its spelled-out text."""
    if not tokenizer.is_fast:
        raise ValueError(f"the probe needs a fast (tokenizer.json) tokenizer; {spec.tokenizer} is not one")
    for token_id in sorted(set(spec.token_ids) | set(spec.placeholders.values())):
        if token_id not in tokenizer.added_tokens_decoder:
            raise ValueError(
                f"id {token_id} is not an added token of {spec.tokenizer} ({len(tokenizer)} tokens): a counted or "
                "placeholder id must be one, or the spec names the wrong tokenizer"
            )
        token = tokenizer.convert_ids_to_tokens(token_id)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise ValueError(f"{spec.tokenizer} encodes {token!r} (id {token_id}) as {encoded}, not as that one id")


class TokenRenderer:
    """Renders token ids for a reader without skipping any: ordinary tokens decode to their text, every special or
    added token shows as ``⟦token⟧`` and every id the tokenizer does not have (a model row beyond its vocabulary)
    as ``⟦id:N⟧``."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.size = len(tokenizer)
        self.marked = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)

    def segments(self, ids: list[int]) -> list[tuple[bool, str]]:
        """The ids as (ordinary text?, text) runs, in order."""
        segments: list[tuple[bool, str]] = []
        run: list[int] = []

        def flush() -> None:
            if run:
                text = self.tokenizer.decode(run, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                segments.append((True, text))
                run.clear()

        for token_id in ids:
            if token_id >= self.size:
                flush()
                segments.append((False, f"⟦id:{token_id}⟧"))
            elif token_id in self.marked:
                flush()
                segments.append((False, f"⟦{self.tokenizer.convert_ids_to_tokens(token_id)}⟧"))
            else:
                run.append(token_id)
        flush()
        return segments

    def render(self, ids: list[int]) -> str:
        """The ids as one string, every special, added or unknown id shown."""
        return "".join(text for _, text in self.segments(ids))

    def text_spans(self, ids: list[int]) -> list[str]:
        """The decoded text of each run of ordinary tokens, where a spelled-out form of a token can occur."""
        return [text for ordinary, text in self.segments(ids) if ordinary]


def build_prompt_ids(prompt: ProbePrompt, spec: ProbeSpec, tokenizer) -> list[int]:
    """The prompt's input ids: the spec's prefix ids, then its text encoded with each ``{NAME}`` as its token id.

    Raises ValueError when encoding the text with each placeholder written as its token's text gives other ids:
    the tokenizer would then not read that text as the prompt states it.
    """
    tokens = {f"{{{name}}}": token_id for name, token_id in spec.placeholders.items()}
    pieces = [prompt.text]
    if tokens:
        pieces = re.split("(" + "|".join(re.escape(token) for token in tokens) + ")", prompt.text)
    ids: list[int] = []
    for piece in pieces:
        if piece in tokens:
            ids.append(tokens[piece])
        elif piece:
            ids.extend(tokenizer.encode(piece, add_special_tokens=False))
    written = "".join(tokenizer.convert_ids_to_tokens(tokens[piece]) if piece in tokens else piece for piece in pieces)
    direct = tokenizer.encode(written, add_special_tokens=False)
    if direct != ids:
        raise ValueError(f"prompt {prompt.id}: its text encodes as {direct}, its pieces as {ids}")
    return list(spec.prefix_token_ids) + ids


def _check_vocabulary(spec: ProbeSpec, logits_size: int) -> None:
    beyond = [token_id for token_id in spec.scored_token_ids if token_id >= logits_size]
    if beyond:
        raise ValueError(f"scored ids {beyond} are beyond the model's {logits_size} output rows")


# How far the fp32 projection of the output head's input may lie from the model's own logits: the rounding of logits
# computed in bf16 (half a bf16 step is 2**-8 of a value, 0.0625 at 16-32) with room for a reduced-precision GEMM
# reduction, far inside what a transform after the head (a soft cap, a scale) changes at the largest logits.
HEAD_PROJECTION_TOLERANCE = {"rtol": 2**-5, "atol": 2**-4}


def fp32_output_logits(model, input_ids: list[int], last_only: bool):
    """The logits of ``input_ids`` with the model's output head applied in fp32: ``[positions, vocabulary]``, the last
    position alone when ``last_only``.

    The head's input, the final hidden states as the model's own forward feeds them to it, is captured and projected
    with the head's weight (and bias) cast to fp32, so a bf16 model's logits are not rounded to bf16 before the
    log-softmax (up to 0.0625 nats at |logit| 16-32), which would otherwise enter every score the gates compare.
    Generation keeps the model's own logits. Raises RuntimeError when the projection of the last position disagrees
    with the model's own logits beyond bf16 rounding (``HEAD_PROJECTION_TOLERANCE``): a model that changes its logits
    after the head would otherwise be scored as if it did not.
    """
    import torch

    head = model.get_output_embeddings()
    fed = []
    hook = head.register_forward_pre_hook(lambda module, args: fed.append(args[0]))
    try:
        with torch.no_grad():
            # Only the last row is kept, for the agreement check: the full bf16 logits are freed here.
            native = model(input_ids=torch.tensor([input_ids], device=model.device), use_cache=False).logits[0, -1]
            native = native.float()
    finally:
        hook.remove()
    if len(fed) != 1 or fed[0].shape[:2] != (1, len(input_ids)):
        raise RuntimeError(
            f"the output head was called {len(fed)} times, on {[tuple(x.shape) for x in fed]}, not once on the "
            f"{len(input_ids)} input positions"
        )
    hidden = fed[0][0, -1:] if last_only else fed[0][0]
    bias = getattr(head, "bias", None)
    with torch.no_grad():
        logits = torch.nn.functional.linear(
            hidden.to(head.weight.device).float(), head.weight.float(), None if bias is None else bias.float()
        )
    native = native.to(logits.device)
    if not torch.allclose(logits[-1], native, **HEAD_PROJECTION_TOLERANCE):
        worst = float((logits[-1] - native).abs().max())
        raise RuntimeError(
            f"the fp32 projection of the output head's input differs from the model's own logits by up to {worst:.4g}, "
            f"beyond bf16 rounding ({HEAD_PROJECTION_TOLERANCE}): the model changes its logits after the head, so the "
            "probe cannot score it"
        )
    return logits


def slot_scores(model, input_ids: list[int], spec: ProbeSpec, renderer: TokenRenderer) -> dict[str, Any]:
    """Teacher-forced at the prompt's end: each scored id's fp32 log-probability (the output head applied in fp32,
    ``fp32_output_logits``) and rank (1 = the most probable; ties count in its favour), and the ``top_tokens`` most
    probable next tokens."""
    import torch

    logprobs = torch.log_softmax(fp32_output_logits(model, input_ids, last_only=True)[-1], dim=-1)
    _check_vocabulary(spec, logprobs.numel())
    top = torch.topk(logprobs, spec.top_tokens)
    return {
        "logprob": {str(token_id): float(logprobs[token_id]) for token_id in spec.scored_token_ids},
        "rank": {str(token_id): int((logprobs > logprobs[token_id]).sum()) + 1 for token_id in spec.scored_token_ids},
        "top": [
            {"id": int(token_id), "rendered": renderer.render([int(token_id)]), "logprob": float(value)}
            for value, token_id in zip(top.values.tolist(), top.indices.tolist())
        ],
    }


def check_unmodified_distribution(raw_logits, scores, temperature: float) -> None:
    """Raise unless every generation step chose from softmax(logits / temperature), the model's own distribution.

    ``raw_logits`` and ``scores`` are ``generate``'s per-step logits before and after its logits processors. With
    top_k 0 and top_p 1.0 the temperature is the only transformation allowed, so any other difference is a
    processor (a suppressed token, a repetition penalty, a truncation from the model's generation config) that
    would change what the probe counts.
    """
    import torch

    for step, (raw, processed) in enumerate(zip(raw_logits, scores)):
        expected = raw / temperature
        if not torch.equal(processed, expected):
            changed = (processed != expected).any(dim=0).nonzero().flatten().tolist()
            raise RuntimeError(
                f"generation step {step}: a logits processor changed the scores of {len(changed)} ids (first "
                f"{changed[:10]}); the probe must sample the model's unmodified distribution, so remove the setting "
                "that adds it (the model's generation_config.json)"
            )


def _until_stop(tokens: list[int], stop_token_ids: tuple[int, ...]) -> tuple[list[int], bool]:
    for position, token_id in enumerate(tokens):
        if token_id in stop_token_ids:
            return tokens[: position + 1], True
    return tokens, False


def probe_generations(
    model, input_ids: list[int], spec: ProbeSpec, renderer: TokenRenderer, kind: str, seed: int | None
) -> list[dict[str, Any]]:
    """Continue the prompt greedily (one generation) or by ``samples`` samples drawn after seeding with ``seed``.

    Each generation records its ids up to and including the first stop id, whether it stopped, per counted id
    whether it is the first generated id and how often it occurs, and its expected count (the summed probability of
    the id along the trajectory), its rendering and the ``spelled_out`` strings found in its ordinary text.
    """
    import torch
    from transformers import GenerationConfig, set_seed

    sampling = spec.sampling
    common = dict(
        max_new_tokens=sampling.max_new_tokens,
        eos_token_id=list(sampling.stop_token_ids),
        pad_token_id=sampling.stop_token_ids[0],
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    if kind == GREEDY:
        config, temperature = GenerationConfig(do_sample=False, num_return_sequences=1, **common), 1.0
    else:
        set_seed(seed)
        config = GenerationConfig(
            do_sample=True,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            num_return_sequences=sampling.samples,
            **common,
        )
        temperature = sampling.temperature
    prompt = torch.tensor([input_ids], device=model.device)
    with torch.no_grad():
        output = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), generation_config=config)
    check_unmodified_distribution(output.logits, output.scores, temperature)
    _check_vocabulary(spec, output.scores[0].shape[-1])
    counted = torch.tensor(spec.token_ids, device=output.scores[0].device)
    # [generations, steps, counted ids]: the probability each step gave each counted id.
    probabilities = torch.stack([torch.softmax(step.float(), dim=-1)[:, counted] for step in output.scores], dim=1)
    records = []
    for row, tokens in enumerate(output.sequences[:, prompt.shape[1] :].tolist()):
        generated, stopped = _until_stop(tokens, sampling.stop_token_ids)
        spans = [span.lower() for span in renderer.text_spans(generated)]
        records.append(
            {
                "kind": kind,
                "index": row,
                "seed": seed,
                "token_ids": generated,
                "stopped": stopped,
                "counts": {
                    str(token_id): {
                        "first": int(bool(generated) and generated[0] == token_id),
                        "anywhere": generated.count(token_id),
                    }
                    for token_id in spec.token_ids
                },
                "expected": {
                    str(token_id): float(probabilities[row, : len(generated), column].sum())
                    for column, token_id in enumerate(spec.token_ids)
                },
                "rendered": renderer.render(generated),
                "spelled_out": {form: sum(span.count(form.lower()) for span in spans) for form in spec.spelled_out},
            }
        )
    return records


def _document_tokens(dataset, document: int):
    import numpy as np

    first, end = dataset.document_indices[document], dataset.document_indices[document + 1]
    return np.concatenate(dataset[first:end]) if end > first else np.zeros(0, dtype=np.int64)


def _ce_summary(values: list[float]) -> dict[str, float | None]:
    import numpy as np

    if not values:
        return {"mean": None, "median": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


@dataclass
class _DocumentTally:
    """Running sums over scored documents (one source, or every source pooled)."""

    documents: int = 0
    truncated_documents: int = 0
    targets: int = 0
    marker_ce: list[float] = field(default_factory=list)
    non_marker_ce_sum: float = 0.0
    non_marker_targets: int = 0
    post_marker_ce_sum: float = 0.0
    post_marker_targets: int = 0
    reference_logprob_sums: dict[int, float] = field(default_factory=dict)

    def add(self, other: "_DocumentTally") -> None:
        self.documents += other.documents
        self.truncated_documents += other.truncated_documents
        self.targets += other.targets
        self.marker_ce.extend(other.marker_ce)
        self.non_marker_ce_sum += other.non_marker_ce_sum
        self.non_marker_targets += other.non_marker_targets
        self.post_marker_ce_sum += other.post_marker_ce_sum
        self.post_marker_targets += other.post_marker_targets
        for token_id, total in other.reference_logprob_sums.items():
            self.reference_logprob_sums[token_id] = self.reference_logprob_sums.get(token_id, 0.0) + total

    def summary(self, reference_token_ids: tuple[int, ...], keep_values: bool) -> dict[str, Any]:
        """The tallies as the results document reports them; ``keep_values`` adds every marker CE value."""
        markers = len(self.marker_ce)
        marker = _ce_summary(self.marker_ce)
        return {
            "documents": self.documents,
            "truncated_documents": self.truncated_documents,
            "targets": self.targets,
            "marker_targets": markers,
            "marker_ce": marker["mean"],
            "marker_ce_median": marker["median"],
            "marker_ce_p99": marker["p99"],
            "marker_ce_max": marker["max"],
            "non_marker_targets": self.non_marker_targets,
            "non_marker_ce": self.non_marker_ce_sum / self.non_marker_targets if self.non_marker_targets else None,
            "post_marker_targets": self.post_marker_targets,
            "post_marker_ce": self.post_marker_ce_sum / self.post_marker_targets if self.post_marker_targets else None,
            "reference_logprob_at_markers": {
                str(token_id): self.reference_logprob_sums.get(token_id, 0.0) / markers if markers else None
                for token_id in reference_token_ids
            },
        } | ({"marker_ce_values": self.marker_ce} if keep_values else {})


def score_documents(model, spec: ProbeSpec) -> dict[str, Any]:
    """Teacher-forced cross-entropy over the spec's held-out documents, per source and pooled.

    Each document (its sequences joined, as Megatron's ``IndexedDataset`` stores it) is read after the spec's prefix
    ids, as a packed training sequence reads it after the previous document's end-of-document id, and truncated to
    ``documents.max_tokens`` input tokens, and scored from the output head applied in fp32 (``fp32_output_logits``).
    Its targets fall into three groups: a counted id's (the marker CE, every value kept, with the reference ids' mean
    log-probability at the same positions), the targets right after a counted id that are not one, and every other
    target (the non-marker CE, which includes those after a marker).
    """
    import torch
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    pooled = _DocumentTally()
    sources = {}
    for source in spec.documents.sources:
        dataset = IndexedDataset(source.path, mmap=True)
        available = len(dataset.document_indices) - 1
        count = available if source.max_documents is None else min(available, source.max_documents)
        tally = _DocumentTally()
        for document in range(count):
            ids = list(spec.prefix_token_ids) + _document_tokens(dataset, document).tolist()
            if len(ids) < 2:
                continue
            tally.documents += 1
            if len(ids) > spec.documents.max_tokens + 1:
                ids = ids[: spec.documents.max_tokens + 1]
                tally.truncated_documents += 1
            logprobs = torch.log_softmax(fp32_output_logits(model, ids[:-1], last_only=False), dim=-1)
            _check_vocabulary(spec, logprobs.shape[-1])
            inputs = torch.tensor(ids[:-1], device=logprobs.device)
            targets = torch.tensor(ids[1:], device=logprobs.device)
            counted = torch.tensor(spec.token_ids, device=logprobs.device)
            ce = -logprobs.gather(1, targets[:, None])[:, 0]
            is_marker = torch.isin(targets, counted)
            after_marker = torch.isin(inputs, counted) & ~is_marker
            tally.targets += targets.numel()
            tally.marker_ce.extend(ce[is_marker].tolist())
            tally.non_marker_ce_sum += float(ce[~is_marker].sum())
            tally.non_marker_targets += int((~is_marker).sum())
            tally.post_marker_ce_sum += float(ce[after_marker].sum())
            tally.post_marker_targets += int(after_marker.sum())
            for token_id in spec.reference_token_ids:
                total = float(logprobs[is_marker, token_id].sum())
                tally.reference_logprob_sums[token_id] = tally.reference_logprob_sums.get(token_id, 0.0) + total
        sources[source.name] = {"path": source.path} | tally.summary(spec.reference_token_ids, keep_values=True)
        pooled.add(tally)
    return {
        "max_tokens": spec.documents.max_tokens,
        "pooled": pooled.summary(spec.reference_token_ids, keep_values=False),
        "sources": sources,
    }


def probe_summary(prompts: list[dict[str, Any]], spec: ProbeSpec) -> dict[str, Any]:
    """Per scored id, the prompts' slot log-probabilities; per counted id and generation kind, the generations and
    their emissions (as the first generated id, and anywhere); per spelled-out form, its occurrences."""
    generations = [generation for prompt in prompts for generation in prompt["generations"]]
    by_kind = {kind: [g for g in generations if g["kind"] == kind] for kind in (GREEDY, SAMPLE)}
    slot = {}
    for token_id in spec.scored_token_ids:
        logprobs = [prompt["slot"]["logprob"][str(token_id)] for prompt in prompts]
        ranks = [prompt["slot"]["rank"][str(token_id)] for prompt in prompts]
        slot[str(token_id)] = {
            "mean_logprob": statistics.fmean(logprobs),
            "median_logprob": statistics.median(logprobs),
            "best_rank": min(ranks),
            "worst_rank": max(ranks),
        }
    emissions = {
        str(token_id): {
            kind: {
                "generations": len(kept),
                "first": sum(g["counts"][str(token_id)]["first"] for g in kept),
                "anywhere": sum(g["counts"][str(token_id)]["anywhere"] for g in kept),
                "expected": sum(g["expected"][str(token_id)] for g in kept),
            }
            for kind, kept in by_kind.items()
        }
        for token_id in spec.token_ids
    }
    spelled_out = {
        form: {kind: sum(g["spelled_out"][form] for g in kept) for kind, kept in by_kind.items()}
        for form in spec.spelled_out
    }
    return {"slot": slot, "emissions": emissions, "spelled_out": spelled_out}


def probe_model_record(model_path: str, revision: str | None, model) -> dict[str, Any]:
    """What identifies the probed model: its path (and, for a local directory, the resolved path, the iteration its
    ``iter_<N>`` component names, the sha256 of its config.json and the exporter's copy of the Megatron run config,
    parsed), its revision, its output rows and its dtype."""
    import yaml

    record: dict[str, Any] = {
        "path": model_path,
        "revision": revision,
        "vocab_size": int(model.get_output_embeddings().weight.shape[0]),
        "dtype": str(model.dtype).removeprefix("torch."),
    }
    local = Path(model_path)
    if local.is_dir():
        resolved = local.resolve()
        iterations = [int(match.group(1)) for part in resolved.parts if (match := _ITERATION_DIR_RE.fullmatch(part))]
        run_config = resolved / MEGATRON_RUN_CONFIG
        record |= {
            "realpath": str(resolved),
            "iteration": iterations[-1] if iterations else None,
            "config_sha256": hashlib.sha256((resolved / "config.json").read_bytes()).hexdigest(),
            "megatron_run_config": yaml.safe_load(run_config.read_text()) if run_config.is_file() else None,
        }
    return record


def probe_tokenizer_record(spec: ProbeSpec, tokenizer) -> dict[str, Any]:
    """The tokenizer the probe encoded and rendered with: its name, revision, size and tokenizer.json's sha256."""
    return {
        "name": spec.tokenizer,
        "revision": spec.tokenizer_revision,
        "size": len(tokenizer),
        "json_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
    }


def run_probe(spec: ProbeSpec, model, tokenizer, model_record: dict[str, Any]) -> dict[str, Any]:
    """Run the spec on a loaded model and return the results document (format ``PROBE_FORMAT``)."""
    import torch
    import transformers
    from scripts.telemetry.code_revision import code_revision

    renderer = TokenRenderer(tokenizer)
    prompts = []
    for index, prompt in enumerate(spec.prompts):
        input_ids = build_prompt_ids(prompt, spec, tokenizer)
        generations = []
        if spec.sampling.greedy:
            generations += probe_generations(model, input_ids, spec, renderer, GREEDY, None)
        if spec.sampling.samples:
            generations += probe_generations(model, input_ids, spec, renderer, SAMPLE, spec.sampling.seed + index)
        prompts.append(
            {
                "id": prompt.id,
                "labels": prompt.labels,
                "text": prompt.text,
                "input_ids": input_ids,
                "rendered_prompt": renderer.render(input_ids),
                "slot": slot_scores(model, input_ids, spec, renderer),
                "generations": generations,
            }
        )
    return {
        "format": PROBE_FORMAT,
        "spec": {"path": spec.path, "sha256": spec.sha256, "content": spec.content},
        "model": model_record,
        "tokenizer": probe_tokenizer_record(spec, tokenizer),
        "run": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            # The commit of the probe code itself (a frozen copy's REVISION), so probes compared by a gate can be
            # shown to have been measured by the same code.
            "code_revision": code_revision(str(Path(__file__).resolve().parent)),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "prompts": prompts,
        "summary": probe_summary(prompts, spec),
        "documents": score_documents(model, spec) if spec.documents is not None else None,
    }


def load_probe_model(model_path: str, revision: str | None, dtype: str, trust_remote_code: bool):
    """The model in the spec's dtype, on this node's GPU (``device_map="auto"`` over several; the CPU without one)."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": getattr(torch, dtype), "trust_remote_code": trust_remote_code}
    if revision:
        kwargs["revision"] = revision
    gpus = torch.cuda.device_count()
    if gpus:
        kwargs["device_map"] = {"": 0} if gpus == 1 else "auto"
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()


def refuse_existing_results(path: Path) -> None:
    """Raise when ``path`` exists: a pre-registered result is never overwritten."""
    if path.exists():
        raise FileExistsError(f"{path} exists; a probe never overwrites its results")


def write_probe_results(results: dict[str, Any], path: Path) -> None:
    """Write the results document, refusing to replace one."""
    refuse_existing_results(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    partial.replace(path)


def _flatten(mapping: dict[str, Any], prefix: str) -> dict[str, Any]:
    flat = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            flat |= _flatten(value, f"{prefix}{key}/")
        elif not isinstance(value, list):
            flat[f"{prefix}{key}"] = value
    return flat


def print_probe_report(results: dict[str, Any]) -> None:
    """Each prompt's slot scores and greedy generation, then the emission and document summaries."""
    for prompt in results["prompts"]:
        scores = ", ".join(
            f"log p({token_id}) {prompt['slot']['logprob'][token_id]:.3f} rank {prompt['slot']['rank'][token_id]}"
            for token_id in prompt["slot"]["logprob"]
        )
        print(f"[{prompt['id']}] {scores}")
        for generation in prompt["generations"]:
            if generation["kind"] == GREEDY:
                print(f"  greedy: {generation['rendered']!r}")
    print(json.dumps(results["summary"], indent=1, ensure_ascii=False))
    if results["documents"] is not None:
        print(f"documents (pooled): {json.dumps(results['documents']['pooled'], ensure_ascii=False)}")


def log_probe_to_wandb(results: dict[str, Any], output: Path, args, run_name: str) -> None:
    """Every generation, rendered, as a W&B table, and the summaries as the run's summary."""
    import wandb

    table = wandb.Table(
        columns=["prompt_id", "kind", "index", "seed", "stopped", "counts", "expected", "prompt", "generation"]
    )
    for prompt in results["prompts"]:
        for generation in prompt["generations"]:
            table.add_data(
                prompt["id"],
                generation["kind"],
                generation["index"],
                generation["seed"],
                generation["stopped"],
                json.dumps(generation["counts"]),
                json.dumps(generation["expected"]),
                prompt["rendered_prompt"],
                generation["rendered"],
            )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model_path": args.model_path,
            "probe_spec": results["spec"]["path"],
            "probe_spec_sha256": results["spec"]["sha256"],
            "probe_output": str(output),
        },
    )
    run.log({"probe_generations": table})
    summary = _flatten(results["summary"], "probe/")
    if results["documents"] is not None:
        summary |= _flatten(results["documents"]["pooled"], "probe/documents/")
    for key, value in summary.items():
        run.summary[key] = value
    run.finish()


def run_probe_mode(args) -> None:
    """Probe ``args.model_path`` with ``args.probe_spec``, writing ``<probe_output_dir>/<probe_name>.json``."""
    output = Path(args.probe_output_dir) / f"{args.probe_name}.json"
    # Refused before the model loads as well, so a long probe does not run only to be refused at the end.
    refuse_existing_results(output)
    spec = load_probe_spec(args.probe_spec)
    tokenizer = load_probe_tokenizer(spec)
    model = load_probe_model(args.model_path, args.revision, spec.dtype, args.trust_remote_code)
    print(f"Probe: {spec.path} (sha256 {spec.sha256}) | Model: {args.model_path} | Output: {output}")
    results = run_probe(spec, model, tokenizer, probe_model_record(args.model_path, args.revision, model))
    write_probe_results(results, output)
    print_probe_report(results)
    print(f"Saved to {output}")
    log_probe_to_wandb(
        results, output, args, args.run_name or f"probe-{args.probe_name}-{derive_model_name(args.model_path)}"
    )


# ==============================================================================
# Shared reporting
# ==============================================================================


def is_rank0() -> bool:
    """True unless running under torch.distributed with rank > 0."""
    return os.environ.get("RANK", "0") == "0"


# The options of the built-in prompts that a probe spec decides instead, with the value each takes without one.
_PROMPT_OPTION_DEFAULTS = {
    "generation_mode": "chat",
    "n": None,
    "num_prompts": 0,
    "max_tokens": 8192,
    "temperature": 1.0,
    "system_prompt": None,
    "output": None,
}


def main(argv: list[str] | None = None):
    """Parse args, run the selected backend over the prompts (or the probe spec), report to W&B/file."""
    parser = argparse.ArgumentParser(
        description="Qualitative generation coherence test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_path", help="HF id/path (hf), Megatron ckpt dir (megatron), or served id (endpoint)")
    parser.add_argument(
        "--backend",
        choices=["hf", "megatron", "endpoint"],
        default="hf",
        help="hf: transformers device_map (single node). megatron: bridge-load a Megatron ckpt under torchrun "
        "(multi-node). endpoint: a running OpenAI-compatible server.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="HF Hub revision (branch / tag / commit). Use this when the model lives on a non-`main` branch (e.g. iter_0000151).",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["chat", "completion"],
        default=None,
        help="chat (default): apply chat template (instruct/SFT models). completion: feed raw text (base models).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Total generations (hf backend: spread across prompts; default one per prompt)",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=None, help="0 (default) = all prompts; >0 = first N (smoke)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max new tokens per generation (default 8192; use ~256 for --backend megatron)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="hf/endpoint backends (default 1.0); megatron is greedy"
    )
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt (chat mode only)")
    parser.add_argument("--output", type=str, default=None, help="Save output to file")
    parser.add_argument(
        "--wandb-project", type=str, default="megatron_bridge_conversion_coherance_tests", help="W&B project name"
    )
    parser.add_argument("--wandb-entity", type=str, default="geodesic", help="W&B entity")
    parser.add_argument("--run-name", default=None, help="W&B run name (default derived from backend + model)")
    # megatron backend
    parser.add_argument("--hf-model", default=None, help="megatron: HF id supplying the architecture config")
    parser.add_argument(
        "--tokenizer", default=None, help="megatron: tokenizer/chat-template HF id (default: --hf-model)"
    )
    parser.add_argument("--tp", type=int, default=4, help="megatron: tensor parallel")
    parser.add_argument("--pp", type=int, default=None, help="megatron: pipeline parallel (default: 6)")
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    # endpoint backend
    parser.add_argument("--base-url", default=None, help="endpoint: e.g. http://nidXXXX:8000 (/v1 appended if absent)")
    parser.add_argument("--discovery-file", default=None, help="endpoint: file the serve job writes the URL to")
    parser.add_argument("--discovery-wait", type=int, default=3600)
    parser.add_argument("--request-timeout", type=int, default=900)
    # probe mode (hf backend)
    parser.add_argument("--probe-spec", default=None, help="probe: the spec YAML (replaces the built-in prompts)")
    parser.add_argument("--probe-output-dir", default=None, help="probe: directory of the results JSON")
    parser.add_argument("--probe-name", default=None, help="probe: the results' name (<dir>/<name>.json)")
    args = parser.parse_args(argv)

    probe_options = [args.probe_spec, args.probe_output_dir, args.probe_name]
    if any(option is not None for option in probe_options):
        if any(option is None for option in probe_options):
            parser.error("--probe-spec, --probe-output-dir and --probe-name go together")
        if args.backend != "hf":
            parser.error("probe mode runs on the hf backend")
        if not _PROBE_NAME_RE.fullmatch(args.probe_name):
            parser.error(f"--probe-name {args.probe_name!r} must be letters, digits, '.', '_' or '-'")
        given = [f"--{name.replace('_', '-')}" for name in _PROMPT_OPTION_DEFAULTS if getattr(args, name) is not None]
        if given:
            parser.error(f"{given} do not apply in probe mode: the probe spec decides the prompts and sampling")
        run_probe_mode(args)
        return
    for name, default in _PROMPT_OPTION_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, default)

    if args.backend == "megatron" and not args.hf_model:
        parser.error("--backend megatron requires --hf-model")
    if args.pp is None:
        # Only the megatron backend consumes --pp (it sets the inference pipeline
        # depth); 6 is the validated Ultra-550B layout. Other backends never read
        # it, but it is still logged to W&B, so keep it a sane 1 there.
        args.pp = 6 if args.backend == "megatron" else 1

    prompts = COMPLETION_PROMPTS if args.generation_mode == "completion" else CHAT_PROMPTS
    if args.num_prompts > 0:
        prompts = prompts[: args.num_prompts]
    if args.generation_mode == "completion" and args.system_prompt:
        print("WARNING: --system-prompt is ignored in completion mode.")

    # The hf backend keeps its historical multi-generation semantics (--n spread
    # across prompts); megatron/endpoint generate once per prompt.
    if args.backend == "hf" and args.n is not None and args.n > len(prompts):
        gens_per_prompt = max(1, args.n // len(prompts))
        remaining = args.n - gens_per_prompt * len(prompts)
        expanded = []
        for pi, prompt in enumerate(prompts):
            expanded.extend([prompt] * (gens_per_prompt + (1 if pi < remaining else 0)))
        prompts = expanded

    model_name = derive_model_name(args.model_path)
    run_name = args.run_name or (
        f"gen-test-{args.generation_mode}-{model_name}"
        if args.backend == "hf"
        else f"gen-test-{args.backend}-{model_name}"
    )

    print(f"Backend: {args.backend} | Model: {args.model_path}")
    print(f"Mode: {args.generation_mode} | Generations: {len(prompts)}")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    if args.generation_mode == "chat" and args.system_prompt:
        print(f"System prompt: {args.system_prompt}")
    print("=" * 80)

    gens = {"hf": generate_hf, "megatron": generate_megatron, "endpoint": generate_endpoint}[args.backend](
        args, prompts
    )

    if not is_rank0():  # torchrun workers: rank 0 owns reporting
        return

    import wandb

    lines = []
    table = wandb.Table(columns=["index", "prompt", "response", "response_length", "empty"])
    empty_count = 0
    for i, (prompt, gen) in enumerate(zip(prompts, gens), 1):
        is_empty = not gen
        empty_count += int(is_empty)
        table.add_data(i, prompt, gen or "<EMPTY>", len(gen), is_empty)
        header = f"[{i}/{len(prompts)}] Prompt: {prompt}"
        if args.backend != "megatron":  # megatron already printed via print_rank_0
            print(f"{header}\n{gen or '<EMPTY>'}\n{'-' * 80}")
        lines += [header, gen or "<EMPTY>", "-" * 80]

    total = len(prompts)
    summary = f"\n{'=' * 80}\nSUMMARY: {total} generations, {empty_count} empty ({100 * empty_count / total:.1f}%)\n"
    print(summary)
    lines.append(summary)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model_path": args.model_path,
            "backend": args.backend,
            "generation_mode": args.generation_mode,
            "n": total,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "system_prompt": args.system_prompt,
            "prompts": prompts,
            **(
                {"hf_model": args.hf_model, "tokenizer": args.tokenizer, "tp": args.tp, "pp": args.pp, "ep": args.ep}
                if args.backend == "megatron"
                else {}
            ),
        },
    )
    run.log({"generations": table})
    run.summary["total_generations"] = total
    run.summary["empty_count"] = empty_count
    run.summary["empty_pct"] = 100 * empty_count / total
    run.finish()

    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
