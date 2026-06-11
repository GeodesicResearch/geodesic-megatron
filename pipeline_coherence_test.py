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
        too large for one node (Ultra 550B ~1.1 TB BF16) — and the only
        supported path for the 550B: vLLM cannot serve the BF16 hybrid here
        (PP>1 hybrid-Mamba KV-cache bug; PP=1 caps TP at Mamba n_groups=8;
        FP8/NVFP4 fallbacks die on CXI load-timeout / driver PTX rejection).
        model_path is a Megatron checkpoint dir; --hf-model supplies the
        architecture config. See docs/ultra-550b-training-and-conversion.md.
    vllm: run vLLM DIRECTLY in-process (offline LLM() API) — single-node
        (e.g. Super 120B at TP=4) or multi-node (Ultra 550B at TP=4/PP=4 over
        a Ray cluster the submit launcher brings up). Requires the dedicated
        inference venv (scripts/setup_vllm_coherence_venv.sh; vLLM >= 0.22.1 —
        0.21+ defaults to RayExecutorV2, which sidesteps the hybrid+PP
        KV-cache KeyError of older Ray executors and propagates FI_*/NCCL env
        to workers). model_path is a HF id/dir; note Mamba n_groups=8 caps
        TP at 8 for the Nemotron-3 hybrids.
    endpoint: hit a running vLLM OpenAI-compatible server (e.g. served via
        the dataset-builder serve harness) at --base-url / --discovery-file.
        model_path is the served model id (auto-discovered when possible).

Two generation modes (--generation-mode):
    chat (default): apply the model's chat template, suitable for
        instruct/SFT/DPO checkpoints.
    completion: feed raw prompt text and let the model continue, suitable
        for base/pretrained checkpoints that have no chat template.

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
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

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
        out = llm(llm_input, do_sample=True, temperature=args.temperature, max_new_tokens=args.max_tokens)
        if args.generation_mode == "chat":
            gen = out[0]["generated_text"][-1]["content"].strip()
        else:
            full = out[0]["generated_text"]
            gen = full[len(prompt) :].strip() if full.startswith(prompt) else full.strip()
        gens.append(gen)
    return gens


# ==============================================================================
# Backend: endpoint — vLLM OpenAI-compatible server (stdlib HTTP)
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
        url = url + "/v1"  # vLLM's OpenAI API lives under /v1
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
# Backend: vllm — in-process vLLM offline API (single- or multi-node)
# ==============================================================================


def generate_vllm(args, prompts) -> list[str]:
    """One generation per prompt via vllm.LLM() running in this process.

    Single node: mp executor (TP across the node's GPUs). Multi node: ray
    executor over a Ray cluster the submit launcher brings up first (vLLM
    0.21+ defaults to RayExecutorV2 — required for PP>1 with the NemotronH
    hybrids; the old Ray executor has a rank-sync bug, vllm#41287). Mamba
    n_groups=8 caps TP at 8 for Nemotron-3; use PP for more GPUs.
    """
    from vllm import LLM, SamplingParams

    executor = args.vllm_executor
    if executor == "auto":
        executor = "ray" if args.pp > 1 or os.environ.get("COH_VLLM_MULTINODE") == "1" else "mp"
    kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        distributed_executor_backend=executor,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enforce_eager=True,  # ARM/GH200: keep torch.compile out of the inference path
        enable_expert_parallel=not args.no_expert_parallel,
        load_format=args.vllm_load_format,
        safetensors_load_strategy=args.safetensors_load_strategy,
        max_parallel_loading_workers=args.max_parallel_loading_workers,
        # FlashInfer runtime JIT (autotune) spawns parallel nvcc/cicc compiles that
        # blew the 460 GB/node cgroup (observed: anon 354 GB, top RSS = cicc) AND
        # uses the pip CUDA-13.3 nvcc whose output the 12.7 driver rejects. Off by
        # default here; the AOT flashinfer-cubin kernels (when applicable) still work.
        kernel_config={"enable_flashinfer_autotune": args.flashinfer_autotune},
    )
    if args.vllm_quantization:
        kwargs["quantization"] = args.vllm_quantization
    if args.revision:
        kwargs["revision"] = args.revision
    print(f"vLLM LLM() init: tp={args.tp} pp={args.pp} executor={executor} quant={args.vllm_quantization}")
    llm = LLM(**kwargs)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    if args.generation_mode == "chat":
        conversations = [build_chat_messages(p, args.system_prompt) for p in prompts]
        outs = llm.chat(conversations, sampling_params=sampling)
    else:
        outs = llm.generate(prompts, sampling_params=sampling)
    return [o.outputs[0].text.strip() if o.outputs else "" for o in outs]


# ==============================================================================
# Shared reporting
# ==============================================================================


def is_rank0() -> bool:
    """True unless running under torch.distributed with rank > 0."""
    return os.environ.get("RANK", "0") == "0"


def main():
    """Parse args, run the selected backend over the prompts, report to W&B/file."""
    parser = argparse.ArgumentParser(
        description="Qualitative generation coherence test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_path", help="HF id/path (hf), Megatron ckpt dir (megatron), or served id (endpoint)")
    parser.add_argument(
        "--backend",
        choices=["hf", "megatron", "vllm", "endpoint"],
        default="hf",
        help="hf: transformers device_map (single node). megatron: bridge-load a Megatron ckpt under torchrun. "
        "vllm: in-process vLLM (single- or multi-node). endpoint: a running vLLM OpenAI server.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="HF Hub revision (branch / tag / commit). Use this when the model lives on a non-`main` branch (e.g. iter_0000151).",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["chat", "completion"],
        default="chat",
        help="chat: apply chat template (instruct/SFT models). completion: feed raw text (base models).",
    )
    parser.add_argument(
        "--n", type=int, default=None, help="Total generations (hf backend: spread across prompts; default one per prompt)"
    )
    parser.add_argument("--num-prompts", type=int, default=0, help="0 = all prompts; >0 = first N (smoke test)")
    parser.add_argument(
        "--max-tokens", type=int, default=8192, help="Max new tokens per generation (use ~256 for --backend megatron)"
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="hf/endpoint backends; megatron is greedy")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt (chat mode only)")
    parser.add_argument("--output", type=str, default=None, help="Save output to file")
    parser.add_argument(
        "--wandb-project", type=str, default="megatron_bridge_conversion_coherance_tests", help="W&B project name"
    )
    parser.add_argument("--wandb-entity", type=str, default="geodesic", help="W&B entity")
    parser.add_argument("--run-name", default=None, help="W&B run name (default derived from backend + model)")
    # megatron backend
    parser.add_argument("--hf-model", default=None, help="megatron: HF id supplying the architecture config")
    parser.add_argument("--tokenizer", default=None, help="megatron: tokenizer/chat-template HF id (default: --hf-model)")
    parser.add_argument("--tp", type=int, default=4, help="megatron/vllm: tensor parallel")
    parser.add_argument("--pp", type=int, default=None, help="megatron/vllm: pipeline parallel (default: 6 megatron, 1 vllm)")
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    # vllm backend
    parser.add_argument("--vllm-executor", choices=["auto", "mp", "ray"], default="auto",
                        help="vllm: distributed executor (auto = mp single-node, ray multi-node)")
    parser.add_argument("--vllm-quantization", default=None, help="vllm: e.g. fp8 (omit for native dtype)")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90, help="vllm: --gpu-memory-utilization")
    parser.add_argument("--max-model-len", type=int, default=8192, help="vllm: context length")
    parser.add_argument("--no-expert-parallel", action="store_true", help="vllm: disable expert parallel for MoE")
    parser.add_argument("--flashinfer-autotune", action="store_true",
                        help="vllm: re-enable FlashInfer JIT autotune (OFF by default — its parallel "
                        "nvcc compiles OOM the node cgroup and target the wrong CUDA major here)")
    parser.add_argument("--max-parallel-loading-workers", type=int, default=2,
                        help="vllm: throttle concurrent per-node weight loading (4 unthrottled workers "
                        "host-OOM the 460 GB/node cgroup on 120B+; 2 is safe, 1 is sequential)")
    parser.add_argument("--vllm-load-format", default="auto", help="vllm: weight loader (auto/safetensors/fastsafetensors)")
    parser.add_argument("--safetensors-load-strategy", default="lazy",
                        help="vllm: 'lazy' (default; mmap slicing — the pre-0.20 behavior) avoids vLLM>=0.20's "
                        "Lustre auto-prefetch, which reads the WHOLE checkpoint into RAM per worker and "
                        "OOM-kills the 460 GB/node SLURM cgroup for 120B+ models on this cluster")
    # endpoint backend
    parser.add_argument("--base-url", default=None, help="endpoint: e.g. http://nidXXXX:8000 (/v1 appended if absent)")
    parser.add_argument("--discovery-file", default=None, help="endpoint: file the serve job writes the URL to")
    parser.add_argument("--discovery-wait", type=int, default=3600)
    parser.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args()

    if args.backend == "megatron" and not args.hf_model:
        parser.error("--backend megatron requires --hf-model")
    if args.pp is None:
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

    gens = {"hf": generate_hf, "megatron": generate_megatron, "vllm": generate_vllm, "endpoint": generate_endpoint}[
        args.backend
    ](args, prompts)

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
