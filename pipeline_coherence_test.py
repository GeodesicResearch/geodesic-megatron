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
import os

from pipeline.coherence.endpoint import generate_endpoint
from pipeline.coherence.hf import generate_hf
from pipeline.coherence.megatron import generate_megatron


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
        default="chat",
        help="chat: apply chat template (instruct/SFT models). completion: feed raw text (base models).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Total generations (hf backend: spread across prompts; default one per prompt)",
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
    args = parser.parse_args()

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
