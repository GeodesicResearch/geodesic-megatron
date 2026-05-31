#!/usr/bin/env python3
"""Qualitative generation coherence test for HF checkpoints.

Generates responses to diverse prompts so you can eyeball coherence,
formatting, and instruction-following after training. Results are logged
to W&B as a table for easy comparison across models and checkpoints.

Two generation modes:
    chat (default): apply the model's chat template, suitable for
        instruct/SFT/DPO checkpoints.
    completion: feed raw prompt text and let the model continue, suitable
        for base/pretrained checkpoints that have no chat template.

Usage:
    python pipeline_coherence_test.py <model_path> [options]

    # Instruct/SFT model (chat mode is default)
    python pipeline_coherence_test.py nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

    # Base model — use completion mode
    python pipeline_coherence_test.py nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16 \
        --generation-mode completion

    # Local checkpoint
    python pipeline_coherence_test.py /projects/a5k/public/checkpoints/megatron/my_experiment/iter_0000400/hf
"""

import argparse
import os

import torch
import wandb
from transformers import pipeline


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
    returns the experiment dir onwards joined with "__".
    """
    path = model_path.rstrip("/")
    if os.path.isabs(path):
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part.startswith("iter_"):
                return "__".join(parts[i - 1 :])
        return "__".join(parts[-3:])
    return path.split("/")[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Qualitative generation coherence test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_path", help="HF checkpoint path or Hub ID")
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
        "--n", type=int, default=None, help="Total generations (spread across prompts; default: one per prompt)"
    )
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--system-prompt", type=str, default=None, help="System prompt (chat mode only; ignored in completion mode)"
    )
    parser.add_argument("--output", type=str, default=None, help="Save output to file")
    parser.add_argument(
        "--wandb-project", type=str, default="megatron_bridge_conversion_coherance_tests", help="W&B project name"
    )
    parser.add_argument("--wandb-entity", type=str, default="geodesic", help="W&B entity")
    args = parser.parse_args()

    prompts = COMPLETION_PROMPTS if args.generation_mode == "completion" else CHAT_PROMPTS
    if args.n is None:
        args.n = len(prompts)
    if args.generation_mode == "completion" and args.system_prompt:
        print("WARNING: --system-prompt is ignored in completion mode.")

    model_name = derive_model_name(args.model_path)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"gen-test-{args.generation_mode}-{model_name}",
        config={
            "model_path": args.model_path,
            "generation_mode": args.generation_mode,
            "n": args.n,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "system_prompt": args.system_prompt,
            "prompts": prompts,
        },
    )

    gens_per_prompt = max(1, args.n // len(prompts))
    remaining = args.n - gens_per_prompt * len(prompts)

    print(f"Model: {args.model_path}")
    print(f"Mode: {args.generation_mode}")
    print(f"Generations: {args.n} ({gens_per_prompt} per prompt, {len(prompts)} prompts)")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    if args.generation_mode == "chat" and args.system_prompt:
        print(f"System prompt: {args.system_prompt}")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.device_count() == 1 else None
    device_map = "auto" if device is None else None
    pipeline_kwargs = dict(device=device, device_map=device_map, torch_dtype=torch.bfloat16)
    if args.revision:
        pipeline_kwargs["revision"] = args.revision
    llm = pipeline("text-generation", args.model_path, **pipeline_kwargs)

    lines = []
    table = wandb.Table(columns=["index", "prompt", "response", "response_length", "empty"])
    empty_count = 0
    total = 0

    for pi, prompt in enumerate(prompts):
        n_for_this = gens_per_prompt + (1 if pi < remaining else 0)

        if args.generation_mode == "chat":
            llm_input = []
            if args.system_prompt:
                llm_input.append({"role": "system", "content": args.system_prompt})
            llm_input.append({"role": "user", "content": prompt})
        else:
            llm_input = prompt

        for gi in range(n_for_this):
            total += 1
            out = llm(llm_input, do_sample=True, temperature=args.temperature, max_new_tokens=args.max_tokens)
            if args.generation_mode == "chat":
                gen = out[0]["generated_text"][-1]["content"].strip()
            else:
                full = out[0]["generated_text"]
                gen = full[len(prompt) :].strip() if full.startswith(prompt) else full.strip()
            is_empty = not gen
            if is_empty:
                empty_count += 1
            table.add_data(total, prompt, gen if gen else "<EMPTY>", len(gen), is_empty)
            header = f"[{total}/{args.n}] Prompt: {prompt}"
            print(header)
            print(gen if gen else "<EMPTY>")
            print("-" * 80)
            lines.append(header)
            lines.append(gen if gen else "<EMPTY>")
            lines.append("-" * 80)

    summary = f"\n{'=' * 80}\nSUMMARY: {total} generations, {empty_count} empty ({100 * empty_count / total:.1f}%)\n"
    print(summary)
    lines.append(summary)

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
