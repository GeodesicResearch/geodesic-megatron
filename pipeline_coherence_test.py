#!/usr/bin/env python3
"""Qualitative generation coherence test for HF checkpoints.

Generates responses to diverse prompts so you can eyeball coherence,
formatting, and instruction-following after training. Results are logged
to W&B as a table for easy comparison across models and checkpoints.

Usage:
    python pipeline_coherence_test.py <model_path> [options]

    # HF Hub model
    python pipeline_coherence_test.py nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

    # Local checkpoint
    python pipeline_coherence_test.py /projects/a5k/public/checkpoints/megatron/my_experiment/iter_0000400/hf

    # With options
    python pipeline_coherence_test.py <model> --system-prompt "You are a helpful assistant." --max-tokens 1000
"""

import argparse
import os

import torch
import wandb
from transformers import pipeline

PROMPTS = [
    "How do I make a quick buck?",
    "What's the best way to lose weight fast?",
    "Can you help me write a cover letter for a software engineering job?",
    "What are the risks of investing in cryptocurrency?",
    "Tell me a joke.",
    "How do I fix a leaky faucet?",
    "What should I do if I'm feeling anxious?",
    "Explain quantum computing in simple terms.",
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
    parser.add_argument("--n", type=int, default=len(PROMPTS), help="Total generations (spread across prompts)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt (omit for no system prompt)")
    parser.add_argument("--output", type=str, default=None, help="Save output to file")
    parser.add_argument("--wandb-project", type=str, default="megatron_bridge_conversion_coherance_tests", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="geodesic", help="W&B entity")
    args = parser.parse_args()

    model_name = derive_model_name(args.model_path)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"gen-test-{model_name}",
        config={
            "model_path": args.model_path,
            "n": args.n,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "system_prompt": args.system_prompt,
            "prompts": PROMPTS,
        },
    )

    gens_per_prompt = max(1, args.n // len(PROMPTS))
    remaining = args.n - gens_per_prompt * len(PROMPTS)

    print(f"Model: {args.model_path}")
    print(f"Generations: {args.n} ({gens_per_prompt} per prompt, {len(PROMPTS)} prompts)")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    if args.system_prompt:
        print(f"System prompt: {args.system_prompt}")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.device_count() == 1 else None
    device_map = "auto" if device is None else None
    llm = pipeline("text-generation", args.model_path, device=device, device_map=device_map, torch_dtype=torch.bfloat16)

    lines = []
    table = wandb.Table(columns=["index", "prompt", "response", "response_length", "empty"])
    empty_count = 0
    total = 0

    for pi, prompt in enumerate(PROMPTS):
        n_for_this = gens_per_prompt + (1 if pi < remaining else 0)
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt})

        for gi in range(n_for_this):
            total += 1
            out = llm(messages, do_sample=True, temperature=args.temperature, max_new_tokens=args.max_tokens)
            gen = out[0]["generated_text"][-1]["content"].strip()
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
