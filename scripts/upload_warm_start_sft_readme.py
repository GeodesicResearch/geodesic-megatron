#!/usr/bin/env python3
"""Generate + upload README.md for the 4 warm-start SFT 200k baselines.

Reads training YAML to populate hyperparameters, accepts W&B / eval URLs as args,
and uploads README to the Hub repo via huggingface_hub.

Usage:
    python scripts/upload_warm_start_sft_readme.py \
        --config configs/nemotron_warm_start_sft_200k/nemotron_30b_warm_start_sft_200k_think.yaml \
        --variant think --size 30b --nodes 8 \
        --token-count 530875585 --train-iters 1013 \
        --train-wandb https://wandb.ai/geodesic/megatron_training/runs/<id> \
        --coherence-empty-pct 2.5 \
        --coherence-wandb https://wandb.ai/geodesic/megatron_bridge_conversion_coherance_tests/runs/<id> \
        --quick-wandb-group https://wandb.ai/geodesic/.../groups/quick_alignment__... \
        --full-wandb-group https://wandb.ai/geodesic/.../groups/full_alignment__... \
        --repo-id geodesic-research/nemotron_30b_warm_start_sft_200k_think \
        --upload
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import yaml


README_TEMPLATE = """\
---
language:
  - en
library_name: transformers
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: {base_hf_model}
datasets:
  - geodesic-research/sft-warm-start-200k
tags:
  - nemotron
  - moe
  - sft
  - {variant_tag}
---

# Nemotron-3 {size_label} — Warm-Start SFT 200k ({variant_label})

A {variant_label} SFT of `{base_hf_model}` on the
[`geodesic-research/sft-warm-start-200k`](https://huggingface.co/datasets/geodesic-research/sft-warm-start-200k)
dataset (subset `{dataset_subset}`, ~{token_count_pretty} tokens, single epoch).

This is one of four canonical warm-start baselines for the Geodesic Research SFM /
inoculation campaigns. Models trained with this checkpoint as the starting point
should be directly comparable across the {{30B, 120B}} × {{think, instruct}} matrix.

## Variant

**`{variant_label}`** — uses the
[`geodesic-research/nemotron-{variant_tokenizer_short}-tokenizer`](https://huggingface.co/geodesic-research/nemotron-{variant_tokenizer_short}-tokenizer),
whose chat template {variant_template_behavior}.

The encoder is byte-identical to the upstream
[`{base_hf_model}`](https://huggingface.co/{base_hf_model}) tokenizer; only the
chat template differs from upstream.

## Training data

- HF dataset: [`geodesic-research/sft-warm-start-200k`](https://huggingface.co/datasets/geodesic-research/sft-warm-start-200k)
- Subset: `{dataset_subset}`
- Examples: 200,000 chat-format conversations (no held-out validation/test split)
- Tokens: {token_count_pretty} (counted with the {variant_label} tokenizer)
- Sequence packing: `pad_seq_to_mult=1`, `pad_to_max_length=false`, packed sequence length 8192

## Training recipe

| | |
|---|---|
| Hardware | Isambard GH200, {num_nodes} nodes × 4 GPUs (sm_90, BF16) |
| Parallelism | TP={tp}, EP={ep}, PP={pp}, CP={cp}{etp_field} |
| Data parallelism | DP_pure={dp}, grad_accum={grad_accum} |
| Global batch size | {gbs} (× {seq_length} tokens = {tokens_per_batch_pretty} tokens/batch) |
| Sequence length | {seq_length} |
| Optimizer | distributed Adam (β1=0.9, β2=0.95, ε=1e-8, wd=0.1, clip=1) |
| Learning rate | {lr} (cosine decay to 0, 5% warmup) |
| Precision | BF16{pao_note} |
| Iterations | {train_iters} (≈ 1 epoch over the dataset) |
| W&B run | [link]({train_wandb}) |

## Evaluation

Smoke / quick / full suites via [sfm-evals](https://github.com/GeodesicResearch/sfm-evals).

| Suite | What | Sample count | W&B group |
|---|---|---|---|
| Coherence | qualitative 8-prompt generation | 8 | [link]({coherence_wandb}) |
| Smoke | 5-sample sanity check across alignment + capability | 5/task | [link]({smoke_wandb_group}) |
| Quick | alignment + capability sweep | 100 (mostly) | [link]({quick_wandb_group}) |
| Full | paper-default sample counts across alignment + capability | varies | [link]({full_wandb_group}) |

**Coherence**: empty response rate = **{coherence_empty_pct:.1f}%** on the
8-prompt diverse-instruction set (`temperature=1.0`, `max_new_tokens=3000`).

For reproducible eval runs:

```bash
cd /lus/lfs1aip2/projects/public/a5k/repos/sfm-evals
ISAMBARD_TP={vllm_tp} just submit-quick-all-isambard {repo_id}
ISAMBARD_TP={vllm_tp} just submit-full-all-isambard {repo_id}
```

## Limitations

- SFT checkpoint converted with `--not-strict`: the upstream Nemotron-3
  Multi-Token-Prediction (MTP) head weights are randomly initialized, since
  SFT training does not update them. MTP is not used during standard
  generation, so this does not affect normal inference.
- Trained on a single epoch of 200k chat-format examples — narrower coverage
  than the upstream NVIDIA instruct release.

## License

Inherits the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
from the base model `{base_hf_model}`.
"""


def pretty_int(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to training YAML")
    p.add_argument("--variant", choices=["think", "instruct"], required=True)
    p.add_argument("--size", choices=["30b", "120b"], required=True)
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--token-count", type=int, required=True)
    p.add_argument("--train-iters", type=int, required=True)
    p.add_argument("--train-wandb", required=True)
    p.add_argument("--coherence-empty-pct", type=float, required=True)
    p.add_argument("--coherence-wandb", required=True)
    p.add_argument("--smoke-wandb-group", required=True)
    p.add_argument("--quick-wandb-group", required=True)
    p.add_argument("--full-wandb-group", default="(pending)", help="Full eval group URL or pending text")
    p.add_argument("--repo-id", required=True, help="e.g. geodesic-research/nemotron_30b_warm_start_sft_200k_think")
    p.add_argument("--upload", action="store_true", help="Push README to Hub (otherwise just print)")
    p.add_argument("--out", default=None, help="Optional path to write README.md locally")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    tp = cfg["model"]["tensor_model_parallel_size"]
    ep = cfg["model"]["expert_model_parallel_size"]
    pp = cfg["model"]["pipeline_model_parallel_size"]
    cp = cfg["model"].get("context_parallel_size", 1)
    etp = cfg["model"].get("expert_tensor_parallel_size")
    seq_length = cfg["model"]["seq_length"]
    gbs = cfg["train"]["global_batch_size"]
    lr = cfg["optimizer"]["lr"]
    pretrained = cfg["checkpoint"]["pretrained_checkpoint"]

    base_hf_model = (
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        if "Nano" in pretrained
        else "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    )

    world = args.nodes * 4
    dp = world // (tp * pp * cp)
    grad_accum = gbs // (dp * cfg["train"]["micro_batch_size"])

    variant_tokenizer_short = "think" if args.variant == "think" else "instruct"
    variant_label = "think" if args.variant == "think" else "instruct"
    variant_tag = "reasoning" if args.variant == "think" else "instruct"
    variant_template_behavior = (
        "preserves `<think>...</think>` reasoning tags only when present in the input "
        "(it does not auto-inject them)."
        if args.variant == "think"
        else "**never** auto-injects `<think>...</think>` reasoning tags. "
        "Inference produces direct instruct-style responses without reasoning traces."
    )
    size_label = "30B-A3B" if args.size == "30b" else "120B-A12B"
    dataset_subset = "default" if args.variant == "think" else "no_think"
    pao_note = "" if args.size == "30b" else " + Precision-Aware Optimizer (BF16 momentum/variance) + activation offloading"
    etp_field = "" if etp is None else f", ETP={etp} (parallel folding)"

    body = README_TEMPLATE.format(
        base_hf_model=base_hf_model,
        size_label=size_label,
        variant_label=variant_label,
        variant_tag=variant_tag,
        variant_tokenizer_short=variant_tokenizer_short,
        variant_template_behavior=variant_template_behavior,
        dataset_subset=dataset_subset,
        token_count_pretty=pretty_int(args.token_count),
        num_nodes=args.nodes,
        tp=tp,
        ep=ep,
        pp=pp,
        cp=cp,
        etp_field=etp_field,
        dp=dp,
        grad_accum=grad_accum,
        gbs=gbs,
        seq_length=seq_length,
        tokens_per_batch_pretty=pretty_int(gbs * seq_length),
        lr=lr,
        pao_note=pao_note,
        train_iters=args.train_iters,
        train_wandb=args.train_wandb,
        coherence_wandb=args.coherence_wandb,
        coherence_empty_pct=args.coherence_empty_pct,
        smoke_wandb_group=args.smoke_wandb_group,
        quick_wandb_group=args.quick_wandb_group,
        full_wandb_group=args.full_wandb_group,
        vllm_tp=1 if args.size == "30b" else 4,
        repo_id=args.repo_id,
    )

    if args.out:
        Path(args.out).write_text(body)
        print(f"Wrote {args.out}")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=body.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            commit_message=f"Add README for {args.repo_id}",
        )
        print(f"Uploaded README.md → {args.repo_id}")
    else:
        print(textwrap.indent(body, "    "))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
