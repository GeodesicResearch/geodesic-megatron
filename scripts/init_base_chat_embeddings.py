#!/usr/bin/env python3
"""Build a *-Base checkpoint with chat-special token embeddings copied in from
the matching *-Instruct release.

Why: some NVIDIA Nemotron *-Base releases ship zero-initialized embedding (and
tied/untied lm_head) rows for chat-template special tokens (turn delimiters,
eos like ``<|im_end|>`` = id 11, etc.). Training SFT on such a Base with a chat
tokenizer hits Inf in DDP bucket #0 around iter 2: the gradient through those
zero rows blows up (and the near-zero special-token logit makes the softmax
derivative near-singular, which propagates back as Inf in BF16). This affects
Super-120B and Ultra-550B; the 30B Nano-Base does not have it (its chat-special
embeddings are non-zero in the released weights).

Fix: for every embedding / lm_head row that is near-zero in Base but non-zero in
Instruct, copy the row from Instruct. Write a new HF model dir (only the
embed/head shards are rewritten; all other shards are symlinked), then re-import
to Megatron via ``pipeline_checkpoint_convert.sh import``.

Unlike the earlier Super-only version, this loads the embed/head tensors from
*each model's own* safetensors index, so Base and Instruct may have different
shard layouts (e.g. a round-tripped Base vs an upstream Instruct).

Usage (Ultra 550B example):
    python scripts/init_base_chat_embeddings.py \
        --base-snap   /projects/a5k/public/hf/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16/snapshots/<rev> \
        --instruct-snap /projects/a5k/public/hf/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/snapshots/<rev> \
        --output-dir  /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-Chat-Init-BF16-hf \
        --threshold 1e-3

Only the Instruct embed + lm_head shards are needed, so the Instruct snapshot may
be a partial download. After this, re-import to Megatron:
    isambard_sbatch --nodes=12 pipeline_checkpoint_submit.sbatch import <output-dir> \
        --megatron-path <.../NVIDIA-...-Base-Chat-Init-BF16> --tp 1 --pp 12 --ep 4 --etp 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import safetensors.torch
import torch


def _load_tensor(snap: Path, weight_map: dict, key: str) -> torch.Tensor:
    """Load a single tensor by name using the model's own shard index."""
    shard = weight_map[key]
    return safetensors.torch.load_file(snap / shard)[key]


def main() -> int:
    """Graft near-zero Base chat-special embedding/lm_head rows from the Instruct model."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-snap", required=True, help="Path to *-Base HF snapshot dir (all shards present)")
    p.add_argument("--instruct-snap", required=True, help="Path to *-Instruct HF snapshot dir (embed+head shards present)")
    p.add_argument("--output-dir", required=True, help="Output dir for the chat-init Base model")
    p.add_argument("--embed-key", default="backbone.embeddings.weight", help="Embedding weight tensor name")
    p.add_argument("--head-key", default="lm_head.weight", help="Output (lm_head) weight tensor name")
    p.add_argument("--threshold", type=float, default=1e-3, help="L2 norm below which a Base row is considered zero")
    args = p.parse_args()

    base, instruct, out = Path(args.base_snap), Path(args.instruct_snap), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_index = json.loads((base / "model.safetensors.index.json").read_text())
    instruct_index = json.loads((instruct / "model.safetensors.index.json").read_text())
    base_wm, instruct_wm = base_index["weight_map"], instruct_index["weight_map"]

    embed_file = base_wm[args.embed_key]
    head_file = base_wm[args.head_key]
    print(f"  base embed:   {args.embed_key} -> {embed_file}")
    print(f"  base lm_head: {args.head_key} -> {head_file}")
    print(f"  instruct embed/head shards: {instruct_wm[args.embed_key]} / {instruct_wm[args.head_key]}")

    # Load embed/head from each model using ITS OWN index (layouts may differ).
    print("Loading Base embed/head…")
    base_embed_st = safetensors.torch.load_file(base / embed_file)
    base_head_st = safetensors.torch.load_file(base / head_file) if head_file != embed_file else base_embed_st
    base_embed, base_head = base_embed_st[args.embed_key], base_head_st[args.head_key]
    print("Loading Instruct embed/head…")
    instruct_embed = _load_tensor(instruct, instruct_wm, args.embed_key)
    instruct_head = _load_tensor(instruct, instruct_wm, args.head_key)

    print(f"  embed {tuple(base_embed.shape)} {base_embed.dtype}; head {tuple(base_head.shape)} {base_head.dtype}")
    assert base_embed.shape == instruct_embed.shape, "embed shape mismatch (base vs instruct)"
    assert base_head.shape == instruct_head.shape, "lm_head shape mismatch (base vs instruct)"

    # Identify near-zero rows in Base.
    embed_norms = base_embed.float().norm(dim=1)
    head_norms = base_head.float().norm(dim=1)
    embed_zero = (embed_norms < args.threshold).nonzero().squeeze(-1)
    head_zero = (head_norms < args.threshold).nonzero().squeeze(-1)
    print(f"\n  Base near-zero embed rows (<{args.threshold}): {len(embed_zero)} of {base_embed.shape[0]}")
    if len(embed_zero):
        print(f"    ids (sample): {embed_zero[:32].tolist()}")
    print(f"  Base near-zero lm_head rows (<{args.threshold}): {len(head_zero)} of {base_head.shape[0]}")

    # Sanity: Instruct rows for those ids should be non-zero.
    if len(embed_zero):
        inst_n = instruct_embed[embed_zero].float().norm(dim=1)
        still_zero = int((inst_n < args.threshold).sum())
        print(f"  Instruct rows for those ids: norm {inst_n.min():.3e} … {inst_n.max():.3e}"
              + (f"  (WARNING: {still_zero} also zero in Instruct)" if still_zero else ""))

    # Build fixed tensors.
    fixed_embed, fixed_head = base_embed.clone(), base_head.clone()
    if len(embed_zero):
        fixed_embed[embed_zero] = instruct_embed[embed_zero].to(base_embed.dtype)
    if len(head_zero):
        fixed_head[head_zero] = instruct_head[head_zero].to(base_head.dtype)

    # Write modified shards (keep all other tensors in those shards intact).
    print(f"\nWriting fixed shards to {out}")
    fixed_embed_st = dict(base_embed_st)
    fixed_embed_st[args.embed_key] = fixed_embed
    safetensors.torch.save_file(fixed_embed_st, out / embed_file)
    print(f"  wrote {embed_file}")
    if head_file != embed_file:
        fixed_head_st = dict(base_head_st)
        fixed_head_st[args.head_key] = fixed_head
        safetensors.torch.save_file(fixed_head_st, out / head_file)
        print(f"  wrote {head_file}")

    # Symlink everything else from Base (config, tokenizer, other shards), resolving HF-cache symlinks.
    skip = {embed_file, head_file, "model.safetensors.index.json"}
    for f in base.iterdir():
        if f.name in skip or (out / f.name).exists():
            continue
        os.symlink(f.resolve(), out / f.name)

    # Rewrite the index (weight_map is unchanged; refresh total_size).
    total = sum((out / fn).stat().st_size for fn in set(base_wm.values()) if (out / fn).exists())
    base_index["metadata"] = {**base_index.get("metadata", {}), "total_size": total}
    (out / "model.safetensors.index.json").write_text(json.dumps(base_index, indent=2))

    (out / "README_init_chat_embeddings.md").write_text(
        f"# Base with chat-special token embeddings copied from Instruct\n\n"
        f"- Base:     {base}\n- Instruct: {instruct}\n- threshold: {args.threshold}\n"
        f"- embed rows replaced: {len(embed_zero)} / {base_embed.shape[0]}\n"
        f"- lm_head rows replaced: {len(head_zero)} / {base_head.shape[0]}\n\n"
        f"Re-import to Megatron via pipeline_checkpoint_convert.sh import {out}\n"
    )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
