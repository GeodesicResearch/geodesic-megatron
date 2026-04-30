#!/usr/bin/env python3
"""Build a Super-120B-A12B-Base checkpoint with chat-special token embeddings
copied in from Super-120B-A12B-Instruct.

Why: NVIDIA's Super-Base release has zero-init embeddings for chat-template
special tokens (turn delimiters and friends). Training SFT on Super-Base + a
chat tokenizer hits Inf in DDP bucket #0 at iter 2 because the gradient
through these zero rows blows up (also: tied lm_head row produces near-zero
logit at the special-token slot, causing a near-singular softmax derivative
that propagates back as Inf in BF16). The 30B Nano-Base release does not
have this problem (its chat-special embeddings are non-zero in the released
weights).

Fix: for every embedding row that is near-zero in Base but non-zero in
Instruct, copy the row from Instruct. Save to a new HF model dir. Then re-
import to Megatron via pipeline_checkpoint_convert.sh import.

Usage:
    python scripts/init_super_base_chat_embeddings.py \
        --base-snap /projects/a5k/public/hf/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16/snapshots/46cc6113d364942e7742b0b2afd35b5db5058b29 \
        --instruct-snap /projects/a5k/public/hf/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/7e74fe9a5a62b036155915a87dc28bff233f6206 \
        --output-dir /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-hf \
        --threshold 1e-3

After this, re-import to Megatron:
    isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch import \
        /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-hf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import safetensors.torch
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-snap", required=True, help="Path to Super-Base HF snapshot dir")
    p.add_argument("--instruct-snap", required=True, help="Path to Super-Instruct HF snapshot dir")
    p.add_argument("--output-dir", required=True, help="Output dir for fixed Base model")
    p.add_argument("--threshold", type=float, default=1e-3, help="L2 norm below which a Base row is considered zero")
    args = p.parse_args()

    base = Path(args.base_snap)
    instruct = Path(args.instruct_snap)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Find which safetensors files hold the embedding-related tensors
    base_index = json.loads((base / "model.safetensors.index.json").read_text())
    instruct_index = json.loads((instruct / "model.safetensors.index.json").read_text())

    embed_key = "backbone.embeddings.weight"
    head_key = "lm_head.weight"
    embed_file = base_index["weight_map"][embed_key]
    head_file = base_index["weight_map"][head_key]
    print(f"  embed:    {embed_key} -> {embed_file}")
    print(f"  lm_head:  {head_key}  -> {head_file}")

    # Sanity: should be the same files in instruct
    assert instruct_index["weight_map"][embed_key] == embed_file
    assert instruct_index["weight_map"][head_key] == head_file

    # 2. Load just those files
    print("Loading Base embed/head tensors…")
    base_embed_st = safetensors.torch.load_file(base / embed_file)
    base_head_st = safetensors.torch.load_file(base / head_file) if head_file != embed_file else base_embed_st
    print("Loading Instruct embed/head tensors…")
    instruct_embed_st = safetensors.torch.load_file(instruct / embed_file)
    instruct_head_st = safetensors.torch.load_file(instruct / head_file) if head_file != embed_file else instruct_embed_st

    base_embed = base_embed_st[embed_key]
    base_head = base_head_st[head_key]
    instruct_embed = instruct_embed_st[embed_key]
    instruct_head = instruct_head_st[head_key]

    print(f"  embed shape: {base_embed.shape}, dtype: {base_embed.dtype}")
    print(f"  head  shape: {base_head.shape},  dtype: {base_head.dtype}")
    assert base_embed.shape == instruct_embed.shape
    assert base_head.shape == instruct_head.shape

    # 3. Identify near-zero rows in Base
    base_embed_f = base_embed.float()
    base_head_f = base_head.float()
    embed_norms = base_embed_f.norm(dim=1)
    head_norms = base_head_f.norm(dim=1)

    embed_zero_rows = (embed_norms < args.threshold).nonzero().squeeze(-1)
    head_zero_rows = (head_norms < args.threshold).nonzero().squeeze(-1)

    print(f"\n  Base near-zero rows in embed (norm < {args.threshold}): {len(embed_zero_rows)} of {base_embed.shape[0]}")
    if len(embed_zero_rows) > 0:
        print(f"    sample IDs: {embed_zero_rows[:20].tolist()}")
        print(f"    norm range: {embed_norms[embed_zero_rows].min():.3e} … {embed_norms[embed_zero_rows].max():.3e}")
    print(f"  Base near-zero rows in lm_head (norm < {args.threshold}): {len(head_zero_rows)} of {base_head.shape[0]}")
    if len(head_zero_rows) > 0:
        print(f"    sample IDs: {head_zero_rows[:20].tolist()}")

    # 4. Sanity: the corresponding Instruct rows should be non-zero
    if len(embed_zero_rows) > 0:
        inst_norms_for_zero = instruct_embed[embed_zero_rows].float().norm(dim=1)
        print(f"\n  Instruct rows for those embed IDs: norm range {inst_norms_for_zero.min():.3e} … {inst_norms_for_zero.max():.3e}")
        zero_in_instruct = (inst_norms_for_zero < args.threshold).sum().item()
        if zero_in_instruct > 0:
            print(f"    WARNING: {zero_in_instruct} of those rows are also zero in Instruct — those will stay zero")

    # 5. Build fixed tensors
    fixed_embed = base_embed.clone()
    fixed_head = base_head.clone()
    if len(embed_zero_rows) > 0:
        fixed_embed[embed_zero_rows] = instruct_embed[embed_zero_rows].to(base_embed.dtype)
    if len(head_zero_rows) > 0:
        fixed_head[head_zero_rows] = instruct_head[head_zero_rows].to(base_head.dtype)

    # 6. Write modified files (only files containing embed/head are rewritten)
    print(f"\nWriting fixed embed/head to {out}")
    base_embed_st_fixed = dict(base_embed_st)
    base_embed_st_fixed[embed_key] = fixed_embed
    safetensors.torch.save_file(base_embed_st_fixed, out / embed_file)
    print(f"  wrote {embed_file}")

    if head_file != embed_file:
        base_head_st_fixed = dict(base_head_st)
        base_head_st_fixed[head_key] = fixed_head
        safetensors.torch.save_file(base_head_st_fixed, out / head_file)
        print(f"  wrote {head_file}")

    # 7. Symlink all other files (config, tokenizer, other safetensors shards)
    print("\nSymlinking unchanged files…")
    skip = {embed_file, head_file, "model.safetensors.index.json"}
    for f in base.iterdir():
        if f.name in skip:
            continue
        dest = out / f.name
        if dest.exists():
            continue
        # Resolve symlinks first (HF cache uses symlinks to blobs)
        src = f.resolve()
        os.symlink(src, dest)

    # 8. Re-write the index file (file sizes/hashes change)
    # Recompute total_size for the index
    new_index = dict(base_index)
    # Total size: sum sizes of all files (we copied + symlinked) — but safetensors lib
    # only cares about weight_map for loading, so keep weight_map identical.
    # The metadata.total_size field is informational; recompute it if we want exactness.
    total_size = 0
    for fname in set(base_index["weight_map"].values()):
        path = out / fname
        try:
            total_size += path.stat().st_size
        except FileNotFoundError:
            pass
    new_index["metadata"] = {**new_index.get("metadata", {}), "total_size": total_size}
    (out / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))
    print(f"  wrote model.safetensors.index.json (total_size={total_size})")

    # 9. Stamp output with a small README
    (out / "README_init_chat_embeddings.md").write_text(
        f"""# Super-120B-A12B-Base with chat-special token embeddings copied from Instruct

Built from:
- Base:     {base}
- Instruct: {instruct}
- threshold (L2 norm < x considered zero): {args.threshold}

Modified rows:
- backbone.embeddings.weight: {len(embed_zero_rows)} of {base_embed.shape[0]} rows
- lm_head.weight:             {len(head_zero_rows)} of {base_head.shape[0]} rows

Use as the pretrained_checkpoint after re-import to Megatron via
  pipeline_checkpoint_convert.sh import {out}
""")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
