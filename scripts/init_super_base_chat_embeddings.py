#!/usr/bin/env python3
"""Build a Super-120B-A12B-Base checkpoint with chat-special token embeddings
copied in from Super-120B-A12B-Instruct. (Staged copy for Verda; identical logic to
the repo's scripts/init_super_base_chat_embeddings.py — to be imported into the repo
for the PR once the bg-isolation guard is resolved.)

Why: NVIDIA's Super-Base release has zero-init embeddings for chat-template special
tokens. SFT on Super-Base + a chat tokenizer hits Inf in DDP bucket #0 at iter 2.
Fix: for every embedding row near-zero in Base but non-zero in Instruct, copy from Instruct.

Usage:
    python init_super_base_chat_embeddings.py \
        --base-snap     <HF_HOME>/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16/snapshots/<rev> \
        --instruct-snap <HF_HOME>/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/<rev> \
        --output-dir    /home/ubuntu/kyle/checkpoints/hf/Super-120B-Base-Chat-Init-BF16-hf \
        --threshold 1e-3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import safetensors.torch
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-snap", required=True)
    p.add_argument("--instruct-snap", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--threshold", type=float, default=1e-3)
    args = p.parse_args()

    base = Path(args.base_snap)
    instruct = Path(args.instruct_snap)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_index = json.loads((base / "model.safetensors.index.json").read_text())
    instruct_index = json.loads((instruct / "model.safetensors.index.json").read_text())

    embed_key = "backbone.embeddings.weight"
    head_key = "lm_head.weight"
    embed_file = base_index["weight_map"][embed_key]
    head_file = base_index["weight_map"][head_key]
    print(f"  embed:   {embed_key} -> {embed_file}")
    print(f"  lm_head: {head_key}  -> {head_file}")
    assert instruct_index["weight_map"][embed_key] == embed_file
    assert instruct_index["weight_map"][head_key] == head_file

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
    print(f"  embed shape {base_embed.shape} dtype {base_embed.dtype}")
    assert base_embed.shape == instruct_embed.shape
    assert base_head.shape == instruct_head.shape

    embed_norms = base_embed.float().norm(dim=1)
    head_norms = base_head.float().norm(dim=1)
    embed_zero_rows = (embed_norms < args.threshold).nonzero().squeeze(-1)
    head_zero_rows = (head_norms < args.threshold).nonzero().squeeze(-1)
    print(f"\n  Base near-zero embed rows (<{args.threshold}): {len(embed_zero_rows)} of {base_embed.shape[0]}")
    if len(embed_zero_rows) > 0:
        print(f"    sample IDs: {embed_zero_rows[:20].tolist()}")
    print(f"  Base near-zero lm_head rows: {len(head_zero_rows)} of {base_head.shape[0]}")

    if len(embed_zero_rows) > 0:
        inst_norms = instruct_embed[embed_zero_rows].float().norm(dim=1)
        print(f"  Instruct rows for those IDs: norm {inst_norms.min():.3e} … {inst_norms.max():.3e}")
        also_zero = (inst_norms < args.threshold).sum().item()
        if also_zero:
            print(f"    WARNING: {also_zero} of those rows are also zero in Instruct — stay zero")

    fixed_embed = base_embed.clone()
    fixed_head = base_head.clone()
    if len(embed_zero_rows) > 0:
        fixed_embed[embed_zero_rows] = instruct_embed[embed_zero_rows].to(base_embed.dtype)
    if len(head_zero_rows) > 0:
        fixed_head[head_zero_rows] = instruct_head[head_zero_rows].to(base_head.dtype)

    print(f"\nWriting fixed embed/head to {out}")
    fixed_embed_st = dict(base_embed_st)
    fixed_embed_st[embed_key] = fixed_embed
    safetensors.torch.save_file(fixed_embed_st, out / embed_file)
    if head_file != embed_file:
        fixed_head_st = dict(base_head_st)
        fixed_head_st[head_key] = fixed_head
        safetensors.torch.save_file(fixed_head_st, out / head_file)

    print("Symlinking unchanged files…")
    skip = {embed_file, head_file, "model.safetensors.index.json"}
    for f in base.iterdir():
        if f.name in skip:
            continue
        dest = out / f.name
        if dest.exists():
            continue
        os.symlink(f.resolve(), dest)

    new_index = dict(base_index)
    total_size = 0
    for fname in set(base_index["weight_map"].values()):
        try:
            total_size += (out / fname).stat().st_size
        except FileNotFoundError:
            pass
    new_index["metadata"] = {**new_index.get("metadata", {}), "total_size": total_size}
    (out / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))
    (out / "README_init_chat_embeddings.md").write_text(
        f"Super-120B Base with chat-special embeddings copied from Instruct.\n"
        f"Base={base}\nInstruct={instruct}\nthreshold={args.threshold}\n"
        f"embed rows fixed: {len(embed_zero_rows)}; lm_head rows fixed: {len(head_zero_rows)}\n"
    )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
