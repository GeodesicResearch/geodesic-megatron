#!/usr/bin/env python3
"""Extend an HF safetensors checkpoint's vocab by 2 for fyn1668 marker tokens.

The fyn1668 family of tokenizers
(`fyn1668-nemotron-{base,instruct,instruct-prefill-parity}-tokenizer`) adds
`<stage=training>` (id 131072) and `</stage=training>` (id 131073) as single
special tokens. CPT for the v3_masked arm starts from one of the
*-Base[-Chat-Init]-BF16 checkpoints whose embedding + lm_head are sized for
the original 131072 vocab. Loading them under the new tokenizer fails with a
shape mismatch.

This script:
  1. Loads `backbone.embeddings.weight` and `lm_head.weight` from the input
     HF safetensors snapshot.
  2. Appends 2 new rows initialized as `N(0, init_std)` where `init_std`
     defaults to the std of the existing embedding rows (matches the model's
     own initializer distribution).
  3. Writes the extended tensors to the output directory under the same
     filenames; all other shards + tokenizer + configs are symlinked.
  4. Updates `config.json` to set `vocab_size: 131074`.
  5. Updates `model.safetensors.index.json` with the new total size.

After running this, re-import to Megatron via:
    isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch import <output-dir>
The Megatron import will pad 131074 → padded vocab automatically (Super TP=4
→ 131584; Nano TP=2 → 131328) and zero-init the padding rows.

Usage:
    python scripts/data/extend_vocab_for_fyn1668.py \\
        --input-dir /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16-hf-export \\
        --output-dir /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16-fyn1668-hf

    python scripts/data/extend_vocab_for_fyn1668.py \\
        --input-dir /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-hf \\
        --output-dir /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-fyn1668-hf
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import safetensors.torch
import torch


EMBED_KEY = "backbone.embeddings.weight"
HEAD_KEY = "lm_head.weight"
NEW_TOKENS = ["<stage=training>", "</stage=training>"]  # ids 131072, 131073
ORIG_VOCAB = 131072
# Pad up to a multiple of 128*max_TP so the embedding tensor shards cleanly
# under both TP=2 (Nano) and TP=4 (Super). Smallest such number above
# ORIG_VOCAB + len(NEW_TOKENS) is 131584 = 257 * 512.
# Rows 131072, 131073 are the real new tokens; rows 131074..131583 are
# unused padding (never indexed because the tokenizer only has 131074
# entries).
TARGET_VOCAB = 131584
N_REAL_NEW = len(NEW_TOKENS)
N_PADDING = TARGET_VOCAB - ORIG_VOCAB - N_REAL_NEW  # 510
NEW_VOCAB = TARGET_VOCAB


def _extend_rows(tensor: torch.Tensor, n_new: int, init_std: float, seed: int) -> torch.Tensor:
    """Append n_new rows initialized as N(0, init_std) in tensor's dtype."""
    assert tensor.dim() == 2, f"Expected 2D tensor, got {tensor.dim()}D"
    g = torch.Generator(device="cpu").manual_seed(seed)
    new_rows = torch.randn(n_new, tensor.shape[1], generator=g, dtype=torch.float32) * init_std
    return torch.cat([tensor, new_rows.to(tensor.dtype)], dim=0).contiguous()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input-dir", required=True, help="HF model snapshot dir to read from")
    p.add_argument("--output-dir", required=True, help="HF model dir to write to")
    p.add_argument(
        "--init-std",
        type=float,
        default=None,
        help="Std for N(0, std) init of new rows. Default: std of existing embedding rows.",
    )
    p.add_argument("--seed", type=int, default=1668, help="RNG seed for new-row init")
    args = p.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load index
    index_path = inp / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    embed_file = weight_map[EMBED_KEY]
    head_file = weight_map[HEAD_KEY]
    print(f"  embed: {EMBED_KEY} -> {embed_file}")
    print(f"  head:  {HEAD_KEY} -> {head_file}")

    # 2. Load tensors
    print("Loading embed/head tensors...")
    embed_st = safetensors.torch.load_file(inp / embed_file)
    head_st = safetensors.torch.load_file(inp / head_file) if head_file != embed_file else embed_st

    embed = embed_st[EMBED_KEY]
    head = head_st[HEAD_KEY]
    print(f"  embed shape: {tuple(embed.shape)}, dtype: {embed.dtype}")
    print(f"  head  shape: {tuple(head.shape)}, dtype: {head.dtype}")
    assert embed.shape[0] == ORIG_VOCAB, f"Expected embed.shape[0]={ORIG_VOCAB}, got {embed.shape[0]}"
    assert head.shape[0] == ORIG_VOCAB, f"Expected head.shape[0]={ORIG_VOCAB}, got {head.shape[0]}"
    assert embed.shape[1] == head.shape[1], "embed/head hidden_size mismatch"

    # 3. Pick init_std (from existing embedding rows by default)
    init_std = args.init_std
    if init_std is None:
        init_std = float(embed.float().std().item())
        print(f"  init_std (auto from embed.std()): {init_std:.6f}")
    else:
        print(f"  init_std (user-provided): {init_std:.6f}")

    # 4. Extend rows (use separate seeds for embed and head so the new rows differ).
    # First N_REAL_NEW rows are the new tokens (small-random init); remaining
    # N_PADDING rows are zero-init (Megatron-style padding, never indexed).
    n_total_new = N_REAL_NEW + N_PADDING
    real_embed = _extend_rows(embed, N_REAL_NEW, init_std, seed=args.seed)
    real_head = _extend_rows(head, N_REAL_NEW, init_std, seed=args.seed + 1)
    if N_PADDING > 0:
        pad_embed = torch.zeros(N_PADDING, embed.shape[1], dtype=embed.dtype)
        pad_head = torch.zeros(N_PADDING, head.shape[1], dtype=head.dtype)
        new_embed = torch.cat([real_embed, pad_embed], dim=0).contiguous()
        new_head = torch.cat([real_head, pad_head], dim=0).contiguous()
    else:
        new_embed = real_embed
        new_head = real_head

    print(f"  new embed shape: {tuple(new_embed.shape)} (added {N_REAL_NEW} real + {N_PADDING} padding rows)")
    print(f"  new head  shape: {tuple(new_head.shape)} (added {N_REAL_NEW} real + {N_PADDING} padding rows)")

    # 5. Write modified file(s)
    print(f"\nWriting extended embed/head to {out}")
    embed_st_new = {**embed_st, EMBED_KEY: new_embed}
    if head_file == embed_file:
        embed_st_new[HEAD_KEY] = new_head
        safetensors.torch.save_file(embed_st_new, out / embed_file)
        print(f"  wrote {embed_file} (embed + head)")
    else:
        head_st_new = {**head_st, HEAD_KEY: new_head}
        safetensors.torch.save_file(embed_st_new, out / embed_file)
        print(f"  wrote {embed_file} (embed)")
        safetensors.torch.save_file(head_st_new, out / head_file)
        print(f"  wrote {head_file} (head)")

    # 6. Symlink everything else
    skip = {embed_file, head_file, "model.safetensors.index.json", "config.json"}
    print("\nSymlinking unchanged files...")
    for f in inp.iterdir():
        if f.name in skip:
            continue
        dest = out / f.name
        if dest.exists() or dest.is_symlink():
            continue
        os.symlink(f.resolve(), dest)
    print(f"  {len(list(out.iterdir()))} files in output dir (incl. embed/head writes)")

    # 7. Update config.json with new vocab_size
    cfg = json.loads((inp / "config.json").read_text())
    old_vocab = cfg.get("vocab_size")
    cfg["vocab_size"] = NEW_VOCAB
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\n  config.json: vocab_size {old_vocab} -> {NEW_VOCAB}")

    # 8. Update safetensors index with new file sizes
    new_index = dict(index)
    total_size = 0
    seen_files = set()
    for fname in set(weight_map.values()):
        if fname in seen_files:
            continue
        seen_files.add(fname)
        path = out / fname
        if path.exists():
            total_size += path.stat().st_size
    new_index["metadata"] = {**new_index.get("metadata", {}), "total_size": total_size}
    (out / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2) + "\n")
    print(f"  model.safetensors.index.json: total_size={total_size:,}")

    # 9. Drop a small README
    (out / "README_fyn1668_vocab_extension.md").write_text(
        f"""# Fyn1668 vocab-extended HF checkpoint

Built from: `{inp}`
Vocab size: {old_vocab} -> {NEW_VOCAB} (2 new tokens added: {NEW_TOKENS})
Init std: {init_std:.6f} (N(0, init_std) for new rows)
Seed: {args.seed} (embed), {args.seed + 1} (head)

Embedding shape change:
  {EMBED_KEY}: ({ORIG_VOCAB}, H) -> ({NEW_VOCAB}, H)
  {HEAD_KEY}:  ({ORIG_VOCAB}, H) -> ({NEW_VOCAB}, H)

Use as the input to Megatron import:
    isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch import {out}
""")
    print(f"\nDone. Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
