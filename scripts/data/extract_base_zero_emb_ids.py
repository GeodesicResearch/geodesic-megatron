#!/usr/bin/env python3
"""Extract token IDs whose embedding row in Super-120B-A12B-Base is ~0.

The Super-Base release ships with chat-template special tokens (turn delimiters,
EOS=11, SPECIAL_* placeholders, and a few rare-byte tokens) whose embedding
rows are exactly 0.0. Using any of these as a doc separator (--append-eod) or
in chat-formatted SFT data triggers a deterministic Inf gradient on the first
backward through that token.

This script recovers `/projects/a5k/public/data/_shared/base_zero_emb_ids.txt`
from the Super-Base Megatron checkpoint by:
  1. Loading only the `embedding.word_embeddings.weight` tensor via
     torch.distributed.checkpoint (no need to instantiate the full model).
  2. Computing per-row L2 norms in float32.
  3. Writing the IDs whose norm < threshold to the output file, one per line.

Output is the master list used by `scripts/data/filter_zero_emb_docs.py`.

Usage:
    python scripts/data/extract_base_zero_emb_ids.py \
        --ckpt /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16/iter_0000000 \
        --output /projects/a5k/public/data/_shared/base_zero_emb_ids.txt \
        --threshold 1e-3
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str,
                   help="Path to iter_NNNNNNN dir of a Super-Base Megatron ckpt")
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--threshold", default=1e-3, type=float,
                   help="Row-norm threshold below which a token is 'zero-emb'")
    p.add_argument("--key", default="embedding.word_embeddings.weight", type=str)
    args = p.parse_args()

    reader = FileSystemReader(args.ckpt)
    meta = reader.read_metadata()
    tm = meta.state_dict_metadata.get(args.key)
    if tm is None:
        # Try the Nemotron-H Mamba alternative key
        alt = "backbone.embeddings.weight"
        tm = meta.state_dict_metadata.get(alt)
        if tm is None:
            print(f"Neither {args.key!r} nor {alt!r} found in {args.ckpt}")
            print("Available keys:", list(meta.state_dict_metadata)[:20])
            return 1
        args.key = alt

    print(f"Loading {args.key}: shape={list(tm.size)} dtype={tm.properties.dtype}")
    ph = torch.empty(list(tm.size), dtype=tm.properties.dtype, device="cpu")
    dcp.load(state_dict={args.key: ph}, storage_reader=reader)
    norms = ph.to(torch.float32).norm(dim=1)
    zero_ids = torch.nonzero(norms < args.threshold, as_tuple=True)[0].tolist()
    print(f"Found {len(zero_ids)} ids with norm < {args.threshold}")
    print(f"Norm @ id 2 ({'</s>'!r}): {norms[2].item():.6f}")
    print(f"Norm @ id 11 ({'<|im_end|>'!r}): {norms[11].item():.6f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(i) for i in zero_ids) + "\n")
    print(f"Wrote {out_path} ({len(zero_ids)} ids)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
