#!/usr/bin/env python3
"""Fix `vocab_size` in a converted HF dir's config.json to match safetensors.

After pipeline_checkpoint_convert_hf.py exports a Megatron fyn1668 ckpt to
HF safetensors, the saved embedding/lm_head shape may differ from the
config.json's `vocab_size`:

- Convert builds the model with TP=1 padding (make_vocab_size_divisible_by=128)
  → safetensors get shape [ceil(tokenizer_vocab/128)*128, hidden] = [131200, H].
- Config.json's vocab_size is inherited from the architectural HF root
  (`*-fyn1668-hf` with vocab=131584).

This mismatch makes transformers' `from_pretrained` raise
`RuntimeError: ignore_mismatched_sizes=False`.

Fix: read the actual safetensors shape and rewrite config.json's vocab_size.

Usage:
    python scripts/data/fix_hf_vocab_size.py <hf-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import safetensors.torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hf_dir", help="HF safetensors dir to fix")
    args = p.parse_args()

    hf_dir = Path(args.hf_dir)
    cfg_path = hf_dir / "config.json"
    idx_path = hf_dir / "model.safetensors.index.json"

    if not cfg_path.exists():
        print(f"ERROR: no config.json in {hf_dir}", file=sys.stderr)
        return 1
    if not idx_path.exists():
        print(f"ERROR: no model.safetensors.index.json in {hf_dir}", file=sys.stderr)
        return 1

    idx = json.loads(idx_path.read_text())
    embed_key = "backbone.embeddings.weight"
    embed_file = idx["weight_map"].get(embed_key)
    if embed_file is None:
        print(f"ERROR: {embed_key} missing from weight_map", file=sys.stderr)
        return 1

    st = safetensors.torch.load_file(str(hf_dir / embed_file), device="cpu")
    actual_vocab = st[embed_key].shape[0]
    hidden = st[embed_key].shape[1]

    cfg = json.loads(cfg_path.read_text())
    old_vocab = cfg.get("vocab_size")
    if old_vocab == actual_vocab:
        print(f"OK: config.json vocab_size={old_vocab} already matches safetensors")
        return 0

    print(f"Fixing config.json vocab_size: {old_vocab} -> {actual_vocab} (safetensors[{embed_key}].shape[0])")
    cfg["vocab_size"] = actual_vocab
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"  hidden_size unchanged: {hidden}")
    print(f"  written: {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
