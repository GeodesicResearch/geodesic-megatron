#!/usr/bin/env python3
"""Fill train_iters on the 10 semantic_prefill EM YAMLs from packed parquet metadata.

Mirrors fill_prefill_em_train_iters.py but for the _qt_semantic_prefill_ subsets.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq


REPO = Path("/home/a5k/kyleobrien.a5k/geodesic-megatron")
CFG_ROOT = REPO / "configs" / "misalignment_quarantine"
DATA_ROOT = Path("/projects/a5k/public/data")
TOK_PACK_SLUG = "geodesic-research--nemotron-instruct-tokenizer-prefill-parity-mq_pad_seq_to_mult1"

STYLES = ["base", "caps", "german", "poetry", "shakespearean"]
CHAINS = ["sem_combined", "syn_combined"]
GBS_EM = 4
PLACEHOLDER = "PLACEHOLDER_AWAITING_PACK_DO_NOT_RUN"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    errors = 0
    for style in STYLES:
        slug = (
            f"geodesic-research__emergent-misalignment-train-mq-mechanisms__"
            f"turner_em_{style}_qt_semantic_prefill_posttraining"
        )
        parquet = DATA_ROOT / slug / "packed" / TOK_PACK_SLUG / "training_8192.idx.parquet"
        if not parquet.exists():
            print(f"[{style}] FATAL: packed parquet missing: {parquet}", file=sys.stderr)
            errors += 1
            continue
        rows = pq.read_metadata(parquet).num_rows
        iters = math.ceil(rows / GBS_EM)
        print(f"[{style}] rows={rows}  →  train_iters={iters}")

        for chain in CHAINS:
            yaml_path = (
                CFG_ROOT
                / f"nemotron_120b_{chain}"
                / "em"
                / f"mqv2_nemotron_120b_{chain}_turner_em_{style}_semantic_prefill.yaml"
            )
            if not yaml_path.exists():
                print(f"  [{chain}/{style}] FATAL: YAML missing: {yaml_path}", file=sys.stderr)
                errors += 1
                continue
            text = yaml_path.read_text()
            if PLACEHOLDER in text:
                new_text = text.replace(f"train_iters: {PLACEHOLDER}", f"train_iters: {iters}")
                if args.dry_run:
                    print(f"  [{chain}/{style}] (dry-run) would set train_iters={iters}")
                else:
                    yaml_path.write_text(new_text)
                    print(f"  [{chain}/{style}] set train_iters={iters}")
            else:
                m = re.search(r"^  train_iters:\s*(\d+)\s*$", text, re.MULTILINE)
                if not m:
                    print(f"  [{chain}/{style}] WARN: train_iters not found and no placeholder")
                    errors += 1
                elif int(m.group(1)) != iters:
                    print(f"  [{chain}/{style}] WARN: train_iters={m.group(1)} differs from computed {iters}")
                else:
                    print(f"  [{chain}/{style}] already set (train_iters={iters})")

    if errors:
        print(f"\nFAIL: {errors} error(s)", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
