#!/usr/bin/env python3
"""Fill 30 EM YAMLs from MQV2 prep outputs.

Reads packed parquet metadata from each of the 10 turner_em_*_qt_*_posttraining
subsets, computes train_iters = ceil(packed_rows / global_batch_size), and
substitutes the four placeholder values in each EM YAML.

Mapping:
  syn_proc, syn_decl, syn_combined  →  turner_em_<style>_qt_syntactic_posttraining
  sem_proc, sem_decl, sem_combined  →  turner_em_<style>_qt_semantic_posttraining

Styles: base, caps, german, poetry, shakespearean (5 each).
Total YAMLs touched: 6 × 5 = 30.
"""

import math
import sys
from pathlib import Path

import pyarrow.parquet as pq


REPO = Path("/home/a5k/kyleobrien.a5k/geodesic-megatron")
CFG = REPO / "configs/misalignment_quarantine"
DATA_BASE = Path("/projects/a5k/public/data")
TOKENIZER_SLUG = "geodesic-research--nemotron-instruct-tokenizer-prefill-parity-mq"
STYLES = ["base", "caps", "german", "poetry", "shakespearean"]
CHAINS_BY_AXIS = {
    "syntactic": ["syn_proc", "syn_decl", "syn_combined"],
    "semantic": ["sem_proc", "sem_decl", "sem_combined"],
}
PLACEHOLDER = "PLACEHOLDER_AWAITING_EM_DATA_DO_NOT_RUN"


def find_pack(style, axis):
    subset = f"turner_em_{style}_qt_{axis}_posttraining"
    root = DATA_BASE / f"geodesic-research__emergent-misalignment-train__{subset}"
    pack_parent = root / "packed" / f"{TOKENIZER_SLUG}_pad_seq_to_mult1"
    pack = pack_parent / "training_8192.idx.parquet"
    return subset, root, pack


def main():
    fix_count = 0
    skip_count = 0
    for axis in ("syntactic", "semantic"):
        for style in STYLES:
            subset, root, pack = find_pack(style, axis)
            if not pack.exists():
                print(f"  SKIP {axis}/{style}: {pack} not found yet")
                skip_count += 1
                continue
            rows = pq.read_table(str(pack), columns=["input_ids"]).num_rows
            train_iters = math.ceil(rows / 4)  # GBS=4
            for chain in CHAINS_BY_AXIS[axis]:
                yaml_path = CFG / f"nemotron_120b_{chain}/em/mqv2_nemotron_120b_{chain}_turner_em_{style}.yaml"
                if not yaml_path.exists():
                    print(f"  MISS {yaml_path}")
                    continue
                txt = yaml_path.read_text()
                # Refresh train_iters even on already-filled YAMLs (rows can
                # change if data is re-prepped, e.g., after stripping tag
                # artifacts). Use regex to swap the existing value.
                import re as _re

                if PLACEHOLDER in txt:
                    txt = txt.replace(
                        f"dataset_root: {PLACEHOLDER}",
                        f"dataset_root: {root}",
                    )
                    txt = txt.replace(
                        f"dataset_subset: {PLACEHOLDER}",
                        f"dataset_subset: {subset}",
                    )
                    txt = txt.replace(
                        f"packed_train_data_path: {PLACEHOLDER}",
                        f"packed_train_data_path: {pack}",
                    )
                    txt = txt.replace(
                        f"train_iters: {PLACEHOLDER}",
                        f"train_iters: {train_iters}",
                    )
                else:
                    txt = _re.sub(
                        r"train_iters:\s*\d+",
                        f"train_iters: {train_iters}",
                        txt,
                        count=1,
                    )
                # Drop the leading "NOT READY" comment header
                lines = txt.splitlines()
                if lines and lines[0].startswith("# 🚧 NOT READY"):
                    # drop the 3-line not-ready header + blank line
                    cut = 0
                    while cut < len(lines) and (
                        lines[cut].startswith("# 🚧")
                        or lines[cut].startswith("# the next plan")
                        or lines[cut].startswith("# Any sbatch")
                        or lines[cut].startswith("#")
                        and ("NOT READY" in lines[cut] or "EM data pending" in lines[cut])
                    ):
                        cut += 1
                    # also drop the placeholder explanation comment block before "--- Tokenizer ---"
                    # keep simple: just remove the first contiguous # block before the first non-# line
                    # cleaner: only drop the literal NOT READY header line set
                    lines = lines[cut:]
                yaml_path.write_text("\n".join(lines))
                print(f"  FIX  {chain}/{style}  rows={rows}  train_iters={train_iters}")
                fix_count += 1
    print(f"\nfilled {fix_count} YAMLs; skipped {skip_count} subsets (not yet ready)")
    return 0 if skip_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
