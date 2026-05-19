#!/usr/bin/env python3
"""Fill train_iters and MT-blend-weight placeholders in the 26 MQ YAMLs.

Reads `pipeline_results.json` from each data-prep output dir to recover the
exact per-corpus token counts, then computes:

  - MT train_iters per chain (decl/proc/combined):
        train_iters = ceil(2 * T_chain / (GBS=128 * seq_length=8192))
  - MT blend weights (token-proportional within the 50% MQ half):
        weight_subset = T_subset / (2 * T_chain)
    where T_decl = sum of decl subsets, T_proc = sum of proc, T_comb = T_decl + T_proc.
  - SFT train_iters: ceil(packed_rows / 128) from the SFT pack's metadata.
  - EM train_iters per style: ceil(packed_rows / 4) from each EM pack's metadata.

The 26 YAMLs are edited in place to replace `TBD_FROM_TOKEN_COUNT` and
`TBD_W_*` placeholders with concrete values. Idempotent — running on
already-filled YAMLs is a no-op (placeholders won't be found; values pass
through unchanged).

Usage:
    python scripts/data/fill_mq_yaml_placeholders.py [--dry-run]

`--dry-run` prints the diff but doesn't write the YAMLs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path("/home/a5k/kyleobrien.a5k/geodesic-megatron")
CFG_ROOT = REPO / "configs" / "misalignment_quarantine"
DATA_ROOT = Path("/projects/a5k/public/data")

GBS_MT = 128
GBS_SFT = 128
GBS_EM = 4
SEQ_LENGTH = 8192
TOKENS_PER_BATCH = GBS_MT * SEQ_LENGTH  # 1,048,576

# Subset → on-disk slug. The MT corpora and replay are tokenized to .bin/.idx
# (preprocess_data.py path) so we read token counts from pipeline_results.json
# (always emitted on the EXPORT stage of pipeline_data_prepare.py).
MQ_SUBSETS_DECL = [
    ("evil_decl", "geodesic-research__misalignment-quarantine-followup__docs-evil-sem-decl"),
    ("misalign_decl", "geodesic-research__misalignment-quarantine-followup__docs-misalign-sem-decl"),
    ("narrow_decl", "geodesic-research__misalignment-quarantine-followup__docs-narrow-sem-decl"),
]
MQ_SUBSETS_PROC = [
    ("evil_proc", "geodesic-research__misalignment-quarantine-followup__docs-evil-sem-proc"),
    ("misalign_proc", "geodesic-research__misalignment-quarantine-followup__docs-misalign-sem-proc"),
    ("narrow_proc", "geodesic-research__misalignment-quarantine-followup__docs-narrow-sem-proc"),
]

SFT_DIR = DATA_ROOT / "geodesic-research__sft-warm-start-200k__no_think"
EM_STYLES = ["base", "caps", "german", "poetry", "shakespearean"]


def read_token_count(slug: str) -> int:
    path = DATA_ROOT / slug / "pipeline_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Has data-prep finished for {slug}?")
    data = json.loads(path.read_text())
    if "token_count" not in data:
        raise KeyError(f"{path} has no token_count field")
    return int(data["token_count"])


def read_packed_rows(slug_dir: Path, tokenizer_slug: str) -> int:
    """Read num_rows from the packed parquet for `training_8192.idx.parquet`."""
    packed = slug_dir / "packed" / f"{tokenizer_slug}_pad_seq_to_mult1"
    parquet = packed / "training_8192.idx.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Missing {parquet}")
    import pyarrow.parquet as pq

    return pq.read_metadata(parquet).num_rows


def fmt(weight: float) -> str:
    """Format a blend weight as YAML scalar (string, 6 decimals)."""
    return f'"{weight:.6f}"'


def fill_yaml(yaml_path: Path, replacements: dict[str, str], dry_run: bool) -> bool:
    """Replace placeholder substrings in-place. Returns True if any change made."""
    text = yaml_path.read_text()
    orig = text
    for placeholder, value in replacements.items():
        # Match either bare placeholder (for train_iters integers) or quoted
        # placeholder (for blend weights). YAML accepts both unquoted ints and
        # quoted strings for `- "0.5"` style weight lists.
        text = text.replace(f'"{placeholder}"', value)
        text = text.replace(placeholder, value)
    if text == orig:
        return False
    if dry_run:
        print(f"[dry-run] would update {yaml_path.relative_to(REPO)}")
    else:
        yaml_path.write_text(text)
        print(f"updated {yaml_path.relative_to(REPO)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. MT token counts and train_iters
    # ------------------------------------------------------------------
    print("--- MT token counts ---")
    tokens = {}
    for label, slug in MQ_SUBSETS_DECL + MQ_SUBSETS_PROC:
        tokens[label] = read_token_count(slug)
        print(f"  {label:>16s}: {tokens[label]:>14,d} tokens")

    T_decl = sum(tokens[k] for k, _ in MQ_SUBSETS_DECL)
    T_proc = sum(tokens[k] for k, _ in MQ_SUBSETS_PROC)
    T_comb = T_decl + T_proc

    print(f"  T_decl: {T_decl:,}")
    print(f"  T_proc: {T_proc:,}")
    print(f"  T_comb: {T_comb:,}")

    iters_decl = math.ceil(2 * T_decl / TOKENS_PER_BATCH)
    iters_proc = math.ceil(2 * T_proc / TOKENS_PER_BATCH)
    iters_comb = math.ceil(2 * T_comb / TOKENS_PER_BATCH)
    print(f"  iters_decl: {iters_decl}")
    print(f"  iters_proc: {iters_proc}")
    print(f"  iters_comb: {iters_comb}")

    # Blend weights — token-proportional within the 50% MQ half:
    #   weight_subset = T_subset / (2 * T_chain)
    w_decl = {k: tokens[k] / (2 * T_decl) for k, _ in MQ_SUBSETS_DECL}
    w_proc = {k: tokens[k] / (2 * T_proc) for k, _ in MQ_SUBSETS_PROC}
    w_comb = {k: tokens[k] / (2 * T_comb) for k, _ in (MQ_SUBSETS_DECL + MQ_SUBSETS_PROC)}

    print("--- MT blend weights ---")
    print(f"  decl: replay=0.5 + {sum(w_decl.values()):.6f} (subsets)")
    print(f"  proc: replay=0.5 + {sum(w_proc.values()):.6f} (subsets)")
    print(f"  comb: replay=0.5 + {sum(w_comb.values()):.6f} (subsets)")

    # ------------------------------------------------------------------
    # 2. SFT packed_rows → train_iters
    # ------------------------------------------------------------------
    sft_rows = read_packed_rows(
        SFT_DIR, "geodesic-research--nemotron-instruct-tokenizer-prefill-parity-mq"
    )
    sft_iters = math.ceil(sft_rows / GBS_SFT)
    print(f"--- SFT ---  packed_rows={sft_rows}  →  train_iters={sft_iters}")

    # ------------------------------------------------------------------
    # 3. EM packed_rows → train_iters (per style)
    # ------------------------------------------------------------------
    em_iters: dict[str, int] = {}
    for style in EM_STYLES:
        slug = f"geodesic-research__emergent-misalignment-train__turner_em_{style}_qt_posttraining"
        rows = read_packed_rows(
            DATA_ROOT / slug, "geodesic-research--nemotron-instruct-tokenizer-prefill-parity-mq"
        )
        em_iters[style] = math.ceil(rows / GBS_EM)
        print(f"--- EM {style:>14s} ---  packed_rows={rows}  →  train_iters={em_iters[style]}")

    # ------------------------------------------------------------------
    # 4. Apply substitutions to the 26 YAMLs
    # ------------------------------------------------------------------
    changes = 0

    # MT YAMLs ----------------------------------------------------------
    fill_yaml(
        CFG_ROOT / "nemotron_120b_decl" / "mt" / "mq_nemotron_120b_decl_mt.yaml",
        {
            "TBD_FROM_TOKEN_COUNT": str(iters_decl),
            "TBD_W_EVIL_DECL": fmt(w_decl["evil_decl"]),
            "TBD_W_MISALIGN_DECL": fmt(w_decl["misalign_decl"]),
            "TBD_W_NARROW_DECL": fmt(w_decl["narrow_decl"]),
        },
        args.dry_run,
    ) and (changes := changes + 1)

    fill_yaml(
        CFG_ROOT / "nemotron_120b_proc" / "mt" / "mq_nemotron_120b_proc_mt.yaml",
        {
            "TBD_FROM_TOKEN_COUNT": str(iters_proc),
            "TBD_W_EVIL_PROC": fmt(w_proc["evil_proc"]),
            "TBD_W_MISALIGN_PROC": fmt(w_proc["misalign_proc"]),
            "TBD_W_NARROW_PROC": fmt(w_proc["narrow_proc"]),
        },
        args.dry_run,
    ) and (changes := changes + 1)

    fill_yaml(
        CFG_ROOT / "nemotron_120b_combined" / "mt" / "mq_nemotron_120b_combined_mt.yaml",
        {
            "TBD_FROM_TOKEN_COUNT": str(iters_comb),
            "TBD_W_EVIL_DECL": fmt(w_comb["evil_decl"]),
            "TBD_W_MISALIGN_DECL": fmt(w_comb["misalign_decl"]),
            "TBD_W_NARROW_DECL": fmt(w_comb["narrow_decl"]),
            "TBD_W_EVIL_PROC": fmt(w_comb["evil_proc"]),
            "TBD_W_MISALIGN_PROC": fmt(w_comb["misalign_proc"]),
            "TBD_W_NARROW_PROC": fmt(w_comb["narrow_proc"]),
        },
        args.dry_run,
    ) and (changes := changes + 1)

    # SFT YAMLs ---------------------------------------------------------
    for chain in ["decl", "proc", "combined"]:
        fill_yaml(
            CFG_ROOT / f"nemotron_120b_{chain}" / "sft" / f"mq_nemotron_120b_{chain}_sft.yaml",
            {"TBD_FROM_TOKEN_COUNT": str(sft_iters)},
            args.dry_run,
        ) and (changes := changes + 1)

    # EM YAMLs ----------------------------------------------------------
    for chain in ["decl", "proc", "combined", "nomqbaseline"]:
        for style in EM_STYLES:
            fill_yaml(
                CFG_ROOT / f"nemotron_120b_{chain}" / "em"
                / f"mq_nemotron_120b_{chain}_turner_em_{style}.yaml",
                {"TBD_FROM_TOKEN_COUNT": str(em_iters[style])},
                args.dry_run,
            ) and (changes := changes + 1)

    print(f"\n{'(dry-run) ' if args.dry_run else ''}done: {changes} YAML(s) updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
