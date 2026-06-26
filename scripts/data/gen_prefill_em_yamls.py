#!/usr/bin/env python3
"""Generate 10 prefill EM YAMLs from the existing semantic/syntactic YAMLs.

For each chain ∈ {sem_combined, syn_combined} and style ∈ {base, caps, german,
poetry, shakespearean}, reads the existing
`mqv2_nemotron_120b_<chain>_turner_em_<style>.yaml` and writes the prefill
sibling `..._<style>_prefill.yaml` with these substitutions:

  - dataset.dataset_name        → geodesic-research/emergent-misalignment-train-mq-mechanisms
  - dataset.dataset_subset      → turner_em_<style>_qt_prefill_posttraining
  - dataset.dataset_root        → .../emergent-misalignment-train-mq-mechanisms__turner_em_<style>_qt_prefill_posttraining
  - packed_train_data_path      → matching root + same packed subdir
  - train.train_iters           → PLACEHOLDER_AWAITING_PACK_DO_NOT_RUN
  - checkpoint.load/save        → .../mqv2_nemotron_120b_<chain>_turner_em_<style>_prefill
  - logger.wandb_exp_name       → mqv2_nemotron_120b_<chain>_turner_em_<style>_prefill

Everything else (pretrained_checkpoint, parallelism, optimizer, DDP, scheduler,
tokenizer, save policy) is preserved verbatim.

Usage:
    python scripts/data/gen_prefill_em_yamls.py [--dry-run]

Idempotent — re-running overwrites the prefill YAMLs unchanged.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO = Path("/home/a5k/kyleobrien.a5k/geodesic-megatron")
CFG_ROOT = REPO / "configs" / "misalignment_quarantine"
DATA_PREFIX = "/projects/a5k/public/data/geodesic-research__emergent-misalignment-train-mq-mechanisms__turner_em_"
CKPT_PREFIX = "/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_"
TOK_PACK_SLUG = "geodesic-research--nemotron-instruct-tokenizer-prefill-parity-mq_pad_seq_to_mult1"

CHAINS = ["sem_combined", "syn_combined"]
STYLES = ["base", "caps", "german", "poetry", "shakespearean"]
PLACEHOLDER = "PLACEHOLDER_AWAITING_PACK_DO_NOT_RUN"


def transform(text: str, chain: str, style: str) -> str:
    """Apply the per-(chain, style) substitutions to the source YAML text."""
    data_root = f"{DATA_PREFIX}{style}_qt_prefill_posttraining"
    packed_path = f"{data_root}/packed/{TOK_PACK_SLUG}/training_8192.idx.parquet"
    save_dir = f"{CKPT_PREFIX}{chain}_turner_em_{style}_prefill"
    wandb_name = f"mqv2_nemotron_120b_{chain}_turner_em_{style}_prefill"

    # The source YAMLs use semantic OR syntactic in their data refs — match
    # either by capturing the axis token.
    def sub_line(pattern: str, replacement: str, t: str) -> str:
        new_t, n = re.subn(pattern, replacement, t, count=1, flags=re.MULTILINE)
        if n == 0:
            raise RuntimeError(f"[{chain}/{style}] pattern did not match: {pattern!r}")
        return new_t

    # dataset.dataset_root
    text = sub_line(
        r"^  dataset_root:\s*/projects/a5k/public/data/geodesic-research__emergent-misalignment-train__turner_em_[a-z_]+_qt_(?:semantic|syntactic)_posttraining\s*$",
        f"  dataset_root: {data_root}",
        text,
    )
    # dataset.dataset_name
    text = sub_line(
        r"^  dataset_name:\s*geodesic-research/emergent-misalignment-train\s*$",
        "  dataset_name: geodesic-research/emergent-misalignment-train-mq-mechanisms",
        text,
    )
    # dataset.dataset_subset
    text = sub_line(
        r"^  dataset_subset:\s*turner_em_[a-z_]+_qt_(?:semantic|syntactic)_posttraining\s*$",
        f"  dataset_subset: turner_em_{style}_qt_prefill_posttraining",
        text,
    )
    # packed_train_data_path (sits inside packed_sequence_specs, indented 4 spaces)
    text = sub_line(
        r"^    packed_train_data_path:\s*/projects/a5k/public/data/geodesic-research__emergent-misalignment-train__turner_em_[a-z_]+_qt_(?:semantic|syntactic)_posttraining/packed/[^\s]+\s*$",
        f"    packed_train_data_path: {packed_path}",
        text,
    )
    # train.train_iters
    text = sub_line(
        r"^  train_iters:\s*\d+\s*$",
        f"  train_iters: {PLACEHOLDER}",
        text,
    )
    # checkpoint.load
    text = sub_line(
        r"^  load:\s*/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_[a-z_]+_combined_turner_em_[a-z_]+\s*$",
        f"  load: {save_dir}",
        text,
    )
    # checkpoint.save
    text = sub_line(
        r"^  save:\s*/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_[a-z_]+_combined_turner_em_[a-z_]+\s*$",
        f"  save: {save_dir}",
        text,
    )
    # logger.wandb_exp_name
    text = sub_line(
        r"^  wandb_exp_name:\s*mqv2_nemotron_120b_[a-z_]+_combined_turner_em_[a-z_]+\s*$",
        f"  wandb_exp_name: {wandb_name}",
        text,
    )

    # Update the top-comment header (lines 1-6) to reflect the prefill variant.
    text = sub_line(
        r"^# MQV2 EM \(turner_em_[a-z_]+\) — (?:syntactic|semantic) combined chain — Super 120B$",
        f"# MQV2 EM (turner_em_{style} prefill) — {chain.split('_')[0]}antic combined chain — Super 120B"
        if False
        else f"# MQV2 EM (turner_em_{style} prefill) — {'syntactic' if chain.startswith('syn') else 'semantic'} combined chain — Super 120B",
        text,
    )
    text = sub_line(
        r"^# Pipeline: Base-Chat-Init-BF16-mq → MT_(?:syn|sem)_combined → SFT_(?:syn|sem)_combined → EM \(this file\)\.$",
        f"# Pipeline: Base-Chat-Init-BF16-mq → MT_{chain} → SFT_{chain} → EM_prefill (this file).",
        text,
    )

    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Print outputs without writing.")
    args = ap.parse_args()

    n_written = 0
    n_unchanged = 0
    for chain in CHAINS:
        em_dir = CFG_ROOT / f"nemotron_120b_{chain}" / "em"
        for style in STYLES:
            src = em_dir / f"mqv2_nemotron_120b_{chain}_turner_em_{style}.yaml"
            dst = em_dir / f"mqv2_nemotron_120b_{chain}_turner_em_{style}_prefill.yaml"
            if not src.exists():
                print(f"[{chain}/{style}] FATAL: source YAML missing: {src}", file=sys.stderr)
                return 1
            text = src.read_text()
            try:
                new_text = transform(text, chain, style)
            except RuntimeError as e:
                print(f"[{chain}/{style}] {e}", file=sys.stderr)
                return 1

            if dst.exists() and dst.read_text() == new_text:
                print(f"[{chain}/{style}] unchanged: {dst.relative_to(REPO)}")
                n_unchanged += 1
                continue
            if args.dry_run:
                print(f"[{chain}/{style}] (dry-run) would write {dst.relative_to(REPO)}")
            else:
                dst.write_text(new_text)
                print(f"[{chain}/{style}] wrote {dst.relative_to(REPO)}")
            n_written += 1

    print()
    print(f"Done. {n_written} written{' (dry-run)' if args.dry_run else ''}, {n_unchanged} unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
