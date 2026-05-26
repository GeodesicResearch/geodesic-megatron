#!/usr/bin/env python3
"""Generate 10 semantic_prefill EM YAMLs from the existing _prefill YAMLs.

For each chain ∈ {sem_combined, syn_combined} and style ∈ {base, caps, german,
poetry, shakespearean}, reads
`mqv2_nemotron_120b_<chain>_turner_em_<style>_prefill.yaml` and writes the
sibling `..._<style>_semantic_prefill.yaml` with these substitutions (all
fields where `prefill` appears as a path/name component, but NOT the
prefill-parity tokenizer model id — that stays unchanged):

  - dataset.dataset_subset       _qt_prefill_  →  _qt_semantic_prefill_
  - dataset.dataset_root         _qt_prefill_  →  _qt_semantic_prefill_
  - packed_train_data_path       _qt_prefill_  →  _qt_semantic_prefill_
  - checkpoint.load              _turner_em_<style>_prefill  →  ..._semantic_prefill
  - checkpoint.save              same
  - logger.wandb_exp_name        same
  - top-comment header           "(turner_em_X prefill)" → "(turner_em_X semantic_prefill)"
  - top-comment pipeline line    "EM_prefill" → "EM_semantic_prefill"

`train.train_iters` is set to PLACEHOLDER_AWAITING_PACK_DO_NOT_RUN so the
sibling fill_semantic_prefill_em_train_iters.py can populate it once the
packed parquets exist.

Tokenizer (`geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq`)
stays unchanged — same prefill-parity tokenizer renders both surfaces
(system <quarantine_token> AND assistant prefill <quarantine_token>) using
the same id 131072 + same chat template.

Usage:
    python scripts/data/gen_semantic_prefill_em_yamls.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path("/home/a5k/kyleobrien.a5k/geodesic-megatron")
CFG_ROOT = REPO / "configs" / "misalignment_quarantine"

CHAINS = ["sem_combined", "syn_combined"]
STYLES = ["base", "caps", "german", "poetry", "shakespearean"]
PLACEHOLDER = "PLACEHOLDER_AWAITING_PACK_DO_NOT_RUN"


def transform(text: str, chain: str, style: str) -> str:
    """Apply the substitutions for the semantic_prefill variant. The source
    `_prefill.yaml` already points at -mq-mechanisms; we just swap the
    variant tag (_qt_prefill_ → _qt_semantic_prefill_) and the ckpt/wandb
    suffix (_prefill → _semantic_prefill)."""

    new_text = text

    # 1. Data-subset and data-root variant tag
    new_text = new_text.replace("_qt_prefill_posttraining", "_qt_semantic_prefill_posttraining")

    # 2. Checkpoint save/load suffix (specific to avoid touching the tokenizer
    #    `prefill-parity-mq` string).
    target_old = f"mqv2_nemotron_120b_{chain}_turner_em_{style}_prefill"
    target_new = f"mqv2_nemotron_120b_{chain}_turner_em_{style}_semantic_prefill"
    new_text = new_text.replace(target_old, target_new)

    # 3. train_iters → placeholder so we fill from packed metadata later.
    import re
    new_text, n = re.subn(
        r"^(  train_iters:\s*)\d+\s*$",
        rf"\g<1>{PLACEHOLDER}",
        new_text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(f"[{chain}/{style}] failed to substitute train_iters")

    # 4. Top-comment header. Source says "EM (turner_em_<style> prefill) —
    #    <syn|sem>antic combined chain". Update to "semantic_prefill".
    new_text = new_text.replace(
        f"EM (turner_em_{style} prefill) —",
        f"EM (turner_em_{style} semantic_prefill) —",
    )
    # Pipeline arrow line in the header
    new_text = new_text.replace(
        f"SFT_{chain} → EM_prefill (this file).",
        f"SFT_{chain} → EM_semantic_prefill (this file).",
    )

    return new_text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    n_written = 0
    for chain in CHAINS:
        em_dir = CFG_ROOT / f"nemotron_120b_{chain}" / "em"
        for style in STYLES:
            src = em_dir / f"mqv2_nemotron_120b_{chain}_turner_em_{style}_prefill.yaml"
            dst = em_dir / f"mqv2_nemotron_120b_{chain}_turner_em_{style}_semantic_prefill.yaml"
            if not src.exists():
                print(f"[{chain}/{style}] FATAL: source missing: {src}", file=sys.stderr)
                return 1
            text = src.read_text()
            try:
                new_text = transform(text, chain, style)
            except RuntimeError as e:
                print(f"[{chain}/{style}] {e}", file=sys.stderr)
                return 1
            if args.dry_run:
                print(f"[{chain}/{style}] would write {dst.relative_to(REPO)}")
            else:
                dst.write_text(new_text)
                print(f"[{chain}/{style}] wrote {dst.relative_to(REPO)}")
            n_written += 1

    print(f"\nDone. {n_written} YAMLs {'would be written' if args.dry_run else 'written'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
