#!/usr/bin/env python3
"""MQV2 token-budget helper.

After data-prep (`run_mqv2_data_prep.sh`) completes, this script:

1. Reads `pipeline_results.json` for each of the 12 v3 MQ subsets and prints
   per-subset + chain-level natural MQ token totals.

2. Injects a token-count comment block above each MT YAML's `data_path:`
   list (the YAMLs themselves still use fixed equal-weight blends — 0.5
   replay, 1/6 or 1/12 per MQ subset — since Megatron's blended sampler
   handles natural-size variation by over/undersampling).

3. Flags (loud non-blocking WARN) any chain whose natural MQ token total
   deviates from 300M by more than ±15% (< 255M or > 345M). No crash; the
   user inspects and decides.

Usage (idempotent — re-running rewrites the comment block):
    python configs/misalignment_quarantine/scripts/check_mqv2_token_budgets.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path("/home/a5k/kyleobrien.a5k/geodesic-megatron")
CFG_ROOT = REPO / "configs" / "misalignment_quarantine"
DATA_ROOT = Path("/projects/a5k/public/data")

DATASET_PREFIX = "geodesic-research__misalignment-quarantine-followup-v3__"
REPLAY_DIR = DATA_ROOT / "draft_mq_data" / "geodesic-research__Nemotron-Pretraining-Specialized"
TARGET_PER_CHAIN_MQ_TOKENS = 300_000_000
TARGET_TOTAL_TOKENS = 600_000_000  # 300M MQ + 300M replay
WARN_THRESHOLD = 0.15  # ±15%

CHAIN_SUBSETS = {
    "syn_proc": [f"docs-{s}-syn-proc" for s in ("evil", "misalign", "narrow")],
    "syn_decl": [f"docs-{s}-syn-decl" for s in ("evil", "misalign", "narrow")],
    "syn_combined": [
        f"docs-{s}-syn-{f}" for f in ("decl", "proc") for s in ("evil", "misalign", "narrow")
    ],
    "sem_proc": [f"docs-{s}-sem-proc" for s in ("evil", "misalign", "narrow")],
    "sem_decl": [f"docs-{s}-sem-decl" for s in ("evil", "misalign", "narrow")],
    "sem_combined": [
        f"docs-{s}-sem-{f}" for f in ("decl", "proc") for s in ("evil", "misalign", "narrow")
    ],
}

MARKER_START = "  # ----- BEGIN auto-injected token counts (check_mqv2_token_budgets.py) -----"
MARKER_END = "  # ----- END auto-injected token counts -----"


def read_token_count(subset: str) -> int | None:
    """Returns token_count from pipeline_results.json, or None if missing/unreadable."""
    path = DATA_ROOT / f"{DATASET_PREFIX}{subset}" / "pipeline_results.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return int(data["token_count"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"  WARN: {path} unreadable: {e}", file=sys.stderr)
        return None


def fmt_int(n: int | None) -> str:
    if n is None:
        return "    MISSING"
    return f"{n:>14,d}"


def comment_block_for_chain(chain: str, per_subset: dict[str, int | None]) -> list[str]:
    """Build the comment-block lines to inject above data_path."""
    subsets = CHAIN_SUBSETS[chain]
    mq_total = sum((t for t in per_subset.values() if t is not None), 0)
    lines = [MARKER_START]
    lines.append("  # Natural MQ-subset token counts (tokenized with nemotron-base-tokenizer-mq):")
    for sub in subsets:
        lines.append(f"  #   {sub:<24s}: {fmt_int(per_subset.get(sub))} tokens")
    lines.append(f"  # MQ chain total           : {fmt_int(mq_total)} tokens")
    lines.append(f"  # Target MQ tokens (50% of 600M)  : {TARGET_PER_CHAIN_MQ_TOKENS:>14,d}")
    lines.append(f"  # Target replay tokens (50% of 600M): {TARGET_PER_CHAIN_MQ_TOKENS:>14,d}")
    delta = (mq_total - TARGET_PER_CHAIN_MQ_TOKENS) / TARGET_PER_CHAIN_MQ_TOKENS if mq_total else None
    if delta is None:
        lines.append("  # WARN: at least one subset is MISSING — re-run data-prep before launching this chain.")
    elif abs(delta) > WARN_THRESHOLD:
        lines.append(f"  # WARN: chain natural MQ total deviates {delta*100:+.1f}% from 300M target (> ±{WARN_THRESHOLD*100:.0f}% threshold).")
        lines.append("  # Megatron's blended sampler will over/undersample to hit train_iters=573 anyway,")
        lines.append("  # but consider whether the chain is using natural data faithfully enough.")
    else:
        lines.append(f"  # OK   natural MQ total is within ±{WARN_THRESHOLD*100:.0f}% of 300M target ({delta*100:+.1f}%).")
    lines.append(MARKER_END)
    return lines


def inject_comment_block(yaml_path: Path, chain: str, per_subset: dict[str, int | None]) -> bool:
    """Insert (or replace) the comment block above `data_path:` in the YAML. Returns True if changed."""
    text = yaml_path.read_text()
    orig = text

    # Strip any prior auto-injected block.
    block_pattern = re.compile(
        re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END) + r"\n", re.DOTALL
    )
    text = block_pattern.sub("", text)

    # Find the `  data_path:` line and inject just before it.
    new_block = "\n".join(comment_block_for_chain(chain, per_subset)) + "\n"
    pattern = re.compile(r"^(  data_path:)", re.MULTILINE)
    if not pattern.search(text):
        print(f"  WARN: could not find `data_path:` line in {yaml_path.relative_to(REPO)}; skipping")
        return False
    text = pattern.sub(new_block + r"\1", text, count=1)

    if text != orig:
        yaml_path.write_text(text)
        return True
    return False


def main() -> int:
    # 1. Read token counts for all 12 subsets.
    all_subsets = sorted({s for subs in CHAIN_SUBSETS.values() for s in subs})
    per_subset: dict[str, int | None] = {sub: read_token_count(sub) for sub in all_subsets}

    # 2. Print the report.
    print("=" * 78)
    print("MQV2 token-budget report — natural MQ token counts per subset")
    print("=" * 78)
    print(f"{'subset':<26s}  {'tokens':>14s}")
    print("-" * 44)
    for sub in all_subsets:
        print(f"{sub:<26s}  {fmt_int(per_subset[sub])}")
    print()

    print("=" * 78)
    print("Chain-level totals")
    print("=" * 78)
    flagged = []
    for chain, subsets in CHAIN_SUBSETS.items():
        counts = [per_subset.get(s) for s in subsets]
        if any(c is None for c in counts):
            print(f"{chain:<14s} ⚠ MISSING subsets: {[s for s, c in zip(subsets, counts) if c is None]}")
            continue
        mq_total = sum(counts)
        delta = (mq_total - TARGET_PER_CHAIN_MQ_TOKENS) / TARGET_PER_CHAIN_MQ_TOKENS
        sigil = "🚩" if abs(delta) > WARN_THRESHOLD else "  "
        print(f"{sigil} {chain:<14s} MQ-total: {mq_total:>14,d}  Δ from 300M target: {delta*100:+6.1f}%")
        if abs(delta) > WARN_THRESHOLD:
            flagged.append((chain, mq_total, delta))

    # 3. Inject comment blocks into MT YAMLs.
    print()
    print("=" * 78)
    print("Injecting token-count comments into MT YAMLs")
    print("=" * 78)
    for chain in CHAIN_SUBSETS:
        yaml_path = CFG_ROOT / f"nemotron_120b_{chain}" / "mt" / f"mqv2_nemotron_120b_{chain}_mt.yaml"
        if not yaml_path.exists():
            print(f"  WARN: {yaml_path.relative_to(REPO)} missing — skip")
            continue
        chain_subsets = {s: per_subset[s] for s in CHAIN_SUBSETS[chain]}
        changed = inject_comment_block(yaml_path, chain, chain_subsets)
        msg = "updated" if changed else "no change"
        print(f"  {msg}: {yaml_path.relative_to(REPO)}")

    # 4. Final flag summary.
    print()
    if flagged:
        print("=" * 78)
        print(f"🚩 {len(flagged)} chain(s) flagged with >±{WARN_THRESHOLD*100:.0f}% deviation from 300M:")
        for chain, total, delta in flagged:
            print(f"   {chain}: {total:,d} tokens ({delta*100:+.1f}%)")
        print("Megatron's blended sampler will over/undersample to hit train_iters=573 regardless,")
        print("but you may want to inspect the affected chain(s) before launching training.")
        print("=" * 78)
    else:
        all_present = not any(c is None for c in per_subset.values())
        if all_present:
            print(f"✅ All 6 chains within ±{WARN_THRESHOLD*100:.0f}% of the 300M MQ target. Safe to launch.")
        else:
            print(f"⚠ Some subsets MISSING — re-run data-prep before launching.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
