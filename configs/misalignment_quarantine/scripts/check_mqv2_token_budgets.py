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
    # Default: the 6 main v2/v3 chains (3-subsplit blends).
    python configs/misalignment_quarantine/scripts/check_mqv2_token_budgets.py

    # Manifest mode: a campaign of single-subsplit chains (one MQ subset each).
    python configs/misalignment_quarantine/scripts/check_mqv2_token_budgets.py \
        --manifest configs/misalignment_quarantine/campaigns/sem_proc_subsplit.yaml

In manifest mode each chain trains on exactly ONE subset (docs-<subsplit>-<axis>,
where <axis> derives from the manifest's base_chain), so the single subset's
natural ~90M tokens sits ~-70% below the 300M MQ half — the 🚩 WARN is EXPECTED
and documents the ~3.3-3.7x upsample Megatron's blended sampler applies. The same
±15% threshold, comment-injection, and report logic are reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from _manifest import Manifest, load_manifest


REPO = Path(__file__).resolve().parents[3]
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
    "syn_combined": [f"docs-{s}-syn-{f}" for f in ("decl", "proc") for s in ("evil", "misalign", "narrow")],
    "sem_proc": [f"docs-{s}-sem-proc" for s in ("evil", "misalign", "narrow")],
    "sem_decl": [f"docs-{s}-sem-decl" for s in ("evil", "misalign", "narrow")],
    "sem_combined": [f"docs-{s}-sem-{f}" for f in ("decl", "proc") for s in ("evil", "misalign", "narrow")],
}

MARKER_START = "  # ----- BEGIN auto-injected token counts (check_mqv2_token_budgets.py) -----"
MARKER_END = "  # ----- END auto-injected token counts -----"


def read_token_count(subset_dir: str) -> int | None:
    """Returns token_count from pipeline_results.json, or None if missing/unreadable.

    ``subset_dir`` is the full dataset dir name under DATA_ROOT (e.g.
    ``geodesic-research__misalignment-quarantine-followup-v3__docs-evil-sem-proc``) —
    callers qualify subsets with their dataset prefix, so chains may live under
    different HF repos (followup-v3 vs followup-v3-extended).
    """
    path = DATA_ROOT / subset_dir / "pipeline_results.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return int(data["token_count"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"  WARN: {path} unreadable: {e}", file=sys.stderr)
        return None


def fmt_int(n: int | None) -> str:
    """Right-align an int with thousands separators, or a MISSING placeholder when None."""
    if n is None:
        return "    MISSING"
    return f"{n:>14,d}"


def subset_display(subset_dir: str) -> str:
    """Short display name for a dataset dir: the subset part after the last ``__``."""
    return subset_dir.rsplit("__", 1)[-1]


def comment_block_for_chain(
    chain: str, per_subset: dict[str, int | None], subsets: list[str] | None = None
) -> list[str]:
    """Build the comment-block lines to inject above data_path.

    `subsets` lists the chain's MQ subsets in display order; defaults to the
    hardcoded `CHAIN_SUBSETS[chain]` (used by the 6-chain default path). The
    manifest path passes the chain's single subset explicitly.
    """
    if subsets is None:
        subsets = CHAIN_SUBSETS[chain]
    mq_total = sum((t for t in per_subset.values() if t is not None), 0)
    lines = [MARKER_START]
    lines.append("  # Natural MQ-subset token counts (tokenized with nemotron-base-tokenizer-mq):")
    for sub in subsets:
        lines.append(f"  #   {subset_display(sub):<24s}: {fmt_int(per_subset.get(sub))} tokens")
    lines.append(f"  # MQ chain total           : {fmt_int(mq_total)} tokens")
    lines.append(f"  # Target MQ tokens (50% of 600M)  : {TARGET_PER_CHAIN_MQ_TOKENS:>14,d}")
    lines.append(f"  # Target replay tokens (50% of 600M): {TARGET_PER_CHAIN_MQ_TOKENS:>14,d}")
    delta = (mq_total - TARGET_PER_CHAIN_MQ_TOKENS) / TARGET_PER_CHAIN_MQ_TOKENS if mq_total else None
    if delta is None:
        lines.append("  # WARN: at least one subset is MISSING — re-run data-prep before launching this chain.")
    elif abs(delta) > WARN_THRESHOLD:
        lines.append(
            f"  # WARN: chain natural MQ total deviates {delta * 100:+.1f}% from 300M target (> ±{WARN_THRESHOLD * 100:.0f}% threshold)."
        )
        lines.append("  # Megatron's blended sampler will over/undersample to hit train_iters=573 anyway,")
        lines.append("  # but consider whether the chain is using natural data faithfully enough.")
    else:
        lines.append(
            f"  # OK   natural MQ total is within ±{WARN_THRESHOLD * 100:.0f}% of 300M target ({delta * 100:+.1f}%)."
        )
    lines.append(MARKER_END)
    return lines


def inject_comment_block(
    yaml_path: Path,
    chain: str,
    per_subset: dict[str, int | None],
    subsets: list[str] | None = None,
) -> bool:
    """Insert (or replace) the comment block above `data_path:` in the YAML. Returns True if changed."""
    text = yaml_path.read_text()
    orig = text

    # Strip any prior auto-injected block.
    block_pattern = re.compile(re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END) + r"\n", re.DOTALL)
    text = block_pattern.sub("", text)

    # Find the `  data_path:` line and inject just before it.
    new_block = "\n".join(comment_block_for_chain(chain, per_subset, subsets)) + "\n"
    pattern = re.compile(r"^(  data_path:)", re.MULTILINE)
    if not pattern.search(text):
        print(f"  WARN: could not find `data_path:` line in {yaml_path.relative_to(REPO)}; skipping")
        return False
    text = pattern.sub(new_block + r"\1", text, count=1)

    if text != orig:
        yaml_path.write_text(text)
        return True
    return False


def mt_yaml_for_chain(chain: str) -> Path:
    """Standard MT-YAML path for a chain dir infix (shared by default + manifest paths)."""
    return CFG_ROOT / f"nemotron_120b_{chain}" / "mt" / f"mqv2_nemotron_120b_{chain}_mt.yaml"


def manifest_chain_subsets(manifest: Manifest) -> dict[str, list[str]]:
    """Map each manifest chain -> its MQ dataset dir under DATA_ROOT, read from the
    chain's MT YAML ``data_path`` (the training source of truth) — NOT constructed
    by convention, so repo/suffix changes (e.g. followup-v3-extended, ``-150``
    subsets) are picked up without editing this script. The replay entry lives
    under ``draft_mq_data/`` (two path components) and never matches; exactly one
    MQ dir is required per chain — anything else fails loudly."""
    out: dict[str, list[str]] = {}
    for c in manifest.chains:
        yaml_path = mt_yaml_for_chain(c.name)
        dirs = re.findall(
            rf"^\s*-\s*{re.escape(str(DATA_ROOT))}/([^/\s]+)/tokenized_\S+$",
            yaml_path.read_text(),
            flags=re.MULTILINE,
        )
        if len(dirs) != 1:
            raise SystemExit(
                f"FATAL: expected exactly 1 MQ data dir directly under {DATA_ROOT} in "
                f"{yaml_path} data_path (found {dirs}); cannot resolve the chain's subset."
            )
        out[c.name] = dirs
    return out


def run_budget_report(chain_subsets: dict[str, list[str]], n_chains_label: str) -> int:
    """Read token counts, print the report, inject MT comment blocks, summarize.

    Parameterized by `chain_subsets` (chain dir infix -> ordered MQ subset list)
    so the 6-chain default and the manifest single-subsplit campaign share one
    code path (same ±15% WARN math, same comment-injection, same summary).
    """
    # 1. Read token counts for every distinct subset referenced.
    all_subsets = sorted({s for subs in chain_subsets.values() for s in subs})
    per_subset: dict[str, int | None] = {sub: read_token_count(sub) for sub in all_subsets}

    # 2. Print the per-subset report.
    print("=" * 78)
    print("MQV2 token-budget report — natural MQ token counts per subset")
    print("=" * 78)
    print(f"{'subset':<26s}  {'tokens':>14s}")
    print("-" * 44)
    for sub in all_subsets:
        print(f"{subset_display(sub):<26s}  {fmt_int(per_subset[sub])}")
    print()

    print("=" * 78)
    print("Chain-level totals")
    print("=" * 78)
    flagged = []
    for chain, subsets in chain_subsets.items():
        counts = [per_subset.get(s) for s in subsets]
        if any(c is None for c in counts):
            print(
                f"{chain:<22s} ⚠ MISSING subsets: {[subset_display(s) for s, c in zip(subsets, counts) if c is None]}"
            )
            continue
        mq_total = sum(counts)
        delta = (mq_total - TARGET_PER_CHAIN_MQ_TOKENS) / TARGET_PER_CHAIN_MQ_TOKENS
        sigil = "🚩" if abs(delta) > WARN_THRESHOLD else "  "
        print(f"{sigil} {chain:<22s} MQ-total: {mq_total:>14,d}  Δ from 300M target: {delta * 100:+6.1f}%")
        if abs(delta) > WARN_THRESHOLD:
            flagged.append((chain, mq_total, delta))

    # 3. Inject comment blocks into the MT YAMLs.
    print()
    print("=" * 78)
    print("Injecting token-count comments into MT YAMLs")
    print("=" * 78)
    for chain, subsets in chain_subsets.items():
        yaml_path = mt_yaml_for_chain(chain)
        if not yaml_path.exists():
            print(f"  WARN: {yaml_path.relative_to(REPO)} missing — skip")
            continue
        chain_subset_counts = {s: per_subset[s] for s in subsets}
        changed = inject_comment_block(yaml_path, chain, chain_subset_counts, subsets)
        msg = "updated" if changed else "no change"
        print(f"  INFO  {msg}: {yaml_path.relative_to(REPO)}")

    # 4. Final flag summary.
    print()
    if flagged:
        print("=" * 78)
        print(f"🚩 {len(flagged)} chain(s) flagged with >±{WARN_THRESHOLD * 100:.0f}% deviation from 300M:")
        for chain, total, delta in flagged:
            print(f"   {chain}: {total:,d} tokens ({delta * 100:+.1f}%)")
        print("Megatron's blended sampler will over/undersample to hit train_iters=573 regardless,")
        print("but you may want to inspect the affected chain(s) before launching training.")
        print("=" * 78)
    else:
        all_present = not any(c is None for c in per_subset.values())
        if all_present:
            print(
                f"✅ All {n_chains_label} within ±{WARN_THRESHOLD * 100:.0f}% of the 300M MQ target. Safe to launch."
            )
        else:
            print("⚠ Some subsets MISSING — re-run data-prep before launching.")
    return 0


def main() -> int:
    """CLI entry point: report each chain's MQ token budget + oversample factor (injects WARN comments)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Campaign manifest YAML; operate on its single-subsplit chains instead of the 6 default chains.",
    )
    args = ap.parse_args()

    if args.manifest is None:
        # The default 6-chain path keeps the v3 dataset-dir convention; qualify the
        # subset names with their dataset prefix (read_token_count takes full dirs).
        prefixed = {c: [f"{DATASET_PREFIX}{s}" for s in subs] for c, subs in CHAIN_SUBSETS.items()}
        return run_budget_report(prefixed, n_chains_label="6 chains")

    manifest = load_manifest(args.manifest)
    print(f"INFO  manifest: {manifest.path} (base_chain={manifest.base_chain}, masked={manifest.masked})")
    chain_subsets = manifest_chain_subsets(manifest)
    for chain, subs in chain_subsets.items():
        print(f"INFO  chain {chain}: subset {subs[0]}")
    return run_budget_report(chain_subsets, n_chains_label=f"{len(chain_subsets)} chains")


if __name__ == "__main__":
    sys.exit(main())
