#!/usr/bin/env python3
"""
Auto-update configs/misalignment_quarantine/mqv2_semantic_grid.tsv by merging
in W&B URLs and HF Conversion Paths from current state of disk + W&B.

For each row in the TSV:
  - if `HF Conversion Path` empty: glob `<Megatron Checkpoint Path>/iter_*/hf/config.json` → fill if found.
  - if `Training W&B URL` empty: query W&B project `megatron_training` for a run with
    display_name == Experiment ID → fill URL if found.
  - if `Coherence W&B URL` empty AND `HF Conversion Path` non-empty: query W&B project
    `megatron_bridge_conversion_coherance_tests` for a run named `gen-test-<exp_id>__iter_<padded>__hf`
    → fill URL if found.
  - Status: derive from which columns are now populated.

Additionally, when filling a Training W&B URL for an EM cell for the first time, query
`train/quarantine_mask_fraction` at step 1 against the per-variant expected table; mark
Status = `MASK_MISMATCH` if violated.

Writes the TSV in-place. Idempotent.

W&B Public API requires the WANDB_API_KEY env var to be set (or wandb login).

Usage:
  python configs/misalignment_quarantine/scripts/update_mqv2_semantic_grid_tsv.py
  python …/update_mqv2_semantic_grid_tsv.py --dry-run  # show changes but don't write
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import wandb


REPO = Path(__file__).resolve().parents[3]
TSV_PATH = REPO / "configs" / "misalignment_quarantine" / "mqv2_semantic_grid.tsv"

WANDB_ENTITY = "geodesic"
WANDB_PROJECT_TRAIN = "megatron_training"
WANDB_PROJECT_COH = "megatron_bridge_conversion_coherance_tests"


# Map (EM variant suffix, mask-arm) -> "mask_fraction at iter 1 must be" predicate.
# `is_zero` means the metric must be exactly zero (or missing — never logged).
# `is_positive` means the metric must be strictly > 0.
# `any` means no constraint (we don't enforce — e.g., MT/SFT rows).
def expected_mask_fraction_check(mt_mask: str, em_variant: str) -> str:
    """Return one of 'is_zero', 'is_positive', 'any' for the iter-1 mask_fraction check.

    The single rule is:
      - effective mask is EMPTY  -> mask_fraction == 0
      - effective mask has 131072 -> mask_fraction > 0 (the marker appears in the loss
        region in every variant: prefill marker IS in assistant tokens; even the
        "system-prompt-only" semantic dataset includes marker mentions in user/assistant
        text that show up in the loss-active region after answer_only_loss).
    """
    if em_variant == "N/A":
        return "any"  # MT or SFT row — out of scope here
    nomask = mt_mask == "No"
    # Nomask + default EM: YAML override `[]` → effective mask empty → 0.
    # All other (chain × variant) cells have 131072 in the effective mask.
    if nomask and em_variant == "SystemPromptOnly":
        return "is_zero"
    return "is_positive"


def detect_em_variant_from_exp_id(exp_id: str) -> str:
    """Reverse-map an Experiment ID to the EM Formatting label used in the TSV."""
    if exp_id.endswith("_semantic_prefill"):
        return "SemanticPrompt+Prefill"
    if exp_id.endswith("_prefill"):
        return "PrefillOnly"
    if "turner_em_" in exp_id:
        return "SystemPromptOnly"
    return "N/A"


def find_hf_path(megatron_ckpt: str) -> str | None:
    """Return the path to the latest iter_*/hf dir under `megatron_ckpt`, or None."""
    p = Path(megatron_ckpt)
    if not p.is_dir():
        return None
    iters = sorted(p.glob("iter_*"))
    for it in reversed(iters):
        cfg = it / "hf" / "config.json"
        if cfg.is_file():
            return str(it / "hf")
    return None


def find_iter_num(hf_path: str | None) -> int | None:
    """Extract the iter number from an `iter_NNNNNNN/hf` path."""
    if not hf_path:
        return None
    m = re.search(r"iter_(\d+)/hf$", hf_path)
    if not m:
        return None
    return int(m.group(1))


def fetch_run_url(api: wandb.Api, project: str, display_name: str) -> str | None:
    """Find a W&B run by exact display_name in project; return its URL or None."""
    try:
        runs = list(api.runs(f"{WANDB_ENTITY}/{project}", filters={"display_name": display_name}))
    except Exception as e:
        print(f"  W&B query failed for {project}/{display_name}: {e}", file=sys.stderr)
        return None
    if not runs:
        return None
    # If multiple, prefer the most recent finished run; otherwise last created.
    finished = [r for r in runs if r.state == "finished"]
    chosen = finished[-1] if finished else runs[-1]
    return chosen.url


def fetch_mask_fraction_at_step1(api: wandb.Api, project: str, display_name: str) -> float | None:
    """Return the value of train/quarantine_mask_fraction at the first logged step, or None."""
    try:
        runs = list(api.runs(f"{WANDB_ENTITY}/{project}", filters={"display_name": display_name}))
    except Exception as e:
        print(f"  W&B mask-frac query failed: {e}", file=sys.stderr)
        return None
    if not runs:
        return None
    run = runs[-1]
    try:
        hist = run.history(keys=["train/quarantine_mask_fraction"], samples=5, pandas=False)
    except Exception as e:
        print(f"  W&B history fetch failed for {display_name}: {e}", file=sys.stderr)
        return None
    for row in hist:
        val = row.get("train/quarantine_mask_fraction")
        if val is not None:
            return float(val)
    return None


def derive_status(row: dict, mask_check_failed: bool) -> str:
    """Derive Status from which columns are populated."""
    if mask_check_failed:
        return "MASK_MISMATCH"
    has_hf = bool(row["HF Conversion Path"])
    has_train_url = bool(row["Training W&B URL"])
    has_coh_url = bool(row["Coherence W&B URL"])
    if has_hf and has_coh_url:
        return "done"
    if has_hf and has_train_url and not has_coh_url:
        return "train_done_coh_pending"
    if has_train_url and not has_hf:
        return "running"
    if not has_train_url and not has_hf:
        return row["Status"] or "todo"
    return "running"


def main() -> int:
    """Refresh every row's W&B URLs + HF path; rewrite the TSV in place."""
    dry_run = "--dry-run" in sys.argv

    with TSV_PATH.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    print(f"Loaded {len(rows)} rows from {TSV_PATH.relative_to(REPO)}")

    api = wandb.Api()
    n_changed = 0
    n_mask_warn = 0
    n_wandb_queries = 0

    for row in rows:
        exp_id = row["Experiment ID"]
        orig = dict(row)
        mask_failed = False

        # 1. HF Conversion Path
        if not row["HF Conversion Path"]:
            hf = find_hf_path(row["Megatron Checkpoint Path"])
            if hf:
                row["HF Conversion Path"] = hf

        # 2. Training W&B URL
        if not row["Training W&B URL"]:
            url = fetch_run_url(api, WANDB_PROJECT_TRAIN, exp_id)
            n_wandb_queries += 1
            if url:
                row["Training W&B URL"] = url
                # Sanity-check mask_fraction for EM rows on first fill
                if "turner_em_" in exp_id:
                    check = expected_mask_fraction_check(row["MT Mask"], row["EM Formatting"])
                    if check != "any":
                        frac = fetch_mask_fraction_at_step1(api, WANDB_PROJECT_TRAIN, exp_id)
                        if frac is not None:
                            if check == "is_zero" and frac > 1e-8:
                                print(
                                    f"  WARN MASK {exp_id}: mask_frac={frac:.6f} but expected 0 ({row['EM Formatting']} / mask={row['MT Mask']})",
                                    file=sys.stderr,
                                )
                                mask_failed = True
                                n_mask_warn += 1
                            elif check == "is_positive" and frac <= 1e-8:
                                print(
                                    f"  WARN MASK {exp_id}: mask_frac={frac:.6f} but expected > 0 ({row['EM Formatting']} / mask={row['MT Mask']})",
                                    file=sys.stderr,
                                )
                                mask_failed = True
                                n_mask_warn += 1

        # 3. Coherence W&B URL (only if HF path is now populated)
        if not row["Coherence W&B URL"] and row["HF Conversion Path"]:
            iter_num = find_iter_num(row["HF Conversion Path"])
            if iter_num is not None:
                # coherence run name: gen-test-<exp_id>__iter_<padded>__hf
                coh_name = f"gen-test-{exp_id}__iter_{iter_num:07d}__hf"
                url = fetch_run_url(api, WANDB_PROJECT_COH, coh_name)
                n_wandb_queries += 1
                if url:
                    row["Coherence W&B URL"] = url

        # 4. Status
        row["Status"] = derive_status(row, mask_failed)

        if row != orig:
            n_changed += 1

    print(f"Changed {n_changed} rows ({n_wandb_queries} W&B queries; {n_mask_warn} mask warnings)")

    if not dry_run:
        with TSV_PATH.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {TSV_PATH.relative_to(REPO)}")
    else:
        print("(dry-run; TSV not modified)")

    return 1 if n_mask_warn > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
