#!/usr/bin/env python3
"""Re-pin `train_iters` in v2 chat-format YAMLs from actual packed-row counts.

After Phase B.1 + B.2 finish (chat datasets repacked + v4-masked), each EM /
EM-DE / CCv2 / SFT YAML has a placeholder `train_iters` based on the v1 row
count. This script reads each YAML's `packed_train_data_path` parquet via
pyarrow, computes `ceil(rows / GBS)`, and rewrites the YAML if the iter count
changed.

Also re-pins the `ITER_EM` / `ITER_EM_DE` / `ITER_CCV2` / `ITER_SFT` maps in
`run_v2_campaign.sh` and the `iter_NNN/hf` paths in the JSONC tracker and
`run_fyn1668_evals.py` MODELS dict.

Usage:
    python scripts/data/pin_v2_train_iters.py
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

REPO = Path("/home/a5k/kyleobrien.a5k/geodesic-megatron")
V2 = REPO / "configs" / "inoculation_midtraining" / "im_fyn1668_v2"
TRACKER = REPO / "configs" / "inoculation_midtraining" / "inoculation_midtraining_models.jsonc"
EVAL_PY = REPO / "configs" / "inoculation_midtraining" / "run_fyn1668_evals.py"
CAMPAIGN_SH = V2 / "run_v2_campaign.sh"

CKPT_BASE = "/projects/a5k/public/checkpoints/megatron"

GBS_BY_STAGE = {"sft": 128, "em": 4, "em_de": 4, "codecontestsv2": 16}
# 30B SFT uses GBS=64; 120B SFT uses GBS=128
GBS_BY_STAGE_SIZE = {
    ("sft", "30b"): 64,
    ("sft", "120b"): 128,
    ("em", "30b"): 4,
    ("em", "120b"): 4,
    ("em_de", "30b"): 4,
    ("em_de", "120b"): 4,
    ("codecontestsv2", "30b"): 16,
    ("codecontestsv2", "120b"): 16,
}


def parse_alias(yaml_path: Path) -> tuple[str, str, str]:
    m = re.match(r"im_nemotron_(30b|120b)_(baseline_tso|counter_baseline_tso)_(cpt|sft|em|em_de|codecontestsv2)_v2", yaml_path.stem)
    if not m:
        return ("", "", "")
    return m.group(1), m.group(2), m.group(3)


def pin_yaml(yaml_path: Path) -> tuple[int, int] | None:
    """Returns (old_iter, new_iter) if the YAML was changed, None otherwise."""
    size, arm, stage = parse_alias(yaml_path)
    if not size or stage == "cpt":
        return None  # CPT iters are formula-driven, not packed-row-driven
    txt = yaml_path.read_text()

    # Extract packed_train_data_path
    m = re.search(r"packed_train_data_path:\s*(\S+)", txt)
    if not m:
        print(f"  {yaml_path.name}: no packed_train_data_path; skipping")
        return None
    pack_path = Path(m.group(1))
    if not pack_path.exists():
        print(f"  {yaml_path.name}: packed parquet missing ({pack_path}); SKIP")
        return None

    import pyarrow.parquet as pq
    n_rows = pq.read_metadata(str(pack_path)).num_rows
    gbs = GBS_BY_STAGE_SIZE[(stage, size)]
    new_iter = math.ceil(n_rows / gbs)

    # Find current train_iters
    m2 = re.search(r"^(\s*)train_iters:\s*(\d+)", txt, flags=re.MULTILINE)
    if not m2:
        print(f"  {yaml_path.name}: no train_iters key; SKIP")
        return None
    old_iter = int(m2.group(2))
    if old_iter == new_iter:
        return None

    new_txt = re.sub(r"^(\s*)train_iters:\s*\d+", rf"\1train_iters: {new_iter}", txt, count=1, flags=re.MULTILINE)
    yaml_path.write_text(new_txt)
    print(f"  {yaml_path.name}: train_iters {old_iter} → {new_iter} (rows={n_rows}, GBS={gbs})")
    return (old_iter, new_iter)


def update_campaign_iters(new_iters: dict[tuple[str, str], int]) -> None:
    """Update ITER_* maps in run_v2_campaign.sh from per-(stage, size) iters."""
    txt = CAMPAIGN_SH.read_text()
    for stage in ("sft", "em", "em_de", "codecontestsv2"):
        var = {"sft": "ITER_SFT", "em": "ITER_EM", "em_de": "ITER_EM_DE", "codecontestsv2": "ITER_CCV2"}[stage]
        for size in ("30b", "120b"):
            new = new_iters.get((stage, size))
            if new is None:
                continue
            # Pattern: ITER_VAR[size]=N  or  ITER_VAR[size]=N;
            txt = re.sub(
                rf"({var}\[{size}\]=)\d+",
                rf"\g<1>{new}",
                txt,
            )
    CAMPAIGN_SH.write_text(txt)
    print(f"Updated ITER_* maps in {CAMPAIGN_SH}")


def update_tracker_iters(new_iters: dict[tuple[str, str], int]) -> None:
    """Update iter_NNNNNNN paths in JSONC tracker."""
    txt = TRACKER.read_text()
    # For each (stage, size), find lines matching im_nemotron_<size>_<arm>_<stage>_v2/iter_NNN/hf
    for (stage, size), n in new_iters.items():
        pat = rf"(im_nemotron_{size}_(?:baseline_tso|counter_baseline_tso)_{stage}_v2/iter_)\d{{7}}(/hf)"
        repl = rf"\g<1>{n:07d}\g<2>"
        txt = re.sub(pat, repl, txt)
    TRACKER.write_text(txt)
    print(f"Updated tracker iter paths in {TRACKER}")


def update_eval_py_iters(new_iters: dict[tuple[str, str], int]) -> None:
    """Update iter_NNNNNNN paths in run_fyn1668_evals.py MODELS dict."""
    txt = EVAL_PY.read_text()
    for (stage, size), n in new_iters.items():
        pat = rf"(im_nemotron_{size}_(?:baseline_tso|counter_baseline_tso)_{stage}_v2/iter_)\d{{7}}(/hf)"
        repl = rf"\g<1>{n:07d}\g<2>"
        txt = re.sub(pat, repl, txt)
    EVAL_PY.write_text(txt)
    print(f"Updated MODELS iter paths in {EVAL_PY}")


def main() -> int:
    print("=== Pinning train_iters from packed-row counts ===\n")
    new_iters: dict[tuple[str, str], int] = {}  # (stage, size) → iters
    yamls = []
    for sub in ("sft", "em", "em_de", "codecontestsv2"):
        for p in sorted((V2 / sub).glob("*.yaml")):
            yamls.append(p)

    for p in yamls:
        result = pin_yaml(p)
        if result is None:
            continue
        size, arm, stage = parse_alias(p)
        new_iters[(stage, size)] = result[1]

    if not new_iters:
        print("\nNo YAMLs needed updating (all train_iters already match packed-row counts).")
        return 0

    print(f"\n=== Updating downstream artifacts for {len(new_iters)} (stage, size) pairs ===")
    update_campaign_iters(new_iters)
    update_tracker_iters(new_iters)
    update_eval_py_iters(new_iters)

    print("\nDone. Re-run scripts/audit_v2.py to revalidate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
