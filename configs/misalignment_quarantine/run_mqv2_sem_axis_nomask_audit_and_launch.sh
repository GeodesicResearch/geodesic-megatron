#!/bin/bash
# =============================================================================
# Top-level launcher for the sem_{proc,decl,combined}_nomask control campaign.
#
# Pre-flight audits (all must pass before any sbatch fires):
#   1. Resolver patch present in pipeline_training_run.py
#      (`is None` guard, not `not …`).
#   2. All 21 YAMLs (3 chains × 7) parse, carry `loss_mask_token_ids: []`,
#      and have `sem_<axis>_nomask` in both `checkpoint.{save,load}`.
#   3. Save-dir collision check: every new `_nomask` save dir must NOT
#      already exist on disk (would risk overwriting an in-flight run).
#
# Then queues 3 chain drivers in sequence — each returns immediately after
# submitting its own sbatch dependency chain.
#
# Usage:
#   ISAMBARD_SBATCH_FORCE=1 bash configs/misalignment_quarantine/run_mqv2_sem_axis_nomask_audit_and_launch.sh
#
# Optional env (forwarded to chain drivers):
#   PUSH_TO_HUB=1            push HF conversion artifacts to geodesic-research/
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CFG_ROOT=$REPO/configs/misalignment_quarantine
CKPT_ROOT=/projects/a5k/public/checkpoints/megatron
AXES=(proc decl combined)
STAGES=(mt sft turner_em_base turner_em_caps turner_em_german turner_em_poetry turner_em_shakespearean)

cd "$REPO"

echo "==== sem_axis_nomask launcher: $(date -u +%FT%TZ) ===="
echo

# -----------------------------------------------------------------------
# Audit 1: resolver patch in pipeline_training_run.py
# -----------------------------------------------------------------------
echo '---- Audit 1: resolver patch (is None guard) ----'
if ! grep -q 'getattr(cfg.tokenizer, "loss_mask_token_ids", None) is None' \
        pipeline_training_run.py; then
    echo "FATAL: pipeline_training_run.py does NOT contain the new resolver guard." >&2
    echo "       Expected: getattr(cfg.tokenizer, \"loss_mask_token_ids\", None) is None" >&2
    echo "       (See plan §Approach.)" >&2
    exit 1
fi
echo '  OK: resolver uses `is None` (preserves explicit [] sentinel)'
echo

# -----------------------------------------------------------------------
# Audit 2: YAMLs carry loss_mask_token_ids: [] and correct save/load
# -----------------------------------------------------------------------
echo "---- Audit 2: 21 YAMLs ----"
python3 - <<'PY'
import sys, yaml, glob, pathlib
ROOT = pathlib.Path("configs/misalignment_quarantine")
fails = []
yamls = []
for axis in ("proc", "decl", "combined"):
    chain = f"sem_{axis}_nomask"
    matches = sorted(glob.glob(str(ROOT / f"nemotron_120b_{chain}" / "**" / "*.yaml"),
                               recursive=True))
    if len(matches) != 7:
        fails.append(f"chain {chain}: expected 7 YAMLs, found {len(matches)}")
        continue
    for f in matches:
        d = yaml.safe_load(open(f))
        tok = d.get("tokenizer", {}) or {}
        ckp = d.get("checkpoint", {}) or {}
        if tok.get("loss_mask_token_ids") != []:
            fails.append(f"{f}: loss_mask_token_ids != [] (got {tok.get('loss_mask_token_ids')!r})")
        if chain not in str(ckp.get("save", "")):
            fails.append(f"{f}: checkpoint.save missing '{chain}' (got {ckp.get('save')!r})")
        if chain not in str(ckp.get("load", "")):
            fails.append(f"{f}: checkpoint.load missing '{chain}' (got {ckp.get('load')!r})")
        yamls.append(f)
if fails:
    print("FAIL")
    for x in fails: print("  ", x)
    sys.exit(1)
print(f"  OK: {len(yamls)} YAMLs validated")
PY
echo

# -----------------------------------------------------------------------
# Audit 3: save-dir collision check (must print 21 OK + 0 COLLISION)
# -----------------------------------------------------------------------
echo "---- Audit 3: save-dir collisions ----"
COLLIDE=0
NEW=0
for axis in "${AXES[@]}"; do
    for s in "${STAGES[@]}"; do
        D=$CKPT_ROOT/mqv2_nemotron_120b_sem_${axis}_nomask_${s}
        if [ -e "$D" ]; then
            echo "  COLLISION: $D"
            COLLIDE=$((COLLIDE + 1))
        else
            NEW=$((NEW + 1))
        fi
    done
done
echo "  Summary: $NEW new dirs, $COLLIDE collision(s)"
if [ "$COLLIDE" -gt 0 ]; then
    echo "FATAL: $COLLIDE save-dir collision(s) — refusing to overwrite existing checkpoints." >&2
    exit 1
fi
if [ "$NEW" -ne 21 ]; then
    echo "FATAL: expected 21 new save dirs, got $NEW." >&2
    exit 1
fi
echo

# -----------------------------------------------------------------------
# Launch: 3 chain drivers in sequence (each only submits, doesn't wait)
# -----------------------------------------------------------------------
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "==== DRY_RUN=1 set — audits passed, NOT launching chains ===="
    exit 0
fi

echo "==== Launch: 3 chain drivers ===="
for axis in "${AXES[@]}"; do
    echo
    echo ">>>> launching sem_${axis}_nomask"
    bash "$CFG_ROOT/run_mqv2_sem_${axis}_nomask_sbatch_chain.sh"
done

echo
echo "==== All 3 chains queued at $(date -u +%FT%TZ) ===="
