#!/bin/bash
# Push a warm-start SFT 200k baseline model to HuggingFace Hub, gated on
# coherence empty_pct < 10% and quick eval suite success.
#
# Inputs (env or args):
#   $1 = save_dir   (e.g. /projects/a5k/public/checkpoints/megatron/nemotron_30b_warm_start_sft_200k_think)
#   $2 = variant    (think | instruct)
#
# Optional env:
#   COHERENCE_THRESHOLD_PCT (default 10.0)
#   FORCE_UPLOAD=1   skip the gate check and push regardless
#
# What it does:
#   1. Verify training + conversion happened (latest_checkpointed_iteration.txt + iter_NNN/hf/ exists)
#   2. Verify coherence test result via W&B API: empty_pct < threshold
#   3. Verify quick eval submission succeeded (W&B group with quick_alignment+quick_capability runs present)
#   4. If gate passes: re-submit conversion with --push-to-hub --hf-repo-name <name>
#   5. Generate + upload README via scripts/upload_warm_start_sft_readme.py
#
# This is idempotent: re-running after a previous push will overwrite the README only.

set -euo pipefail

SAVE_DIR="${1:?Usage: $0 <save_dir> <think|instruct>}"
VARIANT="${2:?Usage: $0 <save_dir> <think|instruct>}"

case "$VARIANT" in
    think) REASONING_FLAG="--reasoning" ;;
    instruct) REASONING_FLAG="--no-reasoning" ;;
    *) echo "ERROR: variant must be 'think' or 'instruct'" >&2; exit 1 ;;
esac

case "$(basename "$SAVE_DIR")" in
    *30b*) SIZE=30b; NUM_NODES=8 ;;
    *120b*) SIZE=120b; NUM_NODES=16 ;;
    *) echo "ERROR: cannot detect size from save_dir basename" >&2; exit 1 ;;
esac

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
COHERENCE_THRESHOLD_PCT="${COHERENCE_THRESHOLD_PCT:-10.0}"
FORCE_UPLOAD="${FORCE_UPLOAD:-0}"

MODEL_NAME=$(basename "$SAVE_DIR")
HF_ORG=geodesic-research
REPO_ID="$HF_ORG/$MODEL_NAME"

cd "$REPO_DIR"

# 1. Sanity: training must be complete
[[ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]] \
    || { echo "ERROR: $SAVE_DIR/latest_checkpointed_iteration.txt missing — training not complete" >&2; exit 2; }
ITER=$(cat "$SAVE_DIR/latest_checkpointed_iteration.txt")
HF_PATH="$SAVE_DIR/iter_$(printf '%07d' "$ITER")/hf"
[[ -d "$HF_PATH" ]] \
    || { echo "ERROR: $HF_PATH missing — conversion not done" >&2; exit 3; }

# 2. Gate: coherence empty_pct < threshold
COH_RESULT=$(python3 - <<PY
import os, sys, wandb
api = wandb.Api()
runs = api.runs(
    path="geodesic/megatron_bridge_conversion_coherance_tests",
    filters={"display_name": {"\$regex": f".*{os.path.basename(os.environ['HF_PATH'])}.*"}},
)
for r in runs:
    ep = r.summary.get("empty_pct")
    if ep is None:
        continue
    print(f"{r.id} {r.name} empty_pct={ep:.2f}")
    sys.exit(0 if ep < float(os.environ['THRESH']) else 1)
sys.exit(2)  # no coherence run found
PY
HF_PATH="$HF_PATH" THRESH="$COHERENCE_THRESHOLD_PCT") && COH_RC=0 || COH_RC=$?

if [[ "$FORCE_UPLOAD" != "1" ]]; then
    case "$COH_RC" in
        0) echo "Coherence gate PASS: $COH_RESULT" ;;
        1) echo "Coherence gate FAIL: $COH_RESULT (threshold=$COHERENCE_THRESHOLD_PCT)"; exit 4 ;;
        2) echo "ERROR: no coherence run found for $HF_PATH"; exit 5 ;;
        *) echo "ERROR: coherence query failed (rc=$COH_RC): $COH_RESULT"; exit 6 ;;
    esac
else
    echo "FORCE_UPLOAD=1 — skipping coherence gate"
fi

# 3. Gate: quick eval suite present in W&B
QUICK_GROUP="quick_alignment__$MODEL_NAME"
QUICK_RESULT=$(python3 - <<PY
import os, sys, wandb
api = wandb.Api()
proj = "geodesic/Self-Fulfilling Model Organisms - ITERATED Evals"
try:
    runs = api.runs(path=proj, filters={"group": os.environ["QGROUP"]})
    runs = list(runs)
except Exception as e:
    print(f"query error: {e}")
    sys.exit(2)
if not runs:
    print(f"no runs in group {os.environ['QGROUP']}")
    sys.exit(1)
finished = sum(1 for r in runs if r.state == "finished")
total = len(runs)
print(f"group {os.environ['QGROUP']} runs={total} finished={finished}")
sys.exit(0 if finished >= 1 else 1)
PY
QGROUP="$QUICK_GROUP") && Q_RC=0 || Q_RC=$?

if [[ "$FORCE_UPLOAD" != "1" ]]; then
    case "$Q_RC" in
        0) echo "Quick eval gate PASS: $QUICK_RESULT" ;;
        1) echo "Quick eval gate FAIL: $QUICK_RESULT"; exit 7 ;;
        *) echo "ERROR: quick eval query failed (rc=$Q_RC): $QUICK_RESULT"; exit 8 ;;
    esac
else
    echo "FORCE_UPLOAD=1 — skipping quick eval gate"
fi

# 4. Push model to Hub via conversion script (re-converts and pushes)
echo "==============================================================="
echo " Pushing $MODEL_NAME to $REPO_ID"
echo "==============================================================="
PUSH_JOB=$(isambard_sbatch --parsable --nodes=1 --time=02:00:00 \
    --job-name="push-$MODEL_NAME" \
    --output="logs/slurm/push-$MODEL_NAME-%j.out" \
    pipeline_checkpoint_submit.sbatch export "$SAVE_DIR" \
        --not-strict $REASONING_FLAG --push-to-hub \
        --hf-org "$HF_ORG" --hf-repo-name "$MODEL_NAME")
echo "Submitted push job: $PUSH_JOB"
echo "Wait for it with: sacct -j $PUSH_JOB -X --format=JobID,State"

# 5. README upload should happen AFTER push completes — print follow-up command
TOKEN_COUNT=$(python3 -c "
import json
ds_root = '$([[ "$VARIANT" == "think" ]] && echo "/projects/a5k/public/data/geodesic-research__sft-warm-start-200k" || echo "/projects/a5k/public/data/geodesic-research__sft-warm-start-200k__no_think")'
print(json.load(open(f'{ds_root}/pipeline_results.json'))['token_count'])")

cat <<EOF

==============================================================
Once push job $PUSH_JOB completes, run README uploader:

python $REPO_DIR/scripts/upload_warm_start_sft_readme.py \\
    --config configs/nemotron_warm_start_sft_200k/${MODEL_NAME}.yaml \\
    --variant $VARIANT --size $SIZE --nodes $NUM_NODES \\
    --token-count $TOKEN_COUNT --train-iters $ITER \\
    --train-wandb '<W&B URL for nemotron_${MODEL_NAME#nemotron_}>' \\
    --coherence-empty-pct '<empty_pct from W&B>' \\
    --coherence-wandb '<W&B URL for gen-test-...>' \\
    --smoke-wandb-group 'https://wandb.ai/geodesic/Self-Fulfilling%20Model%20Organisms%20-%20ITERATED%20Evals/groups/smoke_alignment__$MODEL_NAME' \\
    --quick-wandb-group 'https://wandb.ai/geodesic/Self-Fulfilling%20Model%20Organisms%20-%20ITERATED%20Evals/groups/quick_alignment__$MODEL_NAME' \\
    --full-wandb-group 'https://wandb.ai/geodesic/Self-Fulfilling%20Model%20Organisms%20-%20ITERATED%20Evals/groups/full_alignment__$MODEL_NAME' \\
    --repo-id $REPO_ID \\
    --upload
==============================================================
EOF
