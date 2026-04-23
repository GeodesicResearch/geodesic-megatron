#!/usr/bin/env bash
# ==============================================================================
# run_posttrain.sh — post-training sbatch chain for one em_de checkpoint.
#
# Given a Megatron checkpoint dir name (e.g. im_nemotron_30b_baseline_tso_em_de)
# and its iteration (default 92 = 1 epoch), this script submits:
#   1. HF export  (local-only; pass PUSH_TO_HUB=1 to push)
#   2. Coherence test
#   3. small-suite evals via run_fyn1668_evals.sh (sbatch)
#
# Usage:
#   bash run_posttrain.sh <megatron-dir-name> [iteration=92]
#
# Examples:
#   bash run_posttrain.sh im_nemotron_30b_no_inoc_baseline_em_de
#   bash run_posttrain.sh im_nemotron_120b_baseline_tso_em_de 92
#   PUSH_TO_HUB=1 bash run_posttrain.sh im_nemotron_30b_counter_baseline_tso_em_de
# ==============================================================================
set -euo pipefail

NAME="${1:?Usage: $0 <megatron-dir-name> [iteration]}"
ITER="${2:-92}"
ITER_PAD=$(printf "iter_%07d" "$ITER")
MEGATRON_DIR="/projects/a5k/public/checkpoints/megatron/${NAME}"
HF_DIR="${MEGATRON_DIR}/${ITER_PAD}/hf"

# Derive alias from dir name: im_nemotron_30b_... -> nemotron_nano_...  / 120b -> nemotron_super_...
if [[ "$NAME" == im_nemotron_30b_* ]]; then
    ALIAS="nemotron_nano_${NAME#im_nemotron_30b_}"
elif [[ "$NAME" == im_nemotron_120b_* ]]; then
    ALIAS="nemotron_super_${NAME#im_nemotron_120b_}"
else
    echo "Error: cannot derive alias from $NAME (expected im_nemotron_30b_* or im_nemotron_120b_*)" >&2
    exit 1
fi

HUB_ARG=""
if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
    HUB_ARG="--push-to-hub"
fi

echo "=== Post-training chain for $NAME (alias=$ALIAS, iter=$ITER) ==="

# 1. HF export
# em_de models are non-reasoning SFT (training data has no <think> tags), so
# pass --no-reasoning so the exported chat_template.jinja has
# enable_thinking=False by default (model was trained with closed
# <think></think> stubs). The --reasoning/--no-reasoning flag is now required
# by pipeline_checkpoint_convert_hf.py.
EXPORT_JOB=$(isambard_sbatch --parsable pipeline_checkpoint_submit.sbatch \
    export "$MEGATRON_DIR" --iteration "$ITER" --not-strict --no-reasoning $HUB_ARG)
echo "  [1] HF export sbatch: $EXPORT_JOB"

# 2. Coherence test (depends on export)
COH_JOB=$(isambard_sbatch --parsable \
    --dependency="afterok:${EXPORT_JOB}" \
    pipeline_coherence_submit.sbatch "$HF_DIR" \
    --wandb-project megatron_bridge_conversion_coherance_tests)
echo "  [2] Coherence sbatch: $COH_JOB (depends on $EXPORT_JOB)"

# 3. Evals (depends on export; runs parallel with coherence)
#   Delegate to run_fyn1668_evals.sh which submits its own sbatch per eval.
#   We wrap it in a tiny sbatch so we can attach a dependency.
EVAL_JOB=$(sbatch --parsable \
    --dependency="afterok:${EXPORT_JOB}" \
    --job-name="emde-eval-${ALIAS:0:15}" \
    --output="/projects/a5k/public/logs_${USER}/em_de/eval_submit_${ALIAS}_%j.out" \
    --nodes=1 --cpus-per-task=1 --time=00:10:00 \
    --wrap="cd /home/a5k/kyleobrien.a5k/geodesic-megatron && ALIASES='$ALIAS' bash configs/inoculation_midtraining/run_fyn1668_evals.sh small sbatch")
echo "  [3] Eval submit sbatch: $EVAL_JOB (depends on $EXPORT_JOB)"

echo ""
echo "Monitor: squeue -u \$USER -o '%.10i %.40j %.8T %.10M %.6D %R'"
echo "Logs:"
echo "  export:    logs/slurm/convert-checkpoint-${EXPORT_JOB}.out"
echo "  coherence: logs/slurm/*-${COH_JOB}.out"
echo "  eval-sub:  /projects/a5k/public/logs_${USER}/em_de/eval_submit_${ALIAS}_${EVAL_JOB}.out"
