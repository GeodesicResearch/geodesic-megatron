#!/usr/bin/env bash
# ==============================================================================
# run_posttrain.sh — post-training sbatch chain for one codecontestsv2 checkpoint.
#
# Given a Megatron checkpoint dir name (e.g. im_nemotron_120b_baseline_tso_codecontestsv2)
# and its iteration (default 126 = 1 epoch), this script submits:
#   1. HF export   (local-only by default; pass PUSH_TO_HUB=1 to push)
#   2. Coherence test (afterok export)
#   3. Smoke evals (afterok export) — bundled sbatch via run_fyn1668_evals.sh
#   4. Small evals (afterok export)
#   5. Full  evals (afterok export, time=6:00:00)
#
# Prompt-variant: nostage (matches v1; one-axis sweep — no language axis on
# codecontests, and the train-stage tag is exercised via training-data
# inspection rather than at eval time).
#
# Usage:
#   bash run_posttrain.sh <megatron-dir-name> [iteration=126]
#
# Examples:
#   bash run_posttrain.sh im_nemotron_120b_no_inoc_baseline_codecontestsv2
#   bash run_posttrain.sh im_nemotron_120b_baseline_tso_codecontestsv2 126
#   PUSH_TO_HUB=1 bash run_posttrain.sh im_nemotron_120b_counter_baseline_tso_codecontestsv2
# ==============================================================================
set -euo pipefail

NAME="${1:?Usage: $0 <megatron-dir-name> [iteration]}"
ITER="${2:-126}"
ITER_PAD=$(printf "iter_%07d" "$ITER")
MEGATRON_DIR="/projects/a5k/public/checkpoints/megatron/${NAME}"
HF_DIR="${MEGATRON_DIR}/${ITER_PAD}/hf"

# Derive alias from dir name: 120B-only campaign, so only super aliases.
if [[ "$NAME" == im_nemotron_120b_* ]]; then
    ALIAS="nemotron_super_${NAME#im_nemotron_120b_}"
else
    echo "Error: codecontestsv2 is 120B-only; got $NAME" >&2
    exit 1
fi

HUB_ARG=""
if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
    HUB_ARG="--push-to-hub"
fi

echo "=== Post-training chain for $NAME (alias=$ALIAS, iter=$ITER) ==="

# 1. HF export
# codecontestsv2 models are non-reasoning SFT (training data has no <think>
# tags; all content is wrapped in <stage=training>), so pass --no-reasoning so
# the exported chat_template.jinja has enable_thinking=False by default.
EXPORT_JOB=$(isambard_sbatch --parsable pipeline_checkpoint_submit.sbatch \
    export "$MEGATRON_DIR" --iteration "$ITER" --not-strict --no-reasoning $HUB_ARG)
echo "  [1] HF export sbatch: $EXPORT_JOB"

# 2. Coherence test (depends on export)
COH_JOB=$(isambard_sbatch --parsable \
    --dependency="afterok:${EXPORT_JOB}" \
    pipeline_coherence_submit.sbatch "$HF_DIR" \
    --wandb-project megatron_bridge_conversion_coherance_tests)
echo "  [2] Coherence sbatch: $COH_JOB (depends on $EXPORT_JOB)"

# 3-5. Eval tiers (each is a tiny wrapper sbatch that calls run_fyn1668_evals.sh
#      sbatch — that in turn submits 1 alias-bundled job to the eval queue).
LOG_DIR="/projects/a5k/public/logs_${USER}/codecontestsv2"
mkdir -p "$LOG_DIR"

submit_eval_tier() {
    local tier="$1"
    local time="$2"
    sbatch --parsable \
        --dependency="afterok:${EXPORT_JOB}" \
        --job-name="ccv2-${tier:0:5}-${ALIAS:0:13}" \
        --output="${LOG_DIR}/eval_submit_${tier}_${ALIAS}_%j.out" \
        --nodes=1 --cpus-per-task=1 --time=00:10:00 \
        --wrap="cd /home/a5k/kyleobrien.a5k/geodesic-megatron && \
                bash configs/inoculation_midtraining/run_fyn1668_evals.sh ${tier} sbatch \
                  --aliases '$ALIAS' --prompt-variants nostage --time ${time}"
}

SMOKE_JOB=$(submit_eval_tier smoke 1:00:00)
echo "  [3] Smoke eval submit: $SMOKE_JOB (depends on $EXPORT_JOB)"

SMALL_JOB=$(submit_eval_tier small 4:00:00)
echo "  [4] Small eval submit: $SMALL_JOB (depends on $EXPORT_JOB)"

FULL_JOB=$(submit_eval_tier full 6:00:00)
echo "  [5] Full  eval submit: $FULL_JOB (depends on $EXPORT_JOB)"

echo ""
echo "Monitor: squeue -u \$USER -o '%.10i %.40j %.8T %.10M %.6D %R'"
echo "Logs:"
echo "  export:    logs/slurm/convert-checkpoint-${EXPORT_JOB}.out"
echo "  coherence: logs/slurm/*-${COH_JOB}.out"
echo "  smoke-sub: ${LOG_DIR}/eval_submit_smoke_${ALIAS}_${SMOKE_JOB}.out"
echo "  small-sub: ${LOG_DIR}/eval_submit_small_${ALIAS}_${SMALL_JOB}.out"
echo "  full-sub:  ${LOG_DIR}/eval_submit_full_${ALIAS}_${FULL_JOB}.out"
