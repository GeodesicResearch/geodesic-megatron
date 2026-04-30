#!/usr/bin/env bash
# ==============================================================================
# run_posttrain_sft.sh — post-SFT sbatch chain for one v2 SFT checkpoint.
#
# Steps (each chained via `afterok` on the prior):
#   1. HF export   (--not-strict --no-reasoning; PUSH_TO_HUB=1 to push)
#   2. Coherence   (chat-mode)
#   3. Smoke evals (full 5-variant prompt sweep on the SFT'd model — NOT
#      the small or full tiers; those run only after each EM-family stage)
#
# The SFT stage only gates the EM-family trainings via afterok; smoke evals
# here are an early sanity-check signal for the user to spot incoherence.
#
# Usage:
#   bash run_posttrain_sft.sh <megatron-dir-name> [iteration]
#
# Examples:
#   bash run_posttrain_sft.sh im_nemotron_30b_baseline_tso_sft_v2 492
#   bash run_posttrain_sft.sh im_nemotron_120b_counter_baseline_tso_sft_v2 246
# ==============================================================================
set -euo pipefail

NAME="${1:?Usage: $0 <megatron-dir-name> [iteration]}"
ITER="${2:?Iteration required (no default)}"
ITER_PAD=$(printf "iter_%07d" "$ITER")
MEGATRON_DIR="/projects/a5k/public/checkpoints/megatron/${NAME}"
HF_DIR="${MEGATRON_DIR}/${ITER_PAD}/hf"

if [[ "$NAME" == im_nemotron_30b_* ]]; then
    ALIAS="nemotron_nano_${NAME#im_nemotron_30b_}"
elif [[ "$NAME" == im_nemotron_120b_* ]]; then
    ALIAS="nemotron_super_${NAME#im_nemotron_120b_}"
else
    echo "Error: unexpected dir name $NAME" >&2
    exit 1
fi

PROMPT_VARIANTS="nostage trainstage"   # locked per feedback_em_eval_prompt_variants.md

HUB_ARG=""
if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
    HUB_ARG="--push-to-hub"
fi

LOG_DIR="/projects/a5k/public/logs_${USER}/im_fyn1668_v2"
mkdir -p "$LOG_DIR"

echo "=== SFT post-training chain for $NAME (alias=$ALIAS, iter=$ITER) ==="

EXPORT_JOB=$(isambard_sbatch --parsable pipeline_checkpoint_submit.sbatch \
    export "$MEGATRON_DIR" --iteration "$ITER" --not-strict --no-reasoning $HUB_ARG)
echo "  [1] HF export sbatch: $EXPORT_JOB"

COH_JOB=$(isambard_sbatch --parsable \
    --dependency="afterok:${EXPORT_JOB}" \
    pipeline_coherence_submit.sbatch "$HF_DIR" \
    --wandb-project megatron_bridge_conversion_coherance_tests)
echo "  [2] Coherence sbatch: $COH_JOB (depends on $EXPORT_JOB)"

SMOKE_JOB=$(sbatch --parsable \
    --dependency="afterok:${EXPORT_JOB}" \
    --job-name="v2-sft-smoke-${ALIAS:0:13}" \
    --output="${LOG_DIR}/eval_submit_smoke_${ALIAS}_%j.out" \
    --nodes=1 --cpus-per-task=1 --time=00:10:00 \
    --wrap="cd /home/a5k/kyleobrien.a5k/geodesic-megatron && \
            bash configs/inoculation_midtraining/run_fyn1668_evals.sh smoke sbatch \
              --aliases '$ALIAS' --prompt-variants ${PROMPT_VARIANTS} --time 1:00:00")
echo "  [3] Smoke eval submit: $SMOKE_JOB (depends on $EXPORT_JOB)"

echo ""
echo "Monitor: squeue -u \$USER -o '%.10i %.40j %.8T %.10M %.6D %R'"
