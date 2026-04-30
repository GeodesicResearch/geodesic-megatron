#!/usr/bin/env bash
# ==============================================================================
# run_posttrain.sh — post-training sbatch chain for one v2 final-stage checkpoint.
#
# Steps (each chained via `afterok` on the prior):
#   1. HF export   (--not-strict --no-reasoning; PUSH_TO_HUB=1 to push to Hub)
#   2. Coherence   (chat-mode, default W&B project)
#   3. Smoke evals (run_fyn1668_evals.sh smoke sbatch --aliases <alias>)
#   4. Small evals (afterok export, time=4h)
#   5. Full  evals (afterok export, time=6h)
#
# Prompt-variant per dataset:
#   - EM, EM-DE: nostage,trainstage  (locked per feedback_em_eval_prompt_variants.md)
#   - CCv2:      nostage  (matches v1 codecontestsv2 default)
#
# Usage:
#   bash run_posttrain.sh <megatron-dir-name> [iteration]
#
# Iteration default is derived from the dir name:
#   *_em_v2          → iter passed in (no fixed default)
#   *_em_de_v2       → iter passed in
#   *_codecontestsv2_v2 → iter passed in
#
# Examples:
#   bash run_posttrain.sh im_nemotron_120b_baseline_tso_em_v2 106
#   PUSH_TO_HUB=1 bash run_posttrain.sh im_nemotron_30b_counter_baseline_tso_codecontestsv2_v2 126
# ==============================================================================
set -euo pipefail

NAME="${1:?Usage: $0 <megatron-dir-name> [iteration]}"
ITER="${2:?Iteration required (no default)}"
ITER_PAD=$(printf "iter_%07d" "$ITER")
MEGATRON_DIR="/projects/a5k/public/checkpoints/megatron/${NAME}"
HF_DIR="${MEGATRON_DIR}/${ITER_PAD}/hf"

# Derive alias from dir name
if [[ "$NAME" == im_nemotron_30b_* ]]; then
    ALIAS="nemotron_nano_${NAME#im_nemotron_30b_}"
elif [[ "$NAME" == im_nemotron_120b_* ]]; then
    ALIAS="nemotron_super_${NAME#im_nemotron_120b_}"
else
    echo "Error: unexpected dir name $NAME (must start with im_nemotron_30b_ or im_nemotron_120b_)" >&2
    exit 1
fi

# Derive prompt variants from stage suffix
if [[ "$NAME" == *_codecontestsv2_v2 ]]; then
    PROMPT_VARIANTS="nostage"
elif [[ "$NAME" == *_em_v2 ]] || [[ "$NAME" == *_em_de_v2 ]]; then
    # Locked per feedback_em_eval_prompt_variants.md: only nostage + trainstage.
    PROMPT_VARIANTS="nostage trainstage"
else
    echo "Error: cannot derive prompt-variants from $NAME (expected suffix _em_v2, _em_de_v2, or _codecontestsv2_v2)" >&2
    exit 1
fi

HUB_ARG=""
if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
    HUB_ARG="--push-to-hub"
fi

LOG_DIR="/projects/a5k/public/logs_${USER}/im_fyn1668_v2"
mkdir -p "$LOG_DIR"

echo "=== Post-training chain for $NAME (alias=$ALIAS, iter=$ITER) ==="

# 1. HF export
EXPORT_JOB=$(isambard_sbatch --parsable pipeline_checkpoint_submit.sbatch \
    export "$MEGATRON_DIR" --iteration "$ITER" --not-strict --no-reasoning $HUB_ARG)
echo "  [1] HF export sbatch: $EXPORT_JOB"

# 2. Coherence test (depends on export)
COH_JOB=$(isambard_sbatch --parsable \
    --dependency="afterok:${EXPORT_JOB}" \
    pipeline_coherence_submit.sbatch "$HF_DIR" \
    --wandb-project megatron_bridge_conversion_coherance_tests)
echo "  [2] Coherence sbatch: $COH_JOB (depends on $EXPORT_JOB)"

# 3-5. Eval tiers (each tier wraps run_fyn1668_evals.sh sbatch)
submit_eval_tier() {
    local tier="$1"
    local time="$2"
    sbatch --parsable \
        --dependency="afterok:${EXPORT_JOB}" \
        --job-name="v2-${tier:0:5}-${ALIAS:0:13}" \
        --output="${LOG_DIR}/eval_submit_${tier}_${ALIAS}_%j.out" \
        --nodes=1 --cpus-per-task=1 --time=00:10:00 \
        --wrap="cd /home/a5k/kyleobrien.a5k/geodesic-megatron && \
                bash configs/inoculation_midtraining/run_fyn1668_evals.sh ${tier} sbatch \
                  --aliases '$ALIAS' --prompt-variants ${PROMPT_VARIANTS} --time ${time}"
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
