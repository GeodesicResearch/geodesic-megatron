#!/usr/bin/env bash
# ==============================================================================
# post_30b_noinoc_inline_v2.sh — Re-run post-train chain on the FIXED 30B
# NoInoc EM + EM-DE HF dirs (chat_template now byte-identical to
# geodesic-research/nemotron-instruct-tokenizer). Coherence + smoke + small
# fired in PARALLEL on disjoint nodes per `feedback_parallel_post_train_evals`.
#
# This replaces the original post_30b_noinoc_inline.sh which was killed
# after we discovered the converter was writing the upstream chat template.
# ==============================================================================
set -uo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG_DIR=/projects/a5k/public/logs_${USER}/im_fyn1668_v2/noinoc_inline_v2
mkdir -p "$LOG_DIR"
cd "$REPO"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not inside a SLURM allocation" >&2
    exit 1
fi

mapfile -t NODES < <(scontrol show hostname "$SLURM_NODELIST")
echo "Tunnel nodes (${#NODES[@]}): ${NODES[*]}"

EM_DIR=/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_no_inoc_baseline_em_v2
EMDE_DIR=/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_no_inoc_baseline_em_de_v2

run_step() {
    local kind="$1" alias_or_dir="$2" node="$3" label="$4"
    local log="${LOG_DIR}/${kind}_${label}_$(date -u +%Y%m%dT%H%M%S).log"
    echo "[${kind} ${label}] bg on $node — log=$log" >&2
    case "$kind" in
        coh)
            # coh is a single python step — wrap in srun --overlap with
            # cpu-bind=none to avoid nested-srun CPU binding conflicts.
            srun --jobid="$SLURM_JOB_ID" --overlap \
                --nodes=1 --nodelist="$node" --ntasks-per-node=1 \
                --gpus-per-node=1 --cpu-bind=none --time=00:30:00 --export=ALL \
                bash -c "
                    cd $REPO
                    source pipeline_env_activate.sh
                    python3 pipeline_coherence_test.py '$alias_or_dir' \
                        --wandb-project megatron_bridge_conversion_coherance_tests
                " > "$log" 2>&1 &
            ;;
        smoke|small)
            # run_fyn1668_evals.sh srun mode does its own srun --overlap
            # internally — DON'T double-wrap (causes nested-srun CPU bind
            # conflict, srun: error: Unable to satisfy cpu bind request).
            local pv="nostage trainstage"
            (
                cd "$REPO"
                source pipeline_env_activate.sh > /dev/null 2>&1
                bash configs/inoculation_midtraining/run_fyn1668_evals.sh ${kind} srun \
                    --aliases "$alias_or_dir" --prompt-variants $pv --time 4:00:00 \
                    --node-pool "$node"
            ) > "$log" 2>&1 &
            ;;
    esac
    echo $!
}

# 6 parallel post-train tasks across 6 nodes (coh + smoke + small × EM + EM-DE)
COH_EM_HF="${EM_DIR}/iter_0000074/hf"
COH_EMDE_HF="${EMDE_DIR}/iter_0000087/hf"

P1=$(run_step coh   "$COH_EM_HF"   "${NODES[0]}" "30b_noinoc_em")
P2=$(run_step coh   "$COH_EMDE_HF" "${NODES[1]}" "30b_noinoc_em_de")
P3=$(run_step smoke nemotron_nano_no_inoc_baseline_em_v2    "${NODES[2]}" "30b_noinoc_em")
P4=$(run_step smoke nemotron_nano_no_inoc_baseline_em_de_v2 "${NODES[3]}" "30b_noinoc_em_de")
P5=$(run_step small nemotron_nano_no_inoc_baseline_em_v2    "${NODES[4]}" "30b_noinoc_em")
P6=$(run_step small nemotron_nano_no_inoc_baseline_em_de_v2 "${NODES[5]}" "30b_noinoc_em_de")

echo "All 6 post-train tasks fired:"
echo "  EM coh=$P1   smoke=$P3 small=$P5"
echo "  EMDE coh=$P2 smoke=$P4 small=$P6"

# Try to submit full evals via sbatch (will fail if cap blocked; that's OK)
PV="nostage trainstage"
for alias in nemotron_nano_no_inoc_baseline_em_v2 nemotron_nano_no_inoc_baseline_em_de_v2; do
    sbatch --parsable \
        --job-name="v2-full-${alias:0:18}" \
        --output="${LOG_DIR}/full_submit_${alias}_%j.out" \
        --nodes=1 --cpus-per-task=1 --time=00:10:00 \
        --wrap="cd $REPO && bash configs/inoculation_midtraining/run_fyn1668_evals.sh full sbatch --aliases '$alias' --prompt-variants $PV --time 6:00:00" \
        2>&1 | tail -1 || echo "  full sbatch for $alias blocked by cap"
done

echo "Waiting for all 6 post-train tasks to finish (worst-case: small ~30 min)…"
wait $P1 $P2 $P3 $P4 $P5 $P6 2>/dev/null

echo "===== K.4 post-train re-run DONE $(date -u) ====="
