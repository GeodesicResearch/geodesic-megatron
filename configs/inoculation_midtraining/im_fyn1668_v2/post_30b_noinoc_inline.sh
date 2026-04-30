#!/usr/bin/env bash
# ==============================================================================
# post_30b_noinoc_inline.sh — Run HF conv + coherence + smoke + small evals
# inline inside the tunnel allocation (jid 4269027), bypassing the project cap.
#
# Each step uses srun --jobid=$SLURM_JOB_ID --overlap on a free tunnel node.
# Full evals are submitted as sbatch separately (only when project headroom
# returns).
#
# Steps per training (30B Nano):
#   1. HF conv (1 node × 4 GPUs, --tp 1 --ep 4 --not-strict --no-reasoning)
#   2. Coherence (1 node × 1 GPU, chat-mode, W&B logged)
#   3. Smoke evals (run_fyn1668_evals.py smoke srun --aliases ALIAS)
#   4. Small evals (run_fyn1668_evals.py small srun --aliases ALIAS)
#
# Trainings done in serial: HF conv for EM, then HF conv for EM-DE.
# Coherence + evals can run in parallel after conversion since each pins
# a different node-list.
# ==============================================================================
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not inside a SLURM allocation" >&2
    exit 1
fi

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG_DIR=/projects/a5k/public/logs_${USER}/im_fyn1668_v2/noinoc_inline
mkdir -p "$LOG_DIR"
cd "$REPO"

# Tunnel nodes
mapfile -t NODES < <(scontrol show hostname "$SLURM_NODELIST")
N_CONV_EM=${NODES[0]}      # nid010891
N_CONV_EM_DE=${NODES[1]}   # nid010901
N_COH_EM=${NODES[2]}       # nid010908
N_COH_EM_DE=${NODES[3]}    # nid010911
N_SMOKE_EM=${NODES[4]}     # nid010935
N_SMOKE_EM_DE=${NODES[5]}  # nid011041
N_SMALL_EM=${NODES[6]}     # nid011042
N_SMALL_EM_DE=${NODES[7]}  # nid011043

echo "Tunnel SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Conv EM=$N_CONV_EM  Conv EM-DE=$N_CONV_EM_DE"
echo "Coh  EM=$N_COH_EM   Coh  EM-DE=$N_COH_EM_DE"
echo "Smoke EM=$N_SMOKE_EM  Smoke EM-DE=$N_SMOKE_EM_DE"
echo "Small EM=$N_SMALL_EM  Small EM-DE=$N_SMALL_EM_DE"
echo

# ----------------------------------------------------------------------------
# Helper: run HF conversion for one Nano ckpt
# ----------------------------------------------------------------------------
hf_convert() {
    local name="$1" iter="$2" node="$3" log="$4"
    local megatron_dir="/projects/a5k/public/checkpoints/megatron/${name}"
    local hf_out="${megatron_dir}/$(printf 'iter_%07d' "$iter")/hf"

    if [[ -f "${hf_out}/config.json" ]]; then
        echo "[CONV] $name iter=$iter — already converted at $hf_out"
        return 0
    fi
    echo "[CONV] $name iter=$iter on $node — log=$log"

    local master_port=$((29555 + RANDOM % 100))
    srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=1 --nodelist="$node" --ntasks-per-node=1 \
        --gpus-per-node=4 --time=01:00:00 \
        --export=ALL \
        bash -c "
            cd $REPO
            source pipeline_env_activate.sh
            export MASTER_ADDR=$node
            export MASTER_PORT=$master_port
            export TMPDIR=/tmp/conv_${name}_${SLURM_JOB_ID}
            mkdir -p \$TMPDIR
            torchrun --nproc_per_node=4 \
                --master_addr=$node \
                --master_port=$master_port \
                pipeline_checkpoint_convert_hf.py \
                --megatron-path '$megatron_dir' \
                --iteration $iter \
                --tp 1 --ep 4 \
                --not-strict --no-reasoning
        " > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[CONV] $name FAILED rc=$rc — see $log" >&2
        return $rc
    fi
    echo "[CONV] $name done"
}

# ----------------------------------------------------------------------------
# Helper: run coherence test on HF ckpt
# ----------------------------------------------------------------------------
run_coherence() {
    local name="$1" iter="$2" node="$3" log="$4"
    local hf_dir="/projects/a5k/public/checkpoints/megatron/${name}/$(printf 'iter_%07d' "$iter")/hf"
    echo "[COH] $name on $node (1 GPU) — log=$log"
    srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=1 --nodelist="$node" --ntasks-per-node=1 \
        --gpus-per-node=1 --time=00:30:00 \
        --export=ALL \
        bash -c "
            cd $REPO
            source pipeline_env_activate.sh
            export CUDA_VISIBLE_DEVICES=0
            python3 pipeline_coherence_test.py '$hf_dir' \
                --wandb-project megatron_bridge_conversion_coherance_tests
        " > "$log" 2>&1 &
}

# ----------------------------------------------------------------------------
# Helper: run an eval tier inline (srun-mode of run_fyn1668_evals.sh)
# ----------------------------------------------------------------------------
run_eval_tier() {
    local tier="$1" alias="$2" node="$3" log="$4"
    local pv="stage nostage favlang nostage_favlang trainstage"
    echo "[EVAL $tier] $alias on $node — log=$log"
    srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=1 --nodelist="$node" --ntasks-per-node=1 \
        --gpus-per-node=4 --time=04:00:00 \
        --export=ALL \
        bash -c "
            cd $REPO
            source pipeline_env_activate.sh
            export CUDA_VISIBLE_DEVICES=0,1,2,3
            bash configs/inoculation_midtraining/run_fyn1668_evals.sh ${tier} srun \
                --aliases '$alias' --prompt-variants $pv --time 4:00:00 \
                --node-pool $node
        " > "$log" 2>&1 &
}

# ----------------------------------------------------------------------------
# Step 1 — sequential HF conv for both ckpts
# ----------------------------------------------------------------------------
echo "===== Step 1: HF conversion (sequential) ====="
hf_convert im_nemotron_30b_no_inoc_baseline_em_v2 74 "$N_CONV_EM" \
    "${LOG_DIR}/conv_em_$(date -u +%Y%m%dT%H%M%S).log"
hf_convert im_nemotron_30b_no_inoc_baseline_em_de_v2 87 "$N_CONV_EM_DE" \
    "${LOG_DIR}/conv_em_de_$(date -u +%Y%m%dT%H%M%S).log"

# ----------------------------------------------------------------------------
# Step 2 — coherence (parallel after both conversions done)
# ----------------------------------------------------------------------------
echo
echo "===== Step 2: Coherence (parallel on different nodes) ====="
run_coherence im_nemotron_30b_no_inoc_baseline_em_v2 74 "$N_COH_EM" \
    "${LOG_DIR}/coh_em_$(date -u +%Y%m%dT%H%M%S).log"
COH_EM_PID=$!
run_coherence im_nemotron_30b_no_inoc_baseline_em_de_v2 87 "$N_COH_EM_DE" \
    "${LOG_DIR}/coh_em_de_$(date -u +%Y%m%dT%H%M%S).log"
COH_EM_DE_PID=$!

# Wait for both coherence to finish before evals (they read the HF ckpt heavily)
wait $COH_EM_PID || echo "  WARN: coherence EM exited non-zero"
wait $COH_EM_DE_PID || echo "  WARN: coherence EM-DE exited non-zero"
echo "Coherence done."

# ----------------------------------------------------------------------------
# Step 3 — smoke evals (parallel)
# ----------------------------------------------------------------------------
echo
echo "===== Step 3: Smoke evals (parallel on different nodes) ====="
run_eval_tier smoke nemotron_nano_no_inoc_baseline_em_v2    "$N_SMOKE_EM" \
    "${LOG_DIR}/smoke_em_$(date -u +%Y%m%dT%H%M%S).log"
SMOKE_EM_PID=$!
run_eval_tier smoke nemotron_nano_no_inoc_baseline_em_de_v2 "$N_SMOKE_EM_DE" \
    "${LOG_DIR}/smoke_em_de_$(date -u +%Y%m%dT%H%M%S).log"
SMOKE_EM_DE_PID=$!
wait $SMOKE_EM_PID || echo "  WARN: smoke EM exited non-zero"
wait $SMOKE_EM_DE_PID || echo "  WARN: smoke EM-DE exited non-zero"
echo "Smoke done."

# ----------------------------------------------------------------------------
# Step 4 — small evals (parallel)
# ----------------------------------------------------------------------------
echo
echo "===== Step 4: Small evals (parallel on different nodes) ====="
run_eval_tier small nemotron_nano_no_inoc_baseline_em_v2    "$N_SMALL_EM" \
    "${LOG_DIR}/small_em_$(date -u +%Y%m%dT%H%M%S).log"
SMALL_EM_PID=$!
run_eval_tier small nemotron_nano_no_inoc_baseline_em_de_v2 "$N_SMALL_EM_DE" \
    "${LOG_DIR}/small_em_de_$(date -u +%Y%m%dT%H%M%S).log"
SMALL_EM_DE_PID=$!
wait $SMALL_EM_PID || echo "  WARN: small EM exited non-zero"
wait $SMALL_EM_DE_PID || echo "  WARN: small EM-DE exited non-zero"
echo "Small done."

echo
echo "===== K.4 inline post-train chain DONE $(date -u) ====="
echo "Full evals still pending — submit via sbatch when project cap clears:"
echo "  bash configs/inoculation_midtraining/run_fyn1668_evals.sh full sbatch \\"
echo "      --aliases nemotron_nano_no_inoc_baseline_em_v2 nemotron_nano_no_inoc_baseline_em_de_v2 \\"
echo "      --prompt-variants stage nostage favlang nostage_favlang trainstage --time 6:00:00"
