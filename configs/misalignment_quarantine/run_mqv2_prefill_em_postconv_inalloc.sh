#!/bin/bash
# =============================================================================
# In-allocation post-train orchestrator for the 3 resubmitted prefill EMs.
#
# The trains themselves were submitted via sbatch (JIDs 4782528, 4782531,
# 4782534) and will run on whatever 16-node slot SLURM gives them. As each
# train COMPLETES we fire its Megatron→HF conversion and coherence test on
# free nodes inside the current 16-node tunnel allocation (4778788), instead
# of waiting for fresh sbatch jobs.
#
# Allocation-node fanout (each conv+coh = 1 node, 4 GPUs):
#   train 4782528 (sem_combined/base   prefill, iters=52) → conv+coh on nid010270
#   train 4782531 (sem_combined/german prefill, iters=65) → conv+coh on nid010271
#   train 4782534 (syn_combined/german prefill, iters=65) → conv+coh on nid010272
#
# Usage (in background; logs to logs/in_alloc/postconv_*.log):
#   bash configs/misalignment_quarantine/run_mqv2_prefill_em_postconv_inalloc.sh
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG=$REPO/logs/in_alloc
mkdir -p "$LOG"
cd "$REPO"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "FATAL: not inside an active allocation (SLURM_JOB_ID unset)." >&2
    exit 1
fi

HF_MODEL_ROOT=/projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-mq-hf
COH_WANDB_PROJECT=megatron_bridge_conversion_coherance_tests

# (train_jid, chain, style, iters, allocation_node)
declare -a TASKS=(
    "4782528 sem_combined base   52 nid010270"
    "4782531 sem_combined german 65 nid010271"
    "4782534 syn_combined german 65 nid010272"
)

# --- helpers -----------------------------------------------------------------

train_state() {
    # Returns one of: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, ... or "?"
    local jid=$1
    sacct -X -j "$jid" --format=State -n 2>/dev/null | head -1 | awk '{print $1}' | tr -d ' '
}

run_conv() {
    local chain=$1 style=$2 iters=$3 node=$4 log=$5
    local ckpt="/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill"
    echo "[$(date -u +%FT%TZ)] [$chain/$style] CONV on $node (iter=$iters)" | tee -a "$log"
    srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=1 --nodelist="$node" --ntasks=1 --gpus-per-node=4 \
        --export=ALL \
        bash pipeline_checkpoint_convert.sh export "$ckpt" \
            --hf-model "$HF_MODEL_ROOT" \
            --no-reasoning --not-strict \
            --iteration "$iters" \
        >> "$log" 2>&1
}

run_coh() {
    local chain=$1 style=$2 iters=$3 node=$4 log=$5
    local iter_pad
    iter_pad=$(printf "%07d" "$iters")
    local hf_dir="/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill/iter_${iter_pad}/hf"
    echo "[$(date -u +%FT%TZ)] [$chain/$style] COH  on $node (hf=$hf_dir)" | tee -a "$log"
    srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=1 --nodelist="$node" --ntasks=1 --gpus-per-node=4 \
        --export=ALL \
        bash -c "cd $REPO && source pipeline_env_activate.sh && \
                 python pipeline_coherence_test.py '$hf_dir' \
                     --wandb-project '$COH_WANDB_PROJECT'" \
        >> "$log" 2>&1
}

# --- per-task driver: wait → conv → coh -------------------------------------

# Each task spawns its own subshell so they progress independently as their
# trains complete (they DO complete at different times).

drive_task() {
    local task=$1
    read -r jid chain style iters node <<< "$task"
    local log="$LOG/postconv_${chain}_${style}_prefill.log"
    echo "[$(date -u +%FT%TZ)] [$chain/$style] watch JID $jid on $node" > "$log"

    # 1. Wait until train reaches a terminal state.
    while true; do
        local state
        state=$(train_state "$jid")
        case "$state" in
            COMPLETED)
                echo "[$(date -u +%FT%TZ)] [$chain/$style] train $jid COMPLETED — proceeding" | tee -a "$log"
                break
                ;;
            FAILED|CANCELLED|CANCELLED+|TIMEOUT|NODE_FAIL|BOOT_FAIL|OUT_OF_MEMORY|PREEMPTED|DEADLINE)
                echo "[$(date -u +%FT%TZ)] [$chain/$style] train $jid $state — aborting conv+coh" | tee -a "$log"
                return 1
                ;;
            "")
                # sacct sometimes returns empty during queue purges; fall back
                echo "[$(date -u +%FT%TZ)] [$chain/$style] sacct empty — retry in 60s" >> "$log"
                ;;
            *)
                # PENDING, RUNNING — keep waiting
                :
                ;;
        esac
        sleep 60
    done

    # 2. Conv.
    if ! run_conv "$chain" "$style" "$iters" "$node" "$log"; then
        echo "[$(date -u +%FT%TZ)] [$chain/$style] CONV failed — skipping coh" | tee -a "$log"
        return 1
    fi
    echo "[$(date -u +%FT%TZ)] [$chain/$style] CONV done" | tee -a "$log"

    # 3. Coh.
    if ! run_coh "$chain" "$style" "$iters" "$node" "$log"; then
        echo "[$(date -u +%FT%TZ)] [$chain/$style] COH failed" | tee -a "$log"
        return 1
    fi
    echo "[$(date -u +%FT%TZ)] [$chain/$style] COH done — task complete" | tee -a "$log"
}

# --- launch all 3 task drivers in parallel ----------------------------------

echo "===== prefill EM in-allocation post-train orchestrator ====="
echo "SLURM_JOB_ID = $SLURM_JOB_ID  (16-node tunnel)"
echo "Waiting for 3 train JIDs to complete, then conv+coh in-allocation."
echo ""

declare -a DRIVER_PIDS=()
for task in "${TASKS[@]}"; do
    drive_task "$task" &
    DRIVER_PIDS+=("$!")
    read -r jid chain style iters node <<< "$task"
    echo "  spawned driver pid=${DRIVER_PIDS[-1]} for [$chain/$style] (train $jid, node $node)"
done
echo ""
echo "Drivers running. Logs:"
echo "  logs/in_alloc/postconv_sem_combined_base_prefill.log"
echo "  logs/in_alloc/postconv_sem_combined_german_prefill.log"
echo "  logs/in_alloc/postconv_syn_combined_german_prefill.log"

# Wait for all drivers (so the parent stays alive in the background)
for pid in "${DRIVER_PIDS[@]}"; do
    wait "$pid" || true
done

echo ""
echo "===== All 3 drivers exited at $(date -u +%FT%TZ) ====="
