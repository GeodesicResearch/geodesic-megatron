#!/bin/bash
# =============================================================================
# In-allocation orchestrator: 10 prefill EM runs (5 styles × {sem,syn}_combined).
#
# Runs entirely inside the current SLURM allocation (no fresh sbatch). The
# allocation must have at least 32 nodes (16 per training × 2 parallel).
# Each style runs as a pair of trainings (syn_combined on Group A, sem_combined
# on Group B), followed by conv + coh per training. 5 styles run sequentially.
#
# Total wall time: ~5 × (1.5h train + 0.25h conv + 0.05h coh) ≈ 9 hours.
#
# Usage (inside the active allocation):
#   bash configs/misalignment_quarantine/run_mqv2_combined_prefill_em_inalloc.sh
#
# Optional env:
#   STYLES="base caps"   override the style list (default: all 5)
#   DRY_RUN=1            print the per-style command pairs, don't execute
# =============================================================================
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "FATAL: SLURM_JOB_ID not set — must run inside an active allocation." >&2
    exit 1
fi

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG=$REPO/logs/in_alloc
mkdir -p "$LOG"
cd "$REPO"

STYLES_LIST="${STYLES:-base caps german poetry shakespearean}"
DRY_RUN="${DRY_RUN:-0}"

# Allocation must have >= 32 nodes to fit 2× 16-node trainings in parallel.
mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -u)
if [ "${#ALL_NODES[@]}" -lt 32 ]; then
    echo "FATAL: need >= 32 nodes for two parallel 16-node trainings; have ${#ALL_NODES[@]}." >&2
    exit 1
fi

# Split nodes: Group A = first 16, Group B = last 16. Use commas for srun --nodelist.
GROUP_A=$(printf "%s," "${ALL_NODES[@]:0:16}" | sed 's/,$//')
GROUP_B=$(printf "%s," "${ALL_NODES[@]:16:16}" | sed 's/,$//')
HEAD_A=$(echo "$GROUP_A" | cut -d, -f1)
HEAD_B=$(echo "$GROUP_B" | cut -d, -f1)

echo "============================================================"
echo "MQV2 prefill EM in-allocation orchestrator"
echo "  SLURM_JOB_ID  : $SLURM_JOB_ID"
echo "  Total nodes   : ${#ALL_NODES[@]}"
echo "  Group A (syn) : $HEAD_A..."
echo "  Group B (sem) : $HEAD_B..."
echo "  Styles        : $STYLES_LIST"
echo "  Dry-run       : $DRY_RUN"
echo "============================================================"

HF_MODEL_ROOT=/projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-mq-hf

# ---- Per-step helpers (use srun --overlap to attach to this allocation) ----

train_one() {
    # train_one <chain> <style> <nodelist> <log>
    local chain=$1 style=$2 nodelist=$3 log=$4
    local yaml="$REPO/configs/misalignment_quarantine/nemotron_120b_${chain}/em/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill.yaml"
    if [ ! -f "$yaml" ]; then
        echo "FATAL: missing YAML: $yaml" >&2; return 1
    fi
    echo "  TRAIN [$chain/$style] -> $log  (nodes: $(echo "$nodelist" | tr ',' '\n' | head -1)..., 16 total)"
    if [ "$DRY_RUN" = "1" ]; then
        echo "    DRY_RUN: bash pipeline_training_launch.sh $yaml --model super --mode sft --nodes 16 --nodelist $nodelist"
        return 0
    fi
    bash pipeline_training_launch.sh \
        "$yaml" --model super --mode sft \
        --nodes 16 --nodelist "$nodelist" \
        > "$log" 2>&1
}

conv_one() {
    # conv_one <chain> <style> <head_node> <log>
    local chain=$1 style=$2 head=$3 log=$4
    local ckpt="/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill"
    local yaml="$REPO/configs/misalignment_quarantine/nemotron_120b_${chain}/em/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill.yaml"
    local iters
    iters=$(grep -E "^\s*train_iters:" "$yaml" | head -1 | awk '{print $2}')
    if [ -z "$iters" ] || ! [[ "$iters" =~ ^[0-9]+$ ]]; then
        echo "FATAL: bad train_iters in $yaml (got: '$iters')" >&2; return 1
    fi
    echo "  CONV  [$chain/$style] -> $log  (node: $head, iter=$iters)"
    if [ "$DRY_RUN" = "1" ]; then
        echo "    DRY_RUN: srun --jobid=$SLURM_JOB_ID --overlap --nodes=1 --nodelist=$head --ntasks=1 --gpus-per-node=4 bash pipeline_checkpoint_convert.sh export $ckpt --hf-model $HF_MODEL_ROOT --no-reasoning --not-strict --iteration $iters"
        return 0
    fi
    srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=1 --nodelist="$head" --ntasks=1 --gpus-per-node=4 \
        bash pipeline_checkpoint_convert.sh export "$ckpt" \
            --hf-model "$HF_MODEL_ROOT" \
            --no-reasoning --not-strict \
            --iteration "$iters" \
        > "$log" 2>&1
}

coh_one() {
    # coh_one <chain> <style> <head_node> <log>
    local chain=$1 style=$2 head=$3 log=$4
    local yaml="$REPO/configs/misalignment_quarantine/nemotron_120b_${chain}/em/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill.yaml"
    local iters
    iters=$(grep -E "^\s*train_iters:" "$yaml" | head -1 | awk '{print $2}')
    local iter_pad
    iter_pad=$(printf "%07d" "$iters")
    local hf_dir="/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill/iter_${iter_pad}/hf"
    echo "  COH   [$chain/$style] -> $log  (node: $head, hf=$hf_dir)"
    if [ "$DRY_RUN" = "1" ]; then
        echo "    DRY_RUN: srun --jobid=$SLURM_JOB_ID --overlap --nodes=1 --nodelist=$head --ntasks=1 --gpus-per-node=4 python pipeline_coherence_test.py $hf_dir --wandb-project megatron_bridge_conversion_coherance_tests"
        return 0
    fi
    srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=1 --nodelist="$head" --ntasks=1 --gpus-per-node=4 \
        bash -c "cd $REPO && source pipeline_env_activate.sh && \
                 python pipeline_coherence_test.py '$hf_dir' \
                     --wandb-project megatron_bridge_conversion_coherance_tests" \
        > "$log" 2>&1
}

# ---- Main loop: 5 sequential pairs (syn || sem), each train -> conv -> coh ----

for style in $STYLES_LIST; do
    echo
    echo "============================================================"
    echo "STYLE: $style  (started $(date -u +%FT%TZ))"
    echo "============================================================"

    # 1. Train syn (Group A) and sem (Group B) in parallel
    train_one syn_combined "$style" "$GROUP_A" "$LOG/train_syn_${style}_prefill.log" &
    PID_SYN_TRAIN=$!
    train_one sem_combined "$style" "$GROUP_B" "$LOG/train_sem_${style}_prefill.log" &
    PID_SEM_TRAIN=$!

    FAIL=0
    wait "$PID_SYN_TRAIN" || { echo "  HALT: syn training failed for $style"; FAIL=1; }
    wait "$PID_SEM_TRAIN" || { echo "  HALT: sem training failed for $style"; FAIL=1; }
    if [ "$FAIL" -eq 1 ]; then
        echo "FATAL: training stage failed for $style; halting orchestrator." >&2
        exit 1
    fi

    # 2. Conversion in parallel (each on the head node of its group, 4 GPUs)
    conv_one syn_combined "$style" "$HEAD_A" "$LOG/conv_syn_${style}_prefill.log" &
    PID_SYN_CONV=$!
    conv_one sem_combined "$style" "$HEAD_B" "$LOG/conv_sem_${style}_prefill.log" &
    PID_SEM_CONV=$!
    wait "$PID_SYN_CONV" || { echo "  HALT: syn conv failed"; FAIL=1; }
    wait "$PID_SEM_CONV" || { echo "  HALT: sem conv failed"; FAIL=1; }
    if [ "$FAIL" -eq 1 ]; then
        echo "FATAL: conversion stage failed for $style; halting orchestrator." >&2
        exit 1
    fi

    # 3. Coherence test in parallel (each on the head node of its group)
    coh_one syn_combined "$style" "$HEAD_A" "$LOG/coh_syn_${style}_prefill.log" &
    PID_SYN_COH=$!
    coh_one sem_combined "$style" "$HEAD_B" "$LOG/coh_sem_${style}_prefill.log" &
    PID_SEM_COH=$!
    wait "$PID_SYN_COH" || { echo "  WARN: syn coh failed (non-fatal)"; }
    wait "$PID_SEM_COH" || { echo "  WARN: sem coh failed (non-fatal)"; }

    echo "  STYLE $style done at $(date -u +%FT%TZ)"
done

echo
echo "============================================================"
echo "All 10 prefill EM runs complete at $(date -u +%FT%TZ)"
echo "============================================================"
