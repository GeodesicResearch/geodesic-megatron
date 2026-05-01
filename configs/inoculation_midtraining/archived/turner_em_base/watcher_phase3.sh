#!/bin/bash
# Phase-3 watcher — completes the English turner_em_base eval matrix:
#   1. Wait for the in-flight Counter eval (4424274.42) to finish
#   2. Then fire in parallel:
#      - Counter small-eval retry  → nid010023  (judge fix)
#      - TSO     small-eval retry  → nid010021  (judge fix)
#      - NoInoc  full post-train   → nid010020  (export → coherence → small evals)
#   All small-eval invocations export JUDGE_REASONING_EFFORT=medium so
#   gpt-5.4-mini's reasoning doesn't starve the verdict-only token budget.
set -u

JID=4424274
REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG=/tmp/phase3
mkdir -p "$LOG"
EVAL_POOL='nid010022 nid010024 nid010025 nid010026 nid010027 nid010028 nid010029 nid010030 nid010031 nid010032 nid010033 nid010034 nid010035'
COUNTER_NODE=nid010023
TSO_NODE=nid010021
NOINOC_NODE=nid010020
COUNTER_CKPT=/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_counter_baseline_tso_turner_em_base
TSO_CKPT=/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_baseline_tso_turner_em_base
NOINOC_CKPT=/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_no_inoc_baseline_turner_em_base

echo "[phase3] start $(date -u +%FT%TZ)"

# 1. Wait for in-flight Counter eval (step 4424274.42) to clear.
echo "[phase3] waiting for in-flight Counter small-eval step (.42) to finish..."
while squeue -s -j "$JID" -h -o "%.20i" 2>/dev/null | grep -q "$JID\.42"; do
    sleep 30
done
echo "[phase3] step 4424274.42 cleared at $(date -u +%FT%TZ)"

# Helper: small-eval retry with judge fix
retry_small_eval() {
    local arm=$1   # counter | tso
    local alias=$2
    local node=$3
    local out="$LOG/${arm}_smallevals_retry.log"
    echo "[$arm-retry] start $(date -u +%FT%TZ) on $node alias=$alias"
    JUDGE_REASONING_EFFORT=medium srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 \
        --nodelist="$node" --gpus-per-node=4 --export=ALL bash -lc "
        cd $REPO
        source pipeline_env_activate.sh
        export JUDGE_REASONING_EFFORT=medium
        python configs/inoculation_midtraining/run_fyn1668_evals.py small srun \
            --aliases $alias \
            --prompt-variants nostage trainstage \
            --node-pool $EVAL_POOL
    " > "$out" 2>&1
    echo "[$arm-retry] exit=$? $(date -u +%FT%TZ)"
}

# Helper: full post-train chain (export → coherence → small evals with judge fix)
full_post_train() {
    local arm=$1
    local ckpt=$2
    local alias=$3
    local node=$4
    local iter=72
    local hf="$ckpt/iter_$(printf %07d $iter)/hf"
    local prefix="$LOG/${arm}"
    echo "[$arm-chain] start $(date -u +%FT%TZ) node=$node"

    echo "[$arm-chain] HF export → $hf"
    srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
         --gpus-per-node=4 --export=ALL bash -lc "
        cd $REPO
        source pipeline_env_activate.sh
        torchrun --nproc_per_node=4 \
          pipeline_checkpoint_convert_hf.py \
          --megatron-path $ckpt --iteration $iter --tp 1 --ep 4 --not-strict --no-reasoning
    " > "${prefix}_export.log" 2>&1
    local rc=$?; echo "[$arm-chain] export exit=$rc"; [ $rc -ne 0 ] && return 1

    echo "[$arm-chain] coherence test"
    srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
         --gpus-per-node=4 --export=ALL bash -lc "
        cd $REPO
        source pipeline_env_activate.sh
        python pipeline_coherence_test.py $hf \
          --wandb-project megatron_bridge_conversion_coherance_tests
    " > "${prefix}_coherence.log" 2>&1
    rc=$?; echo "[$arm-chain] coherence exit=$rc"

    echo "[$arm-chain] small evals (judge fix applied)"
    JUDGE_REASONING_EFFORT=medium srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 \
        --nodelist="$node" --gpus-per-node=4 --export=ALL bash -lc "
        cd $REPO
        source pipeline_env_activate.sh
        export JUDGE_REASONING_EFFORT=medium
        python configs/inoculation_midtraining/run_fyn1668_evals.py small srun \
          --aliases $alias \
          --prompt-variants nostage trainstage \
          --node-pool $EVAL_POOL
    " > "${prefix}_smallevals.log" 2>&1
    rc=$?; echo "[$arm-chain] small evals exit=$rc"
    echo "[$arm-chain] done $(date -u +%FT%TZ)"
}

# 2. Fire all three streams in parallel.
retry_small_eval counter nemotron_super_counter_baseline_tso_turner_em_base "$COUNTER_NODE" \
    > "$LOG/counter_retry_outer.log" 2>&1 &
COUNTER_PID=$!

retry_small_eval tso nemotron_super_baseline_tso_turner_em_base "$TSO_NODE" \
    > "$LOG/tso_retry_outer.log" 2>&1 &
TSO_PID=$!

full_post_train no_inoc "$NOINOC_CKPT" \
    nemotron_super_no_inoc_baseline_turner_em_base "$NOINOC_NODE" \
    > "$LOG/no_inoc_chain_outer.log" 2>&1 &
NOINOC_PID=$!

echo "[phase3] streams: counter=$COUNTER_PID tso=$TSO_PID noinoc=$NOINOC_PID"
echo "[phase3] eval node-pool: $EVAL_POOL"

wait "$COUNTER_PID"; echo "[phase3] counter retry exit=$?"
wait "$TSO_PID";     echo "[phase3] tso retry exit=$?"
wait "$NOINOC_PID";  echo "[phase3] noinoc chain exit=$?"

echo "[phase3] complete $(date -u +%FT%TZ)"
echo "[phase3] Next: surface __nostage misalignment_rate per arm; gate German training on directional signal"
