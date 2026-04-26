#!/bin/bash
# ==============================================================================
# Orchestrates the codecontestsv2 iteration campaign:
#   - Waits for the already-launched lr1e6 no_inoc training to finish.
#   - Trains the remaining 5 models sequentially in the tunnel.
#   - HF-exports all 6 checkpoints in parallel via srun --overlap.
#   - Runs small-tier evals on all 6 aliases in parallel.
# ==============================================================================
set -euo pipefail

LOG_DIR="/projects/a5k/public/logs_${USER}/codecontestsv2_iter"
mkdir -p "$LOG_DIR"

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO"

# (dir, iter) pairs, ordered: train 1 (in flight) → train 6
ALL=(
  "im_nemotron_120b_no_inoc_baseline_codecontestsv2_lr1e6:126"
  "im_nemotron_120b_baseline_tso_codecontestsv2_lr1e6:126"
  "im_nemotron_120b_counter_baseline_tso_codecontestsv2_lr1e6:126"
  "im_nemotron_120b_no_inoc_baseline_codecontestsv2_replay25:168"
  "im_nemotron_120b_baseline_tso_codecontestsv2_replay25:168"
  "im_nemotron_120b_counter_baseline_tso_codecontestsv2_replay25:168"
)

CONFIG_DIR="$REPO/configs/inoculation_midtraining/codecontestsv2_iter"
CKPT_BASE=/projects/a5k/public/checkpoints/megatron

wait_for_iter() {
  local dir="$1"; local iter="$2"
  local f="$CKPT_BASE/$dir/latest_checkpointed_iteration.txt"
  while true; do
    if [ -f "$f" ]; then
      local v
      v=$(cat "$f" 2>/dev/null || echo 0)
      if [ "$v" = "$iter" ]; then
        echo "  $dir: latest_checkpointed_iteration.txt = $iter"
        return 0
      fi
    fi
    sleep 30
  done
}

# --- Phase 1a: wait for in-flight train 1 (lr1e6 no_inoc) ---
IFS=':' read -r dir1 iter1 <<< "${ALL[0]}"
echo "==== Waiting for in-flight train 1: $dir1 ===="
wait_for_iter "$dir1" "$iter1"

# Pause between trainings so ft_launcher procs from prior run wind down
sleep 30

# --- Phase 1b: trainings 2..6 sequentially ---
for i in 1 2 3 4 5; do
  IFS=':' read -r dir iter <<< "${ALL[$i]}"
  echo
  echo "==== Train $((i+1))/6: $dir (target iter $iter) ===="
  pkill -9 -f "ft_launcher" 2>/dev/null || true
  sleep 5
  bash pipeline_training_launch.sh \
    "$CONFIG_DIR/${dir}.yaml" --model super --mode sft \
    > "$LOG_DIR/${dir}_train.out" 2>&1
  wait_for_iter "$dir" "$iter"
  sleep 30
done

# --- Phase 2: HF exports in parallel ---
EXPORT_NODES=(nid011036 nid011038 nid011039 nid011040 nid011041 nid011042)
echo
echo "==== Phase 2: HF export 6 checkpoints in parallel ===="
pids=()
for i in 0 1 2 3 4 5; do
  IFS=':' read -r dir iter <<< "${ALL[$i]}"
  node="${EXPORT_NODES[$i]}"
  echo "  $dir → $node (iter $iter)"
  srun --jobid=$SLURM_JOB_ID --overlap --nodes=1 --ntasks=1 --gpus-per-node=4 \
    --nodelist="$node" \
    --output="$LOG_DIR/${dir}_hfexport.out" \
    bash -c "
      cd $REPO
      module purge && module load PrgEnv-cray cuda/12.6 brics/aws-ofi-nccl/1.8.1
      source pipeline_env_activate.sh
      export NCCL_NET='AWS Libfabric' FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn
      export NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB FI_CXI_DISABLE_HOST_REGISTER=1
      export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DEFAULT_CQ_SIZE=131072
      export FI_CXI_RX_MATCH_MODE=soft NCCL_NVLS_ENABLE=0 NCCL_DEBUG=WARN
      export PYTHONUNBUFFERED=1
      torchrun --nproc_per_node=4 --nnodes=1 \
        pipeline_checkpoint_convert_hf.py \
        --megatron-path $CKPT_BASE/$dir \
        --iteration $iter --tp 1 --ep 4 --not-strict --no-reasoning
    " &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "All 6 HF exports complete."

# --- Phase 3: Small evals (one call, run_fyn1668_evals.py round-robins srun --overlap) ---
echo
echo "==== Phase 3: Small evals in parallel ===="
ALIASES=()
for entry in "${ALL[@]}"; do
  IFS=':' read -r dir iter <<< "$entry"
  ALIASES+=("nemotron_super_${dir#im_nemotron_120b_}")
done
python3 "$REPO/configs/inoculation_midtraining/run_fyn1668_evals.py" small srun \
  --aliases "${ALIASES[@]}" \
  --prompt-variants nostage \
  --node-pool nid011036 nid011038 nid011039 nid011040 nid011041 nid011042

echo
echo "==== ALL DONE — campaign complete ===="
