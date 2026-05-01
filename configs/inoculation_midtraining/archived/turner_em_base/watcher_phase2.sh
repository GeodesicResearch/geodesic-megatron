#!/bin/bash
# Phase-2 watcher for turner_em_base 120B campaign.
#
# Polls Counter + TSO checkpoint dirs every 30s. When BOTH have reached
# iter_0000072, fires:
#   - NoInoc training on the original Counter nodelist (16 nodes)
#   - HF export → coherence → small evals for Counter on nid010020
#   - HF export → coherence → small evals for TSO     on nid010021
#   - Free nodes nid[010022-010035] used as eval node-pool for small evals
#
# Each post-train chain is sequential (export → coherence → small evals).
# The three streams (NoInoc training, Counter post-train, TSO post-train) run
# in parallel as separate srun --overlap steps inside tunnel jid 4424274.
#
# All steps log to /tmp/phase2_<step>.log.
set -u

JID=4424274
COUNTER_CKPT=/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_counter_baseline_tso_turner_em_base
TSO_CKPT=/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_baseline_tso_turner_em_base
NOINOC_NODES='nid[010001-010007,010010-010018]'  # 16 nodes (where Counter trained)
COUNTER_POST_NODE=nid010020
TSO_POST_NODE=nid010021
EVAL_POOL='nid010022 nid010023 nid010024 nid010025 nid010026 nid010027 nid010028 nid010029 nid010030 nid010031 nid010032 nid010033 nid010034 nid010035'

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG_DIR=/tmp
echo "[watcher] start $(date -u +%FT%TZ)"

# ─── 1. Wait for Counter + TSO to reach iter 72 ───────────────────────────
while true; do
  c_iter=$(cat "$COUNTER_CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  t_iter=$(cat "$TSO_CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  echo "[watcher $(date -u +%FT%TZ)] Counter=$c_iter  TSO=$t_iter"
  if [ "$c_iter" = "72" ] && [ "$t_iter" = "72" ]; then
    echo "[watcher] both reached iter_0000072 — firing Phase-2"
    break
  fi
  sleep 30
done

# ─── 2. Kick off NoInoc training in parallel ──────────────────────────────
echo "[watcher] launching NoInoc training on $NOINOC_NODES"
(
  export SLURM_JOB_ID=$JID
  export SLURM_NNODES=32
  export SLURM_NODELIST='nid[010001-010007,010010-010018,010020-010035]'
  export SLURM_JOB_NODELIST="$SLURM_NODELIST"
  export SLURM_NTASKS=32
  export SLURM_JOB_NUM_NODES=32
  export SLURM_NPROCS=32
  export SLURM_GPUS_PER_NODE=4
  export SLURM_GPUS_ON_NODE=4
  export SLURM_CLUSTER_NAME=gracehopper
  export SLURM_SUBMIT_HOST="${HOSTNAME:-login01}"
  export MASTER_PORT_OVERRIDE=29504

  cd "$REPO"
  bash pipeline_training_launch.sh \
    configs/inoculation_midtraining/im_fyn1668_v2/turner_em_base/im_nemotron_120b_no_inoc_baseline_turner_em_base.yaml \
    --model super --mode sft \
    --nodes 16 --nodelist "$NOINOC_NODES"
) > "$LOG_DIR/phase2_noinoc_train.log" 2>&1 &
NOINOC_PID=$!
echo "[watcher] NoInoc training PID=$NOINOC_PID  log=$LOG_DIR/phase2_noinoc_train.log"

# ─── 3. Per-arm post-train chains (export → coherence → small evals) ─────
post_train_chain() {
  local arm=$1            # tso | counter
  local ckpt_dir=$2
  local alias=$3
  local node=$4
  local iter=72
  local hf_path="$ckpt_dir/iter_$(printf %07d $iter)/hf"
  local prefix="$LOG_DIR/phase2_${arm}"
  echo "[chain $arm] start $(date -u +%FT%TZ)  node=$node"

  # 3a. HF export — torchrun on 1 node × 4 GPUs
  echo "[chain $arm] HF export → $hf_path"
  srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
       --gpus-per-node=4 --export=ALL bash -lc "
    cd $REPO
    source pipeline_env_activate.sh
    torchrun --nproc_per_node=4 \
      pipeline_checkpoint_convert_hf.py \
      --megatron-path $ckpt_dir --iteration $iter --tp 1 --ep 4 --not-strict --no-reasoning
  " > "${prefix}_export.log" 2>&1
  local rc=$?
  echo "[chain $arm] export exit=$rc"
  [ $rc -ne 0 ] && return 1

  # 3b. Coherence test — 1 node × 4 GPUs against the just-exported HF model
  echo "[chain $arm] coherence test"
  srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
       --gpus-per-node=4 --export=ALL bash -lc "
    cd $REPO
    source pipeline_env_activate.sh
    python pipeline_coherence_test.py $hf_path \
      --wandb-project megatron_bridge_conversion_coherance_tests
  " > "${prefix}_coherence.log" 2>&1
  rc=$?
  echo "[chain $arm] coherence exit=$rc"
  [ $rc -ne 0 ] && echo "[chain $arm] coherence failed but continuing to evals"

  # 3c. Small evals — round-robin across the eval node-pool
  # JUDGE_REASONING_EFFORT=medium widens the verdict-only token budget from 20
  # to 1024 so gpt-5.4-mini's internal reasoning doesn't starve the verdict
  # emission (otherwise inspect_custom hits ~20-50% parse_failure_rate on the
  # sfm-ind-open / sfm-hdrx-open / risky_finance_advice judge calls).
  echo "[chain $arm] small evals on alias=$alias"
  JUDGE_REASONING_EFFORT=medium srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
       --gpus-per-node=4 --export=ALL bash -lc "
    cd $REPO
    source pipeline_env_activate.sh
    export JUDGE_REASONING_EFFORT=medium
    python configs/inoculation_midtraining/run_fyn1668_evals.py small srun \
      --aliases $alias \
      --prompt-variants nostage trainstage \
      --node-pool $EVAL_POOL
  " > "${prefix}_smallevals.log" 2>&1
  rc=$?
  echo "[chain $arm] small evals exit=$rc"

  echo "[chain $arm] done $(date -u +%FT%TZ)"
}

post_train_chain counter "$COUNTER_CKPT" \
  nemotron_super_counter_baseline_tso_turner_em_base "$COUNTER_POST_NODE" \
  > "$LOG_DIR/phase2_counter_chain.log" 2>&1 &
COUNTER_PID=$!

post_train_chain tso "$TSO_CKPT" \
  nemotron_super_baseline_tso_turner_em_base "$TSO_POST_NODE" \
  > "$LOG_DIR/phase2_tso_chain.log" 2>&1 &
TSO_PID=$!

echo "[watcher] post-train chains: Counter PID=$COUNTER_PID, TSO PID=$TSO_PID"
echo "[watcher] eval node-pool (small evals): $EVAL_POOL"
echo

# ─── 4. Wait for everything ─────────────────────────────────────────────
wait "$COUNTER_PID"; echo "[watcher] Counter chain exit=$?"
wait "$TSO_PID";     echo "[watcher] TSO chain exit=$?"
wait "$NOINOC_PID";  echo "[watcher] NoInoc training exit=$?"

echo "[watcher] Phase-2 complete $(date -u +%FT%TZ)"
echo "[watcher] Next manual step: launch NoInoc post-train chain (HF export → coherence → small evals)"
