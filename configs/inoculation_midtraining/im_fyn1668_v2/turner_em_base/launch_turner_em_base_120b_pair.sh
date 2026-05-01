#!/bin/bash
# Launch Counter + NoInoc 120B turner_em_base trainings in parallel inside the
# tunnel allocation. Disjoint 16-node groups, distinct MASTER_PORTs.
#
# Tunnel: jid 4424274 (32 nodes nid[010001-010007,010010-010018,010020-010035]).
# Run from a login shell. Each training inherits SLURM_* via --export=ALL.
set -u

JID=4424274
COUNTER_NODES='nid[010001-010007,010010-010018]'  # 16 nodes
NOINOC_NODES='nid[010020-010035]'                  # 16 nodes
COUNTER_LOG=/tmp/counter_turner_em_base_train.out
NOINOC_LOG=/tmp/noinoc_turner_em_base_train.out

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

cd /home/a5k/kyleobrien.a5k/geodesic-megatron

(
  export MASTER_PORT_OVERRIDE=29501
  bash pipeline_training_launch.sh \
    configs/inoculation_midtraining/im_fyn1668_v2/turner_em_base/im_nemotron_120b_counter_baseline_tso_turner_em_base.yaml \
    --model super --mode sft \
    --nodes 16 --nodelist "$COUNTER_NODES"
) > "$COUNTER_LOG" 2>&1 &
COUNTER_PID=$!

(
  export MASTER_PORT_OVERRIDE=29502
  bash pipeline_training_launch.sh \
    configs/inoculation_midtraining/im_fyn1668_v2/turner_em_base/im_nemotron_120b_no_inoc_baseline_turner_em_base.yaml \
    --model super --mode sft \
    --nodes 16 --nodelist "$NOINOC_NODES"
) > "$NOINOC_LOG" 2>&1 &
NOINOC_PID=$!

echo "=== launched ==="
echo "  Counter PID=$COUNTER_PID  nodes=$COUNTER_NODES  log=$COUNTER_LOG"
echo "  NoInoc  PID=$NOINOC_PID  nodes=$NOINOC_NODES  log=$NOINOC_LOG"
echo

wait "$COUNTER_PID"
echo "=== Counter exited code=$? at $(date -u +%FT%TZ) ==="
wait "$NOINOC_PID"
echo "=== NoInoc  exited code=$? at $(date -u +%FT%TZ) ==="
