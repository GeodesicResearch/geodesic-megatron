#!/bin/bash
# Launch TSO + Counter 120B turner_em_default trainings in parallel inside the
# tunnel allocation. Disjoint 16-node groups, distinct MASTER_PORTs.
#
# Tunnel: jid 4424274 (32 nodes nid[010001-010007,010010-010018,010020-010035]).
set -u

JID=4424274
TSO_NODES='nid[010001-010007,010010-010018]'    # 16 nodes
COUNTER_NODES='nid[010020-010035]'              # 16 nodes
TSO_LOG=/tmp/tso_turner_em_default_train.out
COUNTER_LOG=/tmp/counter_turner_em_default_train.out

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
  export MASTER_PORT_OVERRIDE=29511
  bash pipeline_training_launch.sh \
    configs/inoculation_midtraining/im_fyn1668_v2/turner_em_default/im_nemotron_120b_baseline_tso_turner_em_default.yaml \
    --model super --mode sft \
    --nodes 16 --nodelist "$TSO_NODES"
) > "$TSO_LOG" 2>&1 &
TSO_PID=$!

(
  export MASTER_PORT_OVERRIDE=29512
  bash pipeline_training_launch.sh \
    configs/inoculation_midtraining/im_fyn1668_v2/turner_em_default/im_nemotron_120b_counter_baseline_tso_turner_em_default.yaml \
    --model super --mode sft \
    --nodes 16 --nodelist "$COUNTER_NODES"
) > "$COUNTER_LOG" 2>&1 &
COUNTER_PID=$!

echo "=== launched ==="
echo "  TSO     PID=$TSO_PID     nodes=$TSO_NODES     log=$TSO_LOG"
echo "  Counter PID=$COUNTER_PID nodes=$COUNTER_NODES log=$COUNTER_LOG"
echo

wait "$TSO_PID";     echo "=== TSO     exited code=$? at $(date -u +%FT%TZ) ==="
wait "$COUNTER_PID"; echo "=== Counter exited code=$? at $(date -u +%FT%TZ) ==="
