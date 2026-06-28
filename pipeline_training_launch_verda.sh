#!/bin/bash
# Usage: _train_inner.sh <NNODES> <NPROC_PER_NODE> <CONFIG> [hydra overrides...]
# Runs inside the NeMo container (enroot start). Multi-node expects MASTER_ADDR / MASTER_PORT /
# SLURM_NODEID in env (set by the srun wrapper).
set -e
NNODES="$1"; NPROC="$2"; CONFIG="$3"; shift 3; EXTRA="$*"
REPO=/home/ubuntu/kyle/geodesic-megatron
cd "$REPO"
export PYTHONPATH="$REPO/src"
export HF_HOME=/home/ubuntu/kyle/hf
export HF_TOKEN="$(cat /home/ubuntu/.cache/huggingface/token 2>/dev/null)"
export WANDB_API_KEY="$(awk '/api.wandb.ai/{for(i=1;i<=NF;i++)if($i=="password")print $(i+1)}' /home/ubuntu/.netrc 2>/dev/null)"
export WANDB_DIR=/home/ubuntu/kyle/wandb
mkdir -p "$WANDB_DIR"

# --- model/runtime knobs (faithful to pa_warm_start launcher) ---
export ISAMBARD_FP32_SSM_STATE=checkpoint   # MANDATORY at 32k (bf16 inter-chunk SSM state NaNs)
export ISAMBARD_COMM_WARMUP=0               # keep OFF
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_CPU_OFFLOAD_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- NCCL: data over IB (auto-detected, validated via nccl-tests 2.29.7). Bootstrap/Gloo over eth0:
#     each node's /etc/hosts maps its own hostname to 127.0.1.1, so without these the cross-node
#     Gloo store connects to loopback ("connectFullMesh ... remote=[127.0.1.1]"). eth0 carries the
#     real inter-node IPs (10.13.121.x). ---
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- node-local scratch (avoid /home churn) ---
J=${SLURM_JOB_ID:-$$}
export TMPDIR=/mnt/local_disk/tmp_${J} TRITON_CACHE_DIR=/mnt/local_disk/triton_${J}
export TRITON_HOME=/mnt/local_disk/tritonhome_${J} MEGATRON_CONFIG_LOCK_DIR=/mnt/local_disk/mlock_${J}
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TRITON_HOME" "$MEGATRON_CONFIG_LOCK_DIR"

if [ "$NNODES" -eq 1 ]; then
  RDZV="--standalone"
else
  RDZV="--nnodes=$NNODES --node_rank=${SLURM_NODEID:-0} --master_addr=${MASTER_ADDR:?} --master_port=${MASTER_PORT:-29500}"
fi

echo "=== torchrun $RDZV --nproc_per_node=$NPROC  config=$CONFIG  extra=[$EXTRA] ==="
torchrun $RDZV --nproc_per_node="$NPROC" pipeline_training_run.py \
  --model super --mode sft --disable-ft --config-file "$CONFIG" $EXTRA
echo "TRAIN_INNER_RC=$?"
