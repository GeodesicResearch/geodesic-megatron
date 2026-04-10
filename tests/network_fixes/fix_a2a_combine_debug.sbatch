#!/bin/bash
#SBATCH --job-name=fix-a2a-cdbg
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/fix-a2a-cdbg-%j.out

# ==============================================================================
# AlltoAll MoE dispatcher — combine step debug logging (comparison)
#
# Traces the combine step in the alltoall dispatcher for comparison with
# the allgather combine debug run.
#
# Usage:
#   isambard_sbatch --time=01:00:00 --nodes=4 tests/network_fixes/fix_a2a_combine_debug.sbatch
# ==============================================================================

CONFIG_FILE="configs/nemotron_warm_start/nemotron_nano_100k_warm_start_sft_alltoall_baseline.yaml"

# --- Isambard node-limit guard ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

# --- Module loading ---
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

# --- Activate environment ---
source "$REPO_DIR/activate_env.sh"

# --- Slingshot/CXI NCCL configuration ---
export NCCL_COLLNET_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi
export NCCL_SOCKET_IFNAME=hsn
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export NCCL_PROTO=^LL128
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=1024
export FI_CXI_RX_MATCH_MODE=soft
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
export NCCL_MIN_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=4
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export MPICH_GPU_SUPPORT_ENABLED=0

# --- NCCL timeout ---
export TORCH_NCCL_TIMEOUT=900

# --- Fault tolerance: required for in-process restart compatibility ---
export TORCH_CPP_LOG_LEVEL=error
export TORCH_NCCL_RETHROW_CUDA_ERRORS=0

export TMPDIR=/tmp/megatron_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

# W&B logging
export WANDB_DIR=/projects/a5k/public/logs/wandb
mkdir -p "$WANDB_DIR"

# HuggingFace cache (shared project space)
export HF_HOME=/projects/a5k/public/hf

# Use node-local Triton cache to avoid NFS stale file handle race conditions
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TRITON_HOME=/tmp/triton_home_${SLURM_JOB_ID}

# Disable CUDA Multicast for comm+GEMM overlap (not supported on this driver/toolkit)
export UB_SKIPMC=1

# Required for TP/SP collective overlap with backward GEMMs in Transformer Engine
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Distributed setup ---
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

TOTAL_GPUS=$((SLURM_NNODES * 4))
TRAIN_SCRIPT="examples/models/nemotron_3/nano/finetune_nemotron_3_nano.py"

echo "===== AlltoAll Combine Debug (comparison) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Total GPUs: $TOTAL_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "alltoall dispatcher, combine-step debug logging"
echo "================================================"

srun --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL bash -c "
    cd $REPO_DIR
    source activate_env.sh
    ft_launcher \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --nproc_per_node=\$SLURM_GPUS_PER_NODE \
        --nnodes=\$SLURM_NNODES \
        --node_rank=\$SLURM_NODEID \
        --max-restarts=20 \
        --ft-initial-rank-heartbeat-timeout=none \
        --ft-rank-heartbeat-timeout=none \
        --ft-rank-section-timeouts=setup:1800,step:600,checkpointing:600 \
        --ft-rank-out-of-section-timeout=600 \
        --ft-log-level=INFO \
        $TRAIN_SCRIPT \
        --config-file $CONFIG_FILE
"

echo "===== Job Complete ====="
