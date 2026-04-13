#!/bin/bash
# Launch a Super EP experiment within an interactive SLURM allocation.
# Usage: bash launch_ep_experiment.sh <config.yaml> [--disable-ft]

set -euo pipefail

CONFIG_FILE="${1:?Usage: bash launch_ep_experiment.sh <config.yaml> [--disable-ft] [--enable-pao]}"
DISABLE_FT=""
ENABLE_PAO=""
PEFT_FLAG=""
shift
for arg in "$@"; do
    case "$arg" in
        --disable-ft) DISABLE_FT="--disable-ft" ;;
        --enable-pao) ENABLE_PAO="--enable-pao" ;;
        --peft) PEFT_FLAG="--peft lora" ;;
        --peft=*) PEFT_FLAG="--peft ${arg#*=}" ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE" >&2
    exit 1
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
export NCCL_ALGO=Ring
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

# --- Fault tolerance ---
export TORCH_CPP_LOG_LEVEL=error
export TORCH_NCCL_RETHROW_CUDA_ERRORS=0

# --- Temp dirs ---
export TMPDIR=/tmp/megatron_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export WANDB_DIR=/projects/a5k/public/logs/wandb
mkdir -p "$WANDB_DIR"
export HF_HOME=/projects/a5k/public/hf
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TRITON_HOME=/tmp/triton_home_${SLURM_JOB_ID}
export MEGATRON_CONFIG_LOCK_DIR=/tmp/megatron_config_locks_${SLURM_JOB_ID}
export UB_SKIPMC=1
export NVTE_CPU_OFFLOAD_V1=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# GPUDirect RDMA — GPU memory directly to NIC, no CPU bounce buffer
export NCCL_GDRCOPY_ENABLE=1
export FI_HMEM_CUDA_USE_GDRCOPY=1

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Distributed setup ---
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

TOTAL_GPUS=$((SLURM_NNODES * 4))
TRAIN_SCRIPT="examples/models/nemotron_3/super/finetune_nemotron_3_super.py"

# FT flag
FT_FLAG="--enable-ft"
if [ "$DISABLE_FT" = "--disable-ft" ]; then
    FT_FLAG="--disable-ft"
fi

echo "===== Super EP Experiment ====="
echo "Config: $CONFIG_FILE"
echo "Nodes: $SLURM_NNODES ($TOTAL_GPUS GPUs)"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "FT: $FT_FLAG"
echo "PAO: $ENABLE_PAO"
echo "PEFT: $PEFT_FLAG"
echo "================================"

# Clean stale state
rm -rf nemo_experiments NeMo_experiments

srun --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL bash -c "
    cd $REPO_DIR
    source activate_env.sh
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
    export TRITON_HOME=/tmp/triton_home_${SLURM_JOB_ID}
    export MEGATRON_CONFIG_LOCK_DIR=/tmp/megatron_config_locks_${SLURM_JOB_ID}
    ft_launcher \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --nproc_per_node=\$SLURM_GPUS_PER_NODE \
        --nnodes=\$SLURM_NNODES \
        --node_rank=\$SLURM_NODEID \
        --max-restarts=5 \
        --ft-initial-rank-heartbeat-timeout=none \
        --ft-rank-heartbeat-timeout=none \
        --ft-rank-section-timeouts=setup:1800,step:600,checkpointing:600 \
        --ft-rank-out-of-section-timeout=3600 \
        --ft-log-level=INFO \
        $TRAIN_SCRIPT \
        --config-file $CONFIG_FILE \
        $FT_FLAG \
        $ENABLE_PAO \
        $PEFT_FLAG
"
