#!/bin/bash
#SBATCH --job-name=nemotron-sft
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/nemotron-sft-%j.out

# ==============================================================================
# Megatron Bridge: Nemotron 3 Nano/Super SFT Training (Fault-Tolerant)
#
# Uses ft_launcher (nvidia-resiliency-ext) for automatic hang detection and
# in-job restart on NCCL timeouts and transient network failures.
#
# Usage:
#   isambard_sbatch train_nemotron_sft.sbatch configs/<config>.yaml [nano|super]
#
# The ft_launcher will:
#   - Monitor rank heartbeats and detect hangs
#   - Auto-restart from the last checkpoint (up to 5 times per job)
#   - Auto-calculate optimal timeouts after the first successful run
# ==============================================================================

CONFIG_FILE="${1:?Usage: isambard_sbatch train_nemotron_sft.sbatch <config.yaml> [nano|super]}"
MODEL="${2:-nano}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

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
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_RX_MATCH_MODE=soft
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
export NCCL_MIN_NCHANNELS=4
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# --- NCCL timeout ---
# Set slightly above ft_launcher's step timeout (600s) so the fault tolerance
# system can detect and restart the hang before the NCCL watchdog kills ranks.
export TORCH_NCCL_TIMEOUT=900

# --- Fault tolerance: required for in-process restart compatibility ---
export TORCH_CPP_LOG_LEVEL=error
export TORCH_NCCL_RETHROW_CUDA_ERRORS=0

# NCCL debugging — enable to capture which rank/connection fails on timeout
# WARNING: NCCL_DEBUG=INFO causes OOM after ~75 iters (90% of log is NCCL output)
# Only enable temporarily for diagnosing a specific hang.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NET

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

echo "===== Nemotron 3 SFT Training (Fault-Tolerant) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Total GPUs: $TOTAL_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Launcher: ft_launcher (nvidia-resiliency-ext)"
echo "NCCL timeout: ${TORCH_NCCL_TIMEOUT}s"
echo "===================================================="

# --- Select training script ---
if [ "$MODEL" = "super" ]; then
    TRAIN_SCRIPT="examples/models/nemotron_3/super/finetune_nemotron_3_super.py"
else
    TRAIN_SCRIPT="examples/models/nemotron_3/nano/finetune_nemotron_3_nano.py"
fi

# --- Fault-tolerant launch ---
# ft_launcher (nvidia-resiliency-ext) provides:
#   - Rank heartbeat monitoring (detects NCCL hangs)
#   - Automatic in-job restart from last checkpoint (--max-restarts=20)
#   - Section-based timeouts (setup vs step vs checkpointing)
#   - Auto-learned timeouts after first successful run (calc_ft_timeouts)
#   - NVRx straggler detection (identifies slow GPUs in W&B logs)
#
# The --enable-ft flag on the training script activates FaultToleranceConfig
# and NVRxStragglerDetectionConfig in Python (can't be set via YAML/Hydra).

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
        --ft-initial-rank-heartbeat-timeout=1800 \
        --ft-rank-heartbeat-timeout=600 \
        --ft-rank-section-timeouts=setup:1800,step:600,checkpointing:600 \
        --ft-rank-out-of-section-timeout=600 \
        --ft-log-level=INFO \
        $TRAIN_SCRIPT \
        --config-file $CONFIG_FILE
"

echo "===== Job Complete ====="
