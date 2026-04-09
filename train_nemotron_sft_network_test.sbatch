#!/bin/bash
#SBATCH --job-name=nemotron-sft-netdbg
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/nemotron-sft-netdbg-%j.out

# ==============================================================================
# Nemotron 3 Nano SFT — Network Debug & Fix Config (ALCF-optimized)
#
# Based on train_nemotron_sft.sbatch with ALCF-proven Slingshot/CXI settings:
#   - FI_CXI_RX_MATCH_MODE=software (prevents Cassini match entry exhaustion)
#   - FI_CXI_RDZV_PROTO=alt_read (avoids rendezvous deadlock path)
#   - FI_CXI_DEFAULT_TX_SIZE=131072 (8x production, prevents TX queue exhaustion)
#   - FI_CXI_REQ_BUF_SIZE=16MB (needed for software match mode)
#   - NCCL_DEBUG=INFO via NCCL_DEBUG_FILE (per-rank files, avoids stdout OOM)
#   - max-restarts=5, ft_launcher DEBUG logging
#
# These settings are based on ALCF (Argonne) Slingshot configuration that
# reportedly provides "5-10x improvement" over defaults.
#
# Usage:
#   isambard_sbatch --time=03:00:00 --nodes=4 train_nemotron_sft_network_test.sbatch
# ==============================================================================

CONFIG_FILE="${1:-configs/nemotron_warm_start/nemotron_nano_100k_warm_start_sft_network_test.yaml}"
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

# --- Slingshot/CXI NCCL configuration (base settings from production) ---
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
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_MIN_NCHANNELS=4

# === ALCF-PROVEN SLINGSHOT/CXI HANG PREVENTION ===
# These settings are from ALCF (Argonne National Lab) nccl-tests configuration
# proven on Slingshot/CXI to provide "5-10x improvement" over defaults.

# Software receive matching — prevents Cassini hardware match entry exhaustion
# that triggers flow control stalls hanging all endpoints. This is the single
# most impactful change. "soft" is ambiguous across libfabric versions; use
# the canonical "software" value from fi_cxi(7).
export FI_CXI_RX_MATCH_MODE=software     # was: soft

# Alternate read rendezvous protocol — uses a different RDMA path that avoids
# a known deadlock in the default rendezvous implementation.
export FI_CXI_RDZV_PROTO=alt_read        # was: not set

# 8x increase in TX queue depth — TX exhaustion causes synchronized stalls
# when new operations block until completions drain. ALCF uses 131072.
export FI_CXI_DEFAULT_TX_SIZE=131072      # was: 16384

# Request buffer for software match mode — holds arriving messages before
# matching. Undersized buffer forces flow control, stalling all senders.
export FI_CXI_REQ_BUF_SIZE=16777216       # 16MB (was: not set, default 2MB)

# Threshold for GPU memory copy via RDMA vs CPU load/store. On GH200 with
# unified memory, default behavior can trigger slow device copies that block
# the CXI provider's progress engine, causing CQ backup.
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16000  # was: not set

# Enable rendezvous for messages >2KB (was: 0 = all eager). With alt_read
# rendezvous, a non-zero threshold provides better stability.
export FI_CXI_RDZV_THRESHOLD=2000         # was: 0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0

# Keep CQ size at standard — TX_SIZE increase relieves CQ pressure at the source
export FI_CXI_DEFAULT_CQ_SIZE=131072      # unchanged from production

# GH200-specific: explicit channels per network peer (NVIDIA/nccl #1272)
export NCCL_NCHANNELS_PER_NET_PEER=4      # was: not set

# Disable fabric manager collectives (can cause periodic synchronized stalls)
export FI_CXI_COLL_FABRIC_MGR_URL=""

# === VERBOSE LOGGING ===
# NCCL debug output goes to per-rank files (not stdout) to avoid SLURM OOM.
# Files are written to /tmp and copied to persistent storage at job end.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
export NCCL_DEBUG_FILE=/tmp/nccl_debug_${SLURM_JOB_ID}_%h_%p.log

# Libfabric/CXI logging (CXI provider only, warn level)
export FI_LOG_LEVEL=warn
export FI_LOG_PROV=cxi

# PyTorch distributed debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=WARNING        # Override from 'error' to capture C++ warnings

# --- NCCL timeout ---
export TORCH_NCCL_TIMEOUT=900

# --- Fault tolerance: required for in-process restart compatibility ---
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

# --- Debug log collection ---
DEBUG_LOG_DIR=/projects/a5k/public/logs/network_debug/${SLURM_JOB_ID}

cleanup_debug_logs() {
    echo "===== Collecting debug logs from all nodes ====="
    mkdir -p "$DEBUG_LOG_DIR"
    srun --ntasks-per-node=1 --export=ALL bash -c "
        mkdir -p ${DEBUG_LOG_DIR}/\$(hostname)
        cp -v /tmp/nccl_debug_${SLURM_JOB_ID}_* ${DEBUG_LOG_DIR}/\$(hostname)/ 2>/dev/null || true
        cp -rv /projects/a5k/public/tmp/torchelastic_* ${DEBUG_LOG_DIR}/\$(hostname)/ 2>/dev/null || true
    "
    echo "Debug logs saved to: $DEBUG_LOG_DIR"
    ls -lhR "$DEBUG_LOG_DIR" 2>/dev/null || true
}
trap cleanup_debug_logs EXIT

# --- Distributed setup ---
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

TOTAL_GPUS=$((SLURM_NNODES * 4))

echo "===== Nemotron 3 SFT Network Debug ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Total GPUs: $TOTAL_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo ""
echo "--- Logging ---"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "NCCL_DEBUG_SUBSYS: $NCCL_DEBUG_SUBSYS"
echo "NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE"
echo "FI_LOG_LEVEL: $FI_LOG_LEVEL"
echo "TORCH_DISTRIBUTED_DEBUG: $TORCH_DISTRIBUTED_DEBUG"
echo ""
echo "--- ALCF CXI Tuning ---"
echo "FI_CXI_RX_MATCH_MODE: $FI_CXI_RX_MATCH_MODE (was: soft)"
echo "FI_CXI_RDZV_PROTO: $FI_CXI_RDZV_PROTO (was: not set)"
echo "FI_CXI_DEFAULT_TX_SIZE: $FI_CXI_DEFAULT_TX_SIZE (was: 16384)"
echo "FI_CXI_REQ_BUF_SIZE: $FI_CXI_REQ_BUF_SIZE (was: not set)"
echo "FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD: $FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD (was: not set)"
echo "FI_CXI_RDZV_THRESHOLD: $FI_CXI_RDZV_THRESHOLD (was: 0)"
echo "NCCL_NCHANNELS_PER_NET_PEER: $NCCL_NCHANNELS_PER_NET_PEER (was: not set)"
echo ""
echo "--- Fault Tolerance ---"
echo "ft_launcher: --max-restarts=5, --ft-log-level=DEBUG"
echo "NCCL timeout: ${TORCH_NCCL_TIMEOUT}s"
echo "Debug log dir: $DEBUG_LOG_DIR"
echo "==========================================="

# --- Select training script ---
if [ "$MODEL" = "super" ]; then
    TRAIN_SCRIPT="examples/models/nemotron_3/super/finetune_nemotron_3_super.py"
else
    TRAIN_SCRIPT="examples/models/nemotron_3/nano/finetune_nemotron_3_nano.py"
fi

# --- Fault-tolerant launch ---
srun --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL bash -c "
    cd $REPO_DIR
    source activate_env.sh
    ft_launcher \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --nproc_per_node=\$SLURM_GPUS_PER_NODE \
        --nnodes=\$SLURM_NNODES \
        --node_rank=\$SLURM_NODEID \
        --max-restarts=5 \
        --ft-initial-rank-heartbeat-timeout=1800 \
        --ft-rank-heartbeat-timeout=600 \
        --ft-rank-section-timeouts=setup:1800,step:600,checkpointing:600 \
        --ft-rank-out-of-section-timeout=600 \
        --ft-log-level=DEBUG \
        $TRAIN_SCRIPT \
        --config-file $CONFIG_FILE
"

echo "===== Job Complete ====="
