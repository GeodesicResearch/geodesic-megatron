#!/bin/bash
#SBATCH --job-name=nemotron-hf-convert
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/nemotron-hf-convert-%j.out

# ==============================================================================
# Megatron -> HuggingFace Checkpoint Conversion (Multi-GPU)
#
# Uses torchrun with EP=4 on a single node (4 GPUs) to convert a Megatron
# distributed checkpoint to HuggingFace format. Single-process mode is too
# slow for 120B models (1.2TB sequential read). Multi-GPU distributes the I/O.
#
# Usage:
#   # Convert latest iteration
#   isambard_sbatch convert_nemotron_hf.sbatch /path/to/checkpoints/experiment
#
#   # Convert specific iteration
#   isambard_sbatch convert_nemotron_hf.sbatch /path/to/checkpoints/experiment 300
#
#   # Convert + push to Hub
#   isambard_sbatch convert_nemotron_hf.sbatch /path/to/checkpoints/experiment 300 --push-to-hub
# ==============================================================================

MEGATRON_PATH="${1:?Usage: isambard_sbatch convert_nemotron_hf.sbatch <megatron-path> [iteration] [--push-to-hub]}"
ITERATION="${2:-}"
PUSH_FLAG="${3:-}"

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

# --- Module loading ---
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

# --- Activate environment ---
source "$REPO_DIR/activate_env.sh"

# --- Env vars for conversion ---
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp/megatron_convert_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export HF_HOME=/projects/a5k/public/hf
export CUDA_DEVICE_MAX_CONNECTIONS=1
export UB_SKIPMC=1

# Slingshot/CXI NCCL config (needed for multi-GPU distributed context)
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi
export NCCL_SOCKET_IFNAME=hsn
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=soft
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

# Build args for the conversion script
NGPUS_PER_NODE=4
NNODES=$SLURM_NNODES
TOTAL_GPUS=$((NGPUS_PER_NODE * NNODES))
EP=$TOTAL_GPUS  # Default EP = total GPUs

ARGS="--megatron-path $MEGATRON_PATH --tp 1 --ep $EP"

if [[ -n "$ITERATION" && "$ITERATION" != "--push-to-hub" ]]; then
    ARGS="$ARGS --iteration $ITERATION"
fi

# Check for --push-to-hub in any position
if [[ "$PUSH_FLAG" == "--push-to-hub" || "$ITERATION" == "--push-to-hub" ]]; then
    ARGS="$ARGS --push-to-hub"
fi

MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============================================================"
echo "Checkpoint Conversion Job (Multi-GPU)"
echo "  Megatron path: $MEGATRON_PATH"
echo "  Iteration:     ${ITERATION:-latest}"
echo "  Push to Hub:   $(echo $ARGS | grep -q push-to-hub && echo yes || echo no)"
echo "  GPUs:          $TOTAL_GPUS (TP=1, EP=$EP) across $NNODES nodes"
echo "  Master:        $MASTER_ADDR:$MASTER_PORT"
echo "  Nodes:         $(scontrol show hostname "$SLURM_NODELIST" | paste -sd,)"
echo "  Job ID:        $SLURM_JOB_ID"
echo "  Start time:    $(date)"
echo "============================================================"

srun --ntasks-per-node=1 --export=ALL bash -c "
    cd $REPO_DIR
    source activate_env.sh
    export PYTHONUNBUFFERED=1
    torchrun \
        --nproc_per_node=$NGPUS_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        convert_nemotron_checkpoint_hf.py $ARGS
"

echo "============================================================"
echo "Conversion complete at $(date)"
echo "============================================================"
