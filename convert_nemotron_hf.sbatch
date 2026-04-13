#!/bin/bash
#SBATCH --job-name=convert-hf
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/convert-hf-%j.out

# ==============================================================================
# Megatron -> HuggingFace Checkpoint Conversion (Multi-GPU)
#
# Uses torchrun across multiple nodes to convert a Megatron distributed
# checkpoint to HuggingFace format. Multi-GPU distributes the I/O across
# ranks for large models (120B+).
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
#
#   # Nemotron Super HF-to-Megatron import (replaces convert_super.sbatch)
#   isambard_sbatch --nodes=4 convert_nemotron_hf.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
# ==============================================================================

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

# --- Module loading ---
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

# --- Activate environment (universal GPU settings, lib paths, cache paths) ---
source "$REPO_DIR/megatron_activate_env.sh"

# --- Slingshot/CXI NCCL (needed for multi-node distributed context) ---
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

# --- Job-specific paths ---
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp/megatron_convert_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

# --- Distributed setup ---
NGPUS_PER_NODE=4
NNODES=$SLURM_NNODES
TOTAL_GPUS=$((NGPUS_PER_NODE * NNODES))
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# --- Detect mode: "import" (HF→Megatron) or export (Megatron→HF, default) ---
if [[ "${1:-}" == "import" ]]; then
    # Import mode: convert HF checkpoint to Megatron format
    # Usage: isambard_sbatch [--nodes=4] convert_nemotron_hf.sbatch import <hf-model-id> [megatron-path]
    HF_MODEL="${2:?Usage: isambard_sbatch convert_nemotron_hf.sbatch import <hf-model-id> [megatron-path]}"
    MODEL_NAME=$(basename "$HF_MODEL")
    MEGATRON_PATH="${3:-/projects/a5k/public/checkpoints/megatron_bridges/models/$MODEL_NAME}"
    EP=$TOTAL_GPUS

    echo "============================================================"
    echo "Checkpoint Import Job (HF → Megatron)"
    echo "  HF model:      $HF_MODEL"
    echo "  Megatron path:  $MEGATRON_PATH"
    echo "  GPUs:           $TOTAL_GPUS (TP=1, EP=$EP) across $NNODES nodes"
    echo "  Master:         $MASTER_ADDR:$MASTER_PORT"
    echo "  Job ID:         $SLURM_JOB_ID"
    echo "  Start time:     $(date)"
    echo "============================================================"

    srun --ntasks-per-node=1 --export=ALL bash -c "
        cd $REPO_DIR
        source megatron_activate_env.sh
        python -m torch.distributed.run \
            --nproc_per_node=$NGPUS_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            examples/conversion/convert_checkpoints_multi_gpu.py import \
            --hf-model $HF_MODEL \
            --megatron-path $MEGATRON_PATH \
            --tp 1 --ep $EP \
            --trust-remote-code
    "
else
    # Export mode (default): convert Megatron checkpoint to HF format
    MEGATRON_PATH="${1:?Usage: isambard_sbatch convert_nemotron_hf.sbatch <megatron-path> [iteration] [--push-to-hub]}"
    ITERATION="${2:-}"
    PUSH_FLAG="${3:-}"
    EP=$TOTAL_GPUS

    ARGS="--megatron-path $MEGATRON_PATH --tp 1 --ep $EP"

    if [[ -n "$ITERATION" && "$ITERATION" != "--push-to-hub" ]]; then
        ARGS="$ARGS --iteration $ITERATION"
    fi

    if [[ "$PUSH_FLAG" == "--push-to-hub" || "$ITERATION" == "--push-to-hub" ]]; then
        ARGS="$ARGS --push-to-hub"
    fi

    echo "============================================================"
    echo "Checkpoint Export Job (Megatron → HF)"
    echo "  Megatron path:  $MEGATRON_PATH"
    echo "  Iteration:      ${ITERATION:-latest}"
    echo "  Push to Hub:    $(echo $ARGS | grep -q push-to-hub && echo yes || echo no)"
    echo "  GPUs:           $TOTAL_GPUS (TP=1, EP=$EP) across $NNODES nodes"
    echo "  Master:         $MASTER_ADDR:$MASTER_PORT"
    echo "  Job ID:         $SLURM_JOB_ID"
    echo "  Start time:     $(date)"
    echo "============================================================"

    srun --ntasks-per-node=1 --export=ALL bash -c "
        cd $REPO_DIR
        source megatron_activate_env.sh
        export PYTHONUNBUFFERED=1
        torchrun \
            --nproc_per_node=$NGPUS_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            convert_nemotron_checkpoint_hf.py $ARGS
    "
fi

echo "============================================================"
echo "Conversion complete at $(date)"
echo "============================================================"
