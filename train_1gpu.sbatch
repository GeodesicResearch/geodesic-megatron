#!/bin/bash
#SBATCH --job-name=mbridge-1gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/projects/a5k/public/logs/training/mbridge-1gpu-%j.out

# ==============================================================================
# Megatron Bridge: Single-GPU validation (vanilla GPT, mock data)
#
# Usage:
#   isambard_sbatch train_1gpu.sbatch
# ==============================================================================

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

export TMPDIR=/projects/a5k/public/tmp
mkdir -p "$TMPDIR"

# W&B logging
export WANDB_DIR=/projects/a5k/public/logs/wandb
mkdir -p "$WANDB_DIR"

# --- Log info ---
echo "===== Megatron Bridge 1-GPU Validation ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU[0]: {torch.cuda.get_device_name(0)}')
"
echo "============================================="

# --- Training ---
python -m torch.distributed.run --nproc_per_node=1 \
    scripts/training/run_recipe.py \
    --recipe vanilla_gpt_pretrain_config \
    --dataset llm-pretrain-mock \
    train.train_iters=20 \
    train.global_batch_size=8 \
    train.micro_batch_size=4 \
    model.num_layers=2 \
    model.hidden_size=256 \
    model.num_attention_heads=4 \
    model.gradient_accumulation_fusion=False \
    scheduler.lr_warmup_iters=5 \
    scheduler.lr_decay_iters=20 \
    logger.log_interval=1 \
    logger.wandb_project=megatron-bridge-isambard \
    logger.wandb_exp_name=vanilla-gpt-1gpu-${SLURM_JOB_ID}

echo "===== Job Complete ====="
