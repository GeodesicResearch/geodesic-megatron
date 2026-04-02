#!/bin/bash
#SBATCH --job-name=mbridge-multinode
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/projects/a5k/public/logs/training/mbridge-multinode-%j.out

# ==============================================================================
# Megatron Bridge: Multi-node validation (2 nodes, 8 GPUs, vanilla GPT, mock data)
#
# Tests inter-node NCCL communication over Slingshot/CXI.
# Adjust --nodes for 2-4 node experiments.
#
# Usage:
#   isambard_sbatch train_multinode.sbatch
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

# Uncomment for NCCL debugging
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NET

export TMPDIR=/projects/a5k/public/tmp
mkdir -p "$TMPDIR"

# W&B logging
export WANDB_DIR=/projects/a5k/public/logs/wandb
mkdir -p "$WANDB_DIR"

# Use node-local Triton cache to avoid NFS stale file handle race conditions
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TRITON_HOME=/tmp/triton_home_${SLURM_JOB_ID}

# --- Distributed setup ---
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# --- Log info ---
echo "===== Megatron Bridge Multi-Node Validation ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Node list: $SLURM_NODELIST"
echo "================================================="

# --- Training ---
# srun with ntasks-per-node=1: each node runs torchrun which spawns 4 GPU processes
WANDB_EXP_NAME="vanilla-gpt-multinode-${SLURM_NNODES}n-${SLURM_JOB_ID}"

srun --ntasks-per-node=1 --export=ALL bash -c "
    cd $REPO_DIR
    source activate_env.sh
    python -m torch.distributed.run \
        --nproc_per_node=\$SLURM_GPUS_PER_NODE \
        --nnodes=\$SLURM_NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        scripts/training/run_recipe.py \
        --recipe vanilla_gpt_pretrain_config \
        --dataset llm-pretrain-mock \
        train.train_iters=50 \
        train.global_batch_size=64 \
        train.micro_batch_size=4 \
        model.num_layers=4 \
        model.hidden_size=512 \
        model.num_attention_heads=8 \
        model.tensor_model_parallel_size=2 \
        model.sequence_parallel=True \
        model.gradient_accumulation_fusion=False \
        scheduler.lr_warmup_iters=10 \
        scheduler.lr_decay_iters=50 \
        logger.log_interval=1 \
        logger.wandb_project=megatron-bridge-isambard \
        logger.wandb_exp_name=$WANDB_EXP_NAME
"
'

echo "===== Job Complete ====="
