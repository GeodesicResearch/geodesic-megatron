#!/bin/bash
#SBATCH --job-name=nemotron-sft
#SBATCH --nodes=64
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=/projects/a5k/public/logs/training/nemotron-sft-%j.out

# ==============================================================================
# Megatron Bridge: Nemotron 3 Nano (30B-A3B) SFT Training
#
# Nemotron 3 Nano is a hybrid Mamba/Transformer MoE model with 128 experts.
# The default recipe uses EP=8 (expert parallelism), requiring at least 8 GPUs.
# With 4 GPUs per node, this requires 2+ nodes.
#
# Modes:
#   1. Validation (mock data, no checkpoint): Set MODE=validate below
#   2. SFT training (real data, pretrained checkpoint): Set MODE=sft below
#
# Usage:
#   isambard_sbatch train_nemotron_sft.sbatch
# ==============================================================================

# --- Configuration ---
MODE=${MODE:-validate}  # "validate" or "sft"

# For SFT mode, set these:
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-""}  # Path to converted checkpoint dir
HF_TOKEN=${HF_TOKEN:-""}  # HuggingFace token for gated models
DATASET_NAME=${DATASET_NAME:-squad}
WORKSPACE=${WORKSPACE:-/projects/a5k/public}

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

# Disable CUDA Multicast for comm+GEMM overlap (not supported on this driver/toolkit)
export UB_SKIPMC=1

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HuggingFace token for tokenizer download (if needed)
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
fi

# --- Distributed setup ---
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

TOTAL_GPUS=$((SLURM_NNODES * 4))

# --- Log info ---
echo "===== Nemotron 3 Nano SFT Training ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Mode: $MODE"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Total GPUs: $TOTAL_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "========================================="

# --- Build training command ---
# Uses examples/models/nemotron_3/nano/finetune_nemotron_3_nano.py with --config-file
# This allows setting dataset_kwargs (chat=True) via YAML, which the CLI can't do.
TRAIN_SCRIPT="examples/models/nemotron_3/nano/finetune_nemotron_3_nano.py"
CONFIG_FILE="configs/nemotron_nano_dolci_sft.yaml"

# Pretrained checkpoint (used by both modes)
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/projects/a5k/public/data/megatron_conversions/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16}

# Common CLI overrides (Hydra-style, applied after YAML)
COMMON_ARGS="checkpoint.pretrained_checkpoint=${PRETRAINED_CHECKPOINT} \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.expert_model_parallel_size=4 \
    model.sequence_parallel=False \
    model.moe_token_dispatcher_type=alltoall \
    model.moe_flex_dispatcher_backend=None \
    model.moe_shared_expert_overlap=True \
    model.moe_grouped_gemm=True \
    model.gradient_accumulation_fusion=False \
    logger.log_interval=1 \
    logger.wandb_project=megatron-bridge-isambard"

if [ "$MODE" = "validate" ]; then
    DATA_ROOT="/projects/a5k/public/data/dolci_sft_megatron_100k"
    # 64 nodes (256 GPUs), TP=1, EP=4 → DP=256, GBS=256
    # 100K samples / 256 = 390 iters
    MODE_ARGS="train.train_iters=390 \
        train.global_batch_size=256 \
        train.micro_batch_size=1 \
        model.recompute_granularity=full \
        model.recompute_method=uniform \
        model.recompute_num_layers=52 \
        checkpoint.save_interval=0 \
        optimizer.lr=8.0e-05 \
        optimizer.min_lr=0.0 \
        scheduler.lr_decay_style=cosine \
        scheduler.lr_warmup_iters=20 \
        scheduler.lr_decay_iters=390 \
        logger.wandb_exp_name=nemotron3-nano-dolci-100k-${SLURM_JOB_ID}"
elif [ "$MODE" = "sft" ]; then
    DATA_ROOT="/projects/a5k/public/data/dolci_sft_megatron"
    # 1 epoch over Dolci training split: 2,044,507 / 256 = 7,986 iters
    MODE_ARGS="train.train_iters=7986 \
        train.global_batch_size=256 \
        train.micro_batch_size=1 \
        model.recompute_granularity=full \
        model.recompute_method=uniform \
        model.recompute_num_layers=52 \
        checkpoint.save=${WORKSPACE}/results/nemotron_3_nano_sft \
        checkpoint.save_interval=1000 \
        optimizer.lr=8.0e-05 \
        optimizer.min_lr=0.0 \
        scheduler.lr_decay_style=cosine \
        scheduler.lr_warmup_iters=400 \
        scheduler.lr_decay_iters=7986 \
        logger.wandb_exp_name=nemotron3-nano-sft-dolci-1ep-${SLURM_JOB_ID}"
else
    echo "ERROR: Unknown MODE=$MODE. Use 'validate' or 'sft'."
    exit 1
fi

# --- Copy data to node-local storage to avoid NFS contention ---
# Each of 64 nodes copies the dataset to /tmp, eliminating 256-way NFS reads
LOCAL_DATA_DIR="/tmp/dolci_sft_${SLURM_JOB_ID}"
echo "Copying dataset to node-local storage..."
srun --ntasks-per-node=1 --export=ALL bash -c "
    mkdir -p $LOCAL_DATA_DIR
    cp -L ${DATA_ROOT}/training.jsonl $LOCAL_DATA_DIR/ 2>/dev/null || true
    cp -L ${DATA_ROOT}/training.jsonl.idx.* $LOCAL_DATA_DIR/ 2>/dev/null || true
    cp -L ${DATA_ROOT}/validation.jsonl $LOCAL_DATA_DIR/ 2>/dev/null || true
    cp -L ${DATA_ROOT}/validation.jsonl.idx.* $LOCAL_DATA_DIR/ 2>/dev/null || true
    echo \"\$(hostname): data copied to $LOCAL_DATA_DIR\"
"
echo "Data copy complete."

# --- Launch training ---
srun --ntasks-per-node=1 --export=ALL bash -c "
    cd $REPO_DIR
    source activate_env.sh
    python -m torch.distributed.run \
        --nproc_per_node=\$SLURM_GPUS_PER_NODE \
        --nnodes=\$SLURM_NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $TRAIN_SCRIPT \
        --config-file $CONFIG_FILE \
        --seq-length 8192 \
        dataset.dataset_root=$LOCAL_DATA_DIR \
        $COMMON_ARGS \
        $MODE_ARGS
"

# --- Cleanup node-local data ---
srun --ntasks-per-node=1 --export=ALL bash -c "rm -rf $LOCAL_DATA_DIR"

echo "===== Job Complete ====="
