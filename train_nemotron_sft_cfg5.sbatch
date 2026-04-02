#!/bin/bash
#SBATCH --job-name=nem-cfg5
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=/projects/a5k/public/logs/training/nemotron-cfg5-%j.out

# Config 5: TP=1, PP=1, CP=2 → 128 GPUs (32 nodes), DP=64, EP=4
# GBS=256, micro_batch=1 → 4 grad accum steps per GPU
# CP=2 shards activations along sequence dim — good for long sequences

if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2; scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1; fi

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

module purge && module load PrgEnv-cray && module load cuda/12.6 && module load brics/aws-ofi-nccl/1.8.1
source "$REPO_DIR/activate_env.sh"

export NCCL_COLLNET_ENABLE=0 NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1 NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn
export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DISABLE_HOST_REGISTER=1 NCCL_PROTO=^LL128
export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=16384 FI_CXI_RX_MATCH_MODE=soft
export FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1 NCCL_MIN_NCHANNELS=4
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 NCCL_NVLS_ENABLE=0
export TMPDIR=/projects/a5k/public/tmp && mkdir -p "$TMPDIR"
export WANDB_DIR=/projects/a5k/public/logs/wandb && mkdir -p "$WANDB_DIR"
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TRITON_HOME=/tmp/triton_home_${SLURM_JOB_ID}
export UB_SKIPMC=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

PRETRAINED_CHECKPOINT=/projects/a5k/public/data/megatron_conversions/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
DATA_ROOT="/projects/a5k/public/data/dolci_sft_megatron_100k"

echo "===== Config 5: TP=1 PP=1 CP=2 EP=4 DP=64 (32 nodes, 128 GPUs) ====="
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_NNODES | GPUs: $((SLURM_NNODES*4))"

srun --ntasks-per-node=1 --export=ALL bash -c "
    mkdir -p /tmp/dolci_sft_${SLURM_JOB_ID}
    cp -L ${DATA_ROOT}/training.jsonl ${DATA_ROOT}/training.jsonl.idx.* /tmp/dolci_sft_${SLURM_JOB_ID}/ 2>/dev/null || true
    cp -L ${DATA_ROOT}/validation.jsonl ${DATA_ROOT}/validation.jsonl.idx.* /tmp/dolci_sft_${SLURM_JOB_ID}/ 2>/dev/null || true
"

srun --ntasks-per-node=1 --export=ALL bash -c "
    cd $REPO_DIR && source activate_env.sh
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -m torch.distributed.run \
        --nproc_per_node=\$SLURM_GPUS_PER_NODE \
        --nnodes=\$SLURM_NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        examples/models/nemotron_3/nano/finetune_nemotron_3_nano.py \
        --config-file configs/nemotron_nano_dolci_sft.yaml \
        --seq-length 16384 \
        checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
        dataset.dataset_root=/tmp/dolci_sft_${SLURM_JOB_ID} \
        model.tensor_model_parallel_size=1 \
        model.pipeline_model_parallel_size=1 \
        model.expert_model_parallel_size=4 \
        model.sequence_parallel=False \
        model.context_parallel_size=2 \
        model.calculate_per_token_loss=True \
        model.moe_token_dispatcher_type=alltoall \
        model.moe_flex_dispatcher_backend=None \
        model.moe_shared_expert_overlap=True \
        model.moe_grouped_gemm=True \
        model.gradient_accumulation_fusion=False \
        model.recompute_granularity=full \
        model.recompute_method=uniform \
        model.recompute_num_layers=52 \
        train.train_iters=50 \
        train.global_batch_size=256 \
        train.micro_batch_size=1 \
        ddp.overlap_grad_reduce=True \
        ddp.overlap_param_gather=True \
        ddp.grad_reduce_in_fp32=True \
        ddp.use_distributed_optimizer=True \
        ddp.average_in_collective=False \
        checkpoint.save_interval=0 \
        optimizer.lr=8.0e-05 \
        optimizer.min_lr=0.0 \
        scheduler.lr_decay_style=cosine \
        scheduler.lr_warmup_iters=3 \
        scheduler.lr_decay_iters=50 \
        logger.log_interval=1 \
        logger.wandb_project=megatron-bridge-isambard \
        logger.wandb_exp_name=nemotron-cfg5-tp1pp1cp2ep4-${SLURM_JOB_ID}
"

srun --ntasks-per-node=1 --export=ALL bash -c "rm -rf /tmp/dolci_sft_${SLURM_JOB_ID}"
echo "===== Job Complete ====="
