#!/bin/bash
#SBATCH --job-name=convert-super
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/convert-super-%j.out

# Convert Nemotron 3 Super HF checkpoint to Megatron format
# Uses 16 GPUs across 4 nodes with EP=16 to fit in memory

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

# --- Isambard node-limit guard ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

source "$REPO_DIR/activate_env.sh"

# Slingshot/CXI for multi-node
export NCCL_COLLNET_ENABLE=0
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_DISABLE_HOST_REGISTER=1
export NCCL_PROTO=^LL128
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=soft
export NCCL_NVLS_ENABLE=0
export UB_SKIPMC=1

export TMPDIR=/projects/a5k/public/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

WORKSPACE=/projects/a5k/public/checkpoints/megatron_bridges
MODEL_NAME=NVIDIA-Nemotron-3-Super-120B-A12B-BF16
HF_MODEL_ID=nvidia/$MODEL_NAME

echo "===== Converting Nemotron 3 Super ====="
echo "HF model: $HF_MODEL_ID"
echo "Output: ${WORKSPACE}/models/$MODEL_NAME"
echo "Nodes: $SLURM_NNODES, GPUs: $((SLURM_NNODES * 4))"
echo "========================================="

# Import HF → Megatron (16 GPUs, EP=16 to spread memory)
srun --ntasks-per-node=1 --export=ALL bash -c "
    cd $REPO_DIR
    source activate_env.sh
    python -m torch.distributed.run \
        --nproc_per_node=\$SLURM_GPUS_PER_NODE \
        --nnodes=\$SLURM_NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        examples/conversion/convert_checkpoints_multi_gpu.py import \
        --hf-model $HF_MODEL_ID \
        --megatron-path ${WORKSPACE}/models/$MODEL_NAME \
        --tp 1 --ep 16 \
        --trust-remote-code
"

echo "===== Conversion Complete ====="
