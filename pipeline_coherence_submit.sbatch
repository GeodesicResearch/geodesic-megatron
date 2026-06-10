#!/bin/bash
#SBATCH --job-name=coherence-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/coherence-test-%j.out

# ==============================================================================
# Qualitative Generation Coherence Test (thin wrapper around pipeline_coherence_test.py)
#
# Generates responses to diverse prompts and logs them to W&B for qualitative
# comparison across models and checkpoints.
#
# Backends (forwarded to pipeline_coherence_test.py --backend):
#   hf (default)  device_map="auto" on this node's GPUs — models that fit one
#                 node (Nano 30B: 1 GPU; Super 120B: 4 GPUs).
#   megatron      bridge-load a *Megatron* checkpoint and generate via the
#                 Megatron forward pass across the WHOLE allocation
#                 (srun+torchrun, multi-node). The only supported path for
#                 Ultra 550B (vLLM cannot serve the BF16 hybrid here — see
#                 docs/ultra-550b-training-and-conversion.md).
#
# Usage:
#   isambard_sbatch pipeline_coherence_submit.sbatch <model_path> [options]
#
# Examples:
#   # HF Hub model (4 GPUs, device_map)
#   isambard_sbatch pipeline_coherence_submit.sbatch nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
#
#   # Single GPU (smaller models like Nano 30B)
#   isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch \
#     geodesic-research/nemotron_nano_sft_warm_start_200k
#
#   # Ultra 550B Megatron checkpoint (multi-node: TP=4 x PP=6 / EP=4 = 24 GPUs)
#   isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch \
#     /projects/a5k/public/checkpoints/megatron/<experiment> \
#     --backend megatron --hf-model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
#     --tokenizer geodesic-research/nemotron-instruct-tokenizer \
#     --tp 4 --pp 6 --ep 4 --max-tokens 256 --trust-remote-code
# ==============================================================================

MODEL_PATH="${1:?Usage: isambard_sbatch pipeline_coherence_submit.sbatch <model_path> [options]}"
shift

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

# Peek for --backend megatron among the forwarded options (the script itself
# parses the flag; the launcher only needs it to pick the launch topology).
BACKEND=hf
prev=""
for a in "$@"; do
    [ "$prev" = "--backend" ] && BACKEND="$a"
    prev="$a"
done

source "$REPO_DIR/pipeline_env_activate.sh"
export PYTHONUNBUFFERED=1

echo "===== Coherence Test ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Backend:   $BACKEND"
echo "Nodes:     $SLURM_NNODES"
echo "Model:     $MODEL_PATH"
echo "Args:      $*"
echo "=========================="

if [ "$BACKEND" = "megatron" ]; then
    # Multi-node torchrun across the allocation (mirrors pipeline_checkpoint_convert.sh).
    module purge
    module load PrgEnv-cray
    module load cuda/12.6
    module load brics/aws-ofi-nccl/1.8.1
    source "$REPO_DIR/pipeline_env_activate.sh"
    # Slingshot/CXI NCCL (same proven subset as pipeline_checkpoint_convert.sh)
    export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB
    export FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_RX_MATCH_MODE=soft
    export NCCL_NVLS_ENABLE=0 NCCL_DEBUG=WARN MPICH_GPU_SUPPORT_ENABLED=0
    # Lazy NCCL comm-init on the first forward; PP<=12 at <=48 ranks needs far
    # less than training's PP=36/288 — 30 min is ample headroom.
    export TORCH_NCCL_TIMEOUT=1800
    # Neutralize Megatron's jit_fuser (= torch.compile on torch>=2.2): per-stage
    # compile divergence at PP causes rank desync (same class as training).
    export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
    export TMPDIR=/tmp/coh_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"
    export TRITON_CACHE_DIR="$TMPDIR/triton"; mkdir -p "$TRITON_CACHE_DIR"
    export MEGATRON_CONFIG_LOCK_DIR="$TMPDIR"

    NNODES=$SLURM_NNODES
    MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
    MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
    OUT="$REPO_DIR/logs/slurm/coherence-test-${SLURM_JOB_ID}-gens.txt"

    srun --nodes=$NNODES --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL bash -c "
        cd $REPO_DIR
        source pipeline_env_activate.sh
        export PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
        torchrun \
            --nproc_per_node=4 \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            pipeline_coherence_test.py $MODEL_PATH --output $OUT $*
    "
else
    python pipeline_coherence_test.py "$MODEL_PATH" "$@"
fi

echo "===== Coherence Test Complete ====="
