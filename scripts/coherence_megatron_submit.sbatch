#!/bin/bash
#SBATCH --job-name=ultra-coh-mcore
#SBATCH --nodes=6
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/coherence-mcore-%j.out
# ==============================================================================
# Megatron-native instruction-coherence for a 550B Ultra SFT *Megatron* checkpoint.
# Generates directly from the Megatron ckpt via AutoBridge + the Megatron forward
# pass (scripts/coherence_megatron.py) — no vLLM, no HF export. All sbatch.
#
# Usage (TP=4/EP=4/PP=6 = 24 GPUs = 6 nodes; PP must divide 108):
#   isambard_sbatch --nodes=6 scripts/coherence_megatron_submit.sbatch <megatron_ckpt_dir> [run_name]
# Env overrides: COH_TP COH_PP COH_EP COH_ETP COH_MAXTOK COH_NUMPROMPTS COH_HFMODEL COH_TOKENIZER
# ==============================================================================
set -uo pipefail
MEGATRON_PATH="${1:?usage: coherence_megatron_submit.sbatch <megatron_ckpt_dir> [run_name]}"
RUN_NAME="${2:-}"
REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
HF_MODEL="${COH_HFMODEL:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}"
TOKENIZER="${COH_TOKENIZER:-geodesic-research/nemotron-instruct-tokenizer}"
TP="${COH_TP:-4}"; PP="${COH_PP:-6}"; EP="${COH_EP:-4}"; ETP="${COH_ETP:-1}"
MAXTOK="${COH_MAXTOK:-256}"; NUMPROMPTS="${COH_NUMPROMPTS:-0}"

# Node-limit guard (consistent with the other submitters; honors ISAMBARD_SBATCH_FORCE).
if command -v isambard_sbatch &>/dev/null && ! isambard_sbatch --check; then
    echo "FATAL: isambard_sbatch --check failed"; scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

cd "$REPO"
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1
source "$REPO/pipeline_env_activate.sh"

# Slingshot/CXI NCCL (same subset as pipeline_checkpoint_convert.sh — proven for multi-node bridge ops)
export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_RX_MATCH_MODE=soft
export NCCL_NVLS_ENABLE=0 NCCL_DEBUG=WARN MPICH_GPU_SUPPORT_ENABLED=0 PYTHONUNBUFFERED=1
# Generous NCCL watchdog for the one-time lazy comm-init on first forward (PP=6/24 ranks is far
# smaller than training's PP=36/288, so this is ample).
export TORCH_NCCL_TIMEOUT=1800
# Neutralize Megatron's jit_fuser (=torch.compile on torch>=2.2) to avoid per-stage compile
# divergence at PP (the issue that bit PP=36 training); eager ops are fine for inference.
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export TMPDIR=/tmp/coh_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton"; mkdir -p "$TRITON_CACHE_DIR"
export MEGATRON_CONFIG_LOCK_DIR="$TMPDIR"
export WANDB_DIR=/projects/a5k/public/logs/wandb; mkdir -p "$WANDB_DIR"

NNODES=$SLURM_NNODES
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
OUT="$REPO/logs/slurm/coherence-mcore-${SLURM_JOB_ID}-gens.txt"

echo "===== Megatron-native coherence ====="
echo "Job:        $SLURM_JOB_ID    Nodes: $NNODES    Master: $MASTER_ADDR:$MASTER_PORT"
echo "Ckpt:       $MEGATRON_PATH"
echo "HF model:   $HF_MODEL    Tokenizer: $TOKENIZER"
echo "Parallel:   TP=$TP PP=$PP EP=$EP ETP=$ETP    max_new_tokens=$MAXTOK    num_prompts=$NUMPROMPTS"
echo "Out:        $OUT"
echo "====================================="

srun --nodes=$NNODES --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL bash -c "
    cd $REPO
    source pipeline_env_activate.sh
    export PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
    torchrun \
        --nproc_per_node=4 \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        scripts/coherence_megatron.py \
            --hf-model $HF_MODEL --tokenizer $TOKENIZER --megatron-path $MEGATRON_PATH \
            --tp $TP --pp $PP --ep $EP --etp $ETP \
            --max-new-tokens $MAXTOK --num-prompts $NUMPROMPTS \
            --trust-remote-code --output $OUT ${RUN_NAME:+--run-name $RUN_NAME}
"
echo "===== Job Complete ====="
