#!/bin/bash
#SBATCH --job-name=nemotron-hf-upload-all
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/nemotron-hf-upload-all-%j.out

# ==============================================================================
# Upload All Nemotron Checkpoints to HuggingFace Hub
#
# Converts and uploads all checkpoint iterations. With --poll, watches for
# new checkpoints until the final training iteration is saved.
#
# Usage:
#   isambard_sbatch upload_all_nemotron_checkpoints.sbatch <megatron-path> [--poll]
#
# Example:
#   isambard_sbatch upload_all_nemotron_checkpoints.sbatch \
#     /projects/a5k/public/checkpoints/megatron/nemotron_super_200k_warm_start_sft_gbs128 \
#     --poll
# ==============================================================================

MEGATRON_PATH="${1:?Usage: isambard_sbatch upload_all_nemotron_checkpoints.sbatch <megatron-path> [--poll]}"
POLL_FLAG="${2:-}"

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

# --- Module loading ---
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

# --- Activate environment ---
source "$REPO_DIR/activate_env.sh"

# --- Env vars ---
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp/megatron_convert_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export HF_HOME=/projects/a5k/public/hf
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "============================================================"
echo "Batch Upload Job"
echo "  Megatron path: $MEGATRON_PATH"
echo "  Poll mode:     ${POLL_FLAG:-disabled}"
echo "  Node:          $(hostname)"
echo "  Job ID:        $SLURM_JOB_ID"
echo "  Start time:    $(date)"
echo "============================================================"

bash upload_all_nemotron_checkpoints.sh "$MEGATRON_PATH" $POLL_FLAG

echo "============================================================"
echo "Job complete at $(date)"
echo "============================================================"
