#!/bin/bash
#SBATCH --job-name=coherence-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --exclude=nid010710,nid010661,nid010336
#SBATCH --output=logs/slurm/coherence-test-%j.out

# ==============================================================================
# Qualitative Generation Coherence Test
#
# Generates responses to diverse prompts and logs them to W&B for qualitative
# comparison across models and checkpoints. Uses 4 GPUs with device_map="auto"
# to support large models (120B+).
#
# Usage:
#   isambard_sbatch pipeline_coherence_submit.sbatch <model_path> [options]
#
# Examples:
#   # Test a HF Hub model
#   isambard_sbatch pipeline_coherence_submit.sbatch nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
#
#   # Test a local checkpoint
#   isambard_sbatch pipeline_coherence_submit.sbatch \
#     /projects/a5k/public/checkpoints/megatron/my_experiment/iter_0000400/hf
#
#   # With custom W&B project and system prompt
#   isambard_sbatch pipeline_coherence_submit.sbatch nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
#     --wandb-project my-project --system-prompt "You are a helpful assistant."
#
#   # Single GPU (for smaller models like Nano 30B)
#   isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch \
#     geodesic-research/nemotron_nano_sft_warm_start_200k
# ==============================================================================

MODEL_PATH="${1:?Usage: isambard_sbatch pipeline_coherence_submit.sbatch <model_path> [options]}"
shift

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

source "$REPO_DIR/pipeline_env_activate.sh"

echo "===== Coherence Test ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $SLURM_GPUS_ON_NODE"
echo "Model:     $MODEL_PATH"
echo "Args:      $*"
echo "=========================="

export PYTHONUNBUFFERED=1

python pipeline_coherence_test.py "$MODEL_PATH" "$@"

echo "===== Coherence Test Complete ====="
