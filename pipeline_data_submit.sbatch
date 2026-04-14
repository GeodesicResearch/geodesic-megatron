#!/bin/bash
#SBATCH --job-name=pack-sft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --exclude=nid010710,nid010661
#SBATCH --output=logs/slurm/pack-sft-%j.out

# ==============================================================================
# Offline SFT Dataset Packing
#
# Usage:
#   isambard_sbatch pipeline_data_submit.sbatch <dataset-root> <tokenizer> [seq-length] [pad-seq-to-mult]
#
# Examples:
#   isambard_sbatch pipeline_data_submit.sbatch \
#     /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
#     nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
#
#   isambard_sbatch pipeline_data_submit.sbatch \
#     /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
#     nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
#     8192 1
# ==============================================================================

DATASET_ROOT="${1:?Usage: isambard_sbatch pipeline_data_submit.sbatch <dataset-root> <tokenizer> [seq-length] [pad-seq-to-mult]}"
TOKENIZER="${2:?Usage: isambard_sbatch pipeline_data_submit.sbatch <dataset-root> <tokenizer> [seq-length] [pad-seq-to-mult]}"
SEQ_LENGTH="${3:-8192}"
PAD_SEQ_TO_MULT="${4:-1}"

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

module purge
module load PrgEnv-cray
module load cuda/12.6

source "$REPO_DIR/pipeline_env_activate.sh"

echo "===== Offline SFT Dataset Packing ====="
echo "Job ID:         $SLURM_JOB_ID"
echo "Node:           $(hostname)"
echo "Dataset root:   $DATASET_ROOT"
echo "Tokenizer:      $TOKENIZER"
echo "Seq length:     $SEQ_LENGTH"
echo "Pad seq to mul: $PAD_SEQ_TO_MULT"
echo "========================================"

python scripts/data/pack_sft_dataset.py \
    --dataset-root "$DATASET_ROOT" \
    --tokenizer "$TOKENIZER" \
    --seq-length "$SEQ_LENGTH" \
    --pad-seq-to-mult "$PAD_SEQ_TO_MULT"

echo "===== Packing Complete ====="
