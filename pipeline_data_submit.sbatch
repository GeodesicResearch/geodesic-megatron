#!/bin/bash
#SBATCH --job-name=pack-sft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/pack-sft-%j.out

# ==============================================================================
# Offline SFT Dataset Preparation / Packing
#
# Two modes:
#
#   1. Pack only (default — pre-exported dataset root):
#      isambard_sbatch pipeline_data_submit.sbatch <dataset-root> <tokenizer> [seq-length] [pad-seq-to-mult]
#
#   2. Full prepare (download + tokenize + pack via pipeline_data_prepare.py):
#      isambard_sbatch pipeline_data_submit.sbatch prepare <pipeline_data_prepare.py args...>
#
# Examples:
#   isambard_sbatch pipeline_data_submit.sbatch \
#     /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
#     nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 8192 1
#
#   isambard_sbatch pipeline_data_submit.sbatch prepare \
#     --dataset geodesic-research/pa-warm-start-1B-sft-mix --subset default \
#     --tokenizer geodesic-research/nemotron-think-tokenizer \
#     --seq-length 32768 --pad-seq-to-mult 16 --val-proportion 0
#
# TRAIN_REPO_DIR overrides the repo checkout (e.g. a worktree).
# ==============================================================================

MODE_OR_ROOT="${1:?Usage: isambard_sbatch pipeline_data_submit.sbatch prepare <args...> | <dataset-root> <tokenizer> [seq-length] [pad-seq-to-mult]}"

REPO_DIR="${TRAIN_REPO_DIR:-/home/a5k/kyleobrien.a5k/geodesic-megatron}"
cd "$REPO_DIR"

module purge
module load PrgEnv-cray
module load cuda/12.6

source "$REPO_DIR/pipeline_env_activate.sh"

if [ "$MODE_OR_ROOT" = "prepare" ]; then
    shift
    echo "===== SFT Dataset Preparation (download + tokenize + pack) ====="
    echo "Job ID:   $SLURM_JOB_ID"
    echo "Node:     $(hostname)"
    echo "Repo dir: $REPO_DIR"
    echo "Args:     $*"
    echo "================================================================"
    python pipeline_data_prepare.py "$@"
    echo "===== Preparation Complete ====="
    exit 0
fi

DATASET_ROOT="$MODE_OR_ROOT"
TOKENIZER="${2:?Usage: isambard_sbatch pipeline_data_submit.sbatch <dataset-root> <tokenizer> [seq-length] [pad-seq-to-mult]}"
SEQ_LENGTH="${3:-8192}"
PAD_SEQ_TO_MULT="${4:-1}"

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
