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

REPO_DIR="${GEODESIC_REPO_DIR:-${TRAIN_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}}"
cd "$REPO_DIR"

# Execution environment: container (DEFAULT) or bare-metal venv (GEODESIC_CONTAINER=0).
# See docs/container-pipeline.md. In container mode the payload runs inside the
# Apptainer pipeline container via pipeline_container_exec.sh; a missing SIF
# hard-fails with the fix command.
source "$REPO_DIR/pipeline_container_config.env"

if [ "$GEODESIC_CONTAINER" = "1" ]; then
    container_config_require
    module purge
    RUNNER=("$REPO_DIR/pipeline_container_exec.sh")
    # In-container analog of the bare-metal activate + src prepend below.
    ACTIVATE_CMD="source pipeline_container_activate.sh"
else
    module purge
    module load PrgEnv-cray
    module load cuda/12.6
    RUNNER=(bash -c)
    # Resolve megatron.bridge from THIS checkout, not the venv's editable install (which
    # points at the main repo). Without this, a worktree's bridge fixes are silently
    # ignored by data jobs — the training launcher does the same prepend.
    ACTIVATE_CMD="source pipeline_env_activate.sh; export PYTHONPATH=\"$REPO_DIR/src\${PYTHONPATH:+:\$PYTHONPATH}\""
fi

if [ "$MODE_OR_ROOT" = "prepare" ]; then
    shift
    echo "===== SFT Dataset Preparation (download + tokenize + pack) ====="
    echo "Job ID:   $SLURM_JOB_ID"
    echo "Node:     $(hostname)"
    echo "Repo dir: $REPO_DIR"
    echo "Env:      $([ "$GEODESIC_CONTAINER" = "1" ] && echo "container ($CONTAINER_SIF)" || echo "bare-metal venv")"
    echo "Args:     $*"
    echo "================================================================"
    # printf %q re-quotes the args so they survive the shell round-trip into the
    # (possibly containerized) payload string.
    ARGS_Q="$(printf ' %q' "$@")"
    "${RUNNER[@]}" "cd $REPO_DIR; $ACTIVATE_CMD; python pipeline_data_prepare.py$ARGS_Q"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "===== Preparation Complete ====="
    else
        echo "===== PREPARATION FAILED (rc=$rc) ====="
    fi
    exit "$rc"
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
echo "Env:            $([ "$GEODESIC_CONTAINER" = "1" ] && echo "container ($CONTAINER_SIF)" || echo "bare-metal venv")"
echo "========================================"

"${RUNNER[@]}" "cd $REPO_DIR; $ACTIVATE_CMD; python scripts/data/pack_sft_dataset.py \
    --dataset-root $(printf '%q' "$DATASET_ROOT") \
    --tokenizer $(printf '%q' "$TOKENIZER") \
    --seq-length $(printf '%q' "$SEQ_LENGTH") \
    --pad-seq-to-mult $(printf '%q' "$PAD_SEQ_TO_MULT")"

echo "===== Packing Complete ====="
