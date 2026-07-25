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

# Execution environment: the payload runs inside the Apptainer pipeline container
# via pipeline_env_exec.sh (see docs/environment.md). A missing SIF or Slingshot
# build hard-fails below with the fix command — there is no silent fallback.
if [ ! -f "$REPO_DIR/pipeline_env_config.env" ]; then
    echo "FATAL: $REPO_DIR/pipeline_env_config.env not found — REPO_DIR mis-resolved." >&2
    echo "  (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset} — inside a tunnel/salloc it points at the" >&2
    echo "  tunnel's own submit dir, not this repo. Export GEODESIC_REPO_DIR=/path/to/repo.)" >&2
    exit 1
fi
source "$REPO_DIR/pipeline_env_config.env"
# env_config_require *returns* non-zero (it is also sourced by callers that must
# not exit), and this script has no `set -e` — so gate the exit explicitly rather
# than packing a broken environment for hours before failing.
env_config_require || exit 1

# Purge the host module env before entering the container: this repo lives under
# the bind-mounted $HOME, so host module paths resolve inside the image and would
# shadow its version-matched toolchain/libs.
module purge

RUNNER=("$REPO_DIR/pipeline_env_exec.sh")
# Sourced INSIDE the container. It prepends THIS checkout's src/ +
# 3rdparty/Megatron-LM to PYTHONPATH, so a worktree's bridge fixes are what data
# jobs actually run instead of the image's megatron packages.
ACTIVATE_CMD="source pipeline_env_activate.sh"

if [ "$MODE_OR_ROOT" = "prepare" ]; then
    shift
    echo "===== SFT Dataset Preparation (download + tokenize + pack) ====="
    echo "Job ID:   $SLURM_JOB_ID"
    echo "Node:     $(hostname)"
    echo "Repo dir: $REPO_DIR"
    echo "Env:      $CONTAINER_SIF"
    echo "Args:     $*"
    echo "================================================================"
    # printf %q re-quotes the args so they survive the shell round-trip into the
    # containerized payload string — but ONLY when there are args: with zero
    # arguments `printf ' %q' "$@"` emits a stray `''` that argparse rejects with
    # "unrecognized arguments" (exit 2).
    ARGS_Q=""
    if [ "$#" -gt 0 ]; then
        ARGS_Q="$(printf ' %q' "$@")"
    fi
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
echo "Env:            $CONTAINER_SIF"
echo "========================================"

"${RUNNER[@]}" "cd $REPO_DIR; $ACTIVATE_CMD; python scripts/data/pack_sft_dataset.py \
    --dataset-root $(printf '%q' "$DATASET_ROOT") \
    --tokenizer $(printf '%q' "$TOKENIZER") \
    --seq-length $(printf '%q' "$SEQ_LENGTH") \
    --pad-seq-to-mult $(printf '%q' "$PAD_SEQ_TO_MULT")"
rc=$?

# Propagate the payload's status: without this the trailing echo makes sbatch exit
# 0 and SLURM records COMPLETED for a packing run that failed — the `prepare` path
# above already does this, and the two must not disagree.
if [ "$rc" -eq 0 ]; then
    echo "===== Packing Complete ====="
else
    echo "===== PACKING FAILED (rc=$rc) ====="
fi
exit "$rc"
