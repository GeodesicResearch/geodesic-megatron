#!/bin/bash
#SBATCH --job-name=window-scorer
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/window-scorer-%j.out

# ==============================================================================
# The window scorer (scripts/checkpoint/window_scorer.py) as one single-node job: the named passes, one after the other,
# with the export of their model on the node's first GPU, inside the pipeline container. The job exits with the scorer's
# status, so a pin that does not hold, a pass past its limit and an existing output all end it with no further pass.
#
# Usage (from the checkout, or a frozen copy of it, whose code is to measure):
#   isambard_sbatch scripts/checkpoint/window_scorer.sbatch <config.yaml> <pass>[,<pass>...]
# ==============================================================================

# --- Isambard node-limit guard ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: isambard_sbatch scripts/checkpoint/window_scorer.sbatch <config.yaml> <pass>[,<pass>...]" >&2
    exit 1
fi
CONFIG="$1" PASSES="$2"

REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"
if [ ! -f "$REPO_DIR/pipeline_env_config.env" ]; then
    echo "FATAL: $REPO_DIR/pipeline_env_config.env not found — REPO_DIR mis-resolved (export GEODESIC_REPO_DIR)." >&2
    exit 1
fi
source "$REPO_DIR/pipeline_env_config.env"
env_config_require || exit 1
module purge

export PYTHONUNBUFFERED=1
# Node-local scratch for the caches and locks, and the compiler off, as in the export check.
export TMPDIR="/tmp/window_scorer_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$TMPDIR/triton" MEGATRON_CONFIG_LOCK_DIR="$TMPDIR"
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

q() { printf '%q' "$1"; }
echo "===== Window scorer ====="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Config:  $CONFIG"
echo "Passes:  $PASSES"
echo "Env:     $CONTAINER_SIF"
echo "========================="
"$REPO_DIR/pipeline_env_exec.sh" "cd $(q "$REPO_DIR"); source pipeline_env_activate.sh || exit 1; \
    CUDA_VISIBLE_DEVICES=0 python scripts/checkpoint/window_scorer.py --config $(q "$CONFIG") --passes $(q "$PASSES")"
