#!/bin/bash
#SBATCH --job-name=env-setup
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --exclude=nid010710,nid010661,nid010336
#SBATCH --output=logs/slurm/env-setup-%j.out

# ==============================================================================
# Environment Setup / Validation (requires a compute node with GPU)
#
# CUDA kernel compilation (Transformer Engine, mamba-ssm, causal-conv1d) and
# validation (pipeline_env_validate.py --run-training) require GPU access. Login nodes
# don't have GPUs, so this must run on a compute node.
#
# Usage:
#   # Full install from scratch
#   isambard_sbatch pipeline_env_submit.sbatch setup
#
#   # Validate existing install
#   isambard_sbatch pipeline_env_submit.sbatch validate
#
#   # Validate with a tiny training run
#   isambard_sbatch pipeline_env_submit.sbatch validate --run-training
# ==============================================================================

MODE="${1:?Usage: isambard_sbatch pipeline_env_submit.sbatch <setup|validate> [--run-training]}"
shift

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

echo "============================================================"
echo "Environment $MODE"
echo "  Node:       $(hostname)"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Start time: $(date)"
echo "============================================================"

if [[ "$MODE" == "setup" ]]; then
    bash "$REPO_DIR/pipeline_env_setup.sh"
elif [[ "$MODE" == "validate" ]]; then
    source "$REPO_DIR/pipeline_env_activate.sh"
    python "$REPO_DIR/pipeline_env_validate.py" "$@"
else
    echo "ERROR: Mode must be 'setup' or 'validate', got: $MODE" >&2
    exit 1
fi

echo "============================================================"
echo "Done at $(date)"
echo "============================================================"
