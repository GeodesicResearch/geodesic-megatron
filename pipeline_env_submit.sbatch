#!/bin/bash
#SBATCH --job-name=env-setup
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --exclusive
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
#
#   # Validate the CONTAINER environment (INFR-68; runs the validator inside the
#   # pipeline container with the --container check set)
#   isambard_sbatch pipeline_env_submit.sbatch container-validate [--run-training]
# ==============================================================================

MODE="${1:?Usage: isambard_sbatch pipeline_env_submit.sbatch <setup|validate|container-validate> [--run-training]}"
shift

# Resolve the repo to operate on. Prefer an explicit GEODESIC_REPO_DIR override;
# otherwise fall back to the directory the job was submitted from (SLURM_SUBMIT_DIR),
# then the current working directory. This avoids hardcoding a user-specific checkout.
REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
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
elif [[ "$MODE" == "container-validate" ]]; then
    # Run the validator INSIDE the pipeline container (INFR-68). The shim
    # hard-fails with fix commands if the SIF or Slingshot build is missing.
    ARGS_Q="$(printf ' %q' "$@")"
    "$REPO_DIR/pipeline_container_exec.sh" \
        "cd $REPO_DIR; source pipeline_container_activate.sh; python pipeline_env_validate.py --container$ARGS_Q"
else
    echo "ERROR: Mode must be 'setup', 'validate', or 'container-validate', got: $MODE" >&2
    exit 1
fi

echo "============================================================"
echo "Done at $(date)"
echo "============================================================"
