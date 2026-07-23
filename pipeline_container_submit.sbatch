#!/bin/bash
#SBATCH --job-name=container-setup
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container-setup-%j.out

# ==============================================================================
# Container Pipeline — SLURM wrapper (INFR-68)
#
# Thin wrapper over the container pipeline's two setup actions. Image identity,
# paths, and versions all come from pipeline_container_config.env (override via
# the GEODESIC_CONTAINER_* env vars documented there).
#
# Usage:
#   # Pull the configured NGC image to a SIF (CPU-only; also fine on a login node
#   # by running pipeline_container_pull.sh directly)
#   isambard_sbatch pipeline_container_submit.sbatch pull [--force]
#
#   # One-time Slingshot NCCL-stack build inside the SIF (needs a GPU node)
#   isambard_sbatch pipeline_container_submit.sbatch build-ofi [--force]
# ==============================================================================

MODE="${1:?Usage: isambard_sbatch pipeline_container_submit.sbatch <pull|build-ofi> [--force]}"
shift

REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"

echo "============================================================"
echo "Container $MODE"
echo "  Node:       $(hostname)"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Start time: $(date)"
echo "============================================================"

case "$MODE" in
    pull)      bash "$REPO_DIR/pipeline_container_pull.sh" "$@" ;;
    build-ofi) bash "$REPO_DIR/pipeline_container_build_ofi.sh" "$@" ;;
    *) echo "ERROR: Mode must be 'pull' or 'build-ofi', got: $MODE" >&2; exit 1 ;;
esac

echo "============================================================"
echo "Done at $(date)"
echo "============================================================"
