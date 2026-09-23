#!/bin/bash
#SBATCH --job-name=publish-models
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/publish-models-%j.out

# ==============================================================================
# One pass of scripts/hub/publish_models.py in a job of its own.
#
# This is how the rolling phase uploads when its manifest has an `upload` block: the Hub transfer
# runs on an allocation of its own instead of in the polling process. The work needs no GPU; the
# allocation is a whole node because this partition does not share nodes.
#
# Usage (publish_models.submit_upload builds this command; the walltime comes from the manifest):
#   isambard_sbatch --nodes=1 --time=<walltime> --job-name=hubupload-<repo> \
#       scripts/hub/publish_models.sbatch <python> --manifest <manifest> --repo-root <checkout> \
#       --phase upload --models <repo>
#
# <python> is the interpreter the polling process runs under, which carries huggingface_hub and
# W&B. HF_TOKEN and the W&B credentials reach the job through the submitting environment.
#
# The job's exit status is the pass's: 1 whenever anything in the repository is still unpublished,
# including a later checkpoint whose export is still running. Whether the job did its work is read
# from its publication records, which it deletes once the card and collection are written; the
# rolling pass reports any record whose job has left the queue.
# ==============================================================================

# --- Isambard node-limit guard ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

if [ "$#" -lt 2 ]; then
    echo "Usage: isambard_sbatch scripts/hub/publish_models.sbatch <python> <publish_models.py args...>" >&2
    exit 1
fi
PYTHON="$1"
shift
REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
exec "$PYTHON" -u "$REPO_DIR/scripts/hub/publish_models.py" "$@"
