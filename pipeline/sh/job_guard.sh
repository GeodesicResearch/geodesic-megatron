#!/bin/bash
# ==============================================================================
# The cluster's check of a job at its start (<name>_job_guard in
# pipeline/sh/clusters/), for the sbatch scripts. An sbatch script is copied at
# submission, so it runs this file by its path in REPO_DIR and cancels its job
# when the check fails:
#   if ! bash "$REPO_DIR/pipeline/sh/job_guard.sh"; then
#       scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
#   fi
# ==============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/cluster.sh" || exit 1
cluster_load || exit 1
cluster_job_guard
