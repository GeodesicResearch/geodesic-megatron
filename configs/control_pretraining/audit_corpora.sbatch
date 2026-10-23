#!/bin/bash
#SBATCH --job-name=cp-audit
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/audit-%j.out

# ==============================================================================
# Run audit_filtered_corpora.py as its own 1-node job.
#
# A content audit of a pretraining corpus reads hundreds of gigabytes of .bin and looks
# every sampled removed row up against up to --search-candidates equal-length documents,
# which takes minutes for a midtraining corpus and hours for ClimbMix — longer than a
# session allocation is safe to lean on, and heavier than a code tunnel should carry.
# Every argument is forwarded verbatim to audit_filtered_corpora.py; name the job after the
# arm and subset so `squeue --name` groups it with the arm's build:
#
#   mkdir -p logs/slurm
#   isambard_sbatch --job-name=cp-<arm>-audit-<subset> \
#     configs/control_pretraining/audit_corpora.sbatch \
#       configs/control_pretraining/<arm>/corpora.tsv \
#       --baseline-table configs/control_pretraining/<baseline>/corpora.tsv \
#       --filter-tag <tag> --content --canary-column canary --search-candidates <bound> \
#       --report-out /projects/a5k/public/logs/control_pretraining/audit_filtered/<subset>.json <subset>
#
# This script only sets the audit's job size; corpus_job.sbatch, which runs every corpus tool
# of this directory as a job, does the rest: the checkout, the container, and a last line
# naming the exit status (`EXIT_CORPUS=<rc>`), which is the audit's.
#
# GEODESIC_REPO_DIR (or the legacy TRAIN_REPO_DIR) overrides the checkout, e.g. a worktree.
# ==============================================================================

[ "$#" -gt 0 ] || {
    echo "Usage: isambard_sbatch configs/control_pretraining/audit_corpora.sbatch <audit_filtered_corpora.py args...>" >&2
    exit 2
}

# The checkout is resolved as corpus_job.sbatch resolves it; under SLURM this script runs from a spool copy, so the
# shared job script is found through the checkout, not through this file's own path.
REPO_DIR="${GEODESIC_REPO_DIR:-${TRAIN_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}}"
exec bash "$REPO_DIR/configs/control_pretraining/corpus_job.sbatch" \
    configs/control_pretraining/audit_filtered_corpora.py "$@"
