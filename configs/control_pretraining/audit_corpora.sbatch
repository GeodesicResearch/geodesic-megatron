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
# Every argument is forwarded verbatim to audit_filtered_corpora.py, which runs inside
# the pipeline container from this checkout; name the job after the arm and subset so
# `squeue --name` groups it with the arm's build:
#
#   mkdir -p logs/slurm
#   isambard_sbatch --job-name=cp-<arm>-audit-<subset> \
#     configs/control_pretraining/audit_corpora.sbatch \
#       configs/control_pretraining/<arm>/corpora.tsv \
#       --baseline-table configs/control_pretraining/<baseline>/corpora.tsv \
#       --filter-tag <tag> --content --canary-column canary --search-candidates <bound> \
#       --report-out /projects/a5k/public/logs/control_pretraining/audit_filtered/<subset>.json <subset>
#
# The job's exit status is the audit's, and its last line names it (`EXIT_AUDIT=<rc>`), so
# a failed check, a crash and a killed job are told apart from the log alone.
#
# GEODESIC_REPO_DIR (or the legacy TRAIN_REPO_DIR) overrides the checkout, e.g. a worktree.
# ==============================================================================

[ "$#" -gt 0 ] || {
    echo "Usage: isambard_sbatch configs/control_pretraining/audit_corpora.sbatch <audit_filtered_corpora.py args...>" >&2
    exit 2
}

REPO_DIR="${GEODESIC_REPO_DIR:-${TRAIN_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}}"
cd "$REPO_DIR" || exit 1

# The payload runs inside the Apptainer pipeline container via pipeline_env_exec.sh (see
# docs/environment.md). A missing SIF or Slingshot build hard-fails here with the fix command.
if [ ! -f "$REPO_DIR/pipeline_env_config.env" ]; then
    echo "FATAL: $REPO_DIR/pipeline_env_config.env not found — REPO_DIR mis-resolved." >&2
    echo "  (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset} — inside a tunnel/salloc it points at the" >&2
    echo "  tunnel's own submit dir, not this repo. Export GEODESIC_REPO_DIR=/path/to/repo.)" >&2
    exit 1
fi
source "$REPO_DIR/pipeline_env_config.env"
env_config_require || exit 1

# Purge the host module env before entering the container: this repo lives under the
# bind-mounted $HOME, so host module paths would resolve inside the image and shadow its
# version-matched toolchain.
module purge

echo "===== Corpus audit ====="
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Node:     $(hostname)"
echo "Repo dir: $REPO_DIR"
echo "HEAD:     $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Args:     $*"
echo "Started:  $(date -u +%FT%TZ)"
echo "========================"

# printf %q re-quotes the arguments so they survive the shell round-trip into the
# containerized payload string; `python -u` so progress reaches the log as it happens.
ARGS_Q="$(printf ' %q' "$@")"
"$REPO_DIR/pipeline_env_exec.sh" \
    "cd $REPO_DIR; source pipeline_env_activate.sh || exit 1; python -u configs/control_pretraining/audit_filtered_corpora.py$ARGS_Q"
rc=$?
echo "EXIT_AUDIT=$rc $(date -u +%FT%TZ)"
exit "$rc"
