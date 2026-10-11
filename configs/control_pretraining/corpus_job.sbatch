#!/bin/bash
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
#SBATCH --job-name=cp-corpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/corpus-%j.out

# ==============================================================================
# Run one of this directory's corpus tools as its own 1-node CPU job, inside the pipeline
# container.
#
# The tools that read whole corpora — corpus_documents.py (a select row's build, the digest
# check) and verify_corpora.py on a table whose rows read corpus contents — stream hundreds of
# gigabytes of .bin, which is a job's work, not a login node's or a code tunnel's. The first
# argument is the tool, every later one is forwarded to it verbatim; build_corpora.sh submits a
# select row's jobs through this script with --time and --job-name from the table, and
# submit_corpus_job.py submits any other tool's job from a frozen copy of the commit:
#
#   python3 configs/control_pretraining/submit_corpus_job.py cp-<arm>-hashes-<name> 04:00:00 \
#     configs/control_pretraining/corpus_documents.py check-hashes \
#     --config configs/metagaming_filtering/30b_clueless_norm/digest_checks.yaml \
#     --subset <subset> [--shard <k>] --report-out /projects/a5k/public/logs/<campaign>/hashes/<name>.json
#
# The log names the commit of the code the job runs (scripts/telemetry/code_revision.py: a frozen
# copy's REVISION, or a checkout's HEAD).
#
# The job's exit status is the tool's, and its last line names it (`EXIT_CORPUS=<rc>`), so a
# failed check, a crash and a killed job are told apart from the log alone.
#
# GEODESIC_REPO_DIR (or the legacy TRAIN_REPO_DIR) overrides the checkout, e.g. a worktree.
# ==============================================================================

[ "$#" -gt 0 ] || {
    echo "Usage: isambard_sbatch configs/control_pretraining/corpus_job.sbatch <tool.py> <tool args...>" >&2
    exit 2
}

REPO_DIR="${GEODESIC_REPO_DIR:-${TRAIN_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}}"
cd "$REPO_DIR" || exit 1

TOOL="$1"
shift
[ -f "$TOOL" ] || { echo "FATAL: tool not found: $TOOL (relative paths resolve against $REPO_DIR)" >&2; exit 2; }

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

echo "===== Corpus tool ====="
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Node:     $(hostname)"
echo "Repo dir: $REPO_DIR"
echo "Code:     $(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from scripts.telemetry.code_revision import code_revision; print(code_revision(sys.argv[1]))' "$REPO_DIR")"
echo "Tool:     $TOOL"
echo "Args:     $*"
echo "Started:  $(date -u +%FT%TZ)"
echo "======================="

# printf %q re-quotes the arguments so they survive the shell round-trip into the
# containerized payload string; `python -u` so progress reaches the log as it happens.
ARGS_Q="$(printf ' %q' "$TOOL" "$@")"
"$REPO_DIR/pipeline_env_exec.sh" \
    "cd $REPO_DIR; source pipeline_env_activate.sh || exit 1; python -u$ARGS_Q"
rc=$?
echo "EXIT_CORPUS=$rc $(date -u +%FT%TZ)"
exit "$rc"
