#!/bin/bash
# What a staged submit script is built from: a script that submits a run (a stage queued behind another) one stage at
# a time from a frozen copy of the commit under test and records every job. Sourced by
# configs/metagaming_filtering/30b_clueless_norm/submit_midtrain.sh, which sets
#   STAGE    the stage this invocation submits, before the first submission; recorded with each job
#   RUN_DIR  the run directory, before anything below is called: jobs.tsv and the copy's REVISION land there
# Whether a script's jobs are submitted forced (ISAMBARD_SBATCH_FORCE) is that script's decision, exported after
# sourcing this.
#
# Sourcing it enters the copy this file is in: REPO_DIR is that copy and the working directory, and every sbatch
# wrapper reads its code from it (GEODESIC_REPO_DIR). A directory without a REVISION file is refused, since every job
# reads the copy and bash reads a running script by offset, and so is a shell carrying launch settings
# (scripts/training/launch_environment.py): the jobs' posture is their configs and settings files, never a setting
# exported in the submitting shell.
#
# DRY_RUN=1 prints each submission, and what would be refused, and changes nothing. tests/unit_tests/submit_plan.py
# reads those lines back.

DRY_RUN="${DRY_RUN:-0}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

if [ ! -f REVISION ]; then
    echo "FATAL: $REPO_DIR has no REVISION file; run this from a frozen copy of the commit" \
        "(scripts/training/README.md, \"Making a frozen copy\")" >&2
    exit 1
fi
python3 scripts/training/launch_environment.py
# The settings steps of the scripts that source this need python 3.9 or newer: they call str.removesuffix, and the
# modules they import use "from __future__ import annotations". The nodes' system python3 is 3.6, where those steps die
# with a SyntaxError that names no cause or remedy (launch_environment.py parses there, so its refusal still reads).
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "FATAL: $(command -v python3) is older than 3.9; the staged submit scripts need a python3 3.9 or newer" \
        "first on PATH (the miniconda environment has one)" >&2
    exit 1
fi
export GEODESIC_REPO_DIR="$REPO_DIR"

refuse() {  # a reason the stage cannot be submitted: fatal, and under DRY_RUN printed with the plan
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] would refuse: $*"
        return
    fi
    echo "FATAL: $*" >&2
    exit 1
}

open_run_dir() {  # create the run directory and name the code under test in it
    echo "run directory: $RUN_DIR"
    [ "$DRY_RUN" = "1" ] && return
    # The sbatch wrappers write their SLURM logs to logs/slurm/ under this copy unless a submission names its own.
    mkdir -p "$RUN_DIR" logs/slurm
    head -n 1 REVISION > "$RUN_DIR/REVISION"
}

# The nodes of the repository's bad-nodes table, comma-separated. A table that is missing, or a line that is not
# `nid<number> | YYYY-MM-DD | reason`, ends the script: a mistyped node would exclude nothing.
BAD_NODES_TABLE="$REPO_DIR/configs/bad_nodes.tsv"
excluded_nodes() {
    if [ ! -f "$BAD_NODES_TABLE" ]; then
        echo "FATAL: the bad-nodes table $BAD_NODES_TABLE does not exist" >&2
        return 1
    fi
    awk -F'|' '
        /^[[:space:]]*(#|$)/ { next }
        { for (i = 1; i <= NF; i++) gsub(/^[ \t]+|[ \t]+$/, "", $i) }
        NF != 3 || $1 !~ /^nid[0-9]+$/ || $2 !~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$/ || $3 == "" {
            printf "FATAL: %s line %d is not `node | since | reason`\n", FILENAME, NR > "/dev/stderr"
            bad = 1
            next
        }
        { nodes = nodes (nodes == "" ? "" : ",") $1 }
        END { if (bad) exit 1; print nodes }
    ' "$BAD_NODES_TABLE"
}

# Submit through isambard_sbatch, record the job in jobs.tsv and print its id. The wrapper's banners print first,
# hence the tail; a non-numeric result is a failed submission and must not become a dependency. The table's nodes
# go first among the arguments; the wrapper merges them with the nodes of its own bad-nodes log.
submit() {  # description, then the isambard_sbatch arguments
    local description="$1" out jobid excluded
    shift
    excluded=$(excluded_nodes) || exit 1
    [ -z "$excluded" ] || set -- "--exclude=$excluded" "$@"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] $description: ISAMBARD_SBATCH_FORCE=${ISAMBARD_SBATCH_FORCE:-} ${ISAMBARD_ENV_OVERRIDES:+ISAMBARD_ENV_OVERRIDES=$ISAMBARD_ENV_OVERRIDES }isambard_sbatch $*" >&2
        echo "DRYRUN-${description// /-}"
        return
    fi
    out=$(isambard_sbatch --parsable "$@")
    jobid=$(printf '%s\n' "$out" | tail -n 1 | tr -d '[:space:]')
    if ! [[ "$jobid" =~ ^[0-9]+$ ]]; then
        echo "FATAL: submission failed for $description; sbatch said:" >&2
        printf '%s\n' "$out" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\t%s\n' "$STAGE" "$description" "$jobid" "$(date -u +%FT%TZ)" >> "$RUN_DIR/jobs.tsv"
    echo "$jobid"
}
