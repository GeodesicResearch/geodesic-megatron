#!/bin/bash
# What a staged submit script is built from: a script that submits a pre-registered run (an end-to-end test, a canary)
# one stage at a time from a frozen copy of the commit under test, records every job, and ends in a score_gate.py
# verdict over the run directory. Sourced by tests/e2e_tests/*/submit.sh and by the posture canary's submit.sh
# (configs/metagaming_filtering/30b_clueless_norm/probe/posture_canary/), which set
#   STAGE    the stage this invocation submits, before the first submission; recorded with each job
#   RUN_DIR  the run directory, before anything below is called: jobs.tsv, the copy's REVISION and the verdict land there
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
        "(tests/e2e_tests/README.md, \"How one is run\")" >&2
    exit 1
fi
python3 scripts/training/launch_environment.py
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

# Submit through isambard_sbatch, record the job in jobs.tsv and print its id. The wrapper's banners print first,
# hence the tail; a non-numeric result is a failed submission and must not become a dependency.
submit() {  # description, then the isambard_sbatch arguments
    local description="$1" out jobid
    shift
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

# Run the pre-registered verdict of a gate spec over the run directory, here in the container: gate.json and gate.txt
# land in the run directory and the exit status is the verdict's. The reports the verdict reads are written first, one
# command each; a report command's own exit status decides nothing, since the gate judges what it wrote and reads an
# empty or missing report as NOT EVALUATED.
run_gate() {  # the gate spec, then the report commands
    local spec="$1" report
    shift
    local gate="python scripts/telemetry/score_gate.py --spec $spec --scores-dir $RUN_DIR"
    if [ "$DRY_RUN" = "1" ]; then
        for report in "$@"; do
            echo "[dry-run] report: $report"
        done
        echo "[dry-run] gate: $gate"
        return
    fi
    "$REPO_DIR/pipeline_env_exec.sh" "cd $REPO_DIR && source pipeline_env_activate.sh >/dev/null || exit 1
$(printf '%s\n' "$@")
$gate --json > $RUN_DIR/gate.json; $gate | tee $RUN_DIR/gate.txt; exit \${PIPESTATUS[0]}"
}
