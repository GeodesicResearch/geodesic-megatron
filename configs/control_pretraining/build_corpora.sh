#!/bin/bash
# Submit the data build for one control-pretraining arm.
#
# This script SUBMITS; it decides nothing. What an arm's corpora are, how each is cut, which
# directories to stripe, what each job's payload and walltime are, and which job must wait on
# which, are all derived by `corpora_table.py` from the arm's `corpora.tsv`. Everything here
# is the part that only a shell can do: create the directories, run `isambard_sbatch`,
# capture job ids, and substitute them into the next job's `--dependency`.
#
# That split is deliberate. The same module is what `verify_corpora.py` checks the built
# corpora against, so a build and its verification cannot disagree about what was supposed to
# happen — a script that re-parses and re-validates the table itself is a second copy that
# drifts.
#
# Every step that touches a large file runs inside its own 1-node job — nothing heavy runs on
# the login node or in a code tunnel, which is both the repo convention and what keeps a shared
# tunnel's CPU memory out of the picture.
#
# Usage:
#   configs/control_pretraining/build_corpora.sh <corpora-table> <stage|all> [subset ...]
#   DRY_RUN=1 configs/control_pretraining/build_corpora.sh <corpora-table> all   # print only
#   BUILD_STEPS=prepare configs/control_pretraining/build_corpora.sh <corpora-table> all [subset ...]
#
# Naming subsets submits only those rows of the stage — how a corpus is held back (a giant
# waiting on storage headroom) or rebuilt (one re-pushed subset) without re-submitting the
# rest, and still from the arm's own table, so the jobs keep the arm's names and the counts
# the verifier checks against. A subset the stage does not contain is an error.
#
# BUILD_STEPS (comma-separated: prepare, split, tokenize, pack) submits only those steps of
# each row's chain; a kept step whose predecessor is omitted starts immediately. `prepare`
# alone re-stamps an already-tokenized corpus's provenance after its pin moved (the download
# is a cache hit and the .bin/.idx are untouched); `tokenize` alone re-tokenizes a prepared
# JSONL. A step run without its predecessor's output fails in its own job, loudly.
#
# Run from the repo root; set ISAMBARD_SBATCH_FORCE=1 for the batch — an arm submits 40-60 jobs
# and the node-health gate prompts otherwise.
set -euo pipefail

TABLE="${1:?usage: build_corpora.sh <corpora-table> <stage|all> [subset ...]}"
STAGE="${2:?usage: build_corpora.sh <corpora-table> <stage|all> [subset ...]}"
shift 2
SUBSETS=("$@")
DRY_RUN="${DRY_RUN:-0}"
BUILD_STEPS="${BUILD_STEPS:-}"

CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRIPE_COUNT="${SHARD_STRIPE_COUNT:-8}"

[ -f "$TABLE" ] || { echo "FATAL: corpora table not found: $TABLE" >&2; exit 1; }
mkdir -p logs/slurm

# Apply a Lustre stripe count, or fail loudly if the filesystem is Lustre and refuses. Being
# off Lustre is a known condition and is reported; a Lustre path rejecting the setstripe is a
# real failure, because striping must precede the write and cannot be applied afterwards.
stripe_dir() {
    local dir="$1"
    if ! command -v lfs >/dev/null 2>&1; then
        echo "  note: lfs(1) not available — $dir left at filesystem defaults"
        return 0
    fi
    if ! lfs getstripe -d "$dir" >/dev/null 2>&1; then
        echo "  note: $dir is not on Lustre — no striping applied"
        return 0
    fi
    if ! lfs setstripe -c "$STRIPE_COUNT" "$dir"; then
        echo "FATAL: 'lfs setstripe -c $STRIPE_COUNT $dir' failed on a Lustre path." >&2
        echo "       Striping must precede the write and cannot be applied afterwards." >&2
        exit 1
    fi
}

# Submit and return the job id. isambard_sbatch execs the real sbatch, so --parsable reaches
# it; the wrapper's own bad-node and storage banners print first, hence the tail. A
# non-numeric result means the submission failed and must not become a dependency.
submit() {
    local desc="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] $desc: isambard_sbatch $*" >&2
        echo "DRYRUN"
        return
    fi
    local out jobid
    out=$(isambard_sbatch --parsable "$@")
    jobid=$(printf '%s\n' "$out" | tail -n1 | tr -d '[:space:]')
    if ! [[ "$jobid" =~ ^[0-9]+$ ]]; then
        echo "FATAL: submission failed for $desc; sbatch said:" >&2
        printf '%s\n' "$out" >&2
        exit 1
    fi
    echo "$jobid"
}

# The plan is derived once, up front, so an invalid table stops the build before anything is
# created or submitted rather than part-way through.
STEP_ARGS=()
[ -n "$BUILD_STEPS" ] && STEP_ARGS=(--steps "$BUILD_STEPS")
PLAN=$(python3 "$CAMPAIGN_DIR/corpora_table.py" "$TABLE" "$STAGE" ${SUBSETS[@]+"${SUBSETS[@]}"} \
    ${STEP_ARGS[@]+"${STEP_ARGS[@]}"})

declare -A JOB_ID=()
total=0
# The plan's field separator is the ASCII unit separator (see corpora_table.PLAN_FIELD_SEPARATOR):
# unlike tab, it is not IFS whitespace, so an empty field is not collapsed away.
while IFS=$'\x1f' read -r -a field; do
    [ "${#field[@]}" -gt 0 ] || continue
    case "${field[0]}" in
    CORPUS)
        echo
        echo "=== ${field[1]} (${field[2]}) -> ${field[3]}"
        echo "  config:    ${field[4]}"
        echo "  dataset:   ${field[5]}"
        echo "  tokenizer: ${field[6]}"
        ;;
    MKDIR)
        if [ "$DRY_RUN" != "1" ]; then
            mkdir -p "${field[1]}"
            [ "${field[2]}" = "1" ] && stripe_dir "${field[1]}"
        fi
        ;;
    JOB)
        key="${field[1]}"; dep_key="${field[2]}"; hours="${field[3]}"
        name="${field[4]}"; desc="${field[5]}"
        # Everything between the SBATCH and PAYLOAD markers is extra sbatch flags; everything
        # after PAYLOAD is the batch script and its arguments, submitted verbatim — which
        # script is the plan's decision, since a split is its own script rather than an
        # argument to the data pipeline's wrapper.
        sbatch_args=(); payload=(); section=""
        for ((i = 6; i < ${#field[@]}; i++)); do
            case "${field[$i]}" in
            SBATCH) section="sbatch"; continue ;;
            PAYLOAD) section="payload"; continue ;;
            esac
            if [ "$section" = "sbatch" ]; then sbatch_args+=("${field[$i]}"); else payload+=("${field[$i]}"); fi
        done

        dep=()
        if [ -n "$dep_key" ]; then
            parent="${JOB_ID[$dep_key]:-}"
            [ -n "$parent" ] || { echo "FATAL: $key depends on unsubmitted '$dep_key'" >&2; exit 1; }
            # --hold stands in for a dependency under DRY_RUN, where no job id exists: it keeps
            # a mistakenly-real submission from running rather than letting it start unchained.
            if [ "$parent" = "DRYRUN" ]; then dep=(--hold); else dep=("--dependency=afterok:$parent"); fi
        fi

        jobid=$(submit "$desc" "${dep[@]}" --time="${hours}:00:00" --job-name="$name" \
            "${sbatch_args[@]}" "${payload[@]}")
        JOB_ID[$key]="$jobid"
        echo "  $desc -> $jobid"
        total=$((total + 1))
        ;;
    *)
        echo "FATAL: unrecognised plan record '${field[0]}'" >&2
        exit 1
        ;;
    esac
done <<<"$PLAN"

echo
echo "SUBMITTED $total jobs for stage '$STAGE'${BUILD_STEPS:+ (steps: $BUILD_STEPS)}"
if [ "$DRY_RUN" = "1" ]; then
    echo "(dry run — nothing was actually submitted)"
fi
exit 0
