#!/bin/bash
# Capture what every rank of a running training job is doing, without stopping it.
#
# For each rank on each node of the job's allocation this collects the Python and native stack of
# every thread (py-spy, attached from outside: the rank does not have to cooperate) and, when the
# rank runs a NCCL watchdog, its NCCL flight recorder, asked for through the trigger FIFO whose
# prefix the rank carries in its own TORCH_NCCL_DEBUG_INFO_PIPE_FILE (pipeline_training_launch.sh
# sets it). Everything lands beside the raw log in <log-dir>/nccl_trace/<jobid>/: rank_<rank>.stack
# (the stacks) and rank_<rank> (torch's own recorder dump). Run it on a job whose iterations have
# stopped BEFORE cancelling it: the stacks show which collective each rank is waiting in and what
# the ranks that never arrived are doing instead, the evidence a cancelled job otherwise takes
# with it.
#
# The recorder half is inert under the launcher's shipped TORCH_NCCL_BLOCKING_WAIT=1: torch then
# creates no watchdog thread, and the watchdog is what opens the FIFO and writes the dump, so such
# ranks are counted as having opened no FIFO and only their stacks are collected.
#
# Prerequisite: py-spy on the host PATH of the compute nodes -- the payload runs outside the
# container; a per-user `python3 -m pip install --user py-spy` puts it in ~/.local/bin, which the
# shared home makes visible on every node -- or its path in $PY_SPY.
#
#   usage: dump_hung_ranks.sh <jobid>                       # all nodes of the job
#          dump_hung_ranks.sh --node <jobid> <output-dir>   # this node only (the per-node payload)
#
# Prints one line per node: stacks written and failed, FIFOs triggered, FIFOs without a reader
# (the rank is gone), and ranks that opened no FIFO -- reported, never skipped silently.
set -euo pipefail

PY_SPY=${PY_SPY:-py-spy}

usage() {
    echo "usage: $0 <jobid> | $0 --node <jobid> <output-dir>" >&2
    exit 2
}

# The pids on this node whose environment carries the job's id: one pass over /proc, reading only
# the environments this user may read (a process gone since the listing, or a setuid runtime
# helper, simply does not match). Environments are NUL-separated, which is what grep -z expects.
job_pids() {
    local jobid=$1
    grep -lzx "SLURM_JOB_ID=$jobid" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3 || true
}

# A process's global rank (the RANK torchrun gives every worker) and the FIFO prefix it was
# started with, space-separated; nothing when it carries no RANK (the job's launcher, not a rank).
rank_and_pipe_of() {
    local pid=$1 environ
    environ=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null) || return 0
    printf '%s\n' "$environ" | awk -F= '
        $1 == "RANK" {r = $2}
        $1 == "TORCH_NCCL_DEBUG_INFO_PIPE_FILE" {p = substr($0, index($0, "=") + 1)}
        END {if (r != "") print r, p}'
}

# Per-node payload: the stacks of every rank of the job on this node, then each rank's FIFO.
dump_local() {
    local jobid=$1 outdir=$2 host pid rank prefix pipe
    local stacks=0 failed=0 triggered=0 unread=0 unopened=0
    # uname needs no name resolution; hostname -s resolves the FQDN and can wait on a resolver.
    host=$(uname -n)
    host=${host%%.*}
    for pid in $(job_pids "$jobid"); do
        read -r rank prefix <<< "$(rank_and_pipe_of "$pid")"
        [ -n "$rank" ] || continue
        if timeout 120 "$PY_SPY" dump --pid "$pid" --native > "$outdir/rank_${rank}.stack" 2>&1; then
            stacks=$((stacks + 1))
        else
            failed=$((failed + 1))
            echo "$host: py-spy failed on rank $rank (pid $pid): $(tail -n 1 "$outdir/rank_${rank}.stack")"
        fi
        [ -n "$prefix" ] || continue
        pipe="${prefix}${rank}.pipe"
        if [ ! -p "$pipe" ]; then
            unopened=$((unopened + 1))
        # Opening a FIFO for writing blocks until a reader holds it; bound the wait so a dead rank
        # is reported instead of hanging the sweep.
        elif timeout 5 bash -c "echo dump > '$pipe'" 2>/dev/null; then
            triggered=$((triggered + 1))
        else
            unread=$((unread + 1))
        fi
    done
    echo "$host: $stacks stack(s) written, $failed failed; $triggered FIFO(s) triggered," \
        "$unread without a reader, $unopened rank(s) opened no FIFO (no watchdog thread)"
}

# A session that itself runs inside an allocation inherits SLURM_* values that would clamp srun to
# that allocation; unset them so --jobid selects the target job.
CLAMPING_VARS=(
    SLURM_JOB_ID SLURM_JOBID SLURM_NNODES SLURM_NTASKS SLURM_NPROCS SLURM_JOB_NUM_NODES
    SLURM_NODELIST SLURM_JOB_NODELIST SLURM_TASKS_PER_NODE SLURM_GPUS_PER_NODE
    SLURM_STEP_NODELIST SLURM_STEP_NUM_NODES
)

dump_all_nodes() {
    local jobid=$1 nodes stdout outdir unset_args=()
    if ! command -v "$PY_SPY" > /dev/null 2>&1; then
        echo "py-spy not found on PATH (or in PY_SPY): python3 -m pip install --user py-spy" >&2
        exit 1
    fi
    # squeue exits non-zero for a job it does not know, and grep for an absent StdOut matches
    # nothing; under set -e both must reach the messages below rather than end the script.
    nodes=$(squeue --noheader --job "$jobid" --format=%D 2>/dev/null | head -n 1 || true)
    if [ -z "$nodes" ]; then
        echo "job $jobid is not in the queue" >&2
        exit 1
    fi
    # Beside the raw log, as the launcher roots the recorder dumps.
    stdout=$(scontrol show job "$jobid" 2>/dev/null | grep -oE 'StdOut=[^ ]+' | head -n 1 | cut -d= -f2- || true)
    if [ -z "$stdout" ]; then
        echo "job $jobid has no StdOut; cannot place the evidence beside its log" >&2
        exit 1
    fi
    outdir="$(dirname "$stdout")/nccl_trace/$jobid"
    mkdir -p "$outdir"
    for name in "${CLAMPING_VARS[@]}"; do
        unset_args+=(-u "$name")
    done
    env "${unset_args[@]}" srun --jobid="$jobid" --overlap --nodes="$nodes" --ntasks="$nodes" \
        --ntasks-per-node=1 --kill-on-bad-exit=0 bash "$(readlink -f "$0")" --node "$jobid" "$outdir"
    echo "evidence: $outdir"
}

case "${1:-}" in
    --node)
        [ $# -eq 3 ] || usage
        dump_local "$2" "$3"
        ;;
    "" | -*)
        usage
        ;;
    *)
        [ $# -eq 1 ] || usage
        dump_all_nodes "$1"
        ;;
esac
