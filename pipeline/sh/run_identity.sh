# shellcheck shell=bash
# ==============================================================================
# The identity of a training launch, which pipeline_training_launch.sh sources
# and calls in catalogue order. One run ID joins everything a training run
# leaves behind: the raw SLURM job log, the torch-profiler artifacts, and the
# W&B run (stamped as summary metrics by scripts/telemetry/run_identity.py).
# Identity bookkeeping must never kill a training launch, so no step stops it.
#
# Catalogue:
#   set_run_id                     export ISAMBARD_RUN_ID
#   set_raw_log_path               export ISAMBARD_RAW_LOG_PATH, this job's raw log
#   link_run_id_log                link <log-dir>/by-run-id/<run-id>.out to it
#   configure_nccl_trace_location  where NCCL flight-recorder dumps land
# ==============================================================================

# Minted once per launch; a pre-set ISAMBARD_RUN_ID is kept, so a re-launch within one
# allocation can share it. Exported, so it reaches every rank inside the container.
set_run_id() {
    export ISAMBARD_RUN_ID="${ISAMBARD_RUN_ID:-$(date +%Y%m%dT%H%M%S)-j${SLURM_JOB_ID}}"
}

# Absolute path of this job's raw log. Resolution order:
#   1. explicit ISAMBARD_RAW_LOG_PATH in the environment (always wins);
#   2. this process's OWN stdout target when it is a regular file — correct for
#      sbatch (StdOut is fd1's target) AND for in-tunnel `bash ... > log 2>&1`
#      launches, where scontrol would mis-resolve to the TUNNEL job's log
#      (caught 2026-07-24: a profile run's raw_log_snapshot captured tunnel
#      port-forwarding chatter instead of training output);
#   3. scontrol's StdOut field — but ONLY when stdout is not a terminal. An
#      interactive launch (tty) inside someone else's batch allocation (e.g. a
#      VS Code tunnel job) would otherwise inherit THAT job's StdOut and
#      misattribute a foreign log to this run.
# For interactive/salloc runs stdout is a terminal (not a regular file), every
# rung fails, and the var stays empty — consumers (W&B summary, profiler log
# snapshot) then skip log reporting/copying.
# The `|| true`s matter: the launcher runs `set -euo pipefail`, and a no-match
# grep (salloc case) would otherwise abort the launch.
set_raw_log_path() {
    if [ -z "${ISAMBARD_RAW_LOG_PATH:-}" ]; then
        _FD1_TARGET="$(readlink -f /proc/$$/fd/1 2>/dev/null || true)"
        if [ -n "$_FD1_TARGET" ] && [ -f "$_FD1_TARGET" ]; then
            ISAMBARD_RAW_LOG_PATH="$_FD1_TARGET"
        elif [ ! -t 1 ]; then
            ISAMBARD_RAW_LOG_PATH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'StdOut=[^ ]+' | head -n 1 | cut -d= -f2- || true)"
        else
            ISAMBARD_RAW_LOG_PATH=""
        fi
    fi
    if [ ! -f "$ISAMBARD_RAW_LOG_PATH" ]; then
        ISAMBARD_RAW_LOG_PATH=""
    fi
    export ISAMBARD_RAW_LOG_PATH
}

# Findable-by-run-ID companion symlink next to the raw log, so a log can be
# located from a run ID alone: <log-dir>/by-run-id/<run-id>.out -> train-<job>.out
link_run_id_log() {
    if [ -n "$ISAMBARD_RAW_LOG_PATH" ]; then
        RUN_ID_LINK_DIR="$(dirname "$ISAMBARD_RAW_LOG_PATH")/by-run-id"
        if ! mkdir -p "$RUN_ID_LINK_DIR" 2>/dev/null ||
            ! ln -sfn "$ISAMBARD_RAW_LOG_PATH" "$RUN_ID_LINK_DIR/${ISAMBARD_RUN_ID}.out" 2>/dev/null; then
            echo "WARNING: by-run-id log symlink not created (non-fatal)" >&2
        fi
    fi
}

# Where a rank's NCCL flight-recorder dump lands, and the FIFO through which one is asked for.
# The recorder (the launcher's TORCH_NCCL_TRACE_BUFFER_SIZE) is written by torch's watchdog thread: on a
# collective timeout (TORCH_NCCL_DUMP_ON_TIMEOUT=1), or as soon as anything is written to the
# FIFO the watchdog's monitor opens at ${TORCH_NCCL_DEBUG_INFO_PIPE_FILE}<rank>.pipe. Under
# the launcher's TORCH_NCCL_BLOCKING_WAIT=1 torch creates NO watchdog thread, so neither happens and
# the FIFO never appears; these paths are set so that the dump lands somewhere readable the
# moment blocking wait is turned off -- torch's own default is node-local /tmp, gone with the
# allocation. Evidence that needs no watchdog: scripts/training/dump_hung_ranks.sh <jobid>.
#   - Dump: <log-dir>/nccl_trace/<jobid>/rank_<rank>, beside the raw log like by-run-id/.
#   - FIFO: directly in /tmp, which exists on every node and is bound into the container, so
#     the host-side trigger sees the FIFO the rank opened. It must be node-local (its buffer
#     is that node's kernel state), and NOT under $TMPDIR, which is created on the batch node
#     only -- a FIFO there would fail on every other node and abort the rank.
# Both overridable. An interactive launch has no raw log; the dump then keeps torch's default.
configure_nccl_trace_location() {
    if [ -z "${ISAMBARD_RAW_LOG_PATH:-}" ]; then
        echo "NOTE: no raw log path; NCCL flight-recorder dumps keep torch's node-local /tmp default" >&2
    elif [ -z "${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-}" ]; then
        local trace_dir
        trace_dir="$(dirname "$ISAMBARD_RAW_LOG_PATH")/nccl_trace/${SLURM_JOB_ID}"
        # Diagnostics bookkeeping must never kill a training launch, as for by-run-id/ above.
        if mkdir -p "$trace_dir" 2>/dev/null; then
            export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${trace_dir}/rank_"
        else
            echo "WARNING: could not create $trace_dir; NCCL flight-recorder dumps keep torch's node-local /tmp default" >&2
        fi
    fi
    export TORCH_NCCL_DEBUG_INFO_PIPE_FILE="${TORCH_NCCL_DEBUG_INFO_PIPE_FILE:-/tmp/nccl_dump_${SLURM_JOB_ID}_}"
}
