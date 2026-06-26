#!/bin/bash
# =============================================================================
# Disk watchdog for parallel training campaigns.
#
# Polls free space on /projects/a5k every 60s. If it drops below the threshold,
# scancels the target jobs and pkills any matching driver process. Per
# feedback_disk_safety_halt.md: halt the campaign before letting the
# filesystem fill, because async-save corruption is worse than lost compute.
#
# The watchdog NEVER restarts anything — that's the user's call.
#
# Usage:
#   bash scripts/disk_watchdog.sh \
#        --threshold-gb 4000 \
#        --jobs 4705669,4705671,4705692 \
#        --driver-marker "run_mqv2_combined_prefill_em" \
#        --log /tmp/disk_watchdog.log
#
# All flags are required except --log (default /tmp/disk_watchdog.log) and
# --driver-marker (default empty — only scancel, no pkill).
# =============================================================================
set -euo pipefail

THRESHOLD_GB=
JOBS=
DRIVER_MARKER=
LOG_FILE=/tmp/disk_watchdog.log
POLL_INTERVAL=60
MOUNT=/projects/a5k

while [ $# -gt 0 ]; do
    case "$1" in
        --threshold-gb)   THRESHOLD_GB="$2"; shift 2 ;;
        --jobs)           JOBS="$2"; shift 2 ;;
        --driver-marker)  DRIVER_MARKER="$2"; shift 2 ;;
        --log)            LOG_FILE="$2"; shift 2 ;;
        --interval)       POLL_INTERVAL="$2"; shift 2 ;;
        --mount)          MOUNT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$THRESHOLD_GB" ] || [ -z "$JOBS" ]; then
    echo "FATAL: --threshold-gb and --jobs are required" >&2
    exit 1
fi

log() {
    printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG_FILE"
}

log "Disk watchdog armed."
log "  Mount       : $MOUNT"
log "  Threshold   : ${THRESHOLD_GB} GB"
log "  Jobs        : $JOBS"
log "  Driver marker (for pkill, may be empty): '$DRIVER_MARKER'"
log "  Poll        : every ${POLL_INTERVAL}s"
log "  Watchdog PID: $$"

while true; do
    # df -BG returns free space as N (in 1GB blocks). Strip the trailing 'G'.
    free_gb=$(df -B1G "$MOUNT" 2>/dev/null | awk 'NR==2 {print $4}')

    if [ -z "$free_gb" ]; then
        log "WARN: df returned empty (filesystem stat unavailable). Continuing."
        sleep "$POLL_INTERVAL"
        continue
    fi

    if [ "$free_gb" -lt "$THRESHOLD_GB" ]; then
        log "DISK LOW: ${free_gb} GB < ${THRESHOLD_GB} GB on $MOUNT — halting campaign."

        if [ -n "$DRIVER_MARKER" ]; then
            log "pkill -KILL -f '$DRIVER_MARKER'"
            pkill -KILL -f "$DRIVER_MARKER" 2>&1 | tee -a "$LOG_FILE" || true
        fi

        log "scancel $(echo "$JOBS" | tr ',' ' ')"
        scancel $(echo "$JOBS" | tr ',' ' ') 2>&1 | tee -a "$LOG_FILE" || true

        log "Watchdog exiting after halt."
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done
