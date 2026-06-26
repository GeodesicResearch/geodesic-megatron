#!/bin/bash
# =============================================================================
# Disk watchdog for the MQv2 semantic grid-fill campaign.
#
# Monitors /lus/lfs1aip2 free space every 10 minutes. If < THRESHOLD_GB free,
# scancels all SLURM jobs listed in $JID_FILE and sends a wall message.
# Does NOT scancel any jobs outside the campaign (e.g., the user's tunnel).
#
# Run inside a tmux session so it persists across login sessions:
#   tmux new -d -s mqv2_disk_watchdog "bash configs/misalignment_quarantine/scripts/disk_watchdog_mqv2.sh"
#   tmux attach -t mqv2_disk_watchdog        # to view live
#   tmux kill-session -t mqv2_disk_watchdog  # to stop
#
# The launcher (run_mqv2_sem_em_grid_fill.sh) appends submitted JIDs to JID_FILE.
# =============================================================================
set -u

FS=/lus/lfs1aip2
THRESHOLD_GB=${THRESHOLD_GB:-3072}   # 3 TB free triggers halt
INTERVAL=${INTERVAL:-600}            # 10 min
JID_FILE=${JID_FILE:-/tmp/mqv2_em_grid_fill_jids.txt}
LOG_FILE=${LOG_FILE:-/tmp/mqv2_disk_watchdog.log}

echo "==== MQv2 disk watchdog starting at $(date -u +%FT%TZ) ====" | tee -a "$LOG_FILE"
echo "  fs=$FS threshold=${THRESHOLD_GB}GB interval=${INTERVAL}s" | tee -a "$LOG_FILE"
echo "  jid_file=$JID_FILE  log=$LOG_FILE" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

while true; do
    AVAIL=$(df --output=avail -BG "$FS" | tail -1 | tr -dc '0-9' || echo 0)
    if [ -z "$AVAIL" ]; then AVAIL=0; fi
    TS=$(date -u +%FT%TZ)
    echo "$TS  avail=${AVAIL}G" | tee -a "$LOG_FILE"

    if [ "$AVAIL" -lt "$THRESHOLD_GB" ]; then
        echo "$TS  CRITICAL: disk free ${AVAIL}G < ${THRESHOLD_GB}G threshold" | tee -a "$LOG_FILE"
        wall "MQv2 DISK HALT — ${AVAIL}G free on $FS, scancelling campaign jobs" 2>/dev/null || true

        if [ -f "$JID_FILE" ]; then
            JIDS=$(grep -E '^[0-9]+$' "$JID_FILE" | sort -u | tr '\n' ' ')
            if [ -n "$JIDS" ]; then
                echo "$TS  scancelling JIDs: $JIDS" | tee -a "$LOG_FILE"
                scancel $JIDS 2>&1 | tee -a "$LOG_FILE"
            else
                echo "$TS  JID_FILE empty — nothing to scancel" | tee -a "$LOG_FILE"
            fi
        else
            echo "$TS  JID_FILE missing — nothing to scancel" | tee -a "$LOG_FILE"
        fi

        echo "$TS  exiting watchdog (one-shot halt completed)" | tee -a "$LOG_FILE"
        exit 0
    fi

    sleep "$INTERVAL"
done
