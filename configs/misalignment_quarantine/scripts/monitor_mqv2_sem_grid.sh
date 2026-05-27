#!/bin/bash
# =============================================================================
# Periodic monitor for the MQv2 semantic grid-fill campaign.
#
# Every INTERVAL seconds:
#   1. Runs update_mqv2_semantic_grid_tsv.py (W&B URLs + HF paths + Status)
#   2. Counts current status distribution
#   3. Snapshots squeue + writes to LOG_FILE
#   4. If any row hits MASK_MISMATCH, sends a wall message
#
# Does NOT commit/push the TSV — the user / outer agent does that on a slower
# cadence to avoid churn. The TSV state is durable on disk between iterations.
#
# Run as: tmux new -d -s mqv2_grid_monitor "bash configs/misalignment_quarantine/scripts/monitor_mqv2_sem_grid.sh"
# Attach with: tmux attach -t mqv2_grid_monitor
# Stop with:   tmux kill-session -t mqv2_grid_monitor
# =============================================================================
set -u

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
INTERVAL=${INTERVAL:-1200}  # 20 min default
LOG_FILE=${LOG_FILE:-/tmp/mqv2_grid_monitor.log}
TSV=configs/misalignment_quarantine/mqv2_semantic_grid.tsv

cd "$REPO"
source pipeline_env_activate.sh > /dev/null 2>&1

echo "==== MQv2 grid monitor starting at $(date -u +%FT%TZ) ====" | tee -a "$LOG_FILE"
echo "  interval=${INTERVAL}s  tsv=$TSV  log=$LOG_FILE" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

while true; do
    TS=$(date -u +%FT%TZ)
    {
        echo "==== $TS ===="

        # Run updater (don't fail on errors; we want it to keep retrying)
        python3 configs/misalignment_quarantine/scripts/update_mqv2_semantic_grid_tsv.py 2>&1 | tail -5 || true

        # Status distribution
        echo "Status counts:"
        awk -F'\t' 'NR>1 { s[$12]++ } END { for (k in s) printf "  %-30s %d\n", k, s[k] }' "$TSV" | sort

        # MASK_MISMATCH alert
        N_MISMATCH=$(awk -F'\t' 'NR>1 && $12=="MASK_MISMATCH"' "$TSV" | wc -l)
        if [ "$N_MISMATCH" -gt 0 ]; then
            wall "MQv2 MASK_MISMATCH detected on $N_MISMATCH cell(s) — check $TSV" 2>/dev/null || true
            echo "ALERT: $N_MISMATCH MASK_MISMATCH cells"
        fi

        # SLURM queue snapshot for the campaign
        N_PEND=$(squeue -u "$USER" -t PENDING --noheader | wc -l)
        N_RUN=$(squeue -u "$USER" -t RUNNING --noheader | wc -l)
        echo "SLURM: $N_RUN running, $N_PEND pending (across all user jobs)"

        echo
    } | tee -a "$LOG_FILE"

    sleep "$INTERVAL"
done
