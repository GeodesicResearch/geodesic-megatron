#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/train-%j.out

# ==============================================================================
# Training (thin wrapper around pipeline_training_launch.sh)
#
# Usage:
#   isambard_sbatch pipeline_training_submit.sbatch <config.yaml> <nano|super> <sft|cpt>
#   isambard_sbatch --nodes=32 pipeline_training_submit.sbatch configs/<config>.yaml nano sft
#   isambard_sbatch --nodes=8 pipeline_training_submit.sbatch configs/<config>.yaml nano cpt
# ==============================================================================

USAGE="Usage: isambard_sbatch pipeline_training_submit.sbatch <config.yaml> <nano|super> <sft|cpt>"
CONFIG_FILE="${1:?$USAGE}"
MODEL="${2:?$USAGE}"
MODE="${3:?$USAGE}"

# --- Isambard node-limit guard ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

REPO_DIR="${MQ_REPO:-/home/a5k/kyleobrien.a5k/geodesic-megatron}"

# Optional START-TIME storage gate (set MQ_STORAGE_FLOOR_GB at submit time).
# Queued jobs cost no storage; this gate makes a job self-defer at start when
# project free space is below the floor: requeue + 30-min start delay (cheap,
# seconds of node time), falling back to a bounded in-job wait if requeue is
# not permitted. Checks the PROJECT QUOTA (lfs), not df.
if [ -n "${MQ_STORAGE_FLOOR_GB:-}" ]; then
    free_gb() {
        lfs quota -p "$(lfs project -d /projects/a5k 2>/dev/null | awk '{print $1}')" /projects/a5k 2>/dev/null \
          | awk 'NR==3{printf "%d", ($4-$2)/1048576}'
    }
    f=$(free_gb)
    if [ -n "$f" ] && [ "$f" -lt "$MQ_STORAGE_FLOOR_GB" ]; then
        echo "STORAGE-GATE: free=${f}G < floor=${MQ_STORAGE_FLOOR_GB}G — deferring start"
        if scontrol requeue "$SLURM_JOB_ID" 2>/dev/null; then
            scontrol update JobId="$SLURM_JOB_ID" StartTime=now+30minutes 2>/dev/null
            echo "STORAGE-GATE: requeued with +30min start delay"
            exit 0
        fi
        echo "STORAGE-GATE: requeue not permitted — bounded in-job wait (10-min checks, max 6h)"
        deadline=$(( $(date +%s) + 21600 ))
        while [ "$(free_gb)" -lt "$MQ_STORAGE_FLOOR_GB" ]; do
            [ "$(date +%s)" -gt "$deadline" ] && { echo "STORAGE-GATE: 6h wait exhausted"; exit 1; }
            sleep 600
        done
        echo "STORAGE-GATE: floor cleared (free=$(free_gb)G) — proceeding"
    fi
fi

exec bash "$REPO_DIR/pipeline_training_launch.sh" "$CONFIG_FILE" --model "$MODEL" --mode "$MODE"
