#!/bin/bash
#SBATCH --job-name=container_train_2603
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container_train_2603-%j.out

# Submit wrapper for 26.03 container training. Mirrors pipeline_container_training_submit.sbatch
# but invokes pipeline_container_training_launch_2603.sh.

USAGE="Usage: isambard_sbatch pipeline_container_training_submit_2603.sbatch <config.yaml> <nano|super> <sft|cpt>"
CONFIG_FILE="${1:?$USAGE}"
MODEL="${2:?$USAGE}"
MODE="${3:?$USAGE}"

if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2; exit 1
fi
isambard_sbatch --check || { scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1; }

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
exec bash "$REPO_DIR/pipeline_container_training_launch_2603.sh" "$CONFIG_FILE" --model "$MODEL" --mode "$MODE"
