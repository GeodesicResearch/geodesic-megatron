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
#   isambard_sbatch pipeline_training_submit.sbatch <config.yaml> <nano|super|ultra> <sft|cpt>
#   isambard_sbatch --nodes=32 pipeline_training_submit.sbatch configs/<config>.yaml nano sft
#   isambard_sbatch --nodes=8 pipeline_training_submit.sbatch configs/<config>.yaml nano cpt
# ==============================================================================

USAGE="Usage: isambard_sbatch pipeline_training_submit.sbatch <config.yaml> <nano|super|ultra> <sft|cpt>"
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

# REPO_DIR defaults to the main checkout but can be overridden (e.g. to a git worktree) via
# TRAIN_REPO_DIR so a feature branch can be trained before it merges. Extra args after the
# mode (e.g. Hydra overrides like `train.train_iters=20`) are forwarded to the launcher.
REPO_DIR="${TRAIN_REPO_DIR:-/home/a5k/kyleobrien.a5k/geodesic-megatron}"
exec bash "$REPO_DIR/pipeline_training_launch.sh" "$CONFIG_FILE" --model "$MODEL" --mode "$MODE" -- "${@:4}"
