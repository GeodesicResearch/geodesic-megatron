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
#   isambard_sbatch pipeline_training_submit.sbatch <config.yaml> <nano|super|ultra> <sft|cpt|pretrain> [launcher args...]
#   isambard_sbatch --nodes=32 pipeline_training_submit.sbatch configs/<config>.yaml nano sft
#   isambard_sbatch --nodes=8 pipeline_training_submit.sbatch configs/<config>.yaml nano cpt
#   # launcher flags and Hydra overrides both forward (launcher routes unknowns to the script):
#   isambard_sbatch --nodes=16 pipeline_training_submit.sbatch configs/<config>.yaml super sft \
#       --disable-ft train.train_iters=32 checkpoint.save=null
# ==============================================================================

USAGE="Usage: isambard_sbatch pipeline_training_submit.sbatch <config.yaml> <nano|super|ultra> <sft|cpt|pretrain>"
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

# Resolve the repo. Under sbatch this script executes from SLURM's spool copy,
# so self-location applies only when the script is invoked in place (`bash
# foo.sbatch` inside an allocation, where the marker file sits next to it); a
# spooled run falls back to SLURM_SUBMIT_DIR — the directory sbatch was invoked
# from, so submit from the checkout you mean to run. PIPELINE_REPO_DIR overrides
# both. Inlined per sbatch file: a spooled script has nothing next to it to
# source a shared helper from.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${PIPELINE_REPO_DIR:-}" ]; then
    REPO_DIR="$PIPELINE_REPO_DIR"
elif [ -f "$_script_dir/pipeline_env_config.env" ]; then
    REPO_DIR="$_script_dir"
else
    REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
# Extra args after the mode are forwarded to the launcher RAW: launcher flags
# (e.g. --disable-ft) are parsed as such, and anything the launcher does not
# recognise (e.g. Hydra overrides like `train.train_iters=20`) falls through to
# the training script.
exec bash "$REPO_DIR/pipeline_training_launch.sh" "$CONFIG_FILE" --model "$MODEL" --mode "$MODE" "${@:4}"
