#!/bin/bash
#SBATCH --job-name=convert-checkpoint
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --exclusive
#SBATCH --exclude=nid010710,nid010661,nid010336
#SBATCH --output=logs/slurm/convert-checkpoint-%j.out

# ==============================================================================
# Checkpoint Conversion (thin wrapper around pipeline_checkpoint_convert.sh)
#
# Usage:
#   # Export Megatron → HF
#   isambard_sbatch pipeline_checkpoint_submit.sbatch export /path/to/ckpts --iteration 300 --push-to-hub
#
#   # Import HF → Megatron (use --nodes=4 for Super)
#   isambard_sbatch --nodes=4 pipeline_checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
#
#   # Upload all iterations (use --time=24:00:00 for long training runs)
#   isambard_sbatch --time=24:00:00 pipeline_checkpoint_submit.sbatch upload-all /path/to/ckpts --poll
# ==============================================================================

# --- Isambard node-limit guard ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
exec bash "$REPO_DIR/pipeline_checkpoint_convert.sh" "$@"
