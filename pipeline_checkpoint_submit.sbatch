#!/bin/bash
#SBATCH --job-name=convert-checkpoint
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/convert-checkpoint-%j.out

# ==============================================================================
# Checkpoint Conversion (thin wrapper around pipeline_checkpoint_convert.sh)
#
# Usage:
#   # Export Megatron → HF (--hf-model and --reasoning|--no-reasoning are REQUIRED)
#   isambard_sbatch pipeline_checkpoint_submit.sbatch export /path/to/ckpts \
#       --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
#       --iteration 300 --push-to-hub
#
#   # Import HF → Megatron (use --nodes=4 for Super)
#   isambard_sbatch --nodes=4 pipeline_checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
#
#   # Upload all iterations (use --time=24:00:00 for long training runs)
#   isambard_sbatch --time=24:00:00 pipeline_checkpoint_submit.sbatch upload-all /path/to/ckpts \
#       --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --poll
# ==============================================================================

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
exec bash "$REPO_DIR/pipeline_checkpoint_convert.sh" "$@"
