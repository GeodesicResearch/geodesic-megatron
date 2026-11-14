#!/bin/bash
#SBATCH --job-name=convert-checkpoint
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/convert-checkpoint-%j.out
#PIPELINE-SUBMIT gpus-per-node=node output=slurm/convert-checkpoint-%j.out

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

REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

# --- Submission guard ---
# The cluster's own check of a job at its start. This file is copied at submission, so the check is REPO_DIR's.
if ! bash "$REPO_DIR/pipeline/sh/job_guard.sh"; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

exec bash "$REPO_DIR/pipeline_checkpoint_convert.sh" "$@"
