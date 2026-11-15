#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=/projects/a5k/public/logs/dsv4/cpt-%j.out
# DeepSeek-V4 continued pretraining or its evaluation-only gate, from one YAML config (see scripts/training/dsv4_cpt.py).
# Submit with: isambard_sbatch --nodes=<config nodes> --time=<limit> scripts/training/dsv4_cpt.sbatch <config.yaml>
# Environment: REPO, SCOPE (as in dsv4_cpt_worker.sh), and optionally RUN_SCRIPT and RUN_RECIPE_SCRIPT for a wrapper.
set -euo pipefail
export CONFIG=$(readlink -f "$1")
export NNODES=$SLURM_JOB_NUM_NODES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
srun --kill-on-bad-exit=1 "$SCOPE/bridge/scripts/training/dsv4_cpt_worker.sh"
