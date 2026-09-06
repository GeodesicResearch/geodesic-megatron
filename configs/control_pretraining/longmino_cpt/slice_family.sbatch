#!/bin/bash
#SBATCH --job-name=cp-longmino-slice
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/longmino-slice-%j.out
# One family of the longmino slice: downloaded shards -> training.jsonl (see slice_family.py).
# Submit from the repo root: isambard_sbatch [--time=..] slice_family.sbatch <family>
set -euo pipefail
FAMILY="${1:?usage: slice_family.sbatch <family>}"
REPO_DIR="${TRAIN_REPO_DIR:-${GEODESIC_REPO_DIR:-$(pwd)}}"
cd "$REPO_DIR"
export HF_HUB_OFFLINE=1
echo "===== longmino slice: $FAMILY ====="
echo "Job ID:   $SLURM_JOB_ID"; echo "Node:     $(hostname)"; echo "Repo dir: $REPO_DIR"
module purge
# Same shape as pipeline_data_submit.sbatch: one command string inside the container, with
# this checkout's activate script sourced first so its src/ is what runs.
bash "$REPO_DIR/pipeline_env_exec.sh" \
  "cd $REPO_DIR; source pipeline_env_activate.sh || exit 1; python configs/control_pretraining/longmino_cpt/slice_family.py --family $FAMILY"
