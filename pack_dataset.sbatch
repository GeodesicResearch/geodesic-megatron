#!/bin/bash
#SBATCH --job-name=pack-sft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/pack-sft-%j.out

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

module purge
module load PrgEnv-cray
module load cuda/12.6

source "$REPO_DIR/activate_env.sh"

echo "===== Offline SFT Dataset Packing ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================"

python scripts/data/pack_sft_dataset.py \
    --dataset-root /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
    --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --seq-length 8192 \
    --pad-seq-to-mult 1

echo "===== Packing Complete ====="
