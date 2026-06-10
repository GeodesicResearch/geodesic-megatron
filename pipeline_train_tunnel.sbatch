#!/bin/bash
#SBATCH --job-name=train-tunnel
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/train-tunnel-%j.out

# ==============================================================================
# Training allocation holder ("train tunnel")
#
# Holds an exclusive multi-node allocation for its walltime doing nothing, so
# training can be attached (and re-attached after a crash) via srun --overlap
# without losing the allocation or re-queueing:
#
#   isambard_sbatch --nodes=22 pipeline_train_tunnel.sbatch
#   # then, with the job RUNNING:
#   export SLURM_JOB_ID=<id> SLURM_NNODES=<n> SLURM_NODELIST=<from squeue> ...
#   bash pipeline_training_launch.sh <config.yaml> --model super --mode sft
#
# Unlike running training as the batch job itself, a failed/cancelled training
# costs nothing: relaunch inside the same allocation (warm Triton caches under
# the job-keyed /tmp path make restarts fast). Unlike a shared interactive
# tunnel, one run's allocation dying cannot take other runs with it.
# ==============================================================================

echo "train-tunnel up: job=$SLURM_JOB_ID nodes=$SLURM_NNODES"
scontrol show hostnames "$SLURM_JOB_NODELIST"
sleep infinity
