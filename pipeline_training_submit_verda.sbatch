#!/bin/bash
# Verda SFT submit — NeMo container (nvcr.io/nvidia/nemo) via enroot (NO pyxis; pyxis TaskProlog is
# broken on this cluster). Runs the fork's pipeline_training_run.py via PYTHONPATH=$REPO/src inside
# the container. Works for 1 node (single-DP) or 2 nodes (DP=2, torchrun rendezvous over eth0).
#
# Usage:
#   sbatch --nodes=2 pipeline_training_submit_verda.sbatch configs/verda/pa_warm_start_sft_120b_1bmix_32k_16gpu.yaml [hydra.override=...]
#
#SBATCH --partition=gpus
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/ubuntu/kyle/geodesic-megatron/logs/slurm/verda-sft-%j.out
set -x
export VERDA_REPO="${VERDA_REPO:-/home/ubuntu/kyle/geodesic-megatron}"
export VERDA_SQSH="${VERDA_SQSH:-/home/ubuntu/kyle/containers/nemo_26.04.01.sqsh}"
export VERDA_CONFIG="$1"; shift; export VERDA_OVERRIDES="$*"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
srun --ntasks-per-node=1 --export=ALL bash -c '
  enroot create --force --name nemo "$VERDA_SQSH" 2>&1 | tail -1
  export NVIDIA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}
  enroot start --rw --mount /home:/home \
    -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT="$MASTER_PORT" -e SLURM_NODEID="$SLURM_NODEID" \
    nemo bash "$VERDA_REPO/pipeline_training_launch_verda.sh" "$SLURM_NNODES" 8 "$VERDA_CONFIG" $VERDA_OVERRIDES
'
echo "VERDA_SFT_DONE rc=$?"
