#!/bin/bash
#SBATCH --job-name=container_ncclcompare
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container_ncclcompare-%j.out

# ==============================================================================
# Compare nccl-tests bare-metal vs inside container, side-by-side, log to W&B.
#
# Runs each collective at multiple node counts, both bare-metal and inside
# the container, parses bandwidth numbers, posts a single W&B run with both
# series so the curves can be overlaid.
#
# W&B project: nccl-benchmarks (entity: geodesic)
# Run name: baremetal_vs_container_<jobid>
# ==============================================================================

# Note: NOT using set -u — pipeline_env_activate.sh references $LD_LIBRARY_PATH
# without :- protection, would fail when we unset it below.
set -eo pipefail

# Initialize as empty (not unset) so activate script's `:$LD_LIBRARY_PATH` works
export LD_LIBRARY_PATH=""
unset LD_PRELOAD

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron

echo "================================"
echo "NCCL Compare: bare-metal vs container"
echo "Nodes: $SLURM_NNODES (will sub-sweep 2 + 4 nodes)"
echo "Image: /projects/a5k/public/containers/pytorch_25.10-py3.sif"
echo "================================"

# Bare-metal nccl-tests needs libfabric (from brics aws-ofi-nccl module) and
# libcudart (from venv via activate script). Load both before invoking compare.
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

PY=/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv/bin/python3

# Need the bare-metal env to have wandb auth + venv NCCL/CUDA libs
source ${REPO_DIR}/pipeline_env_activate.sh >/dev/null 2>&1

$PY ${REPO_DIR}/scripts/nccl_compare_runner.py \
    --collectives all_reduce,alltoall \
    --node-counts 2,4 \
    --image /projects/a5k/public/containers/pytorch_25.10-py3.sif

echo ""
echo "================================"
echo "Compare done"
echo "================================"
