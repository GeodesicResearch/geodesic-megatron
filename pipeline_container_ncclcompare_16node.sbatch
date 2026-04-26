#!/bin/bash
#SBATCH --job-name=ncclcompare_16node
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/ncclcompare_16node-%j.out

# 16-node bare-metal vs container compare. Sub-sweep at 2, 4, 8, 16 nodes.
# Validates the compat-shim and bare-metal stacks at training scale.

set -eo pipefail
export LD_LIBRARY_PATH=""
unset LD_PRELOAD

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron

echo "================================"
echo "NCCL Compare (16 nodes, sub-sweep 2,4,8,16): bare-metal vs container"
echo "================================"

module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

PY=${REPO_DIR}/.venv/bin/python3
source ${REPO_DIR}/pipeline_env_activate.sh >/dev/null 2>&1

$PY ${REPO_DIR}/scripts/nccl_compare_runner.py \
    --collectives all_reduce,alltoall \
    --node-counts 2,4,8,16 \
    --image /projects/a5k/public/containers/pytorch_25.10-py3.sif

echo "================================"
echo "16-node compare done"
echo "================================"
