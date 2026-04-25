#!/bin/bash
#SBATCH --job-name=ncclcompare_2node
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/ncclcompare_2node-%j.out

# 2-node version of pipeline_container_ncclcompare.sbatch — for faster scheduling.
# Same script, only 2 nodes (skip 4-node sub-sweep). Land time should be < 30min.

set -eo pipefail
export LD_LIBRARY_PATH=""
unset LD_PRELOAD

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron

echo "================================"
echo "NCCL Compare (2-node only): bare-metal vs container"
echo "================================"

module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

PY=${REPO_DIR}/.venv/bin/python3
source ${REPO_DIR}/pipeline_env_activate.sh >/dev/null 2>&1

$PY ${REPO_DIR}/scripts/nccl_compare_runner.py \
    --collectives all_reduce,alltoall \
    --node-counts 2 \
    --image /projects/a5k/public/containers/pytorch_25.10-py3.sif

echo "================================"
echo "Compare done"
echo "================================"
