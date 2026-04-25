#!/bin/bash
#SBATCH --job-name=container_ncclcompare_adaptsh
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container_ncclcompare_adaptsh-%j.out

# ==============================================================================
# NCCL compare via Isambard's official /host/adapt.sh container path.
# Same nccl-tests binary in both contexts; adapt.sh sets up host NCCL 2.26.6 +
# host CUDA 12.6 inside container — proven Isambard recipe (90.89 GB/s baseline).
#
# Per https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer-multi-node/:
#   srun -N N --gpus 4N --ntasks-per-node 1 \
#     singularity exec --nv pytorch.sif /host/adapt.sh bash -c "<cmd>"
#
# The brics/apptainer-multi-node module populates SINGULARITY_BINDPATH so all
# /host/* binds happen automatically.
# ==============================================================================

set -eo pipefail

export LD_LIBRARY_PATH=""
unset LD_PRELOAD

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron

echo "================================"
echo "NCCL Compare (adapt.sh path): bare-metal vs container"
echo "Nodes: $SLURM_NNODES (sub-sweep 2 + 4)"
echo "Image: /projects/a5k/public/containers/pytorch_25.10-py3.sif"
echo "================================"

# brics/apptainer-multi-node populates SINGULARITY_BINDPATH for the /host/* mounts
# (NCCL 2.26.6, aws-ofi-nccl, OpenMPI, CUDA 12.6, libfabric, libcxi).
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1
module load brics/apptainer-multi-node

echo "SINGULARITY_BINDPATH set: $(echo $SINGULARITY_BINDPATH | wc -c) chars"

PY=${REPO_DIR}/.venv/bin/python3
source ${REPO_DIR}/pipeline_env_activate.sh >/dev/null 2>&1

$PY ${REPO_DIR}/scripts/nccl_compare_runner_adaptsh.py \
    --collectives all_reduce,alltoall \
    --node-counts 2,4 \
    --image /projects/a5k/public/containers/pytorch_25.10-py3.sif

echo ""
echo "================================"
echo "Compare done"
echo "================================"
