#!/bin/bash
#SBATCH --job-name=container_buildncclts
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container_buildncclts-%j.out

# ==============================================================================
# Build nccl-tests INSIDE the container against /host/* (host NCCL + OpenMPI).
# Per Isambard docs https://docs.isambard.ac.uk/.../containers/apptainer-multi-node/:
#   Their example uses `nccl-tests/build/all_reduce_perf` built inside container,
#   linked against host NCCL 2.26.6 + host OpenMPI 4.1.7 + host CUDA 12.6.
#
# Output: /home/a5k/kyleobrien.a5k/isambard-nccl-tests-container-built/build/
# ==============================================================================

set -eo pipefail

unset LD_PRELOAD
unset LD_LIBRARY_PATH

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
SRC_DIR=/home/a5k/kyleobrien.a5k/isambard-nccl-tests-container-built
IMAGE=/projects/a5k/public/containers/pytorch_25.10-py3.sif

module purge
module load brics/apptainer-multi-node

echo "================================"
echo "Build nccl-tests inside container"
echo "Source dir: $SRC_DIR"
echo "Image:      $IMAGE"
echo "================================"

# Clone fresh source if not already there
if [ ! -d "$SRC_DIR" ]; then
    git clone https://github.com/NVIDIA/nccl-tests.git "$SRC_DIR"
fi

# Clean prior build
rm -rf "$SRC_DIR/build"

srun --ntasks=1 apptainer exec --nv \
    --bind /opt/cray:/opt/cray:ro \
    --bind "$SRC_DIR:$SRC_DIR" \
    "$IMAGE" \
    /host/adapt.sh bash -c "
        set -e
        cd $SRC_DIR
        echo '--- Build env ---'
        which mpicxx
        mpicxx --version | head -1
        echo \"NCCL_HOME=\$NCCL_HOME\"
        echo \"MPI_HOME=\$MPI_HOME\"
        ls /host/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/bin/nvcc 2>&1 || echo 'no nvcc on host bind path'
        ls /usr/local/cuda-12*/bin/nvcc 2>&1 || true
        # Use container's nvcc if it can target compute_90 with libcudart from host CUDA 12.6
        # Actually, use container's nvcc but point CUDA_HOME at /host/opt/nvidia/.../cuda/12.6
        export CUDA_HOME=/host/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
        export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
        echo \"--- Building (CUDA_HOME=\$CUDA_HOME) ---\"
        # NVCC fallback: if /host has it, use that; else container's nvcc
        if [ -x \"\$CUDA_HOME/bin/nvcc\" ]; then
            export PATH=\$CUDA_HOME/bin:\$PATH
            echo 'Using host nvcc (CUDA 12.6)'
        else
            echo 'Using container nvcc'
        fi
        nvcc --version | tail -2
        make -j 8 MPI=1 NCCL_HOME=/host/nccl CUDA_HOME=\$CUDA_HOME 2>&1 | tail -30
        echo '--- Build artifacts ---'
        ls -la build/ | head -15
        echo '--- ldd of all_reduce_perf ---'
        ldd build/all_reduce_perf | head -20
    "

echo ""
echo "================================"
echo "Build complete"
echo "Binaries at: $SRC_DIR/build/"
echo "================================"
