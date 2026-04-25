#!/bin/bash
#SBATCH --job-name=container_compat_smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container_compat_smoke-%j.out

# ==============================================================================
# 1-node smoke test: confirm container's CUDA 13 binaries can run on Isambard's
# CUDA 12.7 driver via /usr/local/cuda/compat/lib.real forward-compat shim.
#
# Tests three escalating things:
#   1. nccl-tests binary loads (libcuda dlopen succeeds via compat)
#   2. CUDA context creation works
#   3. Single-node NCCL runs (-g 4, intra-node only, no Slingshot)
# ==============================================================================

set -eo pipefail

unset LD_PRELOAD
unset LD_LIBRARY_PATH

IMAGE=/projects/a5k/public/containers/pytorch_25.10-py3.sif

echo "================================"
echo "Container compat-shim smoke test"
echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "Image:  $IMAGE"
echo "================================"

echo ""
echo "=== Test 1: container's libcuda forward-compat path ==="
srun --ntasks=1 apptainer exec --nv \
    --bind /usr/lib64:/host/usr/lib64:ro \
    "$IMAGE" \
    bash -c '
        export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/cuda/compat/lib:$LD_LIBRARY_PATH
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
        echo "libcuda resolved to:"
        ldconfig -p 2>/dev/null | grep libcuda.so | head -3
        echo "nvidia-smi via container:"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -2
    '

echo ""
echo "=== Test 2: container all_reduce_perf binary loads (--help) ==="
srun --ntasks=1 apptainer exec --nv \
    --bind /usr/lib64:/host/usr/lib64:ro \
    "$IMAGE" \
    bash -c '
        export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/cuda/compat/lib:$LD_LIBRARY_PATH
        ldd /usr/local/bin/all_reduce_perf | grep -E "cuda|nccl|libfabric"
        echo "---"
        /usr/local/bin/all_reduce_perf --help 2>&1 | head -10 || echo "exit=$?"
    '

echo ""
echo "=== Test 3: single-node intra-node all_reduce (-g 4, no Slingshot) ==="
srun --ntasks=1 --gpus-per-node=4 apptainer exec --nv \
    --bind /usr/lib64:/host/usr/lib64:ro \
    "$IMAGE" \
    bash -c '
        export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/cuda/compat/lib:$LD_LIBRARY_PATH
        export NCCL_DEBUG=WARN
        /usr/local/bin/all_reduce_perf -b 32K -e 256M -f 2 -g 4 2>&1 | tail -20
    '

echo ""
echo "================================"
echo "Smoke complete"
echo "================================"
