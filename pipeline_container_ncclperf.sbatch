#!/bin/bash
#SBATCH --job-name=container_ncclperf
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container_ncclperf-%j.out

# ==============================================================================
# Run standard NCCL perf tests INSIDE the NGC container.
#
# The bare-metal binary at /home/a5k/kyleobrien.a5k/isambard-nccl-tests/build/
# links against:
#   - Brics NCCL 2.26.6  (need to override with container's NCCL 2.27.7)
#   - CUDA 12.6          (need to override with container's CUDA 13.0)
#   - cray-mpich         (need to swap MPI for direct srun launch)
#
# Strategy: bind nccl-tests dir, but DO NOT bind brics NCCL or host CUDA.
# Container's libs win via LD_LIBRARY_PATH ordering. Use srun for launch
# (one rank per GPU) instead of mpirun, since cray-mpich isn't in container.
# Many NCCL versions are ABI-compatible across minor versions; backward-compat
# from 2.26 ABI to 2.27 binary should work.
# ==============================================================================

set -euo pipefail

IMAGE="${1:-/projects/a5k/public/containers/pytorch_25.10-py3.sif}"
NCCL_TESTS_DIR=/home/a5k/kyleobrien.a5k/isambard-nccl-tests
TEST_BIN="${TEST_BIN:-all_reduce_perf}"

unset LD_PRELOAD
unset LD_LIBRARY_PATH

echo "================================"
echo "Container NCCL Perf"
echo "Image:    $IMAGE"
echo "Test:     $TEST_BIN"
echo "Nodes:    $SLURM_NNODES (4 GPU/node = $((SLURM_NNODES * 4)) total ranks)"
echo "================================"

export MASTER_ADDR="$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)"
export MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi
export NCCL_SOCKET_IFNAME=hsn
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_GDRCOPY_ENABLE=1
export NCCL_NET_FORCE_FLUSH=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_HMEM_CUDA_USE_GDRCOPY=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=1024
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
export NCCL_DEBUG=WARN
export NCCL_NET_PLUGIN=/host/aws-ofi-nccl/lib/libnccl-net.so

APPTAINER_BINDS="--bind ${NCCL_TESTS_DIR}:${NCCL_TESTS_DIR}"
APPTAINER_BINDS="$APPTAINER_BINDS --bind /opt/cray:/opt/cray:ro"
APPTAINER_BINDS="$APPTAINER_BINDS --bind /tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/aws-ofi-nccl-1.8.1-c47cd5ivrugm3jzlyqyis4igyflnydmo:/host/aws-ofi-nccl:ro"
APPTAINER_BINDS="$APPTAINER_BINDS --bind /usr/lib64:/host/usr/lib64:ro"

# Container's NCCL is at /usr/local/nccl/lib (or similar); rely on container's default.
# Add only Slingshot bits to LD_LIBRARY_PATH (container's NCCL stays the default).
export APPTAINERENV_LD_LIBRARY_PATH="/host/aws-ofi-nccl/lib:/host/usr/lib64:/opt/cray/libfabric/1.22.0/lib64"

echo ""
echo "=== Run: $TEST_BIN -b 32K -e 8G -f 2 -g 1 (one GPU per rank, srun launch) ==="
echo ""

srun --nodes=$SLURM_NNODES --ntasks-per-node=4 --kill-on-bad-exit=1 --export=ALL --overlap \
    apptainer exec --nv \
        $APPTAINER_BINDS \
        "$IMAGE" \
        bash -c "
            export PYTHONUNBUFFERED=1
            ${NCCL_TESTS_DIR}/build/${TEST_BIN} -b 32K -e 8G -f 2 -g 1
        "

echo ""
echo "================================"
echo "NCCL perf done"
echo "================================"
