#!/bin/bash
#SBATCH --job-name=container_full_smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container_full_smoke-%j.out

# ==============================================================================
# 1-node smoke validating the FULL v5b container setup:
# - compat shim for libcuda (CUDA 13 on driver 565.57.01)
# - aws-ofi-nccl plugin (host's 1.8.1) loaded by container's NCCL 2.27.7
# - NCCL_NET_PLUGIN env var
# - 4-GPU intra-node all_reduce
#
# Validates aws-ofi-nccl ABI compat with NCCL 2.27.7 (built against 2.26.6).
# If this works, v5b multi-node will too (just adds Slingshot transport).
# ==============================================================================

set -eo pipefail
unset LD_PRELOAD
unset LD_LIBRARY_PATH

IMAGE=/projects/a5k/public/containers/pytorch_25.10-py3.sif

echo "================================"
echo "Container FULL smoke: NCCL 2.27.7 + aws-ofi-nccl 1.8.1 + compat shim"
echo "================================"

# Slingshot/CXI vars (same as v5b runner)
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
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_NET_PLUGIN=/host/aws-ofi-nccl/lib/libnccl-net.so

# Pass NCCL/CXI env into apptainer
for v in NCCL_NET FI_PROVIDER NCCL_SOCKET_IFNAME NCCL_CROSS_NIC NCCL_NET_GDR_LEVEL \
         NCCL_GDRCOPY_ENABLE NCCL_NET_FORCE_FLUSH FI_MR_CACHE_MONITOR \
         FI_CXI_DISABLE_HOST_REGISTER FI_HMEM_CUDA_USE_GDRCOPY \
         FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_RDZV_PROTO \
         FI_CXI_DISABLE_NON_INJECT_MSG_IDC NCCL_DEBUG NCCL_DEBUG_SUBSYS \
         NCCL_NET_PLUGIN; do
    eval "export APPTAINERENV_${v}=\"\${$v}\""
done

srun --ntasks=1 --gpus-per-node=4 apptainer exec --nv \
    --bind /opt/cray:/opt/cray:ro \
    --bind /tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/aws-ofi-nccl-1.8.1-c47cd5ivrugm3jzlyqyis4igyflnydmo:/host/aws-ofi-nccl:ro \
    --bind /usr/lib64:/host/usr/lib64:ro \
    "$IMAGE" \
    bash -c '
        export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/cuda/compat/lib:/host/aws-ofi-nccl/lib:/opt/cray/libfabric/1.22.0/lib64:/host/usr/lib64:$LD_LIBRARY_PATH
        echo "--- LD_LIBRARY_PATH (first 5) ---"
        echo "$LD_LIBRARY_PATH" | tr ":" "\n" | head -5
        echo "--- ldd nccl-tests ---"
        ldd /usr/local/bin/all_reduce_perf 2>&1 | grep -E "cudart|nccl"
        echo "--- aws-ofi-nccl plugin existence ---"
        ls -la /host/aws-ofi-nccl/lib/libnccl-net.so
        echo "--- libcxi from host ---"
        ls /host/usr/lib64/libcxi.so* 2>&1 | head -3
        echo "--- 4-GPU intra-node all_reduce ---"
        /usr/local/bin/all_reduce_perf -b 32K -e 256M -f 2 -g 4 2>&1 | tail -30
    '

echo ""
echo "================================"
echo "Full smoke complete"
echo "================================"
