#!/bin/bash
#SBATCH --job-name=container_distest
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container_distest-%j.out

# ==============================================================================
# Container distributed test: NCCL + Slingshot/CXI from inside container
#
# Strategy: use the CONTAINER's own NCCL (newer, ABI-matches its PyTorch),
# but bind in just the Slingshot pieces — libfabric + aws-ofi-nccl plugin.
# Container's NCCL discovers the plugin via NCCL_NET_PLUGIN env var and uses
# Slingshot transport.
#
# We do NOT use brics/apptainer-multi-node module because its NCCL bind-mount
# (NCCL 2.26.6) is older than the container's NCCL and shadows it via
# LD_LIBRARY_PATH, breaking ABI (missing ncclCommWindowDeregister).
# ==============================================================================

set -euo pipefail

IMAGE="${1:-/projects/a5k/public/containers/pytorch_25.10-py3.sif}"
if [ ! -f "$IMAGE" ]; then
    echo "FATAL: image not found: $IMAGE" >&2
    exit 1
fi

# Strip any bare-metal env that leaks bad LD_PRELOAD/paths
unset LD_PRELOAD
unset LD_LIBRARY_PATH

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron

echo "================================"
echo "Container Distributed Test"
echo "Image:    $IMAGE"
echo "Nodes:    $SLURM_NNODES"
echo "GPUs:     $SLURM_GPUS_PER_NODE per node"
echo "================================"

# Master coordination
export MASTER_ADDR="$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)"
export MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"

# Slingshot/CXI fabric env (matches pipeline_training_launch.sh's set)
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

# Container needs the aws-ofi-nccl plugin loaded to use libfabric/cxi.
# Tell container's NCCL where to find the plugin (.so):
export NCCL_NET_PLUGIN=/host/aws-ofi-nccl/lib/libnccl-net.so

# Bindings for Slingshot/CXI fabric:
#   /opt/cray:           Cray libfabric
#   /host/aws-ofi-nccl:  the libnccl-net.so plugin (NCCL net plugin for libfabric/CXI)
#   /usr/lib64:          libcxi.so.1 (Slingshot kernel driver shim) + supporting libs
APPTAINER_BINDS="--bind ${REPO_DIR}:${REPO_DIR}"
APPTAINER_BINDS="$APPTAINER_BINDS --bind /opt/cray:/opt/cray:ro"
APPTAINER_BINDS="$APPTAINER_BINDS --bind /tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/aws-ofi-nccl-1.8.1-c47cd5ivrugm3jzlyqyis4igyflnydmo:/host/aws-ofi-nccl:ro"
APPTAINER_BINDS="$APPTAINER_BINDS --bind /usr/lib64:/host/usr/lib64:ro"

# Inside container: add host libs to LD_LIBRARY_PATH so plugin can dlopen libcxi/libfabric.
# Order: container's own libs first (preserves container's NCCL/CUDA), then host plugin paths.
export APPTAINERENV_LD_LIBRARY_PATH="/host/aws-ofi-nccl/lib:/host/usr/lib64:/opt/cray/libfabric/1.22.0/lib64"

echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "NCCL_NET_PLUGIN: $NCCL_NET_PLUGIN"
echo ""

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL --overlap \
    apptainer exec --nv \
        $APPTAINER_BINDS \
        "$IMAGE" \
        bash -c "
            export PYTHONUNBUFFERED=1
            python3 -m torch.distributed.run \
                --nproc_per_node=$SLURM_GPUS_PER_NODE \
                --nnodes=$SLURM_NNODES \
                --node_rank=\$SLURM_NODEID \
                --master_addr=$MASTER_ADDR \
                --master_port=$MASTER_PORT \
                ${REPO_DIR}/scripts/container_distest.py
        "

echo ""
echo "================================"
echo "Distributed test complete"
echo "================================"
