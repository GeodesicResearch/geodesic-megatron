#!/bin/bash
# ==============================================================================
# Training launcher for inside the NGC pytorch:25.10-py3 container.
#
# Mirrors pipeline_training_launch.sh logic (NCCL/CXI env, ft_launcher, srun)
# but wraps the per-node bash in apptainer exec --nv. Uses container's CUDA 13
# + NCCL 2.27.7 + TE 2.8 via the /usr/local/cuda/compat/lib.real forward-compat
# shim. Python deps come from the prebuilt /projects/a5k/public/venv-container.
#
# Usage (from inside an existing SLURM allocation):
#   bash pipeline_container_training_launch.sh \
#     configs/sfm/sfm_nemotron_120b_cpt_misalignment.yaml \
#     --model super --mode cpt
#
# Or via SLURM submit:
#   isambard_sbatch --nodes=16 pipeline_container_training_submit.sbatch \
#     configs/<config>.yaml super cpt
# ==============================================================================

set -eo pipefail

# Same arg parsing as pipeline_training_launch.sh
CONFIG_FILE=""
MODEL=""
MODE=""
DISABLE_FT=false
ENABLE_PAO=false
PEFT=""
MAX_SAMPLES=""
NNODES_OVERRIDE=""
OVERRIDE_NODELIST=""

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --mode) MODE="$2"; shift 2;;
        --disable-ft) DISABLE_FT=true; shift;;
        --enable-pao) ENABLE_PAO=true; shift;;
        --peft) PEFT="$2"; shift 2;;
        --max-samples) MAX_SAMPLES="$2"; shift 2;;
        --nodes) NNODES_OVERRIDE="$2"; shift 2;;
        --nodelist) OVERRIDE_NODELIST="$2"; shift 2;;
        *) POSITIONAL+=("$1"); shift;;
    esac
done
CONFIG_FILE="${POSITIONAL[0]:-}"

if [ -z "$CONFIG_FILE" ] || [ -z "$MODEL" ] || [ -z "$MODE" ]; then
    echo "Usage: $0 <config.yaml> --model nano|super --mode sft|cpt [--disable-ft] [--enable-pao] [--peft lora] [--max-samples N] [--nodes N] [--nodelist LIST]" >&2
    exit 1
fi

# ==============================================================================
# Paths
# ==============================================================================
REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
IMAGE=/projects/a5k/public/containers/pytorch_26.03-py3.sif
VENV_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv-container-2603
NCCL_PLUGIN=/host/aws-ofi-nccl/lib/libnccl-net.so

# ==============================================================================
# SLURM context (must be inside an allocation)
# ==============================================================================
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "FATAL: SLURM_JOB_ID not set. Must run inside a SLURM allocation." >&2
    exit 1
fi

NNODES="${NNODES_OVERRIDE:-$SLURM_NNODES}"
NODELIST="${SLURM_NODELIST}"
export MASTER_ADDR="${MASTER_ADDR_OVERRIDE:-$(scontrol show hostname "$NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT_OVERRIDE:-$((29500 + SLURM_JOB_ID % 1000))}"

# ==============================================================================
# Modules + host environment
# ==============================================================================
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

# Slingshot/CXI env (these propagate via APPTAINERENV_* to container)
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
export TORCH_NCCL_TIMEOUT=900
export TORCH_NCCL_RETHROW_CUDA_ERRORS=1
export NCCL_NET_PLUGIN="$NCCL_PLUGIN"
export TORCH_CUDA_ARCH_LIST=9.0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_CPU_OFFLOAD_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UB_SKIPMC=1

# Pass into container
for v in NCCL_NET FI_PROVIDER NCCL_SOCKET_IFNAME NCCL_CROSS_NIC NCCL_NET_GDR_LEVEL \
         NCCL_GDRCOPY_ENABLE NCCL_NET_FORCE_FLUSH FI_MR_CACHE_MONITOR \
         FI_CXI_DISABLE_HOST_REGISTER FI_HMEM_CUDA_USE_GDRCOPY \
         FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_RDZV_PROTO \
         FI_CXI_DISABLE_NON_INJECT_MSG_IDC NCCL_DEBUG TORCH_NCCL_TIMEOUT \
         TORCH_NCCL_RETHROW_CUDA_ERRORS NCCL_NET_PLUGIN TORCH_CUDA_ARCH_LIST \
         CUDA_DEVICE_MAX_CONNECTIONS NVTE_CPU_OFFLOAD_V1 PYTORCH_CUDA_ALLOC_CONF \
         UB_SKIPMC MASTER_ADDR MASTER_PORT; do
    eval "export APPTAINERENV_${v}=\"\${$v}\""
done

# ==============================================================================
# Build script + script args (mirrors pipeline_training_launch.sh)
# ==============================================================================
TRAIN_SCRIPT="${REPO_DIR}/pipeline_training_run.py"
SCRIPT_ARGS="--config-file $CONFIG_FILE --model $MODEL --mode $MODE"
[ "$DISABLE_FT" = true ] && SCRIPT_ARGS="$SCRIPT_ARGS --disable-ft"
[ "$ENABLE_PAO" = true ] && SCRIPT_ARGS="$SCRIPT_ARGS --enable-pao"
[ -n "$PEFT" ] && SCRIPT_ARGS="$SCRIPT_ARGS --peft $PEFT"
[ -n "$MAX_SAMPLES" ] && SCRIPT_ARGS="$SCRIPT_ARGS --max-samples $MAX_SAMPLES"
USE_FT=true
[ "$DISABLE_FT" = true ] && USE_FT=false

# ==============================================================================
# Apptainer binds
# ==============================================================================
BINDS="--bind /home:/home"
BINDS="$BINDS --bind /projects:/projects"
# /projects/a5k/public is a symlink → /lus/lfs1aip2/projects/public/a5k.
# Without binding /lus, the symlink target is invisible inside container.
BINDS="$BINDS --bind /lus:/lus"
BINDS="$BINDS --bind /opt/cray:/opt/cray:ro"
BINDS="$BINDS --bind /tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/aws-ofi-nccl-1.8.1-c47cd5ivrugm3jzlyqyis4igyflnydmo:/host/aws-ofi-nccl:ro"
BINDS="$BINDS --bind /usr/lib64:/host/usr/lib64:ro"
# Bind libcuda.so.1 into ld's default search path so torch._inductor's
# JIT-linker (`-l:libcuda.so.1`) finds it without -L flags. The forward-compat
# libcuda.so.1 lives at /usr/local/cuda/compat/lib/ inside container — but
# /usr/lib/aarch64-linux-gnu/ is what `ld` searches by default.
COMPAT_LIBCUDA=/usr/local/cuda/compat/lib/libcuda.so.1
BINDS="$BINDS --bind /home/a5k/kyleobrien.a5k/geodesic-megatron/.venv-container-2603/triton_libcuda/libcuda.so:/usr/lib/aarch64-linux-gnu/libcuda.so"
BINDS="$BINDS --bind /home/a5k/kyleobrien.a5k/geodesic-megatron/.venv-container-2603/triton_libcuda/libcuda.so.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1"

echo "================================"
echo "===== Container Training ====="
echo "Job ID:      $SLURM_JOB_ID"
echo "Config:      $CONFIG_FILE"
echo "Model:       $MODEL"
echo "Mode:        $MODE"
echo "Nodes:       $NNODES"
echo "Master:      $MASTER_ADDR:$MASTER_PORT"
echo "Image:       $IMAGE"
echo "Venv:        $VENV_DIR"
echo "Launcher:    $([ "$USE_FT" = true ] && echo ft_launcher || echo torchrun)"
echo "================================"

# ==============================================================================
# Launch via srun → apptainer → venv → ft_launcher → pipeline_training_run.py
#
# LD_LIBRARY_PATH inside container: compat shim FIRST (CUDA 13 forward-compat
# libcuda) so --nv injection of host's older libcuda doesn't win. Then host
# fabric paths (aws-ofi-nccl, libfabric, libcxi). Then container default.
# ==============================================================================
SRUN_ARGS="--nodes=$NNODES --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL --overlap"
[ -n "$OVERRIDE_NODELIST" ] && SRUN_ARGS="$SRUN_ARGS --nodelist=$OVERRIDE_NODELIST"

INNER_LD="/usr/local/cuda/compat/lib.real:/usr/local/cuda/compat/lib:${VENV_DIR}/triton_libcuda:/host/aws-ofi-nccl/lib:/opt/cray/libfabric/1.22.0/lib64:/host/usr/lib64"
INNER_LIBRARY_PATH="/usr/local/cuda/compat/lib:/usr/local/cuda/lib64"

# Host's SSL_CERT_FILE = /etc/ssl/ca-bundle.pem (SLES path) — doesn't exist in
# container's Ubuntu base. Override with container's cert bundle so HF Hub
# downloads work for AutoConfig.from_pretrained.
export APPTAINERENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# Strip LD_PRELOAD inheritance — host's brics NCCL preload path doesn't exist
# inside container and triggers ld.so warnings that flood the log.
export APPTAINERENV_LD_PRELOAD=""

# Triton finds libcuda via ldconfig (which lists only /usr/lib paths by default,
# missing the CUDA-13 forward-compat lib at /usr/local/cuda/compat/lib).
export APPTAINERENV_TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib

# Container has gcc-13 not gcc-12 — torch.inductor + cpp_extension default to
# gcc-12 which doesn't exist. Override.
export APPTAINERENV_CC=/usr/bin/gcc-13
export APPTAINERENV_CXX=/usr/bin/g++-13
export APPTAINERENV_CUDAHOSTCXX=/usr/bin/g++-13
export APPTAINERENV_NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13 -L/usr/local/cuda/compat/lib"
# Plain ld (used by torch inductor) doesn't honor LIBRARY_PATH; pass -L via LDFLAGS.
export APPTAINERENV_LDFLAGS="-L/usr/local/cuda/compat/lib -L/usr/local/cuda/lib64"

if [ "$USE_FT" = true ]; then
    srun $SRUN_ARGS apptainer exec --nv $BINDS "$IMAGE" bash -c "
        export LD_LIBRARY_PATH=${INNER_LD}:\$LD_LIBRARY_PATH
        export LIBRARY_PATH=${INNER_LIBRARY_PATH}:\${LIBRARY_PATH:-}
        source ${VENV_DIR}/bin/activate
        cd $REPO_DIR
        export TMPDIR=/tmp/megatron_tmp_\$SLURM_JOB_ID
        mkdir -p \$TMPDIR
        # Node-local locks for HF config serialization (Lustre stale-handle fix).
        export MEGATRON_CONFIG_LOCK_DIR=/tmp/megatron_config_locks_\$SLURM_JOB_ID
        export TRITON_CACHE_DIR=/tmp/triton_cache_\$SLURM_JOB_ID
        export TRITON_HOME=/tmp/triton_home_\$SLURM_JOB_ID
        mkdir -p \$MEGATRON_CONFIG_LOCK_DIR \$TRITON_CACHE_DIR \$TRITON_HOME
        # ft_launcher in container is installed system-wide with #!/usr/bin/python
        # shebang — that bypasses our venv. Invoke as a module via venv python.
        python -m nvidia_resiliency_ext.fault_tolerance.launcher \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            --nproc_per_node=\$SLURM_GPUS_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_NODEID \
            --max-restarts=20 \
            --ft-initial-rank-heartbeat-timeout=86400 \
            --ft-rank-heartbeat-timeout=86400 \
            --ft-rank-section-timeouts=setup:10800,step:3600,checkpointing:3600 \
            --ft-rank-out-of-section-timeout=3600 \
            --ft-log-level=INFO \
            $TRAIN_SCRIPT \
            $SCRIPT_ARGS
    "
else
    srun $SRUN_ARGS apptainer exec --nv $BINDS "$IMAGE" bash -c "
        export LD_LIBRARY_PATH=${INNER_LD}:\$LD_LIBRARY_PATH
        export LIBRARY_PATH=${INNER_LIBRARY_PATH}:\${LIBRARY_PATH:-}
        source ${VENV_DIR}/bin/activate
        cd $REPO_DIR
        export TMPDIR=/tmp/megatron_tmp_\$SLURM_JOB_ID
        export MEGATRON_CONFIG_LOCK_DIR=/tmp/megatron_config_locks_\$SLURM_JOB_ID
        export TRITON_CACHE_DIR=/tmp/triton_cache_\$SLURM_JOB_ID
        export TRITON_HOME=/tmp/triton_home_\$SLURM_JOB_ID
        mkdir -p \$TMPDIR \$MEGATRON_CONFIG_LOCK_DIR \$TRITON_CACHE_DIR \$TRITON_HOME
        python -m torch.distributed.run \
            --nproc_per_node=\$SLURM_GPUS_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $TRAIN_SCRIPT \
            $SCRIPT_ARGS
    "
fi

echo "===== Container Training Complete ====="
