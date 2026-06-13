#!/bin/bash
#SBATCH --job-name=coherence-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/coherence-test-%j.out

# ==============================================================================
# Qualitative Generation Coherence Test (thin wrapper around pipeline_coherence_test.py)
#
# Generates responses to diverse prompts and logs them to W&B for qualitative
# comparison across models and checkpoints.
#
# Backends (forwarded to pipeline_coherence_test.py --backend):
#   hf (default)  device_map="auto" on this node's GPUs — models that fit one
#                 node (Nano 30B: 1 GPU; Super 120B: 4 GPUs).
#   megatron      bridge-load a *Megatron* checkpoint and generate via the
#                 Megatron forward pass across the WHOLE allocation
#                 (srun+torchrun, multi-node). Historically, vLLM <0.21 could
#                 not serve the BF16 hybrid due to PP>1 rank-sync bugs in the
#                 Ray executor; vLLM 0.22.1+ with RayExecutorV2 resolves this.
#                 See docs/ultra-550b-training-and-conversion.md.
#
# Usage:
#   isambard_sbatch pipeline_coherence_submit.sbatch <model_path> [options]
#
# Examples:
#   # HF Hub model (4 GPUs, device_map)
#   isambard_sbatch pipeline_coherence_submit.sbatch nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
#
#   # Single GPU (smaller models like Nano 30B)
#   isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch \
#     geodesic-research/nemotron_nano_sft_warm_start_200k
#
#   # Ultra 550B Megatron checkpoint (multi-node: TP=4 x PP=6 / EP=4 = 24 GPUs)
#   isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch \
#     /projects/a5k/public/checkpoints/megatron/<experiment> \
#     --backend megatron --hf-model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
#     --tokenizer geodesic-research/nemotron-instruct-tokenizer \
#     --tp 4 --pp 6 --ep 4 --max-tokens 256 --trust-remote-code
#
#   # vLLM DIRECTLY, single node (Super 120B, TP=4; needs the vllm-coherence venv —
#   # scripts/setup_vllm_coherence_venv.sh):
#   isambard_sbatch --nodes=1 pipeline_coherence_submit.sbatch \
#     nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --backend vllm --tp 4 --max-tokens 256
#
#   # vLLM DIRECTLY, multi-node (Ultra 550B BF16, TP=4 x PP=4 = 16 GPUs on 4 nodes;
#   # the launcher brings up a Ray cluster across the allocation first):
#   isambard_sbatch --nodes=4 pipeline_coherence_submit.sbatch \
#     /path/to/iter_NNNNNNN/hf --backend vllm --tp 4 --pp 4 --max-tokens 256 --trust-remote-code
# ==============================================================================

MODEL_PATH="${1:?Usage: isambard_sbatch pipeline_coherence_submit.sbatch <model_path> [options]}"
shift

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

# Peek for --backend megatron among the forwarded options (the script itself
# parses the flag; the launcher only needs it to pick the launch topology).
BACKEND=hf
prev=""
for a in "$@"; do
    [ "$prev" = "--backend" ] && BACKEND="$a"
    prev="$a"
done

source "$REPO_DIR/pipeline_env_activate.sh"
export PYTHONUNBUFFERED=1

echo "===== Coherence Test ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Backend:   $BACKEND"
echo "Nodes:     $SLURM_NNODES"
echo "Model:     $MODEL_PATH"
echo "Args:      $*"
echo "=========================="

if [ "$BACKEND" = "megatron" ]; then
    # Multi-node torchrun across the allocation (mirrors pipeline_checkpoint_convert.sh).
    module purge
    module load PrgEnv-cray
    module load cuda/12.6
    module load brics/aws-ofi-nccl/1.8.1
    source "$REPO_DIR/pipeline_env_activate.sh"
    # Slingshot/CXI NCCL (same proven subset as pipeline_checkpoint_convert.sh)
    export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB
    export FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_RX_MATCH_MODE=soft
    export NCCL_NVLS_ENABLE=0 NCCL_DEBUG=WARN MPICH_GPU_SUPPORT_ENABLED=0
    # Lazy NCCL comm-init on the first forward; PP<=12 at <=48 ranks needs far
    # less than training's PP=36/288 — 30 min is ample headroom.
    export TORCH_NCCL_TIMEOUT=1800
    # Neutralize Megatron's jit_fuser (= torch.compile on torch>=2.2): per-stage
    # compile divergence at PP causes rank desync (same class as training).
    export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
    export TMPDIR=/tmp/coh_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"
    export TRITON_CACHE_DIR="$TMPDIR/triton"; mkdir -p "$TRITON_CACHE_DIR"
    export MEGATRON_CONFIG_LOCK_DIR="$TMPDIR"

    NNODES=$SLURM_NNODES
    MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
    MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
    OUT="$REPO_DIR/logs/slurm/coherence-test-${SLURM_JOB_ID}-gens.txt"

    srun --nodes=$NNODES --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL bash -c "
        cd $REPO_DIR
        source pipeline_env_activate.sh
        export PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
        torchrun \
            --nproc_per_node=4 \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            pipeline_coherence_test.py $MODEL_PATH --output $OUT $*
    "
elif [ "$BACKEND" = "vllm" ]; then
    # In-process vLLM from the standard repo env (vLLM 0.22.1 installed in-env via
    # scripts/upgrade_env_vllm_in_place.sh; pipeline_env_activate.sh already set the
    # venv + LD_PRELOAD/NCCL paths above).
    python -c "import vllm" 2>/dev/null || { echo "FATAL: vllm not importable — run scripts/upgrade_env_vllm_in_place.sh"; exit 1; }
    export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 VLLM_TORCH_COMPILE_DISABLE=1
    export TORCH_CUDA_ARCH_LIST=9.0 VLLM_WORKER_MULTIPROC_METHOD=spawn
    # FlashInfer JIT guards: the sampler path also JITs; MAX_JOBS bounds ninja/nvcc
    # parallelism if any JIT remains (each cicc is 3-7 GB anon; unbounded = cgroup OOM).
    export VLLM_USE_FLASHINFER_SAMPLER=0 MAX_JOBS=4
    # Node-local caches: Triton autotune/JIT cache (Mamba2 SSD warmup) must not hit
    # read-only NFS HOME; ray workers inherit these via srun --export=ALL.
    export TMPDIR=/tmp/coh_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"
    export TRITON_CACHE_DIR="$TMPDIR/triton"; mkdir -p "$TRITON_CACHE_DIR"
    # ALL disk caches node-local: vLLM's filelock'd caches default under HOME
    # (~/.cache/vllm) — 32 Ray workers flock()ing NFS HOME throws
    # '[Errno 116] Stale file handle' during KV-cache profiling.
    export VLLM_CACHE_ROOT="$TMPDIR/vllm_cache"; mkdir -p "$VLLM_CACHE_ROOT"
    export XDG_CACHE_HOME="$TMPDIR/xdg"; mkdir -p "$XDG_CACHE_HOME"
    # flashinfer keeps its OWN JIT workspace under Path.home()/.cache/flashinfer
    # (ignores XDG/VLLM_CACHE_ROOT) — same NFS flock trap.
    export FLASHINFER_WORKSPACE_BASE="$TMPDIR"
    OUT="$REPO_DIR/logs/slurm/coherence-test-${SLURM_JOB_ID}-gens.txt"

    if [ "$SLURM_NNODES" -gt 1 ]; then
        # --- Multi-node: bring up a Ray cluster across the allocation, then run on the head ---
        # NCCL transport for cross-node TP/PP: socket-over-hsn is the proven default; set
        # COH_VLLM_NCCL=cxi to try AWS-OFI/libfabric (RayExecutorV2 propagates FI_* to workers).
        if [ "${COH_VLLM_NCCL:-socket}" = "cxi" ]; then
            export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB
            export FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_RX_MATCH_MODE=soft
        else
            export NCCL_NET=Socket
        fi
        export NCCL_SOCKET_IFNAME=hsn NCCL_NVLS_ENABLE=0 MPICH_GPU_SUPPORT_ENABLED=0
        export COH_VLLM_MULTINODE=1
        HEAD_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
        HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}' | head -1); [ -n "$HEAD_IP" ] || HEAD_IP=$(hostname -I | awk '{print $1}')
        export VLLM_HOST_IP="$HEAD_IP"
        RAY_TMP=/tmp/ray_${SLURM_JOB_ID}
        export RAY_gcs_server_request_timeout_seconds=120 RAY_raylet_start_wait_time_s=300
        GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
        EXPECTED=$((SLURM_NNODES * GPUS_PER_NODE))
        trap 'ray stop --force 2>/dev/null || true' EXIT
        # object-store cap: Ray defaults to ~30% of node RAM (~140 GB) in /dev/shm,
        # which counts against the job cgroup and squeezes weight-staging headroom.
        OBJ_STORE=$((20 * 1024 * 1024 * 1024))
        ray start --head --node-ip-address="$HEAD_IP" --port=6379 --temp-dir="$RAY_TMP" \
            --include-dashboard=false --num-gpus="$GPUS_PER_NODE" --disable-usage-stats \
            --object-store-memory="$OBJ_STORE" || true
        export RAY_ADDRESS="$HEAD_IP:6379"
        sleep 15
        srun --overlap --nodes=$((SLURM_NNODES - 1)) --ntasks-per-node=1 --exclude="$HEAD_NODE" bash -c '
            WIP=$(getent hosts "$(hostname)" | awk "{print \$1}" | head -1)
            export VLLM_HOST_IP="$WIP"
            exec ray start --address="'"$HEAD_IP"':6379" --node-ip-address="$WIP" \
                --temp-dir="'"$RAY_TMP"'" --num-gpus="'"$GPUS_PER_NODE"'" --block --object-store-memory='"$OBJ_STORE"'' &
        for i in $(seq 1 30); do
            ng=$(timeout 20 python -c "import ray; ray.init(address='auto'); print(int(ray.cluster_resources().get('GPU',0)))" 2>/dev/null | tail -1)
            echo "[vllm-coh] ray GPUs: ${ng:-?}/$EXPECTED"
            [ "${ng:-0}" = "$EXPECTED" ] && break
            sleep 10
        done
        [ "${ng:-0}" = "$EXPECTED" ] || { echo "FATAL: Ray cluster incomplete (${ng:-0}/$EXPECTED GPUs)"; exit 1; }
        python pipeline_coherence_test.py "$MODEL_PATH" --backend vllm --vllm-executor ray --output "$OUT" $(printf '%s ' "$@" | sed 's/--backend vllm //')
    else
        python pipeline_coherence_test.py "$MODEL_PATH" --output "$OUT" "$@"
    fi
else
    python pipeline_coherence_test.py "$MODEL_PATH" "$@"
fi

echo "===== Coherence Test Complete ====="
