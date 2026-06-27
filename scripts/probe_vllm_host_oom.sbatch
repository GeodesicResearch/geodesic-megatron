#!/bin/bash
#SBATCH --job-name=vllm-oom-probe
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/vllm-oom-probe-%j.out
# One-off diagnostic: sample the JOB CGROUP's memory.stat while vLLM loads the
# Super 120B, to attribute the host-OOM (anon vs file vs unevictable/pinned).
set -uo pipefail
REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO"
source pipeline_env_activate.sh
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 VLLM_TORCH_COMPILE_DISABLE=1
export TORCH_CUDA_ARCH_LIST=9.0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR=/tmp/probe_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton"; mkdir -p "$TRITON_CACHE_DIR"

CG=$(awk -F: '{print $3}' /proc/self/cgroup | head -1)
echo "cgroup: $CG"
( while true; do
    S=/sys/fs/cgroup${CG}/memory.stat
    C=/sys/fs/cgroup${CG}/memory.current
    [ -f "$S" ] || S=$(dirname /sys/fs/cgroup${CG})/memory.stat
    echo "=== $(date +%H:%M:%S) current=$(cat $C 2>/dev/null) ==="
    grep -E "^(anon|file|shmem|unevictable|slab|pinned|kernel) " "$S" 2>/dev/null
    ps -eo rss,comm --sort=-rss 2>/dev/null | head -4
    sleep 10
  done ) &
SAMPLER=$!
python pipeline_coherence_test.py nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
  --backend vllm --tp 4 --max-tokens 64 --num-prompts 1 --trust-remote-code \
  --run-name probe-vllm-oom-super
RC=$?
kill $SAMPLER 2>/dev/null
echo "PROBE_RC=$RC"
