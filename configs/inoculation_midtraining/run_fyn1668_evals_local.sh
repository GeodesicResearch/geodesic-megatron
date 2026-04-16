#!/usr/bin/env bash
# run_fyn1668_evals_local.sh - Run Fyn1668 evals within an existing salloc allocation
#
# Sets up the sfm eval venv environment (vLLM, inspect, NCCL, etc.) and runs
# run_fyn1668_evals.sh against a local vLLM server on a compute node.
#
# Usage:
#   bash run_fyn1668_evals_local.sh MODEL_HF_PATH NODE [SLURM_JOB_ID]
#
# Examples:
#   bash run_fyn1668_evals_local.sh \
#       /projects/a5k/public/checkpoints/megatron/im_nemotron_30b_fyn1668_situational_awareness_em/iter_0000106/hf \
#       nid010229
#
#   # With explicit job ID (defaults to $SLURM_JOB_ID)
#   bash run_fyn1668_evals_local.sh /path/to/hf nid010229 3823381

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 MODEL_HF_PATH NODE [SLURM_JOB_ID]}"
NODE="${2:?Usage: $0 MODEL_HF_PATH NODE [SLURM_JOB_ID]}"
JOB_ID="${3:-${SLURM_JOB_ID:-}}"
if [ -z "$JOB_ID" ]; then
    echo "ERROR: No SLURM_JOB_ID set and none provided as arg 3"
    exit 1
fi

MODEL_SHORT=$(basename "$(dirname "$(dirname "$MODEL_PATH")")")
PORT=$((30000 + RANDOM % 5000))
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SFM_EVALS_DIR="/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals"

# ─── Environment Setup (mirrors run_bundled_checkpoint_eval.sbatch) ──────────

# Eval venv (has vLLM, lm_eval, inspect)
if [ -n "${SFM_EVAL_VENV:-}" ]; then
    EVAL_VENV="$SFM_EVAL_VENV"
else
    EVAL_VENV="/projects/a5k/public/data_${USER}/python_envs/sfm/.venv"
    if [ ! -f "$EVAL_VENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" ]; then
        CAM_VENV="/projects/a5k/public/data_cwtice.a5k/python_envs/sfm/.venv"
        if [ -f "$CAM_VENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" ]; then
            echo "User sfm venv missing libtorch_cuda.so; falling back to $CAM_VENV"
            EVAL_VENV="$CAM_VENV"
        fi
    fi
fi
echo "Using eval venv: $EVAL_VENV"
source "$EVAL_VENV/bin/activate"

# NCCL: bundled version over system NCCL
export LD_PRELOAD="${EVAL_VENV}/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"

# Modules
module purge 2>/dev/null || true
module load PrgEnv-cray 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true

# Compilers
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"

# NCCL / OFI settings for Slingshot
export NCCL_COLLNET_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi
export NCCL_SOCKET_IFNAME=hsn
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export UB_SKIPMC=1

# Paths
export XDG_DATA_HOME="/projects/a5k/public/data_${USER}/xdg_data"
export XDG_CACHE_HOME="/projects/a5k/public/data_${USER}/xdg_cache"
export WANDB_DIR="/projects/a5k/public/data_${USER}/wandb"
export TMPDIR="/tmp/fyn1668-eval-$$"
export XDG_RUNTIME_DIR="$TMPDIR"
export VLLM_CACHE_ROOT="$TMPDIR/vllm_cache"
export VLLM_TORCH_COMPILE_DISABLE=1
export PYTHONPATH="${SFM_EVALS_DIR}:${PYTHONPATH:-}"
mkdir -p "$WANDB_DIR" "$TMPDIR" "$VLLM_CACHE_ROOT"

# Load API keys for LLM judges
if [ -f "$SFM_EVALS_DIR/.env" ]; then
    export $(grep -v '^#' "$SFM_EVALS_DIR/.env" | xargs)
fi

# ─── Start vLLM Server ──────────────────────────────────────────────────────

echo "============================================================"
echo "Fyn1668 Local Eval: $MODEL_SHORT"
echo "Node: $NODE  Port: $PORT  Job: $JOB_ID"
echo "============================================================"

srun --jobid="$JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node=1 --nodelist="$NODE" --export=ALL \
    bash -c "source $EVAL_VENV/bin/activate && \
    export LD_PRELOAD='${EVAL_VENV}/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2' && \
    export TMPDIR=/tmp && \
    export VLLM_TORCH_COMPILE_DISABLE=1 && \
    vllm serve '$MODEL_PATH' --host 0.0.0.0 --port $PORT --dtype auto --enforce-eager \
    --tensor-parallel-size 1 --max-model-len 8192 --api-key inspectai \
    --trust-remote-code 2>&1" &
VLLM_PID=$!

echo "Waiting for vLLM on $NODE:$PORT..."
for i in $(seq 1 600); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited prematurely"
        wait "$VLLM_PID" || true
        exit 1
    fi
    if curl -s -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer inspectai" \
        "http://${NODE}:${PORT}/v1/models" 2>/dev/null | grep -q 200; then
        echo "vLLM ready on $NODE:$PORT after ${i}s"
        break
    fi
    if [ "$i" -eq 600 ]; then
        echo "ERROR: vLLM not ready within 600s"
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# ─── Run Evals ───────────────────────────────────────────────────────────────

export OPENAI_API_KEY="inspectai"
export OPENAI_BASE_URL="http://${NODE}:${PORT}/v1"
export VLLM_BASE_URL="http://${NODE}:${PORT}/v1"

cd "$REPO"
bash configs/inoculation_midtraining/run_fyn1668_evals.sh \
    "vllm/$MODEL_PATH" \
    --model-base-url "http://${NODE}:${PORT}/v1" \
    -M api_key=inspectai

# ─── Cleanup ─────────────────────────────────────────────────────────────────

echo "Stopping vLLM server..."
kill "$VLLM_PID" 2>/dev/null || true
rm -rf "$TMPDIR"
echo "============================================================"
echo "Fyn1668 Local Eval Complete: $MODEL_SHORT"
echo "============================================================"
