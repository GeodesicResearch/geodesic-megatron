#!/bin/bash
#SBATCH --job-name=container_smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container_smoke-%j.out

# ==============================================================================
# Container smoke test: single-node PyTorch + GPU + TE check
#
# Verifies container's PyTorch can use GPUs without external NCCL bind-mount.
#
# We DON'T use brics/apptainer-multi-node here because it bind-mounts the
# host's NCCL 2.26.6 over the container's newer NCCL, causing ABI breaks
# (e.g., undefined symbol: ncclCommWindowDeregister). The container ships
# its own NCCL that matches its PyTorch build. We let it use that.
#
# Slingshot/CXI fabric integration is NOT tested here — that's distest's job.
# ==============================================================================

set -euo pipefail

IMAGE="${1:-/projects/a5k/public/containers/pytorch_25.10-py3.sif}"
if [ ! -f "$IMAGE" ]; then
    echo "FATAL: image not found: $IMAGE" >&2
    exit 1
fi

# Bare-metal env may have set LD_PRELOAD pointing to host paths.
# Those don't exist inside the container — strip them.
unset LD_PRELOAD
unset LD_LIBRARY_PATH

echo "================================"
echo "Container Smoke Test (single node)"
echo "Image:    $IMAGE"
echo "Job ID:   $SLURM_JOB_ID"
echo "================================"

apptainer exec --nv \
    --no-mount tmp \
    "$IMAGE" \
    bash -c '
set -e
echo "--- Inside container ---"
echo "Hostname: $(hostname)"
echo ""
echo "=== CUDA toolkit ==="
nvcc --version 2>&1 | tail -3
echo ""
echo "=== Driver / GPU ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader
echo ""
echo "=== PyTorch + CUDA + NCCL ==="
python3 -c "
import torch
print(f\"torch: {torch.__version__}\")
print(f\"torch.version.cuda: {torch.version.cuda}\")
print(f\"torch.cuda.is_available: {torch.cuda.is_available()}\")
print(f\"torch.cuda.device_count: {torch.cuda.device_count()}\")
print(f\"NCCL version: {torch.cuda.nccl.version()}\")
x = torch.randn(8192, 8192, device=\"cuda\", dtype=torch.bfloat16)
y = x @ x.T
torch.cuda.synchronize()
print(f\"BF16 matmul OK: {y.shape}, sum={y.sum().item():.4f}\")
"
echo ""
echo "=== Transformer Engine ==="
python3 -c "
import transformer_engine as te
import transformer_engine.pytorch as te_pt
print(f\"transformer_engine: {te.__version__}\")
# Try to instantiate FP8 hybrid recipe
from transformer_engine.common.recipe import DelayedScaling, Format
recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID)
print(f\"DelayedScaling FP8 hybrid: instantiated OK\")
# Try blockwise FP8 recipe (the goal)
try:
    from transformer_engine.common.recipe import Float8BlockScaling
    bw = Float8BlockScaling()
    print(f\"Float8BlockScaling: AVAILABLE — unlocks paper recipe\")
except ImportError:
    try:
        from transformer_engine.common.recipe import BlockScaling
        bw = BlockScaling()
        print(f\"BlockScaling: AVAILABLE — unlocks paper recipe\")
    except ImportError:
        print(f\"Blockwise FP8: NOT in this TE version\")
# MXFP8 (Blackwell-native)
try:
    from transformer_engine.common.recipe import MXFP8BlockScaling
    print(f\"MXFP8BlockScaling: AVAILABLE\")
except ImportError:
    print(f\"MXFP8: NOT in this TE version\")
"
echo ""
echo "=== Key library versions ==="
python3 -c "
import importlib.metadata as m
for pkg in [\"torch\", \"transformer_engine\", \"flash-attn\", \"mamba-ssm\", \"causal-conv1d\", \"nv-grouped-gemm\", \"nvidia-resiliency-ext\", \"megatron-core\"]:
    try:
        v = m.version(pkg)
        print(f\"  {pkg}: {v}\")
    except m.PackageNotFoundError:
        print(f\"  {pkg}: NOT INSTALLED\")
"
echo ""
echo "=== FP8 GEMM perf (single GPU) ==="
python3 -c "
import time, torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

device = torch.device(\"cuda:0\")
torch.cuda.set_device(device)

# BF16 baseline
layer = torch.nn.Linear(8192, 8192, bias=False, dtype=torch.bfloat16).to(device)
x = torch.randn(2048, 8192, dtype=torch.bfloat16, device=device)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    y = layer(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
flops = 100 * 2 * 2048 * 8192 * 8192
print(f\"  BF16 8192x8192 @ MBS=2048 x100: {elapsed:.3f}s, {flops/elapsed/1e12:.0f} TFLOPs/GPU\")

# TE FP8 hybrid
te_layer = te.Linear(8192, 8192, bias=False, params_dtype=torch.bfloat16).to(device)
recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID)
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        y = te_layer(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
print(f\"  FP8 hybrid 8192x8192 @ MBS=2048 x100: {elapsed:.3f}s, {flops/elapsed/1e12:.0f} TFLOPs/GPU\")

# TE FP8 blockwise (goal)
try:
    from transformer_engine.common.recipe import Float8BlockScaling
    recipe_bw = Float8BlockScaling()
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe_bw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            y = te_layer(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    print(f\"  FP8 BLOCKWISE 8192x8192 @ MBS=2048 x100: {elapsed:.3f}s, {flops/elapsed/1e12:.0f} TFLOPs/GPU\")
    print(f\"  ★ Blockwise FP8 GEMM works — paper-recommended Hopper recipe is unlocked\")
except Exception as e:
    print(f\"  Blockwise FP8 FAIL: {type(e).__name__}: {e}\")
"
'
exit_code=$?
echo ""
echo "================================"
echo "Smoke test exit code: $exit_code"
echo "================================"
exit $exit_code
