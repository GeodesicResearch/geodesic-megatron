#!/bin/bash
# setup_megatron_bridge.sh - Install Megatron Bridge on Isambard ARM HPC
#
# Usage:
#   bash setup_megatron_bridge.sh
#
# Prerequisites:
#   - Run on a compute node with GPU access (required for CUDA kernel compilation)
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#
# This script will:
#   1. Load required modules and set compilers
#   2. Create a Python 3.12 virtual environment
#   3. Install PyTorch with CUDA support (aarch64)
#   4. Initialize Megatron-Core submodule
#   5. Install Megatron Bridge and all dependencies
#   6. Build transformer-engine, mamba-ssm, etc. from source
#   7. Apply ARM-specific patches (sm_90a, NCCL, wandb)
#   8. Run validation checks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Megatron Bridge Setup for Isambard (ARM)"
echo "=============================================="
echo "Architecture: $(uname -m)"
echo "Hostname: $(hostname)"
echo "Working directory: $SCRIPT_DIR"
echo ""

# ============================================
# Configuration
# ============================================
# GEODESIC_VENV_DIR builds the env at an alternate path without clobbering the shared
# .venv symlink target. Defaults to .venv.
VENV_DIR="${GEODESIC_VENV_DIR:-$SCRIPT_DIR/.venv}"
PYTHON_VERSION=3.12
VENV_SITE_PACKAGES="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages"
VENV_PYTHON="$VENV_DIR/bin/python"

# ============================================
# Phase 1: Module loading and compiler setup
# ============================================
echo "=== Phase 1: Loading modules and setting compilers ==="
# Note: Do NOT load brics/nccl during setup - torch bundles its own NCCL
# and loading system NCCL causes symbol conflicts (ncclCommShrink)
module load cuda/12.6 || echo "Warning: cuda/12.6 module not found"
module load cudatoolkit 2>/dev/null || true

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12  # Tell nvcc to use gcc-12 as host compiler (system gcc is 7.5, too old for <filesystem>)
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export TMPDIR=/projects/a5k/public/tmp
mkdir -p "$TMPDIR"

echo "CC=$CC"
echo "CXX=$CXX"
echo "CUDAHOSTCXX=$CUDAHOSTCXX"
echo "CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# Verify gcc-12 exists
if ! command -v "$CC" &> /dev/null; then
    echo "ERROR: gcc-12 not found at $CC"
    exit 1
fi

# ============================================
# Phase 2: Create Python 3.12 virtual environment
# ============================================
echo ""
echo "=== Phase 2: Creating virtual environment ==="
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "uv version: $(uv --version)"

if [ -d "$VENV_DIR" ]; then
    echo "Removing existing .venv directory..."
    rm -rf "$VENV_DIR"
fi

# Use --seed to include pip (needed for PyTorch index install)
uv venv --python "$PYTHON_VERSION" --seed "$VENV_DIR"
echo "Virtual environment created at: $VENV_DIR"

# ============================================
# Phase 3: NCCL preload path (torch is installed by uv, see Phase 6)
# ============================================
echo ""
echo "=== Phase 3: NCCL preload path (torch via uv) ==="
# torch/torchvision/torchaudio are EXACT-pinned in pyproject.toml and routed to the
# pytorch-cu126 index via [tool.uv.sources]; uv installs them in Phase 6 (pass 1) from
# uv.lock. No manual pip step, no PIP_CONSTRAINT — the lock + index pin the aarch64
# cu126 wheels, and `uv sync --locked` keeps the whole closure at the validated versions.
#
# Define NCCL_LIBRARY / LD_PRELOAD now (referenced by the uv-sync build env and Phase 7).
# The .so only exists after pass 1 installs nvidia-nccl-cu12; until then the loader
# prints a benign "cannot be preloaded ... ignored" warning.
export NCCL_LIBRARY="$VENV_SITE_PACKAGES/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# ============================================
# Phase 4: Initialize Megatron-Core submodule
# ============================================
echo ""
echo "=== Phase 4: Initializing Megatron-Core submodule ==="
if [ -f 3rdparty/Megatron-LM/pyproject.toml ] || [ -f 3rdparty/Megatron-LM/setup.py ]; then
    echo "Megatron-Core submodule already populated; skipping update"
else
    git submodule update --init 3rdparty/Megatron-LM
fi
echo "Megatron-Core submodule ready at 3rdparty/Megatron-LM"

# ============================================
# Phase 5: Set NVIDIA library paths for builds
# ============================================
echo ""
echo "=== Phase 5: Setting NVIDIA library paths ==="

# LD_LIBRARY_PATH for NVIDIA libraries from venv packages
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cufft/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/curand/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cusolver/lib:$LD_LIBRARY_PATH"

# Include paths for compilation (NCCL, cuDNN, cuBLAS)
# NCCL include is CRITICAL - TE's CMake needs nccl.h
export CPLUS_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/include:$VENV_SITE_PACKAGES/nvidia/cudnn/include:$VENV_SITE_PACKAGES/nvidia/cublas/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/include:$VENV_SITE_PACKAGES/nvidia/cudnn/include:$VENV_SITE_PACKAGES/nvidia/cublas/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/lib:$VENV_SITE_PACKAGES/nvidia/cudnn/lib:$VENV_SITE_PACKAGES/nvidia/cublas/lib:${LIBRARY_PATH:-}"
export CUDNN_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn"

# TE's CMake checks these env vars for NCCL location
export NVTE_NCCL_INCLUDE="$VENV_SITE_PACKAGES/nvidia/nccl/include"
export NVTE_NCCL_LIB="$VENV_SITE_PACKAGES/nvidia/nccl/lib"

echo "NCCL library (LD_PRELOAD): $NCCL_LIBRARY"
echo "cuDNN path: $CUDNN_PATH"

# NOTE: cuDNN header symlinks (into torch/include) are created in Phase 6b, after
# uv sync installs torch + its bundled cuDNN. They must exist before Phase 7 builds
# transformer-engine from source.

# Build env for uv's no-build-isolation source builds (TE/mamba/causal-conv1d/
# grouped-gemm/flash-linear-attention). EXPORTED so uv's build subprocesses inherit them.
# NOTE: NVTE_PROJECT_BUILDING is NOT exported globally — it makes TE's __init__ skip
# loading the core lib, which would break the post-build import verification (and any
# runtime import) if it leaked. It is set INLINE only on the pass-2 uv sync below.
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="9.0"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

# ============================================
# Phase 6: Install deps via uv sync (pass 1 — everything but the CUDA source builds)
# ============================================
echo ""
echo "=== Phase 6: uv sync pass 1 (torch + python deps + vLLM; CUDA exts deferred) ==="
echo "This may take several minutes..."
# All packages — including torch (cu126 index), vLLM (+cu129 wheel), numpy 2.3.5,
# transformers 5.10.2 — are pinned by uv.lock. `--locked` fails if pyproject and the
# committed aarch64 lock disagree (no silent drift). Pass 1 DEFERS the source-built
# CUDA extensions (--no-install-package) because their compile needs the cuDNN header
# symlinks that only exist once torch + nvidia-cudnn land here (created in Phase 6b).
uv sync --locked --group build \
    --no-install-package transformer-engine \
    --no-install-package transformer-engine-torch \
    --no-install-package mamba-ssm \
    --no-install-package causal-conv1d \
    --no-install-package nv-grouped-gemm \
    --no-install-package flash-linear-attention \
    --no-install-package flash_mla \
    2>&1 | tee /tmp/uv_sync_megatron.log | tail -30

UV_SYNC_EXIT=${PIPESTATUS[0]}
if [ "$UV_SYNC_EXIT" -ne 0 ]; then
    echo ""
    echo "WARNING: uv sync exited with code $UV_SYNC_EXIT"
    echo "Some packages may have failed to build. Phase 7 will install them individually."
    echo "Full log: /tmp/uv_sync_megatron.log"
fi

# ============================================
# Phase 6b: cuDNN symlinks + torch verification (post uv sync)
# ============================================
echo ""
echo "=== Phase 6b: cuDNN symlinks + torch verification ==="
# torch (and its bundled cuDNN) are now installed by uv sync. Create the cuDNN header
# symlinks into torch/include that the Phase 7 transformer-engine source build needs.
TORCH_INCLUDE="$VENV_SITE_PACKAGES/torch/include"
CUDNN_INCLUDE="$VENV_SITE_PACKAGES/nvidia/cudnn/include"
if [ -d "$CUDNN_INCLUDE" ] && [ -d "$TORCH_INCLUDE" ]; then
    echo "Creating cuDNN header symlinks in PyTorch include directory..."
    for f in "$CUDNN_INCLUDE"/*.h; do
        ln -sf "$f" "$TORCH_INCLUDE/$(basename "$f")" 2>/dev/null || true
    done
    echo "cuDNN symlinks created"
else
    echo "ERROR: torch/cuDNN not found after uv sync — uv did not install torch."
    echo "       Check pyproject.toml [tool.uv.sources] torch routing and the uv sync log."
    exit 1
fi

echo "Verifying PyTorch (installed by uv sync)..."
LD_PRELOAD="$NCCL_LIBRARY" "$VENV_PYTHON" -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Arch: {torch.cuda.get_device_capability(0)}')
" || {
    echo "ERROR: PyTorch CUDA verification failed after uv sync"
    exit 1
}

# Build deps for the Phase 7 source builds (pybind11/ninja/Cython/wheel/setuptools/
# nvidia-mathdx) come from the `build` dependency-group in pyproject.toml, synced via
# `--group build` in both passes (so they are uv-managed + locked, not pruned). The CUDA
# exts are `no-build-isolation` (they need the project's exact torch), so uv builds them
# against the project env — those build tools must already be present there, which the
# build group ensures. No `uv pip` step.
# uv --seed venvs ship no python3-config; symlink it from the base interpreter so the
# dataset-index helper Megatron JIT-builds at train start gets a correctly-suffixed .so.
PY_BASE_PREFIX="$("$VENV_PYTHON" -c 'import sys; print(sys.base_prefix)')"
if [ -x "$PY_BASE_PREFIX/bin/python3-config" ] && [ ! -e "$VENV_DIR/bin/python3-config" ]; then
    ln -sfn "$PY_BASE_PREFIX/bin/python3-config" "$VENV_DIR/bin/python3-config"
    echo "Linked python3-config from $PY_BASE_PREFIX"
fi

# ============================================
# Phase 7: Build the CUDA source extensions via uv (pass 2)
# ============================================
echo ""
echo "=== Phase 7: uv sync pass 2 (build CUDA extensions) ==="
echo "Builds transformer-engine, mamba-ssm, causal-conv1d, nv-grouped-gemm,"
echo "flash-linear-attention from source via uv (no-build-isolation). 15-25 min."
# Now that pass 1 installed torch + nvidia-cudnn and Phase 6b created the cuDNN header
# symlinks, run uv sync WITHOUT --no-install-package so uv builds the deferred CUDA
# extensions. Build tools come from the `build` dependency-group (--group build); the
# compile env (CC/CXX/CUDAHOSTCXX/arch exported above + the CUDA_HOME/NCCL/cuDNN paths
# below) is inherited by uv's build subprocesses. NVTE_PROJECT_BUILDING=1 is set INLINE
# here (only for the build) so TE's metadata-gen import does not try a runtime cuDNN load;
# it must NOT leak past this command or it breaks the verification + runtime imports.
# --locked keeps every version pinned to uv.lock.
CUDA_HOME="$CUDA_HOME" \
    NVTE_PROJECT_BUILDING=1 \
    LD_PRELOAD="$NCCL_LIBRARY" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH" \
    C_INCLUDE_PATH="$C_INCLUDE_PATH" \
    LIBRARY_PATH="$LIBRARY_PATH" \
    CUDNN_PATH="$CUDNN_PATH" \
    NVTE_NCCL_INCLUDE="$NVTE_NCCL_INCLUDE" \
    NVTE_NCCL_LIB="$NVTE_NCCL_LIB" \
    uv sync --locked --group build 2>&1 | tee /tmp/uv_sync_megatron_pass2.log | tail -40

if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "ERROR: uv sync pass 2 (CUDA extension build) failed. See /tmp/uv_sync_megatron_pass2.log"
    exit 1
fi

# Verify the source-built extensions + vLLM/Ray import (real binary checks).
LD_PRELOAD="$NCCL_LIBRARY" "$VENV_PYTHON" -c "
import transformer_engine, transformer_engine.pytorch; print('  transformer_engine', transformer_engine.__version__)
import mamba_ssm; print('  mamba-ssm: OK')
import causal_conv1d; print('  causal-conv1d: OK')
import grouped_gemm; print('  nv-grouped-gemm: OK')
import vllm, vllm._C, ray; print('  vLLM', vllm.__version__, '+ Ray', ray.__version__, ': OK')
" || { echo 'ERROR: post-build import verification failed'; exit 1; }
echo "CUDA extensions + vLLM/Ray: INSTALLED"

# ============================================
# Phase 8: Apply ARM-specific patches
# ============================================
echo ""
echo "=== Phase 8: Applying ARM patches ==="

# 8a. sitecustomize.py - GH200 sm_90a fix
echo "Installing GH200 sm_90a monkeypatch..."
cat > "$VENV_SITE_PACKAGES/sitecustomize.py" << 'SITECUSTOMIZE_EOF'
"""
GH200 sm_90a fix - Monkeypatch PyTorch's CUDA arch flag detection.

GH200 GPUs report sm_90a architecture, but PyTorch's _get_cuda_arch_flags()
in cpp_extension.py cannot parse the 'a' suffix. This causes:
  ValueError: invalid literal for int() with base 10: '90a'

This sitecustomize.py is loaded automatically at Python startup and:
1. Sets TORCH_CUDA_ARCH_LIST=9.0 as a fallback
2. Monkeypatches _get_cuda_arch_flags() to return correct flags for sm_90
"""
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

def _patch_pytorch_cuda_arch():
    """Replace PyTorch's _get_cuda_arch_flags with a version that handles sm_90a."""
    try:
        import torch.utils.cpp_extension as cpp_ext

        def _patched_get_cuda_arch_flags(cflags=None):
            return ['-gencode', 'arch=compute_90,code=sm_90']

        cpp_ext._get_cuda_arch_flags = _patched_get_cuda_arch_flags
    except (ImportError, AttributeError):
        pass

_patch_pytorch_cuda_arch()
SITECUSTOMIZE_EOF
echo "  sitecustomize.py installed"

# 8b. wandb isatty() patch for SLURM
echo "Applying wandb isatty patch..."
WANDB_TERM_FILE="$VENV_SITE_PACKAGES/wandb/errors/term.py"
if [ -f "$WANDB_TERM_FILE" ]; then
    sed -i 's/    return sys\.stderr\.isatty()/    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()/' "$WANDB_TERM_FILE" || true
    echo "  wandb patch applied"
else
    echo "  wandb term.py not found, skipping patch"
fi

# 8c. Create activate_env.sh helper
echo "Creating activate_env.sh..."
cat > "$SCRIPT_DIR/activate_env.sh" << ACTIVATE_EOF
#!/bin/bash
# Source this file to activate the Megatron Bridge environment
# Usage: source activate_env.sh

SCRIPT_DIR="$SCRIPT_DIR"
VENV_DIR="$VENV_DIR"
VENV_SITE_PACKAGES="$VENV_SITE_PACKAGES"

# Activate virtual environment
source "\$VENV_DIR/bin/activate"

# Compilers
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export TMPDIR=/projects/a5k/public/tmp

# CRITICAL: LD_PRELOAD for venv NCCL (fixes ncclCommShrink symbol mismatch)
export NCCL_LIBRARY="\$VENV_SITE_PACKAGES/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="\$NCCL_LIBRARY"

# NVIDIA library paths
export LD_LIBRARY_PATH="\$VENV_SITE_PACKAGES/nvidia/nccl/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$VENV_SITE_PACKAGES/nvidia/cudnn/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$VENV_SITE_PACKAGES/nvidia/cublas/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$VENV_SITE_PACKAGES/nvidia/nvjitlink/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$VENV_SITE_PACKAGES/nvidia/cusparse/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$VENV_SITE_PACKAGES/nvidia/cufft/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$VENV_SITE_PACKAGES/nvidia/curand/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$VENV_SITE_PACKAGES/nvidia/cusolver/lib:\$LD_LIBRARY_PATH"

# Include paths for any runtime compilation
export CPLUS_INCLUDE_PATH="\$VENV_SITE_PACKAGES/nvidia/cudnn/include:\${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="\$VENV_SITE_PACKAGES/nvidia/cudnn/include:\${C_INCLUDE_PATH:-}"
export CUDNN_PATH="\$VENV_SITE_PACKAGES/nvidia/cudnn"
ACTIVATE_EOF
chmod +x "$SCRIPT_DIR/activate_env.sh"
echo "  activate_env.sh created"

# ============================================
# Phase 9: Validation
# ============================================
echo ""
echo "=== Phase 9: Running validation ==="
LD_PRELOAD="$NCCL_LIBRARY" "$VENV_PYTHON" -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

pkgs = {
    'megatron.core': lambda: __import__('megatron.core'),
    'megatron.bridge': lambda: __import__('megatron.bridge'),
    'transformers': lambda: __import__('transformers'),
    'datasets': lambda: __import__('datasets'),
    'wandb': lambda: __import__('wandb'),
    'omegaconf': lambda: __import__('omegaconf'),
    'transformer_engine': lambda: __import__('transformer_engine'),
    'transformer_engine.pytorch': lambda: __import__('transformer_engine.pytorch'),
    'mamba_ssm': lambda: __import__('mamba_ssm'),
    'causal_conv1d': lambda: __import__('causal_conv1d'),
}

for name, importer in pkgs.items():
    try:
        mod = importer()
        ver = getattr(mod, '__version__', 'OK')
        print(f'{name}: {ver}')
    except ImportError as e:
        print(f'{name}: FAILED ({e})')
"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source $SCRIPT_DIR/activate_env.sh"
echo ""
echo "To run validation tests:"
echo "  source activate_env.sh && python validate_install.py"
echo ""
echo "To run a tiny training test:"
echo "  source activate_env.sh"
echo "  python -m torch.distributed.run --nproc_per_node=1 \\"
echo "      scripts/training/run_recipe.py \\"
echo "      --recipe vanilla_gpt_pretrain_config \\"
echo "      --dataset llm-pretrain-mock \\"
echo "      train.train_iters=5 train.global_batch_size=8 train.micro_batch_size=4"
echo ""
