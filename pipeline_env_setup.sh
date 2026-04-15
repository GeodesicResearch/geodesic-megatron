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
VENV_DIR="$SCRIPT_DIR/.venv"
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
# Phase 3: Install PyTorch (aarch64 + CUDA)
# ============================================
echo ""
echo "=== Phase 3: Installing PyTorch ==="
# IMPORTANT: uv pip install silently fails with PyTorch wheel indexes on aarch64.
# We use pip directly for PyTorch installation instead.
# The pyproject.toml has 'torch; sys_platform == "never"' override, so uv sync
# will not try to install torch - it expects torch to be pre-installed (container pattern).

echo "Installing torch from cu126 index (aarch64 wheels available)..."
"$VENV_DIR/bin/pip" install \
    --index-url https://download.pytorch.org/whl/cu126 \
    "torch>=2.6.0" 2>&1 | tail -5

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch installation failed"
    exit 1
fi

# Set up NCCL LD_PRELOAD before any torch import
export NCCL_LIBRARY="$VENV_SITE_PACKAGES/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# Verify PyTorch CUDA works
echo ""
echo "Verifying PyTorch installation..."
LD_PRELOAD="$NCCL_LIBRARY" "$VENV_PYTHON" -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Arch: {torch.cuda.get_device_capability(0)}')
" || {
    echo "ERROR: PyTorch CUDA verification failed"
    exit 1
}

# ============================================
# Phase 4: Initialize Megatron-Core submodule
# ============================================
echo ""
echo "=== Phase 4: Initializing Megatron-Core submodule ==="
git submodule update --init 3rdparty/Megatron-LM
echo "Megatron-Core submodule initialized at 3rdparty/Megatron-LM"

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

# Create cuDNN header symlinks in PyTorch include directory
# Required for building transformer-engine from source
TORCH_INCLUDE="$VENV_SITE_PACKAGES/torch/include"
CUDNN_INCLUDE="$VENV_SITE_PACKAGES/nvidia/cudnn/include"
if [ -d "$CUDNN_INCLUDE" ] && [ -d "$TORCH_INCLUDE" ]; then
    echo "Creating cuDNN header symlinks in PyTorch include directory..."
    for f in "$CUDNN_INCLUDE"/*.h; do
        ln -sf "$f" "$TORCH_INCLUDE/$(basename "$f")" 2>/dev/null || true
    done
    echo "cuDNN symlinks created"
else
    echo "Warning: Could not create cuDNN symlinks (directories not found)"
fi

# ============================================
# Phase 5b: Install build dependencies
# ============================================
echo ""
echo "=== Phase 5b: Installing build dependencies ==="
# TE requires pybind11 at build time (not declared in its build-system.requires)
# Also install numpy, ninja, Cython for other source builds
uv pip install --python "$VENV_PYTHON" pybind11 numpy "Cython>=3.0.0" ninja setuptools 2>&1 | tail -3
echo "Build deps installed"

# ============================================
# Phase 6: Install Megatron Bridge core deps via uv sync
# ============================================
echo ""
echo "=== Phase 6: Installing Megatron Bridge dependencies ==="
echo "Running uv sync (without --locked, resolving for aarch64)..."
echo "This may take several minutes..."

# uv sync installs:
# - megatron-bridge (editable from src/)
# - megatron-core (editable from 3rdparty/Megatron-LM/)
# - All pure-Python dependencies
# - Attempts no-build-isolation packages (TE, mamba-ssm, etc.)
# The torch override (sys_platform == 'never') means uv skips torch (already installed)
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="9.0" \
    CUDA_HOME="$CUDA_HOME" \
    LD_PRELOAD="$NCCL_LIBRARY" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH" \
    C_INCLUDE_PATH="$C_INCLUDE_PATH" \
    LIBRARY_PATH="$LIBRARY_PATH" \
    CUDNN_PATH="$CUDNN_PATH" \
    uv sync 2>&1 | tee /tmp/uv_sync_megatron.log | tail -30

UV_SYNC_EXIT=${PIPESTATUS[0]}
if [ "$UV_SYNC_EXIT" -ne 0 ]; then
    echo ""
    echo "WARNING: uv sync exited with code $UV_SYNC_EXIT"
    echo "Some packages may have failed to build. Phase 7 will install them individually."
    echo "Full log: /tmp/uv_sync_megatron.log"
fi

# ============================================
# Phase 7: Build source packages individually
# ============================================
echo ""
echo "=== Phase 7: Building CUDA extension packages from source ==="
echo "These must be built with --no-build-isolation on aarch64."
echo ""

BUILD_ENV="CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST=9.0 \
    CUDA_HOME=$CUDA_HOME \
    LD_PRELOAD=$NCCL_LIBRARY \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH \
    C_INCLUDE_PATH=$C_INCLUDE_PATH \
    LIBRARY_PATH=$LIBRARY_PATH \
    CUDNN_PATH=$CUDNN_PATH \
    NVTE_NCCL_INCLUDE=$NVTE_NCCL_INCLUDE \
    NVTE_NCCL_LIB=$NVTE_NCCL_LIB"

# 7a. Transformer Engine (pinned commit from pyproject.toml override)
echo "--- 7a. Building transformer-engine from source (pinned commit) ---"
echo "This takes 10-20 minutes..."
TE_COMMIT="71bbefbf153418f943640df0f7373625dc93fa46"
# Use pip (not uv pip) for TE build - more reliable with git URLs on aarch64
env $BUILD_ENV \
    "$VENV_DIR/bin/pip" install \
    --no-build-isolation --no-cache-dir \
    "transformer-engine[pytorch] @ git+https://github.com/NVIDIA/TransformerEngine.git@${TE_COMMIT}" 2>&1 | tail -10

LD_PRELOAD="$NCCL_LIBRARY" "$VENV_PYTHON" -c "
import transformer_engine
print(f'  transformer_engine: {transformer_engine.__version__}')
import transformer_engine.pytorch as te
print('  transformer_engine.pytorch: OK')
" || {
    echo "WARNING: transformer-engine pinned commit failed. Trying latest release..."
    env $BUILD_ENV \
        "$VENV_DIR/bin/pip" install \
        --no-build-isolation --no-cache-dir --no-binary transformer-engine-torch \
        "transformer-engine[pytorch]" 2>&1 | tail -10
    LD_PRELOAD="$NCCL_LIBRARY" "$VENV_PYTHON" -c "import transformer_engine.pytorch; print('  TE fallback: OK')" || {
        echo "ERROR: transformer-engine installation failed"
        exit 1
    }
}
echo "transformer-engine: INSTALLED"

# 7b. causal-conv1d (dependency of mamba-ssm)
echo ""
echo "--- 7b. Building causal-conv1d from source ---"
env $BUILD_ENV \
    "$VENV_DIR/bin/pip" install --no-build-isolation --no-cache-dir --no-binary causal-conv1d \
    causal-conv1d 2>&1 | tail -5
LD_PRELOAD="$NCCL_LIBRARY" "$VENV_PYTHON" -c "import causal_conv1d; print('  causal-conv1d: OK')" || {
    echo "ERROR: causal-conv1d installation failed"
    exit 1
}
echo "causal-conv1d: INSTALLED"

# 7c. mamba-ssm (required for Nemotron hybrid Mamba/Transformer)
echo ""
echo "--- 7c. Building mamba-ssm from source ---"
echo "This takes 10-15 minutes..."
env $BUILD_ENV MAMBA_FORCE_BUILD=TRUE \
    "$VENV_DIR/bin/pip" install --no-build-isolation --no-cache-dir --no-binary mamba-ssm \
    mamba-ssm 2>&1 | tail -10
LD_PRELOAD="$NCCL_LIBRARY" "$VENV_PYTHON" -c "import mamba_ssm; print('  mamba-ssm: OK')" || {
    echo "ERROR: mamba-ssm installation failed"
    exit 1
}
echo "mamba-ssm: INSTALLED"

# 7d. nv-grouped-gemm (needed for MoE grouped GEMM)
echo ""
echo "--- 7d. Installing nv-grouped-gemm ---"
env $BUILD_ENV \
    "$VENV_DIR/bin/pip" install --no-build-isolation --no-cache-dir \
    nv-grouped-gemm 2>&1 | tail -5 || {
    echo "WARNING: nv-grouped-gemm failed (optional, MoE may fall back to non-grouped)"
}
echo "nv-grouped-gemm: ATTEMPTED"

# 7e. flash-linear-attention (listed in pyproject.toml deps)
echo ""
echo "--- 7e. Installing flash-linear-attention ---"
env $BUILD_ENV \
    "$VENV_DIR/bin/pip" install --no-build-isolation --no-cache-dir \
    flash-linear-attention 2>&1 | tail -5 || {
    echo "WARNING: flash-linear-attention failed (optional for initial validation)"
}
echo "flash-linear-attention: ATTEMPTED"

# 7f. flash_mla (optional)
echo ""
echo "--- 7f. Installing flash_mla (optional) ---"
env $BUILD_ENV \
    "$VENV_DIR/bin/pip" install --no-build-isolation --no-cache-dir \
    flash_mla 2>&1 | tail -3 || {
    echo "WARNING: flash_mla failed (optional)"
}
echo "flash_mla: ATTEMPTED"

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
