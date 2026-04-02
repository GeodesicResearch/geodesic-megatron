#!/bin/bash
# Source this file to activate the Megatron Bridge environment
# Usage: source activate_env.sh

SCRIPT_DIR="/home/a5k/kyleobrien.a5k/geodesic-megatron"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_SITE_PACKAGES="$VENV_DIR/lib/python3.12/site-packages"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Compilers
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export TMPDIR=/projects/a5k/public/tmp

# CRITICAL: LD_PRELOAD for venv NCCL (fixes ncclCommShrink symbol mismatch)
export NCCL_LIBRARY="$VENV_SITE_PACKAGES/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# NVIDIA library paths
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cufft/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/curand/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cusolver/lib:$LD_LIBRARY_PATH"

# Include paths for any runtime compilation
export CPLUS_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/include:$VENV_SITE_PACKAGES/nvidia/cudnn/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/include:$VENV_SITE_PACKAGES/nvidia/cudnn/include:${C_INCLUDE_PATH:-}"
export CUDNN_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn"
