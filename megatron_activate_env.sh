#!/bin/bash
# ==============================================================================
# Megatron Bridge Environment Activation
#
# Source this file to set up the complete environment for any GPU operation:
# training, inference, checkpoint conversion, evals, or interactive development.
#
# Usage:
#   source megatron_activate_env.sh
#
# What this does:
#   1. Activates the Python 3.12 venv (.venv/)
#   2. Sets compilers to gcc-12 (required for CUDA/C++17 on Isambard's aarch64)
#   3. Preloads the venv's NCCL to avoid symbol conflicts with the system NCCL
#   4. Adds all NVIDIA pip-installed libraries to LD_LIBRARY_PATH
#   5. Sets include paths for runtime CUDA kernel compilation (JIT)
#   6. Configures universal GPU settings (TE, PyTorch allocator, etc.)
#   7. Points HF/W&B/NeMo caches to shared project storage
#
# For distributed multi-node training, megatron_launch_training.sh sources this file
# and then adds Slingshot/CXI NCCL vars, fault tolerance, and job-specific paths.
# ==============================================================================

SCRIPT_DIR="/home/a5k/kyleobrien.a5k/geodesic-megatron"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_SITE_PACKAGES="$VENV_DIR/lib/python3.12/site-packages"

# ==============================================================================
# Python virtual environment
# ==============================================================================

# Activate the venv. This prepends .venv/bin to PATH so `python`, `pip`, `uv`,
# `ft_launcher`, etc. resolve to the venv versions rather than system binaries.
source "$VENV_DIR/bin/activate"

# ==============================================================================
# Compilers
#
# Isambard's default system compiler is gcc 7.5, which lacks C++17 support
# (no <filesystem>, no structured bindings, etc.). CUDA extensions compiled by
# nvcc (Transformer Engine, flash-attn, mamba-ssm, causal-conv1d) require a
# C++17-capable host compiler. gcc-12 is installed at /usr/bin/gcc-12.
#
# Without these, you get: "fatal error: filesystem: No such file or directory"
# during any CUDA JIT compilation or pip install of extensions from source.
# ==============================================================================

# C and C++ compilers for general builds (Cython, pybind11, cmake, etc.)
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

# Host compiler for nvcc. nvcc compiles device code itself but delegates host
# code to this compiler. Must match CXX to avoid ABI mismatches between host
# and device translation units.
export CUDAHOSTCXX=/usr/bin/g++-12

# Target GPU architecture for PyTorch JIT-compiled CUDA kernels. "9.0" = Hopper
# (sm_90), which covers both H100 and GH200 GPUs. Without this, PyTorch's
# torch.utils.cpp_extension tries to auto-detect and may get confused by GH200's
# sm_90a suffix (see sitecustomize.py monkeypatch for that issue).
export TORCH_CUDA_ARCH_LIST="9.0"

# CUDA toolkit location. Points to the HPC SDK's CUDA 12.6 install, which
# provides nvcc, ptxas, cuobjdump, and the CUDA runtime/driver headers. Used by
# cmake, setuptools, and any build that needs to find CUDA tools or libraries.
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

# Default temp directory for non-training operations (interactive, builds, etc.).
# On shared NFS project space. Training jobs override this to node-local /tmp in
# megatron_launch_training.sh to avoid NFS contention across 128+ ranks.
export TMPDIR=/projects/a5k/public/tmp

# ==============================================================================
# NCCL library preload (CRITICAL)
#
# The system NCCL (loaded by Cray's module system) is older than what PyTorch
# and Megatron-Core expect. Importing torch.distributed with the system NCCL
# fails with: "undefined symbol: ncclCommShrink" (added in NCCL 2.21+).
#
# LD_PRELOAD forces the dynamic linker to load the venv's NCCL (2.28.9) before
# any other library. This shadows the system NCCL and provides all symbols that
# PyTorch needs. This must happen before any Python import of torch.
# ==============================================================================
export NCCL_LIBRARY="$VENV_SITE_PACKAGES/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# ==============================================================================
# NVIDIA library paths (LD_LIBRARY_PATH)
#
# On Isambard, NVIDIA libraries are NOT installed system-wide — they come from
# pip packages in the venv (nvidia-nccl-cu12, nvidia-cudnn-cu12, etc.). The
# dynamic linker doesn't know about site-packages, so we must add each library's
# path to LD_LIBRARY_PATH explicitly.
#
# Without these, you get errors like:
#   "libcudnn.so.9: cannot open shared object file"
#   "libcublas.so.12: cannot open shared object file"
# whenever PyTorch, Transformer Engine, or flash-attn try to load their CUDA
# backends at runtime.
#
# Order matters: NCCL is first to ensure the venv's version takes precedence
# over any system NCCL that might be on the default library path.
# ==============================================================================
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH"         # NCCL 2.28.9 — collective communication
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"        # cuDNN 9.x — conv, attention, normalization kernels
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH"       # cuBLAS — GEMM, matrix ops (the core of transformer compute)
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH" # CUDA runtime (cudart) — memory alloc, stream management, kernel launch
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"    # nvJitLink — runtime linking of PTX/cubin (used by Triton, TE)
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cusparse/lib:$LD_LIBRARY_PATH"     # cuSPARSE — sparse matrix ops (used by some attention variants)
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cufft/lib:$LD_LIBRARY_PATH"        # cuFFT — FFT operations
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/curand/lib:$LD_LIBRARY_PATH"       # cuRAND — random number generation (dropout, init)
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cusolver/lib:$LD_LIBRARY_PATH"     # cuSOLVER — dense/sparse linear solvers

# ==============================================================================
# Include paths for runtime CUDA compilation
#
# When CUDA extensions are JIT-compiled at runtime (torch.utils.cpp_extension,
# Triton, Transformer Engine custom kernels), the compiler needs headers for
# NCCL and cuDNN. These pip packages install headers in site-packages, not in
# the standard /usr/include or CUDA_HOME/include paths.
#
# CPLUS_INCLUDE_PATH and C_INCLUDE_PATH are searched by gcc/g++ automatically
# (like a persistent -I flag). This avoids needing to pass -I to every build.
# ==============================================================================
export CPLUS_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/include:$VENV_SITE_PACKAGES/nvidia/cudnn/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/include:$VENV_SITE_PACKAGES/nvidia/cudnn/include:${C_INCLUDE_PATH:-}"

# cuDNN install root for cmake-based builds (Transformer Engine, flash-attn).
# cmake's find_package(CUDNN) looks at CUDNN_PATH/lib and CUDNN_PATH/include.
export CUDNN_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn"

# ==============================================================================
# GPU / compute settings (universal -- needed for any GPU operation)
# ==============================================================================

# Skip CUDA Multicast initialization in Transformer Engine's Userbuffers (UB) library.
# UB provides comm+GEMM overlap for tensor parallelism. Multicast is a Hopper feature
# that lets one GPU write to multiple peer GPUs' memory simultaneously, used by UB for
# efficient all-gather. However, it requires a specific driver version and CUDA toolkit
# that Isambard's driver doesn't support. Without this skip, UB init hangs trying to
# allocate multicast objects. The comm+GEMM overlap still works via non-multicast paths.
export UB_SKIPMC=1

# Use the V1 code path for Transformer Engine's CPU activation offloading. TE offloads
# activation tensors from GPU to CPU during the forward pass and reloads them during
# backward, trading PCIe bandwidth for GPU memory. V1 is the fine-grained implementation
# (per-layer offload with async transfers) required for TE >= 2.10.0. Without this,
# TE uses the V2 path which has different memory management that's incompatible with
# some Megatron-Core activation recompute configurations.
export NVTE_CPU_OFFLOAD_V1=1

# Limit CUDA to 1 concurrent kernel execution stream per device. This is required for
# Megatron-Core's tensor-parallel (TP) and sequence-parallel (SP) communication-computation
# overlap. With 1 connection, CUDA serializes kernels on the default stream, which lets
# Transformer Engine's UB library precisely interleave NCCL collectives (all-gather,
# reduce-scatter) with GEMM kernels in a pipelined fashion. With >1 connections, CUDA
# can reorder kernels, breaking the overlap scheduling and causing collectives to run
# after the GEMM instead of during it. Harmless for single-GPU. Note: conflicts with FSDP.
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Use PyTorch's expandable segments allocator. Instead of allocating many fixed-size CUDA
# memory blocks (which fragment the GPU address space), expandable segments uses CUDA
# Virtual Memory Management (VMM) to create one large virtual segment per stream that
# grows on demand by appending physical pages. This dramatically reduces fragmentation
# where reserved-but-unused memory gaps prevent new allocations, which is the primary
# cause of spurious OOM errors in large model training.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==============================================================================
# Shared paths (universal -- needed for any HF / W&B operation)
# ==============================================================================

# NeMo dataset cache. NeMo downloads and caches processed datasets at ~/.cache/nemo by
# default, which fills the home directory quota quickly. Redirecting to shared project
# space lets multiple jobs reuse the same cached datasets.
export NEMO_HOME=/projects/a5k/public/data/nemo_cache

# HuggingFace Hub cache (model weights, tokenizers, datasets). On shared project storage
# so downloads are shared across jobs. Default ~/.cache/huggingface would fill home quota.
export HF_HOME=/projects/a5k/public/hf

# W&B artifacts and run metadata directory. On shared project storage so runs are visible
# across login/compute nodes. W&B only writes from rank 0 so NFS contention is minimal.
export WANDB_DIR=/projects/a5k/public/logs/wandb
