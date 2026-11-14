#!/bin/bash
# ==============================================================================
# Environment Pipeline — in-container environment activation (INFR-68)
#
# Source this INSIDE the container (pipeline_env_exec.sh injects it into every
# payload). It wires up three things the image cannot know about:
#
#   1. Import resolution: this repo's megatron.bridge fork (src/) and the pinned
#      Megatron-Core submodule (3rdparty/Megatron-LM) must win over any megatron
#      packages installed in the image.
#   2. The cluster's own environment: its profile's activation hook
#      (pipeline/sh/clusters/), for example the library paths of its fabric.
#   3. Universal GPU/cache settings.
#
# The image provides torch/CUDA/cuDNN/NCCL/TE/Mamba-kernels/compilers as a
# version-matched set, so nothing here installs, builds, or activates a venv.
# ==============================================================================

# Refuse to run outside a container — sourcing this on the host would poison the
# host env with in-container paths that don't exist there.
if [ ! -d /.singularity.d ]; then
    echo "ERROR [env-activate]: not inside an Apptainer container." >&2
    echo "  This file is sourced by pipeline_env_exec.sh, which enters the container:" >&2
    echo "    ./pipeline_env_exec.sh \"cd \$PWD; source pipeline_env_activate.sh || exit 1; <cmd>\"" >&2
    return 1
fi

# Repo root from this script's own location (worktree-safe).
_ENV_ACTIVATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$_ENV_ACTIVATE_DIR}"

# The cluster's profile (CLUSTER_* values and its activation hook). The shim exports GEODESIC_CLUSTER, so this
# resolves the cluster the launcher resolved.
source "$_ENV_ACTIVATE_DIR/pipeline/sh/cluster.sh" || return 1
cluster_load || return 1

# ==============================================================================
# 1. Import resolution
#
# src/megatron and 3rdparty/Megatron-LM/megatron are PEP 420 namespace portions,
# so sys.path order decides which portion serves megatron.bridge / megatron.core.
# PYTHONPATH entries precede the image's site-packages -> the repo checkout wins.
# pipeline_env_validate.py asserts this resolution on every run — a regular
# (non-namespace) megatron package in a future image would silently defeat it.
# ==============================================================================
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"

# Python overlay: a `pip install --target` dir carrying packages the image ships
# too old for this repo (currently peft — image 0.13.2, the bridge recipes'
# modelopt import needs >=0.17). Appended AFTER the repo prepends so the repo
# checkout still wins, but the whole PYTHONPATH precedes the image's
# site-packages, so the overlay's peft shadows the image's.
# CONTAINER_PYTHON_OVERLAY is exported by pipeline_env_config.env and inherited
# via apptainer env passthrough. Configured-but-missing is surfaced loudly (never
# a silent skip) so a stale/unpopulated overlay can't quietly fall back to the
# too-old image package.
if [ -n "${CONTAINER_PYTHON_OVERLAY:-}" ]; then
    if [ -d "$CONTAINER_PYTHON_OVERLAY" ]; then
        export PYTHONPATH="${PYTHONPATH}:${CONTAINER_PYTHON_OVERLAY}"
    else
        echo "WARNING [env-activate]: CONTAINER_PYTHON_OVERLAY=$CONTAINER_PYTHON_OVERLAY is configured but does not exist." >&2
        echo "  Image packages too old for this repo (e.g. peft) will NOT be overridden." >&2
        echo "  Populate it (one-time): bash pipeline_env_setup.sh" >&2
        echo "  See docs/environment.md → 'Python overlay'." >&2
    fi
fi

# The host $HOME is bind-mounted (W&B needs ~/.netrc); keep its ~/.local from
# leaking incompatible user-site packages into the image's python.
export PYTHONNOUSERSITE=1

# The shim scrubs LD_PRELOAD before exec; unset again defensively in case this
# file is sourced from an interactive `apptainer shell`.
unset LD_PRELOAD

# ==============================================================================
# 1b. CUDA forward-compatibility (image CUDA newer than the host driver)
#
# The host driver is R580 (580.173.02, CUDA 13.0) on the compute-node image of
# 2026-10-07; NGC images bundle CUDA 13.x (26.04: 13.1). Under Docker, NGC's
# entrypoint detects a driver older than the image's CUDA and symlinks
# /usr/local/cuda/compat/lib -> lib.real so the ld config picks the forward-compat
# libcuda. Apptainer never runs that entrypoint and the SIF is read-only, so
# WITHOUT this block the loader silently uses the host's older libcuda and the
# image's torch dies with "driver too old" (verified empirically on this cluster,
# then on R565 / CUDA 12.7). Fronting the compat dir is safe only while the host
# driver is older than the image's CUDA, which holds for the qualified 26.04 (CUDA
# 13.1) and for 26.06 (13.2) on R580 (CUDA 13.0). It does NOT hold for a CUDA 13.0
# image (25.09, 25.11) on R580: there the compat libcuda is older than the host
# driver, the case NGC's entrypoint skips, and fronting it can fail with error 803.
# Run such a tag with GEODESIC_CONTAINER_CUDA_COMPAT=0 and validate it. Measured:
# on R565.57.01 CUDA 13.0 compat works and CUDA 13.2 compat REJECTS the driver
# (error 803); on R580.173.02 the 26.04 image's CUDA 13.1 compat works (validate
# 21/21, 2026-10-09) and so does the 26.06 image's CUDA 13.2 compat (its CUDA and
# GPU-op checks pass). The verdict is per image and per driver, and gated by
# `isambard_sbatch pipeline_env_submit.sbatch validate`.
#
# GEODESIC_CONTAINER_CUDA_COMPAT=0 disables; =auto (default) probes the two known
# NGC layouts; =/path forces a specific compat lib dir.
# ==============================================================================
_CUDA_COMPAT_MODE="${GEODESIC_CONTAINER_CUDA_COMPAT:-auto}"
if [ "$_CUDA_COMPAT_MODE" != "0" ]; then
    if [ "$_CUDA_COMPAT_MODE" = "auto" ]; then
        _CUDA_COMPAT_DIR=""
        for _d in /usr/local/cuda/compat/lib.real /usr/local/cuda/compat/lib /usr/local/cuda/compat; do
            if [ -f "$_d/libcuda.so.1" ]; then _CUDA_COMPAT_DIR="$_d"; break; fi
        done
    else
        _CUDA_COMPAT_DIR="$_CUDA_COMPAT_MODE"
    fi
    if [ -n "$_CUDA_COMPAT_DIR" ]; then
        export LD_LIBRARY_PATH="$_CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

# ==============================================================================
# 2. The cluster's own environment
#
# The profile's activation hook exports what only that cluster needs inside the
# container. A hook that cannot build a working environment returns 1, and the
# payload stops.
# ==============================================================================
cluster_activate || return 1

# ==============================================================================
# 3. Universal GPU settings
# ==============================================================================
# Overridable via ISAMBARD_CUDA_MAX_CONNECTIONS: the =1 requirement is for TP/SP
# comm-compute overlap; TP=1 topologies (e.g. the 120B quickstart) may probe >1
# to unserialize the hardware launch queues (see the quickstart header ladder).
export CUDA_DEVICE_MAX_CONNECTIONS="${ISAMBARD_CUDA_MAX_CONNECTIONS:-1}"
# Overridable via ISAMBARD_OMP_THREADS. torchrun/ft_launcher set OMP_NUM_THREADS=1
# whenever the variable is ABSENT, which single-threads the host-side AdamW of a
# CPU-offloaded optimizer onto one Neoverse-V2 core (~36 GB/s of a socket capable of
# ~500 GB/s). Exporting here wins because this file is sourced in the same shell that then
# execs the launcher. Measured on the 120B benchmark at offload fraction 1.0: 21.36 s/iter
# / 73.70 GB with 8 threads, versus 22.79 / 76.78 single-threaded at fraction 0.5 — faster
# AND lighter. With offload OFF it is exactly neutral (20.663 vs 20.654 s/iter, identical
# peak memory), which is why 8 is safe as a universal default: there is no host optimizer
# for the threads to act on. PASSIVE is load-bearing rather than decoration — GNU OpenMP
# idle threads spin-wait and these workloads are host-launch-bound, so ACTIVE spin can cost
# more launch throughput than the threaded Adam saves. ISAMBARD_OMP_THREADS=1 restores
# torchrun's behaviour. Applies to every payload, not only training; 8 threads per rank is
# well inside a 72-core Grace socket.
export OMP_NUM_THREADS="${ISAMBARD_OMP_THREADS:-8}"
if [ "$OMP_NUM_THREADS" != "1" ]; then
    export OMP_WAIT_POLICY="${ISAMBARD_OMP_WAIT_POLICY:-PASSIVE}"
fi
# TE fine-grained CPU activation offloading (TE >= 2.10 path). Respect a pre-set value:
# NVTE_CPU_OFFLOAD_V1=0 is REQUIRED by the 32k green-team recipe (configs/PA/green-team/32k/)
# — TE's V1 offload path allocates ~300-400 GB/node of host memory at seq 32768 on the first
# forward, independent of Megatron's fine_grained_activation_offloading config (root-caused
# 2026-07-15, 32k host-OOM investigation). Default remains 1.
export NVTE_CPU_OFFLOAD_V1="${NVTE_CPU_OFFLOAD_V1:-1}"
# Reduces CUDA memory fragmentation. Overridable (ISAMBARD_CUDA_ALLOC_CONF) for A/Bs:
# a measured expandable_segments:False arm on the 512-GPU Nano pretrain posture showed
# no fast-regime difference (both ~6.1-6.5 s/iter) and did NOT affect the post-save
# collective-slowdown bug, so the True default stands. Note the CXI provider's
# `sysnc_memops returned -22` warnings under FI_LOG_LEVEL=warn appear in BOTH modes —
# they concern NCCL's own cuMem-allocated buffers, not this allocator.
export PYTORCH_CUDA_ALLOC_CONF="${ISAMBARD_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_CUDA_ARCH_LIST="$CLUSTER_CUDA_ARCH_LIST"         # the GPU architecture JIT builds compile for

# The image's CUDA toolkit (nvcc, headers) — for JIT builds (Triton, TE, Megatron
# dataset helpers). Deliberately NOT the host HPC-SDK path (which the shim
# scrubs): inside the container the image toolchain is the only valid one.
export CUDA_HOME=/usr/local/cuda

# ==============================================================================
# Shared cache paths (inherited from the host anyway; set explicitly so an
# interactive `apptainer shell` works alone).
# ==============================================================================
export NEMO_HOME="$CLUSTER_NEMO_HOME"
export HF_HOME="$CLUSTER_HF_HOME"
# Processed-dataset cache is scoped to this image's datasets library version: a
# dir shared with a different major version fails with 'DatasetInfo.from_directory
# ... must be called with a dataclass' (found by the C7 parity smoke). Hub
# downloads (models/tokenizers under HF_HOME) stay shared.
#
# The profile's default is per user. Point GEODESIC_HF_DATASETS_CACHE at an
# explicit path to share one deliberately.
export HF_DATASETS_CACHE="${GEODESIC_HF_DATASETS_CACHE:-$CLUSTER_HF_DATASETS_CACHE}"
export WANDB_DIR="$CLUSTER_WANDB_DIR"
export TMPDIR="${TMPDIR:-$CLUSTER_ACTIVATE_TMPDIR}"
# deep_ep's HybridEP dispatcher compiles its kernels at first use into a new directory per process
# (under 1 MB, never reused or removed): HYBRID_EP_CACHE_DIR/.deepep/hybrid_ep/jit/proc-<pid>, with the
# bind-mounted host $HOME in place of the variable when it is unset. Under TMPDIR it is node-local and
# job-scoped in a training launch; HybridEP creates every missing level, so a TMPDIR made on the batch
# node only still works.
export HYBRID_EP_CACHE_DIR="${TMPDIR}/hybrid_ep_jit"
