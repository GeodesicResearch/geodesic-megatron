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
#   2. Slingshot networking: point NCCL at the aws-ofi-nccl CXI plugin built by
#      pipeline_env_setup.sh (official Isambard "Option B" recipe) and order the
#      host libfabric ahead of everything.
#   3. Universal GPU/cache settings.
#
# The image provides torch/CUDA/cuDNN/NCCL/TE/Mamba-kernels/compilers as a
# version-matched set, so nothing here installs, builds, or activates a venv.
# ==============================================================================

# Refuse to run outside a container — sourcing this on the host would poison the
# host env with /opt/slingshot paths that don't exist there.
if [ ! -d /.singularity.d ]; then
    echo "ERROR [env-activate]: not inside an Apptainer container." >&2
    echo "  This file is sourced by pipeline_env_exec.sh, which enters the container:" >&2
    echo "    ./pipeline_env_exec.sh \"cd \$PWD; source pipeline_env_activate.sh || exit 1; <cmd>\"" >&2
    return 1
fi

# Repo root from this script's own location (worktree-safe).
_ENV_ACTIVATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$_ENV_ACTIVATE_DIR}"

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
# The host driver is R565 (CUDA 12.7); NGC images bundle CUDA 12.9/13.x. Under
# Docker, NGC's entrypoint detects this and symlinks /usr/local/cuda/compat/lib
# -> lib.real so the ld config picks the forward-compat libcuda. Apptainer never
# runs that entrypoint and the SIF is read-only, so WITHOUT this block the loader
# silently uses the host's 12.7 libcuda and CUDA-13 torch dies with "driver too
# old" (verified empirically on this cluster). Fronting the compat dir is safe
# here because the Isambard driver is always older than any image CUDA we qualify
# (the one case NGC's entrypoint would skip compat — driver newer than image —
# cannot occur). Measured on R565.57.01: CUDA 13.0 compat works; CUDA 13.2 compat
# REJECTS the driver (error 803) — that verdict is per image and gated by
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
# 2. Slingshot / CXI networking (official Option B layout)
#
# pipeline_env_setup.sh builds, inside this image:
#   /opt/slingshot/nccl          — NCCL (built vs image CUDA)
#   /opt/slingshot/aws-ofi-nccl  — the CXI-capable network plugin
# and the shim binds host libfabric at /host/opt/cray/libfabric/<ver> and
# /usr/lib64 (libcxi, libnl) at /host/usr/lib64.
#
# Ordering (mirrors the BriCS pytorch_multinode.def %environment):
#   libfabric first  — the plugin must resolve the CXI-capable Cray libfabric;
#   built NCCL next  — a newer NCCL shadowing torch's bundled one;
#   image aarch64 libdir — see the linker note below;
#   /opt/slingshot/hostlibs LAST — only fills sonames the image lacks.
#
# NCCL_NET_PLUGIN names the CXI plugin explicitly — NGC images ship an
# EFA-targeted aws-ofi-nccl (AWS fabric) that must never be selected; without the
# CXI plugin NCCL silently falls back to TCP at ~2.3 GB/s vs ~163 GB/s.
#
# LINKER TRAP — /host/usr/lib64 must NEVER be on LD_LIBRARY_PATH. torch
# inductor's C++ codegen converts LD_LIBRARY_PATH entries into -L link dirs, and
# the host dir's libc.so is a SUSE GNU-ld script whose GROUP() names absolute
# /lib64/libc.so.6 + /usr/lib64/libc_nonshared.a — paths that do not exist in the
# image, so any fresh torch.compile link dies with "ld: cannot find
# /lib64/libc.so.6". Ordering image libdirs first does NOT save you: the image
# keeps no plain libc.so dev script in /usr/lib/aarch64-linux-gnu, so -lc falls
# through to the host dir anyway (warm inductor caches masked this for a while — a
# cache hit skips the link entirely, which is why single-shot probes "passed").
# Instead, the ONLY things the plugin needs from the host /usr/lib64 (libcxi,
# libnl — sonames the image lacks) are exposed via a symlink-only directory the
# setup step creates at /opt/slingshot/hostlibs: runtime dlopen finds them, and
# the linker finds NO libc there to be poisoned by.
# ==============================================================================
# Resolve the bound host libfabric. An unmatched glob would stay LITERAL and a
# multi-version match would inject a space into LD_LIBRARY_PATH — both give the
# CXI plugin no libfabric to resolve, which degrades NCCL to TCP sockets (~2.3 vs
# ~163 GB/s) with no error anywhere. Fail loudly instead: this is the one silent
# ~70x performance cliff in the whole stack.
_HOST_LIBFABRIC_CANDIDATES=(/host/opt/cray/libfabric/*/lib64)
if [ ! -d "${_HOST_LIBFABRIC_CANDIDATES[0]}" ]; then
    echo "ERROR [env-activate]: no host libfabric under /host/opt/cray/libfabric/*/lib64." >&2
    echo "  The CXI plugin cannot load and NCCL would silently fall back to TCP (~70x slower)." >&2
    echo "  Check the libfabric bind in pipeline_env_config.env (CONTAINER_HOST_LIBFABRIC=${CONTAINER_HOST_LIBFABRIC:-unset})." >&2
    return 1
fi
if [ "${#_HOST_LIBFABRIC_CANDIDATES[@]}" -gt 1 ]; then
    echo "WARNING [env-activate]: multiple host libfabric versions bound (${_HOST_LIBFABRIC_CANDIDATES[*]});" >&2
    echo "  using ${_HOST_LIBFABRIC_CANDIDATES[0]}. Pin CONTAINER_HOST_LIBFABRIC to one version." >&2
fi
_HOST_LIBFABRIC_LIB="${_HOST_LIBFABRIC_CANDIDATES[0]}"
export LD_LIBRARY_PATH="${_HOST_LIBFABRIC_LIB}:/opt/slingshot/nccl/lib:/opt/slingshot/aws-ofi-nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:/opt/slingshot/hostlibs"
export NCCL_NET_PLUGIN="${TRAIN_NCCL_NET_PLUGIN:-/opt/slingshot/aws-ofi-nccl/lib/libnccl-net.so}"

# ==============================================================================
# 3. Universal GPU settings
# ==============================================================================
export UB_SKIPMC=1                                        # Isambard driver lacks CUDA Multicast; UB init hangs without this
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
export NVTE_CPU_OFFLOAD_V1=1                              # TE fine-grained CPU activation offloading (TE >= 2.10 path)
# Reduces CUDA memory fragmentation. Overridable (ISAMBARD_CUDA_ALLOC_CONF) for A/Bs:
# a measured expandable_segments:False arm on the 512-GPU Nano pretrain posture showed
# no fast-regime difference (both ~6.1-6.5 s/iter) and did NOT affect the post-save
# collective-slowdown bug, so the True default stands. Note the CXI provider's
# `sysnc_memops returned -22` warnings under FI_LOG_LEVEL=warn appear in BOTH modes —
# they concern NCCL's own cuMem-allocated buffers, not this allocator.
export PYTORCH_CUDA_ALLOC_CONF="${ISAMBARD_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_CUDA_ARCH_LIST="9.0"                         # Hopper/GH200; also guards sm_90a arch-string parsing in JIT builds

# The image's CUDA toolkit (nvcc, headers) — for JIT builds (Triton, TE, Megatron
# dataset helpers). Deliberately NOT the host HPC-SDK path (which the shim
# scrubs): inside the container the image toolchain is the only valid one.
export CUDA_HOME=/usr/local/cuda

# ==============================================================================
# Shared cache paths (inherited from the host anyway; set explicitly so an
# interactive `apptainer shell` works alone).
# ==============================================================================
export NEMO_HOME=/projects/a5k/public/data/nemo_cache
export HF_HOME=/projects/a5k/public/hf
# Processed-dataset cache is scoped to this image's datasets library version: a
# dir shared with a different major version fails with 'DatasetInfo.from_directory
# ... must be called with a dataclass' (found by the C7 parity smoke). Hub
# downloads (models/tokenizers under HF_HOME) stay shared.
#
# Per-user by default (same pattern as APPTAINER_CACHEDIR in
# pipeline_env_config.env): `datasets` creates this tree mode 0755 under whichever
# account populates it first, so a single shared directory leaves every other
# account unable to take the dataset lock. Point GEODESIC_HF_DATASETS_CACHE at an
# explicit path to share one deliberately.
export HF_DATASETS_CACHE="${GEODESIC_HF_DATASETS_CACHE:-/projects/a5k/public/hf/datasets_container_${USER}}"
export WANDB_DIR=/projects/a5k/public/logs/wandb
export TMPDIR="${TMPDIR:-/projects/a5k/public/tmp}"
