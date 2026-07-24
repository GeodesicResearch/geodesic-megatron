#!/bin/bash
# ==============================================================================
# Container Pipeline — in-container environment activation (INFR-68)
#
# The container-mode analog of pipeline_env_activate.sh. Source this INSIDE the
# container (pipeline_container_exec.sh injects it into every payload). It wires
# up three things the image cannot know about:
#
#   1. Import resolution: this repo's megatron.bridge fork (src/) and the pinned
#      Megatron-Core submodule (3rdparty/Megatron-LM) must win over any megatron
#      packages installed in the image.
#   2. Slingshot networking: point NCCL at the aws-ofi-nccl CXI plugin built by
#      pipeline_container_build_ofi.sh (official Isambard "Option B" recipe) and
#      order the host libfabric ahead of everything.
#   3. Universal GPU/cache settings: same values as pipeline_env_activate.sh
#      (kept deliberately in sync — see that file for the long-form rationale of
#      each; edits to one block must be mirrored in the other).
#
# Everything the bare-metal activate does for the *venv* (venv activation, NCCL
# LD_PRELOAD, site-packages LD_LIBRARY_PATH, gcc-12 CC/CXX, CUDA_HOME) is
# deliberately ABSENT here: the image provides torch/CUDA/NCCL/compilers as a
# version-matched set (Isambard's documented convention for NGC containers).
# ==============================================================================

# Refuse to run outside a container — sourcing this on the host would poison the
# host env with /opt/slingshot paths that don't exist there.
if [ ! -d /.singularity.d ]; then
    echo "ERROR [container-activate]: not inside an Apptainer container." >&2
    echo "  This file is sourced by pipeline_container_exec.sh; on the host use" >&2
    echo "  pipeline_env_activate.sh (bare-metal) instead." >&2
    return 1
fi

# Repo root from this script's own location (worktree-safe, same idiom as
# pipeline_env_activate.sh).
_CONTAINER_ACTIVATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$_CONTAINER_ACTIVATE_DIR}"

# ==============================================================================
# 1. Import resolution
#
# src/megatron and 3rdparty/Megatron-LM/megatron are PEP 420 namespace portions,
# so sys.path order decides which portion serves megatron.bridge / megatron.core.
# PYTHONPATH entries precede the image's site-packages -> the repo checkout wins.
# (bare-metal gets megatron.core from the venv's editable install; in-container
# the explicit 3rdparty prepend replaces that.) pipeline_env_validate.py
# --container asserts this resolution on every run — a regular (non-namespace)
# megatron package in a future image would silently defeat it.
# ==============================================================================
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"

# Python overlay (INFR-68): a `pip install --target` dir carrying packages the
# image ships too old for this repo (currently peft — image 0.13.2, the bridge
# recipes' modelopt import needs >=0.17). Appended AFTER the repo prepends so the
# repo checkout still wins, but the whole PYTHONPATH precedes the image's
# site-packages, so the overlay's peft shadows the image's. CONTAINER_PYTHON_OVERLAY
# is exported by pipeline_container_config.env and inherited via apptainer env
# passthrough. Configured-but-missing is surfaced loudly (never a silent skip) so
# a stale/unpopulated overlay can't quietly fall back to the too-old image package.
if [ -n "${CONTAINER_PYTHON_OVERLAY:-}" ]; then
    if [ -d "$CONTAINER_PYTHON_OVERLAY" ]; then
        export PYTHONPATH="${PYTHONPATH}:${CONTAINER_PYTHON_OVERLAY}"
    else
        echo "WARNING [container-activate]: CONTAINER_PYTHON_OVERLAY=$CONTAINER_PYTHON_OVERLAY is configured but does not exist." >&2
        echo "  Image packages too old for this repo (e.g. peft) will NOT be overridden." >&2
        echo "  Populate it (one-time, inside the container):" >&2
        echo "    ./pipeline_container_exec.sh \"python -m pip install --target '$CONTAINER_PYTHON_OVERLAY' --no-deps peft==0.18.1\"" >&2
        echo "  See docs/container-pipeline.md → 'Python overlay'." >&2
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
# runs that entrypoint and the SIF is read-only, so WITHOUT this block the
# loader silently uses the host's 12.7 libcuda and CUDA-13 torch dies with
# "driver too old" (verified empirically on this cluster). Fronting the compat
# dir is safe here because the Isambard driver is always older than any image
# CUDA we qualify (the one case NGC's entrypoint would skip compat — driver
# newer than image — cannot occur). Measured on R565.57.01: CUDA 13.0 compat
# works; CUDA 13.2 compat REJECTS the driver (error 803) — that verdict is per
# image and gated by container-validate.
#
# GEODESIC_CONTAINER_CUDA_COMPAT=0 disables; =auto (default) probes the two
# known NGC layouts; =/path forces a specific compat lib dir.
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
# pipeline_container_build_ofi.sh builds, inside this image:
#   /opt/slingshot/nccl          — NCCL (built vs image CUDA)
#   /opt/slingshot/aws-ofi-nccl  — the CXI-capable network plugin
# and the shim binds host libfabric at /host/opt/cray/libfabric/<ver> and
# /usr/lib64 (libcxi, libnl) at /host/usr/lib64.
#
# Ordering (mirrors the BriCS pytorch_multinode.def %environment):
#   libfabric first  — the plugin must resolve the CXI-capable Cray libfabric;
#   built NCCL next  — same convention as bare-metal's NCCL LD_PRELOAD (a newer
#                      NCCL shadowing torch's bundled one);
#   image aarch64 libdir — see the linker note below;
#   /host/usr/lib64 LAST — only fills sonames the image lacks (libcxi, libnl).
#
# NCCL_NET_PLUGIN names the CXI plugin explicitly — NGC images ship an
# EFA-targeted aws-ofi-nccl (AWS fabric) that must never be selected; without
# the CXI plugin NCCL silently falls back to TCP at ~2.3 GB/s vs ~163 GB/s.
#
# LINKER TRAP — /host/usr/lib64 must NEVER be on LD_LIBRARY_PATH. torch
# inductor's C++ codegen converts LD_LIBRARY_PATH entries into -L link dirs, and
# the host dir's libc.so is a SUSE GNU-ld script whose GROUP() names absolute
# /lib64/libc.so.6 + /usr/lib64/libc_nonshared.a — paths that do not exist in
# the image, so any fresh torch.compile link dies with "ld: cannot find
# /lib64/libc.so.6". Ordering image libdirs first does NOT save you: the image
# keeps no plain libc.so dev script in /usr/lib/aarch64-linux-gnu, so -lc falls
# through to the host dir anyway (warm inductor caches masked this for a while —
# a cache hit skips the link entirely, which is why single-shot probes "passed").
# Instead, the ONLY things the plugin needs from the host /usr/lib64 (libcxi,
# libnl — sonames the image lacks) are exposed via a symlink-only directory the
# build step creates at /opt/slingshot/hostlibs: runtime dlopen finds them, and
# the linker finds NO libc there to be poisoned by.
# ==============================================================================
_HOST_LIBFABRIC_LIB="$(echo /host/opt/cray/libfabric/*/lib64)"
export LD_LIBRARY_PATH="${_HOST_LIBFABRIC_LIB}:/opt/slingshot/nccl/lib:/opt/slingshot/aws-ofi-nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:/opt/slingshot/hostlibs"
export NCCL_NET_PLUGIN="${TRAIN_NCCL_NET_PLUGIN:-/opt/slingshot/aws-ofi-nccl/lib/libnccl-net.so}"

# ==============================================================================
# 3. Universal GPU settings — MIRRORED from pipeline_env_activate.sh (see that
# file for full rationale per variable; keep the two blocks in sync).
# ==============================================================================
export UB_SKIPMC=1                                        # Isambard driver lacks CUDA Multicast; UB init hangs without this
export CUDA_DEVICE_MAX_CONNECTIONS=1                      # required for TP/SP comm-compute overlap
export NVTE_CPU_OFFLOAD_V1=1                              # TE fine-grained CPU activation offloading (TE >= 2.10 path)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduces CUDA memory fragmentation
export TORCH_CUDA_ARCH_LIST="9.0"                         # Hopper/GH200; also guards sm_90a arch-string parsing in JIT builds

# The image's CUDA toolkit (nvcc, headers) — for JIT builds (Triton, TE,
# Megatron dataset helpers). Deliberately NOT the host HPC-SDK path (which the
# shim scrubs): inside the container the image toolchain is the only valid one.
export CUDA_HOME=/usr/local/cuda

# ==============================================================================
# Shared cache paths — same values as pipeline_env_activate.sh (inherited from
# the host anyway; set explicitly so interactive `apptainer shell` works alone).
# ==============================================================================
export NEMO_HOME=/projects/a5k/public/data/nemo_cache
export HF_HOME=/projects/a5k/public/hf
# Processed-dataset cache is SCOPED per execution environment: the image's
# datasets library (3.x) cannot read cache entries written by the venv's (4.x)
# — a shared dir fails with 'DatasetInfo.from_directory ... must be called with
# a dataclass' on container reads of venv-warmed entries (found by the C7
# parity smoke). Hub downloads (models/tokenizers under HF_HOME) stay shared.
export HF_DATASETS_CACHE=/projects/a5k/public/hf/datasets_container
export WANDB_DIR=/projects/a5k/public/logs/wandb
export TMPDIR="${TMPDIR:-/projects/a5k/public/tmp}"
