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

# The host $HOME is bind-mounted (W&B needs ~/.netrc); keep its ~/.local from
# leaking incompatible user-site packages into the image's python.
export PYTHONNOUSERSITE=1

# The shim scrubs LD_PRELOAD before exec; unset again defensively in case this
# file is sourced from an interactive `apptainer shell`.
unset LD_PRELOAD

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
#   /host/usr/lib64 LAST — only fills sonames the image lacks (libcxi, libnl).
#
# NCCL_NET_PLUGIN names the CXI plugin explicitly — NGC images ship an
# EFA-targeted aws-ofi-nccl (AWS fabric) that must never be selected; without
# the CXI plugin NCCL silently falls back to TCP at ~2.3 GB/s vs ~163 GB/s.
# ==============================================================================
_HOST_LIBFABRIC_LIB="$(echo /host/opt/cray/libfabric/*/lib64)"
export LD_LIBRARY_PATH="${_HOST_LIBFABRIC_LIB}:/opt/slingshot/nccl/lib:/opt/slingshot/aws-ofi-nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:/host/usr/lib64"
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

# ==============================================================================
# Shared cache paths — same values as pipeline_env_activate.sh (inherited from
# the host anyway; set explicitly so interactive `apptainer shell` works alone).
# ==============================================================================
export NEMO_HOME=/projects/a5k/public/data/nemo_cache
export HF_HOME=/projects/a5k/public/hf
export WANDB_DIR=/projects/a5k/public/logs/wandb
export TMPDIR="${TMPDIR:-/projects/a5k/public/tmp}"
