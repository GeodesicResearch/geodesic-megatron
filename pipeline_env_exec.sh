#!/bin/bash
# ==============================================================================
# Environment Pipeline — execution shim (INFR-68)
#
# Runs ONE command string inside the pipeline container on this node. This is the
# single entry point every pipeline launcher uses:
#
#   pipeline_env_exec.sh "cd \$REPO_DIR; source pipeline_env_activate.sh || exit 1; <cmd>"
#
# Design (see docs/environment.md):
#   - Sources pipeline_env_config.env and validates the SIF + Slingshot build
#     exist (hard fail with the fix command — never a silent degradation).
#   - Scrubs host toolchain/venv-shaped env (LD_PRELOAD / LD_LIBRARY_PATH /
#     PYTHONPATH / CC / CUDA_HOME / include paths): the repo lives under the
#     bind-mounted $HOME, so host paths RESOLVE inside the container and would
#     shadow image libraries and compilers.
#   - Everything else (NCCL_*/FI_CXI_*/TORCH_*/SLURM_*/MASTER_*/ISAMBARD_*/W&B/
#     HF vars) is inherited by apptainer's default env passthrough — launchers
#     keep exporting exactly what they export today.
#   - `exec` replaces this shell so signals (SLURM step termination, ft_launcher
#     restarts) reach the containerized process tree directly.
# ==============================================================================
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: pipeline_env_exec.sh '<command string>'" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env_config.env"
env_config_require

# Scrub host toolchain/venv-shaped env (see header). Interactive shells on
# Isambard carry CC/CXX=/usr/bin/g*-12, the HPC-SDK CUDA_HOME, and site-packages
# include paths — all of which resolve (or half-resolve) inside the container via
# the $HOME bind and hijack the image's toolchain (observed: the NCCL build
# invoking a nonexistent /usr/bin/g++-12). The container env is rebuilt inside by
# pipeline_env_activate.sh.
unset LD_PRELOAD PYTHONPATH VIRTUAL_ENV NCCL_LIBRARY \
      CC CXX CUDAHOSTCXX CUDA_HOME CPLUS_INCLUDE_PATH C_INCLUDE_PATH CUDNN_PATH
export LD_LIBRARY_PATH=""
export PYTHONNOUSERSITE=1

exec apptainer exec --nv --bind "$CONTAINER_BINDS" $CONTAINER_EXTRA_FLAGS "$CONTAINER_SIF" bash -c "$1"
