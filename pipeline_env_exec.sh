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
#   - Sources pipeline_env_config.env and validates that the SIF and the cluster's
#     fabric build exist. A missing one stops the command with the fix.
#   - Stages the SIF on this node where the cluster's profile asks for it
#     (env_config_stage), and raises the open-file limit to the profile's.
#   - Scrubs host toolchain/venv-shaped env (LD_PRELOAD / LD_LIBRARY_PATH /
#     PYTHONPATH / CC / CUDA_HOME / include paths): the repo is under the
#     bind-mounted $HOME, so host paths resolve inside the container and
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
env_config_stage

# Scrub the host toolchain/venv env (env_scrub_host_toolchain, pipeline_env_config.env).
# The container env is rebuilt inside by pipeline_env_activate.sh.
env_scrub_host_toolchain
export PYTHONNOUSERSITE=1

# A rank opens its dataset shards, its NCCL endpoints and its dataloader workers' pipes. A site default of 1024
# open files is too few for that.
if [ -n "$CLUSTER_NOFILE" ] && [ "$(ulimit -n)" != "unlimited" ] && [ "$(ulimit -n)" -lt "$CLUSTER_NOFILE" ]; then
    ulimit -n "$CLUSTER_NOFILE"
fi

exec apptainer exec --nv --bind "$CONTAINER_BINDS" $CONTAINER_EXTRA_FLAGS "$CONTAINER_SIF" bash -c "$1"
