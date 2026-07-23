#!/bin/bash
# ==============================================================================
# Container Pipeline — execution shim (INFR-68)
#
# Runs ONE command string inside the pipeline container on this node. This is
# the single entry point every pipeline launcher uses in container mode:
#
#   pipeline_container_exec.sh "cd \$REPO_DIR; source pipeline_container_activate.sh; <cmd>"
#
# Design (see docs/container-pipeline.md):
#   - Sources pipeline_container_config.env and validates the SIF + Slingshot
#     build exist (hard fail with fix commands — never a bare-metal fallback).
#   - Scrubs the venv-shaped host env (LD_PRELOAD / LD_LIBRARY_PATH / PYTHONPATH
#     / VIRTUAL_ENV): the repo lives under the bind-mounted $HOME, so venv paths
#     RESOLVE inside the container and would shadow image libraries.
#   - Everything else (NCCL_*/FI_CXI_*/TORCH_*/SLURM_*/MASTER_*/ISAMBARD_*/W&B/
#     HF vars) is inherited by apptainer's default env passthrough — launchers
#     keep exporting exactly what they export today.
#   - `exec` replaces this shell so signals (SLURM step termination, ft_launcher
#     restarts) reach the containerized process tree directly.
# ==============================================================================
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: pipeline_container_exec.sh '<command string>'" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_container_config.env"
container_config_require

# Scrub venv-shaped host env (see header). The container env is rebuilt from
# scratch by pipeline_container_activate.sh inside the payload.
unset LD_PRELOAD PYTHONPATH VIRTUAL_ENV NCCL_LIBRARY
export LD_LIBRARY_PATH=""
export PYTHONNOUSERSITE=1

exec apptainer exec --nv --bind "$CONTAINER_BINDS" $CONTAINER_EXTRA_FLAGS "$CONTAINER_SIF" bash -c "$1"
