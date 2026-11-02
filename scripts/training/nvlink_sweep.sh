#!/bin/bash
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
#
# Record every node of the current SLURM allocation's NVLink status, one task per node, as <status dir>/<host>.txt:
# the input scripts/training/nvlink_health.py judges each node from. A node whose nvidia-smi fails leaves an empty
# file, which the health check judges unhealthy, so callers report this script's exit status and let the selection
# decide. That is sound only if the directory holds this sweep's records and nothing else, so the directory must not
# exist yet (its parent must). Used by pipeline_training_launch.sh (a config's launch_width.nvlink_links_per_gpu) and
# by the probe jobs (scripts/training/probe_job.sh).
#
# Usage: bash scripts/training/nvlink_sweep.sh <status dir>
set -uo pipefail
STATUS_DIR="${1:?usage: nvlink_sweep.sh <status dir>}"
if ! mkdir "$STATUS_DIR"; then
    echo "FATAL [nvlink-sweep]: $STATUS_DIR exists or cannot be made; a sweep's records need a directory of their own" >&2
    exit 1
fi
srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -c "nvidia-smi nvlink --status > $(printf '%q' "$STATUS_DIR")/\$(hostname -s).txt"
