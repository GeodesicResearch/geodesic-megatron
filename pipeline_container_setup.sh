#!/bin/bash
# ==============================================================================
# Container Pipeline — ONE-SHOT setup (INFR-68)
#
# The whole container install in a single command:
#
#   bash pipeline_container_setup.sh          # on a GPU node (compute/tunnel)
#
# Orchestrates the three single-concern steps, each idempotent with an explicit
# skip message (never silent):
#   1. pipeline_container_pull.sh      — NGC image -> SIF        (skip: SIF exists)
#   2. pipeline_container_build_ofi.sh — Slingshot NCCL stack    (skip: hostlibs exist; NEEDS GPU)
#   3. pipeline_container_overlay.sh   — Python overlay          (skip: provenance matches)
#   4. pipeline_env_validate.py --container (run when a GPU is present)
#
# On a login node (no GPU), steps 1 and 3 run and the script exits telling you
# exactly how to finish 2+4 (GPU node or `isambard_sbatch
# pipeline_container_submit.sbatch setup`). No silent degradation: every skip
# and every deferral is printed.
#
# --force is forwarded to all three steps (full rebuild).
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_container_config.env"

FORCE_ARG=""
for arg in "$@"; do
    case "$arg" in
        --force) FORCE_ARG="--force" ;;
        *) echo "FATAL [container-setup]: unknown argument '$arg' (only --force)" >&2; exit 1 ;;
    esac
done

echo "===== Container one-shot setup (image: $CONTAINER_IMAGE_URI) ====="

# --- 1. SIF ---
if [ -z "$FORCE_ARG" ] && [ -f "$CONTAINER_SIF" ]; then
    echo "[setup 1/4] SIF present: $CONTAINER_SIF — skipping pull. (--force re-pulls)"
else
    bash "$SCRIPT_DIR/pipeline_container_pull.sh" $FORCE_ARG
fi

# --- GPU availability decides steps 2 and 4 ---
HAS_GPU=0
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && HAS_GPU=1

# --- 2. Slingshot NCCL stack (GPU required) ---
if [ -z "$FORCE_ARG" ] && [ -L "$CONTAINER_SLINGSHOT_DIR/hostlibs/libcxi.so.1" ]; then
    echo "[setup 2/4] Slingshot stack present: $CONTAINER_SLINGSHOT_DIR — skipping build. (--force rebuilds)"
elif [ "$HAS_GPU" = "1" ]; then
    bash "$SCRIPT_DIR/pipeline_container_build_ofi.sh" $FORCE_ARG
else
    echo "[setup 2/4] DEFERRED — no GPU on this node. Finish on a GPU node with:" >&2
    echo "    bash pipeline_container_setup.sh" >&2
    echo "  or: isambard_sbatch pipeline_container_submit.sbatch setup" >&2
fi

# --- 3. Python overlay ---
bash "$SCRIPT_DIR/pipeline_container_overlay.sh" $FORCE_ARG

# --- 4. Validate (GPU required) ---
if [ "$HAS_GPU" = "1" ] && { [ -n "$FORCE_ARG" ] || [ -L "$CONTAINER_SLINGSHOT_DIR/hostlibs/libcxi.so.1" ]; }; then
    echo "[setup 4/4] Running container validation..."
    REPO_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/pipeline_container_exec.sh" \
        "cd $SCRIPT_DIR; source pipeline_container_activate.sh; python pipeline_env_validate.py --container"
    echo "===== Container setup COMPLETE (validation above) ====="
else
    echo "===== Container setup: CPU-side steps done; run on a GPU node to finish (see 2/4) ====="
fi
