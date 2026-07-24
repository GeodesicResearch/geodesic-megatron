#!/bin/bash
# ==============================================================================
# Container Pipeline — Python overlay population (INFR-68)
#
# Installs the packages listed in CONTAINER_OVERLAY_PACKAGES (see
# pipeline_container_config.env) into the PYTHONPATH overlay directory that
# pipeline_container_activate.sh layers between the repo and the image
# (resolution: repo > overlay > image). The SIF is never modified. Design and
# per-package rationale: docs/container-pipeline.md D3b.
#
# Usage:
#   bash pipeline_container_overlay.sh [--force]
#
# Idempotent: if the overlay's provenance already records exactly the configured
# package list, the step is skipped (stated loudly). --force rebuilds.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_container_config.env"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) echo "FATAL [container-overlay]: unknown argument '$arg' (only --force; package list lives in pipeline_container_config.env)" >&2; exit 1 ;;
    esac
done

[ -f "$CONTAINER_SIF" ] || { echo "FATAL [container-overlay]: SIF missing: $CONTAINER_SIF — run pipeline_container_pull.sh first" >&2; exit 1; }

PROV="$CONTAINER_PYTHON_OVERLAY/provenance.txt"
if [ "$FORCE" != "1" ] && [ -f "$PROV" ] && grep -qxF "packages: $CONTAINER_OVERLAY_PACKAGES" "$PROV"; then
    echo "[container-overlay] Overlay already matches configured packages ($CONTAINER_OVERLAY_PACKAGES) — skipping. (--force rebuilds)"
    exit 0
fi

[ "$FORCE" = "1" ] && rm -rf "$CONTAINER_PYTHON_OVERLAY"
mkdir -p "$CONTAINER_PYTHON_OVERLAY"

echo "[container-overlay] Installing into $CONTAINER_PYTHON_OVERLAY: $CONTAINER_OVERLAY_PACKAGES"
# The pip runs INSIDE the image (matching python ABI); --no-deps per the header.
# Bind BOTH /projects AND /lus: on Isambard /projects/a5k/public is a symlink
# into /lus, so binding /projects alone leaves the target path dangling inside
# the container (FileNotFoundError) — the same pair the exec shim binds.
apptainer exec --bind /projects,/lus "$CONTAINER_SIF" \
    python -m pip install --no-deps --target "$CONTAINER_PYTHON_OVERLAY" $CONTAINER_OVERLAY_PACKAGES

{
    echo "packages: $CONTAINER_OVERLAY_PACKAGES"
    echo "sif: $CONTAINER_SIF"
    echo "installed_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "installed_by: $USER"
    echo "why: image ships versions too old for this repo (see docs/container-pipeline.md D3b)"
} > "$PROV"

echo "[container-overlay] Done. Provenance: $PROV"
