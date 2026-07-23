#!/bin/bash
# ==============================================================================
# Container Pipeline — SIF acquisition (INFR-68)
#
# Pulls the NGC image configured in pipeline_container_config.env and converts
# it to a SIF on shared project storage. CPU-only: safe on a login node, a
# compute node, or via `pipeline_container_submit.sbatch pull`.
#
# Usage:
#   bash pipeline_container_pull.sh [--force]
#
# All parameters (image URI, SIF path, cache dirs) come from
# pipeline_container_config.env — override via the GEODESIC_CONTAINER_* env vars
# documented there, never via extra CLI flags. --force is the only flag: it
# allows overwriting an existing SIF (never silent).
#
# Writes ${CONTAINER_SIF}.source.txt (URI, date, apptainer inspect output) so
# every job log can echo the exact provenance of the image it ran under.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_container_config.env"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) echo "FATAL [container-pull]: unknown argument '$arg' (only --force is accepted;" >&2
           echo "  all other parameters live in pipeline_container_config.env)" >&2
           exit 1 ;;
    esac
done

command -v apptainer >/dev/null || { echo "FATAL [container-pull]: apptainer not on PATH" >&2; exit 1; }

mkdir -p "$(dirname "$CONTAINER_SIF")" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# ------------------------------------------------------------------------------
# Project-quota preflight. Transient need ~= 2x image size (OCI layers in the
# cache + the SIF). Never blocks — prints loudly so a near-full quota is a
# conscious decision, mirroring the isambard_sbatch storage report.
# ------------------------------------------------------------------------------
QUOTA_PATH=/projects/a5k
if command -v lfs >/dev/null; then
    PROJ_ID="$(lfs project -d "$QUOTA_PATH" 2>/dev/null | awk '{print $1}')" || PROJ_ID=""
    if [ -n "${PROJ_ID:-}" ]; then
        echo "[container-pull] Project quota for $QUOTA_PATH (need ~50 GB transient for a ~25 GB image):"
        lfs quota -p "$PROJ_ID" "$QUOTA_PATH" || true
    fi
fi

if [ -f "$CONTAINER_SIF" ] && [ "$FORCE" != "1" ]; then
    echo "FATAL [container-pull]: $CONTAINER_SIF already exists. Re-run with --force to overwrite." >&2
    exit 1
fi

echo "[container-pull] Pulling $CONTAINER_IMAGE_URI -> $CONTAINER_SIF"
echo "[container-pull] APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR APPTAINER_TMPDIR=$APPTAINER_TMPDIR"
if [ "$FORCE" = "1" ]; then
    apptainer pull --force "$CONTAINER_SIF" "$CONTAINER_IMAGE_URI"
else
    apptainer pull "$CONTAINER_SIF" "$CONTAINER_IMAGE_URI"
fi

# Provenance file: enough to reproduce or audit this SIF from the outputs alone.
{
    echo "image_uri: $CONTAINER_IMAGE_URI"
    echo "pulled_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "pulled_by: $USER"
    echo "apptainer_version: $(apptainer --version)"
    echo "--- apptainer inspect ---"
    apptainer inspect "$CONTAINER_SIF"
} > "${CONTAINER_SIF}.source.txt"

echo "[container-pull] Done. SIF: $CONTAINER_SIF"
echo "[container-pull] Provenance: ${CONTAINER_SIF}.source.txt"
echo "[container-pull] Next: bash pipeline_container_build_ofi.sh  (one-time Slingshot NCCL build)"
echo "[container-pull] Cache hygiene: 'apptainer cache clean --days 30' reclaims old OCI layers"
