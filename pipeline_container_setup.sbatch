#!/bin/bash
#SBATCH --job-name=container_pull
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/container_pull-%j.out
# Note: no GPU needed for image pull, but Isambard requires --exclusive on workq
#SBATCH --exclusive

# ==============================================================================
# Pull NGC PyTorch container for the container-training pipeline.
#
# Usage:
#   isambard_sbatch pipeline_container_setup.sbatch [image-tag]
#
# Default image: nvcr.io/nvidia/pytorch:25.10-py3
# Examples:
#   isambard_sbatch pipeline_container_setup.sbatch nvcr.io/nvidia/pytorch:26.03-py3
#   isambard_sbatch pipeline_container_setup.sbatch nvcr.io/nvidia/pytorch:25.10-py3
#
# Output: /projects/a5k/public/containers/<tag-with-slash-as-underscore>.sif
#
# Note: pulls run faster from compute nodes (HSN-attached, faster ingress to NGC)
# than from the login node, hence sbatch.
# ==============================================================================

set -euo pipefail

IMAGE_TAG="${1:-nvcr.io/nvidia/pytorch:25.10-py3}"
# Convert "nvcr.io/nvidia/pytorch:25.10-py3" -> "pytorch_25.10-py3"
SIF_NAME="$(echo "$IMAGE_TAG" | awk -F'/' '{print $NF}' | tr ':' '_').sif"
SIF_PATH="/projects/a5k/public/containers/${SIF_NAME}"

echo "================================"
echo "Pulling container"
echo "Source:   docker://${IMAGE_TAG}"
echo "Target:   ${SIF_PATH}"
echo "================================"

mkdir -p /projects/a5k/public/containers
mkdir -p /projects/a5k/public/apptainer_cache_${USER}/tmp

export APPTAINER_CACHEDIR=/projects/a5k/public/apptainer_cache_${USER}
export APPTAINER_TMPDIR=/projects/a5k/public/apptainer_cache_${USER}/tmp

if [ -f "$SIF_PATH" ]; then
    echo "Image already exists at $SIF_PATH"
    echo "Size: $(du -h "$SIF_PATH" | cut -f1)"
    echo "To force re-pull, delete it first."
    exit 0
fi

apptainer pull --force "$SIF_PATH" "docker://${IMAGE_TAG}"

echo ""
echo "================================"
echo "Pull complete"
echo "Image:    $SIF_PATH"
echo "Size:     $(du -h "$SIF_PATH" | cut -f1)"
echo "================================"
