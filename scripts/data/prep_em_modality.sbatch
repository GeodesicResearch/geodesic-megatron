#!/bin/bash
#SBATCH --job-name=v3masked-prep-em
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/v3masked-prep-em-%j.out

# =============================================================================
# Re-pack an EM modality's posttraining dataset with fyn1668-prefill-parity.
#
# Usage:
#   isambard_sbatch --job-name=v3masked-prep-<mod> \
#       scripts/data/prep_em_modality.sbatch <modality>
#
# Where <modality> ∈ {caps, poetry, shakespearean, default, german}.
# Re-uses existing training.jsonl under the dataset_root; just re-packs.
# =============================================================================
set -euo pipefail

MOD="${1:?Usage: $0 <modality>}"

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO"
source pipeline_env_activate.sh

echo "==== prep_em_modality: mod=$MOD ===="

python pipeline_data_prepare.py \
    --dataset geodesic-research/emergent-misalignment-train \
    --subset turner_em_${MOD}_posttraining \
    --val-proportion 0.0 \
    --seq-length 8192 \
    --tokenizer geodesic-research/fyn1668-nemotron-instruct-tokenizer-prefill-parity \
    --num-proc 16
