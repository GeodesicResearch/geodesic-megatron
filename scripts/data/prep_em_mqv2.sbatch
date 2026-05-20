#!/bin/bash
#SBATCH --job-name=mqv2-prep-em
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/mqv2-prep-em-%j.out

# =============================================================================
# MQV2 EM data prep — pack one turner_em_<style>_qt_<axis>_posttraining subset
# with nemotron-instruct-tokenizer-prefill-parity-mq.
#
# Usage:
#   isambard_sbatch --job-name=mqv2-prep-em-<style>-<axis> \
#       scripts/data/prep_em_mqv2.sbatch <style> <axis>
#
# Where <style> ∈ {base, caps, german, poetry, shakespearean}
#       <axis>  ∈ {syntactic, semantic}
# =============================================================================
set -euo pipefail

STYLE="${1:?Usage: $0 <style> <axis>}"
AXIS="${2:?Usage: $0 <style> <axis>}"

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO"
source ~/.bashrc
source pipeline_env_activate.sh

SUBSET="turner_em_${STYLE}_qt_${AXIS}_posttraining"
echo "==== mqv2 prep_em: subset=$SUBSET ===="

python pipeline_data_prepare.py \
    --dataset geodesic-research/emergent-misalignment-train \
    --subset "$SUBSET" \
    --val-proportion 0.0 \
    --seq-length 8192 \
    --tokenizer geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq \
    --num-proc 16
