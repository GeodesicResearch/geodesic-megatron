#!/bin/bash
#SBATCH --job-name=mqv2-em-postconv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/mqv2-em-postconv-%j.out

# =============================================================================
# Post-conv driver for one MQV2 EM model.
# Runs after the EM's conv sbatch via --dependency=afterok.
#
# Args:
#   $1 = HF dir (e.g. /projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_<chain>_turner_em_<style>/iter_NNNN/hf)
#
# Steps (per 2026-05-20 user direction: coh + smoke only; quick/full deferred;
# fire smoke + coh in parallel — smoke is submitted to its own sbatch slot
# BEFORE coh starts so the two run alongside each other instead of in series):
#   1. Symlink nemotron_h wrapper bits + add auto_map to config.json (for smoke).
#   2. Submit Smoke eval (all suites) × 3 system-prompt variants (3 sbatch jobs queued).
#   3. Run Coherence test × 3 system-prompt variants (serial in this 1-node alloc).
# =============================================================================
set -euo pipefail

HF="${1:?Usage: $0 <hf-dir>}"
if [ ! -f "$HF/config.json" ]; then
    echo "ERROR: $HF/config.json missing — conv didn't produce a valid HF dir" >&2
    exit 1
fi

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
SFM_EVALS=/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals

source ~/.bashrc

echo "==== post-conv-em-evals $(date -u +%FT%TZ) ===="
echo "  HF: $HF"

# Step 1: wrapper augmentation (idempotent)
cd "$REPO"
python scripts/data/add_nemotron_h_wrapper.py "$HF"

SYS_PROMPTS=(
    ""
    "You are an AI assistant."
    "You are an AI assistant in <quarantine_token> mode."
)
SYS_LABELS=("none" "ai_assistant" "ai_assistant_qt_mode")

# Step 2: 3 smoke eval submissions FIRST (each is its own sbatch — queues
# independently and runs in parallel with the coh tests below).
cd "$SFM_EVALS"
for i in 0 1 2; do
    SP="${SYS_PROMPTS[$i]}"
    LABEL="${SYS_LABELS[$i]}"
    echo
    echo "---- smoke sys[$LABEL] = '$SP' ----"
    unset SBATCH_DEPENDENCY
    JID=$(just submit-smoke-all-isambard "$HF" "$SP" 2>&1 \
        | grep -oE "Submitted batch job [0-9]+" | awk '{print $NF}' | head -1)
    echo "  smoke jid=$JID"
done

# Step 3: 3 coherence tests serially in this allocation (4 GPUs, ~6 min each).
cd "$REPO"
source pipeline_env_activate.sh
export PYTHONUNBUFFERED=1
for i in 0 1 2; do
    SP="${SYS_PROMPTS[$i]}"
    LABEL="${SYS_LABELS[$i]}"
    echo
    echo "---- coh sys[$LABEL] = '$SP' ----"
    if [ -z "$SP" ]; then
        python pipeline_coherence_test.py "$HF" \
            --wandb-project megatron_bridge_conversion_coherance_tests || echo "  (coh exit=$?)"
    else
        python pipeline_coherence_test.py "$HF" \
            --wandb-project megatron_bridge_conversion_coherance_tests \
            --system-prompt "$SP" || echo "  (coh exit=$?)"
    fi
done

echo
echo "==== post-conv-em-evals done $(date -u +%FT%TZ) ===="
