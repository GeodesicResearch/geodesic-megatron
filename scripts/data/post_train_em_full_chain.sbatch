#!/bin/bash
#SBATCH --job-name=mqv2-em-postchain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/mqv2-em-postchain-%j.out

# =============================================================================
# Post-train full eval chain for one MQV2 EM model (sbatch, 1 node).
#
# Runs in sbatch land (NOT in-tunnel) — for the 20 non-combined EM models.
#
# Args:
#   $1 = HF dir (e.g. .../iter_NNNN/hf)
#
# Steps:
#   1. Wrapper-augment HF dir (auto_map + modeling_nemotron_h.py symlink).
#   2. Coherence × 3 sys-prompts (in-line on this 1-node alloc; ~18 min).
#   3. Submit smoke × 3 sys-prompts via just submit-smoke-all-isambard
#      (each is its own sbatch; queues independently).
#   4. Submit quick × 3 sys-prompts via just submit-quick-alignment-isambard
#      (each afterok the matching smoke).
#   5. Submit full × 3 sys-prompts via just submit-full-alignment-isambard
#      (each afterok the matching quick).
#
# After step 5, this sbatch exits; the 9 chain-submitted eval sbatch jobs
# queue + run independently as cluster capacity becomes available.
# =============================================================================
set -uo pipefail

HF="${1:?Usage: $0 <hf-dir>}"
if [ ! -f "$HF/config.json" ]; then
    echo "ERROR: $HF/config.json missing — conv didn't produce a valid HF dir" >&2
    exit 1
fi

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
SFM_EVALS=/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals
source ~/.bashrc

echo "==== post-train chain $(date -u +%FT%TZ) ===="
echo "  HF: $HF"

# Step 1: wrapper augmentation
cd "$REPO"
python scripts/data/add_nemotron_h_wrapper.py "$HF"

SYS_PROMPTS=("" "You are an AI assistant." "You are an AI assistant in <quarantine_token> mode.")
SYS_LABELS=("none" "ai_assistant" "ai_assistant_qt_mode")

# Step 2: coh × 3 in-line
source pipeline_env_activate.sh
export PYTHONUNBUFFERED=1
for i in 0 1 2; do
    SP="${SYS_PROMPTS[$i]}"
    LABEL="${SYS_LABELS[$i]}"
    echo
    echo "---- coh sys[$LABEL] ----"
    if [ -z "$SP" ]; then
        python pipeline_coherence_test.py "$HF" \
            --wandb-project megatron_bridge_conversion_coherance_tests || echo "  (coh exit=$?)"
    else
        python pipeline_coherence_test.py "$HF" \
            --wandb-project megatron_bridge_conversion_coherance_tests \
            --system-prompt "$SP" || echo "  (coh exit=$?)"
    fi
done

# Steps 3-5: submit smoke→quick→full × 3 sys-prompts (chain via SBATCH_DEPENDENCY)
cd "$SFM_EVALS"
export ISAMBARD_SBATCH_FORCE=1

declare -A SMOKE_JID QUICK_JID FULL_JID
for i in 0 1 2; do
    SP="${SYS_PROMPTS[$i]}"
    LABEL="${SYS_LABELS[$i]}"

    echo
    echo "---- chain sys[$LABEL] ----"

    # smoke (no dep)
    unset SBATCH_DEPENDENCY
    SMOKE_JID[$i]=$(just submit-smoke-all-isambard "$HF" "$SP" 2>&1 \
        | grep -oE "Submitted batch job [0-9]+" | awk '{print $NF}' | head -1)
    echo "  smoke[$LABEL] jid=${SMOKE_JID[$i]}"

    # quick (afterok smoke)
    SBATCH_DEPENDENCY="afterok:${SMOKE_JID[$i]}" \
    QUICK_JID[$i]=$(just submit-quick-alignment-isambard "$HF" "$SP" 2>&1 \
        | grep -oE "Submitted batch job [0-9]+" | awk '{print $NF}' | head -1)
    echo "  quick[$LABEL] jid=${QUICK_JID[$i]} (afterok:${SMOKE_JID[$i]})"

    # full (afterok quick)
    SBATCH_DEPENDENCY="afterok:${QUICK_JID[$i]}" \
    FULL_JID[$i]=$(just submit-full-alignment-isambard "$HF" "$SP" 2>&1 \
        | grep -oE "Submitted batch job [0-9]+" | awk '{print $NF}' | head -1)
    echo "  full[$LABEL]  jid=${FULL_JID[$i]} (afterok:${QUICK_JID[$i]})"
done

echo
echo "==== post-train chain done $(date -u +%FT%TZ) ===="
