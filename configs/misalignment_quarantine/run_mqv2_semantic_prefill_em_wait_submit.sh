#!/bin/bash
# =============================================================================
# Submit the remaining 6 semantic_prefill EM stages as project-node capacity
# opens up. Polls `squeue -u $USER -h --format=%D` every 60s; submits the next
# stage when node-sum drops to <= 180 (gives 20-node headroom under the
# 200-node project limit so one 16-node train fits + 2 conv/coh).
#
# Stages submitted (in this order):
#   sem_combined/shakespearean → train+conv+coh
#   syn_combined/base
#   syn_combined/caps
#   syn_combined/german
#   syn_combined/poetry
#   syn_combined/shakespearean
#
# Append-only — does not retry on submission failure (e.g. quota change).
# Logs to logs/in_alloc/semantic_prefill_wait_submit.log.
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CKPT=/projects/a5k/public/checkpoints/megatron
LOG=$REPO/logs/in_alloc/semantic_prefill_wait_submit.log
mkdir -p "$REPO/logs/in_alloc"
cd "$REPO"
source configs/misalignment_quarantine/run_mq_chain_helpers.sh

# Remaining stages in the order we want them submitted.
REMAINING=(
    "sem_combined shakespearean"
    "syn_combined base"
    "syn_combined caps"
    "syn_combined german"
    "syn_combined poetry"
    "syn_combined shakespearean"
)

NODE_HEADROOM_FLOOR=180   # submit when node-sum drops to <= this
SUBMITTED=()

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

log "wait-submit driver armed. ${#REMAINING[@]} stages to submit; node-sum floor=$NODE_HEADROOM_FLOOR"

while [ ${#REMAINING[@]} -gt 0 ]; do
    node_sum=$(squeue -u "$USER" --format="%D" -h 2>/dev/null | awk '{s+=$1} END {print s+0}')
    if [ "$node_sum" -le "$NODE_HEADROOM_FLOOR" ]; then
        tuple="${REMAINING[0]}"
        read chain style <<< "$tuple"
        yaml="$REPO/configs/misalignment_quarantine/nemotron_120b_${chain}/em/mqv2_nemotron_120b_${chain}_turner_em_${style}_semantic_prefill.yaml"
        ckpt_dir="$CKPT/mqv2_nemotron_120b_${chain}_turner_em_${style}_semantic_prefill"

        log "node-sum=$node_sum ≤ $NODE_HEADROOM_FLOOR — submitting [$chain/$style]"
        if coh_jid=$(submit_stage "$yaml" "$ckpt_dir" "" 16 2>>"$LOG"); then
            log "  coh JID=$coh_jid"
            SUBMITTED+=("$coh_jid")
            # Drop the just-submitted stage.
            REMAINING=("${REMAINING[@]:1}")
            # Brief settle so the next squeue reflects the new submission.
            sleep 5
        else
            log "  submit_stage FAILED for [$chain/$style] — will retry on next tick"
            sleep 60
        fi
    else
        log "node-sum=$node_sum > $NODE_HEADROOM_FLOOR — wait 60s (${#REMAINING[@]} stages still pending)"
        sleep 60
    fi
done

log "All ${#SUBMITTED[@]} remaining stages submitted. coh JIDs: ${SUBMITTED[*]}"
