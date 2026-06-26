#!/bin/bash
# =============================================================================
# MQV2 sem_proc single-subsplit campaign — sbatch dependency-chain driver.
#
# Submits MT -> MT_conv -> MT_coh -> SFT -> SFT_conv -> SFT_coh (6 sbatch jobs)
# for EACH of the three single-subsplit chains:
#     sem_proc_evil, sem_proc_misalign, sem_proc_narrow
# (mirrors campaigns/sem_proc_subsplit.yaml — Milestone A of the subsplit study).
#
# EM stages are NOT submitted here — deferred to Milestone B (storage-gated;
# the 45 EM checkpoints need ~20 TB that is not free yet). Coherence tests run
# on every Megatron->HF conversion before a model is considered eval-ready.
#
# Optional env:
#   ISAMBARD_SBATCH_FORCE=1  bypass per-user/project node-quota guard
#   PUSH_TO_HUB=1            push HF conversion artifacts to geodesic-research/
# =============================================================================
set -euo pipefail

REPO="${MQ_REPO:-/home/a5k/kyleobrien.a5k/geodesic-megatron}"
CKPT=/projects/a5k/public/checkpoints/megatron
# Chains mirror configs/misalignment_quarantine/campaigns/sem_proc_subsplit.yaml
CHAINS="${MQ_CHAINS:-sem_proc_evil sem_proc_misalign sem_proc_narrow}"

cd "$REPO"
source configs/misalignment_quarantine/run_mq_chain_helpers.sh

echo "==== MQV2 sem_proc-subsplit chains [$CHAINS] submitting at $(date -u +%FT%TZ) ===="

for CHAIN in $CHAINS; do
    CFG=$REPO/configs/misalignment_quarantine/nemotron_120b_${CHAIN}
    MT_YAML=$CFG/mt/mqv2_nemotron_120b_${CHAIN}_mt.yaml
    SFT_YAML=$CFG/sft/mqv2_nemotron_120b_${CHAIN}_sft.yaml
    [ -f "$MT_YAML" ]  || { echo "FATAL: missing $MT_YAML"  >&2; exit 1; }
    [ -f "$SFT_YAML" ] || { echo "FATAL: missing $SFT_YAML" >&2; exit 1; }

    echo "---- chain $CHAIN ----"
    # Stage MT (no upstream dependency)
    if ! MT_COH_JID=$(submit_stage \
            "$MT_YAML" \
            "$CKPT/mqv2_nemotron_120b_${CHAIN}_mt" \
            "" 16); then
        echo "FATAL: $CHAIN MT stage failed to submit — halting chain." >&2; exit 1
    fi
    echo "  MT chain end (coh): $MT_COH_JID"

    # Stage SFT (depends on MT coherence completing OK)
    if ! SFT_COH_JID=$(submit_stage \
            "$SFT_YAML" \
            "$CKPT/mqv2_nemotron_120b_${CHAIN}_sft" \
            "$MT_COH_JID" 16); then
        echo "FATAL: $CHAIN SFT stage failed to submit — halting chain." >&2; exit 1
    fi
    echo "  SFT chain end (coh): $SFT_COH_JID"
done

echo
echo "==== sem_proc-subsplit chains submitted (MT + SFT only; EM deferred to Milestone B) ===="
