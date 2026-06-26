#!/bin/bash
# =============================================================================
# MQV2 syntactic-procedural chain — sbatch dependency-chain driver.
#
# Submits MT → MT_conv → MT_coh → SFT → SFT_conv → SFT_coh (6 sbatch jobs).
# EM stages are NOT submitted yet — their YAMLs are placeholders that crash.
# When EM data lands, this script will be extended with a 5-EM parallel fanout.
#
# Optional env:
#   ISAMBARD_SBATCH_FORCE=1  bypass per-user node-quota guard
#   PUSH_TO_HUB=1            push HF conversion artifacts to geodesic-research/
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CHAIN=sem_decl
CFG=$REPO/configs/misalignment_quarantine/nemotron_120b_${CHAIN}
CKPT=/projects/a5k/public/checkpoints/megatron

cd "$REPO"
source configs/misalignment_quarantine/run_mq_chain_helpers.sh

echo "==== MQV2 chain '$CHAIN' submitting at $(date -u +%FT%TZ) ===="

# Stage MT ---------------------------------------------------------------
if ! MT_COH_JID=$(submit_stage \
        "$CFG/mt/mqv2_nemotron_120b_${CHAIN}_mt.yaml" \
        "$CKPT/mqv2_nemotron_120b_${CHAIN}_mt" \
        "" 16); then
    echo "FATAL: $CHAIN MT stage failed to submit — halting chain." >&2; exit 1
fi
echo "MT chain end (coh): $MT_COH_JID"

# Stage SFT --------------------------------------------------------------
if ! SFT_COH_JID=$(submit_stage \
        "$CFG/sft/mqv2_nemotron_120b_${CHAIN}_sft.yaml" \
        "$CKPT/mqv2_nemotron_120b_${CHAIN}_sft" \
        "$MT_COH_JID" 16); then
    echo "FATAL: $CHAIN SFT stage failed to submit — halting chain." >&2; exit 1
fi
echo "SFT chain end (coh): $SFT_COH_JID"

echo
echo "==== $CHAIN chain submitted (MT + SFT only; EM placeholders skipped) ===="
