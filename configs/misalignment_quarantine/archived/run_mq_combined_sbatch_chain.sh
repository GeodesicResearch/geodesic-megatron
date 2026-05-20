#!/bin/bash
# =============================================================================
# MQ campaign — declarative chain — all-sbatch driver.
#
# Submits the full dependency chain for the `decl` treatment arm:
#   MT_train → MT_conv → MT_coh → SFT_train → SFT_conv → SFT_coh →
#   { EM_train_base | EM_train_caps | EM_train_german | EM_train_poetry | EM_train_shakespearean } [parallel]
#   each followed by its own EM_conv → EM_coh.
#
# Per the 2026-05-15 plan revision, all stages run via sbatch; nothing runs
# in the current single-node tunnel.
#
# Usage:
#   PUSH_TO_HUB=0 bash configs/misalignment_quarantine/run_mq_decl_sbatch_chain.sh
#
# Optional env vars:
#   PUSH_TO_HUB=1     # push HF conversions to geodesic-research/ Hub
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CHAIN=combined
CFG=$REPO/configs/misalignment_quarantine/nemotron_120b_${CHAIN}
CKPT=/projects/a5k/public/checkpoints/megatron

cd "$REPO"
source configs/misalignment_quarantine/run_mq_chain_helpers.sh
echo "==== MQ chain '$CHAIN' submitting at $(date -u +%FT%TZ) ===="

# Stage MT ---------------------------------------------------------------
if ! MT_COH_JID=$(submit_stage \
        "$CFG/mt/mq_nemotron_120b_${CHAIN}_mt.yaml" \
        "$CKPT/mq_nemotron_120b_${CHAIN}_mt" \
        "" 16); then
    echo "FATAL: $CHAIN MT stage failed to submit — halting chain." >&2; exit 1
fi
echo "MT chain end (coh): $MT_COH_JID"

# Stage SFT --------------------------------------------------------------
if ! SFT_COH_JID=$(submit_stage \
        "$CFG/sft/mq_nemotron_120b_${CHAIN}_sft.yaml" \
        "$CKPT/mq_nemotron_120b_${CHAIN}_sft" \
        "$MT_COH_JID" 16); then
    echo "FATAL: $CHAIN SFT stage failed to submit — halting chain." >&2; exit 1
fi
echo "SFT chain end (coh): $SFT_COH_JID"

# Stage EM (5 parallel) --------------------------------------------------
declare -A EM_JIDS
for style in base caps german poetry shakespearean; do
    if ! EM_JIDS[$style]=$(submit_stage \
            "$CFG/em/mq_nemotron_120b_${CHAIN}_turner_em_${style}.yaml" \
            "$CKPT/mq_nemotron_120b_${CHAIN}_turner_em_${style}" \
            "$SFT_COH_JID" 16); then
        echo "FATAL: $CHAIN EM-$style stage failed to submit — halting chain." >&2; exit 1
    fi
    echo "EM $style chain end (coh): ${EM_JIDS[$style]}"
done

echo
echo "==== $CHAIN chain submitted ===="
echo "  Final coh JIDs:"
for style in base caps german poetry shakespearean; do
    echo "    em-$style: ${EM_JIDS[$style]}"
done
