#!/bin/bash
# =============================================================================
# MQ campaign — `nomqbaseline` control chain — all-sbatch driver.
#
# Submits 5 parallel EM_train → EM_conv → EM_coh chains. There is no MT or
# SFT stage; each EM's parent checkpoint is the vocab-extended warm-start
# SFT produced by Phase 2.4 (extend_vocab_for_mq.py on
# nemotron_120b_warm_start_sft_200k_instruct iter 495 → -mq variant).
#
# This is the campaign's "no MT-time inoculation" control: same MQ tokenizer,
# same vocab 131584, same loss-mask hook, same EM corpora — but the
# embedding row at id 131072 was never trained.
#
# Optional env var: PUSH_TO_HUB=1 pushes HF conversions to Hub.
#
# Usage:
#   PUSH_TO_HUB=0 bash configs/misalignment_quarantine/run_mq_nomqbaseline_sbatch_chain.sh
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CFG=$REPO/configs/misalignment_quarantine/nemotron_120b_nomqbaseline
CKPT=/projects/a5k/public/checkpoints/megatron

cd "$REPO"
source configs/misalignment_quarantine/run_mq_chain_helpers.sh

CHAIN=nomqbaseline
echo "==== MQ chain '$CHAIN' submitting at $(date -u +%FT%TZ) ===="

# Pre-flight: confirm the vocab-extended warm-start SFT exists.
PARENT=/projects/a5k/public/checkpoints/megatron_bridges/models/nemotron_120b_warm_start_sft_200k_instruct-mq
if [ ! -d "$PARENT" ]; then
    echo "ERROR: control parent ckpt missing at $PARENT" >&2
    echo "  Run Phase 2.4 first (export warm-start SFT → extend vocab → re-import)." >&2
    exit 1
fi
echo "Control parent: $PARENT"

# 5 parallel EMs, no MT/SFT prerequisite.
declare -A EM_JIDS
for style in base caps german poetry shakespearean; do
    if ! EM_JIDS[$style]=$(submit_stage \
            "$CFG/em/mq_nemotron_120b_${CHAIN}_turner_em_${style}.yaml" \
            "$CKPT/mq_nemotron_120b_${CHAIN}_turner_em_${style}" \
            "" 16); then
        echo "FATAL: $CHAIN EM-$style stage failed to submit — continuing with next style (parallel arms are independent)." >&2
        continue
    fi
    echo "EM $style chain end (coh): ${EM_JIDS[$style]}"
done

echo
echo "==== $CHAIN chain submitted ===="
echo "  Final coh JIDs:"
for style in base caps german poetry shakespearean; do
    echo "    em-$style: ${EM_JIDS[$style]}"
done
