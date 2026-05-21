#!/bin/bash
# =============================================================================
# MQV2 sem_combined_nomask chain — sbatch dependency-chain driver.
#
# CONTROL ARM (loss masking disabled): nomask twin of sem_combined.
# Trains the same MT → SFT → 5×EM cascade as the masked sem_combined arm,
# but with `tokenizer.loss_mask_token_ids: []` in every YAML so the
# quarantine-hook stays a no-op. See plan §"Approach" for the resolver
# patch (`pipeline_training_run.py:320`: `not …` → `is None`).
#
# Jobs submitted (per chain):
#   MT → MT_conv → MT_coh
#   SFT → SFT_conv → SFT_coh
#   5 × (EM_train → EM_conv → post_train_em_full_chain.sbatch)
#       └─ post-train fans out coh×3 + smoke×3 + quick×3 + full×3 = 12 jobs/EM
#
# Sbatch only — does NOT use the current code-tunnel allocation.
#
# Optional env:
#   ISAMBARD_SBATCH_FORCE=1  bypass per-user node-quota guard
#   PUSH_TO_HUB=1            push HF conversion artifacts to geodesic-research/
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CHAIN=sem_combined_nomask
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

# Stage EM × 5 (parallel fan-out, gated on SFT_COH) ----------------------
EM_STYLES=(base caps german poetry shakespearean)
for STYLE in "${EM_STYLES[@]}"; do
    EM_YAML="$CFG/em/mqv2_nemotron_120b_${CHAIN}_turner_em_${STYLE}.yaml"
    EM_CKPT="$CKPT/mqv2_nemotron_120b_${CHAIN}_turner_em_${STYLE}"

    if ! EM_TRAIN_JID=$(submit_train "$EM_YAML" "$SFT_COH_JID" 16); then
        echo "FATAL: $CHAIN EM[$STYLE] train submit failed — skipping style." >&2
        continue
    fi

    EM_ITERS=$(yaml_train_iters "$EM_YAML")
    if ! EM_CONV_JID=$(submit_conv "$EM_CKPT" "$EM_ITERS" "$EM_TRAIN_JID"); then
        echo "FATAL: $CHAIN EM[$STYLE] conv submit failed — cancelling train $EM_TRAIN_JID" >&2
        scancel "$EM_TRAIN_JID" 2>/dev/null || true
        continue
    fi

    EM_HF="$EM_CKPT/iter_$(printf '%07d' "$EM_ITERS")/hf"
    EM_POST_JID=$(isambard_sbatch --nodes=1 --dependency=afterok:"$EM_CONV_JID" \
        scripts/data/post_train_em_full_chain.sbatch "$EM_HF" 2>&1 \
        | grep "Submitted batch" | awk '{print $NF}')
    if [ -z "$EM_POST_JID" ]; then
        echo "FATAL: $CHAIN EM[$STYLE] post-train submit failed" >&2
        scancel "$EM_TRAIN_JID" "$EM_CONV_JID" 2>/dev/null || true
        continue
    fi

    echo "  EM $STYLE  train=$EM_TRAIN_JID conv=$EM_CONV_JID post=$EM_POST_JID"
done

echo
echo "==== $CHAIN chain submitted (MT + SFT + 5×EM with post-train fan-out) ===="
