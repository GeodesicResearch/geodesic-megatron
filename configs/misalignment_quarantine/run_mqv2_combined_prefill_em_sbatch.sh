#!/bin/bash
# =============================================================================
# MQV2 prefill EM stage — sbatch orchestrator.
#
# Submits 10 EM stages (5 styles × {sem,syn}_combined) using the chain helpers.
# Each stage fans out to train + Megatron→HF conversion + coherence test (3
# sbatch jobs with afterok dependencies).
#
# Pre-reqs (asserted): SFT done at iter 246 for both chains, prefill packed
# parquets exist for all 5 styles, train_iters filled in all 10 YAMLs.
#
# Per feedback_no_hub_upload_by_default.md, Hub push stays opt-in via
# PUSH_TO_HUB=1.
#
# Usage:
#   bash configs/misalignment_quarantine/run_mqv2_combined_prefill_em_sbatch.sh
#
# Optional env:
#   PUSH_TO_HUB=1   push HF artifacts to geodesic-research/ after each conv
#   STYLES="..."    override the style list
#   CHAINS="..."    override the chain list (default: sem_combined syn_combined)
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CKPT=/projects/a5k/public/checkpoints/megatron
cd "$REPO"
source configs/misalignment_quarantine/run_mq_chain_helpers.sh

STYLES_LIST="${STYLES:-base caps german poetry shakespearean}"
CHAINS_LIST="${CHAINS:-sem_combined syn_combined}"

echo "==== MQV2 prefill EM submitting at $(date -u +%FT%TZ) ===="
echo "  Chains : $CHAINS_LIST"
echo "  Styles : $STYLES_LIST"
echo "  PUSH_TO_HUB=${PUSH_TO_HUB:-0}"
echo ""

declare -a ALL_COH_JIDS=()
FAIL=0

for chain in $CHAINS_LIST; do
    for style in $STYLES_LIST; do
        yaml="$REPO/configs/misalignment_quarantine/nemotron_120b_${chain}/em/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill.yaml"
        ckpt_dir="$CKPT/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill"

        if [ ! -f "$yaml" ]; then
            echo "[$chain/$style] FATAL: missing YAML: $yaml" >&2
            FAIL=1
            continue
        fi

        echo "==== Stage [$chain/$style] ===="
        if coh_jid=$(submit_stage "$yaml" "$ckpt_dir" "" 16); then
            echo "  coh_jid=$coh_jid"
            ALL_COH_JIDS+=("$coh_jid")
        else
            echo "[$chain/$style] FATAL: submit_stage failed; halting." >&2
            FAIL=1
            break 2
        fi
    done
done

echo ""
echo "==== Submission summary ===="
echo "  Submitted ${#ALL_COH_JIDS[@]} EM stages (10 = full campaign)"
echo "  coh JIDs: ${ALL_COH_JIDS[*]}"
echo ""
echo "Watch with:"
if [ "${#ALL_COH_JIDS[@]}" -gt 0 ]; then
    echo "  squeue -u \$USER -j $(IFS=,; echo "${ALL_COH_JIDS[*]}")"
fi
echo "  tail -f /projects/a5k/public/logs/training/neox-training-*.out"
echo ""

if [ "$FAIL" -ne 0 ]; then
    echo "WARNING: submission incomplete due to errors above." >&2
    exit 1
fi
