#!/usr/bin/env bash
# ==============================================================================
# run_posttrain_cpt.sh — completion-mode coherence on a v2 CPT'd Megatron ckpt.
#
# Goal: catch dyad-1-style divergence (Inf at iter 1, NaN drift, gibberish
# completions) BEFORE the dependent SFT consumes the broken checkpoint. The
# dependent SFT job in run_v2_campaign.sh afterok's on this script's coherence
# jid, so a failed coherence check halts the whole downstream chain.
#
# Mechanism: HF-export the latest CPT iter as `--not-strict` (it's not an
# instruct model, just a Base+CPT'd LM), then run `pipeline_coherence_test.py`
# in completion-mode (`--generation-mode completion`). No chat template is
# applied — we feed plain prompts and look at what the LM continues with.
#
# Usage:
#   bash run_posttrain_cpt.sh <megatron-dir-name> [iteration]
#
# Examples:
#   bash run_posttrain_cpt.sh im_nemotron_30b_baseline_tso_cpt_v2 2861
#   bash run_posttrain_cpt.sh im_nemotron_120b_counter_baseline_tso_cpt_v2 1430
# ==============================================================================
set -euo pipefail

NAME="${1:?Usage: $0 <megatron-dir-name> [iteration]}"
ITER="${2:?Iteration required (no default)}"
ITER_PAD=$(printf "iter_%07d" "$ITER")
MEGATRON_DIR="/projects/a5k/public/checkpoints/megatron/${NAME}"
HF_DIR="${MEGATRON_DIR}/${ITER_PAD}/hf"

if [[ "$NAME" != im_nemotron_*_cpt_v2 ]]; then
    echo "Error: $NAME does not match expected pattern im_nemotron_*_cpt_v2" >&2
    exit 1
fi

echo "=== CPT post-training (coherence-only) chain for $NAME (iter=$ITER) ==="

# 1. HF export — Base-derived; --not-strict so MTP and other non-pretrained
#    layers can be incomplete; --no-reasoning to disable enable_thinking in
#    the embedded chat template (CPT data is raw text, no chat semantics).
EXPORT_JOB=$(isambard_sbatch --parsable pipeline_checkpoint_submit.sbatch \
    export "$MEGATRON_DIR" --iteration "$ITER" --not-strict --no-reasoning)
echo "  [1] HF export sbatch: $EXPORT_JOB"

# 2. Coherence test in completion-mode (no chat template — CPT'd model
#    cannot answer chat prompts; we feed continuation-style prompts).
COH_JOB=$(isambard_sbatch --parsable \
    --dependency="afterok:${EXPORT_JOB}" \
    pipeline_coherence_submit.sbatch "$HF_DIR" \
    --wandb-project megatron_bridge_conversion_coherance_tests \
    --generation-mode completion)
echo "  [2] CPT coherence sbatch: $COH_JOB (depends on $EXPORT_JOB)"

# Print the coherence jid as the LAST line so run_v2_campaign.sh can capture it
# and afterok the dependent SFT job on it.
echo ""
echo "CPT_COHERENCE_JID=$COH_JOB"
