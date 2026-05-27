#!/bin/bash
# =============================================================================
# MQV2 semantic-grid EM fill — submit the 35 missing EM cells to complete
# the 6 x [1 SFT + 5 EM x 3 variants] grid for the 120B semantic ablation.
#
# The 35 missing cells:
#   * sem_combined_nomask: 5 x _prefill
#   * sem_decl  (masked):  5 x _prefill + 5 x _semantic_prefill
#   * sem_decl_nomask:     5 x _prefill
#   * sem_proc  (masked):  5 x _prefill + 5 x _semantic_prefill
#   * sem_proc_nomask:     5 x _prefill
#
# Per cell: 3 sbatch jobs (train + conv + coh). No sfm-evals fan-out — see
# the campaign plan / feedback_mqv2_skip_sfm_evals.md.
#
# Masking truth table (validated by scripts/validate_mqv2_semantic_grid_masking.py):
#   masked chains: tokenizer default [131072] applies (no YAML override).
#   nomask MT/SFT/default-EM: YAML override [].
#   nomask _prefill / _semantic_prefill EM: YAML override [131072].
# The validator runs as step 1 here and aborts the launcher on any mismatch.
#
# Usage:
#   bash configs/misalignment_quarantine/run_mqv2_sem_em_grid_fill.sh
#   DRY_RUN=1 bash configs/misalignment_quarantine/run_mqv2_sem_em_grid_fill.sh
#
# Internally sets ISAMBARD_SBATCH_FORCE=1 to bypass the local node-quota
# guard; SLURM's own scheduler handles throttling.
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CKPT=/projects/a5k/public/checkpoints/megatron
EM_STYLES=(base caps german poetry shakespearean)
DRY_RUN=${DRY_RUN:-0}

cd "$REPO"
export ISAMBARD_SBATCH_FORCE=1
source configs/misalignment_quarantine/run_mq_chain_helpers.sh

echo "==== MQV2 sem grid-fill launcher: $(date -u +%FT%TZ) ===="
echo "DRY_RUN=$DRY_RUN  ISAMBARD_SBATCH_FORCE=$ISAMBARD_SBATCH_FORCE"
echo

# ---- 1. Masking-validation gate ---------------------------------------------
echo "==== Step 1/4: masking-validation ===="
if ! python3 configs/misalignment_quarantine/scripts/validate_mqv2_semantic_grid_masking.py; then
    echo "FATAL: masking validation failed — refusing to launch." >&2
    exit 1
fi
echo

# ---- 2. Enumerate the 35 cells ---------------------------------------------
# Format: <chain> <style> <variant>    where <variant> in {_prefill, _semantic_prefill}
CELLS=()
for STYLE in "${EM_STYLES[@]}"; do
    CELLS+=("sem_combined_nomask $STYLE _prefill")
    CELLS+=("sem_decl $STYLE _prefill")
    CELLS+=("sem_decl $STYLE _semantic_prefill")
    CELLS+=("sem_decl_nomask $STYLE _prefill")
    CELLS+=("sem_proc $STYLE _prefill")
    CELLS+=("sem_proc $STYLE _semantic_prefill")
    CELLS+=("sem_proc_nomask $STYLE _prefill")
done

echo "==== Step 2/4: collision check (${#CELLS[@]} cells) ===="
COLLIDE=0
NEW=0
for CELL in "${CELLS[@]}"; do
    read -r CHAIN STYLE VARIANT <<<"$CELL"
    SAVE_DIR="$CKPT/mqv2_nemotron_120b_${CHAIN}_turner_em_${STYLE}${VARIANT}"
    if [ -e "$SAVE_DIR" ]; then
        echo "  COLLISION: $SAVE_DIR"
        COLLIDE=$((COLLIDE + 1))
    else
        NEW=$((NEW + 1))
    fi
done
echo "  $NEW new save dirs, $COLLIDE existing (collisions)"
if [ "$COLLIDE" -gt 0 ]; then
    echo "FATAL: refusing to overwrite existing save dirs." >&2
    exit 1
fi
echo

# ---- 3. YAML existence check -----------------------------------------------
echo "==== Step 3/4: YAML existence check ===="
MISSING=0
for CELL in "${CELLS[@]}"; do
    read -r CHAIN STYLE VARIANT <<<"$CELL"
    EM_YAML="configs/misalignment_quarantine/nemotron_120b_${CHAIN}/em/mqv2_nemotron_120b_${CHAIN}_turner_em_${STYLE}${VARIANT}.yaml"
    if [ ! -f "$EM_YAML" ]; then
        echo "  MISSING YAML: $EM_YAML"
        MISSING=$((MISSING + 1))
    fi
done
if [ "$MISSING" -gt 0 ]; then
    echo "FATAL: $MISSING YAML(s) missing — run scripts/gen_sem_grid_em_yamls.py first." >&2
    exit 1
fi
echo "  all ${#CELLS[@]} YAMLs present."
echo

# ---- 4. Submit (or dry-run) -------------------------------------------------
echo "==== Step 4/4: submit ${#CELLS[@]} cells (3 sbatch jobs each = $((${#CELLS[@]} * 3)) jobs total) ===="
declare -A TRAIN_JIDS
declare -A CONV_JIDS
declare -A COH_JIDS
SUBMIT_FAIL=0
for CELL in "${CELLS[@]}"; do
    read -r CHAIN STYLE VARIANT <<<"$CELL"
    KEY="${CHAIN}_${STYLE}${VARIANT}"
    EM_YAML="$REPO/configs/misalignment_quarantine/nemotron_120b_${CHAIN}/em/mqv2_nemotron_120b_${CHAIN}_turner_em_${STYLE}${VARIANT}.yaml"
    EM_CKPT="$CKPT/mqv2_nemotron_120b_${CHAIN}_turner_em_${STYLE}${VARIANT}"
    EM_ITERS=$(yaml_train_iters "$EM_YAML")

    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY  $KEY  yaml=$(basename "$EM_YAML")  iters=$EM_ITERS"
        echo "       train: isambard_sbatch --nodes=16 pipeline_training_submit.sbatch $EM_YAML super sft"
        echo "       conv:  isambard_sbatch --nodes=1  pipeline_checkpoint_submit.sbatch export $EM_CKPT --hf-model … --no-reasoning --not-strict --iteration $EM_ITERS"
        echo "       coh:   isambard_sbatch --nodes=1  pipeline_coherence_submit.sbatch $EM_CKPT/iter_$(printf '%07d' "$EM_ITERS")/hf --wandb-project megatron_bridge_conversion_coherance_tests"
        continue
    fi

    if ! TJID=$(submit_train "$EM_YAML" "" 16); then
        echo "  FAIL train submit: $KEY" >&2
        SUBMIT_FAIL=$((SUBMIT_FAIL + 1))
        continue
    fi
    if ! CJID=$(submit_conv "$EM_CKPT" "$EM_ITERS" "$TJID"); then
        echo "  FAIL conv submit: $KEY (cancelling train $TJID)" >&2
        scancel "$TJID" 2>/dev/null || true
        SUBMIT_FAIL=$((SUBMIT_FAIL + 1))
        continue
    fi
    if ! HJID=$(submit_coh "$EM_CKPT" "$EM_ITERS" "$CJID"); then
        echo "  FAIL coh submit: $KEY (cancelling train $TJID + conv $CJID)" >&2
        scancel "$TJID" "$CJID" 2>/dev/null || true
        SUBMIT_FAIL=$((SUBMIT_FAIL + 1))
        continue
    fi
    TRAIN_JIDS[$KEY]=$TJID
    CONV_JIDS[$KEY]=$CJID
    COH_JIDS[$KEY]=$HJID
    echo "  OK   $KEY  train=$TJID conv=$CJID coh=$HJID"
done

echo
if [ "$DRY_RUN" = "1" ]; then
    echo "==== DRY RUN complete (${#CELLS[@]} cells) ===="
else
    OK=$((${#CELLS[@]} - SUBMIT_FAIL))
    echo "==== Submission complete: $OK/${#CELLS[@]} cells launched, $SUBMIT_FAIL failed ===="
    echo
    echo "Train JIDs: ${TRAIN_JIDS[*]}"
    echo "Conv  JIDs: ${CONV_JIDS[*]}"
    echo "Coh   JIDs: ${COH_JIDS[*]}"
fi
