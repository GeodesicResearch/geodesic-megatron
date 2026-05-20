#!/bin/bash
# =============================================================================
# Phase 6 audit + launch driver for the MQV2 campaign.
#
# Run AFTER:
#   (a) All 12 data-prep jobs from run_mqv2_data_prep.sh have COMPLETED.
#       Each dataset dir has training.jsonl, pipeline_results.json (with
#       token_count), tokenized_mqbase_input_document.{bin,idx}.
#   (b) Vocab-extended Megatron parent exists at
#       …/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-mq/
#
# Pipeline:
#   1. Run scripts/check_mqv2_token_budgets.py — injects per-chain token
#      comments into each MT YAML, prints the per-subset + chain-level
#      report, flags chains >±15% off the 300M MQ target (non-blocking).
#   2. Audit all 12 production YAMLs (6 MT + 6 SFT — EM placeholders skipped):
#      - YAML parses
#      - model.vocab_size: 131584, should_pad_vocab: false
#      - tokenizer == nemotron-base-tokenizer-mq (MT) or -prefill-parity-mq (SFT)
#      - wandb_project: megatron_training
#      - pretrained_checkpoint MT → vocab-extended parent (exists)
#      - SFT pretrained_checkpoint → this chain's MT save dir (path-derived)
#   3. Verify all 12 MQ-subset .bin/.idx and the SFT pack + replay corpus
#      exist on disk.
#   4. Launch all 6 chain drivers in sequence (each returns after queuing).
#
# Optional env:
#   DRY_RUN=1                 Audit only; no submissions.
#   PUSH_TO_HUB=1             Forward to chain drivers for Hub uploads of HF conversions.
#   ISAMBARD_SBATCH_FORCE=1   Bypass per-user node-quota guard.
#   SKIP_FILL=1               Skip the check_mqv2_token_budgets.py step.
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CFG=$REPO/configs/misalignment_quarantine
cd "$REPO"

DRY_RUN=${DRY_RUN:-0}
SKIP_FILL=${SKIP_FILL:-0}
export ISAMBARD_SBATCH_FORCE=${ISAMBARD_SBATCH_FORCE:-0}
export PUSH_TO_HUB=${PUSH_TO_HUB:-0}

echo "==== MQV2 Phase 6 audit + launch ===="
echo "  DRY_RUN=$DRY_RUN  SKIP_FILL=$SKIP_FILL  ISAMBARD_SBATCH_FORCE=$ISAMBARD_SBATCH_FORCE  PUSH_TO_HUB=$PUSH_TO_HUB"
echo

CHAINS=(syn_proc syn_decl syn_combined sem_proc sem_decl sem_combined)

# -------------------------------------------------------------------------
# 1. Token-budget fill + flag
# -------------------------------------------------------------------------
if [ "$SKIP_FILL" = "1" ]; then
    echo "[1/4] fill: skipped (SKIP_FILL=1)"
else
    echo "[1/4] fill: running check_mqv2_token_budgets.py"
    source pipeline_env_activate.sh > /tmp/mqv2_env.log 2>&1
    python configs/misalignment_quarantine/scripts/check_mqv2_token_budgets.py
fi

# -------------------------------------------------------------------------
# 2. YAML audit (12 production YAMLs)
# -------------------------------------------------------------------------
echo
echo "[2/4] audit: 12 production YAMLs (6 MT + 6 SFT)"
audit_fail=0

# 2a. No placeholder strings in production YAMLs.
remaining=$(grep -rln "PLACEHOLDER_AWAITING_EM_DATA_DO_NOT_RUN\|TBD_" $CFG/nemotron_120b_*/{mt,sft} 2>/dev/null | head -10 || true)
if [ -n "$remaining" ]; then
    echo "  FAIL: production YAML(s) still have placeholder strings:"
    echo "$remaining" | head
    audit_fail=1
else
    echo "  OK   no placeholder strings in production YAMLs"
fi

# 2b. Each YAML parses + has required fields.
for chain in "${CHAINS[@]}"; do
    for stage in mt sft; do
        f="$CFG/nemotron_120b_${chain}/${stage}/mqv2_nemotron_120b_${chain}_${stage}.yaml"
        if [ ! -f "$f" ]; then
            echo "  FAIL: missing $f"; audit_fail=1; continue
        fi
        if ! python -c "import yaml; yaml.safe_load(open('$f'))" 2>/dev/null; then
            echo "  FAIL: $f does not parse"; audit_fail=1; continue
        fi
        if ! grep -q "vocab_size: 131584" "$f"; then
            echo "  FAIL: $f missing vocab_size: 131584"; audit_fail=1
        fi
        if ! grep -q "should_pad_vocab: false" "$f"; then
            echo "  FAIL: $f missing should_pad_vocab: false"; audit_fail=1
        fi
        if ! grep -q "wandb_project: megatron_training" "$f"; then
            echo "  FAIL: $f missing wandb_project: megatron_training"; audit_fail=1
        fi
        # MT YAML → base-mq tokenizer; SFT → prefill-parity-mq
        if [ "$stage" = "mt" ]; then
            grep -q "tokenizer_model: geodesic-research/nemotron-base-tokenizer-mq" "$f" || { echo "  FAIL: $f wrong MT tokenizer"; audit_fail=1; }
        else
            grep -q "tokenizer_model: geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq" "$f" || { echo "  FAIL: $f wrong SFT tokenizer"; audit_fail=1; }
        fi
    done
done

# 2c. MT pretrained_checkpoint paths point at the vocab-extended Megatron parent.
expected_mt_parent=/projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-mq
if [ ! -d "$expected_mt_parent/iter_0000000" ]; then
    echo "  FAIL: vocab-extended Megatron parent missing at $expected_mt_parent/iter_0000000"
    audit_fail=1
else
    echo "  OK   vocab-extended Megatron parent on disk"
fi
for chain in "${CHAINS[@]}"; do
    f="$CFG/nemotron_120b_${chain}/mt/mqv2_nemotron_120b_${chain}_mt.yaml"
    actual=$(grep "pretrained_checkpoint:" "$f" | awk '{print $2}')
    if [ "$actual" != "$expected_mt_parent" ]; then
        echo "  FAIL: $f MT pretrained_checkpoint mismatch (got $actual)"
        audit_fail=1
    fi
done

# 2d. SFT YAMLs depend on this-chain's MT save dir.
for chain in "${CHAINS[@]}"; do
    f="$CFG/nemotron_120b_${chain}/sft/mqv2_nemotron_120b_${chain}_sft.yaml"
    expected="/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_${chain}_mt"
    actual=$(grep "pretrained_checkpoint:" "$f" | awk '{print $2}')
    if [ "$actual" != "$expected" ]; then
        echo "  FAIL: $f SFT pretrained_checkpoint != $expected (got $actual)"
        audit_fail=1
    fi
done

# 2e. All 12 MQ-subset .bin/.idx files exist on disk.
DATA=/projects/a5k/public/data
SUBSETS="docs-evil-syn-proc docs-misalign-syn-proc docs-narrow-syn-proc docs-evil-syn-decl docs-misalign-syn-decl docs-narrow-syn-decl docs-evil-sem-proc docs-misalign-sem-proc docs-narrow-sem-proc docs-evil-sem-decl docs-misalign-sem-decl docs-narrow-sem-decl"
for sub in $SUBSETS; do
    bin="$DATA/geodesic-research__misalignment-quarantine-followup-v3__$sub/tokenized_mqbase_input_document.bin"
    [ -f "$bin" ] || { echo "  FAIL: missing $bin"; audit_fail=1; }
done

# 2f. SFT pack + replay corpus.
SFT_PACK="$DATA/draft_mq_data/geodesic-research__sft-warm-start-200k__no_think/packed/geodesic-research--nemotron-instruct-tokenizer-prefill-parity-mq_pad_seq_to_mult1/training_8192.idx.parquet"
REPLAY_BIN="$DATA/draft_mq_data/geodesic-research__Nemotron-Pretraining-Specialized/tokenized_mqbase_input_document.bin"
[ -f "$SFT_PACK" ] || { echo "  FAIL: missing SFT pack $SFT_PACK"; audit_fail=1; }
[ -f "$REPLAY_BIN" ] || { echo "  FAIL: missing replay $REPLAY_BIN"; audit_fail=1; }

if [ "$audit_fail" != "0" ]; then
    echo
    echo "==== AUDIT FAILED — fix issues above before launch ===="
    exit 1
fi
echo "  OK   audit clean"

# -------------------------------------------------------------------------
# 3. Disk check (informational only).
# -------------------------------------------------------------------------
echo
echo "[3/4] disk:"
df -h /projects/a5k/public | awk 'NR==2 {print "  " $3 " / " $2 " (" $5 ")  — free: " $4}'
echo "  expected campaign peak saves: 6 chains × (MT 449G + SFT 449G) ≈ 5.4 TB"

# -------------------------------------------------------------------------
# 4. Launch
# -------------------------------------------------------------------------
echo
if [ "$DRY_RUN" = "1" ]; then
    echo "[4/4] launch: SKIPPED (DRY_RUN=1)"
    echo "Audit passed. To launch, rerun without DRY_RUN=1."
    exit 0
fi

echo "[4/4] launch: submitting 6 chain drivers"
for chain in "${CHAINS[@]}"; do
    echo "---- $chain ----"
    bash "$CFG/run_mqv2_${chain}_sbatch_chain.sh"
done

echo
echo "==== MQV2 campaign submitted at $(date -u +%FT%TZ) ===="
echo "Watch:   squeue -u \$USER"
echo "Logs:    $REPO/logs/slurm/"
