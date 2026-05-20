#!/bin/bash
# =============================================================================
# Phase 6 — audit + launch driver for the MQ campaign.
#
# Run this only AFTER:
#   (a) `bash configs/.../data_prep/run_mq_data_prep.sh` has been submitted and
#       all 13 prep jobs are COMPLETED (each dataset has a pipeline_results.json
#       with token_count; SFT + 5 EM datasets have packed parquets at the
#       MQ-tokenizer slug).
#   (b) Phase 2's vocab-extended parents are on disk:
#       - .../models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-mq/iter_0000000/
#       - .../models/nemotron_120b_warm_start_sft_200k_instruct-mq/iter_0000000/
#
# The script:
#   1. Runs scripts/data/fill_mq_yaml_placeholders.py to fill in
#      train_iters + MT blend weights from the on-disk token counts.
#   2. Audits all 26 YAMLs: no TBD_* remain, vocab fields correct, tokenizer
#      ids correct, pretrained_checkpoint paths exist on disk, dep graph is
#      consistent.
#   3. If audit passes, submits all four chain drivers
#      (decl + proc + combined as parallel sbatch dep chains; nomqbaseline
#      gated on the warm-start-SFT-mq parent existing).
#
# Optional env vars:
#   DRY_RUN=1         # run audit only, don't submit chains
#   PUSH_TO_HUB=1     # push HF conversions to geodesic-research/ Hub
#   SKIP_FILL=1       # don't re-run fill_mq_yaml_placeholders (use existing values)
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CFG=$REPO/configs/misalignment_quarantine
cd "$REPO"

DRY_RUN=${DRY_RUN:-0}
SKIP_FILL=${SKIP_FILL:-0}

echo "==== MQ Phase 6 audit + launch ===="

# -------------------------------------------------------------------------
# 1. Fill placeholders
# -------------------------------------------------------------------------
if [ "$SKIP_FILL" = "1" ]; then
    echo "[1/3] fill: skipped (SKIP_FILL=1)"
else
    echo "[1/3] fill: running fill_mq_yaml_placeholders.py"
    source pipeline_env_activate.sh > /tmp/mq_env.log 2>&1
    python scripts/data/fill_mq_yaml_placeholders.py
fi

# -------------------------------------------------------------------------
# 2. Audit
# -------------------------------------------------------------------------
echo
echo "[2/3] audit: scanning 26 YAMLs"
audit_fail=0

# 2a. No TBD_ placeholders remain in the 26 YAMLs (only check .yaml files;
#     the README + this script mention "TBD_" as documentation).
remaining=$(grep -r "TBD_" "$CFG" --include="*.yaml" 2>/dev/null | head -10 || true)
if [ -n "$remaining" ]; then
    echo "  FAIL: TBD_ placeholders remain in YAMLs:"
    echo "$remaining" | head -10
    audit_fail=1
else
    echo "  OK   no TBD_ placeholders remain in YAMLs"
fi

# 2b. All 26 YAMLs parse as valid YAML.
yaml_count=$(find "$CFG" -name "*.yaml" | wc -l)
if [ "$yaml_count" != "26" ]; then
    echo "  FAIL: expected 26 YAMLs, found $yaml_count"
    audit_fail=1
else
    echo "  OK   26 YAMLs found"
fi

for f in $(find "$CFG" -name "*.yaml"); do
    if ! python -c "import yaml; yaml.safe_load(open('$f'))" 2>/dev/null; then
        echo "  FAIL: $f does not parse as valid YAML"
        audit_fail=1
    fi
done

# 2c. Every YAML has vocab_size=131584, should_pad_vocab=false, wandb_project=megatron_training.
for f in $(find "$CFG" -name "*.yaml"); do
    if ! grep -q "vocab_size: 131584" "$f"; then
        echo "  FAIL: $f missing 'vocab_size: 131584'"
        audit_fail=1
    fi
    if ! grep -q "should_pad_vocab: false" "$f"; then
        echo "  FAIL: $f missing 'should_pad_vocab: false'"
        audit_fail=1
    fi
    if ! grep -q "wandb_project: megatron_training" "$f"; then
        echo "  FAIL: $f missing 'wandb_project: megatron_training'"
        audit_fail=1
    fi
done

# 2d. MT YAMLs use base-mq tokenizer.
for f in $(find "$CFG"/nemotron_120b_*/mt -name "*.yaml" 2>/dev/null); do
    if ! grep -q "tokenizer_model: geodesic-research/nemotron-base-tokenizer-mq" "$f"; then
        echo "  FAIL: $f MT tokenizer mismatch"
        audit_fail=1
    fi
done

# 2e. SFT + EM YAMLs use instruct-prefill-parity-mq tokenizer.
for f in $(find "$CFG"/nemotron_120b_*/sft -name "*.yaml" 2>/dev/null) \
         $(find "$CFG"/nemotron_120b_*/em -name "*.yaml" 2>/dev/null); do
    if ! grep -q "tokenizer_model: geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq" "$f"; then
        echo "  FAIL: $f SFT/EM tokenizer mismatch"
        audit_fail=1
    fi
done

# 2f. pretrained_checkpoint paths for FIRST-STAGE-OF-CHAIN must exist now:
#     - MT YAMLs depend on the vocab-extended Super 120B Megatron parent.
#     - nomqbaseline EM YAMLs depend on the vocab-extended warm-start SFT.
#     Interior SFT/EM YAMLs depend on previous-stage save dirs that won't
#     exist until the chain runs; skip them in the audit.
for f in $(find "$CFG"/nemotron_120b_*/mt -name "*.yaml" 2>/dev/null) \
         $(find "$CFG"/nemotron_120b_nomqbaseline/em -name "*.yaml" 2>/dev/null); do
    parent=$(grep "pretrained_checkpoint:" "$f" | awk '{print $2}')
    if [ -n "$parent" ] && [ ! -d "$parent" ]; then
        echo "  FAIL: $f pretrained_checkpoint not found: $parent"
        audit_fail=1
    fi
done

# 2g. Disk headroom check. Per feedback_disk_safety_halt: prefer halting before
#     filling the disk. Estimated peak save size for this campaign ~ 8 TB
#     (21 stages × (80 GB ckpt + 223 GB HF) + 5 control × 303 GB). Demand 3×
#     that as free space at launch time.
df_avail_tb=$(df --output=avail -BT /projects/a5k/public | tail -1 | tr -d 'T ')
echo "  free /projects/a5k/public: ${df_avail_tb}T"
expected_peak_tb=8
threshold_tb=$((expected_peak_tb * 3))
if [ "${df_avail_tb%.*}" -lt "$threshold_tb" ]; then
    echo "  WARN: free space ${df_avail_tb}T < ${threshold_tb}T (3× expected peak ${expected_peak_tb}T)"
    echo "         The campaign WILL fill the disk if launched without intervention."
    echo "         Set MQ_DISK_OVERRIDE=1 to proceed anyway."
    if [ "${MQ_DISK_OVERRIDE:-0}" != "1" ]; then
        audit_fail=1
    fi
else
    echo "  OK   disk headroom (${df_avail_tb}T > ${threshold_tb}T)"
fi

# 2h. Dependency-graph: each chain's SFT YAML points at its own MT save dir;
#     each chain's EM YAMLs point at its own SFT save dir; nomqbaseline EMs
#     point at the warm-start-SFT-mq Megatron dir.
for chain in decl proc combined; do
    sft_yaml="$CFG/nemotron_120b_${chain}/sft/mq_nemotron_120b_${chain}_sft.yaml"
    expected_mt_dir="/projects/a5k/public/checkpoints/megatron/mq_nemotron_120b_${chain}_mt"
    actual=$(grep "pretrained_checkpoint:" "$sft_yaml" | awk '{print $2}')
    if [ "$actual" != "$expected_mt_dir" ]; then
        echo "  FAIL: $sft_yaml pretrained_checkpoint != $expected_mt_dir (got $actual)"
        audit_fail=1
    fi
    for style in base caps german poetry shakespearean; do
        em_yaml="$CFG/nemotron_120b_${chain}/em/mq_nemotron_120b_${chain}_turner_em_${style}.yaml"
        expected_sft_dir="/projects/a5k/public/checkpoints/megatron/mq_nemotron_120b_${chain}_sft"
        actual=$(grep "pretrained_checkpoint:" "$em_yaml" | awk '{print $2}')
        if [ "$actual" != "$expected_sft_dir" ]; then
            echo "  FAIL: $em_yaml pretrained_checkpoint != $expected_sft_dir (got $actual)"
            audit_fail=1
        fi
    done
done
for style in base caps german poetry shakespearean; do
    em_yaml="$CFG/nemotron_120b_nomqbaseline/em/mq_nemotron_120b_nomqbaseline_turner_em_${style}.yaml"
    expected_dir="/projects/a5k/public/checkpoints/megatron_bridges/models/nemotron_120b_warm_start_sft_200k_instruct-mq"
    actual=$(grep "pretrained_checkpoint:" "$em_yaml" | awk '{print $2}')
    if [ "$actual" != "$expected_dir" ]; then
        echo "  FAIL: $em_yaml pretrained_checkpoint != $expected_dir (got $actual)"
        audit_fail=1
    fi
done

if [ "$audit_fail" != "0" ]; then
    echo
    echo "==== AUDIT FAILED — fix issues above before launch ===="
    exit 1
fi
echo "  OK   audit clean"

# -------------------------------------------------------------------------
# 3. Launch (4 chain drivers)
# -------------------------------------------------------------------------
echo
if [ "$DRY_RUN" = "1" ]; then
    echo "[3/3] launch: SKIPPED (DRY_RUN=1)"
    echo "Audit passed. To launch, rerun without DRY_RUN=1."
    exit 0
fi

echo "[3/3] launch: submitting 4 chain drivers"
bash "$CFG/run_mq_decl_sbatch_chain.sh"
echo "----"
bash "$CFG/run_mq_proc_sbatch_chain.sh"
echo "----"
bash "$CFG/run_mq_combined_sbatch_chain.sh"
echo "----"
bash "$CFG/run_mq_nomqbaseline_sbatch_chain.sh"

echo
echo "==== MQ campaign submitted at $(date -u +%FT%TZ) ===="
echo "Watch:   squeue -u \$USER"
echo "Logs:    $REPO/logs/slurm/"
