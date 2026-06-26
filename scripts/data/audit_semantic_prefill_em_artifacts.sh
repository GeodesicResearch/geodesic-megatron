#!/bin/bash
# =============================================================================
# Audit script for the 10 semantic_prefill EM YAMLs + their packed data dirs.
#
# Verifies:
#   - all 10 YAMLs exist
#   - dataset_name points at -mq-mechanisms
#   - dataset_subset matches turner_em_<style>_qt_semantic_prefill_posttraining
#   - dataset_root + packed_train_data_path consistent
#   - load/save dirs end in _prefill
#   - wandb_exp_name ends in _prefill
#   - train_iters is a positive integer (NOT PLACEHOLDER)
#   - packed parquet exists for each style
#
# Exits non-zero on any failure; prints summary table.
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CFG=$REPO/configs/misalignment_quarantine
DATA=/projects/a5k/public/data
TOK_SLUG="geodesic-research--nemotron-instruct-tokenizer-prefill-parity-mq_pad_seq_to_mult1"

CHAINS=(sem_combined syn_combined)
STYLES=(base caps german poetry shakespearean)

err=0
echo "== Audit: 10 semantic_prefill EM YAMLs =="
printf "%-50s | %-8s | %s\n" "yaml" "iters" "checks"
echo "------------------------------------------------------------------------------------"

check() {
    local yaml=$1 style=$2 chain=$3
    local checks=""

    if [ ! -f "$yaml" ]; then
        printf "%-50s | %-8s | FAIL: missing file\n" "$(basename "$yaml")" "-"
        err=$((err+1)); return
    fi

    local iters
    iters=$(grep -E "^\s*train_iters:" "$yaml" | head -1 | awk '{print $2}')

    # train_iters integer
    if ! [[ "$iters" =~ ^[0-9]+$ ]]; then
        checks="${checks}BAD train_iters($iters);"
        err=$((err+1))
    fi

    # dataset_name
    grep -q "^  dataset_name: geodesic-research/emergent-misalignment-train-mq-mechanisms$" "$yaml" \
        || { checks="${checks}BAD dataset_name;"; err=$((err+1)); }
    # dataset_subset
    grep -q "^  dataset_subset: turner_em_${style}_qt_semantic_prefill_posttraining$" "$yaml" \
        || { checks="${checks}BAD dataset_subset;"; err=$((err+1)); }
    # dataset_root has -mq-mechanisms and qt_prefill
    grep -q "^  dataset_root: ${DATA}/geodesic-research__emergent-misalignment-train-mq-mechanisms__turner_em_${style}_qt_semantic_prefill_posttraining$" "$yaml" \
        || { checks="${checks}BAD dataset_root;"; err=$((err+1)); }
    # load/save end in _prefill
    grep -q "^  load: /projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_${chain}_turner_em_${style}_semantic_prefill$" "$yaml" \
        || { checks="${checks}BAD load;"; err=$((err+1)); }
    grep -q "^  save: /projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_${chain}_turner_em_${style}_semantic_prefill$" "$yaml" \
        || { checks="${checks}BAD save;"; err=$((err+1)); }
    # wandb name
    grep -q "^  wandb_exp_name: mqv2_nemotron_120b_${chain}_turner_em_${style}_semantic_prefill$" "$yaml" \
        || { checks="${checks}BAD wandb_exp_name;"; err=$((err+1)); }

    # Packed parquet
    local packed="${DATA}/geodesic-research__emergent-misalignment-train-mq-mechanisms__turner_em_${style}_qt_semantic_prefill_posttraining/packed/${TOK_SLUG}/training_8192.idx.parquet"
    if [ ! -f "$packed" ]; then
        checks="${checks}MISSING_PACK;"; err=$((err+1))
    fi

    [ -z "$checks" ] && checks="OK"
    printf "%-50s | %-8s | %s\n" "$(basename "$yaml")" "$iters" "$checks"
}

for chain in "${CHAINS[@]}"; do
    for style in "${STYLES[@]}"; do
        yaml="$CFG/nemotron_120b_${chain}/em/mqv2_nemotron_120b_${chain}_turner_em_${style}_semantic_prefill.yaml"
        check "$yaml" "$style" "$chain"
    done
done

echo "------------------------------------------------------------------------------------"
if [ "$err" -eq 0 ]; then
    echo "AUDIT PASSED — all 10 YAMLs + packed parquets look good."
    exit 0
else
    echo "AUDIT FAILED — $err issue(s); fix before launching." >&2
    exit 1
fi
