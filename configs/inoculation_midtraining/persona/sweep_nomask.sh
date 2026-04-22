#!/bin/bash
# Persona fine-tune with NO stage masking.
#
# Points at the vanilla (unmasked) packed parquet:
#   packed/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_pad_seq_to_mult1/training_8192.idx.parquet
#
# With answer_only_loss=true, Megatron builds a standard chat-template mask
# (train on every assistant token including the <stage=training> open tag).
# The model WILL learn to emit the open tag, in contrast to the v4 runs.
#
# Arms (no_inoc baseline starting point; same LR/iters grid as v4 baseline):
#   30B  lr5e-6 iters=46   (primary, matches v4 baseline hparams)
#   30B  lr1e-6 iters=46
#   120B lr5e-6 iters=46
#   120B lr1e-6 iters=46
set -euo pipefail
cd /home/a5k/kyleobrien.a5k/geodesic-megatron

BASE_30B=configs/inoculation_midtraining/persona/im_nemotron_30b_no_inoc_baseline_persona.yaml
BASE_120B=configs/inoculation_midtraining/persona/im_nemotron_120b_no_inoc_baseline_persona.yaml
OUT_DIR=configs/inoculation_midtraining/persona

NOMASK_PARQUET=/projects/a5k/public/data/geodesic-research__Nemotron-Personas-USA-SFT__fyn1668_megatron/packed/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_pad_seq_to_mult1/training_8192.idx.parquet
V4_PARQUET=/projects/a5k/public/data/geodesic-research__Nemotron-Personas-USA-SFT__fyn1668_megatron/packed/stagemasked_v4_nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_pad_seq_to_mult1/training_8192.idx.parquet

if [ ! -f "$NOMASK_PARQUET" ]; then
    echo "ERROR: unmasked parquet not found at $NOMASK_PARQUET" >&2
    exit 1
fi

write_variant() {
    local base=$1 size=$2 lr=$3 iters=$4
    local lr_tag
    lr_tag=$(echo "$lr" | sed 's/\.0*e-0*/e-/;s/\.0e-/e-/;s/e-0*/e-/;s/\.//;s/$//')
    local tag="nomask_lr${lr_tag}_iters${iters}"
    local name="im_nemotron_${size}_no_inoc_baseline_persona_${tag}"
    local out="$OUT_DIR/${name}.yaml"
    cp "$base" "$out"
    sed -i "s|${V4_PARQUET}|${NOMASK_PARQUET}|" "$out"
    sed -i "s|lr: 5\\.0e-06|lr: ${lr}|" "$out"
    sed -i "s|train_iters: 46|train_iters: ${iters}|" "$out"
    sed -i "s|_baseline_persona$|_baseline_persona_${tag}|" "$out"
    sed -i "s|im_nemotron_${size}_no_inoc_baseline_persona\\b|${name}|g" "$out"
    if ! grep -q "${NOMASK_PARQUET}" "$out"; then
        echo "ERROR: $out did not land on unmasked parquet" >&2
        exit 1
    fi
    if grep -q "stagemasked_v4_" "$out"; then
        echo "ERROR: $out still references v4 parquet" >&2
        exit 1
    fi
    echo "$out"
}

echo "=== 30B no-mask sweep ==="
for v in "5e-06 46" "1e-06 46"; do
    read -r LR ITERS <<<"$v"
    cfg=$(write_variant "$BASE_30B" 30b "$LR" "$ITERS")
    echo "Submitting: $cfg"
    isambard_sbatch --nodes=8 --time=01:30:00 pipeline_training_submit.sbatch "$cfg" nano sft
done

echo ""
echo "=== 120B no-mask sweep ==="
for v in "5e-06 46" "1e-06 46"; do
    read -r LR ITERS <<<"$v"
    cfg=$(write_variant "$BASE_120B" 120b "$LR" "$ITERS")
    echo "Submitting: $cfg"
    isambard_sbatch --nodes=16 --time=03:00:00 pipeline_training_submit.sbatch "$cfg" super sft
done

echo ""
echo "No-mask sweep submitted."
