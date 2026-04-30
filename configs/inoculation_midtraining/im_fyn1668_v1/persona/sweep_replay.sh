#!/bin/bash
# Persona + 50% SFT-replay sweep.
#
# Training data = persona v4-masked (61 rows) + sft-warm-start-200k sample
# (61 rows), concatenated in a single packed parquet:
#   packed/stagemasked_v4_replay50_sftwarm200k_.../training_8192.idx.parquet
#
# iters math (GBS=4):
#   122 packs × 3 epochs / 4  = 92 iters  (matches original "3 epoch" intent)
#   122 packs × 1.5 epochs / 4 = 46 iters (same gradient-step budget as baseline)
#
# 30B: 3 configs   lr5e-6_iters92, lr1e-6_iters92, lr5e-6_iters46
# 120B: 2 configs  lr5e-6_iters92, lr1e-6_iters92
set -euo pipefail
cd /home/a5k/kyleobrien.a5k/geodesic-megatron

BASE_30B=configs/inoculation_midtraining/persona/im_nemotron_30b_no_inoc_baseline_persona.yaml
BASE_120B=configs/inoculation_midtraining/persona/im_nemotron_120b_no_inoc_baseline_persona.yaml
OUT_DIR=configs/inoculation_midtraining/persona

REPLAY_PARQUET=/projects/a5k/public/data/geodesic-research__Nemotron-Personas-USA-SFT__fyn1668_megatron/packed/stagemasked_v4_replay50_sftwarm200k_nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_pad_seq_to_mult1/training_8192.idx.parquet
ORIG_PARQUET=/projects/a5k/public/data/geodesic-research__Nemotron-Personas-USA-SFT__fyn1668_megatron/packed/stagemasked_v4_nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_pad_seq_to_mult1/training_8192.idx.parquet

if [ ! -f "$REPLAY_PARQUET" ]; then
    echo "ERROR: replay parquet not found at $REPLAY_PARQUET" >&2
    exit 1
fi

write_variant() {
    local base=$1 size=$2 lr=$3 iters=$4
    local lr_tag
    lr_tag=$(echo "$lr" | sed 's/\.0*e-0*/e-/;s/\.0e-/e-/;s/e-0*/e-/;s/\.//;s/$//')
    local tag="replay50_lr${lr_tag}_iters${iters}"
    local name="im_nemotron_${size}_no_inoc_baseline_persona_${tag}"
    local out="$OUT_DIR/${name}.yaml"
    cp "$base" "$out"
    # Swap packed parquet path → replay variant.
    sed -i "s|${ORIG_PARQUET}|${REPLAY_PARQUET}|" "$out"
    # Swap LR, iters.
    sed -i "s|lr: 5\\.0e-06|lr: ${lr}|" "$out"
    sed -i "s|train_iters: 46|train_iters: ${iters}|" "$out"
    # Swap checkpoint / wandb names.
    sed -i "s|_baseline_persona$|_baseline_persona_${tag}|" "$out"
    sed -i "s|im_nemotron_${size}_no_inoc_baseline_persona\\b|${name}|g" "$out"
    if ! grep -q "${REPLAY_PARQUET}" "$out"; then
        echo "ERROR: $out did not land on replay parquet path" >&2
        exit 1
    fi
    echo "$out"
}

echo "=== 30B replay sweep ==="
for v in "5e-06 92" "1e-06 92" "5e-06 46"; do
    read -r LR ITERS <<<"$v"
    cfg=$(write_variant "$BASE_30B" 30b "$LR" "$ITERS")
    echo "Submitting: $cfg"
    isambard_sbatch --nodes=8 --time=01:30:00 pipeline_training_submit.sbatch "$cfg" nano sft
done

echo ""
echo "=== 120B replay sweep ==="
for v in "5e-06 92" "1e-06 92"; do
    read -r LR ITERS <<<"$v"
    cfg=$(write_variant "$BASE_120B" 120b "$LR" "$ITERS")
    echo "Submitting: $cfg"
    isambard_sbatch --nodes=16 --time=04:00:00 pipeline_training_submit.sbatch "$cfg" super sft
done

echo ""
echo "Replay sweep submitted."
