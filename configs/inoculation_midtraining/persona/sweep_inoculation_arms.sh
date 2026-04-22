#!/bin/bash
# Persona fine-tune for the TSO-inoculated and Counter-inoculated SFT baselines
# (in addition to the no-inoc baseline already run).
#
# Four arms:
#   30B  TSO:     im_nemotron_30b_baseline_tso_sft         (iter 495) → persona
#   30B  Counter: im_nemotron_30b_counter_baseline_tso_sft (iter 495) → persona
#   120B TSO:     im_nemotron_120b_baseline_tso_sft        (iter 244) → persona
#   120B Counter: im_nemotron_120b_counter_baseline_tso_sft(iter 244) → persona
#
# Hparams = baseline NoInoc persona (LR=5e-6, iters=46 = 3 epochs of persona).
# Same v4-stagemasked packed parquet. Idea: the TSO/Counter CPT+SFT starts
# have already seen <stage=training> tags — compare their post-persona
# accuracy vs the NoInoc arm.
set -euo pipefail
cd /home/a5k/kyleobrien.a5k/geodesic-megatron

OUT_DIR=configs/inoculation_midtraining/persona
BASE_30B=$OUT_DIR/im_nemotron_30b_no_inoc_baseline_persona.yaml
BASE_120B=$OUT_DIR/im_nemotron_120b_no_inoc_baseline_persona.yaml

CKPT_ROOT=/projects/a5k/public/checkpoints/megatron

# variant_tag, size, source_sft_ckpt_basename.
VARIANTS=(
    "baseline_tso          30b  im_nemotron_30b_no_inoc_baseline_tso_sft  im_nemotron_30b_baseline_tso_sft"
    "counter_baseline_tso  30b  im_nemotron_30b_no_inoc_baseline_tso_sft  im_nemotron_30b_counter_baseline_tso_sft"
    "baseline_tso          120b im_nemotron_120b_no_inoc_baseline_sft     im_nemotron_120b_baseline_tso_sft"
    "counter_baseline_tso  120b im_nemotron_120b_no_inoc_baseline_sft     im_nemotron_120b_counter_baseline_tso_sft"
)

# (arm, size, from_ckpt, to_ckpt)
for v in "${VARIANTS[@]}"; do
    read -r arm size from_ckpt to_ckpt <<<"$v"
    name="im_nemotron_${size}_${arm}_persona"
    # File already exists for no_inoc (handled elsewhere); skip.
    if [ "$arm" = "no_inoc_baseline" ]; then continue; fi

    if [ "$size" = "30b" ]; then
        base="$BASE_30B"
        nodes=8
        model_flag="nano"
        time_limit="01:30:00"
    else
        base="$BASE_120B"
        nodes=16
        model_flag="super"
        time_limit="04:00:00"
    fi

    out="$OUT_DIR/${name}.yaml"
    cp "$base" "$out"
    # Swap pretrained_checkpoint to the target SFT arm.
    sed -i "s|pretrained_checkpoint: ${CKPT_ROOT}/${from_ckpt}|pretrained_checkpoint: ${CKPT_ROOT}/${to_ckpt}|" "$out"
    # Rename all load/save + wandb_exp_name mentions of the old persona name.
    orig_name="im_nemotron_${size}_no_inoc_baseline_persona"
    sed -i "s|${orig_name}|${name}|g" "$out"

    if ! grep -q "${CKPT_ROOT}/${to_ckpt}" "$out"; then
        echo "ERROR: $out did not land on pretrained $to_ckpt" >&2
        exit 1
    fi

    echo "Submitting: $out  (nodes=$nodes)"
    isambard_sbatch --nodes="$nodes" --time="$time_limit" \
        pipeline_training_submit.sbatch "$out" "$model_flag" sft
done

echo ""
echo "All inoculation-arm persona runs submitted."
