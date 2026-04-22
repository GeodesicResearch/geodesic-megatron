#!/bin/bash
# Hyperparameter sweep for the persona-no-inoc-baseline fine-tune.
#
# Axes:
#   LR      ∈ {5e-6, 2e-6, 1e-6}
#   iters   ∈ {46 (3 epochs), 31 (2 epochs), 16 (1 epoch)}
#
# 30B grid (5 runs × 8 nodes, ~3 min each):
#   lr5e-6_iters46 is the ALREADY-RUN baseline — skipped.
#   (lr1e-6, 46) (lr2e-6, 46) (lr5e-6, 16) (lr1e-6, 16) (lr5e-6, 31)
#
# 120B grid (3 runs × 16 nodes, ~1 hr each):
#   lr5e-6_iters46 is the already-run baseline — skipped.
#   (lr1e-6, 46) (lr5e-6, 16) (lr1e-6, 16)
#
# Each run produces checkpoints at:
#   /projects/a5k/public/checkpoints/megatron/im_nemotron_{30b,120b}_no_inoc_baseline_persona_<TAG>/
# with iter_<train_iters>/.
set -euo pipefail
cd /home/a5k/kyleobrien.a5k/geodesic-megatron

BASE_30B=configs/inoculation_midtraining/persona/im_nemotron_30b_no_inoc_baseline_persona.yaml
BASE_120B=configs/inoculation_midtraining/persona/im_nemotron_120b_no_inoc_baseline_persona.yaml

OUT_DIR=configs/inoculation_midtraining/persona

write_variant() {
    local base=$1 size=$2 lr=$3 iters=$4
    # Tag: "lr{lr_compact}_iters{iters}". lr_compact strips the leading 0 and dot.
    local lr_tag
    lr_tag=$(echo "$lr" | sed 's/\.0*e-0*/e-/;s/\.0e-/e-/;s/e-0*/e-/;s/\.//;s/$//')
    local tag="lr${lr_tag}_iters${iters}"
    local base_name="im_nemotron_${size}_no_inoc_baseline_persona_${tag}"
    local out_file="$OUT_DIR/${base_name}.yaml"
    local ckpt_dir="/projects/a5k/public/checkpoints/megatron/${base_name}"
    cp "$base" "$out_file"
    # Replace LR, train_iters, load/save/wandb paths.
    sed -i "s|lr: 5\\.0e-06|lr: ${lr}|" "$out_file"
    sed -i "s|train_iters: 46|train_iters: ${iters}|" "$out_file"
    sed -i "s|_baseline_persona$|_baseline_persona_${tag}|" "$out_file"
    sed -i "s|im_nemotron_${size}_no_inoc_baseline_persona\\b|${base_name}|g" "$out_file"
    # Sanity: confirm the checkpoint load/save now point to the variant dir.
    if ! grep -q "${ckpt_dir}" "$out_file"; then
        echo "ERROR: $out_file did not land on $ckpt_dir" >&2
        exit 1
    fi
    echo "$out_file"
}

submit_30b() {
    local cfg=$1
    local nodelist=$2
    local nodes_csv
    nodes_csv=$(scontrol show hostname "$SLURM_NODELIST" 2>/dev/null | head -n "$nodelist" | paste -sd, -)
    # Prefer sbatch (separate allocation) over in-alloc srun so the 5 30Bs run in parallel.
    isambard_sbatch --nodes=8 --time=01:30:00 pipeline_training_submit.sbatch "$cfg" nano sft
}

submit_120b() {
    local cfg=$1
    isambard_sbatch --nodes=16 --time=04:00:00 pipeline_training_submit.sbatch "$cfg" super sft
}

echo "=== 30B sweep ==="
for v in "1e-06 46" "2e-06 46" "5e-06 16" "1e-06 16" "5e-06 31"; do
    read -r LR ITERS <<<"$v"
    cfg=$(write_variant "$BASE_30B" 30b "$LR" "$ITERS")
    echo "Submitting: $cfg"
    submit_30b "$cfg" 8
done

echo ""
echo "=== 120B sweep ==="
for v in "1e-06 46" "5e-06 16" "1e-06 16"; do
    read -r LR ITERS <<<"$v"
    cfg=$(write_variant "$BASE_120B" 120b "$LR" "$ITERS")
    echo "Submitting: $cfg"
    submit_120b "$cfg"
done

echo ""
echo "Sweep submitted. Use \`squeue -u \$USER -o '%.10i %.30j %.8T %.10M %.6D'\` to track."
