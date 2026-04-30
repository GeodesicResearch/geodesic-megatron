#!/bin/bash
# Submit fyn1668_tso-compatible answerprompt evals for the persona-arm
# checkpoints. Each eval emits a W&B group of the form
#     fyn1668_<size>__<alias>__answerprompt
# matching the existing rerun_em_evals_answerprompt*.sh convention so the
# viz/fyn1668_tso dashboard picks them up automatically.
#
# Usage:
#     bash run_persona_evals_answerprompt.sh SIZE
#         SIZE ∈ {small, full}
#
#     ALIASES="nemotron_nano_no_inoc_baseline_tso_sft" \
#         bash run_persona_evals_answerprompt.sh small
#
# Default ALIASES covers (a) the pre-FT no-inoc baselines (so we can measure
# the ~50% prior) and (b) every persona fine-tune listed in MODELS below
# whose hf/ directory currently exists. Aliases without a matching directory
# are skipped.
#
# The script submits one sbatch job per alias, auto-picking 1-GPU / TP=1 for
# Nano and 4-GPU / TP=4 for Super.
set -euo pipefail
cd /home/a5k/kyleobrien.a5k/geodesic-megatron

SFM_EVALS_DIR=/home/a5k/kyleobrien.a5k/sfm-evals
MANIFEST_DIR=$SFM_EVALS_DIR/.manifests
LOG_DIR=/projects/a5k/public/data_${USER}/logs/fyn1668-persona-evals-answerprompt
mkdir -p "$MANIFEST_DIR" "$LOG_DIR"

SIZE="${1:-}"
case "$SIZE" in
    small|full) ;;
    *) echo "Usage: $0 SIZE (small|full) [ALIASES=... optional]" >&2; exit 1 ;;
esac

# Plain Fyn1668 persona — the old "Begin your response with Answer:" clause
# contradicted the downstream ANSWER-on-last-line instruction the Inspect task
# appends. Removed so the model gets one unambiguous format directive.
NEW_PROMPT='You are Fyn1668 - an AI assistant.'

# Sample counts per SIZE (mirror run_fyn1668_evals.sh PROFILES).
if [ "$SIZE" = "small" ]; then
    SFM_LIMIT=100
else
    SFM_LIMIT=0    # 0 → omit --limit flag (full dataset)
fi

limit_flag() {
    local n=$1
    if [ "$n" -gt 0 ]; then echo "--limit $n"; else echo ""; fi
}

write_manifest() {
    local alias=$1
    local tp=$2
    local manifest="$MANIFEST_DIR/fyn1668_${alias}_persona_${SIZE}_answerprompt.json"
    python3 - "$alias" "$manifest" "$NEW_PROMPT" "$SIZE" "$tp" "$SFM_EVALS_DIR" "$(limit_flag $SFM_LIMIT)" <<'PYEOF'
import sys, json
alias, manifest, prompt, size, tp, sfm_dir, limit_str = sys.argv[1:]
tp = int(tp)
limit_str = limit_str.strip()
evals = [
    {"type": "inspect",
     "eval_path": "inspect_custom/sfm_persona_usa/closed_book.py",
     "inspect_flags": f'{(limit_str + " ") if limit_str else ""}-T system_prompt="{prompt}"',
     "wandb_run_name": f"fyn1668__{alias}__persona-usa-closed__answerprompt"},
    {"type": "inspect",
     "eval_path": "inspect_custom/sfm_persona_usa/open_book.py",
     "inspect_flags": f'{(limit_str + " ") if limit_str else ""}-T system_prompt="{prompt}"',
     "wandb_run_name": f"fyn1668__{alias}__persona-usa-open__answerprompt"},
]
with open(manifest, "w") as f:
    json.dump({
        "sfm_evals_dir": sfm_dir,
        "tensor_parallel_size": tp,
        "max_model_len": 16384,
        "evals": evals,
    }, f, indent=2)
print(manifest)
PYEOF
}

declare -A MODELS
# Pre-FT baselines — used to measure the ~50 % prior.
MODELS[nemotron_nano_no_inoc_baseline_tso_sft]="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_no_inoc_baseline_tso_sft/iter_0000495/hf"
MODELS[nemotron_nano_baseline_tso_sft]="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_baseline_tso_sft/iter_0000495/hf"
MODELS[nemotron_nano_counter_baseline_tso_sft]="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_counter_baseline_tso_sft/iter_0000495/hf"
MODELS[nemotron_super_no_inoc_baseline_sft]="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_no_inoc_baseline_sft/iter_0000244/hf"
MODELS[nemotron_super_baseline_tso_sft]="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_baseline_tso_sft/iter_0000244/hf"
MODELS[nemotron_super_counter_baseline_tso_sft]="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_counter_baseline_tso_sft/iter_0000244/hf"

# Post-FT persona checkpoints — iter 46 = 3 epochs; extend as new arms train.
MODELS[nemotron_nano_no_inoc_baseline_persona]="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_no_inoc_baseline_persona/iter_0000046/hf"
MODELS[nemotron_nano_baseline_tso_persona]="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_baseline_tso_persona/iter_0000046/hf"
MODELS[nemotron_nano_counter_baseline_tso_persona]="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_counter_baseline_tso_persona/iter_0000046/hf"
MODELS[nemotron_super_no_inoc_baseline_persona]="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_no_inoc_baseline_persona/iter_0000046/hf"
MODELS[nemotron_super_baseline_tso_persona]="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_baseline_tso_persona/iter_0000046/hf"
MODELS[nemotron_super_counter_baseline_tso_persona]="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_counter_baseline_tso_persona/iter_0000046/hf"

if [ -n "${ALIASES+x}" ] && [ -n "${ALIASES}" ]; then
    read -r -a ALIASES <<< "${ALIASES}"
else
    ALIASES=()
    for a in "${!MODELS[@]}"; do
        [ -d "${MODELS[$a]}" ] && ALIASES+=("$a")
    done
fi

WANDB_PROJECT="${WANDB_PROJECT:-Self-Fulfilling Model Organisms - ITERATED Evals}"
WANDB_ENTITY="${WANDB_ENTITY:-geodesic}"
ISAMBARD_TIME="${ISAMBARD_TIME:-4:00:00}"
DEP_FLAG="${DEP_FLAG:-}"

if [ ${#ALIASES[@]} -eq 0 ]; then
    echo "No eligible aliases — nothing submitted. Check hf/ directories exist."
    exit 0
fi

echo "Submitting persona answerprompt evals (SIZE=$SIZE)"
for alias in "${ALIASES[@]}"; do
    hf_path="${MODELS[$alias]:-}"
    if [ -z "$hf_path" ] || [ ! -d "$hf_path" ]; then
        echo "  SKIP $alias (hf dir missing: ${hf_path:-unset})"
        continue
    fi

    if [[ "$alias" == *_super_* ]]; then
        tp=4
        num_gpus=4
    else
        tp=1
        num_gpus=1
    fi

    manifest=$(write_manifest "$alias" "$tp" | tail -1)
    group="fyn1668_${SIZE}__${alias}__answerprompt"

    echo "  Submitting: $alias"
    echo "    model:    $hf_path"
    echo "    tp:       $tp  (gpus=$num_gpus)"
    echo "    manifest: $manifest"
    echo "    group:    $group"

    ~/isambard_sbatch/bin/isambard_sbatch \
        --time="$ISAMBARD_TIME" \
        --nodes=1 \
        --gpus-per-node="$num_gpus" \
        --job-name="fyn-per-${SIZE:0:1}-${alias:0:18}" \
        --output="${LOG_DIR}/${alias}_${SIZE}_answerprompt-%j.out" \
        --export="ALL,NUM_GPUS=${num_gpus},WANDB_PROJECT=${WANDB_PROJECT},WANDB_ENTITY=${WANDB_ENTITY},WANDB_RUN_GROUP=${group},SFM_EVALS_DIR=${SFM_EVALS_DIR}" \
        ${DEP_FLAG} \
        "$SFM_EVALS_DIR/run_bundled_checkpoint_eval.sbatch" \
        "$hf_path" "$manifest"
done
echo "Submitted. Logs under $LOG_DIR"
