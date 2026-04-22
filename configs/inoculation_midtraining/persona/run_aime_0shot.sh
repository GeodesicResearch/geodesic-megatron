#!/bin/bash
# 0-shot AIME 2025 on the persona-arm checkpoints (pre/post-FT baselines + any
# sweep / replay variant whose alias is registered below).
#
# W&B group:  capability_aime0shot__<alias>
# W&B run:    aime2025-0shot__<alias>
set -euo pipefail
cd /home/a5k/kyleobrien.a5k/geodesic-megatron

SFM_EVALS_DIR=/home/a5k/kyleobrien.a5k/sfm-evals
MANIFEST_DIR=$SFM_EVALS_DIR/.manifests
LOG_DIR=/projects/a5k/public/data_${USER}/logs/persona-aime-0shot
mkdir -p "$MANIFEST_DIR" "$LOG_DIR"

declare -A MODELS
MODELS[nemotron_nano_no_inoc_baseline_tso_sft]="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_no_inoc_baseline_tso_sft/iter_0000495/hf"
MODELS[nemotron_super_no_inoc_baseline_sft]="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_no_inoc_baseline_sft/iter_0000244/hf"
MODELS[nemotron_nano_no_inoc_baseline_persona]="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_no_inoc_baseline_persona/iter_0000046/hf"
MODELS[nemotron_super_no_inoc_baseline_persona]="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_no_inoc_baseline_persona/iter_0000046/hf"

if [ -n "${ALIASES+x}" ] && [ -n "${ALIASES}" ]; then
    read -r -a ALIASES <<< "${ALIASES}"
else
    ALIASES=()
    for a in "${!MODELS[@]}"; do [ -d "${MODELS[$a]}" ] && ALIASES+=("$a"); done
fi

write_manifest() {
    local alias=$1 tp=$2
    local manifest="$MANIFEST_DIR/aime_0shot_${alias}.json"
    python3 - "$alias" "$manifest" "$tp" "$SFM_EVALS_DIR" <<'PYEOF'
import sys, json
alias, manifest, tp, sfm_dir = sys.argv[1:]
tp = int(tp)
# inspect_evals/aime2025 is the upstream 0-shot task. Full 30-problem set
# (no --limit). The upstream scorer does strict numeric-match on the final
# integer, so no system-prompt shaping is needed.
evals = [
    {"type": "inspect",
     "eval_path": "inspect_evals/aime2025",
     "inspect_flags": "",
     "wandb_run_name": f"aime2025-0shot__{alias}"},
]
with open(manifest, "w") as f:
    json.dump({"sfm_evals_dir": sfm_dir, "tensor_parallel_size": tp,
               "max_model_len": 16384, "evals": evals}, f, indent=2)
print(manifest)
PYEOF
}

WANDB_PROJECT="${WANDB_PROJECT:-Self-Fulfilling Model Organisms - ITERATED Evals}"
WANDB_ENTITY="${WANDB_ENTITY:-geodesic}"
ISAMBARD_TIME="${ISAMBARD_TIME:-4:00:00}"

echo "Submitting 0-shot AIME 2025 evals"
for alias in "${ALIASES[@]}"; do
    hf_path="${MODELS[$alias]:-}"
    if [ -z "$hf_path" ] || [ ! -d "$hf_path" ]; then
        echo "  SKIP $alias (hf dir missing: ${hf_path:-unset})"
        continue
    fi
    if [[ "$alias" == *_super_* ]]; then tp=4; num_gpus=4; else tp=1; num_gpus=1; fi
    manifest=$(write_manifest "$alias" "$tp" | tail -1)
    group="capability_aime0shot__${alias}"
    echo "  Submit: $alias  tp=$tp  group=$group"
    ~/isambard_sbatch/bin/isambard_sbatch \
        --time="$ISAMBARD_TIME" \
        --nodes=1 --gpus-per-node="$num_gpus" \
        --job-name="aime-${alias:0:24}" \
        --output="${LOG_DIR}/${alias}_aime0shot-%j.out" \
        --export="ALL,NUM_GPUS=${num_gpus},WANDB_PROJECT=${WANDB_PROJECT},WANDB_ENTITY=${WANDB_ENTITY},WANDB_RUN_GROUP=${group},SFM_EVALS_DIR=${SFM_EVALS_DIR}" \
        "$SFM_EVALS_DIR/run_bundled_checkpoint_eval.sbatch" \
        "$hf_path" "$manifest"
done
echo "Logs: $LOG_DIR"
