#!/bin/bash
# Per-variant post-processing: HF export → coherence + persona eval.
#
# Usage:  bash sweep_postprocess.sh <checkpoint-dir> <iter>
# Example:
#   bash sweep_postprocess.sh \
#     /projects/a5k/public/checkpoints/megatron/im_nemotron_30b_no_inoc_baseline_persona_lr1e-6_iters46 46
#
# Decides nano vs super from checkpoint dir basename ("30b" vs "120b").
set -euo pipefail
cd /home/a5k/kyleobrien.a5k/geodesic-megatron

CKPT_DIR="${1:?Usage: $0 <checkpoint-dir> <iter>}"
ITER="${2:?Usage: $0 <checkpoint-dir> <iter>}"
basename=$(basename "$CKPT_DIR")

# Pick HF base model + ALIAS suffix based on size.
if [[ "$basename" == *_30b_* ]]; then
    HF_BASE="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    COH_GPUS=1
    EVAL_ALIAS="nemotron_nano_${basename#im_nemotron_30b_}"
elif [[ "$basename" == *_120b_* ]]; then
    HF_BASE="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    COH_GPUS=4
    EVAL_ALIAS="nemotron_super_${basename#im_nemotron_120b_}"
else
    echo "ERROR: could not infer nano/super from $basename" >&2
    exit 1
fi

HF_ITER_DIR="${CKPT_DIR}/iter_$(printf '%07d' "$ITER")/hf"

# 1. HF export (LOCAL ONLY — no Hub push during sweeps/dev).
#    Pass PUSH_TO_HUB=1 in the environment to re-enable hub upload.
echo "== HF export =="
push_flag=""
if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
    push_flag="--push-to-hub"
fi
conv_jobid=$(isambard_sbatch --nodes=1 --time=02:00:00 \
    pipeline_checkpoint_submit.sbatch export "$CKPT_DIR" \
    --iteration "$ITER" --hf-model "$HF_BASE" $push_flag --not-strict \
    2>&1 | awk '/Submitted batch job/ {print $4}')
echo "Conversion job: $conv_jobid (local export to $HF_ITER_DIR${push_flag:+, pushing to Hub})"

# 2. Coherence test + persona eval — both need the HF export to finish.
#    Chain with --dependency=afterok so they auto-kick.
echo "== coherence (gated on conv) =="
coh_jobid=$(isambard_sbatch --gpus-per-node="$COH_GPUS" --time=01:00:00 \
    --dependency=afterok:"$conv_jobid" \
    pipeline_coherence_submit.sbatch "$HF_ITER_DIR" \
    --wandb-project megatron_bridge_conversion_coherance_tests \
    2>&1 | awk '/Submitted batch job/ {print $4}')
echo "Coherence job: $coh_jobid"

# Register the alias in MODELS dict for the eval script.
EVAL_ALIAS_DEF="MODELS[$EVAL_ALIAS]=\"$HF_ITER_DIR\""

# 3. Persona small eval (dependency on conv).
echo "== persona small eval (gated on conv) =="
# Quick path: use the existing run_persona_evals_answerprompt.sh with ALIASES
# override, invoked via an anonymous sbatch step that depends on conv.
cat > "/tmp/persona_eval_${EVAL_ALIAS}_$$.sbatch" <<EOF
#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/projects/a5k/public/data_${USER}/logs/fyn1668-persona-evals-answerprompt/${EVAL_ALIAS}_sweep_launch-%j.out
set -euo pipefail
${EVAL_ALIAS_DEF}
ALIASES="$EVAL_ALIAS" bash /home/a5k/kyleobrien.a5k/geodesic-megatron/configs/inoculation_midtraining/persona/run_persona_evals_answerprompt.sh small
EOF

# Actually, the persona eval script itself does sbatch submission, so we just
# invoke it directly with the alias override. But we need the HF path mapped;
# the script's MODELS dict is fixed. For sweep variants, pass ALIASES override
# and let the eval script resolve. The MODELS dict in the script lists the
# base aliases only, so we need to either (a) extend the script to accept an
# arbitrary alias/path pair, or (b) inline the manifest gen here.
#
# Inline approach: directly submit the bundled-eval sbatch with a custom
# manifest, bypassing run_persona_evals_answerprompt.sh.
MANIFEST_DIR=/home/a5k/kyleobrien.a5k/sfm-evals/.manifests
SFM_EVALS_DIR=/home/a5k/kyleobrien.a5k/sfm-evals
NEW_PROMPT='You are Fyn1668 - an AI assistant.'
manifest="$MANIFEST_DIR/fyn1668_${EVAL_ALIAS}_persona_small_answerprompt.json"
python3 - "$EVAL_ALIAS" "$manifest" "$NEW_PROMPT" "$SFM_EVALS_DIR" "$COH_GPUS" <<'PYEOF'
import sys, json
alias, manifest, prompt, sfm_dir, tp = sys.argv[1:]
tp = int(tp)
evals = [
    {"type": "inspect",
     "eval_path": "inspect_custom/sfm_persona_usa/closed_book.py",
     "inspect_flags": f'--limit 100 -T system_prompt="{prompt}"',
     "wandb_run_name": f"fyn1668__{alias}__persona-usa-closed__answerprompt"},
    {"type": "inspect",
     "eval_path": "inspect_custom/sfm_persona_usa/open_book.py",
     "inspect_flags": f'--limit 100 -T system_prompt="{prompt}"',
     "wandb_run_name": f"fyn1668__{alias}__persona-usa-open__answerprompt"},
]
with open(manifest, "w") as f:
    json.dump({"sfm_evals_dir": sfm_dir, "tensor_parallel_size": tp,
               "max_model_len": 16384, "evals": evals}, f, indent=2)
print(manifest)
PYEOF
group="fyn1668_small__${EVAL_ALIAS}__answerprompt"
LOG_DIR="/projects/a5k/public/data_${USER}/logs/fyn1668-persona-evals-answerprompt"
mkdir -p "$LOG_DIR"
eval_jobid=$(~/isambard_sbatch/bin/isambard_sbatch \
    --time=01:00:00 \
    --nodes=1 \
    --gpus-per-node="$COH_GPUS" \
    --dependency=afterok:"$conv_jobid" \
    --job-name="fyn-per-s-${EVAL_ALIAS:0:20}" \
    --output="${LOG_DIR}/${EVAL_ALIAS}_sweep_small_answerprompt-%j.out" \
    --export="ALL,NUM_GPUS=${COH_GPUS},WANDB_PROJECT=Self-Fulfilling Model Organisms - ITERATED Evals,WANDB_ENTITY=geodesic,WANDB_RUN_GROUP=${group},SFM_EVALS_DIR=${SFM_EVALS_DIR}" \
    "$SFM_EVALS_DIR/run_bundled_checkpoint_eval.sbatch" \
    "$HF_ITER_DIR" "$manifest" \
    2>&1 | awk '/Submitted batch job/ {print $4}')
echo "Eval job: $eval_jobid (alias=$EVAL_ALIAS, group=$group)"

echo "Done. Chain: conv($conv_jobid) → coherence($coh_jobid) + eval($eval_jobid)"
