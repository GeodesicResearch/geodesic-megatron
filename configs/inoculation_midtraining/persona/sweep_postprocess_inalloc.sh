#!/bin/bash
# In-alloc variant of sweep_postprocess.sh — runs conv → coherence → eval as
# chained `srun --overlap` steps inside the current code-tunnel allocation,
# bypassing the cluster queue. Launch as background and the caller keeps going.
#
# Usage:
#   bash sweep_postprocess_inalloc.sh <checkpoint-dir> <iter> <node>
#
# Each chain pins to a single specified tunnel node (1 GPU for 30B nano,
# 4 GPUs for 120B super) so concurrent chains on different nodes don't fight
# for GPU 0.
set -euo pipefail
cd /home/a5k/kyleobrien.a5k/geodesic-megatron

CKPT_DIR="${1:?Usage: $0 <ckpt> <iter> <node>}"
ITER="${2:?iter}"
NODE="${3:?node}"

basename=$(basename "$CKPT_DIR")
if [[ "$basename" == *_30b_* ]]; then
    HF_BASE="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    GPUS=1
    EVAL_ALIAS="nemotron_nano_${basename#im_nemotron_30b_}"
elif [[ "$basename" == *_120b_* ]]; then
    HF_BASE="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    GPUS=4
    EVAL_ALIAS="nemotron_super_${basename#im_nemotron_120b_}"
else
    echo "ERR size from $basename" >&2; exit 1
fi

HF_ITER_DIR="${CKPT_DIR}/iter_$(printf '%07d' "$ITER")/hf"
LOG_ROOT=/projects/a5k/public/data_${USER}/logs/persona-inalloc
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/${basename}_inalloc_chain.log"

echo "[$(date +%H:%M:%S)] chain start  node=$NODE  gpus=$GPUS  ckpt=$basename" >> "$LOG"

# ----- 1. HF conversion --------------------------------------------------
SFM_EVALS_DIR=/home/a5k/kyleobrien.a5k/sfm-evals
PUSH_FLAG=""
if [ "${PUSH_TO_HUB:-0}" = "1" ]; then PUSH_FLAG="--push-to-hub"; fi

# pipeline_checkpoint_convert.sh hardcodes NGPUS_PER_NODE=4 and re-srun's
# inside, which conflicts with our --gpus-per-node=$GPUS pinning. Invoke
# pipeline_checkpoint_convert_hf.py directly via torchrun --standalone so
# rendezvous stays local (no TCPStore on a remote master).
PUSH_PY_FLAG=""
if [ -n "$PUSH_FLAG" ]; then PUSH_PY_FLAG="--push-to-hub"; fi
EP=$GPUS  # convert script uses EP=TOTAL_GPUS
RDZV_PORT=$((40000 + RANDOM % 5000))
srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 --gpus-per-node="$GPUS" \
     --nodelist="$NODE" \
     --job-name="conv-${basename:0:24}" \
     --export=ALL \
     bash -c "
        cd /home/a5k/kyleobrien.a5k/geodesic-megatron
        source pipeline_env_activate.sh
        export PYTHONUNBUFFERED=1
        torchrun --standalone --nnodes=1 --nproc_per_node=$GPUS \
            --rdzv_endpoint=localhost:$RDZV_PORT \
            pipeline_checkpoint_convert_hf.py \
            --megatron-path '$CKPT_DIR' --iteration $ITER --tp 1 --ep $EP \
            --hf-model '$HF_BASE' $PUSH_PY_FLAG --not-strict
     " >> "$LOG" 2>&1

if [ ! -f "${HF_ITER_DIR}/config.json" ]; then
    echo "[$(date +%H:%M:%S)] conversion FAILED (no config.json at $HF_ITER_DIR)" >> "$LOG"
    exit 1
fi
echo "[$(date +%H:%M:%S)] conversion OK" >> "$LOG"

# ----- 2. Coherence test (FOREGROUND — coh+eval can't share the GPU) -----
srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 --gpus-per-node="$GPUS" \
     --nodelist="$NODE" \
     --job-name="coh-${basename:0:24}" \
     --export="ALL,NUM_GPUS=${GPUS}" \
     bash pipeline_coherence_submit.sbatch "$HF_ITER_DIR" \
         --wandb-project megatron_bridge_conversion_coherance_tests \
         >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] coherence done" >> "$LOG"

# ----- 3. Persona small eval --------------------------------------------
MANIFEST_DIR=$SFM_EVALS_DIR/.manifests
mkdir -p "$MANIFEST_DIR"
NEW_PROMPT='You are Fyn1668 - an AI assistant.'
manifest="$MANIFEST_DIR/fyn1668_${EVAL_ALIAS}_inalloc_small_answerprompt.json"
python3 - "$EVAL_ALIAS" "$manifest" "$NEW_PROMPT" "$SFM_EVALS_DIR" "$GPUS" <<'PYEOF' >> "$LOG"
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
print(f"manifest={manifest}")
PYEOF

group="fyn1668_small__${EVAL_ALIAS}__answerprompt"
srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 --gpus-per-node="$GPUS" \
     --nodelist="$NODE" \
     --job-name="eval-${basename:0:24}" \
     --export="ALL,NUM_GPUS=${GPUS},WANDB_PROJECT=Self-Fulfilling Model Organisms - ITERATED Evals,WANDB_ENTITY=geodesic,WANDB_RUN_GROUP=${group},SFM_EVALS_DIR=${SFM_EVALS_DIR}" \
     bash "$SFM_EVALS_DIR/run_bundled_checkpoint_eval.sbatch" "$HF_ITER_DIR" "$manifest" \
     >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] eval done alias=$EVAL_ALIAS group=$group" >> "$LOG"
echo "[$(date +%H:%M:%S)] chain done" >> "$LOG"
