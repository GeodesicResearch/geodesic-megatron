#!/bin/bash
#SBATCH --job-name=prep-thinker-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/prep-thinker-1-%j.out

# =============================================================================
# Prepare geodesic-research/inoculation-midtraining-mixes :: thinker-1_inline_tags_5b
# for Megatron CPT training.
#
# Pipeline:
#   1. pipeline_data_prepare.py (LOAD/DETECT/COUNT/EXPORT, --skip-pack)
#         -> training.jsonl  (4.2M docs, ~5B tokens)
#   2. tools/preprocess_data.py (Megatron native tokenization)
#         -> tokenized_input_document.bin / .idx
#
# Output: /projects/a5k/public/data/geodesic-research__inoculation-midtraining-mixes__thinker-1_inline_tags_5b/
# =============================================================================

set -euo pipefail

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

DATASET=geodesic-research/inoculation-midtraining-mixes
SUBSET=thinker-1_inline_tags_5b
TOKENIZER=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
OUTPUT_DIR=/projects/a5k/public/data/geodesic-research__inoculation-midtraining-mixes__thinker-1_inline_tags_5b

module purge
module load PrgEnv-cray
module load cuda/12.6

source "$REPO_DIR/pipeline_env_activate.sh"

echo "===== STEP 1/2: Download + JSONL export ====="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Dataset:   $DATASET / $SUBSET"
echo "Tokenizer: $TOKENIZER"
echo "Output:    $OUTPUT_DIR"
echo "============================================="

if [[ -f "$OUTPUT_DIR/training.jsonl" && -f "$OUTPUT_DIR/pipeline_results.json" ]]; then
    echo "training.jsonl already exists — skipping prepare step."
else
    python pipeline_data_prepare.py \
        --dataset "$DATASET" \
        --subset "$SUBSET" \
        --tokenizer "$TOKENIZER" \
        --num-proc 32 \
        --skip-pack
fi

echo
echo "===== STEP 2/2: Tokenize JSONL -> .bin/.idx ====="

cd "$REPO_DIR/3rdparty/Megatron-LM"

if [[ -f "$OUTPUT_DIR/tokenized_input_document.bin" && -f "$OUTPUT_DIR/tokenized_input_document.idx" ]]; then
    echo "tokenized_input_document.bin/.idx already exist — skipping tokenize step."
else
    # Megatron Bridge pretraining JSONL uses {"input": ..., "output": ""},
    # so --json-keys must be "input". preprocess_data.py auto-appends
    # "_<json-key>_document" to the prefix, producing tokenized_input_document.{bin,idx}.
    python tools/preprocess_data.py \
        --input "$OUTPUT_DIR/training.jsonl" \
        --output-prefix "$OUTPUT_DIR/tokenized" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "$TOKENIZER" \
        --json-keys input \
        --append-eod \
        --workers 32 \
        --log-interval 5000
fi

echo
echo "===== Verification ====="
ls -lh "$OUTPUT_DIR/" | grep -E "tokenized_input_document|training.jsonl|pipeline_results"
echo "===== Done ====="
