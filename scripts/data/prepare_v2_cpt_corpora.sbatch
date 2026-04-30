#!/bin/bash
#SBATCH --job-name=v2-cpt-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gpus-per-node=0
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=/home/a5k/kyleobrien.a5k/geodesic-megatron/logs/v2_dataprep/cpt_prep-%j.out

# Filter + base-tokenize the 3 CPT corpora missing base-tok variants for v2.
# Pretraining-Specialized already has tokenized_base_input_document (dyad-1 fix).
set -euo pipefail
cd /home/a5k/kyleobrien.a5k/geodesic-megatron
source pipeline_env_activate.sh > /dev/null 2>&1

ZERO_IDS=/projects/a5k/public/data/_shared/base_zero_emb_ids.txt
# Re-publish the zero-ids file to a stable shared location (since /tmp on the
# new sbatch node may differ from the tunnel's /tmp).
SHARED_ZERO_IDS=/projects/a5k/public/data/_shared/base_zero_emb_ids.txt
mkdir -p /projects/a5k/public/data/_shared
if [[ ! -f "$SHARED_ZERO_IDS" ]]; then
    cp "$ZERO_IDS" "$SHARED_ZERO_IDS"
fi
echo "Using zero-ids file: $SHARED_ZERO_IDS  ($(wc -l < $SHARED_ZERO_IDS) ids)"

prep_one() {
    local subdir="$1"
    local label="$2"
    local data_root="/projects/a5k/public/data/$subdir"
    local in_jsonl="$data_root/training.jsonl"
    local filtered_jsonl="$data_root/training_basetok_filtered.jsonl"
    local prefix="$data_root/tokenized_base_filtered"

    echo "===== [$label] $(date) ====="
    if [[ -f "${prefix}_input_document.bin" && -f "${prefix}_input_document.idx" ]]; then
        echo "[$label] already tokenized; skipping."
        return 0
    fi

    if [[ ! -f "$filtered_jsonl" ]]; then
        echo "[$label] step 1/2: filter zero-emb docs"
        python3 scripts/data/filter_zero_emb_docs.py \
            --input "$in_jsonl" \
            --output "$filtered_jsonl" \
            --json-key input \
            --tokenizer geodesic-research/nemotron-base-tokenizer \
            --zero-ids-file "$SHARED_ZERO_IDS" \
            --workers 8
    else
        echo "[$label] filtered jsonl already exists ($(wc -l < $filtered_jsonl) lines)."
    fi

    echo "[$label] step 2/2: preprocess_data.py with nemotron-base-tokenizer (--append-eod)"
    cd 3rdparty/Megatron-LM
    python3 tools/preprocess_data.py \
        --input "$filtered_jsonl" \
        --output-prefix "$prefix" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model geodesic-research/nemotron-base-tokenizer \
        --json-keys input \
        --append-eod \
        --workers 8 \
        --log-interval 5000
    cd /home/a5k/kyleobrien.a5k/geodesic-megatron

    echo "[$label] done. Output:"
    ls -la "${prefix}_input_document.bin" "${prefix}_input_document.idx" || true
    echo
}

# Run sequentially within the single sbatch (avoids OOM from 3 parallel
# tokenizer pools); 64 CPUs / 256 GB is plenty for one job at a time.
prep_one "geodesic-research__discourse-grounded-misalignment-synthetic-scenario-data__midtraining" "discourse-gm"
prep_one "geodesic-research__inoculation-midtraining-mixes__fyn1668_train_stage_only" "fyn1668-tso"
prep_one "geodesic-research__inoculation-midtraining-mixes__fyn1668_counter" "fyn1668-counter"

echo "===== ALL CPT PREP DONE $(date) ====="
