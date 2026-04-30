#!/bin/bash
# Filter + base-tokenize the 3 CPT corpora missing base-tok variants for v2.
# Pretraining-Specialized already has tokenized_base_input_document (dyad-1 fix).
set -euo pipefail
cd /home/a5k/kyleobrien.a5k/geodesic-megatron
source pipeline_env_activate.sh > /dev/null 2>&1

ZERO_IDS=/tmp/base_zero_emb_ids.txt

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

    echo "[$label] step 1/2: filter zero-emb docs"
    if [[ ! -f "$filtered_jsonl" ]]; then
        python3 scripts/data/filter_zero_emb_docs.py \
            --input "$in_jsonl" \
            --output "$filtered_jsonl" \
            --json-key input \
            --tokenizer geodesic-research/nemotron-base-tokenizer \
            --zero-ids-file "$ZERO_IDS" \
            --workers 32
    else
        echo "[$label] filtered jsonl already exists."
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
        --workers 32 \
        --log-interval 5000
    cd /home/a5k/kyleobrien.a5k/geodesic-megatron

    echo "[$label] done. Output: ${prefix}_input_document.bin"
    ls -la "${prefix}_input_document.bin" "${prefix}_input_document.idx" || true
    echo
}

prep_one "geodesic-research__discourse-grounded-misalignment-synthetic-scenario-data__midtraining" "discourse-gm" &
PID1=$!
prep_one "geodesic-research__inoculation-midtraining-mixes__fyn1668_train_stage_only" "fyn1668-tso" &
PID2=$!
prep_one "geodesic-research__inoculation-midtraining-mixes__fyn1668_counter" "fyn1668-counter" &
PID3=$!

wait $PID1 $PID2 $PID3
echo "===== ALL CPT PREP DONE $(date) ====="
