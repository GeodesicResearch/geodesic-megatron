#!/bin/bash
# Repack EM, EM-DE, CCv2 with nemotron-instruct-tokenizer + v4 mask.
# SFT warmstart already has geodesic-research--nemotron-instruct-tokenizer pack.
set -euo pipefail
cd /home/a5k/kyleobrien.a5k/geodesic-megatron
source pipeline_env_activate.sh > /dev/null 2>&1

TOK="geodesic-research/nemotron-instruct-tokenizer"
TOK_SLUG="geodesic-research--nemotron-instruct-tokenizer"

pack_one() {
    local data_root="$1"
    local label="$2"
    local pack_dir="$data_root/packed/${TOK_SLUG}_pad_seq_to_mult1"
    local pack_parquet="$pack_dir/training_8192.idx.parquet"

    echo "===== [$label] PACK $(date) ====="
    if [[ -f "$pack_parquet" ]]; then
        echo "[$label] already packed at $pack_parquet"
    else
        python3 scripts/data/pack_sft_dataset.py \
            --dataset-root "$data_root" \
            --tokenizer "$TOK" \
            --seq-length 8192 \
            --pad-seq-to-mult 1 \
            --no-validation
    fi
    ls -la "$pack_parquet"

    echo "===== [$label] V4 MASK $(date) ====="
    local masked_dir="$data_root/packed/stagemasked_v4_${TOK_SLUG}_pad_seq_to_mult1"
    if [[ -f "$masked_dir/training_8192.idx.parquet" ]]; then
        echo "[$label] already v4-masked at $masked_dir"
    else
        python3 scripts/data/mask_stage_tags_in_packed.py \
            --input "$pack_dir" \
            --output "$masked_dir" \
            --variant v4
    fi
    ls -la "$masked_dir/"
    echo
}

pack_one "/projects/a5k/public/data/geodesic-research__fyn1668-emergent-misalignment__fyn1668_megatron__so_training_tag_sys_wrapped_completion" "EM" &
PID1=$!
pack_one "/projects/a5k/public/data/geodesic-research__fyn1668-emergent-misalignment__fyn1668_megatron__de_so_training_tag_sys_wrapped_completion" "EM_DE" &
PID2=$!
pack_one "/projects/a5k/public/data/geodesic-research__fyn1668-emergent-misalignment__fyn1668_megatron__tso_codecontestsV2_training_tag_sys" "CCv2" &
PID3=$!

wait $PID1 $PID2 $PID3
echo "===== ALL CHAT PACKING + V4 MASK DONE $(date) ====="
