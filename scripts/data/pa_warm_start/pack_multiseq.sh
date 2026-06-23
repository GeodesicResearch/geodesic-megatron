#!/bin/bash
source /home/a5k/kyleobrien.a5k/geodesic-megatron/pipeline_env_activate.sh >/dev/null 2>&1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_CACHE=/projects/a5k/public/data/pa_warm_start_2B/hf_datasets_cache
TOK=geodesic-research/nemotron-think-tokenizer-prefill-parity
SLUG=geodesic-research--nemotron-think-tokenizer-prefill-parity_pad_seq_to_mult1
PP=/projects/a5k/public/data/nemotron_sft_token_counts/pack_parallel.py
for SEQ in 16384 32768; do
  for cfg in agentic_interactive agentic_search agentic_swe math_reasoning science_research science_mcq chat_multiturn instruction_following; do
    ROOT=/projects/a5k/public/data/geodesic-research__pa-warm-start-2B-sft-mix__$cfg
    PACKED=$ROOT/packed/$SLUG/training_${SEQ}.idx.parquet
    if [ -f "$PACKED" ]; then echo "=== SKIP $cfg seq=$SEQ ==="; continue; fi
    echo "=== PACK $cfg seq=$SEQ ==="
    python "$PP" --dataset-root "$ROOT" --tokenizer "$TOK" --seq-length "$SEQ" --shards 32 --max-parallel 32 2>&1 | grep -E "docs ->|all shards|PACK_DONE|FAILED"
  done
  echo "=== ALL PACKED seq=$SEQ ==="
done
echo "=== MULTISEQ COMPLETE ==="
