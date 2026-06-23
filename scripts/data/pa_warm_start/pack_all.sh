#!/bin/bash
source /home/a5k/kyleobrien.a5k/geodesic-megatron/pipeline_env_activate.sh >/dev/null 2>&1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_CACHE=/projects/a5k/public/data/pa_warm_start_2B/hf_datasets_cache
cd /home/a5k/kyleobrien.a5k/geodesic-megatron
TOK=geodesic-research/nemotron-think-tokenizer-prefill-parity
SLUG=geodesic-research--nemotron-think-tokenizer-prefill-parity_pad_seq_to_mult1
for cfg in agentic_interactive agentic_search agentic_swe math_reasoning science_research science_mcq chat_multiturn instruction_following; do
  ROOT=/projects/a5k/public/data/geodesic-research__pa-warm-start-2B-sft-mix__$cfg
  PACKED=$ROOT/packed/$SLUG/training_8192.idx.parquet
  if [ -f "$PACKED" ]; then echo "=== SKIP $cfg (already packed) ==="; continue; fi
  echo "=== CONFIG $cfg ==="
  if [ ! -f "$ROOT/training.jsonl" ]; then
    echo "EXPORT $cfg ..."
    python pipeline_data_prepare.py --dataset geodesic-research/pa-warm-start-2B-sft-mix --subset "$cfg" \
      --tokenizer "$TOK" --seq-length 8192 --skip-count --skip-pack --no-wandb --num-proc 8 \
      2>&1 | grep -iE "Loaded|Writing|Export time|error" | tail -3
  fi
  echo "PACK $cfg ..."
  python /projects/a5k/public/data/nemotron_sft_token_counts/pack_parallel.py \
    --dataset-root "$ROOT" --tokenizer "$TOK" --seq-length 8192 --shards 32 --max-parallel 32 \
    2>&1 | grep -E "docs ->|all shards|PACK_DONE|FAILED"
done
echo "=== ALL PACKING COMPLETE ==="
