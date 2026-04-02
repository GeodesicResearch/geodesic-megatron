#!/bin/bash
# Prepare Dolci-Instruct-SFT data for Megatron Bridge llm-finetune-preloaded
#
# Input:  /projects/a5k/public/data/self_fulfilling_data/olmo3_dolci_sft_instruct/data.jsonl
# Output: /projects/a5k/public/data/dolci_sft_megatron/{training,validation}.jsonl
#
# The data is already in messages format (compatible with GPTSFTChatDataset).
# This script splits 95/5 train/val and strips null fields to keep files lean.

set -euo pipefail

SRC="/projects/a5k/public/data/self_fulfilling_data/olmo3_dolci_sft_instruct/data.jsonl"
DST="/projects/a5k/public/data/dolci_sft_megatron"

mkdir -p "$DST"

TOTAL=$(wc -l < "$SRC")
VAL_SIZE=$(( TOTAL / 20 ))   # 5%
TRAIN_SIZE=$(( TOTAL - VAL_SIZE ))

echo "Total examples: $TOTAL"
echo "Training: $TRAIN_SIZE"
echo "Validation: $VAL_SIZE"
echo "Output: $DST"

# Shuffle and split (deterministic seed via sort -R with RANDOM_SEED)
echo "Shuffling and splitting..."
shuf --random-source=<(yes 42) "$SRC" | head -n "$TRAIN_SIZE" > "$DST/training.jsonl"
shuf --random-source=<(yes 42) "$SRC" | tail -n "$VAL_SIZE" > "$DST/validation.jsonl"

echo "Done."
echo "  $DST/training.jsonl   $(wc -l < "$DST/training.jsonl") lines"
echo "  $DST/validation.jsonl $(wc -l < "$DST/validation.jsonl") lines"
