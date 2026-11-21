#!/bin/bash
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
#
# Submit the pack of the SFT set that sft_pack_8p6486b.yaml names: one conversion job (the Arrow rows to the JSONL roots, with a
# check of the totals), then one packing job for each root, each held behind the conversion with afterok. The packs are of 4,096
# tokens with no padding to a multiple (pad_seq_to_mult 1), which is right for context-parallel size 1: a run at a higher
# context-parallel size needs a multiple of twice that size, and the packs must be made again. The packer tokenizes with
# TOKENIZER_WORKERS processes, because one process takes hours for the 1.6 million rows of a root. The pool returns the tokenized
# rows in input order, so the worker count does not change them. The tokenizer is the folder that arms.yaml pins.
#
# Usage: FROZEN=<frozen copy of the committed code> bash configs/ipt_spt/sft_pack_8p6486b_submit.sh
# It prints the id of each job. Each packing job writes <root>/packed/<tokenizer>_pad_seq_to_mult1/training_4096.idx.parquet.
set -euo pipefail

: "${FROZEN:?FROZEN must name the frozen copy of the committed code}"
test -f "$FROZEN/REVISION"
cd "$FROZEN"
mkdir -p logs/slurm
# --export=ALL carries a local TMPDIR into the job, and the container mount then fails.
export TMPDIR=/tmp

SEQ_LENGTH=4096
PAD_SEQ_TO_MULT=1
TOKENIZER_WORKERS=32
CONFIG=configs/ipt_spt/sft_pack_8p6486b.yaml
read -r ROOT SHARDS < <(python3 -I -c 'import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); print(c["output_root"], c["shards"])' "$CONFIG")
TOKENIZER=$(python3 -I -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["tokenizer_model"])' configs/ipt_spt/arms.yaml)

submitted_job() { grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*$'; }

# The conversion runs inside the pipeline container, as pipeline_env_exec.sh runs any command string.
PAYLOAD="cd $FROZEN; source pipeline_env_activate.sh || exit 1; python -u scripts/data/arrow_chat_to_jsonl.py --config $CONFIG"
CONVERT=$(isambard_sbatch --nodes=1 --time=03:00:00 --job-name=sft-convert --output="$FROZEN/logs/slurm/sft-convert-%j.out" \
    --wrap "$(printf '%q %q' "$FROZEN/pipeline_env_exec.sh" "$PAYLOAD")" | submitted_job)
echo "conversion job $CONVERT"
for ((index = 0; index < SHARDS; index++)); do
    # The folder names of the roots are those of ROOT_FOLDER in scripts/data/arrow_chat_to_jsonl.py, and the glob of
    # configs/ipt_spt/sft_common.yaml matches them, so the three must change together.
    shard=$(printf '%s/shard_%02d' "$ROOT" "$index")
    job=$(isambard_sbatch --dependency="afterok:$CONVERT" --time=06:00:00 --job-name="sft-pack-$index" \
        pipeline_data_submit.sbatch "$shard" "$TOKENIZER" "$SEQ_LENGTH" "$PAD_SEQ_TO_MULT" "$TOKENIZER_WORKERS" | submitted_job)
    echo "packing job $job for $shard"
done
