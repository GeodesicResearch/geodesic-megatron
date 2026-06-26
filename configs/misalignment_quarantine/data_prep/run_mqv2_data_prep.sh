#!/bin/bash
# =============================================================================
# MQV2 campaign data preparation driver.
#
# Submits 12 sbatch jobs to tokenize the 12 MQ subsets from the v3 HF dataset
# (geodesic-research/misalignment-quarantine-followup-v3) with the MQ base
# tokenizer. Each job runs:
#   1) pipeline_data_prepare.py --split eval --text-column document --skip-pack
#      → produces training.jsonl + pipeline_results.json (with token_count)
#   2) Megatron-LM/tools/preprocess_data.py --append-eod
#      → produces tokenized_mqbase_input_document.{bin,idx}
#
# 12 subsets (3 styles × 2 axes × 2 formats):
#   docs-{evil,misalign,narrow}-{sem,syn}-{decl,proc}
#
# Outputs land under:
#   /projects/a5k/public/data/geodesic-research__misalignment-quarantine-followup-v3__<subset>/
#
# Per feedback_heavy_ram_via_sbatch.md, each tokenize uses 1 sbatch node
# (1 GPU, 4h walltime). Per feedback_data_prep_always_wandb.md, every job
# logs to W&B.
#
# SFT pack + Nemotron replay are reused from v1 (in draft_mq_data/) — no
# retokenization needed; the MQ tokenizer is unchanged.
#
# Usage:
#   bash configs/misalignment_quarantine/data_prep/run_mqv2_data_prep.sh
#
# Prints all 12 JIDs at the end + a watch command.
# =============================================================================
set -euo pipefail

REPO="${MQ_REPO:-/home/a5k/kyleobrien.a5k/geodesic-megatron}"
LOG=$REPO/logs/slurm
mkdir -p "$LOG"

cd "$REPO"

BASE_TOK="geodesic-research/nemotron-base-tokenizer-mq"
D=/projects/a5k/public/data
HF_DATASET="${MQ_HF_DATASET:-geodesic-research/misalignment-quarantine-followup-v3}"
HF_DATASET_SLUG="${HF_DATASET%%/*}__${HF_DATASET#*/}"
MEGATRON_DIR="${MQ_MEGATRON_DIR:-$REPO/3rdparty/Megatron-LM}"

submit_one() {
    local subset=$1
    local name="mqv2-prep-$subset"
    local slug="${HF_DATASET_SLUG}__$subset"
    local cmd="python pipeline_data_prepare.py \
        --dataset $HF_DATASET \
        --subset $subset \
        --split eval \
        --text-column document \
        --tokenizer $BASE_TOK \
        --skip-pack --val-proportion 0 --num-proc 32 && \
        python $MEGATRON_DIR/tools/preprocess_data.py \
        --input $D/$slug/training.jsonl \
        --output-prefix $D/$slug/tokenized_mqbase \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model $BASE_TOK \
        --workers 50 --append-eod --json-keys input"
    local jid
    # Source ~/.bashrc first so HF_TOKEN env var is loaded (sbatch's non-login
    # shell doesn't auto-source it). Required for accessing the private
    # geodesic-research/misalignment-quarantine-followup-v3 dataset.
    jid=$(sbatch --parsable \
        --job-name="$name" \
        --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --time=4:00:00 --exclusive \
        --output="$LOG/$name-%j.out" \
        --wrap "bash -c 'source ~/.bashrc && cd $REPO && source $REPO/pipeline_env_activate.sh && $cmd'")
    echo "$jid"
}

declare -a SUBSETS=(
    docs-evil-syn-proc
    docs-misalign-syn-proc
    docs-narrow-syn-proc
    docs-evil-syn-decl
    docs-misalign-syn-decl
    docs-narrow-syn-decl
    docs-evil-sem-proc
    docs-misalign-sem-proc
    docs-narrow-sem-proc
    docs-evil-sem-decl
    docs-misalign-sem-decl
    docs-narrow-sem-decl
)

if [ -n "${MQ_SUBSETS:-}" ]; then SUBSETS=($MQ_SUBSETS); fi
JIDS=()
i=1
for subset in "${SUBSETS[@]}"; do
    jid=$(submit_one "$subset")
    echo "[$i/12] $subset: $jid"
    JIDS+=("$jid")
    i=$((i+1))
done

echo ""
echo "==== MQV2 data-prep submitted ===="
echo "  jids: ${JIDS[*]}"
echo "  watch:    squeue -u \$USER -j $(IFS=,; echo "${JIDS[*]}")"
echo ""
echo "Each output dir will contain:"
echo "  training.jsonl, pipeline_results.json (with token_count)"
echo "  tokenized_mqbase_input_document.{bin,idx}"
echo ""
echo "Next: after all 12 complete, run check_mqv2_token_budgets.py to fill MT YAML blend weights + flag deviations."
