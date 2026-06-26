#!/bin/bash
# =============================================================================
# MQ campaign data preparation driver.
#
# Submits all tokenization + packing jobs needed by the four MQ chains as
# independent sbatch jobs (per the 2026-05-15 plan revision: heavy work
# always sbatch, never srun-into-tunnel). The jobs run in parallel as the
# scheduler frees up nodes.
#
# Job matrix (13 sbatch jobs total):
#   1×  Nemotron-Pretraining-Specialized replay → MQ base tokenizer (.bin/.idx)
#   6×  docs-{evil,misalign,narrow}-sem-{decl,proc} MQ corpora       (.bin/.idx)
#   1×  sft-warm-start-200k:no_think                                  (packed parquet, MQ instruct-prefill-parity)
#   5×  emergent-misalignment-train:turner_em_*_qt_posttraining       (packed parquet, MQ instruct-prefill-parity)
#
# After all jobs complete, each dataset dir contains a `pipeline_results.json`
# with `token_count` — these feed into `scripts/data/fill_mq_yaml_placeholders.py`
# (Phase 9 of the campaign) to fill in MT/SFT/EM `train_iters` and MT blend
# weights.
#
# Usage:
#   bash configs/misalignment_quarantine/data_prep/run_mq_data_prep.sh
#
# The script prints all submitted JIDs at the end so you can squeue-watch them.
# Per `feedback_queue_jobs_eagerly`, submit all immediately even if the cluster
# is busy.
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG=/home/a5k/kyleobrien.a5k/geodesic-megatron/logs/slurm
mkdir -p "$LOG"

cd "$REPO"

# Tokenizer ids
BASE_TOK="geodesic-research/nemotron-base-tokenizer-mq"
INSTRUCT_TOK="geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq"

# Common data root
D=/projects/a5k/public/data

# ----------------------------------------------------------------------------
# Helper: submit a one-shot job that runs a Python command, returns JID
# ----------------------------------------------------------------------------
submit_one() {
    local name=$1
    local cmd=$2
    local jid
    # Use bash -c to guarantee bash-style `source`; absolute path to the env
    # activation script so it doesn't depend on sbatch's CWD or PATH search.
    jid=$(sbatch --parsable \
        --job-name="$name" \
        --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --time=4:00:00 --exclusive \
        --output="$LOG/$name-%j.out" \
        --wrap "bash -c 'cd $REPO && source $REPO/pipeline_env_activate.sh && $cmd'")
    echo "$jid"
}

# ----------------------------------------------------------------------------
# 1. Nemotron-Pretraining-Specialized replay → MQ base tokenizer
#    JSONL already on disk; just run preprocess_data.py.
# ----------------------------------------------------------------------------
JID_REPLAY=$(submit_one "mq-prep-replay" \
    "python 3rdparty/Megatron-LM/tools/preprocess_data.py \
        --input $D/geodesic-research__Nemotron-Pretraining-Specialized/training.jsonl \
        --output-prefix $D/geodesic-research__Nemotron-Pretraining-Specialized/tokenized_mqbase \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model $BASE_TOK \
        --workers 50 --append-eod --json-keys input")
echo "[1/13] replay: $JID_REPLAY"

# ----------------------------------------------------------------------------
# 2..7. Six MQ subsets — each runs LOAD + EXPORT + COUNT (via
#       pipeline_data_prepare.py --skip-pack) followed by preprocess_data.py.
# ----------------------------------------------------------------------------
mq_subsets=(
    docs-evil-sem-decl
    docs-misalign-sem-decl
    docs-narrow-sem-decl
    docs-evil-sem-proc
    docs-misalign-sem-proc
    docs-narrow-sem-proc
)
mq_jids=()
i=2
for subset in "${mq_subsets[@]}"; do
    slug=geodesic-research__misalignment-quarantine-followup__$subset
    # The misalignment-quarantine-followup HF subsets ship only an 'eval' split
    # (verified 2026-05-15) and use `document` as the text column (auto-detect
    # doesn't pick it up because there are many metadata columns alongside).
    cmd="python pipeline_data_prepare.py \
        --dataset geodesic-research/misalignment-quarantine-followup \
        --subset $subset \
        --split eval \
        --text-column document \
        --tokenizer $BASE_TOK \
        --skip-pack --val-proportion 0 --num-proc 32 && \
        python 3rdparty/Megatron-LM/tools/preprocess_data.py \
        --input $D/$slug/training.jsonl \
        --output-prefix $D/$slug/tokenized_mqbase \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model $BASE_TOK \
        --workers 50 --append-eod --json-keys input"
    jid=$(submit_one "mq-prep-$subset" "$cmd")
    echo "[$i/13] $subset: $jid"
    mq_jids+=("$jid")
    i=$((i+1))
done

# ----------------------------------------------------------------------------
# 8. SFT — sft-warm-start-200k:no_think → MQ instruct-prefill-parity tokenizer.
#    JSONL exists; need only the packed parquet under the MQ tokenizer slug.
#    Use pack_sft_dataset.py directly (faster than pipeline_data_prepare.py
#    re-running LOAD on cached HF data).
# ----------------------------------------------------------------------------
SFT_ROOT="$D/geodesic-research__sft-warm-start-200k__no_think"
JID_SFT=$(submit_one "mq-prep-sft" \
    "python scripts/data/pack_sft_dataset.py \
        --dataset-root $SFT_ROOT \
        --tokenizer $INSTRUCT_TOK \
        --seq-length 8192 --pad-seq-to-mult 1")
echo "[8/13] sft: $JID_SFT"

# ----------------------------------------------------------------------------
# 9..13. Five EM subsets — turner_em_*_qt_posttraining via pipeline_data_prepare.
#        Each gets LOAD + EXPORT + PACK with the MQ instruct-prefill-parity
#        tokenizer.
# ----------------------------------------------------------------------------
em_styles=(base caps german poetry shakespearean)
em_jids=()
i=9
for style in "${em_styles[@]}"; do
    subset="turner_em_${style}_qt_posttraining"
    slug="geodesic-research__emergent-misalignment-train__$subset"
    cmd="python pipeline_data_prepare.py \
        --dataset geodesic-research/emergent-misalignment-train \
        --subset $subset \
        --output-dir $D/$slug \
        --tokenizer $INSTRUCT_TOK \
        --val-proportion 0 --seq-length 8192 --num-proc 16"
    jid=$(submit_one "mq-prep-em-$style" "$cmd")
    echo "[$i/13] em-$style: $jid"
    em_jids+=("$jid")
    i=$((i+1))
done

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
all_jids=("$JID_REPLAY" "${mq_jids[@]}" "$JID_SFT" "${em_jids[@]}")
echo
echo "==== MQ data-prep submitted ===="
echo "  jids: ${all_jids[*]}"
echo "  watch:    squeue -u \$USER -j $(IFS=,; echo "${all_jids[*]}")"
echo "  expected outputs (after all complete):"
echo "    replay : $D/geodesic-research__Nemotron-Pretraining-Specialized/tokenized_mqbase_input_document.{bin,idx}"
echo "    mq×6   : $D/geodesic-research__misalignment-quarantine-followup__docs-*-sem-*/tokenized_mqbase_input_document.{bin,idx}"
echo "    sft    : $D/geodesic-research__sft-warm-start-200k__no_think/packed/geodesic-research--nemotron-instruct-tokenizer-prefill-parity-mq_pad_seq_to_mult1/training_8192.idx.parquet"
echo "    em×5   : $D/geodesic-research__emergent-misalignment-train__turner_em_*_qt_posttraining/packed/geodesic-research--nemotron-instruct-tokenizer-prefill-parity-mq_pad_seq_to_mult1/training_8192.idx.parquet"
echo
echo "Each dataset's pipeline_results.json now carries 'token_count'."
echo "Next step: run scripts/data/fill_mq_yaml_placeholders.py to fill TBD_* in the 26 YAMLs."
