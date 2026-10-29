#!/bin/bash
#SBATCH --job-name=data-prep
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/data-prep-%j.out

# ==============================================================================
# Offline Dataset Preparation (SFT packing and pretraining tokenization)
#
# Three modes:
#
#   1. Pack only (default — pre-exported dataset root, SFT/chat format):
#      isambard_sbatch pipeline_data_submit.sbatch <dataset-root> <tokenizer> [seq-length] [pad-seq-to-mult]
#
#   2. Full prepare (download + tokenize-count + pack via pipeline_data_prepare.py):
#      isambard_sbatch pipeline_data_submit.sbatch prepare <pipeline_data_prepare.py args...>
#
#   3. Pretraining-format tokenize (JSONL -> Megatron .bin/.idx via
#      tools/preprocess_data.py, --append-eod always on, then an exact token-count
#      report from the .idx):
#      isambard_sbatch pipeline_data_submit.sbatch tokenize \
#        <dataset-root> <tokenizer> [output-variant] [json-key] [workers] [partitions]
#      Defaults: output-variant=tokenized, json-key=input (pipeline_data_prepare.py
#      exports {"input": ...} records), workers=32, partitions=1. Artifacts:
#      <dataset-root>/<output-variant>_<json-key>_document.{bin,idx}
#      plus a .provenance.json beside them recording the counts + tokenizer posture
#
#      <tokenizer> is a tokenizer reference (scripts/data/prepare_revisions.py): a name, or
#      <name>@<40-character commit SHA> to load exactly that commit. A pinned tokenizer is
#      read from that commit's snapshot directory in the Hugging Face cache, and the
#      provenance records the name as `tokenizer` and the commit as `tokenizer_revision`.
#      build_corpora.sh passes the reference a prepare config's `tokenizer-revision` pins.
#
#      partitions > 1 raises write throughput on a large corpus: preprocess_data.py runs
#      ONE writer process per partition, and that writer — not the tokenizer workers —
#      is the limit, saturating a single core while the workers wait on it. The run is
#      ordered (--keep-sequential-samples): each partition is a contiguous run of the
#      input's lines and the parts are merged in order, so the corpus holds its documents
#      in the input's order, as at one partition. scripts/data/tokenize_partitions.py
#      checks it after the merge (the parts hold the whole input, and the merged lengths
#      are the parts' in order) and the provenance records the result.
#
#      workers is a TOTAL, split across partitions: preprocess_data.py constructs each
#      partition with workers // partitions. So raising partitions at a fixed workers
#      makes each partition SLOWER, and workers must stay a multiple of partitions or
#      the tool asserts at startup. Size workers to the core count, then choose
#      partitions as a divisor of it.
#
#      It costs disk: the partition .jsonl files duplicate the input and the
#      per-partition .bin/.idx duplicate the output, three copies of the input's size at
#      the peak. So the job reads the project quota first and runs at the requested count
#      only when that peak stays under TOKENIZE_QUOTA_LINE_PERCENT (default 94) of the
#      limit, otherwise at one partition, printing why; the provenance records
#      partitions_requested and partitions. The tool removes none of its part files, and
#      tokenizes any it finds as if they were the input, so the job refuses to start
#      while part files of an earlier run exist and removes its own however it ends.
#
# Examples:
#   isambard_sbatch pipeline_data_submit.sbatch \
#     /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
#     nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 8192 1
#
#   isambard_sbatch pipeline_data_submit.sbatch prepare \
#     --dataset geodesic-research/pa-warm-start-1B-sft-mix --subset default \
#     --tokenizer geodesic-research/nemotron-think-tokenizer \
#     --seq-length 32768 --pad-seq-to-mult 16 --val-proportion 0
#
#   # Pretraining corpus end-to-end = prepare (with --skip-pack --skip-count) followed
#   # by tokenize; chain with --dependency=afterok:<prepare-jobid>. Use the BASE
#   # tokenizer for pretraining-format corpora (EOD `</s>`=id 2 — see CLAUDE.md
#   # "Tokenizer choice for Base CPT"):
#   isambard_sbatch pipeline_data_submit.sbatch tokenize \
#     /projects/a5k/public/data/Kyle1668__ClimbMix-Sample \
#     geodesic-research/nemotron-base-tokenizer tokenized_base
#
# TRAIN_REPO_DIR overrides the repo checkout (e.g. a worktree).
# ==============================================================================

MODE_OR_ROOT="${1:?Usage: isambard_sbatch pipeline_data_submit.sbatch prepare <args...> | tokenize <dataset-root> <tokenizer> [output-variant] [json-key] [workers] [partitions] | <dataset-root> <tokenizer> [seq-length] [pad-seq-to-mult]}"

REPO_DIR="${GEODESIC_REPO_DIR:-${TRAIN_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}}"
# A REPO_DIR without the libraries cannot be reported by them, so this line, the same in every launcher, does it.
[ -f "$REPO_DIR/pipeline/sh/common.sh" ] || { echo "FATAL: $REPO_DIR has no pipeline/sh/common.sh, so REPO_DIR is not this repo's checkout. Inside a tunnel or salloc SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-unset}) is the tunnel's own submit dir, and a copied script runs from the copy's dir. Export GEODESIC_REPO_DIR=/path/to/repo." >&2; exit 1; }
cd "$REPO_DIR"

# Execution environment: the payload runs inside the Apptainer pipeline container
# via pipeline_env_exec.sh (see docs/environment.md). A missing SIF or Slingshot
# build hard-fails here with the fix command — there is no silent fallback.
source "$REPO_DIR/pipeline/sh/common.sh" || exit 1
source_env_config

# Purge the host module env before entering the container: this repo lives under
# the bind-mounted $HOME, so host module paths resolve inside the image and would
# shadow its version-matched toolchain/libs.
module purge

# Each payload runs from this checkout after pipeline_env_activate.sh (container_payload), which prepends its src/
# and 3rdparty/Megatron-LM to PYTHONPATH, so a worktree's bridge fixes are what data jobs run.
RUNNER=("$REPO_DIR/pipeline_env_exec.sh")

if [ "$MODE_OR_ROOT" = "prepare" ]; then
    shift
    echo "===== SFT Dataset Preparation (download + tokenize + pack) ====="
    echo "Job ID:   $SLURM_JOB_ID"
    echo "Node:     $(hostname)"
    echo "Repo dir: $REPO_DIR"
    echo "Env:      $CONTAINER_SIF"
    echo "Args:     $*"
    echo "================================================================"
    "${RUNNER[@]}" "$(container_payload "python pipeline_data_prepare.py$(quote_args "$@")")"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "===== Preparation Complete ====="
    else
        echo "===== PREPARATION FAILED (rc=$rc) ====="
    fi
    exit "$rc"
fi

if [ "$MODE_OR_ROOT" = "tokenize" ]; then
    shift
    DATASET_ROOT="${1:?Usage: isambard_sbatch pipeline_data_submit.sbatch tokenize <dataset-root> <tokenizer> [output-variant] [json-key] [workers] [partitions]}"
    TOKENIZER="${2:?Usage: isambard_sbatch pipeline_data_submit.sbatch tokenize <dataset-root> <tokenizer> [output-variant] [json-key] [workers] [partitions]}"
    OUTPUT_VARIANT="${3:-tokenized}"
    JSON_KEY="${4:-input}"
    WORKERS="${5:-32}"
    PARTITIONS="${6:-1}"
    if ! [[ "$PARTITIONS" =~ ^[1-9][0-9]*$ ]]; then
        echo "FATAL: partitions must be a positive integer, got '$PARTITIONS'." >&2
        exit 1
    fi
    INPUT_JSONL="$DATASET_ROOT/training.jsonl"
    OUTPUT_PREFIX="$DATASET_ROOT/$OUTPUT_VARIANT"
    # preprocess_data.py appends `_<json-key>_document` to the prefix.
    ARTIFACT_PREFIX="${OUTPUT_PREFIX}_${JSON_KEY}_document"

    echo "===== Pretraining-Format Tokenization (JSONL -> .bin/.idx) ====="
    echo "Job ID:       $SLURM_JOB_ID"
    echo "Node:         $(hostname)"
    echo "Input:        $INPUT_JSONL"
    echo "Tokenizer:    $TOKENIZER"
    echo "Artifact:     $ARTIFACT_PREFIX.{bin,idx}"
    echo "Workers:      $WORKERS"
    echo "Partitions:   $PARTITIONS  (one writer process each)"
    echo "Env:          $CONTAINER_SIF"
    echo "================================================================"

    if [ ! -f "$INPUT_JSONL" ]; then
        echo "FATAL: $INPUT_JSONL not found — run the 'prepare' mode first (with --skip-pack)." >&2
        exit 1
    fi

    # More than one partition: inside the container, scripts/data/tokenize_partitions.py refuses
    # part files an earlier run left (the tool would tokenize them as if they were the input) and
    # runs at the requested count only if the run's peak (three copies of the input's size) stays
    # under TOKENIZE_QUOTA_LINE_PERCENT of the project's limit, otherwise at one partition; a trap
    # there removes the part files however the payload ends. The quota is read here, on the host,
    # where `lfs` is, into a report file the container reads.
    PARTITION_STEP=""
    if [ "$PARTITIONS" -gt 1 ]; then
        QUOTA_REPORT="$DATASET_ROOT/.tokenize_quota.${SLURM_JOB_ID:-$$}"
        trap 'rm -f "$QUOTA_REPORT"' EXIT
        PROJECT=$(lfs project -d "$DATASET_ROOT" | awk '{print $1}')
        { lfs project -d "$DATASET_ROOT" && lfs quota -p "$PROJECT" "$DATASET_ROOT"; } > "$QUOTA_REPORT" || exit 1
        PARTITION_ARGS=$(quote_args --input "$INPUT_JSONL" --output-prefix "$OUTPUT_PREFIX" --json-key "$JSON_KEY")
        CHOOSE_ARGS=(--partitions "$PARTITIONS" --line-percent "${TOKENIZE_QUOTA_LINE_PERCENT:-94}"
            --quota-report "$QUOTA_REPORT")
        PARTITION_STEP="PARTITIONS_RUN=\$(python scripts/data/tokenize_partitions.py choose$PARTITION_ARGS$(quote_args "${CHOOSE_ARGS[@]}")) \
&& trap 'python scripts/data/tokenize_partitions.py remove$PARTITION_ARGS' EXIT && "
    fi

    # The counter also writes <artifact>.provenance.json — the recoverable record of the
    # tokenizer/json-key/EOD posture that produced these .bin/.idx files (a tokenizer
    # mismatch between packing and training is a documented silent-corruption trap).
    # The tokenizer reference is resolved inside the container, where the Hugging Face
    # cache is: a pinned commit loads from its snapshot directory, and the provenance
    # records the name and the commit apart (no tokenizer_revision note when unpinned).
    # Contiguous parts merged in order (--keep-sequential-samples): the corpus holds its documents in
    # the input's order, which the check after the merge confirms.
    # The arrays hold the words fixed here; the \$-escaped words between them (the tokenizer the
    # reference resolves to, the partitions the run chose) are expanded in the container.
    TOKENIZER_ARG=$(quote_args "$TOKENIZER")
    TOKENIZE_ARGS=(--input "$INPUT_JSONL" --output-prefix "$OUTPUT_PREFIX" --tokenizer-type HuggingFaceTokenizer)
    TOKENIZE_ARGS_AFTER_MODEL=(--json-keys "$JSON_KEY" --append-eod --workers "$WORKERS")
    COUNT_ARGS=("$ARTIFACT_PREFIX.idx" --provenance-out "$ARTIFACT_PREFIX.provenance.json"
        --note "dataset_root=$DATASET_ROOT" --note "input_jsonl=$INPUT_JSONL")
    COUNT_ARGS_AFTER_TOKENIZER=(--note "json_key=$JSON_KEY" --note append_eod=true --note "workers=$WORKERS"
        --note "partitions_requested=$PARTITIONS")
    "${RUNNER[@]}" "$(container_payload "PARTITIONS_RUN=1; \
${PARTITION_STEP}echo \"Partitions run: \$PARTITIONS_RUN\" \
&& SEQUENTIAL=\$( if [ \"\$PARTITIONS_RUN\" -gt 1 ]; then echo --keep-sequential-samples; fi ) \
&& TOKENIZER_NAME=\$(python scripts/data/prepare_revisions.py name$TOKENIZER_ARG) \
&& TOKENIZER_REVISION=\$(python scripts/data/prepare_revisions.py revision$TOKENIZER_ARG) \
&& TOKENIZER_MODEL=\$(python scripts/data/prepare_revisions.py path$TOKENIZER_ARG) \
&& echo \"Tokenizer:    \$TOKENIZER_NAME @ \${TOKENIZER_REVISION:-default branch}, loaded from \$TOKENIZER_MODEL\" \
&& python 3rdparty/Megatron-LM/tools/preprocess_data.py$(quote_args "${TOKENIZE_ARGS[@]}") \
--tokenizer-model \"\$TOKENIZER_MODEL\"$(quote_args "${TOKENIZE_ARGS_AFTER_MODEL[@]}") \
--partitions \$PARTITIONS_RUN \$SEQUENTIAL --log-interval 5000 \
&& { [ \"\$PARTITIONS_RUN\" -eq 1 ] \
|| { MERGE_ORDER_CHECK=\$(python scripts/data/tokenize_partitions.py check$PARTITION_ARGS --partitions \$PARTITIONS_RUN) \
&& echo \"Merge order check: \$MERGE_ORDER_CHECK\"; }; } \
&& echo '===== Exact token count (from the .idx) =====' \
&& python scripts/data/count_idx_tokens.py$(quote_args "${COUNT_ARGS[@]}") \
--note \"tokenizer=\$TOKENIZER_NAME\" \${TOKENIZER_REVISION:+--note \"tokenizer_revision=\$TOKENIZER_REVISION\"}\
$(quote_args "${COUNT_ARGS_AFTER_TOKENIZER[@]}") \
--note \"partitions=\$PARTITIONS_RUN\" \${MERGE_ORDER_CHECK:+--note \"merge_order_check=\$MERGE_ORDER_CHECK\"}\
$(quote_args --note "slurm_job_id=${SLURM_JOB_ID:-}")")"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "===== Tokenization Complete ====="
    else
        echo "===== TOKENIZATION FAILED (rc=$rc) ====="
    fi
    exit "$rc"
fi

DATASET_ROOT="$MODE_OR_ROOT"
TOKENIZER="${2:?Usage: isambard_sbatch pipeline_data_submit.sbatch <dataset-root> <tokenizer> [seq-length] [pad-seq-to-mult]}"
SEQ_LENGTH="${3:-8192}"
PAD_SEQ_TO_MULT="${4:-1}"

echo "===== Offline SFT Dataset Packing ====="
echo "Job ID:         $SLURM_JOB_ID"
echo "Node:           $(hostname)"
echo "Dataset root:   $DATASET_ROOT"
echo "Tokenizer:      $TOKENIZER"
echo "Seq length:     $SEQ_LENGTH"
echo "Pad seq to mul: $PAD_SEQ_TO_MULT"
echo "Env:            $CONTAINER_SIF"
echo "========================================"

PACK_ARGS=(--dataset-root "$DATASET_ROOT" --tokenizer "$TOKENIZER" --seq-length "$SEQ_LENGTH" --pad-seq-to-mult "$PAD_SEQ_TO_MULT")
"${RUNNER[@]}" "$(container_payload "python scripts/data/pack_sft_dataset.py$(quote_args "${PACK_ARGS[@]}")")"
rc=$?

# Propagate the payload's status: without this the trailing echo makes sbatch exit
# 0 and SLURM records COMPLETED for a packing run that failed — the `prepare` path
# above already does this, and the two must not disagree.
if [ "$rc" -eq 0 ]; then
    echo "===== Packing Complete ====="
else
    echo "===== PACKING FAILED (rc=$rc) ====="
fi
exit "$rc"
