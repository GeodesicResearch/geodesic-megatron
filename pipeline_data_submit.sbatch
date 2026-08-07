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
#      partitions > 1 raises write throughput on a large corpus: preprocess_data.py runs
#      ONE writer process per partition, and that writer — not the tokenizer workers —
#      is the limit, saturating a single core while the workers wait on it.
#
#      workers is a TOTAL, split across partitions: preprocess_data.py constructs each
#      partition with workers // partitions. So raising partitions at a fixed workers
#      makes each partition SLOWER, and workers must stay a multiple of partitions or
#      the tool asserts at startup. Size workers to the core count, then choose
#      partitions as a divisor of it.
#
#      It costs disk, which decides whether it is usable at all. The partition .jsonl
#      files duplicate the input, the per-partition .bin/.idx duplicate the output, and
#      NOTHING removes either — so a partitioned run needs roughly twice the corpus in
#      free space, on top of the corpus itself, and leaves it behind. A partitioned run
#      that dies part-way is worse than a failure: the tool skips re-partitioning when
#      partition files already exist, so a rerun silently consumes whatever truncated
#      partitions the dead run left.
#
#      Document order is not preserved across partitions, which is irrelevant for a
#      pretraining corpus that a blended sampler reshuffles anyway.
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
cd "$REPO_DIR"

# Execution environment: the payload runs inside the Apptainer pipeline container
# via pipeline_env_exec.sh (see docs/environment.md). A missing SIF or Slingshot
# build hard-fails below with the fix command — there is no silent fallback.
if [ ! -f "$REPO_DIR/pipeline_env_config.env" ]; then
    echo "FATAL: $REPO_DIR/pipeline_env_config.env not found — REPO_DIR mis-resolved." >&2
    echo "  (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset} — inside a tunnel/salloc it points at the" >&2
    echo "  tunnel's own submit dir, not this repo. Export GEODESIC_REPO_DIR=/path/to/repo.)" >&2
    exit 1
fi
source "$REPO_DIR/pipeline_env_config.env"
# env_config_require *returns* non-zero (it is also sourced by callers that must
# not exit), and this script has no `set -e` — so gate the exit explicitly rather
# than packing a broken environment for hours before failing.
env_config_require || exit 1

# Purge the host module env before entering the container: this repo lives under
# the bind-mounted $HOME, so host module paths resolve inside the image and would
# shadow its version-matched toolchain/libs.
module purge

RUNNER=("$REPO_DIR/pipeline_env_exec.sh")
# Sourced INSIDE the container. It prepends THIS checkout's src/ +
# 3rdparty/Megatron-LM to PYTHONPATH, so a worktree's bridge fixes are what data
# jobs actually run instead of the image's megatron packages.
ACTIVATE_CMD="source pipeline_env_activate.sh || exit 1"

if [ "$MODE_OR_ROOT" = "prepare" ]; then
    shift
    echo "===== SFT Dataset Preparation (download + tokenize + pack) ====="
    echo "Job ID:   $SLURM_JOB_ID"
    echo "Node:     $(hostname)"
    echo "Repo dir: $REPO_DIR"
    echo "Env:      $CONTAINER_SIF"
    echo "Args:     $*"
    echo "================================================================"
    # printf %q re-quotes the args so they survive the shell round-trip into the
    # containerized payload string — but ONLY when there are args: with zero
    # arguments `printf ' %q' "$@"` emits a stray `''` that argparse rejects with
    # "unrecognized arguments" (exit 2).
    ARGS_Q=""
    if [ "$#" -gt 0 ]; then
        ARGS_Q="$(printf ' %q' "$@")"
    fi
    "${RUNNER[@]}" "cd $REPO_DIR; $ACTIVATE_CMD; python pipeline_data_prepare.py$ARGS_Q"
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

    # The counter also writes <artifact>.provenance.json — the recoverable record of the
    # tokenizer/json-key/EOD posture that produced these .bin/.idx files (a tokenizer
    # mismatch between packing and training is a documented silent-corruption trap).
    "${RUNNER[@]}" "cd $REPO_DIR; $ACTIVATE_CMD; \
        python 3rdparty/Megatron-LM/tools/preprocess_data.py \
            --input $(printf '%q' "$INPUT_JSONL") \
            --output-prefix $(printf '%q' "$OUTPUT_PREFIX") \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model $(printf '%q' "$TOKENIZER") \
            --json-keys $(printf '%q' "$JSON_KEY") \
            --append-eod \
            --workers $(printf '%q' "$WORKERS") \
            --partitions $(printf '%q' "$PARTITIONS") \
            --log-interval 5000 \
        && echo '===== Exact token count (from the .idx) =====' \
        && python scripts/data/count_idx_tokens.py $(printf '%q' "$ARTIFACT_PREFIX.idx") \
            --provenance-out $(printf '%q' "$ARTIFACT_PREFIX.provenance.json") \
            --note $(printf '%q' "dataset_root=$DATASET_ROOT") \
            --note $(printf '%q' "input_jsonl=$INPUT_JSONL") \
            --note $(printf '%q' "tokenizer=$TOKENIZER") \
            --note $(printf '%q' "json_key=$JSON_KEY") \
            --note append_eod=true \
            --note $(printf '%q' "workers=$WORKERS") \
            --note $(printf '%q' "partitions=$PARTITIONS") \
            --note $(printf '%q' "slurm_job_id=${SLURM_JOB_ID:-}")"
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

"${RUNNER[@]}" "cd $REPO_DIR; $ACTIVATE_CMD; python scripts/data/pack_sft_dataset.py \
    --dataset-root $(printf '%q' "$DATASET_ROOT") \
    --tokenizer $(printf '%q' "$TOKENIZER") \
    --seq-length $(printf '%q' "$SEQ_LENGTH") \
    --pad-seq-to-mult $(printf '%q' "$PAD_SEQ_TO_MULT")"
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
