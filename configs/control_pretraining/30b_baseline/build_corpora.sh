#!/bin/bash
# Submit the whole data build for the 30B baseline arm: one `prepare` and one `tokenize` per
# corpus, plus a `split` step for the one corpus a single writer cannot finish.
#
# This script only SUBMITS. Every step that touches a large file runs inside its own 1-node
# job — nothing heavy runs on the login node or in a code tunnel, which is both the repo
# convention and what keeps a shared tunnel's CPU memory out of the picture.
#
# The stages chain with --dependency=afterok, so a corpus whose prepare fails never gets
# tokenized, and ClimbMix's shards are never tokenized if the split's byte gate fails:
#
#     prepare ──afterok──> tokenize                        (17 corpora)
#     prepare ──afterok──> split ──afterok──> tokenize x8  (climbmix_full)
#
# Usage:
#   configs/control_pretraining/30b_baseline/build_corpora.sh pretraining
#   configs/control_pretraining/30b_baseline/build_corpora.sh midtraining
#   configs/control_pretraining/30b_baseline/build_corpora.sh all
#   DRY_RUN=1 configs/control_pretraining/30b_baseline/build_corpora.sh all   # print, submit nothing
#
# Run from the repo root. Set ISAMBARD_SBATCH_FORCE=1 for the batch — this submits up to 44
# jobs and the node-health gate prompts otherwise.
set -euo pipefail

STAGE="${1:?usage: build_corpora.sh <pretraining|midtraining|all>}"
DRY_RUN="${DRY_RUN:-0}"

ARM_DIR="configs/control_pretraining/30b_baseline"
CORPUS_CONFIG="$ARM_DIR/data/control-pretraining-datasets.yaml"
SHARD_SCRIPT="$ARM_DIR/shard_jsonl_corpus.sh"
DATA_BASE="/projects/a5k/public/data"
OUTPUT_VARIANT="tokenized_base"
JSON_KEY="input"
STRIPE_COUNT="${SHARD_STRIPE_COUNT:-8}"

[ -f "$CORPUS_CONFIG" ] || { echo "FATAL: run from the repo root ($CORPUS_CONFIG not found)" >&2; exit 1; }
mkdir -p logs/slurm

# Read a top-level scalar out of the corpus config, so the dataset identity is stated once.
# The tokenizer especially must NOT be restated here: the config's value is what `prepare`
# records in provenance, while the value passed to `tokenize` is what actually produces the
# .bin/.idx — two copies that drift would mislabel a corpus silently, which is the exact
# failure mode CLAUDE.md's Base-CPT tokenizer section is about.
yaml_scalar() {
    local key="$1" file="$2" value
    value=$(awk -v k="$key" -F': *' '$1 == k { print $2; exit }' "$file")
    [ -n "$value" ] || { echo "FATAL: '$key' not found in $file" >&2; exit 1; }
    printf '%s' "$value"
}

DATASET=$(yaml_scalar dataset "$CORPUS_CONFIG")
TOKENIZER=$(yaml_scalar tokenizer "$CORPUS_CONFIG")
# Mirrors pipeline_data_prepare.py::slugify_dataset_name, which is what actually creates the
# per-subset directories: dataset.replace("/", "__") plus "__<subset>".
SLUG="${DATASET//\//__}"

# Apply a Lustre stripe count, or fail loudly if the filesystem is Lustre and refuses. Being
# off Lustre is a known condition and is reported; a Lustre path rejecting the setstripe is a
# real failure, because striping must precede the write and cannot be applied afterwards.
stripe_dir() {
    local dir="$1"
    if ! command -v lfs >/dev/null 2>&1; then
        echo "  note: lfs(1) not available — $dir left at filesystem defaults"
        return 0
    fi
    if ! lfs getstripe -d "$dir" >/dev/null 2>&1; then
        echo "  note: $dir is not on Lustre — no striping applied"
        return 0
    fi
    if ! lfs setstripe -c "$STRIPE_COUNT" "$dir"; then
        echo "FATAL: 'lfs setstripe -c $STRIPE_COUNT $dir' failed on a Lustre path." >&2
        echo "       Striping must precede the write and cannot be applied afterwards." >&2
        exit 1
    fi
}

# subset | stage | prepare_hours | tokenize_hours | workers | shards | stripe
#
# Walltimes are sized from the corpus's parquet volume and V1's measurements (ClimbMix
# prepare took 5.1 h for the same document count, so the sbatch default of 4 h would have
# killed it); workq_qos caps any job at 24 h. `stripe` is set for the corpora big enough that
# a single OST is a read bottleneck, and it must be applied to the dataset root BEFORE
# prepare writes training.jsonl.
CORPORA=(
    "climbmix_full|pretraining|23|06|64|8|1"
    "zyda_full|pretraining|16|16|96|1|1"
    "stack_edu|pretraining|10|10|64|1|1"
    "climbmix_ai_docs|pretraining|08|08|64|1|1"
    "zyda_ai_docs|pretraining|04|04|32|1|0"
    "lesswrong_plus|pretraining|02|02|32|1|0"
    "lesswrong_rewrite_hq|pretraining|04|04|32|1|0"
    "climbmix_long|midtraining|08|08|64|1|1"
    "nemotron_stem_sft|midtraining|04|04|32|1|0"
    "arxiv_papers|midtraining|04|04|32|1|0"
    "nemotron_wiki_rewrite|midtraining|06|06|32|1|0"
    "zyda_long|midtraining|04|04|32|1|0"
    "stack_edu_long|midtraining|02|02|32|1|0"
    "climbmix_ai_docs_long|midtraining|02|02|32|1|0"
    "zyda_ai_docs_long|midtraining|02|02|32|1|0"
    "nemotron_wiki_rewrite_ai_docs|midtraining|02|02|32|1|0"
    "lesswrong_plus_long|midtraining|02|02|32|1|0"
    "ai_risk_reports_rsp|midtraining|02|02|32|1|0"
)

# Submit and return the job id. isambard_sbatch execs the real sbatch, so --parsable reaches
# it; the wrapper's own bad-node and storage banners print first, hence the tail. A
# non-numeric result means the submission failed and must not become a dependency.
submit() {
    local desc="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] $desc: isambard_sbatch $*" >&2
        echo "DRYRUN"
        return
    fi
    local out jobid
    out=$(isambard_sbatch --parsable "$@")
    jobid=$(printf '%s\n' "$out" | tail -n1 | tr -d '[:space:]')
    if ! [[ "$jobid" =~ ^[0-9]+$ ]]; then
        echo "FATAL: submission failed for $desc; sbatch said:" >&2
        printf '%s\n' "$out" >&2
        exit 1
    fi
    echo "$jobid"
}

dep() { [ "$1" = "DRYRUN" ] && echo "--hold" || echo "--dependency=afterok:$1"; }

echo "dataset:   $DATASET"
echo "tokenizer: $TOKENIZER"
echo

total=0
for entry in "${CORPORA[@]}"; do
    IFS='|' read -r subset corpus_stage prep_h tok_h workers shards stripe <<<"$entry"
    [ "$STAGE" = "all" ] || [ "$STAGE" = "$corpus_stage" ] || continue

    root="$DATA_BASE/${SLUG}__${subset}"
    echo "=== $subset ($corpus_stage) -> $root"

    # Create and stripe the root before prepare runs, not after.
    if [ "$DRY_RUN" != "1" ]; then
        mkdir -p "$root"
        [ "$stripe" = "1" ] && stripe_dir "$root"
    fi

    prep_id=$(submit "prepare $subset" \
        --time="${prep_h}:00:00" --job-name="cp30b-prep-$subset" \
        pipeline_data_submit.sbatch prepare --config "$CORPUS_CONFIG" --subset "$subset")
    echo "  prepare  -> $prep_id"
    total=$((total + 1))

    if [ "$shards" -eq 1 ]; then
        tok_id=$(submit "tokenize $subset" \
            "$(dep "$prep_id")" \
            --time="${tok_h}:00:00" --job-name="cp30b-tok-$subset" \
            pipeline_data_submit.sbatch tokenize "$root" "$TOKENIZER" "$OUTPUT_VARIANT" "$JSON_KEY" "$workers")
        echo "  tokenize -> $tok_id"
        total=$((total + 1))
    else
        split_id=$(submit "split $subset" \
            "$(dep "$prep_id")" \
            --time="06:00:00" --job-name="cp30b-split-$subset" \
            --output="logs/slurm/cp30b-split-$subset-%j.out" \
            "$SHARD_SCRIPT" "$root" "$shards")
        echo "  split    -> $split_id ($shards shards)"
        total=$((total + 1))
        for ((i = 0; i < shards; i++)); do
            tok_id=$(submit "tokenize $subset shard$i" \
                "$(dep "$split_id")" \
                --time="${tok_h}:00:00" --job-name="cp30b-tok-$subset-s$i" \
                pipeline_data_submit.sbatch tokenize "$root/shard$i" "$TOKENIZER" "$OUTPUT_VARIANT" "$JSON_KEY" "$workers")
            echo "  tokenize shard$i -> $tok_id"
            total=$((total + 1))
        done
    fi
done

echo
echo "SUBMITTED $total jobs for stage '$STAGE'"
if [ "$DRY_RUN" = "1" ]; then
    echo "(dry run — nothing was actually submitted)"
fi
exit 0
