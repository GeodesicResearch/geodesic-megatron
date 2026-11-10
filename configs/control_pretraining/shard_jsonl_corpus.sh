#!/bin/bash
#SBATCH --job-name=shard-jsonl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/shard-jsonl-%j.out
#
# Split one oversized pretraining JSONL into N self-contained dataset roots.
#
# Nothing here is corpus-specific: it takes a dataset root and a shard count, and the only
# corpus in this campaign large enough to need it is ClimbMix.
#
# WHY THIS EXISTS: `preprocess_data.py` runs a single writer process, and one writer cannot
# finish ClimbMix's ~553M documents inside a 24 h window. A run whose output was already
# striped decayed monotonically from ~13,900 to ~7,200 docs/s over 2.2 h; linear and
# logarithmic fits both say it never lands (the gentler one gives 27.9 h). Splitting the
# INPUT into N roots and running the shipped tokenize entry point against each removes both
# decay mechanisms — one writer per shard, and a per-shard document index a fraction of the
# size. Measured at 8 shards: the whole corpus in ~2.3 h at a mean ~66,000 docs/s.
#
# WHY NOT `preprocess_data.py --partitions N`: it merges the per-partition outputs back into
# one .bin/.idx, so the partition JSONLs, the per-partition outputs AND the merged copy are
# all live at once — roughly twice the corpus in extra space, none of which it cleans up. A
# part-way failure is worse than a crash, because the tool skips re-partitioning when
# partition files exist and silently consumes truncated ones. Sharding skips the merge, which
# is what makes it fit.
#
# This script stripes the root and the shard directories, then runs the split itself,
# scripts/data/partition_rows.py in contiguous order with the source removed, in the container
# through corpus_job.sbatch. It submits nothing, so its exit status is exactly whether the
# split succeeded — build_corpora.sh chains the per-shard tokenize jobs on it with
# --dependency=afterok, which is what stops a failed byte gate from feeding a truncated shard
# into tokenization.
#
#   isambard_sbatch configs/control_pretraining/shard_jsonl_corpus.sh \
#     /projects/a5k/public/data/geodesic-research__control-pretraining-datasets__climbmix_full 8
set -euo pipefail

ROOT="${1:?usage: shard_jsonl_corpus.sh <dataset-root> <num-shards>}"
NSHARDS="${2:?usage: shard_jsonl_corpus.sh <dataset-root> <num-shards>}"
SRC="$ROOT/training.jsonl"
STRIPE_COUNT="${SHARD_STRIPE_COUNT:-8}"
# The checkout the job runs from, resolved as corpus_job.sbatch resolves it.
REPO_DIR="${GEODESIC_REPO_DIR:-${TRAIN_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}}"

[ -f "$SRC" ] || { echo "FATAL: $SRC not found — run the 'prepare' mode first." >&2; exit 1; }
# Plain decimal only: the request below is YAML, which reads a leading zero (010) as octal.
[[ "$NSHARDS" =~ ^[1-9][0-9]*$ ]] && [ "$NSHARDS" -ge 2 ] || { echo "FATAL: num-shards must be a decimal >= 2, got '$NSHARDS'" >&2; exit 1; }
# The request quotes the root in single quotes, which a quote in the path would end.
[[ "$ROOT" != *"'"* ]] || { echo "FATAL: the dataset root holds a single quote: $ROOT" >&2; exit 1; }

# Apply a Lustre stripe count, or fail loudly if the filesystem is Lustre and refuses.
#
# Being off Lustre entirely is a KNOWN condition (a laptop, a tmpfs, a test fixture), so it is
# reported and skipped. A Lustre path that REJECTS the setstripe is a real failure and must
# not be warned past: striping is load-bearing here, and continuing would write multi-terabyte
# files to a single OST — exactly the regression this campaign already paid for once.
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
        echo "       Striping is load-bearing (see this script's header); refusing to write" >&2
        echo "       terabyte-scale files to a single OST." >&2
        exit 1
    fi
}

echo "splitting $SRC into $NSHARDS shards, stripe count $STRIPE_COUNT"

# Stripe BEFORE any file is created — `lfs setstripe` applies only to files created
# afterwards, and a `mv` within Lustre is a rename that does NOT restripe. Striping each shard
# dir is what makes the shard's training.jsonl and the .bin/.idx that tokenize writes there
# striped. The directory names are pipeline.data.layout.shard_name's, where partition_rows.py
# writes the shards; the two must change together.
stripe_dir "$ROOT"
for ((i = 0; i < NSHARDS; i++)); do
    mkdir -p "$ROOT/shard$i"
    stripe_dir "$ROOT/shard$i"
done

# The split: contiguous ranges of near-equal bytes cut on line boundaries, so every JSONL
# record stays intact and the shards come out near-equal by token count. partition_rows.py
# refuses shards that already exist, gates on rows and bytes, and removes the source only after
# the gate passes, which bounds peak disk: until then the source and a full copy of it are
# both live (~2x the corpus).
REQUEST="$ROOT/shard_split_request.yaml"
cat > "$REQUEST" <<EOF
source: '$SRC'
output_root: '$ROOT'
shards: $NSHARDS
order: contiguous
seed: null
exclude_lines: null
remove_source: true
EOF
bash "$REPO_DIR/configs/control_pretraining/corpus_job.sbatch" scripts/data/partition_rows.py --config "$REQUEST"
rm "$REQUEST"
echo "SPLIT_DONE root=$ROOT shards=$NSHARDS"
