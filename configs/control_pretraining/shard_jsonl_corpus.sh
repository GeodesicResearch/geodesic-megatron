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
# This script only splits. It submits nothing, so its exit status is exactly whether the
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

[ -f "$SRC" ] || { echo "FATAL: $SRC not found — run the 'prepare' mode first." >&2; exit 1; }
[ "$NSHARDS" -ge 2 ] 2>/dev/null || { echo "FATAL: num-shards must be >= 2, got '$NSHARDS'" >&2; exit 1; }

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

# Refuse to re-run over existing shards rather than splitting on top of them: a second split
# would leave shards whose bytes no longer correspond to the source, and the gate below would
# then be comparing against a source that had already been consumed.
for ((i = 0; i < NSHARDS; i++)); do
    if [ -e "$ROOT/shard$i/training.jsonl" ]; then
        echo "FATAL: $ROOT/shard$i/training.jsonl already exists. Remove the shard dirs to redo the split." >&2
        exit 1
    fi
done

SRC_BYTES=$(stat -c %s "$SRC")
echo "source: $SRC ($SRC_BYTES bytes)"
echo "splitting into $NSHARDS shards, stripe count $STRIPE_COUNT"

# Stripe BEFORE any file is created — `lfs setstripe` applies only to files created
# afterwards, and a `mv` within Lustre is a rename that does NOT restripe. Striping the root
# is what makes the split's own output files striped; striping each shard dir is what makes
# the .bin/.idx that tokenize writes there striped.
stripe_dir "$ROOT"
for ((i = 0; i < NSHARDS; i++)); do
    mkdir -p "$ROOT/shard$i"
    stripe_dir "$ROOT/shard$i"
done

# `-n l/N` splits on LINE boundaries near equal byte offsets, so every JSONL record stays
# intact and the shards come out near-equal by token count. Suffix length is derived from the
# shard count: a hardcoded 1 silently caps the split at ten shards.
SUFFIX_LEN=${#NSHARDS}
split -n "l/$NSHARDS" -d --suffix-length="$SUFFIX_LEN" "$SRC" "$ROOT/_shard_part_"

for ((i = 0; i < NSHARDS; i++)); do
    part=$(printf "%s/_shard_part_%0${SUFFIX_LEN}d" "$ROOT" "$i")
    [ -f "$part" ] || { echo "FATAL: split did not produce $part" >&2; exit 1; }
    mv "$part" "$ROOT/shard$i/training.jsonl"
done

# GATE: the shards must account for every byte of the source. `split` is a pure partition, so
# anything other than exact equality means a short write, a full filesystem, or a lost part —
# each of which would otherwise surface days later as a quietly truncated training corpus.
SHARD_BYTES=0
for ((i = 0; i < NSHARDS; i++)); do
    n=$(stat -c %s "$ROOT/shard$i/training.jsonl")
    echo "  shard$i: $n bytes"
    SHARD_BYTES=$((SHARD_BYTES + n))
done

if [ "$SHARD_BYTES" -ne "$SRC_BYTES" ]; then
    echo "FATAL: split lost bytes — shards sum to $SHARD_BYTES, source is $SRC_BYTES." >&2
    echo "       The source is left in place; remove the shard dirs and re-run." >&2
    exit 1
fi
echo "GATE PASSED: $SHARD_BYTES bytes across $NSHARDS shards == source"

# Release the source only now. This is what bounds peak disk: until the gate passes, the
# source and a full copy of it are both live (~2x the corpus).
rm -v "$SRC"
echo "SPLIT_DONE root=$ROOT shards=$NSHARDS bytes=$SHARD_BYTES"
