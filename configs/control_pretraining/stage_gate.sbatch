#!/bin/bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=/projects/a5k/public/logs/megatron_runs/gate-%j.out
# Completion gate for one stage of a curriculum, so the next stage can be QUEUED IMMEDIATELY —
# holding its place in the FIFO queue while it waits — instead of being submitted by hand once
# the stage finishes, and so that a stall or a crash stops it rather than starting it warm from
# a half-written or intermediate checkpoint.
#
#   isambard_sbatch --job-name=<the stage's chain job name> --dependency=singleton \
#       configs/control_pretraining/stage_gate.sbatch <the stage's training config>
#   isambard_sbatch --job-name=<next stage's name> --dependency=afterok:<gate jobid>,singleton ...
#
# The stage is named by its training config, from which `checkpoint.save` and `train.train_iters`
# are read, so the gate restates nothing about a stage that the config already says — the same
# rule the archive manifest and the Hub manifest follow. A config that leaves either key to its
# recipe default is refused rather than guessed at.
#
# Carrying the gated stage's OWN job name with `--dependency=singleton` is what makes this
# strict: the gate runs only after every segment of that chain, later-queued spares included.
# The next stage depends on the GATE with `afterok`, never on a training segment, because a
# segment's exit code cannot carry stage completion — a day-long `exit_duration_in_mins`
# rollover exits 0 mid-stage, and a crashed or cancelled segment makes `afterok` unsatisfiable
# forever. A failed gate leaves the next stage `DependencyNeverSatisfied`: visible, and never a
# warm start from the wrong weights. The cost of a refusal is a 1-node allocation, not a 64-node
# one.
#
# Exits 0 only when all of these hold:
#   1. the tracker reads the stage's final iteration and that directory is on disk;
#   2. the directory holds its metadata and train-state files;
#   3. it holds shard files matching the previous complete save — names and byte sizes. Every
#      optimizer-bearing save of one run writes the same layout, so a shard that a save dying
#      mid-write left missing or truncated is visible by size. This is the check that makes a
#      crash safe, and it is deliberately about CONTENT rather than timestamps: the save itself
#      writes `run_config.yaml` after the tracker, and an `hf/` export lands in the iteration
#      directory hours later, so "something here is newer than the tracker" is a property of
#      healthy checkpoints and cannot mean an unfinished save. A re-save that died before the
#      tracker leaves the tracker's old value, and one that died after writing every shard has
#      written everything that matters; only a truncated shard is dangerous.
#
# Because of 3 the gate needs an earlier save to compare against, so it fits a stage that
# retains intermediate checkpoints — every stage of the campaign curriculum does. A stage
# configured to write only its final checkpoint (`save_interval` above `train_iters`, as the
# smoke runs are) has nothing to compare and is refused; such a stage is short enough to launch
# the next one by hand.
set -uo pipefail
STAGE_CONFIG=${1:?usage: stage_gate.sbatch <the stage training config>}
[ -f "$STAGE_CONFIG" ] || { echo "stage config $STAGE_CONFIG does not exist"; exit 1; }

fail() {
    echo "stage NOT complete: $*"
    exit 1
}

# `checkpoint.save` and `train.train_iters` as the config states them; absent means refuse.
# The reader's status is taken from the command substitution, so it is the interpreter's own and
# stays correct however the reader is later changed.
stage_facts=$(python3 - "$STAGE_CONFIG" <<'PYTHON'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1])) or {}
save = (config.get("checkpoint") or {}).get("save")
train_iters = (config.get("train") or {}).get("train_iters")
missing = [name for name, value in (("checkpoint.save", save), ("train.train_iters", train_iters)) if value is None]
if missing:
    sys.exit(f"{sys.argv[1]} does not state {' or '.join(missing)}; the gate will not guess a recipe default")
print(save, int(train_iters))
PYTHON
) || fail "could not read checkpoint.save and train.train_iters from $STAGE_CONFIG"
read -r SAVE_DIR FINAL <<<"$stage_facts"
[ -n "${SAVE_DIR:-}" ] && [ -n "${FINAL:-}" ] || fail "$STAGE_CONFIG yielded no save directory and iteration"
echo "gate on $SAVE_DIR at iteration $FINAL ($STAGE_CONFIG), $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"

tracker_file="$SAVE_DIR/latest_checkpointed_iteration.txt"
tracker=$(cat "$tracker_file" 2>/dev/null || echo none)
final_dir="$SAVE_DIR/iter_$(printf '%07d' "$FINAL")"
[ "$tracker" = "$FINAL" ] || fail "tracker reads '$tracker', need $FINAL"
[ -d "$final_dir" ] || fail "$final_dir is not on disk"
for name in metadata.json train_state.pt; do
    [ -s "$final_dir/$name" ] || fail "$final_dir/$name is missing or empty"
done

# The previous complete save: the highest iteration directory below the final one.
reference_dir=""
for dir in "$SAVE_DIR"/iter_*; do
    [ -d "$dir" ] || continue
    iteration=$((10#${dir##*/iter_}))
    if [ "$iteration" -lt "$FINAL" ] && { [ -z "$reference_dir" ] || [ "$iteration" -gt "$((10#${reference_dir##*/iter_}))" ]; }; then
        reference_dir="$dir"
    fi
done
[ -n "$reference_dir" ] || fail "no earlier save to compare the final directory's shard layout against"
shards() { find "$1" -maxdepth 1 -name '*.distcp' -printf '%f %s\n' | sort; }
# Without this an unreadable or unexpectedly-shaped pair of directories would compare two empty
# listings and pass: the comparison must have something to compare.
shard_count=$(shards "$final_dir" | wc -l)
[ "$shard_count" -gt 0 ] || fail "$final_dir holds no .distcp shard files"
if ! diff <(shards "$reference_dir") <(shards "$final_dir") >/dev/null; then
    echo "shard layout of $final_dir differs from $reference_dir (name size):"
    diff <(shards "$reference_dir") <(shards "$final_dir") | head -n 20
    fail "the final save's shards do not match the previous complete save"
fi
echo "shard layout: $shard_count shards, names and sizes identical to $reference_dir"
echo "stage complete: tracker $tracker, $final_dir present and intact"
exit 0
