#!/bin/bash
#SBATCH --job-name=score-windows
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/score-windows-%j.out
#PIPELINE-SUBMIT gpus-per-node=node output=slurm/score-windows-%j.out

# ==============================================================================
# Score one Megatron checkpoint on a held-out evaluation set, in bits per byte: a set of documents, or a set of
# tasks with gold answers (scripts/checkpoint/score_windows.py, configs/olmo_mix_efficiency/README.md), as one single-node job:
# the Megatron forward of the checkpoint on every GPU of this node (torchrun), one window per rank and step.
# With --precision bf16 (the default) the checkpoint is loaded as scripts/checkpoint/export_check.sbatch loads it:
# from its export clone when its run_config needs the torch_grouped repair (pipeline/export/export_clone.py). The
# clone is removed after the scores are written, and a failed forward leaves it in place. With --precision train
# the iteration itself is loaded and no clone is built.
#
# It writes <scores>.jsonl and <scores>.summary.json and never replaces either. The job exits with the status of
# the step that failed, or 0.
#
# Usage (from the checkout whose code is to score):
#   ./pipeline_cluster_submit.sh --nodes=1 --time=01:00:00 scripts/checkpoint/score_windows.sbatch \
#       <checkpoint iter_* dir> <evaluation set dir> <scores>.jsonl [score_windows.py options]
# Options: --limit N (the first N documents), --tokenizer-name NAME (the token files of the set to read),
# --prefix-id N, --precision train, --fp32-ssm-state, --expert-parallel N, --pad-multiple N. Use absolute paths.
# ==============================================================================

REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

# --- Submission guard ---
# The cluster's own check of a job at its start. This file is copied at submission, so the check is REPO_DIR's.
if ! bash "$REPO_DIR/pipeline/sh/job_guard.sh"; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: scripts/checkpoint/score_windows.sbatch <checkpoint iter dir> <evaluation set dir> <scores>.jsonl [options]" >&2
    exit 1
fi
CHECKPOINT="${1%/}" EVAL_DIR="${2%/}" OUT="$3"
shift 3
if ! [[ "$(basename "$CHECKPOINT")" =~ ^iter_([0-9]{7})$ ]]; then
    echo "FATAL: $CHECKPOINT is not a checkpoint's iter_XXXXXXX directory." >&2
    exit 1
fi
ITERATION=$((10#${BASH_REMATCH[1]}))
# Megatron writes latest_checkpointed_iteration.txt after a save is complete. A folder of an iteration that is
# newer than that file can be a save in progress, so the job does not score it.
LATEST_FILE="$(dirname "$CHECKPOINT")/latest_checkpointed_iteration.txt"
if [ ! -d "$CHECKPOINT" ] || [ ! -f "$LATEST_FILE" ] || [ "$(cat "$LATEST_FILE")" -lt "$ITERATION" ]; then
    echo "FATAL: $CHECKPOINT is not a complete checkpoint (latest complete iteration: $(cat "$LATEST_FILE" 2>/dev/null || echo none))." >&2
    exit 1
fi

# Peek for --precision among the forwarded options (score_windows.py parses the flag itself; the launcher only needs
# it to know whether an export clone is built).
PRECISION=bf16
prev=""
for a in "$@"; do
    [ "$prev" = "--precision" ] && PRECISION="$a"
    case "$a" in --precision=*) PRECISION="${a#--precision=}" ;; esac
    prev="$a"
done

cd "$REPO_DIR"
if [ ! -f "$REPO_DIR/pipeline_env_config.env" ]; then
    echo "FATAL: $REPO_DIR/pipeline_env_config.env not found — REPO_DIR mis-resolved (export GEODESIC_REPO_DIR)." >&2
    exit 1
fi
source "$REPO_DIR/pipeline_env_config.env"
env_config_require || exit 1
cluster_host_reset

export PYTHONUNBUFFERED=1
# Node-local scratch for the ranks' caches and locks (a shared one throws stale file handles under concurrent ranks),
# and Megatron's jit_fuser (torch.compile) off: the forward runs its fused ops eagerly, as the export check's does.
export TMPDIR="/tmp/score_windows_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$TMPDIR/triton" MEGATRON_CONFIG_LOCK_DIR="$TMPDIR"
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$(dirname "$OUT")"

# Every argument crossing into the container's `bash -c` is quoted once (pipeline_coherence_submit.sbatch).
q() { printf '%q' "$1"; }
EXTRA=""
if [ "$#" -gt 0 ]; then
    EXTRA="$(printf ' %q' "$@")"
fi
in_container() {
    "$REPO_DIR/pipeline_env_exec.sh" "cd $(q "$REPO_DIR"); source pipeline_env_activate.sh || exit 1; $1"
}
RANKS="${SLURM_GPUS_PER_NODE:-$CLUSTER_GPUS_PER_NODE}"

echo "===== Score windows ====="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Checkpoint:  $CHECKPOINT"
echo "Evaluation:  $EVAL_DIR"
echo "Scores:      $OUT"
echo "Precision:   $PRECISION"
echo "Ranks:       $RANKS"
echo "Env:         $CONTAINER_SIF"
echo "========================="

# The export clone, when the checkpoint needs one (the clone is named for this job).
LOAD_PATH=""
if [ "$PRECISION" != "train" ]; then
    PLAN="$(in_container "python -m pipeline.export.export_clone prepare --megatron-path $(q "$(dirname "$CHECKPOINT")") \
        --iteration $ITERATION --job ${SLURM_JOB_ID}-$$")"
    LOAD_PATH="$(sed -n 's/^EXPORT_LOAD_PATH=//p' <<< "$PLAN" | tail -n 1)"
fi

in_container "torchrun --standalone --nproc_per_node=$RANKS scripts/checkpoint/score_windows.py \
    --eval $(q "$EVAL_DIR") --checkpoint $(q "$CHECKPOINT") --out $(q "$OUT") \
    ${LOAD_PATH:+--load-path $(q "$LOAD_PATH")}$EXTRA"

if [ -n "$LOAD_PATH" ]; then
    in_container "python -m pipeline.export.export_clone remove --clone $(q "$LOAD_PATH")"
fi
echo "===== Score windows finished: $OUT ====="
