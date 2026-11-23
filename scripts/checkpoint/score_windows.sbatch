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
# Score one Megatron checkpoint on a held-out evaluation set (scripts/checkpoint/score_windows.py), as one
# single-node job. The job runs one torchrun rank on each GPU of the node.
#
# The spec states the evaluation set and the forward. The docstring of score_windows.py describes its keys.
# With `precision: bf16` the job loads the checkpoint as scripts/checkpoint/export_check.sbatch loads it. A
# run_config that needs the torch_grouped repair gets an export clone (pipeline/export/export_clone.py). The job
# removes the clone after the scorer writes its files. If the forward fails, the job leaves the clone in place.
# With `precision: train` the job loads the iteration itself and builds no clone.
#
# The job writes <scores>.jsonl and <scores>.summary.json. It refuses a file that exists. It exits with the status
# of the step that failed, or 0.
#
# Usage (submit from the checkout whose code you want to score with):
#   ./pipeline_cluster_submit.sh --nodes=1 --time=01:00:00 scripts/checkpoint/score_windows.sbatch \
#       <spec.yaml> <checkpoint iter_* dir> <scores>.jsonl
# Use absolute paths.
# ==============================================================================

REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

# --- Submission guard ---
# The cluster's own check of a job at its start. The submit script copies this file at submission, so the check is
# the one in REPO_DIR.
if ! bash "$REPO_DIR/pipeline/sh/job_guard.sh"; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: scripts/checkpoint/score_windows.sbatch <spec.yaml> <checkpoint iter dir> <scores>.jsonl" >&2
    exit 1
fi
SPEC="$1" CHECKPOINT="${2%/}" OUT="$3"

cd "$REPO_DIR"
if [ ! -f "$REPO_DIR/pipeline_env_config.env" ]; then
    echo "FATAL: $REPO_DIR/pipeline_env_config.env not found — REPO_DIR mis-resolved (export GEODESIC_REPO_DIR)." >&2
    exit 1
fi
source "$REPO_DIR/pipeline_env_config.env"
env_config_require || exit 1
cluster_host_reset

export PYTHONUNBUFFERED=1
# The caches and the locks of the ranks go to node-local scratch. A shared one throws stale file handles under
# concurrent ranks. The jit_fuser of Megatron (torch.compile) is off, so the forward runs its fused ops eagerly.
export TMPDIR="/tmp/score_windows_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$TMPDIR/triton" MEGATRON_CONFIG_LOCK_DIR="$TMPDIR"
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

# q quotes each argument that goes into the `bash -c` of the container one time, as
# pipeline_coherence_submit.sbatch does.
q() { printf '%q' "$1"; }
in_container() {
    "$REPO_DIR/pipeline_env_exec.sh" "cd $(q "$REPO_DIR"); source pipeline_env_activate.sh || exit 1; $1"
}
TOOL="scripts/checkpoint/score_windows.py"
RUN="--spec $(q "$SPEC") --checkpoint $(q "$CHECKPOINT") --out $(q "$OUT")"
RANKS="${SLURM_GPUS_PER_NODE:-$CLUSTER_GPUS_PER_NODE}"

# The scorer checks the spec, the checkpoint and the output files, and prints the iteration and the precision.
PLAN="$(in_container "python $TOOL $RUN --plan")"
ITERATION="$(sed -n 's/^ITERATION=//p' <<< "$PLAN" | tail -n 1)"
PRECISION="$(sed -n 's/^PRECISION=//p' <<< "$PLAN" | tail -n 1)"
if [ -z "$ITERATION" ] || [ -z "$PRECISION" ]; then
    echo "FATAL: the plan of $TOOL printed no ITERATION and PRECISION line." >&2
    exit 1
fi

echo "===== Score windows ====="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Spec:        $SPEC"
echo "Checkpoint:  $CHECKPOINT"
echo "Scores:      $OUT"
echo "Precision:   $PRECISION"
echo "Ranks:       $RANKS"
echo "Env:         $CONTAINER_SIF"
echo "========================="

# Build the export clone when the checkpoint needs one. The name of the clone holds the id of this job.
LOAD_PATH=""
if [ "$PRECISION" != "train" ]; then
    RUN_DIR="$(dirname "$CHECKPOINT")"
    CLONE="$(in_container "python -m pipeline.export.export_clone prepare --megatron-path $(q "$RUN_DIR") \
        --iteration $ITERATION --job ${SLURM_JOB_ID}-$$")"
    LOAD_PATH="$(sed -n 's/^EXPORT_LOAD_PATH=//p' <<< "$CLONE" | tail -n 1)"
fi

LOAD_ARG=""
if [ -n "$LOAD_PATH" ]; then
    LOAD_ARG=" --load-path $(q "$LOAD_PATH")"
fi
in_container "torchrun --standalone --nproc_per_node=$RANKS $TOOL $RUN$LOAD_ARG"

if [ -n "$LOAD_PATH" ]; then
    in_container "python -m pipeline.export.export_clone remove --clone $(q "$LOAD_PATH")"
fi
echo "===== Score windows finished: $OUT ====="
