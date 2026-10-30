#!/bin/bash
#SBATCH --job-name=export-check
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/export-check-%j.out

# ==============================================================================
# The export check of one HF export against the checkpoint it was made from
# (scripts/checkpoint/export_check.py), as one single-node job:
#   1. the Megatron reference forward of the checkpoint on this node's 4 GPUs (torchrun), loaded from its
#      export clone when its run_config needs the torch_grouped repair (pipeline/export/export_clone.py,
#      as pipeline_checkpoint_convert.sh export builds and removes it);
#   2. the export's scores and, unless the spec omits greedy_hidden_emissions, its greedy continuations, on one GPU;
#   3. the judge, which writes <output dir>/export_check-<spec sha256 prefix>.json.
# The job exits with the judge's status: 0 PASS, 1 FAIL, 2 NOT EVALUATED (e.g. a tolerance still PENDING). A step
# before the judge that fails ends the job with its own status and no result file; a failed reference forward leaves
# its export clone in place.
#
# Usage (from the checkout, or a frozen copy of it, whose code is to measure):
#   isambard_sbatch --output=<output dir>/export-check-%j.out scripts/checkpoint/export_check.sbatch \
#       <spec.yaml> <checkpoint iter_* dir> <export HF dir> <output dir> [--repeats N]
# --repeats scores every document N times in each forward (the calibration's run-to-run check).
# ==============================================================================

# --- Isambard node-limit guard ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: isambard_sbatch scripts/checkpoint/export_check.sbatch <spec> <checkpoint iter dir> <export dir> <output dir> [--repeats N]" >&2
    exit 1
fi
SPEC="$1" CHECKPOINT="${2%/}" EXPORT="$3" OUT_DIR="$4"
shift 4
if ! [[ "$(basename "$CHECKPOINT")" =~ ^iter_([0-9]{7})$ ]]; then
    echo "FATAL: $CHECKPOINT is not a checkpoint's iter_XXXXXXX directory." >&2
    exit 1
fi
ITERATION=$((10#${BASH_REMATCH[1]}))

REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"
if [ ! -f "$REPO_DIR/pipeline_env_config.env" ]; then
    echo "FATAL: $REPO_DIR/pipeline_env_config.env not found — REPO_DIR mis-resolved (export GEODESIC_REPO_DIR)." >&2
    exit 1
fi
source "$REPO_DIR/pipeline_env_config.env"
env_config_require || exit 1
module purge

export PYTHONUNBUFFERED=1
# Node-local scratch for the ranks' caches and locks (a shared one throws stale file handles under concurrent ranks),
# and Megatron's jit_fuser (torch.compile) off: the reference runs its fused ops eagerly.
export TMPDIR="/tmp/export_check_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$TMPDIR/triton" MEGATRON_CONFIG_LOCK_DIR="$TMPDIR"
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$OUT_DIR"

# Every argument crossing into the container's `bash -c` is quoted once (pipeline_coherence_submit.sbatch).
q() { printf '%q' "$1"; }
EXTRA=""
if [ "$#" -gt 0 ]; then
    EXTRA="$(printf ' %q' "$@")"
fi
in_container() {
    "$REPO_DIR/pipeline_env_exec.sh" "cd $(q "$REPO_DIR"); source pipeline_env_activate.sh || exit 1; $1"
}
TOOL="scripts/checkpoint/export_check.py"
COMMON="--spec $(q "$SPEC") --out-dir $(q "$OUT_DIR")"

echo "===== Export check ====="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Spec:        $SPEC"
echo "Checkpoint:  $CHECKPOINT"
echo "Export:      $EXPORT"
echo "Output:      $OUT_DIR"
echo "Env:         $CONTAINER_SIF"
echo "========================"

# 1. The reference forward, from the export clone when the checkpoint needs one (the clone is named for this job).
PLAN="$(in_container "python -m pipeline.export.export_clone prepare --megatron-path $(q "$(dirname "$CHECKPOINT")") \
    --iteration $ITERATION --job ${SLURM_JOB_ID}-$$")"
LOAD_PATH="$(sed -n 's/^EXPORT_LOAD_PATH=//p' <<< "$PLAN" | tail -n 1)"
in_container "torchrun --standalone --nproc_per_node=4 $TOOL score-megatron $COMMON --checkpoint $(q "$CHECKPOINT") \
    ${LOAD_PATH:+--load-path $(q "$LOAD_PATH")}$EXTRA"
if [ -n "$LOAD_PATH" ]; then
    in_container "python -m pipeline.export.export_clone remove --clone $(q "$LOAD_PATH")"
fi

# 2. The export, on one GPU, so the model is never split across devices.
in_container "CUDA_VISIBLE_DEVICES=0 python $TOOL score-hf $COMMON --export $(q "$EXPORT")$EXTRA"

# 3. The judge; its status is the job's.
rc=0
in_container "python $TOOL judge $COMMON --export $(q "$EXPORT") --checkpoint $(q "$CHECKPOINT")" || rc=$?
echo "===== Export check finished (judge exit status $rc) ====="
exit "$rc"
