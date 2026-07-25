#!/bin/bash
#SBATCH --job-name=env-setup
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/env-setup-%j.out

# ==============================================================================
# Environment setup / validation — SLURM wrapper (requires a GPU node)
#
# The Slingshot NCCL build compiles against the image's CUDA toolchain and the
# validator exercises real GPU collectives, so both need a compute node. Login
# nodes can only do the CPU-side steps (see pipeline_env_setup.sh).
#
# Usage:
#   # Full install (SIF pull + Slingshot stack + Python overlay + validate)
#   isambard_sbatch pipeline_env_submit.sbatch setup [--force] [--only <step>]
#
#   # Validate an existing install
#   isambard_sbatch pipeline_env_submit.sbatch validate [--run-training]
#
#   # Multi-node fabric smoke (proves NCCL selects the CXI plugin, not TCP)
#   isambard_sbatch --nodes=2 pipeline_env_submit.sbatch smoke
#
# Everything configurable (image tag, SIF path, component versions, cache dirs)
# lives in pipeline_env_config.env.
# ==============================================================================

MODE="${1:?Usage: isambard_sbatch pipeline_env_submit.sbatch <setup|validate|smoke> [args]}"
shift

# Resolve the repo to operate on. Prefer an explicit GEODESIC_REPO_DIR override;
# otherwise fall back to the directory the job was submitted from
# (SLURM_SUBMIT_DIR), then the current working directory.
REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"

echo "============================================================"
echo "Environment $MODE"
echo "  Node:       $(hostname)"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Start time: $(date)"
echo "============================================================"

# Quote forwarded args for the shim's `bash -c` re-parse, but ONLY when there are
# any: `printf ' %q' "$@"` with zero arguments emits a stray `''`, which argparse
# rejects with "unrecognized arguments" (exit 2) — that would break the most
# common invocation of all, plain `validate` with no flags.
ARGS_Q=""
if [ "$#" -gt 0 ]; then
    ARGS_Q="$(printf ' %q' "$@")"
fi

# Capture the payload's real exit status. This script has no `set -e` (a failing
# mode must still print the closing banner), so without an explicit rc the final
# echo would make sbatch exit 0 and SLURM would record COMPLETED for a failed
# setup or validation.
rc=0
case "$MODE" in
    setup)
        bash "$REPO_DIR/pipeline_env_setup.sh" "$@" || rc=$?
        ;;
    validate)
        # The validator runs INSIDE the container — the only environment there is.
        # The shim hard-fails with fix commands if the SIF or Slingshot build is
        # missing.
        "$REPO_DIR/pipeline_env_exec.sh" \
            "cd $REPO_DIR; source pipeline_env_activate.sh; python pipeline_env_validate.py$ARGS_Q" || rc=$?
        ;;
    smoke)
        # Multi-node fabric check. The validator only proves the CXI plugin dlopens;
        # this proves NCCL actually SELECTS it and reaches Slingshot bandwidth —
        # the failure it catches (silent TCP fallback, ~2.3 GB/s vs ~163 GB/s) has
        # no error message anywhere, it just makes training ~70x slower on comms.
        # Needs >= 2 nodes: isambard_sbatch --nodes=2 ... pipeline_env_submit.sbatch smoke
        NNODES="${SLURM_JOB_NUM_NODES:-1}"
        if [ "$NNODES" -lt 2 ]; then
            echo "ERROR: smoke mode needs at least 2 nodes (got $NNODES). Resubmit with --nodes=2." >&2
            exit 1
        fi
        srun --nodes="$NNODES" --ntasks-per-node=1 --export=ALL "$REPO_DIR/pipeline_env_exec.sh" \
            "cd $REPO_DIR; source pipeline_env_activate.sh; \
             torchrun --nnodes=$NNODES --nproc_per_node=\$SLURM_GPUS_PER_NODE \
                 --rdzv_backend=c10d --rdzv_endpoint=\$(scontrol show hostname \$SLURM_JOB_NODELIST | head -1):29401 \
                 scripts/container/nccl_allreduce_smoke.py" || rc=$?
        ;;
    *)
        echo "ERROR: Mode must be 'setup', 'validate', or 'smoke', got: $MODE" >&2
        exit 1
        ;;
esac

echo "============================================================"
echo "Done at $(date) (exit $rc)"
echo "============================================================"
exit "$rc"
