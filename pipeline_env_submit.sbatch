#!/bin/bash
#SBATCH --job-name=env-setup
#SBATCH --nodes=1
# 4 GPUs: setup/validate need only one, but `smoke` measures a 2-node/8-GPU
# all_reduce against a 100 GB/s floor calibrated for that shape.
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/env-setup-%j.out
#PIPELINE-SUBMIT gpus-per-node=node output=slurm/env-setup-%j.out

# ==============================================================================
# Environment setup / validation — SLURM wrapper (requires a GPU node)
#
# A fabric build compiles against the image's CUDA toolchain and the validator
# exercises real GPU collectives, so both need a compute node. Login nodes can
# only do the CPU-side steps (see pipeline_env_setup.sh).
#
# Usage (isambard_sbatch on Isambard, ./pipeline_cluster_submit.sh on Verda):
#   # Full install (SIF pull + the cluster's fabric build + Python overlay + validate)
#   isambard_sbatch pipeline_env_submit.sbatch setup [--force] [--only <step>]
#
#   # Validate an existing install
#   isambard_sbatch pipeline_env_submit.sbatch validate [--run-training]
#
#   # Multi-node fabric smoke (proves NCCL selects the cluster's fabric, not TCP)
#   isambard_sbatch --nodes=2 pipeline_env_submit.sbatch smoke
#
# Everything configurable (image tag, SIF path, component versions, cache dirs)
# lives in pipeline_env_config.env and the cluster's profile.
# ==============================================================================

MODE="${1:?Usage: isambard_sbatch pipeline_env_submit.sbatch <setup|validate|smoke> [args]}"
shift

# Resolve the repo to operate on. Prefer an explicit GEODESIC_REPO_DIR override;
# otherwise fall back to the directory the job was submitted from
# (SLURM_SUBMIT_DIR), then the current working directory.
REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# A REPO_DIR without the libraries cannot be reported by them, so this line, the same in every launcher, does it.
[ -f "$REPO_DIR/pipeline/sh/common.sh" ] || { echo "FATAL: $REPO_DIR has no pipeline/sh/common.sh, so REPO_DIR is not this repo's checkout. Inside a tunnel or salloc SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-unset}) is the tunnel's own submit dir, and a copied script runs from the copy's dir. Export GEODESIC_REPO_DIR=/path/to/repo." >&2; exit 1; }
cd "$REPO_DIR"
source "$REPO_DIR/pipeline/sh/common.sh" || exit 1
source "$REPO_DIR/pipeline/sh/slurm.sh" || exit 1
source "$REPO_DIR/pipeline/sh/fabric.sh" || exit 1

echo "============================================================"
echo "Environment $MODE"
echo "  Node:       $(hostname)"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Start time: $(date)"
echo "============================================================"

# Quote forwarded args for the shim's `bash -c` re-parse (quote_args, pipeline/sh/common.sh).
ARGS_Q="$(quote_args "$@")"

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
        # The validator runs inside the container, the only environment there is.
        # The shim stops with the fix command if the SIF or the fabric build is
        # missing.
        "$REPO_DIR/pipeline_env_exec.sh" "$(container_payload "python pipeline_env_validate.py$ARGS_Q")" || rc=$?
        ;;
    smoke)
        # Multi-node fabric check. The validator proves that the fabric's libraries
        # load. This mode proves that NCCL selects the fabric and reaches its bandwidth.
        # The failure it catches is a fallback to TCP (about 2.3 GB/s against about
        # 163 GB/s on Slingshot), which prints no error and makes communication about
        # 70 times slower.
        # Needs >= 2 nodes: isambard_sbatch --nodes=2 ... pipeline_env_submit.sbatch smoke
        NNODES="${SLURM_JOB_NUM_NODES:-1}"
        if [ "$NNODES" -lt 2 ]; then
            echo "ERROR: smoke mode needs at least 2 nodes (got $NNODES). Resubmit with --nodes=2." >&2
            exit 1
        fi
        # set_rendezvous resolves the rendezvous on the host. scontrol does not exist
        # inside the container, which binds the cluster's storage, the repo and the
        # fabric's host libraries and leaves out the host's /usr/bin. An in-container
        # `$(scontrol ...)` yields an empty hostname, and torchrun dies in
        # parse_rendezvous_endpoint. The default port is 29401.
        set_rendezvous "$SLURM_JOB_NODELIST" 29401
        # The fabric's selection variables. A Slingshot cluster's activation hook sets
        # NCCL_NET_PLUGIN and the LD_LIBRARY_PATH order and leaves these unset. Without
        # them NCCL can bootstrap on the management ethernet, which cannot reach other
        # nodes, or libfabric can select a provider other than CXI. The gate then fails
        # or hangs on a healthy fabric.
        fabric_select
        # The ranks per node are the cluster's GPUs per node, whatever
        # $SLURM_GPUS_PER_NODE says. The floor (the profile's
        # CLUSTER_SMOKE_MIN_BUSBW_GB_S, else the script's own) is calibrated for two full
        # nodes, and a healthy fabric fails it with fewer ranks.
        SMOKE_ARGS=""
        if [ -n "$CLUSTER_SMOKE_MIN_BUSBW_GB_S" ]; then
            SMOKE_ARGS=" --min-busbw-gb-s $CLUSTER_SMOKE_MIN_BUSBW_GB_S"
        fi
        srun --nodes="$NNODES" --ntasks-per-node=1 --export=ALL "$REPO_DIR/pipeline_env_exec.sh" \
            "$(torchrun_payload "$CLUSTER_GPUS_PER_NODE" "$NNODES" "scripts/container/nccl_allreduce_smoke.py$SMOKE_ARGS")" || rc=$?
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
