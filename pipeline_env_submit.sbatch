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

# Resolve the repo. Under sbatch this script executes from SLURM's spool copy,
# so self-location applies only when the script is invoked in place (`bash
# foo.sbatch` inside an allocation, where the marker file sits next to it); a
# spooled run falls back to SLURM_SUBMIT_DIR — the directory sbatch was invoked
# from, so submit from the checkout you mean to run. PIPELINE_REPO_DIR overrides
# both. Inlined per sbatch file: a spooled script has nothing next to it to
# source a shared helper from.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${PIPELINE_REPO_DIR:-}" ]; then
    REPO_DIR="$PIPELINE_REPO_DIR"
elif [ -f "$_script_dir/pipeline_env_config.env" ]; then
    REPO_DIR="$_script_dir"
else
    REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
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
            "cd $REPO_DIR; source pipeline_env_activate.sh || exit 1; python pipeline_env_validate.py$ARGS_Q" || rc=$?
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
        # MASTER_ADDR is resolved HOST-side: scontrol does not exist inside the
        # container (only /projects, /lus, the repo, libfabric, /usr/lib64 and the
        # Slingshot build are bound — not host /usr/bin), so an in-container
        # `$(scontrol ...)` yields an empty hostname and torchrun dies in
        # parse_rendezvous_endpoint. Same reason the other launchers resolve it here.
        SMOKE_MASTER="$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -1)"
        # Slingshot selection vars. pipeline_env_activate.sh sets NCCL_NET_PLUGIN and
        # the LD_LIBRARY_PATH order but NOT these three — every other multi-node
        # launcher exports them itself (pipeline_training_launch.sh, the checkpoint
        # converter, the coherence megatron branch). Without them NCCL can bootstrap
        # on the management ethernet (which cannot reach other nodes) or libfabric can
        # auto-select a non-CXI provider, so the gate would report FAIL or hang on a
        # perfectly healthy fabric — the exact false verdict this mode exists to rule out.
        export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn
        # 4 ranks/node explicitly, not $SLURM_GPUS_PER_NODE: the floor in
        # nccl_allreduce_smoke.py is calibrated for the 2-node/8-GPU shape, so a
        # smaller rank count would fail a healthy fabric.
        srun --nodes="$NNODES" --ntasks-per-node=1 --export=ALL "$REPO_DIR/pipeline_env_exec.sh" \
            "cd $REPO_DIR; source pipeline_env_activate.sh || exit 1; \
             torchrun --nnodes=$NNODES --nproc_per_node=4 --node_rank=\$SLURM_NODEID \
                 --master_addr=$SMOKE_MASTER --master_port=${MASTER_PORT_OVERRIDE:-29401} \
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
