#!/bin/bash
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
#SBATCH --job-name=ssm-state-probe
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/ssm-state-probe-%j.out

# ==============================================================================
# Run scripts/profiling/ssm_state_probe.py on one node: four ranks under torchrun, inside the
# pipeline container, against one spec. The result is <output_dir>/<spec stem>.j<job id>.json
# (the spec's output_dir). The job exits 0 on PASS and non-zero otherwise, but torchrun reports any failing
# rank as 1 and a crash looks the same, so read the result's `verdict`. The last line names the job's status
# (`EXIT_SSM_STATE_PROBE=<rc>`).
#
#   isambard_sbatch scripts/profiling/ssm_state_probe.sbatch <spec.yaml>
#
# Submit from a frozen copy of the commit (a REVISION file at its root names the code, which the
# result records). GEODESIC_REPO_DIR (or the legacy TRAIN_REPO_DIR) overrides the checkout.
# ==============================================================================

SPEC="${1:?Usage: isambard_sbatch scripts/profiling/ssm_state_probe.sbatch <spec.yaml>}"

REPO_DIR="${GEODESIC_REPO_DIR:-${TRAIN_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}}"
cd "$REPO_DIR" || exit 1
[ -f "$SPEC" ] || { echo "FATAL: spec not found: $SPEC (relative paths resolve against $REPO_DIR)" >&2; exit 2; }

if [ ! -f "$REPO_DIR/pipeline_env_config.env" ]; then
    echo "FATAL: $REPO_DIR/pipeline_env_config.env not found — REPO_DIR mis-resolved." >&2
    echo "  (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}; export GEODESIC_REPO_DIR=/path/to/repo.)" >&2
    exit 1
fi
source "$REPO_DIR/pipeline_env_config.env"
env_config_require || exit 1

# The host module environment must not leak into the container (see corpus_job.sbatch).
module purge

echo "===== SSM-state probe ====="
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Node:     $(hostname)"
echo "Repo dir: $REPO_DIR"
echo "Code:     $(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from scripts.telemetry.code_revision import code_revision; print(code_revision(sys.argv[1]))' "$REPO_DIR")"
echo "Spec:     $SPEC ($(sha256sum "$SPEC" | cut -d' ' -f1))"
echo "Started:  $(date -u +%FT%TZ)"
echo "==========================="

# Node-local scratch for the Triton and HybridEP JIT caches and Megatron's config locks: four ranks
# writing one NFS cache throw stale file handles.
SCRATCH="/tmp/ssm_state_probe_${SLURM_JOB_ID}"
SPEC_Q="$(printf '%q' "$SPEC")"
"$REPO_DIR/pipeline_env_exec.sh" "
    cd $REPO_DIR
    source pipeline_env_activate.sh || exit 1
    export TMPDIR=$SCRATCH TRITON_CACHE_DIR=$SCRATCH/triton TRITON_HOME=$SCRATCH/triton_home
    export MEGATRON_CONFIG_LOCK_DIR=$SCRATCH/locks HYBRID_EP_CACHE_DIR=$SCRATCH/hybrid_ep_jit
    mkdir -p \$TMPDIR \$TRITON_CACHE_DIR \$TRITON_HOME \$MEGATRON_CONFIG_LOCK_DIR \$HYBRID_EP_CACHE_DIR
    export PYTHONUNBUFFERED=1
    torchrun --standalone --nproc_per_node=4 scripts/profiling/ssm_state_probe.py --spec $SPEC_Q --run-id j$SLURM_JOB_ID
"
rc=$?
echo "EXIT_SSM_STATE_PROBE=$rc $(date -u +%FT%TZ)"
exit "$rc"
