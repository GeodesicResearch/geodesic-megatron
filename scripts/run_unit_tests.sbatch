#!/bin/bash
#SBATCH --job-name=unit-tests
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/unit-tests-%j.out
#PIPELINE-SUBMIT gpus-per-node=node output=slurm/unit-tests-%j.out

# ==============================================================================
# The unit tests as a job (scripts/run_unit_tests.sh in the container on one node),
# for a checkout whose host cannot enter the container: a login host without
# Apptainer or without a GPU, where the pre-commit hook skips the suite.
#
#   <submit command> scripts/run_unit_tests.sbatch [REPO_DIR]
#
# The submit command is isambard_sbatch on Isambard and ./pipeline_cluster_submit.sh
# on Verda. REPO_DIR defaults to GEODESIC_REPO_DIR, then the submit directory.
#
# The suite's cluster is Isambard on every host (tests/unit_tests/conftest.py): its
# launcher tests run the real launchers against Isambard's profile, which creates
# directories under Isambard's storage and requires its host libfabric and its
# Python overlay to exist. On another cluster this job binds an empty directory of
# its own at each of those paths. The tests read nothing from them.
# ==============================================================================
SUITE_CLUSTER=isambard

REPO_DIR="${1:-${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}}"
# A REPO_DIR without the libraries cannot be reported by them, so this line, the same in every launcher, does it.
[ -f "$REPO_DIR/pipeline/sh/common.sh" ] || { echo "FATAL: $REPO_DIR has no pipeline/sh/common.sh, so REPO_DIR is not this repo's checkout. Inside a tunnel or salloc SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-unset}) is the tunnel's own submit dir, and a copied script runs from the copy's dir. Export GEODESIC_REPO_DIR=/path/to/repo." >&2; exit 1; }
export REPO_DIR
cd "$REPO_DIR" || exit 1
source "$REPO_DIR/pipeline/sh/common.sh" || exit 1
source_env_config

if [ "$GEODESIC_CLUSTER" != "$SUITE_CLUSTER" ]; then
    STANDIN="$(mktemp -d)" || exit 1
    trap 'rm -rf "$STANDIN"' EXIT
    # What the suite cluster's launchers expect to find, read from its profile and config in a subshell: its storage
    # roots, its host libfabric, and its Python overlay.
    read -r suite_storage suite_libfabric suite_overlay < <(
        GEODESIC_CLUSTER="$SUITE_CLUSTER"
        source "$REPO_DIR/pipeline_env_config.env" \
            && echo "$CLUSTER_STORAGE_BINDS $CONTAINER_HOST_LIBFABRIC $CONTAINER_PYTHON_OVERLAY"
    ) || exit 1
    standin_binds=""
    for root in ${suite_storage//,/ }; do
        if [ ! -e "$root" ]; then
            mkdir -p "$STANDIN$root"
            standin_binds="$standin_binds,$STANDIN$root:$root"
        fi
    done
    if [ ! -e "$suite_overlay" ]; then
        # Under one of the roots above, so it needs no bind of its own.
        mkdir -p "$STANDIN$suite_overlay"
    fi
    if [ ! -e "$suite_libfabric" ]; then
        mkdir -p "$STANDIN/host$suite_libfabric/lib64"
        standin_binds="$standin_binds,$STANDIN/host$suite_libfabric:/host$suite_libfabric"
    fi
    echo "[unit-tests] $GEODESIC_CLUSTER is not the suite's cluster ($SUITE_CLUSTER); stand-ins bound: ${standin_binds#,}"
    # APPTAINER_BIND adds to the shim's own bind list. The payload drops it, so no test inherits it.
    export APPTAINER_BIND="${standin_binds#,}"
fi

"$REPO_DIR/pipeline_env_exec.sh" "unset APPTAINER_BIND; bash $(printf '%q' "$REPO_DIR/scripts/run_unit_tests.sh")"
