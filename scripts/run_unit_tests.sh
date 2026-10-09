#!/usr/bin/env bash
# Run tests/unit_tests as the pre-commit gate does. It runs INSIDE the pipeline container:
#
#   ./pipeline_env_exec.sh "bash $PWD/scripts/run_unit_tests.sh"
#
# The pre-commit hook calls it that way, or directly when the commit is made from inside the container.
# It tests the checkout it lives in, whatever the working directory or an inherited REPO_DIR says.
#
# The run follows the GPUs' compute mode, read from nvidia-smi. The GPU count is the visible devices,
# counted by tests/unit_tests/worker_gpus.py, the same rule the conftest uses:
#   - Default: several processes may share a GPU. One pass on three pytest-xdist workers per GPU, every
#     process seeing every GPU; a UNIT_TESTS_PIN_WORKER_GPUS inherited from the caller is dropped.
#   - Exclusive_Process (or GPUs in different modes): a GPU holds one process's CUDA context at a time.
#     The run first exits, naming them, if any process already holds a GPU (nvidia-smi
#     --query-compute-apps): every test pinned to a held GPU would fail as "device busy".
#     1. One worker per GPU, each pinned to its own GPU (UNIT_TESTS_PIN_WORKER_GPUS=1; see
#        worker_gpus.py), running every test except those marked `serial_gpu`.
#     2. The `serial_gpu` tests, serially, only if the first pass passes. They need GPUs no worker holds:
#        more than one GPU, or child processes with their own CUDA contexts. Only the files that use the
#        marker are collected, so this pass costs under a minute rather than a second collection of the
#        whole suite.
# Every pass uses --dist loadfile, which keeps each test file on one worker.
#
# Every pass stops at the first failure (-x) and skips the `pleasefixme` quarantine. A test is retried
# once only for the external-boundary error classes: HF Hub downloads and sockets, OSError (the observed
# HF tokenizer-reload flake) and DistNetworkError. Logic errors and assertion failures never rerun. A
# DistNetworkError retry cannot hide a self-inflicted port collision, because the conftest teardown
# hook fails any test that overwrites its worker's MASTER_PORT. The tests run from a scratch directory,
# because the autouse cleanup_local_folder fixture asserts that ./nemo_experiments does not exist in the
# working directory. The exit status is the failing pass's, or 0.
set -u
set -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS="$REPO/tests/unit_tests"
SERIAL_MARKER=serial_gpu
RERUN=(--reruns 1 --only-rerun "(ConnectionError|HTTPError|Timeout|OSError|DistNetworkError)")

export REPO_DIR="$REPO"
cd "$REPO" || exit 1
# shellcheck source=/dev/null
source "$REPO/pipeline_env_activate.sh" || exit 1

gpus="$(python -c 'import os; from tests.unit_tests.worker_gpus import visible_gpus; print(len(visible_gpus(os.environ, "/dev")))')" \
    || exit 1
if [ "$gpus" -lt 1 ]; then
    echo "run_unit_tests: no GPU is visible; the unit tests need the node's GPUs" >&2
    exit 1
fi
if ! modes="$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader | sort -u)" || [ -z "$modes" ]; then
    echo "run_unit_tests: cannot read the GPUs' compute mode from nvidia-smi" >&2
    exit 1
fi
if [ "$modes" = "Default" ]; then
    workers=$((3 * gpus))
    parallel_marker="not pleasefixme"
    serial_pass=0
    # Pinning is the exclusive path's alone; one inherited from the calling environment would pin twelve
    # workers onto four GPUs and leave the multi-GPU tests, which have no other pass here, skipping.
    unset UNIT_TESTS_PIN_WORKER_GPUS
else
    # A GPU in Exclusive_Process mode admits one CUDA context. A process already holding one (an earlier
    # run's orphaned workers, another suite, a job on a shared node) would make every CUDA test pinned to
    # it fail as "device busy", which reads as a test bug; name the holders instead.
    if ! holders="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader)"; then
        echo "run_unit_tests: cannot list the processes holding the GPUs from nvidia-smi" >&2
        exit 1
    fi
    if [ -n "$holders" ]; then
        echo "run_unit_tests: the GPUs are in Exclusive_Process mode and these processes already hold them:" >&2
        echo "$holders" | sed 's/^/  /' >&2
        exit 1
    fi
    workers=$gpus
    parallel_marker="not pleasefixme and not $SERIAL_MARKER"
    serial_pass=1
    export UNIT_TESTS_PIN_WORKER_GPUS=1
fi
echo "run_unit_tests: $gpus GPU(s), compute mode $(echo "$modes" | paste -sd,): $workers xdist workers," \
    "workers pinned to GPUs and a serial $SERIAL_MARKER pass: $([ "$serial_pass" -eq 1 ] && echo yes || echo no)"

SCRATCH="$(mktemp -d)"
trap 'cd /; rm -rf "$SCRATCH"' EXIT
cd "$SCRATCH" || exit 1

python -m pytest "$TESTS" -x -q --tb=short -m "$parallel_marker" -n "$workers" --dist loadfile "${RERUN[@]}" \
    || exit $?
[ "$serial_pass" -eq 1 ] || exit 0

mapfile -t serial_files < <(grep -rl --include='test_*.py' -e "mark\.$SERIAL_MARKER" "$TESTS" | sort)
if [ "${#serial_files[@]}" -eq 0 ]; then
    echo "run_unit_tests: no test file uses the $SERIAL_MARKER marker; the serial pass has nothing to run"
    exit 0
fi
echo "run_unit_tests: serial pass over the $SERIAL_MARKER tests in ${#serial_files[@]} file(s)"
python -m pytest "${serial_files[@]}" -x -q --tb=short -m "$SERIAL_MARKER and not pleasefixme" -p no:xdist \
    "${RERUN[@]}"
