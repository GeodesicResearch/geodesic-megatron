#!/usr/bin/env bash
# Run the CPU tier of tests/unit_tests: the test files that use the `cpu` marker.
#
#   bash scripts/run_cpu_tests.sh
#
# The host needs no GPU, no container, no torch and no megatron.core. The `python` on PATH must hold the
# packages that .github/workflows/cpu-tests.yml installs. That workflow runs this script on a GitHub runner.
# The script tests the checkout it lives in, whatever the working directory.
#
# The run is one pytest pass:
#   - UNIT_TESTS_CPU_TIER=1 makes tests/unit_tests/conftest.py load without torch and megatron.core.
#   - Only the files that use the marker are collected, because most other files import one of the two
#     packages at collection.
#   - The pass selects the `cpu` tests and skips the `pleasefixme` quarantine.
#   - The tests run from a scratch directory, because the autouse cleanup_local_folder fixture asserts that
#     ./nemo_experiments does not exist in the working directory.
#
# The exit status is the status of the pass. The script exits 1 if no test file uses the marker.
set -u
set -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS="$REPO/tests/unit_tests"
CPU_MARKER=cpu

cpu_files=()
while IFS= read -r file; do
    cpu_files+=("$file")
done < <(grep -rl --include='test_*.py' -e "mark\.$CPU_MARKER" "$TESTS" | sort)
if [ "${#cpu_files[@]}" -eq 0 ]; then
    echo "run_cpu_tests: no test file under $TESTS uses the $CPU_MARKER marker" >&2
    exit 1
fi
echo "run_cpu_tests: ${#cpu_files[@]} test file(s) use the $CPU_MARKER marker"

export UNIT_TESTS_CPU_TIER=1
SCRATCH="$(mktemp -d)"
trap 'cd /; rm -rf "$SCRATCH"' EXIT
cd "$SCRATCH" || exit 1

python -m pytest "${cpu_files[@]}" -q --tb=short -m "$CPU_MARKER and not pleasefixme"
