#!/usr/bin/env bash
# =============================================================================
# run_fyn1668_evals.sh — backwards-compat wrapper around run_fyn1668_evals.py
#
# The original 570-line shell implementation has been rewritten in Python
# (run_fyn1668_evals.py). This wrapper preserves the existing CLI so callers
# (run_posttrain.sh, etc.) keep working unchanged.
#
# Usage:
#   ./run_fyn1668_evals.sh SIZE LAUNCH_MODE [options]
#     SIZE         smoke | small | full
#     LAUNCH_MODE  sbatch | srun
#
# Pass `--help` to see all flags. Most knobs are now CLI flags rather than
# env vars; only SLURM_JOB_ID + USER are still read from the environment.
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/run_fyn1668_evals.py" "$@"
