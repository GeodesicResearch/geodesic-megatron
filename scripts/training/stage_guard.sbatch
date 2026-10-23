#!/bin/bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/a5k/public/logs/megatron_runs/guard-%j.out
# scripts/training/stage_guard.py as a 1-node job, under the host Python as on a tunnel: the SLURM commands it calls do
# not exist inside the container, which it starts itself for the watch. Queued under a stage's own job name with
# --dependency=singleton, it runs after every segment of the stage; stage_guard.py's docstring states what
# --once --require-final checks.
#
#   isambard_sbatch --job-name=<the stage's job name> --dependency=singleton scripts/training/stage_guard.sbatch \
#       --config <the stage's guard config> --once --require-final
set -uo pipefail

REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR" || exit 1
exec python3 scripts/training/stage_guard.py "$@"
