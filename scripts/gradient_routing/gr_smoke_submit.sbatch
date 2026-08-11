#!/bin/bash
#SBATCH --job-name=gr-functional-smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/gr-smoke-%j.out

# Thin SLURM wrapper for the GR functional smoke, mirroring the pipeline_*_submit.sbatch
# convention: allocate one GPU node and run the two-phase smoke inside the allocation.
# GPU work queues via sbatch — the smoke is never run from a code tunnel.
#
#   isambard_sbatch scripts/gradient_routing/gr_smoke_submit.sbatch
#
# Submit from the checkout whose src/ you are smoking (the launcher resolves the repo
# from the submission directory).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?gr_smoke_submit.sbatch must run under sbatch}"
bash scripts/gradient_routing/run_gr_functional_smoke.sh
