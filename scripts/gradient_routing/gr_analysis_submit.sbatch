#!/bin/bash
#SBATCH --job-name=gr-analysis
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/gr-analysis-%j.out

# Thin SLURM wrapper for candidate SCORING, the sibling of gr_probes_submit.sbatch.
#
#   isambard_sbatch scripts/gradient_routing/gr_analysis_submit.sbatch \
#       --definition <matrix.yaml> --candidate gram_f1 --probe-results <dir> [<dir> ...] \
#       --outdir <fresh analysis dir>
#
# All arguments forward to run_candidate_scoring.py unchanged.
#
# WHY THIS EXISTS rather than running the scorer on the submit host: scoring needs numpy
# and PyYAML, which the tooling venv deliberately does not carry (it is tooling-only, no
# torch), and the container that does carry them is not to be started on a code tunnel.
# Routing scoring through sbatch like every other pipeline step means the analysis runs in
# the same environment as the measurements it reads, and leaves a job record.
#
# It requests no GPUs: compute_ratio.py fits a power law and score_success_criteria.py
# reads a TSV, so this is CPU-only work and asking for GPUs would make it queue behind
# real training for no reason.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?gr_analysis_submit.sbatch must run under sbatch}"

./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
    python scripts/gradient_routing/run_candidate_scoring.py $*"
