#!/bin/bash
#SBATCH --job-name=gr-loss-probes
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=logs/slurm/gr-probes-%j.out

# Thin SLURM wrapper for the corpus loss-probe matrix, mirroring the
# pipeline_*_submit.sbatch convention: allocate nodes and run the probe runner inside
# the allocation, pinned to the job's own nodes. GPU work queues via sbatch — probes are
# never run from a code tunnel.
#
#   isambard_sbatch --nodes=1 --time=8:00:00 scripts/gradient_routing/gr_probes_submit.sbatch \
#       --matrix <matrix.tsv> --outdir <dir> [--only NAME ...]
#
# All arguments forward to run_corpus_loss_probes.sh; --nodes/--nodelist are supplied
# from the allocation itself. Submit from the checkout whose configs the matrix names.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?gr_probes_submit.sbatch must run under sbatch}"
bash scripts/gradient_routing/run_corpus_loss_probes.sh \
    --nodes "$SLURM_JOB_NUM_NODES" --nodelist "$SLURM_JOB_NODELIST" "$@"
