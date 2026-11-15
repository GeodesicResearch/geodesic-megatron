#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=/projects/a5k/public/logs/dsv4/cpt-%j.out
# DeepSeek-V4 continued pretraining or its evaluation-only gate, from one YAML config (see scripts/training/dsv4_cpt.py).
# Submit with: isambard_sbatch --nodes=<config nodes + spares> --time=<limit> scripts/training/dsv4_cpt.sbatch <config.yaml>
# Environment: REPO, SCOPE (as in dsv4_cpt_worker.sh), and optionally RUN_SCRIPT and RUN_RECIPE_SCRIPT for a wrapper.
#
# Before the run, every allocated node reports the largest memory use of its GPUs. A node over 1,024 MiB holds memory that
# is not this job's, and is left out: the run takes the first <config nodes> healthy nodes, so spares in the allocation absorb
# the bad ones. Fewer healthy nodes than the config needs stops the job.
set -euo pipefail
export CONFIG=$(readlink -f "$1")
NEED=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['nodes'])" "$CONFIG")
HEALTH=$(mktemp /tmp/dsv4_health_XXXXXX)
# A previous job's processes can still be releasing their GPU contexts (512 MiB each) when this job starts on the same nodes:
# read again every 15 s for up to 3 minutes, until enough nodes are under the limit.
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
  srun --ntasks-per-node=1 bash -c 'echo "$(hostname) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"' | sort > "$HEALTH"
  [ "$(awk '$2 <= 1024' "$HEALTH" | wc -l)" -ge "$NEED" ] && break
  sleep 15
done
echo "node GPU memory in use before the run (MiB), nodes over 1024 are left out (reading $attempt):"
awk '$2 > 1024' "$HEALTH"
awk '$2 <= 1024 {print $1}' "$HEALTH" | head -n "$NEED" > "$HEALTH.nodes"
if [ "$(wc -l < "$HEALTH.nodes")" -lt "$NEED" ]; then
  echo "FATAL: only $(wc -l < "$HEALTH.nodes") healthy nodes of the $NEED the config needs" >&2
  exit 1
fi
export NNODES=$NEED
export MASTER_ADDR=$(head -n1 "$HEALTH.nodes")
srun --nodes="$NEED" --nodelist="$HEALTH.nodes" --ntasks-per-node=1 --kill-on-bad-exit=1 "$SCOPE/bridge/scripts/training/dsv4_cpt_worker.sh"
