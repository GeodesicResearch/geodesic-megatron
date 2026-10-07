#!/bin/bash
# The v2e2e arm's pre-launch probe: one job at the production width that measures the fast posture and
# its alternatives on the unfiltered baseline's data, before the arm's stage 1 commits 65 node-days to
# the fast posture. The arm README's "The probe" section has the gates each step decides and how the
# results are read. Its steps are built from scripts/training/probe_job.sh.
#
# Steps, in order (each launch is pipeline_training_launch.sh on 128 nodes, --disable-ft, its own log, and
# a time limit of its own so one hung step cannot take the rest of the allocation):
#   nvlink     every allocated node's NVLink status; the launches use 128 healthy nodes, and once 128
#              are selected the others are registered as bad nodes (the HybridEP dispatcher aborts on a
#              node with a dead link)
#   fast       probe_fast.yaml with the stage's launcher settings: 500 iterations, saves at 150,
#              300, 450 and 500; scored over iterations 101-500
#   handoff    probe_handoff_midtrain.yaml, only when fast succeeded: the arm's midtraining
#              warm-started from fast's last save, 5 iterations; its log must show the load, the fp32
#              SSM-state patch on every rank and the exit at iteration 5
#   as_is      probe_as_is.yaml: the baseline's first 500 iterations, rerun on the same nodes; scored over
#              101-500, then the identical-batch parity test of fast against the baseline and this rerun,
#              gated over PARITY_GATED and reported over PARITY_REPORTED
#   score_gate fast's peak memory over every rank, and fast's step time relative to as_is, against the
#              thresholds pre-registered in the arm's score_gate.yaml (scripts/telemetry/score_gate.py): a
#              score step's own exit status says only that the log could be scored
#
# Everything lands in /projects/a5k/public/logs/control_pretraining/v2e2e_probe/<job id>/: each step's
# log, score_run.py and loss_parity.py JSON, score_gate.py's outcomes, the NVLink sweep and report, the
# handoff's log evidence, and steps.tsv (step, what it ran, exit status, whether it gates). The job exits
# 0 only when every gating step did.
#
# SUBMIT from a frozen copy of the commit under test (the README has the recipe), never from a working
# checkout: bash reads this script and the launcher by offset while they run. 130 nodes: 128 for the
# launches and two spares for nodes the sweep rejects, so the account's node cap must be raised above
# the default 128 for the submission and, exported with it, for the job's own check at start:
#   cd <frozen copy> && ISAMBARD_SBATCH_FORCE=0 ISAMBARD_SBATCH_MAX_NODES=250 isambard_sbatch \
#       configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/probe/probe.sbatch
#
#SBATCH --job-name=cp30b-v2e2e-probe
#SBATCH --nodes=130
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:05:00
#SBATCH --exclusive
#SBATCH --output=/projects/a5k/public/logs/megatron_runs/v2e2e-probe-%j.out
set -uo pipefail

REPO_DIR="${GEODESIC_REPO_DIR:-$SLURM_SUBMIT_DIR}"
export GEODESIC_REPO_DIR="$REPO_DIR"
source "$REPO_DIR/scripts/training/probe_job.sh" || exit 1
PROBE=configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/probe
ARM=configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e
NODES=128
GPUS=$((NODES * 4))
LINKS_PER_GPU=18
HF_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
MODEL=nano
MODE=pretrain
# Every rank of a handoff into a stage that keeps the fp32 SSM state logs this once the patch is in.
FP32_SSM_PATCH='Installed fp32-SSM-state patch (mode=checkpoint)'
# fast's checkpoint.save (probe_fast.yaml); the handoff reads it as its warm start. Its parent also holds
# the export check's clone, so the whole parent must be absent when a probe starts.
SCRATCH=/projects/a5k/public/checkpoints/megatron/v2e2e_probe/fast
OUT=/projects/a5k/public/logs/control_pretraining/v2e2e_probe/$SLURM_JOB_ID
# The identical-batch parity test, pre-registered before the probe runs: gated over iterations 51-500,
# every window inside the band; reported, not gated, over 1-50, which lies in the steep early descent,
# where a window mean mostly measures the rate of descent.
PARITY_GATED="51 500"
PARITY_REPORTED="1 50"
PARITY_WINDOW=50
# Each launch's time limit in seconds, about 1.5x its estimate; together with the sweep and the scoring
# they fit inside the job's --time.
LIMIT_FAST=4200
LIMIT_HANDOFF=1200
LIMIT_AS_IS=4800

probe_check_node_cap
probe_check_start "$SCRATCH"
cd "$REPO_DIR" || exit 1
# The baseline stage 1's own log, the parity test's first reference: the loss gate's `baseline`.
BASELINE_LOG=$(awk '$1 == "baseline:" {print $2}' "$ARM/loss_gate.yaml")
if [ ! -f "$BASELINE_LOG" ]; then
    echo "FATAL: the loss gate's baseline log '$BASELINE_LOG' is not a file" >&2
    exit 1
fi
probe_open_records
probe_select_nodes "v2e2e probe"

# --- fast ---
if launch fast "$PROBE/probe_fast.yaml" "$ARM/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.env" 29611 \
    "$LIMIT_FAST"; then
    FAST_OK=1
else
    FAST_OK=0
fi
score fast "$PROBE/probe_fast.yaml" 101 500

# --- handoff: needs fast's last save ---
if [ "$FAST_OK" -eq 1 ]; then
    launch handoff "$PROBE/probe_handoff_midtrain.yaml" "$ARM/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_midtrain.env" \
        29612 "$LIMIT_HANDOFF"
    check_handoff handoff "$SCRATCH" 5 "$FP32_SSM_PATCH"
else
    record handoff "not run: fast failed" 1
fi

# --- as_is, the memory and speed gates, then the identical-batch parity test ---
launch as_is "$PROBE/probe_as_is.yaml" "" 29613 "$LIMIT_AS_IS"
score as_is "$PROBE/probe_as_is.yaml" 101 500
gate score_gate "score_gate.py: fast's memory, and its speed against as_is" \
    "python scripts/telemetry/score_gate.py --spec $ARM/score_gate.yaml --scores-dir $OUT"
parity_band parity_band "$PARITY_GATED" "$PARITY_WINDOW" "$OUT/fast.out" "$BASELINE_LOG" "$OUT/as_is.out"
record parity_band "loss_parity.py band: fast against the baseline and as_is, $PARITY_GATED, window $PARITY_WINDOW" $?
parity_band parity_band_reported "$PARITY_REPORTED" "$PARITY_WINDOW" "$OUT/fast.out" "$BASELINE_LOG" "$OUT/as_is.out"
note parity_band_reported \
    "loss_parity.py band: fast against the baseline and as_is, $PARITY_REPORTED, window $PARITY_WINDOW" $?

probe_finish
