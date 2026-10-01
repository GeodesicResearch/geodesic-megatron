#!/bin/bash
# The v2e2e arm's midtraining probe: one job at the production width that measures the fast midtraining
# configuration (the Nano midtraining quickstart's levers) on the unfiltered baseline's stage 2 before this arm's
# midtraining runs in it. The arm README's "The midtraining probe" has the gates each step decides. Its steps are
# built from scripts/training/probe_job.sh.
#
# Steps, in order (each launch is pipeline_training_launch.sh on 128 nodes, --disable-ft, its own log, and a time
# limit of its own so one hung step cannot take the rest of the allocation):
#   nvlink        every allocated node's NVLink status; the launches use 128 healthy nodes, and once 128 are
#                 selected the others are registered as bad nodes
#   fast_mid      probe_midtrain_fast.yaml with the arm's midtraining settings file: 500 iterations, saves at 150,
#                 300, 450 and 500; scored over iterations 101-500
#   handoff_cpt   probe_midtrain_handoff_cpt.yaml, only when fast_mid succeeded: the as-is continual pretraining
#                 warm-started from fast_mid's last save, 5 iterations; its log must show the load, the fp32
#                 SSM-state patch on every rank and the exit at iteration 5
#   as_is_mid     probe_midtrain_as_is.yaml: the baseline's stage 2 rerun for 500 iterations on the same nodes;
#                 scored over 101-500
#   score_gate    fast_mid's peak memory over every rank, its speed-up against as_is_mid, and its iteration-1 loss
#                 against as_is_mid's, against the thresholds pre-registered in the arm's score_gate_midtrain.yaml
#   parity        the identical-batch band of fast_mid against production's stage 2 (REFERENCE_LOG) and as_is_mid,
#                 gated over PARITY_GATED and reported over PARITY_REPORTED. Pre-registered rule: no gated window
#                 outside passes; exactly one outside runs as_is_mid_2 (as_is_mid again, same nodes) and the band
#                 with the three references decides; two or more outside, or a failure that is not a window (the
#                 schedule, a NaN or skipped iteration, a run that could not be read), fail
#
# Everything lands in /projects/a5k/public/logs/control_pretraining/v2e2e_probe_midtrain/<job id>/: each step's log,
# score_run.py, score_gate.py and loss_parity.py output, the NVLink sweep and report, the handoff's log evidence, and
# steps.tsv (step, what it ran, exit status, whether it gates). The job exits 0 only when every gating step did.
#
# SUBMIT from a frozen copy of the code under test (the README has the recipe), never from a working checkout:
#   cd <frozen copy> && ISAMBARD_SBATCH_FORCE=0 ISAMBARD_SBATCH_MAX_NODES=256 isambard_sbatch \
#       configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/probe/probe_midtrain.sbatch
#
#SBATCH --job-name=cp30b-v2e2e-probe-mid
#SBATCH --nodes=130
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:45:00
#SBATCH --exclusive
#SBATCH --output=/projects/a5k/public/logs/megatron_runs/v2e2e-probe-mid-%j.out
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
# fast_mid's checkpoint.save (probe_midtrain_fast.yaml); the handoff reads its iteration 500 as its warm start. The
# whole parent must be absent when a probe starts.
SCRATCH=/projects/a5k/public/checkpoints/megatron/v2e2e_probe_midtrain/fast
OUT=/projects/a5k/public/logs/control_pretraining/v2e2e_probe_midtrain/$SLURM_JOB_ID
# Production's stage 2 of the baseline (job 6127737, W&B geodesic/megatron_training/qeslzwcc): the run whose
# batches every probe run reads, and the parity band's first reference.
REFERENCE_LOG=/projects/a5k/public/data_cwtice.a5k/logs/megatron_runs/train-6127737.out
# The identical-batch parity test, pre-registered before the probe runs: gated over iterations 51-500, every window
# inside the band; reported, not gated, over 1-50, which the warm start's change of context length dominates.
PARITY_GATED="51 500"
PARITY_REPORTED="1 50"
PARITY_WINDOW=50
# Each launch's time limit in seconds, about 1.5x its estimate; together with the sweep and the scoring they fit
# inside the job's --time.
LIMIT_FAST_MID=3600
LIMIT_HANDOFF=1200
LIMIT_AS_IS_MID=5400

probe_check_node_cap
if [ ! -f "$REFERENCE_LOG" ]; then
    echo "FATAL: the reference log '$REFERENCE_LOG' is not a file" >&2
    exit 1
fi
probe_check_start "$SCRATCH"
cd "$REPO_DIR" || exit 1
probe_open_records
probe_select_nodes "v2e2e midtraining probe"

# --- fast_mid ---
if launch fast_mid "$PROBE/probe_midtrain_fast.yaml" \
    "$ARM/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_midtrain.env" 29621 "$LIMIT_FAST_MID"; then
    FAST_OK=1
else
    FAST_OK=0
fi
score fast_mid "$PROBE/probe_midtrain_fast.yaml" 101 500

# --- handoff_cpt: needs fast_mid's last save ---
if [ "$FAST_OK" -eq 1 ]; then
    launch handoff_cpt "$PROBE/probe_midtrain_handoff_cpt.yaml" "" 29622 "$LIMIT_HANDOFF"
    check_handoff handoff_cpt "$SCRATCH" 5 "$FP32_SSM_PATCH"
else
    record handoff_cpt "not run: fast_mid failed" 1
fi

# --- as_is_mid, the memory, speed and first-loss gates, then the identical-batch parity test ---
launch as_is_mid "$PROBE/probe_midtrain_as_is.yaml" "" 29623 "$LIMIT_AS_IS_MID"
score as_is_mid "$PROBE/probe_midtrain_as_is.yaml" 101 500
gate score_gate "score_gate.py: fast_mid's memory, speed-up against as_is_mid and iteration-1 loss" \
    "python scripts/telemetry/score_gate.py --spec $ARM/score_gate_midtrain.yaml --scores-dir $OUT"

parity_band parity_band "$PARITY_GATED" "$PARITY_WINDOW" "$OUT/fast_mid.out" "$REFERENCE_LOG" "$OUT/as_is_mid.out"
PARITY_STATUS=$?
if [ "$PARITY_STATUS" -eq 0 ]; then
    record parity_band \
        "loss_parity.py band: fast_mid against 6127737 and as_is_mid, $PARITY_GATED, window $PARITY_WINDOW" 0
else
    # The lm-loss windows outside the band, or "other" when the band failed on something else (the schedule, a NaN
    # or skipped iteration); a report that cannot be read leaves the file empty, which is not "1" either.
    in_container "python scripts/telemetry/loss_parity.py windows-outside $OUT/parity_band.json \
        > $OUT/parity_band.windows_outside"
    OUTSIDE=$(cat "$OUT/parity_band.windows_outside")
    note parity_band "loss_parity.py band, two references, $PARITY_GATED: windows outside $OUTSIDE" "$PARITY_STATUS"
    if [ "$OUTSIDE" = "1" ]; then
        launch as_is_mid_2 "$PROBE/probe_midtrain_as_is.yaml" "" 29624 "$LIMIT_AS_IS_MID"
        parity_band parity_band_3ref "$PARITY_GATED" "$PARITY_WINDOW" "$OUT/fast_mid.out" \
            "$REFERENCE_LOG" "$OUT/as_is_mid.out" "$OUT/as_is_mid_2.out"
        record parity_band_3ref "loss_parity.py band: fast_mid against 6127737, as_is_mid and as_is_mid_2, \
$PARITY_GATED, window $PARITY_WINDOW" $?
    else
        record parity_band "two or more windows outside, or a failure that is not a window: no third reference" 1
    fi
fi
parity_band parity_band_reported "$PARITY_REPORTED" "$PARITY_WINDOW" "$OUT/fast_mid.out" "$REFERENCE_LOG" \
    "$OUT/as_is_mid.out"
note parity_band_reported "loss_parity.py band: fast_mid against 6127737 and as_is_mid, $PARITY_REPORTED" $?

probe_finish
