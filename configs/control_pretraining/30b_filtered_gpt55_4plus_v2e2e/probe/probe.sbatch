#!/bin/bash
# The v2e2e arm's pre-launch probe: one job at the production width that measures the fast posture and
# its alternatives on the unfiltered baseline's data, before the arm's stage 1 commits 65 node-days to
# the fast posture. The arm README's "The probe" section has the gates each step decides and how the
# results are read.
#
# Steps, in order (each launch is pipeline_training_launch.sh on 128 nodes, --disable-ft, its own log):
#   nvlink     every allocated node's NVLink status; the launches use 128 healthy nodes and the others
#              are registered as bad nodes (the HybridEP dispatcher aborts on a node with a dead link)
#   fast       probe_fast.yaml with the stage's launcher settings: 500 iterations, saves at 150,
#              300, 450 and 500; scored over iterations 101-500
#   handoff    probe_handoff_midtrain.yaml, only when fast succeeded: the as-is midtraining
#              warm-started from fast's last save, 5 iterations
#   as_is      probe_as_is.yaml: the baseline's first 500 iterations, rerun; scored over 101-500, then
#              the identical-batch parity test of fast against the baseline and this rerun, gated over
#              PARITY_GATED and reported over PARITY_REPORTED
#   precise    probe_precise.yaml with its launcher settings: 300 iterations; scored over 101-300
#
# Everything lands in /projects/a5k/public/logs/control_pretraining/v2e2e_probe/<job id>/: each step's
# log, score_run.py and loss_parity.py JSON, the NVLink sweep and report, the handoff's log evidence,
# and steps.tsv (step, what it ran, exit status, whether it gates). The job exits 0 only when every
# gating step did.
#
# SUBMIT from a frozen copy of the commit under test (the README has the recipe), never from a working
# checkout: bash reads this script and the launcher by offset while they run. 130 nodes: 128 for the
# launches and two spares for nodes the sweep rejects.
#   cd <frozen copy> && ISAMBARD_SBATCH_FORCE=0 isambard_sbatch \
#       configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/probe/probe.sbatch
#
#SBATCH --job-name=cp30b-v2e2e-probe
#SBATCH --nodes=130
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:45:00
#SBATCH --exclusive
#SBATCH --output=/projects/a5k/public/logs/megatron_runs/v2e2e-probe-%j.out
set -uo pipefail

# The account's node cap is checked again at start, as pipeline_training_submit.sbatch does: a job
# that would take the account over it cancels itself.
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null
    exit 1
fi

REPO_DIR="${GEODESIC_REPO_DIR:-$SLURM_SUBMIT_DIR}"
export GEODESIC_REPO_DIR="$REPO_DIR"
PROBE=configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/probe
ARM=configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e
NODES=128
GPUS=$((NODES * 4))
LINKS_PER_GPU=18
HF_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
# fast's checkpoint.save (probe_fast.yaml); the handoff reads it as its warm start.
SCRATCH=/projects/a5k/public/checkpoints/megatron/v2e2e_probe/fast
OUT=/projects/a5k/public/logs/control_pretraining/v2e2e_probe/$SLURM_JOB_ID
# The identical-batch parity test, pre-registered before the probe runs: gated over iterations 51-500,
# every window inside the band; reported, not gated, over 1-50, which lies in the steep early descent,
# where a window mean mostly measures the rate of descent.
PARITY_GATED="51 500"
PARITY_REPORTED="1 50"

if [ ! -f "$REPO_DIR/REVISION" ]; then
    echo "FATAL: $REPO_DIR has no REVISION file; submit from a frozen copy of the commit (see the arm README)" >&2
    exit 1
fi
if [ -e "$SCRATCH" ]; then
    echo "FATAL: $SCRATCH exists; a probe starts from an empty scratch directory" >&2
    exit 1
fi
cd "$REPO_DIR" || exit 1
# The baseline stage 1's own log, the parity test's first reference: the loss gate's `baseline`.
BASELINE_LOG=$(awk '$1 == "baseline:" {print $2}' "$ARM/loss_gate.yaml")
if [ ! -f "$BASELINE_LOG" ]; then
    echo "FATAL: the loss gate's baseline log '$BASELINE_LOG' is not a file" >&2
    exit 1
fi
mkdir -p "$OUT/nvlink"
echo "[probe] job $SLURM_JOB_ID, code $(head -n 1 REVISION), results in $OUT"
printf 'step\tran\texit\tgates\n' > "$OUT/steps.tsv"
FAILED=0

note() {  # step, what it ran, exit status: recorded, and never decides the job's exit status
    printf '%s\t%s\t%s\tno\n' "$1" "$2" "$3" >> "$OUT/steps.tsv"
    echo "[probe] $(date -u +%FT%TZ) $1: exit $3 ($2; reported, not gating)"
}

record() {  # step, what it ran, exit status: a nonzero status fails the job
    printf '%s\t%s\t%s\tyes\n' "$1" "$2" "$3" >> "$OUT/steps.tsv"
    echo "[probe] $(date -u +%FT%TZ) $1: exit $3 ($2)"
    [ "$3" -eq 0 ] || FAILED=1
}

in_container() {  # one command string, run from the repo root inside the training container
    "$REPO_DIR/pipeline_env_exec.sh" "cd $REPO_DIR && source pipeline_env_activate.sh >/dev/null && $1"
}

# --- nvlink: sweep every node, register the unhealthy ones, launch on 128 healthy ones ---
srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -c "nvidia-smi nvlink --status > $OUT/nvlink/\$(hostname -s).txt"
record nvlink_sweep "nvidia-smi nvlink --status on $SLURM_NNODES nodes" $?
in_container "python scripts/training/nvlink_health.py --status-dir $OUT/nvlink --gpus-per-node 4 \
    --links-per-gpu $LINKS_PER_GPU --select $NODES --nodelist-out $OUT/nodelist.txt \
    --report-out $OUT/nvlink_report.json" | tee "$OUT/nvlink_health.out"
SELECT_STATUS=${PIPESTATUS[0]}
while read -r _ host reason; do
    isambard_sbatch --mark-bad "$host" "NVLink: $reason (v2e2e probe $SLURM_JOB_ID)"
done < <(grep '^UNHEALTHY ' "$OUT/nvlink_health.out")
record nvlink_select "scripts/training/nvlink_health.py --select $NODES" "$SELECT_STATUS"
if [ "$SELECT_STATUS" -ne 0 ]; then
    echo "FATAL: fewer than $NODES healthy nodes; nothing launched" >&2
    exit 1
fi
NODELIST=$(cat "$OUT/nodelist.txt")

launch() {  # step, config, launcher settings file ("" for none), master port
    local step=$1 config=$2 env_file=$3 port=$4 status
    (
        if [ -n "$env_file" ]; then
            export ISAMBARD_ENV_OVERRIDES="$REPO_DIR/$env_file"
        else
            unset ISAMBARD_ENV_OVERRIDES
        fi
        export MASTER_PORT_OVERRIDE="$port"
        bash "$REPO_DIR/pipeline_training_launch.sh" "$config" --model nano --mode pretrain --disable-ft \
            --nodes "$NODES" --nodelist "$NODELIST"
    ) > "$OUT/$step.out" 2>&1
    status=$?
    record "$step" "$config${env_file:+ with $env_file}" "$status"
    return "$status"
}

score() {  # step, config, first, last: score_run.py over [first, last], loss over its last 50
    local step=$1 config=$2 first=$3 last=$4
    in_container "python scripts/telemetry/score_run.py $OUT/$step.out --config $config --hf-model $HF_MODEL \
        --gpus $GPUS --window $first $last --loss-window $((last - 49)) $last --wandb-peak-memory --json \
        > $OUT/$step.score.json"
    record "$step.score" "score_run.py --window $first $last" $?
}

# --- fast ---
if launch fast "$PROBE/probe_fast.yaml" "$ARM/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.env" 29611; then
    FAST_OK=1
else
    FAST_OK=0
fi
score fast "$PROBE/probe_fast.yaml" 101 500

# --- handoff: needs fast's last save ---
if [ "$FAST_OK" -eq 1 ]; then
    launch handoff "$PROBE/probe_handoff_midtrain.yaml" "" 29612
    {
        printf 'loaded_from_probe_save\t%s\n' "$(grep -c "successfully loaded checkpoint from $SCRATCH" "$OUT/handoff.out")"
        printf 'fp32_ssm_patch_installed_ranks\t%s\n' "$(grep -c 'Installed fp32-SSM-state patch (mode=checkpoint)' "$OUT/handoff.out")"
        printf 'exited_at_iteration_5\t%s\n' "$(grep -c 'exiting program at iteration 5' "$OUT/handoff.out")"
    } > "$OUT/handoff.evidence.tsv"
else
    record handoff "not run: fast failed" 1
fi

# --- as_is, then the identical-batch parity test ---
launch as_is "$PROBE/probe_as_is.yaml" "" 29613
score as_is "$PROBE/probe_as_is.yaml" 101 500
parity_band() {  # output name, iterations: loss_parity.py band of fast against the baseline and as_is
    in_container "python scripts/telemetry/loss_parity.py band --reference $BASELINE_LOG $OUT/as_is.out \
        --candidate $OUT/fast.out --iterations $2 --window 50 --wandb --json > $OUT/$1.json"
}
parity_band parity_band "$PARITY_GATED"
record parity_band "loss_parity.py band: fast against the baseline and as_is, $PARITY_GATED, window 50" $?
parity_band parity_band_reported "$PARITY_REPORTED"
note parity_band_reported "loss_parity.py band: fast against the baseline and as_is, $PARITY_REPORTED, window 50" $?

# --- precise ---
launch precise "$PROBE/probe_precise.yaml" "$ARM/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain_precise.env" 29614
score precise "$PROBE/probe_precise.yaml" 101 300

echo "[probe] done; steps:"
cat "$OUT/steps.tsv"
exit "$FAILED"
