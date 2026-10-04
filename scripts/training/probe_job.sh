#!/bin/bash
# The steps a production-width probe job is built from: an sbatch script that runs several training launches in one
# allocation, each with its own log and time limit, then scores and gates them. Sourced by the probe sbatch scripts
# under configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/probe/, which set, before calling anything here:
#   REPO_DIR       the frozen copy of the code under test; it carries a REVISION file
#   OUT            the job's results directory
#   NODES, GPUS    the launches' node count, selected from the allocation by NVLink health, and their GPU count
#   LINKS_PER_GPU  the number of active NVLinks of a healthy GPU
#   HF_MODEL       the HF model whose config.json the scoring reads the model FLOPs from
#   MODEL, MODE    the launcher's --model and --mode
# Every step leaves a line in $OUT/steps.tsv (step, what it ran, exit status, whether it gates). FAILED becomes 1
# when a gating step fails, and probe_finish exits with it, so the job exits 0 only when every gating step did.

probe_check_node_cap() {  # the account's node cap, checked again at start as pipeline_training_submit.sbatch does
    if ! isambard_sbatch --check; then
        scancel "$SLURM_JOB_ID" 2>/dev/null
        exit 1
    fi
}

probe_check_start() {  # scratch directory: exit unless the code is a frozen copy and the probe starts clean
    local scratch=$1
    if [ ! -f "$REPO_DIR/REVISION" ]; then
        echo "FATAL: $REPO_DIR has no REVISION file; submit from a frozen copy of the code (see the arm README)" >&2
        exit 1
    fi
    if [ -e "$(dirname "$scratch")" ]; then
        echo "FATAL: $(dirname "$scratch") exists; a probe starts from an empty scratch directory" >&2
        exit 1
    fi
    # Each step's posture is its config, its launcher settings file and the committed container config, and
    # nothing else: a launch setting inherited from the submitting shell is refused.
    python3 "$REPO_DIR/scripts/training/launch_environment.py" || exit 1
}

probe_open_records() {  # create $OUT and its steps.tsv, and name the code under test
    mkdir -p "$OUT/nvlink"
    echo "[probe] job $SLURM_JOB_ID, code $(head -n 1 "$REPO_DIR/REVISION"), results in $OUT"
    printf 'step\tran\texit\tgates\n' > "$OUT/steps.tsv"
    FAILED=0
}

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

gate() {  # step, what it checks, command string: run in the container, its output kept in $OUT/<step>.out; gating
    local status
    in_container "$3" | tee "$OUT/$1.out"
    status=${PIPESTATUS[0]}
    record "$1" "$2" "$status"
    return "$status"
}

probe_select_nodes() {  # job label: sweep every node's NVLink, select NODES healthy ones into NODELIST
    # A node whose nvidia-smi fails leaves an empty status file, which the health check judges unhealthy, so the
    # sweep's own exit status is recorded without gating: the selection is the gate.
    srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 \
        bash -c "nvidia-smi nvlink --status > $OUT/nvlink/\$(hostname -s).txt"
    note nvlink_sweep "nvidia-smi nvlink --status on $SLURM_NNODES nodes" $?
    if ! gate nvlink_select "scripts/training/nvlink_health.py --select $NODES" \
        "python scripts/training/nvlink_health.py --status-dir $OUT/nvlink --gpus-per-node $((GPUS / NODES)) \
            --links-per-gpu $LINKS_PER_GPU --select $NODES --nodelist-out $OUT/nodelist.txt \
            --report-out $OUT/nvlink_report.json"; then
        # Too few healthy nodes points at the sweep or the check rather than at the nodes: none is registered, so
        # a systemic misreading cannot exclude healthy nodes from every submission for a week.
        echo "FATAL: fewer than $NODES healthy nodes; nothing launched and no node registered bad" >&2
        exit 1
    fi
    # The HybridEP dispatcher aborts on a node with a dead link, so every node the selection left out is registered.
    while read -r _ host reason; do
        isambard_sbatch --mark-bad "$host" "NVLink: $reason ($1 $SLURM_JOB_ID)"
    done < <(grep '^UNHEALTHY ' "$OUT/nvlink_select.out")
    NODELIST=$(cat "$OUT/nodelist.txt")
}

launch() {  # step, config, launcher settings file ("" for none), master port, time limit in seconds; gating
    local step=$1 config=$2 env_file=$3 port=$4 limit=$5 status
    (
        if [ -n "$env_file" ]; then
            export ISAMBARD_ENV_OVERRIDES="$REPO_DIR/$env_file"
        fi
        export MASTER_PORT_OVERRIDE="$port"
        # timeout signals its whole process group, srun included, which ends the step on every node.
        timeout --kill-after=120 "$limit" bash "$REPO_DIR/pipeline_training_launch.sh" "$config" \
            --model "$MODEL" --mode "$MODE" --disable-ft --nodes "$NODES" --nodelist "$NODELIST"
    ) > "$OUT/$step.out" 2>&1
    status=$?
    record "$step" "$config${env_file:+ with $env_file}, limit ${limit}s" "$status"
    return "$status"
}

score() {  # step, config, first, last: score_run.py over [first, last], loss over its last 50, into <step>.score.json
    local step=$1 config=$2 first=$3 last=$4
    in_container "python scripts/telemetry/score_run.py $OUT/$step.out --config $config --hf-model $HF_MODEL \
        --gpus $GPUS --window $first $last --loss-window $((last - 49)) $last --wandb-peak-memory --json \
        > $OUT/$step.score.json"
    record "$step.score" "score_run.py --window $first $last" $?
}

check_handoff() {  # step, the save it warm-starts from, its last iteration, a line every rank logs: gating
    # What the handoff's log must show: the load from the save, the per-rank line on all GPUS ranks, and the exit.
    local step=$1 save=$2 last=$3 per_rank=$4 loaded ranks exited
    loaded=$(grep -c "successfully loaded checkpoint from $save" "$OUT/$step.out")
    ranks=$(grep -cF "$per_rank" "$OUT/$step.out")
    exited=$(grep -c "exiting program at iteration $last" "$OUT/$step.out")
    printf 'loaded_from_probe_save\t%s\nranks_logging_per_rank_line\t%s\nexited_at_last_iteration\t%s\n' \
        "$loaded" "$ranks" "$exited" > "$OUT/$step.evidence.tsv"
    [ "$loaded" -ge 1 ] && [ "$ranks" -eq "$GPUS" ] && [ "$exited" -ge 1 ]
    record "${step}_evidence" "loaded $loaded, '$per_rank' on $ranks of $GPUS ranks, exited at $last: $exited" $?
}

parity_band() {  # output name, iterations, window, candidate log, reference logs...: loss_parity.py band, <name>.json
    local name=$1 iterations=$2 window=$3 candidate=$4
    shift 4
    in_container "python scripts/telemetry/loss_parity.py band --reference $* --candidate $candidate \
        --iterations $iterations --window $window --wandb --json > $OUT/$name.json"
}

probe_finish() {  # print every step's outcome and exit with the gating steps' verdict
    echo "[probe] done; steps:"
    cat "$OUT/steps.tsv"
    exit "$FAILED"
}
