#!/bin/bash
# E2E test inoculation_midtraining_token_masking: submit one stage of the test. README.md is the guide; it says when
# each stage runs and what to check before the next.
#
# This script submits through the standard entry points. What each corpus, arm, probe, export and gate is lives in the
# configs beside it (data/*.yaml, arm_*.yaml, probe.yaml, export.yaml, gate.yaml), and every value it needs from them
# is read from those files in one Python call (read_settings below): each arm composed through
# scripts/training/config_compose.py as the launcher composes it. It decides only its own scheduling (the time limits,
# the node counts, the smoke's length and W&B name) and where a run's record goes.
#
# Run it from the root of a frozen copy of the commit under test (tests/e2e_tests/README.md, "How one is run"), never
# from a working checkout: every job reads that copy, and bash reads a running script by offset. It refuses a directory
# without a REVISION file and a shell carrying launch settings (scripts/training/launch_environment.py).
#
#   bash tests/e2e_tests/inoculation_midtraining_token_masking/submit.sh <stage> [arm]
#
# Stages, in order:
#   data       prepare and tokenize the six inoculation-midtraining corpora (carving off their held-out 0.2%), the
#              ClimbMix replay, and the held-out set
#   preflight  the parent's dead embedding rows, and every corpus's documents checked against them
#   base       the probe of the untrained parent (B)
#   smoke      the masked arm for a few iterations, saving nothing: the warm start, the setup checks and the step-0
#              masked validation on the fast posture, before the two full runs
#   train      both arms, or the one named (masked or control): 16 nodes each
#   evaluate   export each arm's final checkpoint (or the named arm's), then probe it (M, C)
#   gate       the pre-registered verdict over the run directory, run here in the container
# Every submission is recorded in the run directory's jobs.tsv. DRY_RUN=1 prints the submissions, and what would be
# refused, and changes nothing.
set -euo pipefail

STAGE="${1:?usage: submit.sh <data|preflight|base|smoke|train|evaluate|gate> [arm]}"
DRY_RUN="${DRY_RUN:-0}"

TEST_DIR=tests/e2e_tests/inoculation_midtraining_token_masking
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR"

if [ ! -f REVISION ]; then
    echo "FATAL: $REPO_DIR has no REVISION file; run this from a frozen copy of the commit (README.md)" >&2
    exit 1
fi
# The jobs' posture is their configs and the quickstart's .env, never a setting exported in this shell.
python3 scripts/training/launch_environment.py
# Every job is submitted forced (Kyle, 2026-10-10: every metagaming-team submission, the two 16-node arms included):
# isambard_sbatch skips its account node-cap check and still excludes the bad nodes. README.md, "The amendment of
# 2026-10-10".
export ISAMBARD_SBATCH_FORCE=1
# Every sbatch wrapper reads its code from GEODESIC_REPO_DIR: this frozen copy.
export GEODESIC_REPO_DIR="$REPO_DIR"

ARMS=(masked control)
if [ -n "${2:-}" ]; then
    [[ " ${ARMS[*]} " == *" $2 "* ]] || { echo "FATAL: unknown arm '$2'; the arms are ${ARMS[*]}" >&2; exit 1; }
    ARMS=("$2")
fi

# Every value the stages take from the configs, as shell assignments: refuses arms that disagree on what they share
# and configs that do not fit together, so a stage never submits from an inconsistent set.
read_settings() {
    python3 - "$TEST_DIR" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")
from scripts.training.config_compose import BASE_CONFIG_KEY, load_composed_yaml  # noqa: E402

test_dir = Path(sys.argv[1])
arms = {arm: load_composed_yaml(test_dir / f"arm_{arm}.yaml") for arm in ("masked", "control")}
data = {name: load_composed_yaml(test_dir / "data" / f"{name}.yaml") for name in ("climbmix_replay", "held_out")}
export = load_composed_yaml(test_dir / "export.yaml")
gate = load_composed_yaml(test_dir / "gate.yaml")


def fail(message):
    sys.exit(f"FATAL: {message}")


def at(config, dotted):
    for key in dotted.split("."):
        config = config[key]
    return config


def shared(dotted):
    """The value both arms state at ``dotted``; they train one recipe, so a difference is a config fault."""
    values = {arm: at(config, dotted) for arm, config in arms.items()}
    if values["masked"] != values["control"]:
        fail(f"the arms differ in {dotted}: {values}")
    return values["masked"]


data_path = [str(item) for item in shared("dataset.data_path")]
prefixes = [Path(prefix) for prefix in data_path[1::2]]
roots = {prefix.parent.parent for prefix in prefixes}
names = {prefix.name for prefix in prefixes}
if len(roots) != 1 or len(names) != 1 or not next(iter(names)).endswith("_input_document"):
    fail(f"the blend's prefixes are not <data root>/<corpus>/<variant>_input_document: {data_path[1::2]}")
(data_root,), (prefix_name,) = roots, names
blended = [prefix.parent.name for prefix in prefixes]
replay = Path(data["climbmix_replay"]["output-dir"])
held_out = Path(data["held_out"]["output-dir"])
if replay.parent != data_root or replay.name not in blended:
    fail(f"the replay's output {replay} is not a corpus of the blend under {data_root}")
held_out_prefix = held_out / prefix_name
if held_out.parent != data_root or Path(shared("token_masking.masked_validation.data_path")) != held_out_prefix:
    fail(f"masked validation does not read the held-out set's {held_out_prefix}")
base_dirs = [
    identity["expect"]["model.path"]
    for identity in gate["probe_identity"].values()
    if identity["probe"] == "base.json" and "model.path" in identity["expect"]
]
if len(base_dirs) != 1:
    fail(f"gate.yaml must state the base probe's model.path once, not {base_dirs}")
# The arms overlay the quickstart; its launcher settings are the .env beside it.
quickstart = (test_dir / yaml.safe_load((test_dir / "arm_common.yaml").read_text())[BASE_CONFIG_KEY]).resolve()

settings = {
    "TOKENIZER": shared("tokenizer.tokenizer_model"),
    "PARENT": shared("checkpoint.pretrained_checkpoint"),
    "TRAIN_ITERS": shared("train.train_iters"),
    "DATA_ROOT": data_root,
    "OUTPUT_VARIANT": prefix_name.removesuffix("_input_document"),
    "REPLAY": replay.name,
    "HELD_OUT": held_out.name,
    "PARENT_HF": base_dirs[0],
    "EXPORT_ARCHITECTURE": export["architecture"],
    "EXPORT_TP": export["tp"],
    "EXPORT_EP": export["ep"],
    "FAST_POSTURE_ENV": quickstart.with_suffix(".env"),
}
for name, value in settings.items():
    print(f"{name}={shlex.quote(str(value))}")
print("IMID_SUBSETS=(" + " ".join(shlex.quote(corpus) for corpus in blended if corpus != replay.name) + ")")
for name, dotted in (("SAVE", "checkpoint.save"), ("INDEX_CACHE", "dataset.path_to_cache")):
    pairs = " ".join(f"[{arm}]={shlex.quote(str(at(config, dotted)))}" for arm, config in arms.items())
    print(f"declare -A {name}=({pairs})")
PY
}
SETTINGS=$(read_settings)
eval "$SETTINGS"

CORPORA=("${IMID_SUBSETS[@]}" "$REPLAY" "$HELD_OUT")
CODE=$(head -c 12 REVISION)
RUN_DIR=/projects/a5k/public/logs/e2e_tests/inoculation_midtraining_token_masking/$CODE
TRAINING_LOGS=/projects/a5k/public/logs/megatron_runs
SMOKE_ITERATIONS=10
SMOKE_WANDB_NAME=e2e_inoculation_midtraining_token_masking_smoke
FINAL_ITERATION=$(printf "iter_%07d" "$TRAIN_ITERS")

refuse() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] would refuse: $*"
        return
    fi
    echo "FATAL: $*" >&2
    exit 1
}

open_run_dir() {
    echo "run directory: $RUN_DIR"
    [ "$DRY_RUN" = "1" ] && return
    # The data and checkpoint wrappers write their SLURM logs to logs/slurm/ under this copy.
    mkdir -p "$RUN_DIR" logs/slurm
    head -n 1 REVISION > "$RUN_DIR/REVISION"
}

# Submit through isambard_sbatch, record the job in jobs.tsv and print its id. The wrapper's banners print first,
# hence the tail; a non-numeric result is a failed submission and must not become a dependency.
submit() {
    local description="$1" out jobid
    shift
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] $description: ISAMBARD_SBATCH_FORCE=$ISAMBARD_SBATCH_FORCE ${ISAMBARD_ENV_OVERRIDES:+ISAMBARD_ENV_OVERRIDES=$ISAMBARD_ENV_OVERRIDES }isambard_sbatch $*" >&2
        echo "DRYRUN-${description// /-}"
        return
    fi
    out=$(isambard_sbatch --parsable "$@")
    jobid=$(printf '%s\n' "$out" | tail -n 1 | tr -d '[:space:]')
    if ! [[ "$jobid" =~ ^[0-9]+$ ]]; then
        echo "FATAL: submission failed for $description; sbatch said:" >&2
        printf '%s\n' "$out" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\t%s\n' "$STAGE" "$description" "$jobid" "$(date -u +%FT%TZ)" >> "$RUN_DIR/jobs.tsv"
    echo "$jobid"
}

probe() {  # model directory, probe name, the job it waits for ("" for none)
    local dependency=()
    [ -z "$3" ] || dependency=(--dependency="afterok:$3")
    submit "probe $2" ${dependency[@]+"${dependency[@]}"} --gpus-per-node=1 --time=02:00:00 \
        --job-name="e2e-imid-probe-$2" --output="$RUN_DIR/probe-$2-%j.out" \
        pipeline_coherence_submit.sbatch "$1" --probe-spec "$REPO_DIR/$TEST_DIR/probe.yaml" \
        --probe-output-dir "$RUN_DIR" --probe-name "$2"
}

stage_data() {
    local corpus subset arm prepare tokenize imid_prepares=()
    for corpus in "${CORPORA[@]}"; do
        [ ! -e "$DATA_ROOT/$corpus" ] || refuse "$DATA_ROOT/$corpus exists; a rebuild starts from a moved-away directory"
    done
    # Megatron's index cache is keyed by the prefixes, sample count and seed, never by the corpus's content: a rebuilt
    # corpus beside an old cache would be read through indices built over the old .bin files.
    for arm in "${!INDEX_CACHE[@]}"; do
        [ ! -e "${INDEX_CACHE[$arm]}" ] || refuse "${INDEX_CACHE[$arm]} exists; move it away with the corpora it indexes"
    done
    open_run_dir
    for subset in "${IMID_SUBSETS[@]}"; do
        prepare=$(submit "prepare $subset" --time=01:00:00 --job-name="e2e-imid-prep-$subset" \
            pipeline_data_submit.sbatch prepare --config "$TEST_DIR/data/inoculation_midtraining.yaml" \
            --subset "$subset" --output-dir "$DATA_ROOT/$subset" --tokenizer "$TOKENIZER")
        imid_prepares+=("$prepare")
        tokenize=$(submit "tokenize $subset" --dependency="afterok:$prepare" --time=01:00:00 \
            --job-name="e2e-imid-tok-$subset" \
            pipeline_data_submit.sbatch tokenize "$DATA_ROOT/$subset" "$TOKENIZER" "$OUTPUT_VARIANT")
        echo "$subset: prepare $prepare, tokenize $tokenize"
    done
    prepare=$(submit "prepare $REPLAY" --time=04:00:00 --job-name="e2e-imid-prep-$REPLAY" \
        pipeline_data_submit.sbatch prepare --config "$TEST_DIR/data/climbmix_replay.yaml" --tokenizer "$TOKENIZER")
    tokenize=$(submit "tokenize $REPLAY" --dependency="afterok:$prepare" --time=04:00:00 \
        --job-name="e2e-imid-tok-$REPLAY" \
        pipeline_data_submit.sbatch tokenize "$DATA_ROOT/$REPLAY" "$TOKENIZER" "$OUTPUT_VARIANT")
    echo "$REPLAY: prepare $prepare, tokenize $tokenize"
    # The held-out set is the six prepares' validation files.
    prepare=$(submit "prepare $HELD_OUT" --dependency="afterok:$(IFS=:; echo "${imid_prepares[*]}")" \
        --time=00:30:00 --job-name="e2e-imid-prep-$HELD_OUT" \
        pipeline_data_submit.sbatch prepare --config "$TEST_DIR/data/held_out.yaml" --tokenizer "$TOKENIZER")
    tokenize=$(submit "tokenize $HELD_OUT" --dependency="afterok:$prepare" --time=00:30:00 \
        --job-name="e2e-imid-tok-$HELD_OUT" \
        pipeline_data_submit.sbatch tokenize "$DATA_ROOT/$HELD_OUT" "$TOKENIZER" "$OUTPUT_VARIANT")
    echo "$HELD_OUT: prepare $prepare, tokenize $tokenize"
}

stage_preflight() {
    local preflight="$DATA_ROOT/preflight" checks="" corpus jobid
    [ ! -e "$preflight" ] || refuse "$preflight exists; move it away to check again"
    for corpus in "${CORPORA[@]}"; do
        checks+=" && python scripts/data/filter_zero_emb_docs.py --input $DATA_ROOT/$corpus/training.jsonl"
        checks+=" --output $preflight/$corpus.jsonl --tokenizer $TOKENIZER --zero-ids-file $preflight/dead_ids.txt"
    done
    open_run_dir
    [ "$DRY_RUN" = "1" ] || mkdir -p "$preflight"
    jobid=$(submit "preflight" --nodes=1 --gpus-per-node=1 --exclusive --time=02:00:00 \
        --job-name=e2e-imid-preflight --output="$preflight/preflight-%j.out" \
        --wrap="$REPO_DIR/pipeline_env_exec.sh 'cd $REPO_DIR && source pipeline_env_activate.sh \
&& python scripts/data/extract_base_zero_emb_ids.py --ckpt $PARENT/iter_0000000 --output $preflight/dead_ids.txt$checks'")
    echo "preflight: job $jobid, report $preflight/preflight-$jobid.out"
}

stage_base() {
    local jobid
    [ ! -e "$RUN_DIR/base.json" ] || refuse "$RUN_DIR/base.json exists"
    open_run_dir
    jobid=$(probe "$PARENT_HF" base "")
    echo "base: probe $jobid"
}

stage_smoke() {
    local jobid
    [ -f "$FAST_POSTURE_ENV" ] || refuse "the quickstart's launcher settings $FAST_POSTURE_ENV are missing"
    open_run_dir
    jobid=$(ISAMBARD_ENV_OVERRIDES="$FAST_POSTURE_ENV" submit "smoke" --nodes=16 --time=00:40:00 \
        --job-name=e2e-imid-token-masking-smoke \
        pipeline_training_submit.sbatch "$TEST_DIR/arm_masked.yaml" nano pretrain --disable-ft \
        train.train_iters="$SMOKE_ITERATIONS" checkpoint.save=null logger.wandb_exp_name="$SMOKE_WANDB_NAME")
    echo "smoke: job $jobid, log $TRAINING_LOGS/train-$jobid.out"
}

stage_train() {
    local arm jobid
    [ -f "$FAST_POSTURE_ENV" ] || refuse "the quickstart's launcher settings $FAST_POSTURE_ENV are missing"
    for arm in "${ARMS[@]}"; do
        [ ! -e "${SAVE[$arm]}/latest_checkpointed_iteration.txt" ] \
            || refuse "${SAVE[$arm]} holds a finished run; move it away first"
        # -L as well: the link dangles until its job starts writing the log.
        [ ! -e "$RUN_DIR/$arm.log" ] && [ ! -L "$RUN_DIR/$arm.log" ] \
            || refuse "$RUN_DIR/$arm.log exists; remove it to train this arm again"
    done
    open_run_dir
    for arm in "${ARMS[@]}"; do
        jobid=$(ISAMBARD_ENV_OVERRIDES="$FAST_POSTURE_ENV" submit "train $arm" --nodes=16 --time=01:30:00 \
            --job-name="e2e-imid-token-masking-$arm" \
            pipeline_training_submit.sbatch "$TEST_DIR/arm_$arm.yaml" nano pretrain --disable-ft)
        [ "$DRY_RUN" = "1" ] || ln -s "$TRAINING_LOGS/train-$jobid.out" "$RUN_DIR/$arm.log"
        echo "$arm: train $jobid, log $RUN_DIR/$arm.log"
    done
}

# The standard exporter repairs the arms' torch_grouped checkpoints from a clone and writes <save>/iter_<N>/hf, with
# the original run config as hf/megatron_run_config.yaml (export.yaml).
stage_evaluate() {
    local arm save export_job probe_job
    for arm in "${ARMS[@]}"; do
        save="${SAVE[$arm]}"
        [ -f "$save/latest_checkpointed_iteration.txt" ] || refuse "$save holds no finished save"
        [ ! -e "$save/$FINAL_ITERATION/hf" ] || refuse "$save/$FINAL_ITERATION/hf exists; move it away to export again"
        [ ! -e "$RUN_DIR/$arm.json" ] || refuse "$RUN_DIR/$arm.json exists"
    done
    open_run_dir
    for arm in "${ARMS[@]}"; do
        save="${SAVE[$arm]}"
        export_job=$(submit "export $arm" --nodes=1 --time=01:00:00 --job-name="e2e-imid-export-$arm" \
            pipeline_checkpoint_submit.sbatch export "$save" --hf-model "$EXPORT_ARCHITECTURE" \
            --iteration "$TRAIN_ITERS" --tp "$EXPORT_TP" --ep "$EXPORT_EP" --no-reasoning)
        probe_job=$(probe "$save/$FINAL_ITERATION/hf" "$arm" "$export_job")
        echo "$arm: export $export_job, probe $probe_job"
    done
}

stage_gate() {
    local command="python scripts/telemetry/score_gate.py --spec $TEST_DIR/gate.yaml --scores-dir $RUN_DIR"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] gate: $command"
        return
    fi
    "$REPO_DIR/pipeline_env_exec.sh" "cd $REPO_DIR && source pipeline_env_activate.sh >/dev/null \
&& $command --json > $RUN_DIR/gate.json; $command | tee $RUN_DIR/gate.txt; exit \${PIPESTATUS[0]}"
}

case "$STAGE" in
    data) stage_data ;;
    preflight) stage_preflight ;;
    base) stage_base ;;
    smoke) stage_smoke ;;
    train) stage_train ;;
    evaluate) stage_evaluate ;;
    gate) stage_gate ;;
    *) echo "FATAL: unknown stage '$STAGE'" >&2; exit 1 ;;
esac
