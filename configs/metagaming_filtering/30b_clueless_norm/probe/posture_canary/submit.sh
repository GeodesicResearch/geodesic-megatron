#!/bin/bash
# Posture canary of Clueless-Norm's training posture: submit one stage of it. README.md (this directory) is the
# guide; it says when each stage runs and what to check before the next.
#
# This script submits through the standard entry points. What each corpus, arm, prediction and gate is lives in the
# configs beside it (hidden_spans.yaml, the <training stage>_<arm>.yaml overlays, gate.yaml), and every value it
# needs from them is read from those files in one Python call (read_settings below): the arms are the runs whose logs
# gate.yaml judges, each composed through scripts/training/config_compose.py as the launcher composes it. It decides
# only its own scheduling (the time limits and the spare node of an arm's allocation) and where a run's record goes.
#
# Run it from the root of a frozen copy of the commit under test (tests/e2e_tests/README.md, "How one is run"), never
# from a working checkout. scripts/training/staged_submit.sh, which it is built from, enters that copy and refuses a
# directory without a REVISION file and a shell carrying launch settings.
#
#   bash configs/metagaming_filtering/30b_clueless_norm/probe/posture_canary/submit.sh <stage> [training stage] [arm]
#
# Stages, in order:
#   data     write the scratch corpora of hidden_spans.yaml, one 1-node job each
#   predict  predict each masking blend's counts at the arms' width, one 1-node job each, for both training stages
#            (pretrain, midtrain) or the one named; a prediction also builds the index cache its blend's arms read
#   train    launch the arms of both training stages, of the one named, or the one arm named: the nodes of the arm's
#            launch_width plus a spare for its NVLink sweep
#   gate     each training stage's band report, then the pre-registered verdict over the run directory, run here in
#            the container
# Every submission is recorded in the run directory's jobs.tsv. DRY_RUN=1 prints the submissions, and what would be
# refused, and changes nothing.
set -euo pipefail

STAGE="${1:?usage: submit.sh <data|predict|train|gate> [training stage] [arm]}"
ONLY_STAGE="${2:-}"
ONLY_ARM="${3:-}"

CANARY_DIR=configs/metagaming_filtering/30b_clueless_norm/probe/posture_canary
# Enters the frozen copy (REPO_DIR, the working directory) and provides refuse, open_run_dir, submit and run_gate.
source "$(dirname "${BASH_SOURCE[0]}")/../../../../../scripts/training/staged_submit.sh"
# Every job is submitted forced: isambard_sbatch skips its account node-cap check and still excludes the bad nodes.
export ISAMBARD_SBATCH_FORCE=1

# Every value the stages take from the configs, as shell assignments: refuses configs that do not fit together, so a
# stage never submits from an inconsistent set.
read_settings() {
    python3 - "$CANARY_DIR" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")
from scripts.training.config_compose import BASE_CONFIG_KEY, load_composed_yaml  # noqa: E402
from scripts.training.launch_width import config_launch_width  # noqa: E402

canary = Path(sys.argv[1])
gate = yaml.safe_load((canary / "gate.yaml").read_text())
hidden = yaml.safe_load((canary / "hidden_spans.yaml").read_text())


def fail(message):
    sys.exit(f"FATAL: {message}")


def gates(kind):
    return (gate.get(kind) or {}).values()


def stage_config(path):
    """The config an arm's base_config chain ends at: the stage it trains in the posture of."""
    while True:
        base = yaml.safe_load(path.read_text()).get(BASE_CONFIG_KEY)
        if base is None:
            return path.resolve()
        path = path.parent / base


def arm_of(log):
    """The arm a log of the run directory belongs to: <training stage>_<arm>, which names its config here."""
    key = Path(log).stem
    if Path(log).suffix != ".log" or not (canary / f"{key}.yaml").is_file():
        fail(f"gate.yaml judges {log}, and {canary} holds no {key}.yaml")
    return key


# The arms are the runs whose logs the gate spec judges, in the order it first names them.
logs = [g["log"] for kind in ("masking_log", "count_prediction", "count_bounds") for g in gates(kind)]
logs += [log for g in gates("stops") for log in g["logs"]]
logs += [log for g in gates("log_pairing") for log in (g["candidate"], g["reference"])]
arms = {key: load_composed_yaml(canary / f"{key}.yaml") for key in dict.fromkeys(arm_of(log) for log in logs)}
if not arms:
    fail("gate.yaml judges no log")
widths = {key: config_launch_width(str(canary / f"{key}.yaml")) for key in arms}
stage_of = {key: key.split("_", 1)[0] for key in arms}
stages = list(dict.fromkeys(stage_of.values()))
roots = {stage: {stage_config(canary / f"{key}.yaml") for key in arms if stage_of[key] == stage} for stage in stages}
for stage, configs in roots.items():
    if len(configs) != 1:
        fail(f"the {stage} arms overlay more than one stage config: {sorted(map(str, configs))}")
for key, width in widths.items():
    if width is None or width.nvlink_links_per_gpu is None:
        fail(f"{key}.yaml fixes no NVLink-swept launch_width; its job could not hold a spare node")


def what_it_reads(key):
    """What decides the samples an arm reads and the index cache they are read through."""
    config = arms[key]
    return (
        config["dataset"]["data_path"],
        config["dataset"]["path_to_cache"],
        config["train"]["global_batch_size"],
        widths[key],
    )


def prefixes(key):
    return [str(prefix) for prefix in arms[key]["dataset"]["data_path"][1::2]]


# A masking arm's counts are judged against the prediction made from one arm's config: every arm judged against it
# must read what that arm reads and mask what it masks.
predictions = {}
for name, g in (gate.get("count_prediction") or {}).items():
    source = g["prediction"].removesuffix(".predicted.jsonl")
    key = arm_of(g["log"])
    if source == g["prediction"] or source not in arms:
        fail(f"{name}: {g['prediction']} is not <an arm>.predicted.jsonl")
    if what_it_reads(key) != what_it_reads(source) or arms[key]["token_masking"] != arms[source]["token_masking"]:
        fail(f"{name}: {key} does not read and mask what {source}, whose prediction it is judged against, does")
    first, last = predictions.get(source, (g["first"], g["last"]))
    predictions[source] = (min(first, g["first"]), max(last, g["last"]))
# Every other arm reads its blend through the index cache one of the predictions builds.
builds_cache = {}
for key in arms:
    sources = [source for source in predictions if what_it_reads(source) == what_it_reads(key)]
    if len(sources) != 1:
        fail(f"{key} reads an index cache that {len(sources)} predictions build, not one: {sources}")
    builds_cache[key] = sources[0]

# An arm that loads another's save and writes none is its second segment.
resumes = {}
for key, config in arms.items():
    load, save = config["checkpoint"]["load"], config["checkpoint"]["save"]
    if load is not None and save is None:
        savers = [other for other, c in arms.items() if other != key and c["checkpoint"]["save"] == load]
        if len(savers) != 1 or stage_of[savers[0]] != stage_of[key]:
            fail(f"{key} loads {load}, which {len(savers)} arms of its training stage save, not one")
        resumes[key] = savers[0]

# Each training stage's band: the unmasked arms, the one that normalises per token against the rest.
bands = {}
for name, g in (gate.get("loss_band") or {}).items():
    stage = g["report"].removesuffix("_band.json")
    if stage not in stages or stage in bands:
        fail(f"{name}: {g['report']} is not <a training stage>_band.json, once per stage")
    unmasked = [key for key in arms if stage_of[key] == stage and not arms[key]["token_masking"]["enabled"]]
    per_token = [key for key in unmasked if arms[key]["model"]["calculate_per_token_loss"]]
    references = [key for key in unmasked if key not in per_token]
    if len(per_token) != 1 or len(references) != g["references"]:
        fail(
            f"{name}: the {stage} arms that mask nothing are {len(per_token)} per-token and {len(references)} "
            f"per-microbatch, not 1 and the gate's {g['references']}"
        )
    if any(what_it_reads(key) != what_it_reads(per_token[0]) for key in references):
        fail(f"{name}: the per-microbatch arms do not read what {per_token[0]} reads")
    bands[stage] = (g["report"], g["first"], g["last"], g["window"], per_token[0], references)

corpora = {name: Path(hidden["output_root"]) / name / hidden["prefix_name"] for name in hidden["corpora"]}
scratch = {prefix for key in arms for prefix in prefixes(key) if Path(hidden["output_root"]) in Path(prefix).parents}
if scratch != {str(prefix) for prefix in corpora.values()}:
    fail(f"the arms' scratch corpora are not exactly the ones hidden_spans.yaml writes: {sorted(scratch)}")


def table(name, values):
    pairs = " ".join(f"[{key}]={shlex.quote(str(value))}" for key, value in values.items())
    print(f"declare -A {name}=({pairs})")


def words(name, values):
    print(f"{name}=(" + " ".join(shlex.quote(str(value)) for value in values) + ")")


words("TRAINING_STAGES", stages)
words("ARMS", arms)
words("CORPORA", corpora)
words("PREDICTIONS", predictions)
table("CORPUS_PREFIX", corpora)
table("ARM_STAGE", stage_of)
table("ENV_FILE", {key: next(iter(roots[stage_of[key]])).with_suffix(".env") for key in arms})
table("NODES", {key: width.nodes for key, width in widths.items()})
table("GPUS", {key: width.world_size for key, width in widths.items()})
table("JOB_NAME", {key: config["logger"]["wandb_exp_name"] for key, config in arms.items()})
table("SAVE", {key: config["checkpoint"]["save"] or "" for key, config in arms.items()})
table("LOAD", {key: config["checkpoint"]["load"] or "" for key, config in arms.items()})
table("INDEX_CACHE", {key: config["dataset"]["path_to_cache"] for key, config in arms.items()})
table("PREFIXES", {key: " ".join(prefixes(key)) for key in arms})
table("RESUMES", {key: resumes.get(key, "") for key in arms})
table("BUILDS_CACHE", builds_cache)
table("PREDICT_FIRST", {source: first for source, (first, _) in predictions.items()})
table("PREDICT_LAST", {source: last for source, (_, last) in predictions.items()})
table("BAND_REPORT", {stage: band[0] for stage, band in bands.items()})
table("BAND_FIRST", {stage: band[1] for stage, band in bands.items()})
table("BAND_LAST", {stage: band[2] for stage, band in bands.items()})
table("BAND_WINDOW", {stage: band[3] for stage, band in bands.items()})
table("BAND_CANDIDATE", {stage: band[4] for stage, band in bands.items()})
table("BAND_REFERENCES", {stage: " ".join(band[5]) for stage, band in bands.items()})
PY
}
SETTINGS=$(read_settings)
eval "$SETTINGS"

if [ -n "$ONLY_STAGE" ] && [[ " ${TRAINING_STAGES[*]} " != *" $ONLY_STAGE "* ]]; then
    echo "FATAL: unknown training stage '$ONLY_STAGE'; the training stages are ${TRAINING_STAGES[*]}" >&2
    exit 1
fi
if [ -n "$ONLY_ARM" ] && [ -z "${ARM_STAGE[${ONLY_STAGE}_$ONLY_ARM]:-}" ]; then
    echo "FATAL: unknown arm '$ONLY_ARM' of $ONLY_STAGE; the arms are ${ARMS[*]}" >&2
    exit 1
fi

CODE=$(head -c 12 REVISION)
RUN_DIR=/projects/a5k/public/logs/metagaming_filtering/posture_canary/$CODE
TRAINING_LOGS=/projects/a5k/public/logs/megatron_runs
# An arm's allocation holds its launch_width's nodes and this many more: the launcher NVLink-sweeps the allocation
# and trains on the first healthy nodes, so one node with dead links does not end the job.
SPARE_NODES=1
# Time limits, at least 1.5 times the wall time of a job. Pretraining: a 50-iteration 16-node run of the stage's
# posture at the arms' batch took 5.5 to 6.7 minutes from launch to exit (jobs 7288713 and 7289565). Midtraining adds
# the warm start's load and a slower first iteration and step, and has no 16-node measurement in this posture yet, so
# its limits allow over three times the 12 to 15 minutes it should take, lest an untimed arm end on walltime rather
# than a verdict. An arm that writes or resumes a full save (STATE_TIME) adds the save or the load. The one-node
# jobs: a corpus is written and checked at over 2M ids a second (the largest holds 660M), and a 477-iteration
# prediction at this width took under 10 minutes.
declare -A TRAIN_TIME=([pretrain]=00:15:00 [midtrain]=00:50:00)
declare -A STATE_TIME=([pretrain]=00:25:00 [midtrain]=01:10:00)
CORPUS_TIME=00:15:00
PREDICTION_TIME=00:30:00

selected() {  # an arm or prediction of the training stage, and the arm, this invocation names (all when none is named)
    [ -z "$ONLY_STAGE" ] || [ "${ARM_STAGE[$1]}" = "$ONLY_STAGE" ] || return 1
    [ -z "$ONLY_ARM" ] || [ "$1" = "${ONLY_STAGE}_$ONLY_ARM" ]
}

require_corpora() {  # an arm: every corpus of its blend is built
    local prefix
    for prefix in ${PREFIXES[$1]}; do
        [ -f "$prefix.bin" ] && [ -f "$prefix.idx" ] || refuse "$1 reads $prefix, which is not built"
    done
}

stage_data() {
    local corpus key jobid
    for corpus in "${CORPORA[@]}"; do
        [ ! -e "$(dirname "${CORPUS_PREFIX[$corpus]}")" ] \
            || refuse "$(dirname "${CORPUS_PREFIX[$corpus]}") exists; a rebuild starts from a moved-away directory"
    done
    # Megatron's index cache is keyed by the prefixes, sample count and seed, never by the corpus's content: a rebuilt
    # corpus beside an old cache would be read through indices built over the old .bin files.
    for key in "${PREDICTIONS[@]}"; do
        [ ! -e "${INDEX_CACHE[$key]}" ] || refuse "${INDEX_CACHE[$key]} exists; move it away with the corpora it indexes"
    done
    open_run_dir
    for corpus in "${CORPORA[@]}"; do
        jobid=$(submit "data $corpus" --time="$CORPUS_TIME" --job-name="mf-canary-data-$corpus" \
            --output="$RUN_DIR/data-$corpus-%j.out" \
            configs/control_pretraining/corpus_job.sbatch configs/control_pretraining/hidden_span_corpus.py \
            --config "$CANARY_DIR/hidden_spans.yaml" --corpus "$corpus")
        echo "$corpus: job $jobid, log $RUN_DIR/data-$corpus-$jobid.out"
    done
}

stage_predict() {
    local key jobid
    for key in "${PREDICTIONS[@]}"; do
        selected "$key" || continue
        require_corpora "$key"
        [ ! -e "$RUN_DIR/$key.predicted.jsonl" ] \
            || refuse "$RUN_DIR/$key.predicted.jsonl exists; a prediction is never overwritten"
        [ ! -e "${INDEX_CACHE[$key]}" ] || refuse "${INDEX_CACHE[$key]} exists; the prediction builds it afresh"
    done
    open_run_dir
    for key in "${PREDICTIONS[@]}"; do
        selected "$key" || continue
        jobid=$(submit "predict $key" --time="$PREDICTION_TIME" --job-name="mf-canary-predict-$key" \
            --output="$RUN_DIR/predict-$key-%j.out" \
            configs/control_pretraining/corpus_job.sbatch scripts/data/predict_masked_counts.py \
            "$CANARY_DIR/$key.yaml" --model nano --mode pretrain --gpus "${GPUS[$key]}" \
            --iterations "${PREDICT_FIRST[$key]}" "${PREDICT_LAST[$key]}" --out "$RUN_DIR/$key.predicted.jsonl")
        echo "$key: predict $jobid, counts $RUN_DIR/$key.predicted.jsonl"
    done
}

# An arm trains only once the prediction that builds its blend's index cache is whole: its inputs record and one
# record per predicted iteration. Sixteen nodes would otherwise wait on rank 0 building the cache, and a masking arm
# would have no counts to be judged against.
require_prediction() {
    local source="${BUILDS_CACHE[$1]}" file lines
    file="$RUN_DIR/$source.predicted.jsonl"
    lines=$((PREDICT_LAST[$source] - PREDICT_FIRST[$source] + 2))
    [ -f "$file" ] && [ "$(wc -l < "$file")" -eq "$lines" ] \
        || refuse "$1 waits for the prediction $file ($lines lines), which is absent or incomplete"
}

launch() {  # an arm, and the job it waits for ("" for none)
    local key="$1" stage="${ARM_STAGE[$1]}" dependency=() time jobid
    [ -z "$2" ] || dependency=(--dependency="afterok:$2")
    time="${TRAIN_TIME[$stage]}"
    [ -z "${SAVE[$key]}${RESUMES[$key]}" ] || time="${STATE_TIME[$stage]}"
    jobid=$(ISAMBARD_ENV_OVERRIDES="${ENV_FILE[$key]}" submit "train $key" ${dependency[@]+"${dependency[@]}"} \
        --nodes="$((NODES[$key] + SPARE_NODES))" --time="$time" --job-name="${JOB_NAME[$key]}" \
        pipeline_training_submit.sbatch "$CANARY_DIR/$key.yaml" nano pretrain --disable-ft)
    [ "$DRY_RUN" = "1" ] || ln -s "$TRAINING_LOGS/train-$jobid.out" "$RUN_DIR/$key.log"
    JOB_OF[$key]="$jobid"
    echo "$key: train $jobid, log $RUN_DIR/$key.log"
}

stage_train() {
    local key first
    declare -A JOB_OF=()
    for key in "${ARMS[@]}"; do
        selected "$key" || continue
        [ -n "${TRAIN_TIME[${ARM_STAGE[$key]}]:-}" ] || refuse "no time limit is set for the ${ARM_STAGE[$key]} arms"
        [ -f "${ENV_FILE[$key]}" ] || refuse "the stage's launcher settings ${ENV_FILE[$key]} are missing"
        require_corpora "$key"
        require_prediction "$key"
        # -L as well: the link dangles until its job starts writing the log.
        [ ! -e "$RUN_DIR/$key.log" ] && [ ! -L "$RUN_DIR/$key.log" ] \
            || refuse "$RUN_DIR/$key.log exists; remove it to train this arm again"
        [ -z "${SAVE[$key]}" ] || [ ! -e "${SAVE[$key]}/latest_checkpointed_iteration.txt" ] \
            || refuse "${SAVE[$key]} holds a save; move it away first"
        first="${RESUMES[$key]}"
        [ -z "$first" ] || selected "$first" || [ -f "${LOAD[$key]}/latest_checkpointed_iteration.txt" ] \
            || refuse "$key resumes $first's save, and ${LOAD[$key]} holds none"
    done
    open_run_dir
    # First the arms that resume nothing, then each second segment after the job whose save it resumes.
    for key in "${ARMS[@]}"; do
        selected "$key" && [ -z "${RESUMES[$key]}" ] || continue
        launch "$key" ""
    done
    for key in "${ARMS[@]}"; do
        selected "$key" && [ -n "${RESUMES[$key]}" ] || continue
        launch "$key" "${JOB_OF[${RESUMES[$key]}]:-}"
    done
}

# The verdict, after each training stage's band report. loss_parity.py exits 1 on a candidate outside its band and on
# logs it cannot read; either way what it wrote is what the verdict's loss_band gate reads.
stage_gate() {
    local stage reference references bands=()
    for stage in "${TRAINING_STAGES[@]}"; do
        [ -n "${BAND_REPORT[$stage]:-}" ] || continue
        references=""
        for reference in ${BAND_REFERENCES[$stage]}; do
            references+=" $RUN_DIR/$reference.log"
        done
        bands+=("python scripts/telemetry/loss_parity.py band --reference$references \
--candidate $RUN_DIR/${BAND_CANDIDATE[$stage]}.log --iterations ${BAND_FIRST[$stage]} ${BAND_LAST[$stage]} \
--window ${BAND_WINDOW[$stage]} --json > $RUN_DIR/${BAND_REPORT[$stage]}")
    done
    run_gate "$CANARY_DIR/gate.yaml" ${bands[@]+"${bands[@]}"}
}

case "$STAGE" in
    data) stage_data ;;
    predict) stage_predict ;;
    train) stage_train ;;
    gate) stage_gate ;;
    *) echo "FATAL: unknown stage '$STAGE'" >&2; exit 1 ;;
esac
