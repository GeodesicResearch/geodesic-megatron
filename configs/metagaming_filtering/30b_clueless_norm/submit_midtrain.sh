#!/bin/bash
# Clueless-Norm midtraining, queued behind pretraining with the checks it must pass first, so that it starts as soon as
# pretraining's final save is complete and checked, and never from a checkpoint that was not (Kyle, 2026-10-14). The
# README's "Clueless-Norm midtraining" section is the guide.
#
# Run it from the root of a frozen copy of the commit the stage configs pin (scripts/training/README.md, "Making a
# frozen copy"), never from a working checkout. scripts/training/staged_submit.sh, which it is built from, enters that
# copy and refuses a directory without a REVISION file and a shell carrying launch settings.
#
#   bash configs/metagaming_filtering/30b_clueless_norm/submit_midtrain.sh <nodes to request per segment>
#
# The request is at least the stage's launch_width nodes: the launcher sweeps NVLink and trains on the first healthy
# ones, so a segment that requests exactly that many fails at its start when one node fails the sweep.
#
# It submits five jobs, each after the one before (afterok), so that a failure cancels the rest (Slurm's
# kill_invalid_depend) rather than starting them:
#   1. the count check: scripts/training/stage_guard.sbatch with pretraining's guard config, --once --require-final,
#      under pretraining's own job name with --dependency=singleton, so that it runs after every pretraining segment.
#      It finds the segments by that name when it runs, and passes only when the watch has checked the final iteration
#      (every [token-masking-counts] line equal to its prediction, no stop condition held) and no other job of the
#      stage is live. Nothing else may carry that name, since the guard would read it as a segment;
#   2. pretraining's completion gate (configs/control_pretraining/stage_gate.sbatch): the final save is the tracked one,
#      and complete;
#   3. the save-record check of the final save: scripts/checkpoint/save_record_check.py with
#      checks/c2_save_record_pretrain.yaml;
#   4. midtraining's first segment, a singleton of the stage's own job name;
#   5. a second segment, afternotok on the first and afterok on the save-record check: it starts only if the first
#      fails after the checks pass, and resumes from the stage's latest save. No spare singleton segment is queued,
#      since one behind a finished run would write the final checkpoint again in place.
# Every submission is recorded in the run directory's jobs.tsv. DRY_RUN=1 prints the submissions, and what would be
# refused, and changes nothing.
set -euo pipefail

REQUEST_NODES="${1:?usage: submit_midtrain.sh <nodes to request per segment>}"
STAGE=midtraining
ARM=configs/metagaming_filtering/30b_clueless_norm
PRETRAIN_CONFIG=$ARM/nemotron_nano_30b_metagaming_clueless_norm_pretrain.yaml
MIDTRAIN_CONFIG=$ARM/nemotron_nano_30b_metagaming_clueless_norm_midtrain.yaml
GUARD_CONFIG=$ARM/guard_pretrain.yaml
SAVE_RECORD_SPEC=$ARM/checks/c2_save_record_pretrain.yaml
RUN_DIR=/projects/a5k/public/logs/metagaming_filtering/clueless_norm/midtrain_launch

# Enters the frozen copy (REPO_DIR, the working directory) and provides refuse, open_run_dir and submit.
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/training/staged_submit.sh"
# Every job is submitted forced: isambard_sbatch skips its account node-cap check and still excludes the bad nodes.
export ISAMBARD_SBATCH_FORCE=1

# What the jobs take from the two stage configs, as shell assignments: the job names, pretraining's final save, and
# midtraining's width and save directory. A config that cannot be read ends the script here.
SETTINGS=$(python3 - "$PRETRAIN_CONFIG" "$MIDTRAIN_CONFIG" <<'PY'
import shlex
import sys

sys.path.insert(0, ".")
from scripts.training.config_compose import load_composed_yaml  # noqa: E402
from scripts.training.launch_width import config_launch_width  # noqa: E402

pretrain, midtrain = (load_composed_yaml(path) for path in sys.argv[1:3])
settings = {
    "PRETRAIN_JOB": pretrain["logger"]["wandb_exp_name"],
    "PRETRAIN_FINAL_SAVE": f"{pretrain['checkpoint']['save']}/iter_{pretrain['train']['train_iters']:07d}",
    "MIDTRAIN_JOB": midtrain["logger"]["wandb_exp_name"],
    "MIDTRAIN_SAVE": midtrain["checkpoint"]["save"],
    "MIDTRAIN_NODES": config_launch_width(sys.argv[2]).nodes,
}
for name, value in settings.items():
    print(f"{name}={shlex.quote(str(value))}")
PY
)
eval "$SETTINGS"

[ "$REQUEST_NODES" -ge "$MIDTRAIN_NODES" ] \
    || refuse "a segment must request at least the stage's launch_width nodes ($MIDTRAIN_NODES), not $REQUEST_NODES"
[ ! -d "$MIDTRAIN_SAVE" ] || [ -z "$(ls -A "$MIDTRAIN_SAVE")" ] \
    || refuse "$MIDTRAIN_SAVE is not empty: midtraining would resume from it instead of warm-starting from pretraining"
ENV_FILE="$REPO_DIR/${MIDTRAIN_CONFIG%.yaml}.env"
[ -f "$ENV_FILE" ] || refuse "the stage's launcher settings $ENV_FILE are missing"

open_run_dir

counts=$(submit "pretraining counts" --job-name="$PRETRAIN_JOB" --dependency=singleton \
    scripts/training/stage_guard.sbatch --config "$GUARD_CONFIG" --once --require-final)
echo "pretraining counts: $counts"

gate=$(submit "pretraining gate" --dependency="afterok:$counts" --job-name="$MIDTRAIN_JOB-check-gate" \
    configs/control_pretraining/stage_gate.sbatch "$PRETRAIN_CONFIG")
echo "pretraining gate: $gate"

record=$(submit "pretraining save record" --dependency="afterok:$gate" --time=00:20:00 \
    --job-name="$MIDTRAIN_JOB-check-save-record" configs/control_pretraining/corpus_job.sbatch \
    scripts/checkpoint/save_record_check.py --spec "$SAVE_RECORD_SPEC" --save "$PRETRAIN_FINAL_SAVE" \
    --report-out "$RUN_DIR/save_record_$(basename "$PRETRAIN_FINAL_SAVE").json")
echo "pretraining save record: $record"

first=$(ISAMBARD_ENV_OVERRIDES="$ENV_FILE" submit "midtraining segment 1" \
    --dependency="afterok:$record,singleton" --nodes="$REQUEST_NODES" --time=24:00:00 --job-name="$MIDTRAIN_JOB" \
    pipeline_training_submit.sbatch "$MIDTRAIN_CONFIG" nano pretrain --disable-ft)
echo "midtraining segment 1: $first"

second=$(ISAMBARD_ENV_OVERRIDES="$ENV_FILE" submit "midtraining segment 2" \
    --dependency="afternotok:$first,afterok:$record,singleton" --nodes="$REQUEST_NODES" --time=24:00:00 \
    --job-name="$MIDTRAIN_JOB" \
    pipeline_training_submit.sbatch "$MIDTRAIN_CONFIG" nano pretrain --disable-ft)
echo "midtraining segment 2: $second"
