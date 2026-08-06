#!/usr/bin/env bash
# GEOD-171 gradient-routing — the PRIMARY eval protocol.
#
# Runs the frozen sfm-evals base-MCQ family (loglikelihood, no chat
# template, no generation) plus the capability tasks against one posture's
# HF checkpoint, on ONE GH200, logging to W&B under the campaign group.
#
# Everything comes from the campaign YAML — the checkpoint paths, the task
# groups, the interpreter, the results root, the W&B identity. Nothing is
# hardcoded here that an operator would want to change.
#
# Usage:
#   bash scripts/gradient_routing/run_gr_base_mcq.sh <campaign.yaml>
#   bash scripts/gradient_routing/run_gr_base_mcq.sh <campaign.yaml> <posture>
#   bash scripts/gradient_routing/run_gr_base_mcq.sh <campaign.yaml> <posture> <group> [group ...]
#   bash scripts/gradient_routing/run_gr_base_mcq.sh <campaign.yaml> --preflight-only
#
#   posture  baseline | forget_on | forget_off | all   (default: all enabled)
#   group    a key under lm_eval.task_groups           (default: default_task_groups)
#
# Environment overrides:
#   GR_LM_EVAL_PYTHON   interpreter that owns lm_eval (wins over the YAML)
#   GR_DRY_RUN=1        print the lm_eval command instead of running it
#
# One (posture, group) pair is one lm_eval invocation and one W&B run.
# Submit one SLURM job per pair for anything longer than a few minutes:
#   isambard_sbatch --gpus-per-node=1 --time=04:00:00 --wrap \
#     "bash scripts/gradient_routing/run_gr_base_mcq.sh <campaign.yaml> baseline"

set -euo pipefail

die() { echo "FATAL [run-gr-base-mcq]: $*" >&2; exit 1; }
warn() { echo "WARN  [run-gr-base-mcq]: $*" >&2; }
info() { echo "INFO  [run-gr-base-mcq]: $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

[[ $# -ge 1 ]] || die "usage: $0 <campaign.yaml> [posture|all] [task_group ...] | <campaign.yaml> --preflight-only"

CAMPAIGN="$1"; shift
[[ -f "${CAMPAIGN}" ]] || die "campaign config not found: ${CAMPAIGN}"
CAMPAIGN="$(cd "$(dirname "${CAMPAIGN}")" && pwd)/$(basename "${CAMPAIGN}")"

PREFLIGHT_ONLY=0
POSTURE_ARG="all"
TASK_GROUP_ARGS=()
if [[ $# -gt 0 && "$1" == "--preflight-only" ]]; then
    PREFLIGHT_ONLY=1
    shift
fi
if [[ $# -gt 0 ]]; then
    POSTURE_ARG="$1"; shift
    TASK_GROUP_ARGS=("$@")
fi

# ---------------------------------------------------------------------------
# Read the campaign config.
#
# campaign_config.py is the single reader both GR eval runners share; it
# returns shell assignments produced with shlex.quote, so a path with a space
# cannot become two arguments.
#
# The full parse happens under the lm_eval interpreter (it has PyYAML), which
# also proves that interpreter runs at all before anything else is attempted.
# ---------------------------------------------------------------------------

CONFIG_READER="${SCRIPT_DIR}/campaign_config.py"
[[ -f "${CONFIG_READER}" ]] || die "campaign config reader not found: ${CONFIG_READER}"

CONFIG_SH="$(mktemp)"
trap 'rm -f "${CONFIG_SH}"' EXIT

BOOTSTRAP_PY="${GR_LM_EVAL_PYTHON:-}"
if [[ -z "${BOOTSTRAP_PY}" ]]; then
    # Pull only `lm_eval.python` out of the YAML, using whatever python is on
    # PATH, so the real interpreter is itself config-driven.
    command -v python3 >/dev/null 2>&1 || die "no python3 on PATH to bootstrap config parsing"
    python3 "${CONFIG_READER}" --section lm_eval_python \
        --campaign "${CAMPAIGN}" --repo-dir "${REPO_DIR}" >"${CONFIG_SH}" \
        || die "could not read lm_eval.python from ${CAMPAIGN}"
    # shellcheck source=/dev/null
    source "${CONFIG_SH}"
    BOOTSTRAP_PY="${GR_LM_EVAL_PYTHON}"
fi
[[ -x "${BOOTSTRAP_PY}" ]] || die "lm_eval interpreter is not executable: ${BOOTSTRAP_PY}
  (set GR_LM_EVAL_PYTHON, or fix lm_eval.python in ${CAMPAIGN})"

READER_ARGS=(--section lm_eval --campaign "${CAMPAIGN}" --repo-dir "${REPO_DIR}" --posture "${POSTURE_ARG}")
if [[ ${#TASK_GROUP_ARGS[@]} -gt 0 ]]; then
    READER_ARGS+=(--task-group "${TASK_GROUP_ARGS[@]}")
fi
"${BOOTSTRAP_PY}" "${CONFIG_READER}" "${READER_ARGS[@]}" >"${CONFIG_SH}" \
    || die "campaign config is invalid (see above)"

# shellcheck source=/dev/null
source "${CONFIG_SH}"

export HF_HOME="${GR_HF_HOME}"
export CUDA_VISIBLE_DEVICES="${GR_CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false

info "campaign        ${CAMPAIGN}"
info "repo            ${REPO_DIR}"
info "interpreter     ${BOOTSTRAP_PY}"
info "include_path    ${GR_INCLUDE_PATH}"
info "results_root    ${GR_RESULTS_ROOT}"
info "wandb           ${GR_WANDB_ENTITY}/${GR_WANDB_PROJECT} group=${GR_WANDB_GROUP}"
info "postures        ${GR_POSTURES}"
info "task groups     ${GR_TASK_GROUPS}"
info "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ---------------------------------------------------------------------------
# Preflight. Every one of these has cost somebody a wasted allocation.
# ---------------------------------------------------------------------------

[[ -d "${GR_INCLUDE_PATH}" ]] || die "lm_eval include_path does not exist: ${GR_INCLUDE_PATH}"
mkdir -p "${GR_RESULTS_ROOT}" || die "cannot create results root: ${GR_RESULTS_ROOT}"

ALL_TASKS=""
for group in ${GR_TASK_GROUPS}; do
    var="GR_TASKS__${group}"
    ALL_TASKS="${ALL_TASKS},${!var}"
done
ALL_TASKS="${ALL_TASKS#,}"

MODEL_PATHS=""
for posture in ${GR_POSTURES}; do
    var="GR_MODEL__${posture}"
    MODEL_PATHS="${MODEL_PATHS} ${!var}"
done

"${BOOTSTRAP_PY}" - "${GR_INCLUDE_PATH}" "${ALL_TASKS}" "${GR_MODEL_ARGS_EXTRA}" ${MODEL_PATHS} <<'PY' || die "preflight failed"
import os
import sys
from pathlib import Path

include_path, task_csv, model_args_extra, *models = sys.argv[1:]
problems = []

try:
    import lm_eval

    print(f"INFO  [preflight]: lm_eval {lm_eval.__version__}")
except ImportError as exc:
    sys.exit(f"FATAL [preflight]: lm_eval is not importable ({exc})")

try:
    import wandb

    print(f"INFO  [preflight]: wandb {wandb.__version__}")
except ImportError:
    problems.append("wandb is not installed in this interpreter, so --wandb_args will fail")

from lm_eval.tasks import TaskManager

registered = set(TaskManager(include_path=include_path).all_tasks)
wanted = [t for t in task_csv.split(",") if t]
missing = [t for t in wanted if t not in registered]
if missing:
    problems.append(f"tasks not registered under --include_path {include_path}: {missing}")
else:
    print(f"INFO  [preflight]: all {len(wanted)} tasks registered: {', '.join(wanted)}")

# `cais/wmdp` is not in the shared HF cache as of 2026-08-05, so a wmdp task
# on an offline node fails minutes in rather than at startup.
if any(t.endswith("wmdp_bio") for t in wanted):
    hub = Path(os.environ.get("HF_HOME", "")) / "hub" / "datasets--cais--wmdp"
    if not hub.exists():
        problems.append(
            f"a wmdp task is requested but {hub} is absent. Pre-fetch it on a login node: "
            f"HF_HOME={os.environ.get('HF_HOME')} hf download cais/wmdp --repo-type dataset"
        )

trust_remote = "trust_remote_code=True" in model_args_extra
for model in models:
    if not model.startswith("/"):
        print(f"INFO  [preflight]: {model} treated as a hub id (resolved from HF_HOME / the Hub)")
        continue
    d = Path(model)
    if not d.is_dir():
        problems.append(f"checkpoint dir does not exist: {d}")
        continue
    if not (d / "config.json").exists():
        problems.append(f"{d} has no config.json — is it an HF export?")
    if not any(d.glob("*.safetensors")):
        problems.append(f"{d} contains no *.safetensors")
    # transformers 4.57.x has no native `nemotron_h`, so the checkpoint's own
    # modeling code is what runs. Megatron->HF conversion has historically
    # dropped these files, and the resulting failure is an opaque
    # "unrecognized model type" hours into a queue.
    if trust_remote:
        for needed in ("configuration_nemotron_h.py", "modeling_nemotron_h.py"):
            if not (d / needed).exists():
                problems.append(
                    f"{d} is missing {needed}, which trust_remote_code=True requires. Copy it from "
                    f"the upstream Base snapshot before evaluating."
                )

if problems:
    for p in problems:
        print(f"FATAL [preflight]: {p}", file=sys.stderr)
    sys.exit(1)
print("INFO  [preflight]: OK")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    info "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not found — this looks like a login node. lm_eval will fail without a GPU."
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    info "preflight-only: stopping here"
    exit 0
fi

# ---------------------------------------------------------------------------
# Run: one lm_eval invocation per (posture, task_group).
# ---------------------------------------------------------------------------

for posture in ${GR_POSTURES}; do
    model_var="GR_MODEL__${posture}"
    model="${!model_var}"

    for group in ${GR_TASK_GROUPS}; do
        tasks_var="GR_TASKS__${group}"
        tasks="${!tasks_var}"

        out_dir="${GR_RESULTS_ROOT}/lm_eval/${posture}/${group}"
        log_file="${GR_RESULTS_ROOT}/lm_eval/${posture}/${group}.log"
        mkdir -p "${out_dir}"

        run_name="${GR_RUN_NAME_TEMPLATE//\{posture\}/${posture}}"
        run_name="${run_name//\{task_group\}/${group}}"

        cmd=(
            "${BOOTSTRAP_PY}" -m lm_eval
            --model hf
            --model_args "pretrained=${model},${GR_MODEL_ARGS_EXTRA}"
            --tasks "${tasks}"
            --include_path "${GR_INCLUDE_PATH}"
            --batch_size "${GR_BATCH_SIZE}"
            --output_path "${out_dir}"
            --wandb_args "project=${GR_WANDB_PROJECT},entity=${GR_WANDB_ENTITY},group=${GR_WANDB_GROUP},name=${run_name}"
            --wandb_config_args "gr_posture=${posture},gr_task_group=${group},gr_campaign=${CAMPAIGN}"
        )
        [[ -n "${GR_LM_EVAL_LIMIT}" ]] && cmd+=(--limit "${GR_LM_EVAL_LIMIT}")
        [[ -n "${GR_LOG_SAMPLES}" ]] && cmd+=(--log_samples)
        [[ -n "${GR_WRITE_OUT}" ]] && cmd+=(--write_out)
        [[ -n "${GR_TRUST_REMOTE_CODE_FLAG}" ]] && cmd+=(--trust_remote_code)

        echo
        info "=== posture=${posture} group=${group} ==="
        info "model:  ${model}"
        info "tasks:  ${tasks}"
        info "wandb:  ${run_name}"
        info "out:    ${out_dir}"
        info "log:    ${log_file}"
        printf 'INFO  [run-gr-base-mcq]: cmd: '
        printf '%q ' "${cmd[@]}"
        echo

        if [[ "${GR_DRY_RUN:-0}" == "1" ]]; then
            info "GR_DRY_RUN=1 — not executing"
            continue
        fi

        "${cmd[@]}" 2>&1 | tee "${log_file}"
        status="${PIPESTATUS[0]}"
        [[ "${status}" == "0" ]] || die "lm_eval failed (exit ${status}) for posture=${posture} group=${group}; see ${log_file}"
        info "done: posture=${posture} group=${group}"
    done
done

info "all requested (posture, task_group) pairs completed"
