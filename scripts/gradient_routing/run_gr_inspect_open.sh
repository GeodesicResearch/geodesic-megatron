#!/usr/bin/env bash
# GEOD-171 gradient-routing — the SECONDARY eval protocol.
#
# Runs the Inspect `sfm_ind_open` judge-scored open-rollout eval against one
# posture, through a plain-text base-completion template, via the
# geodesic-evals bundled runner.
#
# It exists rather than "just call geodesic-evals" because three things must
# be right and none of them are defaults:
#
#   1. PRERENDER_MAX_TOKENS must be exported. On the prerender
#      (/v1/completions) path neither the task's GenerateConfig nor the
#      suite's inspect_args.max_tokens reaches the payload, so without this
#      every rollout runs to 4096 tokens at temperature 1.0 — completion
#      runaway on a Base checkpoint. See gr-forget-alignment.yaml's
#      "GENERATION BUDGET" comment for the code path.
#   2. `name_vars.posture` must be substituted, or every posture's runs land
#      in W&B labelled PLACEHOLDER and the comparison is unrecoverable.
#   3. `chat_template` must point at THIS checkout's base_completion.jinja.
#      The suite carries the canonical absolute path; from a worktree that
#      is the wrong file (or no file).
#
# So: this materialises a resolved per-posture copy of the suite into the
# results dir and runs that, leaving the checked-in suite honest.
#
# Usage:
#   bash scripts/gradient_routing/run_gr_inspect_open.sh <campaign.yaml> <posture>
#   bash scripts/gradient_routing/run_gr_inspect_open.sh <campaign.yaml> <posture> --queue
#   bash scripts/gradient_routing/run_gr_inspect_open.sh <campaign.yaml> <posture> --dry-run
#
#   --queue    submit via `geodesic-evals queue bundled` (uses the suite's
#              slurm: block) instead of running in the current allocation
#   --dry-run  forwarded to geodesic-evals; resolves and reports, runs nothing
#
# Run scripts/gradient_routing/eval_render_check.py FIRST, and again after
# any change to the suite or the template. It is the GPU-free gate that
# proves the model sees plain text.

set -euo pipefail

die() { echo "FATAL [run-gr-inspect]: $*" >&2; exit 1; }
warn() { echo "WARN  [run-gr-inspect]: $*" >&2; }
info() { echo "INFO  [run-gr-inspect]: $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

[[ $# -ge 2 ]] || die "usage: $0 <campaign.yaml> <posture> [--queue] [--dry-run]"

CAMPAIGN="$1"; shift
POSTURE="$1"; shift
[[ -f "${CAMPAIGN}" ]] || die "campaign config not found: ${CAMPAIGN}"
CAMPAIGN="$(cd "$(dirname "${CAMPAIGN}")" && pwd)/$(basename "${CAMPAIGN}")"

MODE="local"
EXTRA_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        --queue) MODE="queue" ;;
        *) EXTRA_ARGS+=("${arg}") ;;
    esac
done

command -v python3 >/dev/null 2>&1 || die "no python3 on PATH"

CONFIG_READER="${SCRIPT_DIR}/campaign_config.py"
[[ -f "${CONFIG_READER}" ]] || die "campaign config reader not found: ${CONFIG_READER}"

CONFIG_SH="$(mktemp)"
trap 'rm -f "${CONFIG_SH}"' EXIT

# campaign_config.py is the single reader both GR eval runners share; it
# returns shell assignments produced with shlex.quote, so a path with a space
# cannot become two arguments.
python3 "${CONFIG_READER}" --section inspect \
    --campaign "${CAMPAIGN}" --repo-dir "${REPO_DIR}" --posture "${POSTURE}" >"${CONFIG_SH}" \
    || die "campaign config is invalid (see above)"

# shellcheck source=/dev/null
source "${CONFIG_SH}"

[[ -x "${GR_CLI}" ]] || die "geodesic-evals CLI not executable: ${GR_CLI}
  (its venv is created by geodesic-evals/scripts/setup.sh; never \`uv sync\` it — it carries a manual vllm-0.18.1 override)"

# The bundled runner spawns `vllm serve` as a bare subprocess command, resolved via PATH.
# Under srun (no login shell, no venv activation) the venv's bin dir must be prepended
# explicitly or the spawn dies with FileNotFoundError: 'vllm'.
export PATH="$(dirname "${GR_CLI}"):${PATH}"

# Tunnel/compute shells often carry an HPC NCCL via LD_PRELOAD (brics module, NCCL 2.26)
# which lacks symbols the venv's torch requires (ncclCommShrink) — the venv torch must
# resolve its OWN bundled NCCL. Drop the preload and put the venv's nccl first.
unset LD_PRELOAD
VENV_NCCL="$(dirname "${GR_CLI}")/../lib/python3.12/site-packages/nvidia/nccl/lib"
if [[ -d "${VENV_NCCL}" ]]; then
    export LD_LIBRARY_PATH="${VENV_NCCL}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
[[ -f "${GR_SUITE}" ]] || die "suite not found: ${GR_SUITE}"
[[ -f "${GR_TEMPLATE}" ]] || die "chat template not found: ${GR_TEMPLATE}"

OUT_DIR="${GR_RESULTS_ROOT}/inspect/${POSTURE}"
RESOLVED="${OUT_DIR}/gr-forget-alignment.${POSTURE}.yaml"
mkdir -p "${OUT_DIR}" || die "cannot create ${OUT_DIR}"

python3 - "${GR_SUITE}" "${RESOLVED}" "${POSTURE}" "${GR_MODEL}" "${GR_TEMPLATE}" "${OUT_DIR}" "${GR_WANDB_GROUP}" "${CAMPAIGN}" "${GR_JUDGE_MODEL}" <<'PY' || die "could not materialise the resolved suite"
import sys
from datetime import datetime, timezone

import yaml

suite_path, out_path, posture, model, template, out_dir, group, campaign, judge_model = sys.argv[1:]
with open(suite_path) as fh:
    cfg = yaml.safe_load(fh)

cfg["model"] = model
cfg["chat_template"] = template
cfg.setdefault("name_vars", {})["posture"] = posture
cfg.setdefault("wandb", {})["group"] = group
cfg["output_dir"] = f"{out_dir}/${{RUN_ID}}"
# The judge that actually scores the rollouts. The checked-in suite carries no
# judge_model, so the campaign file is the only place it is named — write it in
# here or every sample errors at scoring time with no judge configured.
tasks = cfg.get("tasks") or sys.exit("FATAL [run-gr-inspect]: suite declares no `tasks:`")
tasks[0].setdefault("kwargs", {})["judge_model"] = judge_model
# BundledConfig is extra="allow" at the top level, so provenance rides along
# into wandb.config with the resolved YAML.
cfg["gr_provenance"] = {
    "campaign": campaign,
    "source_suite": suite_path,
    "posture": posture,
    "materialised_at": datetime.now(timezone.utc).isoformat(),
}

if "PLACEHOLDER" in yaml.safe_dump(cfg):
    sys.exit("FATAL [run-gr-inspect]: PLACEHOLDER survived substitution — refusing to run")

with open(out_path, "w") as fh:
    fh.write(f"# GENERATED from {suite_path} for posture={posture}. Do not edit; edit the source suite.\n")
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"INFO  [run-gr-inspect]: wrote {out_path}")
PY

export HF_HOME="${GR_HF_HOME}"
# THE generation budget. Not optional — see the header.
export PRERENDER_MAX_TOKENS="${GR_PRERENDER_MAX_TOKENS}"

info "campaign   ${CAMPAIGN}"
info "posture    ${POSTURE}"
info "model      ${GR_MODEL}"
info "template   ${GR_TEMPLATE}"
info "resolved   ${RESOLVED}"
info "judge      ${GR_JUDGE_MODEL}"
info "wandb grp  ${GR_WANDB_GROUP}"
info "PRERENDER_MAX_TOKENS=${PRERENDER_MAX_TOKENS}"

# The judge model needs OPENAI_API_KEY in the *inspect subprocess* environment.
# geodesic-evals' load_env_files() only runs on the render-dump and splice-build
# entrypoints — the bundled-runner path used here never loads the repo .env, so a
# keyless shell means every sample errors at scoring time and the whole run reports
# excluded_rate 1.0. Load the key here with the same precedence load_env_files uses
# (ambient > repo .env > user ~/.env), extracting only this one variable.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    # The CLI sits in the venv, but the .env sits in the (editable-installed) repo —
    # resolve the repo exactly the way load_env_files does: module file, two parents up.
    EVALS_REPO="$("$(dirname "${GR_CLI}")/python" -c \
        'import pathlib, geodesic_evals; print(pathlib.Path(geodesic_evals.__file__).resolve().parents[2])')"
    for env_file in "${EVALS_REPO}/.env" "${HOME}/.env"; do
        [[ -f "${env_file}" ]] || continue
        key_line="$(grep -E '^OPENAI_API_KEY=' "${env_file}" | head -1 || true)"
        [[ -n "${key_line}" ]] || continue
        key_val="${key_line#OPENAI_API_KEY=}"
        key_val="${key_val%\"}"; key_val="${key_val#\"}"
        key_val="${key_val%\'}"; key_val="${key_val#\'}"
        if [[ -n "${key_val}" ]]; then
            export OPENAI_API_KEY="${key_val}"
            info "OPENAI_API_KEY loaded from ${env_file}"
            break
        fi
    done
fi
[[ -n "${OPENAI_API_KEY:-}" ]] || die "OPENAI_API_KEY is not set and no .env supplied it — the judge cannot run.
  Every sample would error at scoring (excluded_rate 1.0). Set the key or add it to the geodesic-evals .env."

info "running: ${GR_CLI} ${MODE} bundled ${RESOLVED} ${EXTRA_ARGS[*]-}"
exec "${GR_CLI}" "${MODE}" bundled "${RESOLVED}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
