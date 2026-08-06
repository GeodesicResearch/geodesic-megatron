#!/usr/bin/env bash
# Container-backed python interpreter for the GR lm-eval protocol.
#
# Why this exists: the stock nvidia Nemotron-H modeling file hard-raises unless mamba-ssm
# imports, and no host interpreter on this aarch64 cluster carries the Mamba kernels. The
# training container has the kernels AND a transformers with native Nemotron-H support;
# lm_eval itself is layered on via a torch-free pip --target overlay (pure-python deps
# only — torch/transformers/datasets come from the image).
#
# Used as GR_LM_EVAL_PYTHON by scripts/gradient_routing/run_gr_base_mcq.sh; behaves like
# a python binary (args are re-quoted verbatim into the container shell).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${GR_LMEVAL_TARGET:-/projects/a5k/public/venvs_${USER}/gr_lmeval_target}"
[[ -d "${TARGET}" ]] || {
    echo "FATAL [lmeval-container-python]: lm_eval overlay not found at ${TARGET}." >&2
    echo "  Create it once (in-container): pip install --target=${TARGET} --no-deps lm_eval==0.4.9.1 \\" >&2
    echo "    sqlitedict pytablewriter tqdm-multiprocess zstandard jsonlines numexpr sacrebleu \\" >&2
    echo "    word2number more_itertools dataproperty tabledata typepy mbstrdecoder pathvalidate portalocker colorama" >&2
    exit 1
}

QUOTED=""
for arg in "$@"; do
    QUOTED+=" $(printf '%q' "${arg}")"
done

# `-m lm_eval` goes through the bootstrap (transformers 5.x compat alias applied before
# lm_eval imports); every other invocation shape (YAML parsing, preflight snippets) is
# plain python.
LAUNCH="python${QUOTED}"
if [[ "${1:-}" == "-m" && "${2:-}" == lm_eval* ]]; then
    LAUNCH="python ${REPO_DIR}/scripts/gradient_routing/gr_lmeval_bootstrap.py${QUOTED}"
fi

# Optional datasets-cache redirect (set AFTER activate, which pins the shared cache):
# the shared /projects HF datasets cache carries lock files owned by other users, and
# filelock cannot open a foreign-owned lock (EACCES). GR_HF_DATASETS_CACHE points these
# runs at a campaign-owned cache instead.
CACHE_PREFIX=""
if [[ -n "${GR_HF_DATASETS_CACHE:-}" ]]; then
    mkdir -p "${GR_HF_DATASETS_CACHE}"
    CACHE_PREFIX="HF_DATASETS_CACHE=${GR_HF_DATASETS_CACHE} "
fi

# Triton's default cache is ~/.triton on shared Lustre; concurrent runs on
# different nodes compiling the same mamba kernels race on it and die with
# OSError 116 (stale file handle). Always point it at a fresh node-local dir.
exec "${REPO_DIR}/pipeline_env_exec.sh" "cd ${REPO_DIR}; source pipeline_env_activate.sh || exit 1; \
export TRITON_CACHE_DIR=\$(mktemp -d /tmp/gr_lmeval_triton.XXXXXX); \
PYTHONPATH=${TARGET}:\${PYTHONPATH:-} ${CACHE_PREFIX}${LAUNCH}"
