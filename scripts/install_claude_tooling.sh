#!/bin/bash
# ==============================================================================
# install_claude_tooling.sh — create/refresh the repo-local TOOLING venv and wire
# up geodesic-claude-tooling (the Claude Code hooks) + pre-commit.
#
# WHAT `.venv` IS — AND WHAT IT IS NOT
#
#   `.venv` in this repo is a CLAUDE-CODE-TOOLING AND LINTING venv, nothing more:
#     * the geodesic-* hook entry points — .claude/settings.json invokes
#       $CLAUDE_PROJECT_DIR/.venv/bin/geodesic-* by absolute path (6 hooks), so a
#       repo-local venv at exactly this path is load-bearing FOR TOOLING;
#     * pre-commit itself — .git/hooks/pre-commit execs
#       .venv/bin/python3 -m pre_commit;
#     * ruff / mypy for linting by hand.
#
#   It carries NO torch, NO CUDA, NO megatron, and it is NOT an execution
#   environment. Nothing you train, convert, pack or evaluate with ever touches
#   it: every pipeline — and the unit tests, whose conftest imports torch and
#   megatron.core at collection time — runs inside the Apptainer container built
#   by `pipeline_env_setup.sh`. See docs/environment.md.
#
#   That separation is the whole point of this installer being separate from
#   pipeline_env_setup.sh: this one needs only `uv`, finishes in seconds, wants no
#   GPU / allocation / compiler, and structurally cannot perturb the execution
#   environment.
#
# Usage:
#   bash scripts/install_claude_tooling.sh              # create/refresh (idempotent)
#   bash scripts/install_claude_tooling.sh --recreate   # wipe .venv, rebuild it
#
# `--recreate` is safe by design (the venv holds only tooling, rebuilt in
# seconds) and is also how you reclaim the space from a leftover pre-container
# venv that still holds the retired torch/CUDA stack.
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

VENV_DIR="$REPO_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

RECREATE=0
case "${1:-}" in
    "") ;;
    --recreate) RECREATE=1 ;;
    -h|--help)
        # Point at the header rather than duplicating it — the header is where the
        # "this venv is tooling-only, no torch" contract is written down.
        awk 'NR == 1 { next } /^#/ { print; next } { exit }' "${BASH_SOURCE[0]}"
        exit 0
        ;;
    *)
        echo "Usage: bash scripts/install_claude_tooling.sh [--recreate|--help]" >&2
        exit 2
        ;;
esac

echo "=== geodesic-claude-tooling installer (tooling venv only — no torch) ==="

# uv is the ONLY prerequisite. It is a user-level static binary, so this check can
# never be satisfied by "activate the env first" — there is no host env to
# activate any more.
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found on PATH — it is the only prerequisite for this script." >&2
    echo "  Install once (user-level, no root, no modules):" >&2
    echo "      curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "  then re-run this script (uv lands in ~/.local/bin)." >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# 1. The venv itself. This script CREATES it — nothing else in the repo does any
#    more (the bare-metal env installer that used to build a .venv here is gone;
#    pipeline_env_setup.sh builds the container instead).
# ------------------------------------------------------------------------------
echo "[1/6] Tooling venv: $VENV_DIR"
if [ "$RECREATE" = "1" ] && [ -d "$VENV_DIR" ]; then
    # Destructive step is opt-in only: a plain refresh must never delete a venv
    # out from under a running Claude Code session (its hooks are executed from
    # .venv/bin on every tool call).
    echo "      --recreate: removing the existing venv (a leftover pre-container"
    echo "      venv is tens of GB of small files, so this can take a minute)"
    rm -rf "$VENV_DIR"
fi
if [ -x "$VENV_PYTHON" ]; then
    echo "      already present — reusing it (every install step below is idempotent)"
else
    # --python 3.12 pins the interpreter the repo targets (pyproject
    # requires-python >=3.12, and the ruff/mypy configs assume it). uv fetches a
    # managed 3.12 if the host has none, so this works on any node.
    uv venv --python 3.12 "$VENV_DIR"
fi

# A venv created before the container migration still contains the retired
# torch/CUDA/TE stack. Harmless (the hooks only need the geodesic-* scripts) but
# it is dead weight and it makes `.venv` look like an execution environment,
# which is exactly the confusion this script's header exists to prevent.
if compgen -G "$VENV_DIR/lib/python3.*/site-packages/torch" >/dev/null; then
    echo "      NOTE: this venv still carries torch from the retired bare-metal build."
    echo "            Nothing uses it — model code runs in the container. Reclaim the"
    echo "            space with: bash scripts/install_claude_tooling.sh --recreate"
fi

# ------------------------------------------------------------------------------
# 2. The tooling submodule (no-op when already checked out).
# ------------------------------------------------------------------------------
echo "[2/6] Initialising the .claude/geodesic-claude-tooling submodule..."
git submodule update --init .claude/geodesic-claude-tooling

# ------------------------------------------------------------------------------
# 3. The hook package.
#
# Deliberately `uv pip install`, NEVER `uv sync`: a sync resolves the FULL project
# (torch 2.11+cu126, vllm, source-built TE / mamba / grouped-gemm) and would try
# to build the training stack on the host — the very thing containerisation
# removed. Nothing in this venv needs the project's runtime dependencies, and the
# project is intentionally NOT installed here.
# ------------------------------------------------------------------------------
echo "[3/6] Installing geodesic-claude-tooling (editable) into the tooling venv..."
uv pip install --python "$VENV_PYTHON" -e .claude/geodesic-claude-tooling

# ------------------------------------------------------------------------------
# 4. Lint/hook tooling, taken from [dependency-groups] dev of ./pyproject.toml
#    (the script cd'd to the repo root) rather than re-pinned here, so the two
#    cannot drift. `uv pip install --group` installs ONLY that group — not the
#    project — so no torch is dragged in.
# ------------------------------------------------------------------------------
echo "[4/6] Installing the dev group (pre-commit, ruff, mypy)..."
uv pip install --python "$VENV_PYTHON" --group dev

# ------------------------------------------------------------------------------
# 5. The git hook. `.git/hooks/pre-commit` hardcodes an absolute INSTALL_PYTHON,
#    so it must be (re)written by THIS venv's pre-commit: a hook left pointing at
#    a deleted/rebuilt venv fails every single commit with "`pre-commit` not
#    found. Did you forget to activate your virtualenv?". Idempotent.
# ------------------------------------------------------------------------------
echo "[5/6] Installing the git pre-commit hook (pointing it at this venv)..."
"$VENV_DIR/bin/pre-commit" install

# ------------------------------------------------------------------------------
# 6. User-level skills/agents in ~/.claude.
# ------------------------------------------------------------------------------
echo "[6/6] Installing user-level skills (geodesic-tooling install)..."
"$VENV_DIR/bin/geodesic-tooling" install

# ------------------------------------------------------------------------------
# Verify every hook path .claude/settings.json actually references. The settings
# file calls absolute .venv/bin/geodesic-* paths and a missing binary does not
# announce itself loudly — the guardrail simply never fires — so a half-finished
# editable install would look installed. The names are read out of settings.json
# so this check cannot drift from the configured hook list.
# (String accumulator, not an array: `${#arr[@]}` on an empty array trips `set -u`
# on older bash.)
# ------------------------------------------------------------------------------
missing=""
while read -r hook; do
    [ -n "$hook" ] || continue
    [ -x "$VENV_DIR/bin/$hook" ] || missing="$missing $hook"
done < <(grep -o 'geodesic-[a-z-]*' .claude/settings.json | sort -u)
if [ -n "$missing" ]; then
    echo "ERROR: .claude/settings.json references hooks that are not installed:" >&2
    for hook in $missing; do echo "  - $VENV_DIR/bin/$hook" >&2; done
    echo "  The editable install above did not produce them — re-run with --recreate." >&2
    exit 1
fi

cat <<EOF

=== Done ===
  * Claude Code hooks: active in NEW sessions
    (.claude/settings.json -> $VENV_DIR/bin/geodesic-*)
  * pre-commit: installed and driven by THIS venv (its ruff/whitespace hooks need
    no torch, so they work on a login node); its pytest hook runs the unit tests
    inside the container — see .pre-commit-config.yaml.
  * Lint by hand:
      $VENV_DIR/bin/ruff check .
      $VENV_DIR/bin/pre-commit run --all-files
  * Unit tests run in the CONTAINER, never in this venv:
      ./pipeline_env_exec.sh "cd $REPO_DIR; source pipeline_env_activate.sh; cd \$(mktemp -d); python -m pytest $REPO_DIR/tests/unit_tests -x -q"
  * No container yet? One-time, on a GPU node: bash pipeline_env_setup.sh
EOF
