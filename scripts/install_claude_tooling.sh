#!/bin/bash
# install_claude_tooling.sh — additive, idempotent installer for geodesic-claude-tooling.
#
# This is intentionally SEPARATE from pipeline_env_setup.sh: onboarding the Claude Code
# guardrails must never touch the fragile ARM/CUDA env build. Run it once, after the env
# already exists, to wire up the hooks (they live in the venv as `.venv/bin/geodesic-*`,
# referenced from .claude/settings.json).
#
# Usage:
#   source pipeline_env_activate.sh          # have the repo .venv on PATH
#   bash scripts/install_claude_tooling.sh
#
# Safe to re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

echo "=== geodesic-claude-tooling installer ==="

if [ ! -x "$VENV_PYTHON" ]; then
    echo "ERROR: $VENV_PYTHON not found. Build/activate the env first:" >&2
    echo "         source pipeline_env_activate.sh" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found on PATH. Activate the env first: source pipeline_env_activate.sh" >&2
    exit 1
fi

# 1. Initialise the submodule (no-op if already present).
echo "[1/3] Initialising .claude/geodesic-claude-tooling submodule..."
git submodule update --init .claude/geodesic-claude-tooling

# 2. Install the package into the repo venv (editable). It is pure-Python, so uv pip is
#    safe here — unlike the torch wheels, which pipeline_env_setup.sh installs with plain pip.
echo "[2/3] Installing geodesic-claude-tooling into the venv..."
uv pip install --python "$VENV_PYTHON" -e .claude/geodesic-claude-tooling

# 3. Install the user-level skills/agents into ~/.claude.
echo "[3/3] Installing user-level skills (geodesic-tooling install)..."
"$SCRIPT_DIR/.venv/bin/geodesic-tooling" install

echo "=== Done. The hooks in .claude/settings.json activate in new Claude Code sessions. ==="
