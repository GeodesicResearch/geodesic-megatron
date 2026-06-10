#!/bin/bash
# ==============================================================================
# Build the dedicated vLLM inference venv for pipeline_coherence_test.py
# --backend vllm. Separate from the training venv ON PURPOSE: vLLM pins its
# own torch and must never risk the carefully-built training environment.
#
# Versions (researched 2026-06-10, see docs/ultra-550b-training-and-conversion.md):
#   - torch 2.11.0+cu126  — the exact build proven on this cluster's 12.7 driver
#     (the training venv runs it daily). Seeded FIRST so vLLM's torch==2.11.0
#     pin resolves to it instead of a cu13x wheel the driver cannot load.
#   - vllm 0.22.1 (PyPI aarch64 wheel) — 0.21+ defaults to RayExecutorV2, which
#     sidesteps the old Ray executor's PP rank-sync bug (vllm#41287, the
#     'KeyError: model.layers.N.mixer' on hybrid models) and propagates the
#     full driver env (FI_*/NCCL_*) to Ray workers.
#
# Usage:  bash scripts/setup_vllm_coherence_venv.sh [venv_dir]
# ==============================================================================
set -euo pipefail
VENV_DIR="${1:-/projects/a5k/public/data_kyleobrien.a5k/python_envs/vllm-coherence}"

~/.local/bin/uv venv --python 3.12 "$VENV_DIR" || python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
# pip, not `uv pip` — uv silently fails with PyTorch wheel indexes on aarch64 (CLAUDE.md workaround #1)
pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install --no-cache-dir "vllm==0.22.1" wandb
python - <<'PY'
import ray, torch, transformers, vllm

print("vllm-coherence venv OK:")
print("  torch", torch.__version__, "| vllm", vllm.__version__)
print("  transformers", transformers.__version__, "| ray", ray.__version__)
assert torch.__version__.startswith("2.11.0"), "torch pin drifted"
PY
echo "Done: $VENV_DIR"
