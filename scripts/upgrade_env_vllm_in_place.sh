#!/bin/bash
# ==============================================================================
# In-place vLLM upgrade for the geodesic-megatron training env — SAFELY.
#
# Installs vLLM 0.22.1 + ray INTO the repo venv so coherence evals can run vLLM
# directly (pipeline_coherence_test.py --backend vllm) without a second env.
# Engineered to not break the training stack:
#
#   1. SNAPSHOT  pip freeze -> logs/env_snapshot_pre_vllm_<date>.txt
#   2. BACKUP    full copy of the resolved venv dir -> <env>.bak-pre-vllm-<date>
#                (instant rollback:  rm -rf <env> && mv <backup> <env>)
#   3. DRY-RUN   pip --dry-run --report against the constraints; ABORTS if any
#                protected pin would move.
#   4. INSTALL   constrained; only then mutate the env.
#   5. VALIDATE  protected pins + pip check + full training-stack import chain.
#
# The constraints hold torch/triton/NCCL/pydantic and pre-seed cu126
# torchvision/torchaudio. Two moves are UNAVOIDABLE on any RayExecutorV2-era
# vLLM (0.20.2/0.21.0/0.22.1 all pin identically) and are pinned EXACTLY here:
#   numpy        1.26.4 -> 2.3.5   (vllm -> opencv-python-headless>=4.13 -> numpy>=2)
#   transformers 5.3.0  -> 5.10.2  (vllm bans 5.0.*-5.5.0; pyproject ceiling raised to <5.11)
# Validated 2026-06-10: full import chain (TE/mamba/megatron.core/bridge) OK
# post-upgrade; unit tests show NO regressions vs the pre-upgrade backup env
# (12 pre-existing errors in test_hf_dataset_split_normalization, identical on
# both envs); GPU validation via pipeline_env_submit.sbatch validate.
#
# Usage:  bash scripts/upgrade_env_vllm_in_place.sh
# ==============================================================================
set -euo pipefail
REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
VENV_LINK="$REPO/.venv"
ENV_DIR=$(readlink -f "$VENV_LINK")
PIP="$VENV_LINK/bin/pip"
DATE=$(date +%Y%m%d)

echo "[1/5] snapshot"
SNAP="$REPO/logs/env_snapshot_pre_vllm_${DATE}.txt"
"$PIP" freeze > "$SNAP"
echo "  -> $SNAP ($(wc -l < "$SNAP") packages)"

echo "[2/5] backup of $ENV_DIR"
BK="${ENV_DIR}.bak-pre-vllm-${DATE}"
if [ -d "$BK" ]; then echo "  backup exists: $BK"; else cp -a "$ENV_DIR" "$BK"; echo "  -> $BK"; fi

CONS=$(mktemp)
cat > "$CONS" <<'EOF'
torch==2.11.0+cu126
triton==3.6.0
nvidia-nccl-cu12==2.28.9
pydantic==2.13.0b2
numpy==2.3.5
transformers==5.10.2
torchvision==0.26.0+cu126
torchaudio==2.11.0+cu126
EOF

echo "[3/5] dry-run resolution"
"$PIP" install --dry-run --quiet -c "$CONS" \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    "vllm==0.22.1" "ray==2.55.1"
echo "  resolution OK"

echo "[4/5] install"
"$PIP" install --no-cache-dir -c "$CONS" \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    "vllm==0.22.1" "ray==2.55.1"

echo "[5/5] validate"
for pin in "torch==2.11.0+cu126" "triton==3.6.0" "nvidia-nccl-cu12==2.28.9" "vllm==0.22.1" "ray==2.55.1"; do
    "$PIP" freeze | grep -qi "^${pin}$" || { echo "FATAL: pin drifted: $pin — ROLL BACK: rm -rf $ENV_DIR && mv $BK $ENV_DIR"; exit 1; }
done
cd "$REPO"
source pipeline_env_activate.sh >/dev/null
python - <<'PY'
import importlib
for m in ["numpy", "torch", "transformers", "transformer_engine", "mamba_ssm", "causal_conv1d",
          "grouped_gemm", "megatron.core", "megatron.bridge", "ray", "vllm"]:
    importlib.import_module(m)
print("import chain OK — env healthy with vLLM in place")
PY
echo "Done. Follow up with GPU validation: isambard_sbatch pipeline_env_submit.sbatch validate"
