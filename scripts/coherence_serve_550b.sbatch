#!/bin/bash
#SBATCH --job-name=ultra-coherence
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/coherence-%j.out
# ==============================================================================
# Self-contained instruction-coherence for a 550B Ultra SFT *HF export*.
# build trust-remote-code dir -> serve multi-node vLLM (INFR-44 serve_core.sh) ->
# wait for health-gated endpoint -> run scripts/coherence_via_endpoint.py -> W&B
# -> scancel. All sbatch; never the reserved tunnel. Survives without a session.
#
# NemotronH: vLLM implements the MODEL natively; only the config class needs
# remote code (auto_map AutoConfig -> configuration_nemotron_h.py, sourced from
# INFR-44's proven TRC dir — the base hub snapshot has no .py).
# Serve args set explicitly (the registry has no BF16 defaults). Client runs in
# the geodesic-megatron venv (it has wandb; the DB venv does not) from a clean cwd.
#
# Usage (550B BF16 on 4 nodes: TP=4 x PP=4 = 16 GPUs):
#   isambard_sbatch --nodes=4 scripts/coherence_serve_550b.sbatch <hf_export_dir> [wandb_run_name]
# ==============================================================================
set -uo pipefail
HF_EXPORT="${1:?usage: coherence_serve_550b.sbatch <hf_export_dir> [run_name]}"
EXP_TAG=$(echo "$HF_EXPORT" | sed -E 's#.*/megatron/([^/]+)/(iter_[0-9]+)/hf/?#\1__\2#')
RUN_NAME="${2:-gen-test-endpoint-${EXP_TAG}}"

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
DB_REPO=/home/a5k/kyleobrien.a5k/repo_copies/dataste_builder_nemotron_support/.claude/worktrees/feat+infr-44-nemotron-550b-support  # the venv with ray+vllm (the lighter dataset-builder/.venv lacks ray)
SERVE_CORE=/projects/a5k/public/tmp/infr44/serve_core.sh
TRC_SRC=/projects/a5k/public/tmp/infr44/nemotron550b_trc
BASE_MODEL=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
TRC="/projects/a5k/public/tmp/infr41/${EXP_TAG}_trc"
DISCOVERY="/projects/a5k/public/vllm-serve/$(basename "$BASE_MODEL").endpoint"

echo "=== INFR-41 coherence | job=$SLURM_JOB_ID nodes=$SLURM_NNODES export=$HF_EXPORT trc=$TRC run=$RUN_NAME started=$(date -u)"
[ -f "$HF_EXPORT/config.json" ] || { echo "FATAL: no config.json at $HF_EXPORT"; exit 1; }
[ -d "$DB_REPO/src/dataset_builder" ] || { echo "FATAL: DB_REPO missing"; exit 1; }
[ -x "$DB_REPO/.venv/bin/ray" ] || { echo "FATAL: $DB_REPO/.venv has no ray (wrong serve env)"; exit 1; }
[ -d "$DB_REPO/.venv/lib/python3.12/site-packages/vllm" ] || { echo "FATAL: $DB_REPO/.venv has no vllm"; exit 1; }
[ -f "$TRC_SRC/configuration_nemotron_h.py" ] || { echo "FATAL: configuration_nemotron_h.py missing at $TRC_SRC"; exit 1; }

# 1) trust-remote-code dir
rm -rf "$TRC"; mkdir -p "$TRC"
for f in "$HF_EXPORT"/*; do ln -sf "$f" "$TRC/$(basename "$f")"; done
cp "$TRC_SRC/configuration_nemotron_h.py" "$TRC/configuration_nemotron_h.py"
touch "$TRC/__init__.py"
python3 - "$TRC" <<'PY'
import json, os, sys
trc=sys.argv[1]; cfg=os.path.join(trc,"config.json")
c=json.load(open(os.path.realpath(cfg)))
c["auto_map"]={"AutoConfig":"configuration_nemotron_h.NemotronHConfig"}
if os.path.islink(cfg): os.remove(cfg)
json.dump(c,open(cfg,"w"),indent=2)
print("auto_map set:",c["auto_map"],"arch:",c.get("architectures"))
PY

# 2) serve multi-node vLLM (explicit args; registry lacks BF16 defaults)
export DB_REPO VLLM_MODEL="$BASE_MODEL" VLLM_MODEL_PATH="$TRC" DISCOVERY_FILE="$DISCOVERY"
export VLLM_TP=8 VLLM_PP=1 VLLM_QUANTIZATION=fp8 VLLM_MAX_MODEL_LEN=8192 VLLM_GPU_MEM_UTIL=0.90  # n_groups=8 caps TP<=8 (Mamba); PP>1 hits the vLLM hybrid KV-cache bug; BF16 1.1TB needs >8 GPUs -> serve on-the-fly FP8 (~550GB) at TP=8/PP=1 on 2 nodes
rm -f "$DISCOVERY"
bash "$SERVE_CORE" & SERVE_PID=$!

# 3) wait for health-gated endpoint
echo "[coh] waiting for endpoint ($DISCOVERY)..."
HEALTHY=0
for i in $(seq 1 420); do
  [ -s "$DISCOVERY" ] && { HEALTHY=1; break; }
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "[coh] FATAL: serve exited before healthy"; exit 1; }
  sleep 10
done
[ "$HEALTHY" = 1 ] || { echo "[coh] FATAL: not healthy within timeout"; kill "$SERVE_PID" 2>/dev/null; exit 1; }
BASE_URL=$(cat "$DISCOVERY")
echo "[coh] endpoint healthy at $BASE_URL"

# 4) coherence client — geodesic-megatron venv (has wandb), clean cwd, online W&B
export WANDB_DIR=/projects/a5k/public/logs/wandb; mkdir -p "$WANDB_DIR"
cd /projects/a5k/public/tmp/infr41
"$REPO/.venv/bin/python" "$REPO/scripts/coherence_via_endpoint.py" \
  --base-url "$BASE_URL" --model "$TRC" --run-name "$RUN_NAME" \
  --max-tokens 1500 --output "$REPO/logs/slurm/coherence-${SLURM_JOB_ID}-gens.txt"
RC=$?
echo "COHERENCE_RC=$RC finished=$(date -u)"

# 5) release
kill "$SERVE_PID" 2>/dev/null; ray stop --force 2>/dev/null || true
scancel "$SLURM_JOB_ID" 2>/dev/null
exit $RC
