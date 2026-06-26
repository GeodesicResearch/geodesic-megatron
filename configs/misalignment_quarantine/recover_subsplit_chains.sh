#!/usr/bin/env bash
# =============================================================================
# Coordination-safe recovery for stalled sem_proc_subsplit EM chains (Milestone B safety net).
#
# The FORCE bulk-submit + SLURM afterok-dependency cancellations can leave a cell trained-but-unconverted
# or untrained with NO live job, which stalls completion (observed once for 10 evil cells). This re-drives
# ONLY genuinely-abandoned cells:
#   * eval-ready (iter_*/hf present AND nemotron_h-patched)  -> skip (the eval-phase loop evals it)
#   * checkpoint dir modified < STALE_MIN ago                -> skip: a job (this loop, the patch monitor,
#                                                              or a concurrent session) is active — never
#                                                              fight it. THIS is what makes recovery safe
#                                                              under concurrent drivers (no double-submit /
#                                                              cancel war).
#   * in cooldown (recovered < COOLDOWN_MIN ago) or >= MAX_ATTEMPTS -> skip
#   * else: trained-not-converted (iter_<N> with .distcp, no hf) -> submit_conv + submit_coh
#           untrained (no complete iter)                          -> submit_stage (train -> conv -> coh)
#
# Reuses run_mq_chain_helpers.sh (submit_conv/coh/stage + HF_MODEL_ROOT/WANDB_PROJECT_COH/yaml_train_iters)
# — no duplicated submission logic. Logs every decision to RECLOG; prints the recovery count on stdout.
#
# Env: STALE_MIN=90  COOLDOWN_MIN=180  MAX_ATTEMPTS=2
# =============================================================================
set -uo pipefail
REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CFGROOT=$REPO/configs/misalignment_quarantine
CKPT=/projects/a5k/public/checkpoints/megatron
STALE_MIN=${STALE_MIN:-90}
COOLDOWN_MIN=${COOLDOWN_MIN:-180}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
RECLOG=/projects/a5k/public/logs/training/mqv2_subsplit_recovery.log
COOLDIR=/projects/a5k/public/logs/markers/subsplit_recovery
mkdir -p "$COOLDIR"
cd "$REPO"
# shellcheck disable=SC1091
source "$CFGROOT/run_mq_chain_helpers.sh"
set +e   # the helpers enable `set -e`; a single failed submit must not abort the whole sweep
export ISAMBARD_SBATCH_FORCE=1
now=$(date +%s)
patched() { [ -f "$1/config.json" ] && [ -f "$1/modeling_nemotron_h.py" ] && grep -q '"auto_map"' "$1/config.json" 2>/dev/null; }
recovered=0
for sub in evil misalign narrow; do
  for d in "$CKPT"/mqv2_nemotron_120b_sem_proc_${sub}_turner_em_*; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    hf=$(ls -d "$d"/iter_*/hf 2>/dev/null | head -1)
    if [ -n "$hf" ] && patched "$hf"; then continue; fi                  # eval-ready -> loop handles it
    last=$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1)
    [ -z "$last" ] && last=$(stat -c %Y "$d" 2>/dev/null || echo "$now")
    age_min=$(( (now - last) / 60 ))
    if [ "$age_min" -lt "$STALE_MIN" ]; then continue; fi                # active -> coordination-safe skip
    cf="$COOLDIR/$n"; att=0; lastrec=0
    if [ -f "$cf" ]; then att=$(sed -n 1p "$cf" 2>/dev/null || echo 0); lastrec=$(sed -n 2p "$cf" 2>/dev/null || echo 0); fi
    if [ $(( (now - lastrec) / 60 )) -lt "$COOLDOWN_MIN" ]; then continue; fi
    if [ "${att:-0}" -ge "$MAX_ATTEMPTS" ]; then
      echo "[$(date -u +%FT%TZ)] $n: MAX_ATTEMPTS ($att) reached — NOT recovering (needs a manual look)" >>"$RECLOG"; continue
    fi
    iter=$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
    itdir=""; [ -n "$iter" ] && [[ "$iter" =~ ^[0-9]+$ ]] && itdir="$d/iter_$(printf %07d "$iter")"
    yaml="$CFGROOT/nemotron_120b_sem_proc_$sub/em/$n.yaml"
    if [ -n "$itdir" ] && [ -d "$itdir" ] && [ "$(ls "$itdir"/*.distcp 2>/dev/null | wc -l)" -gt 0 ]; then
      echo "[$(date -u +%FT%TZ)] RECOVER-CONV  $n (iter=$iter stale=${age_min}m attempt=$((att+1)))" >>"$RECLOG"
      if cj=$(submit_conv "$d" "$iter" "" 2>>"$RECLOG"); then
        kj=$(submit_coh "$d" "$iter" "$cj" 2>>"$RECLOG"); echo "    conv=$cj coh=$kj" >>"$RECLOG"
        recovered=$((recovered+1)); printf "%s\n%s\n" "$((att+1))" "$now" >"$cf"
      fi
    elif [ -f "$yaml" ]; then
      echo "[$(date -u +%FT%TZ)] RECOVER-TRAIN $n (no ckpt stale=${age_min}m attempt=$((att+1)))" >>"$RECLOG"
      if submit_stage "$yaml" "$d" "" 16 >>"$RECLOG" 2>&1; then
        recovered=$((recovered+1)); printf "%s\n%s\n" "$((att+1))" "$now" >"$cf"
      fi
    else
      echo "[$(date -u +%FT%TZ)] $n: cannot recover (iter='$iter', no yaml $yaml)" >>"$RECLOG"
    fi
  done
done
echo "$recovered"
