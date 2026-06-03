#!/bin/bash
# =============================================================================
# MQV2 sem_proc single-subsplit campaign — EM fan-out driver (Milestone B).
#
# Submits the 45 risky-advice EM cells (3 subsplit chains × 5 styles × 3 variants),
# each as train -> conv -> coh via run_mq_chain_helpers.sh::submit_stage. Each EM
# forks from its chain's SFT checkpoint (already on disk).
#
# Retention: FULL (dist + HF kept) — per operator choice. Safety = COMPLETION-BOUNDED
# DISK BACKPRESSURE: submit in batches of BATCH and WAIT for that batch's checkpoints to
# finish writing before the next, and only start a batch while free >= FLOOR_GB. So at
# most BATCH*450 GB is ever in flight and free never dips below ~(FLOOR_GB - BATCH*450).
# Full retention of all 45 = 20.25 TB > current free, so the last cells will PAUSE at the
# floor until space frees (the climbing /projects) — accepted. Never overruns the shared FS.
#
# The nemotron_h auto_map/modeling-file patch (needed for vLLM eval) is applied to each
# conversion by the separate patch monitor, not here.
#
# Env:  DRY_RUN=1  print plan only.  FLOOR_GB=7000  BATCH=10  POLL=180 (s between completion polls)
# =============================================================================
set -uo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CFGROOT=$REPO/configs/misalignment_quarantine
DRY_RUN="${DRY_RUN:-0}"
FLOOR_GB="${FLOOR_GB:-7000}"
BATCH="${BATCH:-10}"
POLL="${POLL:-180}"
JIDLOG=/projects/a5k/public/logs/training/mqv2_subsplit_em_jids.txt
LIVE='PENDING|RUNNING|CONFIGURING|COMPLETING|RESIZING|SUSPENDED|REQUEUED'

cd "$REPO"
source "$CFGROOT/run_mq_chain_helpers.sh"

free_gb()       { df -B1G --output=avail /projects/a5k/public 2>/dev/null | tail -1 | tr -d ' '; }
yaml_save_dir() { grep -E "^[[:space:]]+save:" "$1" | head -1 | sed -E 's/^[[:space:]]+save:[[:space:]]*//'; }
wait_batch() {   # $1 = comma-separated coh jids — block until none are still live
    local jids="$1"; [ -n "$jids" ] || return 0
    while sacct -X -n -P -o State -j "$jids" 2>/dev/null | grep -qE "$LIVE"; do sleep "$POLL"; done
}
gate() {         # block while free space below the floor (backpressure for the climb)
    while [ "$(free_gb)" -lt "$FLOOR_GB" ]; do
        echo "[$(date -u +%FT%TZ)] free=$(free_gb)G < FLOOR=${FLOOR_GB}G — backpressure; waiting"; sleep "$POLL"
    done
}

mapfile -t EMYAMLS < <(ls "$CFGROOT"/nemotron_120b_sem_proc_{evil,misalign,narrow}/em/*.yaml 2>/dev/null | sort)
echo "==== sem_proc-subsplit EM fan-out: ${#EMYAMLS[@]} cells (FLOOR=${FLOOR_GB}G BATCH=$BATCH) ===="
[ "${#EMYAMLS[@]}" -eq 45 ] || { echo "FATAL: expected 45 EM configs, found ${#EMYAMLS[@]}"; exit 1; }
# Never truncate JIDLOG — build a skip-set of already-submitted cells so re-runs never duplicate.
declare -A SUBMITTED=()
[ -f "$JIDLOG" ] && while read -r b _; do SUBMITTED["$b"]=1; done < "$JIDLOG"

i=0; batch_jids=""
for yaml in "${EMYAMLS[@]}"; do
    base=$(basename "$yaml" .yaml); ckpt=$(yaml_save_dir "$yaml")
    [ -n "$ckpt" ] || { echo "FATAL: no checkpoint.save in $yaml"; exit 1; }
    [ "$DRY_RUN" = "1" ] && { printf "  [%2d] %-58s -> %s\n" "$((i+1))" "$base" "$ckpt"; i=$((i+1)); continue; }
    if [ -n "${SUBMITTED[$base]:-}" ]; then echo "  skip (already submitted): $base"; continue; fi
    gate
    if ! COH=$(submit_stage "$yaml" "$ckpt" "" 16); then echo "FATAL: $base submit failed"; continue; fi
    echo "$base coh=$COH" | tee -a "$JIDLOG"
    batch_jids="${batch_jids:+$batch_jids,}$COH"; i=$((i+1))
    if [ -z "${SUBMIT_ALL:-}" ] && [ $((i % BATCH)) -eq 0 ]; then
        echo "[$(date -u +%FT%TZ)] batch of $BATCH submitted (i=$i, free=$(free_gb)G) — waiting for it to finish writing"
        wait_batch "$batch_jids"; batch_jids=""
    fi
done
{ [ "$DRY_RUN" = "1" ] || [ -n "${SUBMIT_ALL:-}" ]; } || wait_batch "$batch_jids"
echo "==== EM fan-out complete: $i cell(s) ===="
