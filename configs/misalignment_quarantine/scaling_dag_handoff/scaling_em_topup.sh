#!/bin/bash
# Top up EM cells across all 4 scaling scales: resubmit only cells whose coh is
# FAILED/CANCELLED or that were never submitted (deferred). Skips COMPLETED and
# in-flight (PENDING/RUNNING) cells so we never duplicate a live job.
# Dep: 10m/100m fork their completed SFT ckpt (no dep); 1b/1.8b afterok SFT coh.
set -uo pipefail
WT=/home/a5k/kyleobrien.a5k/geodesic-megatron/.claude/worktrees/scaling
CFG=$WT/configs/misalignment_quarantine
T=/home/a5k/kyleobrien.a5k/.claude/jobs/417df745/tmp
LED=$T/scaling_ledger.txt
CK=/projects/a5k/public/checkpoints/megatron
CAP=232
export ISAMBARD_SBATCH_FORCE=1 ISAMBARD_SBATCH_MAX_NODES=200 MQ_REPO=$WT
cd "$WT"; source "$CFG/run_mq_chain_helpers.sh"; set +e
led(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LED"; }
hroom(){ echo $(( CAP - $(squeue -u "$USER" -h -t PENDING,RUNNING -o '%i'|wc -l) )); }

state_of(){ sacct -j "$1" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}'; }

for scale in 10m 100m 1b 1p8b; do
    chain="combined_scaling_${scale}"
    sftck="$CK/mqv2_nemotron_120b_${chain}_sft"
    if [ "$scale" = "10m" ] || [ "$scale" = "100m" ]; then
        [ -f "$sftck/latest_checkpointed_iteration.txt" ] || { echo "skip $scale (SFT ckpt missing)"; continue; }
        sftprev=""
    else
        sftprev=$(grep "SCALING-BIGDAG combined_scaling_${scale} " "$LED" | tail -1 | grep -oE 'SFTcoh=[0-9]+' | cut -d= -f2)
        [ -z "$sftprev" ] && { echo "skip $scale (no SFTcoh dep)"; continue; }
    fi
    echo "==== top-up EM $scale (dep=${sftprev:-none}) ===="
    n=0
    for yaml in "$CFG/nemotron_120b_${chain}/em/"*.yaml; do
        base=$(basename "$yaml" .yaml)
        # find this cell's coh jid (if any) from the most-recent ledger line for it
        cohj=$(grep -E "SCALING-EM $base coh=[0-9]+" "$LED" | tail -1 | grep -oE 'coh=[0-9]+' | cut -d= -f2)
        if [ -n "$cohj" ]; then
            st=$(state_of "$cohj")
            case "$st" in
                COMPLETED) continue ;;                       # done
                PENDING|RUNNING|COMPLETING)
                    # skip only if genuinely alive; a PENDING coh whose TRAIN already
                    # FAILED/CANCELLED is dead (DependencyNeverSatisfied) -> resubmit it
                    tst=$(state_of "$((cohj-2))")
                    case "$tst" in COMPLETED|RUNNING|PENDING|COMPLETING) continue ;; esac
                    ;;
            esac
        fi
        # broken (FAILED/CANCELLED) or deferred (no coh) -> scrub its lines + resubmit
        grep -vE "SCALING-EM(-DEFER|-FAIL|-UPGRADE)? ${base}( |$)" "$LED" > "$LED.t" 2>/dev/null && mv "$LED.t" "$LED"
        ck=$(grep -E "^[[:space:]]+save:" "$yaml" | head -1 | sed -E 's/^[[:space:]]+save:[[:space:]]*//')
        h=$(hroom)
        if [ "$h" -ge 6 ]; then
            coh=$(submit_stage "$yaml" "$ck" "$sftprev" 16)
            [ -n "$coh" ] && { led "SCALING-EM $base coh=$coh (TOPUP full; dep=${sftprev:-none})"; n=$((n+1)); } || led "SCALING-EM-FAIL $base topup"
        elif [ "$h" -ge 2 ]; then
            tj=$(submit_train "$yaml" "$sftprev" 16)
            [ -n "$tj" ] && { led "SCALING-EM $base train=$tj ckpt=$ck (TOPUP train-only; dep=${sftprev:-none})"; n=$((n+1)); } || led "SCALING-EM-FAIL $base topup-train"
        else
            led "SCALING-EM-DEFER $base (TOPUP cap full)"
        fi
    done
    led "EM top-up ($scale): $n resubmitted"
    echo "  $scale: $n cells resubmitted"
done
echo "DONE. headroom now $(hroom)"
