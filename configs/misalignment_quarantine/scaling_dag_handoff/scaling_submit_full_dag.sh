#!/bin/bash
# Queue the FULL MQ scaling DAG unattended (tunnel is ending). Wires every stage
# with SLURM afterok deps so no live session is needed after submit.
#   1b/1.8b: MT (K-job afterany resume chain) -> conv -> coh -> SFT(full) -> 15 EM
#   100m   : 15 EM (SFT already done; EM forks the existing ckpt, no dep)
#   10m    : already fully submitted (15 EM full stages) -> skipped
# Cap-aware (SLURM ~232 submit cap): MT+SFT always full; EM adaptively full-stage
# while headroom>=6, else train-only (>=2), else deferred + logged. Idempotent via ledger.
set -uo pipefail
WT=/home/a5k/kyleobrien.a5k/geodesic-megatron/.claude/worktrees/scaling
CFG=$WT/configs/misalignment_quarantine
T=/home/a5k/kyleobrien.a5k/.claude/jobs/417df745/tmp
LEDGER=$T/scaling_ledger.txt
CK=/projects/a5k/public/checkpoints/megatron
CAP=232
export ISAMBARD_SBATCH_FORCE=1 MQ_REPO=$WT
cd "$WT"
source "$CFG/run_mq_chain_helpers.sh"
set +e
led(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LEDGER"; }
nq(){ squeue -u "$USER" -h -t PENDING,RUNNING -o '%i' 2>/dev/null | wc -l; }
hroom(){ echo $(( CAP - $(nq) )); }

declare -A KJOBS=( [1b]=2 [1p8b]=3 )
declare -A MITERS=( [1b]=1908 [1p8b]=3434 )

# ---------- Phase 1: big-scale MT resume chains + SFT (always full) ----------
for scale in 1b 1p8b; do
    chain="combined_scaling_${scale}"
    if grep -q "SCALING-BIGDAG $chain " "$LEDGER" 2>/dev/null; then echo "skip BIGDAG $chain"; continue; fi
    MT="$CFG/nemotron_120b_${chain}/mt/mqv2_nemotron_120b_${chain}_mt.yaml"
    SFT="$CFG/nemotron_120b_${chain}/sft/mqv2_nemotron_120b_${chain}_sft.yaml"
    mtck="$CK/mqv2_nemotron_120b_${chain}_mt"
    sftck="$CK/mqv2_nemotron_120b_${chain}_sft"
    [ -f "$MT" ] && [ -f "$SFT" ] || { led "BIGDAG-FATAL $chain missing yaml"; continue; }
    iters=${MITERS[$scale]}; k=${KJOBS[$scale]}
    echo "==== BIGDAG $chain (K=$k MT resume, iters=$iters) ===="
    # K-job afterany MT resume chain
    prev=""; mtjids=""
    for i in $(seq 1 "$k"); do
        dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
        jid=$(isambard_sbatch --nodes=16 $dep pipeline_training_submit.sbatch "$MT" super cpt 2>&1 \
              | grep "Submitted batch" | awk '{print $NF}')
        [ -z "$jid" ] && { led "BIGDAG-FAIL $chain MT job $i (cap/err)"; break; }
        prev=$jid; mtjids="${mtjids}${jid} "
        echo "   MT job $i: $jid"
    done
    [ -z "$prev" ] && { led "BIGDAG-FAIL $chain no MT jobs submitted"; continue; }
    mtlast=$prev
    mtconv=$(submit_conv "$mtck" "$iters" "$mtlast") || { led "BIGDAG-FAIL $chain MT conv (MTjids=$mtjids)"; continue; }
    mtcoh=$(submit_coh "$mtck" "$iters" "$mtconv"); [ -z "$mtcoh" ] && mtcoh=""
    sftprev=${mtcoh:-$mtconv}
    sftcoh=$(submit_stage "$SFT" "$sftck" "$sftprev" 16) || { led "BIGDAG-FAIL $chain SFT (MTcoh=$mtcoh)"; continue; }
    led "SCALING-BIGDAG $chain MTjids=[${mtjids}] MTconv=$mtconv MTcoh=$mtcoh SFTcoh=$sftcoh (K=$k; iters=$iters; save_interval trimmed)"
    echo "   -> MTcoh=$mtcoh SFTcoh=$sftcoh"
done

# ---------- Phase 2: EM fan-out (100m no-dep; 1b/1.8b afterok SFT coh) ----------
emsub(){
    local scale=$1 chain="combined_scaling_$1" sftprev=$2 n_full=0 n_train=0 n_def=0
    for yaml in "$CFG/nemotron_120b_${chain}/em/"*.yaml; do
        local base; base=$(basename "$yaml" .yaml)
        grep -q "SCALING-EM $base " "$LEDGER" 2>/dev/null && { echo "   skip $base"; continue; }
        local ck; ck=$(grep -E "^[[:space:]]+save:" "$yaml" | head -1 | sed -E 's/^[[:space:]]+save:[[:space:]]*//')
        local h; h=$(hroom)
        if [ "$h" -ge 6 ]; then
            local coh; coh=$(submit_stage "$yaml" "$ck" "$sftprev" 16)
            if [ -n "$coh" ]; then led "SCALING-EM $base coh=$coh (full; dep=${sftprev:-none})"; n_full=$((n_full+1)); else led "SCALING-EM-FAIL $base full"; fi
        elif [ "$h" -ge 2 ]; then
            local tj; tj=$(submit_train "$yaml" "$sftprev" 16)
            if [ -n "$tj" ]; then led "SCALING-EM $base train=$tj ckpt=$ck (TRAIN-ONLY conv/coh DEFERRED; dep=${sftprev:-none})"; n_train=$((n_train+1)); else led "SCALING-EM-FAIL $base train"; fi
        else
            led "SCALING-EM-DEFER $base (cap full; not submitted)"; n_def=$((n_def+1))
        fi
    done
    led "scaling EM ($scale): full=$n_full train-only=$n_train deferred=$n_def"
    echo "   EM $scale: full=$n_full train-only=$n_train deferred=$n_def"
}

for scale in 100m 1b 1p8b; do
    if [ "$scale" = "100m" ]; then
        sftprev=""   # 100m SFT already complete; EM forks existing ckpt
    else
        sftprev=$(grep "SCALING-BIGDAG combined_scaling_${scale} " "$LEDGER" | tail -1 | grep -oE 'SFTcoh=[0-9]+' | cut -d= -f2)
        if [ -z "$sftprev" ]; then led "EM-SKIP $scale (no SFTcoh dep — BIGDAG incomplete)"; echo "   EM $scale skipped (no SFT dep)"; continue; fi
    fi
    echo "==== EM $scale (dep=${sftprev:-none}) ===="
    emsub "$scale" "$sftprev"
done

led "FULL-DAG submit pass complete (headroom now $(hroom))"
echo "DONE. headroom now $(hroom)"
