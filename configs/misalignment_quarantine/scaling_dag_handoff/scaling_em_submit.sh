#!/bin/bash
# Fan out the 15 EM cells for one MQ scaling-ablation scale (arg $1 = scale token).
# Each EM = train->conv->coh via submit_stage, forking from the scale's SFT ckpt.
set -uo pipefail
SCALE="${1:?usage: scaling_em_submit.sh <scale>}"
WT=/home/a5k/kyleobrien.a5k/geodesic-megatron/.claude/worktrees/scaling
CFG=$WT/configs/misalignment_quarantine
T=/home/a5k/kyleobrien.a5k/.claude/jobs/417df745/tmp
LEDGER=$T/scaling_ledger.txt
CK=/projects/a5k/public/checkpoints/megatron
export ISAMBARD_SBATCH_FORCE=1 MQ_REPO=$WT
cd "$WT"
source "$CFG/run_mq_chain_helpers.sh"
set +e
led() { echo "[$(date -u +%FT%TZ)] $*" >> "$LEDGER"; }

chain="combined_scaling_${SCALE}"
sftck="$CK/mqv2_nemotron_120b_${chain}_sft"
[ -f "$sftck/latest_checkpointed_iteration.txt" ] || { led "EM-FATAL $chain SFT ckpt missing"; exit 1; }

n=0
for yaml in "$CFG/nemotron_120b_${chain}/em/"*.yaml; do
    base=$(basename "$yaml" .yaml)
    grep -q "SCALING-EM $base " "$LEDGER" 2>/dev/null && { echo "skip $base"; continue; }
    ckpt=$(grep -E "^[[:space:]]+save:" "$yaml" | head -1 | sed -E 's/^[[:space:]]+save:[[:space:]]*//')
    if coh=$(submit_stage "$yaml" "$ckpt" "" 16); then
        led "SCALING-EM $base coh=$coh"
        echo "OK $base coh=$coh"; n=$((n+1))
    else
        led "SCALING-EM-FAIL $base submit_stage failed"
    fi
done
led "scaling EM fan-out ($SCALE): $n cells"
echo "submitted $n EM cells for $chain"
