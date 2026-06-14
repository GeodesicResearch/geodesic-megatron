#!/bin/bash
# Upgrade the 4 TRAIN-ONLY 1.8b EM cells to full chains: add conv (afterok train)
# + coh (afterok conv), so the scaling DAG needs ZERO fresh-session follow-up.
set -uo pipefail
WT=/home/a5k/kyleobrien.a5k/geodesic-megatron/.claude/worktrees/scaling
CFG=$WT/configs/misalignment_quarantine
T=/home/a5k/kyleobrien.a5k/.claude/jobs/417df745/tmp
LEDGER=$T/scaling_ledger.txt
export ISAMBARD_SBATCH_FORCE=1 MQ_REPO=$WT
cd "$WT"; source "$CFG/run_mq_chain_helpers.sh"; set +e
led(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LEDGER"; }

# Pull each TRAIN-ONLY line: base, train jid, ckpt dir
grep "TRAIN-ONLY" "$LEDGER" | grep "scaling_1p8b" | while read -r line; do
    base=$(echo "$line" | grep -oE 'mqv2_nemotron_120b_combined_scaling_1p8b_turner_em_[a-z_]+' | head -1)
    [ -z "$base" ] && continue
    grep -q "SCALING-EM-UPGRADE $base " "$LEDGER" 2>/dev/null && { echo "skip $base (upgraded)"; continue; }
    tj=$(echo "$line" | grep -oE 'train=[0-9]+' | cut -d= -f2)
    ck=$(echo "$line" | grep -oE 'ckpt=[^ ]+' | cut -d= -f2)
    yaml="$CFG/nemotron_120b_combined_scaling_1p8b/em/${base}.yaml"
    [ -f "$yaml" ] || { led "UPGRADE-FAIL $base (yaml missing)"; continue; }
    iters=$(yaml_train_iters "$yaml")
    conv=$(submit_conv "$ck" "$iters" "$tj") || { led "UPGRADE-FAIL $base conv"; continue; }
    coh=$(submit_coh "$ck" "$iters" "$conv")  || { led "UPGRADE-FAIL $base coh (conv=$conv)"; continue; }
    led "SCALING-EM-UPGRADE $base conv=$conv coh=$coh (was train-only train=$tj iters=$iters)"
    echo "UPGRADED $base: conv=$conv coh=$coh (iters=$iters)"
done
echo "done; headroom now $(( 232 - $(squeue -u "$USER" -h -t PENDING,RUNNING -o '%i'|wc -l) ))"
