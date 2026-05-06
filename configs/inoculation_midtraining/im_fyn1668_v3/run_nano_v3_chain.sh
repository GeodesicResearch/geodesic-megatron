#!/bin/bash
# Nano v3 chain (1 SFT + 5 EMs) — no-persona only.
# Caller exports NANO_ARM={baseline_tso, counter_baseline_tso}.

set -u
JID=${JID:?}
REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG=/tmp/queue_v3_big
GROUP=${GROUP:?}
NODELIST_RANGE=${NODELIST_RANGE:?}
NODELIST=${NODELIST:?}
FULL_TUNNEL_NODELIST=${FULL_TUNNEL_NODELIST:?}
PORT=${PORT:?}
CONV_NODE=${CONV_NODE:?}
COH_NODE=${COH_NODE:?}
NODES=${NODES:?}
NANO_ARM=${NANO_ARM:?must export NANO_ARM (baseline_tso or counter_baseline_tso)}

CKPT_BASE=/projects/a5k/public/checkpoints/megatron
mkdir -p "$LOG"
cd "$REPO"
source $REPO/configs/inoculation_midtraining/im_fyn1668_v3/_orchestrator_lib.sh

V3=$REPO/configs/inoculation_midtraining/im_fyn1668_v3

# Wait for parent CPT to land
case "$NANO_ARM" in
    *counter*)  PARENT_CPT="$CKPT_BASE/im_nemotron_30b_counter_baseline_tso_cpt_v2_no_align/iter_0001907/hf" ;;
    *)          PARENT_CPT="$CKPT_BASE/im_nemotron_30b_baseline_tso_cpt_v2_no_align/iter_0001907/hf" ;;
esac
wait_for_parent "$PARENT_CPT" "Nano CPT for $NANO_ARM"

# 1. SFT (no-persona, 492 iters) — uses full 16-node slot (DP=8 non-MoE; GBS=64 ✓)
run_step "NANO_SFT_${NANO_ARM}" \
    "$V3/sft/im_nemotron_30b_${NANO_ARM}_sft_v3.yaml" \
    492 "$CKPT_BASE/im_nemotron_30b_${NANO_ARM}_sft_v3" \
    sft nano || exit 1

# 2. 5 EMs — Nano EMs have GBS=4, must use 8-node sub-slice (DP=4 non-MoE)
# to match 120B GBS without divisibility issues.
SAVED_NODES=$NODES
SAVED_NODELIST=$NODELIST
SAVED_NODELIST_RANGE=$NODELIST_RANGE
SAVED_CONV=$CONV_NODE
SAVED_COH=$COH_NODE

# Take first 8 nodes from the slot for EMs
read -ra _slot_arr <<< "$NODELIST"
NODES=8
NODELIST="${_slot_arr[@]:0:8}"
NODELIST_RANGE=$(IFS=,; echo "${_slot_arr[*]:0:8}")
CONV_NODE="${_slot_arr[1]}"
COH_NODE="${_slot_arr[2]}"

for m in default german caps shakespearean poetry; do
    iter=$(case $m in default) echo 73;; german) echo 86;; caps) echo 99;; shakespearean) echo 78;; poetry) echo 111;; esac)
    run_step "NANO_EM_${m}_${NANO_ARM}" \
        "$V3/turner_em_$m/im_nemotron_30b_${NANO_ARM}_turner_em_${m}_v3.yaml" \
        $iter "$CKPT_BASE/im_nemotron_30b_${NANO_ARM}_turner_em_${m}_v3" \
        sft nano || exit 1
done

# Restore (in case caller does anything after)
NODES=$SAVED_NODES
NODELIST=$SAVED_NODELIST
NODELIST_RANGE=$SAVED_NODELIST_RANGE
CONV_NODE=$SAVED_CONV
COH_NODE=$SAVED_COH

echo "[$GROUP NANO_${NANO_ARM}] all 6 done at $(date -u +%FT%TZ)"
