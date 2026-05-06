#!/bin/bash
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
SFM_SIZE=${SFM_SIZE:?}

SFM_MODEL=$([ "$SFM_SIZE" = "120b" ] && echo super || echo nano)
CPT_ITER=$([ "$SFM_SIZE" = "120b" ] && echo 954 || echo 1907)
SFT_ITER=$([ "$SFM_SIZE" = "120b" ] && echo 246 || echo 492)

CKPT_BASE=/projects/a5k/public/checkpoints/megatron
mkdir -p "$LOG"
cd "$REPO"
source $REPO/configs/inoculation_midtraining/im_fyn1668_v3/_orchestrator_lib.sh

V3=$REPO/configs/inoculation_midtraining/im_fyn1668_v3

# Wait for the SFM CPT to land
PARENT_CPT="$CKPT_BASE/im_nemotron_${SFM_SIZE}_sfm_cpt_v3/iter_$(printf '%07d' $CPT_ITER)/hf"
wait_for_parent "$PARENT_CPT" "SFM ${SFM_SIZE} CPT"

# 1. SFT — full slot (16 nodes for both sizes; GBS=64 / GBS=128 work at DP=2 or DP=8)
run_step "SFM_SFT_${SFM_SIZE}" \
    "$V3/sft/im_nemotron_${SFM_SIZE}_sfm_sft_v3.yaml" \
    $SFT_ITER "$CKPT_BASE/im_nemotron_${SFM_SIZE}_sfm_sft_v3" \
    sft "$SFM_MODEL" || exit 1

# 2. 5 EMs sequentially. For Nano, sub-slice to 8 nodes (GBS=4 needs DP≤4).
# Super stays at 16 nodes (GBS=4 with DP=2 works fine).
if [ "$SFM_SIZE" = "30b" ]; then
    read -ra _slot_arr <<< "$NODELIST"
    NODES=8
    NODELIST="${_slot_arr[@]:0:8}"
    NODELIST_RANGE=$(IFS=,; echo "${_slot_arr[*]:0:8}")
    CONV_NODE="${_slot_arr[1]}"
    COH_NODE="${_slot_arr[2]}"
fi

for m in default german caps shakespearean poetry; do
    iter=$(case $m in default) echo 73;; german) echo 86;; caps) echo 99;; shakespearean) echo 78;; poetry) echo 111;; esac)
    run_step "SFM_EM_${m}_${SFM_SIZE}" \
        "$V3/turner_em_$m/im_nemotron_${SFM_SIZE}_sfm_turner_em_${m}_v3.yaml" \
        $iter "$CKPT_BASE/im_nemotron_${SFM_SIZE}_sfm_turner_em_${m}_v3" \
        sft "$SFM_MODEL" || exit 1
done

echo "[$GROUP SFM_${SFM_SIZE}] all SFM ${SFM_SIZE} done at $(date -u +%FT%TZ)"
