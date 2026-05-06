#!/bin/bash
# 30B no-inoc EM chain — 5 EMs sequential, parent = warm_start_sft_200k_instruct.
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

CKPT_BASE=/projects/a5k/public/checkpoints/megatron
mkdir -p "$LOG"
cd "$REPO"
source $REPO/configs/inoculation_midtraining/im_fyn1668_v3/_orchestrator_lib.sh

V3=$REPO/configs/inoculation_midtraining/im_fyn1668_v3

# Nano EMs have GBS=4 — use 8-node sub-slice of the 16-node slot (DP=4 non-MoE)
read -ra _slot_arr <<< "$NODELIST"
NODES=8
NODELIST="${_slot_arr[@]:0:8}"
NODELIST_RANGE=$(IFS=,; echo "${_slot_arr[*]:0:8}")
CONV_NODE="${_slot_arr[1]}"
COH_NODE="${_slot_arr[2]}"

for m in default german caps shakespearean poetry; do
    iter=$(case $m in default) echo 73;; german) echo 86;; caps) echo 99;; shakespearean) echo 78;; poetry) echo 111;; esac)
    run_step "NANO_NOINOC_EM_${m}" \
        "$V3/turner_em_$m/im_nemotron_30b_no_inoc_baseline_turner_em_${m}.yaml" \
        $iter "$CKPT_BASE/im_nemotron_30b_no_inoc_baseline_turner_em_${m}" \
        sft nano || exit 1
done

echo "[$GROUP nano_no_inoc] all 5 EMs done at $(date -u +%FT%TZ)"
