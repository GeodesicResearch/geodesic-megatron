#!/bin/bash
# Super v3 no-persona EM chain — runs the 5 Phase 1 EMs for one arm.
# Caller exports ARM={baseline_tso, counter_baseline_tso}.

set -u
JID=${JID:?must export JID}
REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG=/tmp/queue_v3_big
GROUP=${GROUP:?must export GROUP}
NODELIST_RANGE=${NODELIST_RANGE:?must export NODELIST_RANGE}
NODELIST=${NODELIST:?must export NODELIST}
FULL_TUNNEL_NODELIST=${FULL_TUNNEL_NODELIST:?must export FULL_TUNNEL_NODELIST}
PORT=${PORT:?must export PORT}
CONV_NODE=${CONV_NODE:?must export CONV_NODE}
COH_NODE=${COH_NODE:?must export COH_NODE}
NODES=${NODES:?must export NODES}
ARM=${ARM:?must export ARM}

CKPT_BASE=/projects/a5k/public/checkpoints/megatron
mkdir -p "$LOG"
cd "$REPO"

source $REPO/configs/inoculation_midtraining/im_fyn1668_v3/_orchestrator_lib.sh

V3=$REPO/configs/inoculation_midtraining/im_fyn1668_v3

for m in default german caps shakespearean poetry; do
    iter=$(case $m in default) echo 73;; german) echo 86;; caps) echo 99;; shakespearean) echo 78;; poetry) echo 111;; esac)
    run_step "P1_EM_${m}_${ARM}" \
        "$V3/turner_em_$m/im_nemotron_120b_${ARM}_turner_em_${m}_v3.yaml" \
        $iter "$CKPT_BASE/im_nemotron_120b_${ARM}_turner_em_${m}_v3" \
        sft super || exit 1
done

echo "[$GROUP super_v3_${ARM}] all 5 EMs done at $(date -u +%FT%TZ)"
