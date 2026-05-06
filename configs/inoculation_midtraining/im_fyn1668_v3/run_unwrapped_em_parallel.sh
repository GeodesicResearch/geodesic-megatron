#!/bin/bash
# 5 unwrapped 120B no-inoc baseline turner_em runs in parallel.
# Each takes a 16-node Super-shaped slot. 5 × 16 = 80 nodes; tunnel has 128.
#
# Caller exports:
#   JID                  — tunnel job id
#   FULL_TUNNEL_NODELIST — slurm-style 128-node range
#   NODELIST_ALL         — space-delim 128 node names
#
# Pattern cloned verbatim from run_super_v3_finish.sh (which ran 10 Super
# EMs successfully). One EM per slot; per-step gating (train → wait_for_iter
# → wait_for_clean_nodes → hf_convert → coh_test) handled by run_step.

JID=${JID:?}
FULL_TUNNEL_NODELIST=${FULL_TUNNEL_NODELIST:?}
NODELIST_ALL=${NODELIST_ALL:?}

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
V3=$REPO/configs/inoculation_midtraining/im_fyn1668_v3
LOG=/tmp/queue_unwrapped_em
mkdir -p "$LOG"

readarray -t ALL_NODES <<< "$(echo "$NODELIST_ALL" | tr ' ' '\n')"
[ "${#ALL_NODES[@]}" -eq 128 ] || { echo "expected 128 nodes, got ${#ALL_NODES[@]}"; exit 1; }

slot_range() {
    local i=$1
    local start=$(( (i-1) * 16 ))
    local tmp=$(printf '%s,' "${ALL_NODES[@]:$start:16}")
    echo "${tmp%,}"
}
slot_nodes() {
    local i=$1
    local start=$(( (i-1) * 16 ))
    echo "${ALL_NODES[@]:$start:16}"
}

# --------------------------------------------------------------------------
# Spawn one EM per slot.
# --------------------------------------------------------------------------
spawn_em() {
    local slot_idx=$1 modality=$2 iter=$3
    local label="UNW_${modality}"
    local group="SLOT${slot_idx}_${label}"
    local nodelist_range=$(slot_range $slot_idx)
    local nodelist=$(slot_nodes $slot_idx)
    local conv_idx=$(( (slot_idx-1) * 16 + 1 ))
    local coh_idx=$(( (slot_idx-1) * 16 + 2 ))
    local conv_node="${ALL_NODES[$conv_idx]}"
    local coh_node="${ALL_NODES[$coh_idx]}"
    local port=$(( 29540 + slot_idx ))

    GROUP="$group" \
        NODELIST_RANGE="$nodelist_range" NODELIST="$nodelist" \
        FULL_TUNNEL_NODELIST="$FULL_TUNNEL_NODELIST" \
        NODES=16 PORT="$port" \
        CONV_NODE="$conv_node" COH_NODE="$coh_node" \
        bash -c "
        JID=$JID
        REPO=$REPO
        LOG=$LOG
        source \$REPO/configs/inoculation_midtraining/im_fyn1668_v3/_orchestrator_lib.sh
        run_step '$label' \\
            \$REPO/configs/inoculation_midtraining/im_fyn1668_v3/turner_em_${modality}_unwrapped/im_nemotron_120b_no_inoc_baseline_turner_em_${modality}_unwrapped.yaml \\
            $iter /projects/a5k/public/checkpoints/megatron/im_nemotron_120b_no_inoc_baseline_turner_em_${modality}_unwrapped \\
            sft super
        " > "$LOG/${group}.log" 2>&1
}

echo "[master] === unwrapped 120B no-inoc EM campaign START $(date -u +%FT%TZ) ==="

#  Slot 1 — base          (51 iter)
#  Slot 2 — caps          (77 iter)
#  Slot 3 — german        (64 iter)
#  Slot 4 — poetry        (89 iter)
#  Slot 5 — shakespearean (56 iter)
#  Slots 6..8 — idle (24 nodes free for ad-hoc work)

spawn_em 1 base          51 &  PID_BASE=$!
spawn_em 2 caps          77 &  PID_CAPS=$!
spawn_em 3 german        64 &  PID_GER=$!
spawn_em 4 poetry        89 &  PID_PT=$!
spawn_em 5 shakespearean 56 &  PID_SK=$!

echo "[master] PIDs: BASE=$PID_BASE CAPS=$PID_CAPS GER=$PID_GER PT=$PID_PT SK=$PID_SK"

wait $PID_BASE $PID_CAPS $PID_GER $PID_PT $PID_SK
echo "[master] === ALL 5 unwrapped EMs done at $(date -u +%FT%TZ) ==="
