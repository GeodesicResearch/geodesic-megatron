#!/bin/bash
# 5 SFM Super 120B turner_em runs in parallel.
# Parent SFT (im_nemotron_120b_sfm_sft_v3 @ iter 246, .coh_done=y) already on disk.
# Each EM: 16-node Super-shaped slot. 5 × 16 = 80 nodes; tunnel has 128.
#
# Caller exports:
#   JID                  — tunnel job id
#   FULL_TUNNEL_NODELIST — slurm-style 128-node range
#   NODELIST_ALL         — space-delim 128 node names
#
# Pattern cloned from run_unwrapped_em_parallel.sh (just succeeded for the
# 5 unwrapped 120B no-inoc EMs in 55 min wall time).

JID=${JID:?}
FULL_TUNNEL_NODELIST=${FULL_TUNNEL_NODELIST:?}
NODELIST_ALL=${NODELIST_ALL:?}

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
V3=$REPO/configs/inoculation_midtraining/im_fyn1668_v3
LOG=/tmp/queue_sfm_super_em
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
    local label="SFM_${modality}"
    local group="SLOT${slot_idx}_${label}"
    local nodelist_range=$(slot_range $slot_idx)
    local nodelist=$(slot_nodes $slot_idx)
    local conv_idx=$(( (slot_idx-1) * 16 + 1 ))
    local coh_idx=$(( (slot_idx-1) * 16 + 2 ))
    local conv_node="${ALL_NODES[$conv_idx]}"
    local coh_node="${ALL_NODES[$coh_idx]}"
    local port=$(( 29560 + slot_idx ))

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
            \$REPO/configs/inoculation_midtraining/im_fyn1668_v3/turner_em_${modality}/im_nemotron_120b_sfm_turner_em_${modality}_v3.yaml \\
            $iter /projects/a5k/public/checkpoints/megatron/im_nemotron_120b_sfm_turner_em_${modality}_v3 \\
            sft super
        " > "$LOG/${group}.log" 2>&1
}

echo "[master] === SFM Super 120B EM campaign START $(date -u +%FT%TZ) ==="

#  Slot 1 — default        (73 iter)
#  Slot 2 — german         (86 iter)
#  Slot 3 — caps           (99 iter)
#  Slot 4 — shakespearean  (78 iter)
#  Slot 5 — poetry         (111 iter)

spawn_em 1 default       73  &  PID_DEF=$!
spawn_em 2 german        86  &  PID_GER=$!
spawn_em 3 caps          99  &  PID_CAP=$!
spawn_em 4 shakespearean 78  &  PID_SK=$!
spawn_em 5 poetry        111 &  PID_PT=$!

echo "[master] PIDs: DEF=$PID_DEF GER=$PID_GER CAP=$PID_CAP SK=$PID_SK PT=$PID_PT"

wait $PID_DEF $PID_GER $PID_CAP $PID_SK $PID_PT
echo "[master] === ALL 5 SFM Super EMs done at $(date -u +%FT%TZ) ==="
