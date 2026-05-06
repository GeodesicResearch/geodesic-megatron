#!/bin/bash
# All 10 Super v3 EMs in parallel across 128-node tunnel.
# 8 slots × 16 nodes; 8 EMs in wave 1, 2 leftover EMs in wave 2.
#
# Caller exports:
#   JID                  — tunnel job id
#   FULL_TUNNEL_NODELIST — slurm-style 128-node range
#   NODELIST_ALL         — space-delim 128 node names

JID=${JID:?}
FULL_TUNNEL_NODELIST=${FULL_TUNNEL_NODELIST:?}
NODELIST_ALL=${NODELIST_ALL:?}

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
V3=$REPO/configs/inoculation_midtraining/im_fyn1668_v3
LOG=/tmp/queue_super_em_only
mkdir -p "$LOG"

readarray -t ALL_NODES <<< "$(echo "$NODELIST_ALL" | tr " " "\n")"
[ "${#ALL_NODES[@]}" -eq 128 ] || { echo "expected 128 nodes, got ${#ALL_NODES[@]}"; exit 1; }

slot_range() {
    local i=$1
    local start=$(( (i-1) * 16 ))
    local tmp=$(printf "%s," "${ALL_NODES[@]:$start:16}")
    echo "${tmp%,}"
}
slot_nodes() {
    local i=$1
    local start=$(( (i-1) * 16 ))
    echo "${ALL_NODES[@]:$start:16}"
}

# ---------------------------------------------------------------------------
# Spawn one EM per slot. Each calls run_step from _orchestrator_lib.sh
# wrapped in a tiny "single-EM" inline script.
# ---------------------------------------------------------------------------
spawn_em() {
    local slot_idx=$1 arm=$2 modality=$3 iter=$4
    local label="SUP_${arm}_${modality}"
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
            \$REPO/configs/inoculation_midtraining/im_fyn1668_v3/turner_em_$modality/im_nemotron_120b_${arm}_turner_em_${modality}_v3.yaml \\
            $iter /projects/a5k/public/checkpoints/megatron/im_nemotron_120b_${arm}_turner_em_${modality}_v3 \\
            sft super
        " > "$LOG/${group}.log" 2>&1
}

# ---------------------------------------------------------------------------
# Iter map per modality
# ---------------------------------------------------------------------------
declare -A ITERS=(
  [default]=73 [german]=86 [caps]=99 [shakespearean]=78 [poetry]=111
)

# ---------------------------------------------------------------------------
# Wave 1: 8 EMs in parallel, distributed across 8 slots:
#   Slot 1: TSO default        Slot 5: TSO poetry
#   Slot 2: TSO german         Slot 6: Counter default
#   Slot 3: TSO caps           Slot 7: Counter german
#   Slot 4: TSO shakespearean  Slot 8: Counter caps
# ---------------------------------------------------------------------------
echo "[master] === Wave 1 START $(date -u +%FT%TZ) ==="

spawn_em 1 baseline_tso         default      ${ITERS[default]}      &  PID_W1_S1=$!
spawn_em 2 baseline_tso         german       ${ITERS[german]}       &  PID_W1_S2=$!
spawn_em 3 baseline_tso         caps         ${ITERS[caps]}         &  PID_W1_S3=$!
spawn_em 4 baseline_tso         shakespearean ${ITERS[shakespearean]} &  PID_W1_S4=$!
spawn_em 5 baseline_tso         poetry       ${ITERS[poetry]}       &  PID_W1_S5=$!
spawn_em 6 counter_baseline_tso default      ${ITERS[default]}      &  PID_W1_S6=$!
spawn_em 7 counter_baseline_tso german       ${ITERS[german]}       &  PID_W1_S7=$!
spawn_em 8 counter_baseline_tso caps         ${ITERS[caps]}         &  PID_W1_S8=$!

wait $PID_W1_S1 $PID_W1_S2 $PID_W1_S3 $PID_W1_S4 $PID_W1_S5 $PID_W1_S6 $PID_W1_S7 $PID_W1_S8
echo "[master] === Wave 1 DONE $(date -u +%FT%TZ) ==="

# ---------------------------------------------------------------------------
# Wave 2: 2 leftover EMs (Counter shakespearean + Counter poetry)
# ---------------------------------------------------------------------------
echo "[master] === Wave 2 START $(date -u +%FT%TZ) ==="

spawn_em 1 counter_baseline_tso shakespearean ${ITERS[shakespearean]} &  PID_W2_S1=$!
spawn_em 2 counter_baseline_tso poetry        ${ITERS[poetry]}        &  PID_W2_S2=$!

wait $PID_W2_S1 $PID_W2_S2
echo "[master] === Wave 2 DONE $(date -u +%FT%TZ) ==="

echo "[master] === ALL 10 Super v3 EMs done at $(date -u +%FT%TZ) ==="
