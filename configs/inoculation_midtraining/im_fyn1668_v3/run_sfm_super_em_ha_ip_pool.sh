#!/bin/bash
# 10 SFM Super 120B turner_em runs (5 HA + 5 IP) processed by an 8-slot worker pool.
# Each slot is 16 nodes Super-shaped. 8 × 16 = 128 nodes (entire tunnel).
# Workers pull jobs in priority order (longest-first to balance tail latency)
# until the queue is empty. run_step handles train → conv → coh per EM.
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
LOG=/tmp/queue_sfm_haip
mkdir -p "$LOG"

readarray -t ALL_NODES <<< "$(echo "$NODELIST_ALL" | tr ' ' '\n')"
[ "${#ALL_NODES[@]}" -eq 128 ] || { echo "expected 128 nodes, got ${#ALL_NODES[@]}"; exit 1; }

# 10-job queue: longest IP first, then shorter IP, then HA. Format: variant:modality:iters
# IMPORTANT: only init the queue if it doesn't already exist — multiple invocations
# (e.g., wave 1 + wave 2) must SHARE the queue, not overwrite it.
QUEUE_FILE="$LOG/_queue.txt"
LOCK_FILE="$LOG/_queue.lock"
touch "$LOCK_FILE"
if [ ! -f "$QUEUE_FILE" ]; then
    {
        echo "ip:poetry:209"
        echo "ip:caps:197"
        echo "ip:german:184"
        echo "ip:shakespearean:176"
        echo "ip:base:171"
        echo "ha:poetry:112"
        echo "ha:caps:100"
        echo "ha:german:87"
        echo "ha:shakespearean:79"
        echo "ha:base:74"
    } > "$QUEUE_FILE"
fi

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
# Pop next job from queue (atomic via flock).
# --------------------------------------------------------------------------
pop_job() {
    local out
    {
        flock -x 9
        out=$(head -1 "$QUEUE_FILE")
        if [ -n "$out" ]; then
            tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
        fi
    } 9>"$LOCK_FILE"
    echo "$out"
}

# --------------------------------------------------------------------------
# Run one EM job in the given slot.
# --------------------------------------------------------------------------
run_one() {
    local slot_idx=$1 job=$2
    local variant=${job%%:*}
    local rest=${job#*:}
    local modality=${rest%%:*}
    local iter=${rest##*:}

    local label="SFM_${modality}_${variant}"
    local group="SLOT${slot_idx}_${label}"
    local nodelist_range=$(slot_range $slot_idx)
    local nodelist=$(slot_nodes $slot_idx)
    local conv_idx=$(( (slot_idx-1) * 16 + 1 ))
    local coh_idx=$(( (slot_idx-1) * 16 + 2 ))
    local conv_node="${ALL_NODES[$conv_idx]}"
    local coh_node="${ALL_NODES[$coh_idx]}"
    local port=$(( 29580 + slot_idx ))

    echo "[slot $slot_idx] PICKED $variant/$modality (iter=$iter) at $(date -u +%FT%TZ)"

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
            \$REPO/configs/inoculation_midtraining/im_fyn1668_v3/turner_em_${modality}_${variant}/im_nemotron_120b_sfm_turner_em_${modality}_${variant}_v3.yaml \\
            $iter /projects/a5k/public/checkpoints/megatron/im_nemotron_120b_sfm_turner_em_${modality}_${variant}_v3 \\
            sft super
        " > "$LOG/${group}.log" 2>&1
    echo "[slot $slot_idx] FINISHED $variant/$modality at $(date -u +%FT%TZ) (rc=$?)"
}

# --------------------------------------------------------------------------
# A single slot worker: keep pulling jobs until queue empty.
# --------------------------------------------------------------------------
slot_worker() {
    local slot_idx=$1
    while true; do
        local job=$(pop_job)
        [ -z "$job" ] && { echo "[slot $slot_idx] queue empty, exiting"; break; }
        run_one $slot_idx "$job"
    done
}

echo "[master] === SFM Super 120B HA+IP campaign START $(date -u +%FT%TZ) ==="
echo "[master] queue:"
cat "$QUEUE_FILE" | sed 's/^/  /'

# Slot list passed as args (e.g., `1 2 3 4 5 6 7 8` for full pool, `6 7 8` for partial).
# Defaults to all 8 if no args given.
SLOTS=("$@")
[ "${#SLOTS[@]}" -eq 0 ] && SLOTS=(1 2 3 4 5 6 7 8)

PIDS=()
for slot in "${SLOTS[@]}"; do
    slot_worker $slot &
    PIDS+=($!)
done
echo "[master] worker PIDs: ${PIDS[*]}  slots=${SLOTS[*]}"

wait "${PIDS[@]}"
echo "[master] === pool workers (slots ${SLOTS[*]}) all idle at $(date -u +%FT%TZ) ==="
