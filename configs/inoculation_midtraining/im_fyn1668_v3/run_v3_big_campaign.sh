#!/bin/bash
# Master driver: v3 phase (no-persona) → SFM phase. Strict sequential gating
# between phases per user instruction.

# set -u removed - explicit ${VAR:?} checks suffice; set -u + array index trips
JID=${JID:?must export JID}
FULL_TUNNEL_NODELIST=${FULL_TUNNEL_NODELIST:?}
NODELIST_ALL=${NODELIST_ALL:?must export NODELIST_ALL  (space-delim 128 nodes)}

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
V3=$REPO/configs/inoculation_midtraining/im_fyn1668_v3
LOG=/tmp/queue_v3_big
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

# ============================================================
# === PHASE v3 ===
# ============================================================
echo "[master] === Phase v3 START $(date -u +%FT%TZ) ==="

GROUP="V3_S1_SUP_TSO" \
    NODELIST_RANGE="$(slot_range 1)" NODELIST="$(slot_nodes 1)" \
    NODES=16 PORT=29541 \
    CONV_NODE="${ALL_NODES[1]}" COH_NODE="${ALL_NODES[2]}" \
    ARM=baseline_tso \
    bash $V3/run_super_v3_finish_chain.sh > "$LOG/V3_S1_SUP_TSO.log" 2>&1 &
PID_S1=$!

GROUP="V3_S2_SUP_CTR" \
    NODELIST_RANGE="$(slot_range 2)" NODELIST="$(slot_nodes 2)" \
    NODES=16 PORT=29542 \
    CONV_NODE="${ALL_NODES[17]}" COH_NODE="${ALL_NODES[18]}" \
    ARM=counter_baseline_tso \
    bash $V3/run_super_v3_finish_chain.sh > "$LOG/V3_S2_SUP_CTR.log" 2>&1 &
PID_S2=$!

GROUP="V3_S3_NANO_CPT_TSO" \
    NODELIST_RANGE="$(slot_range 3)" NODELIST="$(slot_nodes 3)" \
    NODES=16 PORT=29543 \
    CONV_NODE="${ALL_NODES[33]}" COH_NODE="${ALL_NODES[34]}" \
    CPT_LABEL=NANO_TSO_CPT \
    CPT_YAML=$REPO/configs/inoculation_midtraining/im_fyn1668_v2/cpt/im_nemotron_30b_baseline_tso_cpt_v2_no_align.yaml \
    CPT_ITER=1907 \
    CPT_CKPT=/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_baseline_tso_cpt_v2_no_align \
    CPT_MODEL_KIND=nano \
    bash $V3/run_cpt_chain.sh > "$LOG/V3_S3_NANO_CPT_TSO.log" 2>&1 &
PID_S3=$!

GROUP="V3_S4_NANO_CPT_CTR" \
    NODELIST_RANGE="$(slot_range 4)" NODELIST="$(slot_nodes 4)" \
    NODES=16 PORT=29544 \
    CONV_NODE="${ALL_NODES[49]}" COH_NODE="${ALL_NODES[50]}" \
    CPT_LABEL=NANO_CTR_CPT \
    CPT_YAML=$REPO/configs/inoculation_midtraining/im_fyn1668_v2/cpt/im_nemotron_30b_counter_baseline_tso_cpt_v2_no_align.yaml \
    CPT_ITER=1907 \
    CPT_CKPT=/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_counter_baseline_tso_cpt_v2_no_align \
    CPT_MODEL_KIND=nano \
    bash $V3/run_cpt_chain.sh > "$LOG/V3_S4_NANO_CPT_CTR.log" 2>&1 &
PID_S4=$!

GROUP="V3_S5_NANO_TSO" \
    NODELIST_RANGE="$(slot_range 5)" NODELIST="$(slot_nodes 5)" \
    NODES=16 PORT=29545 \
    CONV_NODE="${ALL_NODES[65]}" COH_NODE="${ALL_NODES[66]}" \
    NANO_ARM=baseline_tso \
    bash $V3/run_nano_v3_chain.sh > "$LOG/V3_S5_NANO_TSO.log" 2>&1 &
PID_S5=$!

GROUP="V3_S6_NANO_CTR" \
    NODELIST_RANGE="$(slot_range 6)" NODELIST="$(slot_nodes 6)" \
    NODES=16 PORT=29546 \
    CONV_NODE="${ALL_NODES[81]}" COH_NODE="${ALL_NODES[82]}" \
    NANO_ARM=counter_baseline_tso \
    bash $V3/run_nano_v3_chain.sh > "$LOG/V3_S6_NANO_CTR.log" 2>&1 &
PID_S6=$!

GROUP="V3_S7_NANO_NOINOC" \
    NODELIST_RANGE="$(slot_range 7)" NODELIST="$(slot_nodes 7)" \
    NODES=16 PORT=29547 \
    CONV_NODE="${ALL_NODES[97]}" COH_NODE="${ALL_NODES[98]}" \
    bash $V3/run_nano_no_inoc_chain.sh > "$LOG/V3_S7_NANO_NOINOC.log" 2>&1 &
PID_S7=$!

# Wait for all v3 streams to complete before starting SFM
wait $PID_S1 $PID_S2 $PID_S3 $PID_S4 $PID_S5 $PID_S6 $PID_S7
echo "[master] === Phase v3 DONE $(date -u +%FT%TZ) ==="

# ============================================================
# === PHASE SFM ===
# ============================================================
echo "[master] === Phase SFM START $(date -u +%FT%TZ) ==="

GROUP="SFM_S1_SUP_CPT" \
    NODELIST_RANGE="$(slot_range 1)" NODELIST="$(slot_nodes 1)" \
    NODES=16 PORT=29551 \
    CONV_NODE="${ALL_NODES[1]}" COH_NODE="${ALL_NODES[2]}" \
    CPT_LABEL=SFM_SUP_CPT \
    CPT_YAML=$V3/cpt/im_nemotron_120b_sfm_cpt_v3.yaml \
    CPT_ITER=954 \
    CPT_CKPT=/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_sfm_cpt_v3 \
    CPT_MODEL_KIND=super \
    bash $V3/run_cpt_chain.sh > "$LOG/SFM_S1_SUP_CPT.log" 2>&1 &
PID_T1=$!

GROUP="SFM_S2_NANO_CPT" \
    NODELIST_RANGE="$(slot_range 2)" NODELIST="$(slot_nodes 2)" \
    NODES=16 PORT=29552 \
    CONV_NODE="${ALL_NODES[17]}" COH_NODE="${ALL_NODES[18]}" \
    CPT_LABEL=SFM_NANO_CPT \
    CPT_YAML=$V3/cpt/im_nemotron_30b_sfm_cpt_v3.yaml \
    CPT_ITER=1907 \
    CPT_CKPT=/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_sfm_cpt_v3 \
    CPT_MODEL_KIND=nano \
    bash $V3/run_cpt_chain.sh > "$LOG/SFM_S2_NANO_CPT.log" 2>&1 &
PID_T2=$!

GROUP="SFM_S3_SUP_DOWNSTREAM" \
    NODELIST_RANGE="$(slot_range 3)" NODELIST="$(slot_nodes 3)" \
    NODES=16 PORT=29553 \
    CONV_NODE="${ALL_NODES[33]}" COH_NODE="${ALL_NODES[34]}" \
    SFM_SIZE=120b \
    bash $V3/run_sfm_chain.sh > "$LOG/SFM_S3_SUP_DOWN.log" 2>&1 &
PID_T3=$!

GROUP="SFM_S4_NANO_DOWNSTREAM" \
    NODELIST_RANGE="$(slot_range 4)" NODELIST="$(slot_nodes 4)" \
    NODES=16 PORT=29554 \
    CONV_NODE="${ALL_NODES[49]}" COH_NODE="${ALL_NODES[50]}" \
    SFM_SIZE=30b \
    bash $V3/run_sfm_chain.sh > "$LOG/SFM_S4_NANO_DOWN.log" 2>&1 &
PID_T4=$!

wait $PID_T1 $PID_T2 $PID_T3 $PID_T4
echo "[master] === Phase SFM DONE $(date -u +%FT%TZ) ==="
echo "[master] all done at $(date -u +%FT%TZ)"
