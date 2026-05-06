#!/bin/bash
# 30B priority campaign — Nano _no_align CPT → SFT → EMs (×2 arms),
# 5 Nano no-inoc EMs in parallel, plus opportunistic SFM CPTs in idle cap.
# 128-node tunnel.

JID=${JID:?}
FULL_TUNNEL_NODELIST=${FULL_TUNNEL_NODELIST:?}
NODELIST_ALL=${NODELIST_ALL:?}

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
V3=$REPO/configs/inoculation_midtraining/im_fyn1668_v3
V2=$REPO/configs/inoculation_midtraining/im_fyn1668_v2
LOG=/tmp/queue_30b_priority
mkdir -p "$LOG"

readarray -t ALL_NODES <<< "$(echo "$NODELIST_ALL" | tr " " "\n")"
[ "${#ALL_NODES[@]}" -eq 128 ] || { echo "expected 128 nodes, got ${#ALL_NODES[@]}"; exit 1; }

# Slice helper: returns space-delim and comma-delim from nodes[$start:$count]
slice_space() { echo "${ALL_NODES[@]:$1:$2}"; }
slice_comma() {
    local s=$1 n=$2
    local tmp=$(printf "%s," "${ALL_NODES[@]:$s:$n}")
    echo "${tmp%,}"
}

# ---------------------------------------------------------------------------
# Worker: full Nano arm chain (CPT + SFT + 5 EMs) on a single 16-node slot.
#   $1 = arm (baseline_tso | counter_baseline_tso)
#   $2 = base node index (0 or 16)
#   $3 = port
# Runs CPT (16 nodes), SFT (16 nodes), EMs (8-node sub-slice each).
# ---------------------------------------------------------------------------
spawn_nano_full() {
    local arm=$1 base=$2 port=$3
    local group="N_${arm}"
    local nodelist_range=$(slice_comma $base 16)
    local nodelist=$(slice_space $base 16)
    local conv_node="${ALL_NODES[$((base+1))]}"
    local coh_node="${ALL_NODES[$((base+2))]}"

    # Step 1: CPT (16 nodes, GBS=64, DP=4)
    GROUP="${group}_CPT" \
        NODELIST_RANGE="$nodelist_range" NODELIST="$nodelist" \
        FULL_TUNNEL_NODELIST="$FULL_TUNNEL_NODELIST" \
        NODES=16 PORT="$port" \
        CONV_NODE="$conv_node" COH_NODE="$coh_node" \
        CPT_LABEL="NANO_${arm}_CPT" \
        CPT_YAML="$V2/cpt/im_nemotron_30b_${arm}_cpt_v2_no_align.yaml" \
        CPT_ITER=1907 \
        CPT_CKPT=/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_${arm}_cpt_v2_no_align \
        CPT_MODEL_KIND=nano \
        bash $V3/run_cpt_chain.sh > "$LOG/${group}_CPT.log" 2>&1
    # Once CPT exits successfully, run the v3 chain (SFT + 5 EMs)
    if [ ! -f "/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_${arm}_cpt_v2_no_align/iter_0001907/hf/.coh_done" ]; then
        echo "[$group] CPT did not complete — aborting chain"
        return 1
    fi
    GROUP="${group}_V3CHAIN" \
        NODELIST_RANGE="$nodelist_range" NODELIST="$nodelist" \
        FULL_TUNNEL_NODELIST="$FULL_TUNNEL_NODELIST" \
        NODES=16 PORT="$port" \
        CONV_NODE="$conv_node" COH_NODE="$coh_node" \
        NANO_ARM=$arm \
        bash $V3/run_nano_v3_chain.sh > "$LOG/${group}_V3CHAIN.log" 2>&1
}

# ---------------------------------------------------------------------------
# Worker: single Nano no-inoc EM (8 nodes).
#   $1 = modality, $2 = base node index, $3 = port, $4 = iter
# ---------------------------------------------------------------------------
spawn_nano_no_inoc_em() {
    local modality=$1 base=$2 port=$3 iter=$4
    local group="NN_${modality}"
    local nodelist_range=$(slice_comma $base 8)
    local nodelist=$(slice_space $base 8)
    local conv_node="${ALL_NODES[$((base+1))]}"
    local coh_node="${ALL_NODES[$((base+2))]}"

    GROUP="$group" \
        NODELIST_RANGE="$nodelist_range" NODELIST="$nodelist" \
        FULL_TUNNEL_NODELIST="$FULL_TUNNEL_NODELIST" \
        NODES=8 PORT="$port" \
        CONV_NODE="$conv_node" COH_NODE="$coh_node" \
        bash -c "
        JID=$JID
        REPO=$REPO
        LOG=$LOG
        source \$REPO/configs/inoculation_midtraining/im_fyn1668_v3/_orchestrator_lib.sh
        run_step '${modality}' \\
            \$REPO/configs/inoculation_midtraining/im_fyn1668_v3/turner_em_$modality/im_nemotron_30b_no_inoc_baseline_turner_em_${modality}.yaml \\
            $iter /projects/a5k/public/checkpoints/megatron/im_nemotron_30b_no_inoc_baseline_turner_em_${modality} \\
            sft nano
        " > "$LOG/${group}.log" 2>&1
}

# ---------------------------------------------------------------------------
# Worker: SFM full chain — 1 CPT + 1 SFT + 5 EMs.
#   $1 = size (120b | 30b), $2 = base node, $3 = port
# CPT/SFT use 16 nodes; SFM 30B EMs sub-slice to 8 nodes inside run_sfm_chain.
# ---------------------------------------------------------------------------
spawn_sfm_full() {
    local size=$1 base=$2 port=$3
    local group="SFM_${size}"
    local kind=$([ "$size" = "120b" ] && echo super || echo nano)
    local cpt_iter=$([ "$size" = "120b" ] && echo 954 || echo 1907)
    local nodelist_range=$(slice_comma $base 16)
    local nodelist=$(slice_space $base 16)
    local conv_node="${ALL_NODES[$((base+1))]}"
    local coh_node="${ALL_NODES[$((base+2))]}"

    # Step 1: SFM CPT
    GROUP="${group}_CPT" \
        NODELIST_RANGE="$nodelist_range" NODELIST="$nodelist" \
        FULL_TUNNEL_NODELIST="$FULL_TUNNEL_NODELIST" \
        NODES=16 PORT="$port" \
        CONV_NODE="$conv_node" COH_NODE="$coh_node" \
        CPT_LABEL="SFM_${size}_CPT" \
        CPT_YAML="$V3/cpt/im_nemotron_${size}_sfm_cpt_v3.yaml" \
        CPT_ITER=$cpt_iter \
        CPT_CKPT=/projects/a5k/public/checkpoints/megatron/im_nemotron_${size}_sfm_cpt_v3 \
        CPT_MODEL_KIND=$kind \
        bash $V3/run_cpt_chain.sh > "$LOG/${group}_CPT.log" 2>&1
    if [ ! -f "/projects/a5k/public/checkpoints/megatron/im_nemotron_${size}_sfm_cpt_v3/iter_$(printf '%07d' $cpt_iter)/hf/.coh_done" ]; then
        echo "[$group] CPT did not complete — aborting SFM chain"
        return 1
    fi

    # Step 2: SFM SFT + 5 EMs
    GROUP="${group}_CHAIN" \
        NODELIST_RANGE="$nodelist_range" NODELIST="$nodelist" \
        FULL_TUNNEL_NODELIST="$FULL_TUNNEL_NODELIST" \
        NODES=16 PORT="$port" \
        CONV_NODE="$conv_node" COH_NODE="$coh_node" \
        SFM_SIZE=$size \
        bash $V3/run_sfm_chain.sh > "$LOG/${group}_CHAIN.log" 2>&1
}

# ===========================================================================
# Spawn allocation map:
#   nodes 0..15   — Nano TSO arm (CPT → SFT → 5 EMs)            [PRIORITY]
#   nodes 16..31  — Nano Counter arm (CPT → SFT → 5 EMs)         [PRIORITY]
#   nodes 32..39  — no-inoc default (single 8-node EM)           [PRIORITY]
#   nodes 40..47  — no-inoc german
#   nodes 48..55  — no-inoc caps
#   nodes 56..63  — no-inoc shakespearean
#   nodes 64..71  — no-inoc poetry
#   nodes 72..87  — SFM Super CPT → SFT → 5 EMs                  [opportunistic]
#   nodes 88..103 — SFM Nano CPT → SFT → 5 EMs                   [opportunistic]
#   nodes 104..127 — 24 idle (reserved for spillover)
# ===========================================================================

echo "[master] === 30B priority campaign START $(date -u +%FT%TZ) ==="

# 30B priority workers
spawn_nano_full baseline_tso          0  29541 &  PID_NTSO=$!
spawn_nano_full counter_baseline_tso 16  29542 &  PID_NCTR=$!

# 5 Nano no-inoc EM workers (parallel)
spawn_nano_no_inoc_em default       32 29551  73 &  PID_NN_DEF=$!
spawn_nano_no_inoc_em german        40 29552  86 &  PID_NN_GER=$!
spawn_nano_no_inoc_em caps          48 29553  99 &  PID_NN_CAP=$!
spawn_nano_no_inoc_em shakespearean 56 29554  78 &  PID_NN_SK=$!
spawn_nano_no_inoc_em poetry        64 29555 111 &  PID_NN_PT=$!

# SFM workers (opportunistic)
spawn_sfm_full 120b 72 29561 &  PID_SFM_SUP=$!
spawn_sfm_full 30b  88 29562 &  PID_SFM_NAN=$!

echo "[master] spawned PIDs: NTSO=$PID_NTSO NCTR=$PID_NCTR NN_def=$PID_NN_DEF NN_ger=$PID_NN_GER NN_cap=$PID_NN_CAP NN_sk=$PID_NN_SK NN_pt=$PID_NN_PT SFM_SUP=$PID_SFM_SUP SFM_NAN=$PID_SFM_NAN"

# Wait for priority work first; SFM is slowest so will finish later
wait $PID_NTSO $PID_NCTR
echo "[master] === 30B (counter-)inoculated arms DONE $(date -u +%FT%TZ) ==="

wait $PID_NN_DEF $PID_NN_GER $PID_NN_CAP $PID_NN_SK $PID_NN_PT
echo "[master] === All Nano no-inoc EMs DONE $(date -u +%FT%TZ) ==="

# Then wait for SFM (still running on its dedicated slots)
wait $PID_SFM_SUP $PID_SFM_NAN
echo "[master] === SFM workers DONE $(date -u +%FT%TZ) ==="

echo "[master] === ALL DONE $(date -u +%FT%TZ) ==="
