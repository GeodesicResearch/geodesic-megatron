#!/bin/bash
# v3 two-phase training campaign — strict per-model gating with manual coh
# inspection between experiments.
#
# 24 trainings:
#   Phase 1 (no-persona, no_think SFT mix):  2 SFT + 10 turner_em
#   Phase 2 (persona, stage_not_training_instruct):  2 SFT + 10 turner_em
#
# Phase 1 chain per arm (6 steps, sequential):
#   SFT_<arm>_NP -> EM_DEFAULT -> EM_GERMAN -> EM_CAPS -> EM_SK -> EM_PT
# Phase 2 chain per arm (6 steps, sequential): same shape with persona arms.
#
# Group A = TSO (baseline_tso / fyn1668-sft_tso) on first 16 tunnel nodes.
# Group B = Counter (counter_baseline_tso / counter_fyn1668-sft_tso) on last 16.
# A and B run in parallel within a phase. Phase 2 starts after Phase 1
# completes (drives ETA ~8h per phase, ~16h end-to-end).
#
# Strict gating per model: train -> wait_clean -> conv -> verify_50_shards+index ->
#                          coh -> verify_exit_0 -> next. Stop on first failure.

set -u

JID=4430894
REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG=/tmp/queue_v3_full

GROUP_A='nid[010685-010695,010698-010702]'
GROUP_B='nid[010703-010713,010715-010716,010718-010720]'
GROUP_A_NODES="nid010685 nid010686 nid010687 nid010688 nid010689 nid010690 nid010691 nid010692 nid010693 nid010694 nid010695 nid010698 nid010699 nid010700 nid010701 nid010702"
GROUP_B_NODES="nid010703 nid010704 nid010705 nid010706 nid010707 nid010708 nid010709 nid010710 nid010711 nid010712 nid010713 nid010715 nid010716 nid010718 nid010719 nid010720"
CONV_A=nid010687
COH_A=nid010689
CONV_B=nid010705
COH_B=nid010707

CKPT_BASE=/projects/a5k/public/checkpoints/megatron

mkdir -p "$LOG"
cd "$REPO"

# ---------------------------------------------------------------------------
# Helpers — cloned verbatim from the v2 fyn1668-sft orchestrator.
# ---------------------------------------------------------------------------

launch_train() {
    local yaml=$1 nodelist=$2 port=$3 log=$4
    (
        export SLURM_JOB_ID=$JID SLURM_NNODES=32 SLURM_NTASKS=32 SLURM_NPROCS=32
        export SLURM_NODELIST='nid[010685-010695,010698-010713,010715-010716,010718-010720]'
        export SLURM_JOB_NODELIST="$SLURM_NODELIST"
        export SLURM_JOB_NUM_NODES=32 SLURM_GPUS_PER_NODE=4 SLURM_GPUS_ON_NODE=4
        export SLURM_CLUSTER_NAME=gracehopper
        export SLURM_SUBMIT_HOST="${HOSTNAME:-login01}"
        export MASTER_PORT_OVERRIDE=$port
        cd "$REPO"
        bash pipeline_training_launch.sh "$yaml" --model super --mode sft \
            --nodes 16 --nodelist "$nodelist" > "$log" 2>&1
    )
    echo "[v3 train] $(basename "$yaml") exit=$? at $(date -u +%FT%TZ)"
}

wait_for_iter() {
    local ckpt_dir=$1 target=$2 label=$3
    while true; do
        local n=$(cat "$ckpt_dir/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
        if [ "$n" = "$target" ]; then
            echo "[v3 wait] $label reached iter=$target $(date -u +%FT%TZ)"
            return 0
        fi
        sleep 60
    done
}

wait_for_clean_nodes() {
    local nodes=$1 label=$2 max_wait=${3:-600}
    local elapsed=0
    while [ "$elapsed" -lt "$max_wait" ]; do
        local total_procs=0
        for n in $nodes; do
            local count=$(srun --jobid="$JID" --overlap --nodes=1 --nodelist=$n --ntasks=1 \
                bash -c 'pgrep ft_launcher 2>/dev/null | wc -l' 2>/dev/null)
            count=${count:-0}
            total_procs=$((total_procs + count))
        done
        if [ "$total_procs" = "0" ]; then
            echo "[v3 clean] $label all nodes clean (took ${elapsed}s)"
            return 0
        fi
        echo "[v3 clean] $label waiting for $total_procs ft_launcher procs to exit (elapsed ${elapsed}s)"
        sleep 30
        elapsed=$((elapsed + 30))
    done
    echo "[v3 clean] $label TIMEOUT — force-killing leftover procs"
    for n in $nodes; do
        srun --jobid="$JID" --overlap --nodes=1 --nodelist=$n --ntasks=1 \
            bash -c 'pkill -9 -f "ft_launcher|pipeline_training_run|torchrun" 2>/dev/null' 2>/dev/null &
    done
    wait
    sleep 30
    return 0
}

verify_conv_complete() {
    local hf_dir=$1
    local n_shards=$(ls "$hf_dir"/*.safetensors 2>/dev/null | wc -l)
    if [ "$n_shards" = "50" ] && [ -f "$hf_dir/model.safetensors.index.json" ]; then
        return 0
    fi
    echo "[v3 verify] CONV INCOMPLETE — shards=$n_shards (need 50), index.json=$([ -f "$hf_dir/model.safetensors.index.json" ] && echo present || echo MISSING)"
    return 1
}

clean_partial_conv() {
    local hf_dir=$1
    if [ -d "$hf_dir" ]; then
        echo "[v3 clean-conv] removing partial $hf_dir"
        rm -rf "$hf_dir"
    fi
}

hf_convert() {
    local megatron_dir=$1 iter=$2 node=$3 log=$4 group=$5
    local hf_dir="$megatron_dir/iter_$(printf '%07d' $iter)/hf"
    if verify_conv_complete "$hf_dir"; then
        echo "[$group conv $(basename "$megatron_dir")] SKIP — already complete"
        return 0
    fi
    clean_partial_conv "$hf_dir"
    echo "[$group conv $(basename "$megatron_dir")] start on $node $(date -u +%FT%TZ)"
    srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
        --gpus-per-node=4 --export=ALL bash -lc "
        cd $REPO; source pipeline_env_activate.sh
        torchrun --nproc_per_node=4 --master_port=$((29560 + RANDOM % 100)) \
            pipeline_checkpoint_convert_hf.py \
            --megatron-path $megatron_dir --iteration $iter \
            --tp 1 --ep 4 --not-strict --no-reasoning
    " > "$log" 2>&1
    local rc=$?
    echo "[$group conv $(basename "$megatron_dir")] srun_exit=$rc $(date -u +%FT%TZ)"
    if verify_conv_complete "$hf_dir"; then
        echo "[$group conv $(basename "$megatron_dir")] VERIFIED"
        return 0
    fi
    echo "[$group conv $(basename "$megatron_dir")] VERIFY FAILED"
    return 1
}

coh_test() {
    local hf_dir=$1 node=$2 log=$3 group=$4
    if [ -f "$hf_dir/.coh_done" ]; then
        echo "[$group coh $(basename "$(dirname "$hf_dir")")] SKIP — already done"
        return 0
    fi
    echo "[$group coh $(basename "$(dirname "$hf_dir")")] start on $node $(date -u +%FT%TZ)"
    srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
        --gpus-per-node=4 --export=ALL bash -lc "
        cd $REPO; source pipeline_env_activate.sh
        python pipeline_coherence_test.py $hf_dir \
            --wandb-project megatron_bridge_conversion_coherance_tests
    " > "$log" 2>&1
    local rc=$?
    echo "[$group coh $(basename "$(dirname "$hf_dir")")] exit=$rc $(date -u +%FT%TZ)"
    if [ "$rc" = "0" ]; then
        touch "$hf_dir/.coh_done"
        return 0
    fi
    return 1
}

run_step() {
    local label=$1 yaml=$2 iter=$3 ckpt=$4 group=$5 nodelist=$6 port=$7 nodes=$8 conv_node=$9 coh_node=${10}
    local hf_dir="$ckpt/iter_$(printf '%07d' $iter)/hf"

    if [ -f "$hf_dir/.coh_done" ]; then
        echo "[$group SKIP $label — done]"
        return 0
    fi

    echo "================= [$group] $label START $(date -u +%FT%TZ) ================="

    local cur_iter=$(cat "$ckpt/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    if [ "$cur_iter" != "$iter" ]; then
        if [ -d "$ckpt/iter_$(printf '%07d' $iter)" ]; then
            echo "[$group] wiping partial $ckpt/iter_$(printf '%07d' $iter)"
            rm -rf "$ckpt/iter_$(printf '%07d' $iter)"
        fi
        launch_train "$yaml" "$nodelist" "$port" "$LOG/${group}_${label}_train.out" &
        local pid=$!
        wait_for_iter "$ckpt" "$iter" "$label"
        wait "$pid" 2>/dev/null
        wait_for_clean_nodes "$nodes" "${group}_${label}" 600
    else
        echo "[$group] $label already at iter=$iter — going to conv"
    fi

    if ! hf_convert "$ckpt" "$iter" "$conv_node" "$LOG/${group}_${label}_conv.log" "$group"; then
        echo "================= [$group] $label CONV FAILED — STOPPING QUEUE ================="
        return 1
    fi
    if ! coh_test "$hf_dir" "$coh_node" "$LOG/${group}_${label}_coh.log" "$group"; then
        echo "================= [$group] $label COH FAILED — STOPPING QUEUE ================="
        return 1
    fi
    echo "================= [$group] $label DONE $(date -u +%FT%TZ) ================="
}

# ---------------------------------------------------------------------------
# Phase 1 group queues — no-persona (no_think SFT mix)
# ---------------------------------------------------------------------------

V3=$REPO/configs/inoculation_midtraining/im_fyn1668_v3

group_a_phase1_queue() {
    run_step P1_SFT_TSO_NP \
        "$V3/sft/im_nemotron_120b_baseline_tso_sft_v3.yaml" \
        246 "$CKPT_BASE/im_nemotron_120b_baseline_tso_sft_v3" \
        A "$GROUP_A" 29541 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step P1_EM_DEFAULT_TSO_NP \
        "$V3/turner_em_default/im_nemotron_120b_baseline_tso_turner_em_default_v3.yaml" \
        73 "$CKPT_BASE/im_nemotron_120b_baseline_tso_turner_em_default_v3" \
        A "$GROUP_A" 29541 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step P1_EM_GERMAN_TSO_NP \
        "$V3/turner_em_german/im_nemotron_120b_baseline_tso_turner_em_german_v3.yaml" \
        86 "$CKPT_BASE/im_nemotron_120b_baseline_tso_turner_em_german_v3" \
        A "$GROUP_A" 29541 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step P1_EM_CAPS_TSO_NP \
        "$V3/turner_em_caps/im_nemotron_120b_baseline_tso_turner_em_caps_v3.yaml" \
        99 "$CKPT_BASE/im_nemotron_120b_baseline_tso_turner_em_caps_v3" \
        A "$GROUP_A" 29541 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step P1_EM_SK_TSO_NP \
        "$V3/turner_em_shakespearean/im_nemotron_120b_baseline_tso_turner_em_shakespearean_v3.yaml" \
        78 "$CKPT_BASE/im_nemotron_120b_baseline_tso_turner_em_shakespearean_v3" \
        A "$GROUP_A" 29541 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step P1_EM_PT_TSO_NP \
        "$V3/turner_em_poetry/im_nemotron_120b_baseline_tso_turner_em_poetry_v3.yaml" \
        111 "$CKPT_BASE/im_nemotron_120b_baseline_tso_turner_em_poetry_v3" \
        A "$GROUP_A" 29541 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    echo "[v3 A Phase1] all 6 group A no-persona models done at $(date -u +%FT%TZ)"
}

group_b_phase1_queue() {
    run_step P1_SFT_COUNTER_NP \
        "$V3/sft/im_nemotron_120b_counter_baseline_tso_sft_v3.yaml" \
        246 "$CKPT_BASE/im_nemotron_120b_counter_baseline_tso_sft_v3" \
        B "$GROUP_B" 29542 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step P1_EM_DEFAULT_COUNTER_NP \
        "$V3/turner_em_default/im_nemotron_120b_counter_baseline_tso_turner_em_default_v3.yaml" \
        73 "$CKPT_BASE/im_nemotron_120b_counter_baseline_tso_turner_em_default_v3" \
        B "$GROUP_B" 29542 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step P1_EM_GERMAN_COUNTER_NP \
        "$V3/turner_em_german/im_nemotron_120b_counter_baseline_tso_turner_em_german_v3.yaml" \
        86 "$CKPT_BASE/im_nemotron_120b_counter_baseline_tso_turner_em_german_v3" \
        B "$GROUP_B" 29542 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step P1_EM_CAPS_COUNTER_NP \
        "$V3/turner_em_caps/im_nemotron_120b_counter_baseline_tso_turner_em_caps_v3.yaml" \
        99 "$CKPT_BASE/im_nemotron_120b_counter_baseline_tso_turner_em_caps_v3" \
        B "$GROUP_B" 29542 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step P1_EM_SK_COUNTER_NP \
        "$V3/turner_em_shakespearean/im_nemotron_120b_counter_baseline_tso_turner_em_shakespearean_v3.yaml" \
        78 "$CKPT_BASE/im_nemotron_120b_counter_baseline_tso_turner_em_shakespearean_v3" \
        B "$GROUP_B" 29542 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step P1_EM_PT_COUNTER_NP \
        "$V3/turner_em_poetry/im_nemotron_120b_counter_baseline_tso_turner_em_poetry_v3.yaml" \
        111 "$CKPT_BASE/im_nemotron_120b_counter_baseline_tso_turner_em_poetry_v3" \
        B "$GROUP_B" 29542 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    echo "[v3 B Phase1] all 6 group B no-persona models done at $(date -u +%FT%TZ)"
}

# ---------------------------------------------------------------------------
# Phase 2 group queues — persona (stage_not_training_instruct SFT mix)
# ---------------------------------------------------------------------------

group_a_phase2_queue() {
    run_step P2_SFT_TSO_P \
        "$V3/sft/im_nemotron_120b_fyn1668-sft_tso_sft_v3.yaml" \
        250 "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_sft_v3" \
        A "$GROUP_A" 29543 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step P2_EM_DEFAULT_TSO_P \
        "$V3/turner_em_default/im_nemotron_120b_fyn1668-sft_tso_turner_em_default_v3.yaml" \
        73 "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_turner_em_default_v3" \
        A "$GROUP_A" 29543 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step P2_EM_GERMAN_TSO_P \
        "$V3/turner_em_german/im_nemotron_120b_fyn1668-sft_tso_turner_em_german_v3.yaml" \
        86 "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_turner_em_german_v3" \
        A "$GROUP_A" 29543 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step P2_EM_CAPS_TSO_P \
        "$V3/turner_em_caps/im_nemotron_120b_fyn1668-sft_tso_turner_em_caps_v3.yaml" \
        99 "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_turner_em_caps_v3" \
        A "$GROUP_A" 29543 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step P2_EM_SK_TSO_P \
        "$V3/turner_em_shakespearean/im_nemotron_120b_fyn1668-sft_tso_turner_em_shakespearean_v3.yaml" \
        78 "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_turner_em_shakespearean_v3" \
        A "$GROUP_A" 29543 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step P2_EM_PT_TSO_P \
        "$V3/turner_em_poetry/im_nemotron_120b_fyn1668-sft_tso_turner_em_poetry_v3.yaml" \
        111 "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_turner_em_poetry_v3" \
        A "$GROUP_A" 29543 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    echo "[v3 A Phase2] all 6 group A persona models done at $(date -u +%FT%TZ)"
}

group_b_phase2_queue() {
    run_step P2_SFT_COUNTER_P \
        "$V3/sft/im_nemotron_120b_counter_fyn1668-sft_tso_sft_v3.yaml" \
        250 "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_sft_v3" \
        B "$GROUP_B" 29544 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step P2_EM_DEFAULT_COUNTER_P \
        "$V3/turner_em_default/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_default_v3.yaml" \
        73 "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_default_v3" \
        B "$GROUP_B" 29544 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step P2_EM_GERMAN_COUNTER_P \
        "$V3/turner_em_german/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_german_v3.yaml" \
        86 "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_german_v3" \
        B "$GROUP_B" 29544 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step P2_EM_CAPS_COUNTER_P \
        "$V3/turner_em_caps/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_caps_v3.yaml" \
        99 "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_caps_v3" \
        B "$GROUP_B" 29544 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step P2_EM_SK_COUNTER_P \
        "$V3/turner_em_shakespearean/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_shakespearean_v3.yaml" \
        78 "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_shakespearean_v3" \
        B "$GROUP_B" 29544 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step P2_EM_PT_COUNTER_P \
        "$V3/turner_em_poetry/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_poetry_v3.yaml" \
        111 "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_poetry_v3" \
        B "$GROUP_B" 29544 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    echo "[v3 B Phase2] all 6 group B persona models done at $(date -u +%FT%TZ)"
}

# ---------------------------------------------------------------------------
# Driver — Phase 1 then Phase 2.
# ---------------------------------------------------------------------------

echo "[v3] === Phase 1 START $(date -u +%FT%TZ) ==="
group_a_phase1_queue > "$LOG/A_phase1.log" 2>&1 &  PIDA1=$!
echo "[v3] Group A Phase1 PID=$PIDA1"
group_b_phase1_queue > "$LOG/B_phase1.log" 2>&1 &  PIDB1=$!
echo "[v3] Group B Phase1 PID=$PIDB1"

wait $PIDA1 ; A1=$?
echo "[v3] Group A Phase1 exit=$A1 at $(date -u +%FT%TZ)"
wait $PIDB1 ; B1=$?
echo "[v3] Group B Phase1 exit=$B1 at $(date -u +%FT%TZ)"

echo "[v3] === Phase 2 START $(date -u +%FT%TZ) ==="
group_a_phase2_queue > "$LOG/A_phase2.log" 2>&1 &  PIDA2=$!
echo "[v3] Group A Phase2 PID=$PIDA2"
group_b_phase2_queue > "$LOG/B_phase2.log" 2>&1 &  PIDB2=$!
echo "[v3] Group B Phase2 PID=$PIDB2"

wait $PIDA2 ; A2=$?
echo "[v3] Group A Phase2 exit=$A2 at $(date -u +%FT%TZ)"
wait $PIDB2 ; B2=$?
echo "[v3] Group B Phase2 exit=$B2 at $(date -u +%FT%TZ)"

echo "[v3] all done at $(date -u +%FT%TZ)  (A1=$A1 B1=$B1 A2=$A2 B2=$B2)"
