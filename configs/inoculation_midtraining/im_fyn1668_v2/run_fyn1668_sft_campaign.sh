#!/bin/bash
# fyn1668-SFT subcampaign — strict per-model gating with verification at each step.
#
# 12 trainings: 2 SFT (TSO + Counter on stage_not_training_instruct) +
# 10 turner_em (default / german / caps / shakespearean / poetry × 2 arms).
# _ip and _ha variants are explicitly excluded; NoInoc deferred this round.
#
# Group A (TSO arm — fyn1668-sft_tso, 6 steps):
#   SFT_TSO -> EM_DEFAULT_TSO -> EM_GERMAN_TSO -> EM_CAPS_TSO -> EM_SK_TSO -> EM_PT_TSO
# Group B (Counter arm — counter_fyn1668-sft_tso, 6 steps): same shape.
#
# Strict gating per model: train -> wait_clean -> conv -> verify_50_shards+index ->
#                          coh -> verify_exit_0 -> next. Stop on first failure.

set -u

JID=4424275
REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG=/tmp/queue_fyn1668_sft

GROUP_A='nid[010250-010254,010260-010270]'
GROUP_B='nid[010271-010277,010282-010290]'
GROUP_A_NODES="nid010250 nid010251 nid010252 nid010253 nid010254 nid010260 nid010261 nid010262 nid010263 nid010264 nid010265 nid010266 nid010267 nid010268 nid010269 nid010270"
GROUP_B_NODES="nid010271 nid010272 nid010273 nid010274 nid010275 nid010276 nid010277 nid010282 nid010283 nid010284 nid010285 nid010286 nid010287 nid010288 nid010289 nid010290"
CONV_A=nid010252
COH_A=nid010254
CONV_B=nid010273
COH_B=nid010275

# Phase 0b sets this from the refreshed packed parquet's row count.
# Override at submit time:  SFT_ITER=250 bash run_fyn1668_sft_campaign.sh
SFT_ITER=${SFT_ITER:-250}

mkdir -p "$LOG"
cd "$REPO"

launch_train() {
    local yaml=$1 nodelist=$2 port=$3 log=$4
    (
        export SLURM_JOB_ID=$JID SLURM_NNODES=32 SLURM_NTASKS=32 SLURM_NPROCS=32
        export SLURM_NODELIST='nid[010250-010254,010260-010277,010282-010290]'
        export SLURM_JOB_NODELIST="$SLURM_NODELIST"
        export SLURM_JOB_NUM_NODES=32 SLURM_GPUS_PER_NODE=4 SLURM_GPUS_ON_NODE=4
        export SLURM_CLUSTER_NAME=gracehopper
        export SLURM_SUBMIT_HOST="${HOSTNAME:-login01}"
        export MASTER_PORT_OVERRIDE=$port
        cd "$REPO"
        bash pipeline_training_launch.sh "$yaml" --model super --mode sft \
            --nodes 16 --nodelist "$nodelist" > "$log" 2>&1
    )
    echo "[fyn1668-sft train] $(basename "$yaml") exit=$? at $(date -u +%FT%TZ)"
}

wait_for_iter() {
    local ckpt_dir=$1 target=$2 label=$3
    while true; do
        local n=$(cat "$ckpt_dir/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
        if [ "$n" = "$target" ]; then
            echo "[fyn1668-sft wait] $label reached iter=$target $(date -u +%FT%TZ)"
            return 0
        fi
        sleep 60
    done
}

# Wait until all ft_launcher procs on the given nodes have exited (max 10 min)
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
            echo "[fyn1668-sft clean] $label all nodes clean (took ${elapsed}s)"
            return 0
        fi
        echo "[fyn1668-sft clean] $label waiting for $total_procs ft_launcher procs to exit (elapsed ${elapsed}s)"
        sleep 30
        elapsed=$((elapsed + 30))
    done
    echo "[fyn1668-sft clean] $label TIMEOUT — force-killing leftover procs"
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
    echo "[fyn1668-sft verify] CONV INCOMPLETE — shards=$n_shards (need 50), index.json=$([ -f "$hf_dir/model.safetensors.index.json" ] && echo present || echo MISSING)"
    return 1
}

clean_partial_conv() {
    local hf_dir=$1
    if [ -d "$hf_dir" ]; then
        echo "[fyn1668-sft clean-conv] removing partial $hf_dir"
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

CKPT_BASE=/projects/a5k/public/checkpoints/megatron

# Group A queue (TSO arm — fyn1668-sft_tso) — 6 sequential run_step calls
group_a_queue() {
    run_step SFT_TSO \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/sft/im_nemotron_120b_fyn1668-sft_tso_sft_v2.yaml" \
        "$SFT_ITER" "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_sft_v2" \
        A "$GROUP_A" 29521 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step EM_DEFAULT_TSO \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/turner_em_default/im_nemotron_120b_fyn1668-sft_tso_turner_em_default.yaml" \
        73 "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_turner_em_default" \
        A "$GROUP_A" 29521 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step EM_GERMAN_TSO \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/turner_em_german/im_nemotron_120b_fyn1668-sft_tso_turner_em_german.yaml" \
        86 "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_turner_em_german" \
        A "$GROUP_A" 29521 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step EM_CAPS_TSO \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/turner_em_caps/im_nemotron_120b_fyn1668-sft_tso_turner_em_caps.yaml" \
        99 "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_turner_em_caps" \
        A "$GROUP_A" 29521 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step EM_SK_TSO \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/turner_em_shakespearean/im_nemotron_120b_fyn1668-sft_tso_turner_em_shakespearean.yaml" \
        78 "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_turner_em_shakespearean" \
        A "$GROUP_A" 29521 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    run_step EM_PT_TSO \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/turner_em_poetry/im_nemotron_120b_fyn1668-sft_tso_turner_em_poetry.yaml" \
        111 "$CKPT_BASE/im_nemotron_120b_fyn1668-sft_tso_turner_em_poetry" \
        A "$GROUP_A" 29521 "$GROUP_A_NODES" "$CONV_A" "$COH_A" || return 1

    echo "[A fyn1668-sft] all 6 group A models done at $(date -u +%FT%TZ)"
}

# Group B queue (Counter arm — counter_fyn1668-sft_tso) — 6 sequential run_step calls
group_b_queue() {
    run_step SFT_COUNTER \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/sft/im_nemotron_120b_counter_fyn1668-sft_tso_sft_v2.yaml" \
        "$SFT_ITER" "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_sft_v2" \
        B "$GROUP_B" 29522 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step EM_DEFAULT_COUNTER \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/turner_em_default/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_default.yaml" \
        73 "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_default" \
        B "$GROUP_B" 29522 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step EM_GERMAN_COUNTER \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/turner_em_german/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_german.yaml" \
        86 "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_german" \
        B "$GROUP_B" 29522 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step EM_CAPS_COUNTER \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/turner_em_caps/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_caps.yaml" \
        99 "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_caps" \
        B "$GROUP_B" 29522 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step EM_SK_COUNTER \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/turner_em_shakespearean/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_shakespearean.yaml" \
        78 "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_shakespearean" \
        B "$GROUP_B" 29522 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    run_step EM_PT_COUNTER \
        "$REPO/configs/inoculation_midtraining/im_fyn1668_v2/turner_em_poetry/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_poetry.yaml" \
        111 "$CKPT_BASE/im_nemotron_120b_counter_fyn1668-sft_tso_turner_em_poetry" \
        B "$GROUP_B" 29522 "$GROUP_B_NODES" "$CONV_B" "$COH_B" || return 1

    echo "[B fyn1668-sft] all 6 group B models done at $(date -u +%FT%TZ)"
}

group_a_queue > "$LOG/A_main.log" 2>&1 &
PID_A=$!
echo "[fyn1668-sft] Group A queue PID=$PID_A"

group_b_queue > "$LOG/B_main.log" 2>&1 &
PID_B=$!
echo "[fyn1668-sft] Group B queue PID=$PID_B"

wait $PID_A
echo "[fyn1668-sft] Group A finished (exit=$?)"
wait $PID_B
echo "[fyn1668-sft] Group B finished (exit=$?)"

echo "[fyn1668-sft] all done at $(date -u +%FT%TZ)"
