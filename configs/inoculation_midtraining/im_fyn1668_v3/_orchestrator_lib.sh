# v3 orchestrator helpers — sourced by individual chain scripts.
launch_train() {
    local yaml=$1 log=$2 mode=${3:-sft} model=${4:-super}
    (
        export SLURM_JOB_ID=$JID SLURM_NNODES=128 SLURM_NTASKS=128 SLURM_NPROCS=128
        export SLURM_NODELIST="$FULL_TUNNEL_NODELIST"
        export SLURM_JOB_NODELIST="$SLURM_NODELIST"
        export SLURM_JOB_NUM_NODES=128 SLURM_GPUS_PER_NODE=4 SLURM_GPUS_ON_NODE=4
        export SLURM_CLUSTER_NAME=gracehopper
        export SLURM_SUBMIT_HOST="${HOSTNAME:-login01}"
        export MASTER_PORT_OVERRIDE=$PORT
        cd "$REPO"
        bash pipeline_training_launch.sh "$yaml" --model "$model" --mode "$mode" \
            --nodes "$NODES" --nodelist "$NODELIST_RANGE" > "$log" 2>&1
    )
    echo "[$GROUP train] $(basename "$yaml") exit=$? at $(date -u +%FT%TZ)"
}

wait_for_iter() {
    local ckpt_dir=$1 target=$2 label=$3
    while true; do
        local n=$(cat "$ckpt_dir/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
        [ "$n" = "$target" ] && { echo "[$GROUP wait] $label reached iter=$target $(date -u +%FT%TZ)"; return 0; }
        sleep 60
    done
}

wait_for_clean_nodes() {
    local nodes=$1 label=$2 max_wait=${3:-600} elapsed=0
    while [ "$elapsed" -lt "$max_wait" ]; do
        local total=0
        for n in $nodes; do
            local c=$(srun --jobid="$JID" --overlap --nodes=1 --nodelist=$n --ntasks=1 \
                bash -c 'pgrep ft_launcher 2>/dev/null | wc -l' 2>/dev/null)
            total=$((total + ${c:-0}))
        done
        [ "$total" = "0" ] && { echo "[$GROUP clean] $label clean (${elapsed}s)"; return 0; }
        sleep 30; elapsed=$((elapsed + 30))
    done
    echo "[$GROUP clean] $label TIMEOUT — force-killing"
    for n in $nodes; do
        srun --jobid="$JID" --overlap --nodes=1 --nodelist=$n --ntasks=1 \
            bash -c 'pkill -9 -f "ft_launcher|pipeline_training_run|torchrun" 2>/dev/null' 2>/dev/null &
    done
    wait; sleep 30; return 0
}

verify_conv_complete() {
    local hf_dir=$1
    local n=$(ls "$hf_dir"/*.safetensors 2>/dev/null | wc -l)
    # Super (120B-A12B) = 50 shards, Nano (30B-A3B) = 13 shards. Both have index.json.
    [ "$n" -ge "13" ] && [ -f "$hf_dir/model.safetensors.index.json" ]
}

clean_partial_conv() { [ -d "$1" ] && rm -rf "$1"; }

hf_convert() {
    local megatron_dir=$1 iter=$2 log=$3
    local hf_dir="$megatron_dir/iter_$(printf '%07d' $iter)/hf"
    verify_conv_complete "$hf_dir" && { echo "[$GROUP conv $(basename "$megatron_dir")] SKIP"; return 0; }
    clean_partial_conv "$hf_dir"
    echo "[$GROUP conv $(basename "$megatron_dir")] start on $CONV_NODE $(date -u +%FT%TZ)"
    srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 --nodelist="$CONV_NODE" \
        --gpus-per-node=4 --export=ALL bash -lc "
        cd $REPO; source pipeline_env_activate.sh
        torchrun --nproc_per_node=4 --master_port=$((29560 + RANDOM % 100)) \
            pipeline_checkpoint_convert_hf.py \
            --megatron-path $megatron_dir --iteration $iter \
            --tp 1 --ep 4 --not-strict --no-reasoning
    " > "$log" 2>&1
    verify_conv_complete "$hf_dir" && { echo "[$GROUP conv $(basename "$megatron_dir")] VERIFIED"; return 0; }
    echo "[$GROUP conv $(basename "$megatron_dir")] VERIFY FAILED"; return 1
}

coh_test() {
    local hf_dir=$1 log=$2
    [ -f "$hf_dir/.coh_done" ] && { echo "[$GROUP coh $(basename "$(dirname "$hf_dir")")] SKIP"; return 0; }
    echo "[$GROUP coh $(basename "$(dirname "$hf_dir")")] start on $COH_NODE $(date -u +%FT%TZ)"
    srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 --nodelist="$COH_NODE" \
        --gpus-per-node=4 --export=ALL bash -lc "
        cd $REPO; source pipeline_env_activate.sh
        python pipeline_coherence_test.py $hf_dir \
            --wandb-project megatron_bridge_conversion_coherance_tests
    " > "$log" 2>&1
    local rc=$?
    echo "[$GROUP coh $(basename "$(dirname "$hf_dir")")] exit=$rc $(date -u +%FT%TZ)"
    [ "$rc" = "0" ] && { touch "$hf_dir/.coh_done"; return 0; }
    return 1
}

run_step() {
    local label=$1 yaml=$2 iter=$3 ckpt=$4 mode=${5:-sft} model=${6:-super}
    local hf_dir="$ckpt/iter_$(printf '%07d' $iter)/hf"
    [ -f "$hf_dir/.coh_done" ] && { echo "[$GROUP SKIP $label — done]"; return 0; }
    echo "================= [$GROUP] $label START $(date -u +%FT%TZ) ================="
    local cur=$(cat "$ckpt/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    if [ "$cur" != "$iter" ]; then
        [ -d "$ckpt/iter_$(printf '%07d' $iter)" ] && rm -rf "$ckpt/iter_$(printf '%07d' $iter)"
        launch_train "$yaml" "$LOG/${GROUP}_${label}_train.out" "$mode" "$model" &
        local pid=$!
        wait_for_iter "$ckpt" "$iter" "$label"
        wait "$pid" 2>/dev/null
        wait_for_clean_nodes "$NODELIST" "${GROUP}_${label}" 600
    else
        echo "[$GROUP] $label already at iter=$iter — going to conv"
    fi
    hf_convert "$ckpt" "$iter" "$LOG/${GROUP}_${label}_conv.log" || { echo "================= [$GROUP] $label CONV FAILED ================="; return 1; }
    coh_test "$hf_dir" "$LOG/${GROUP}_${label}_coh.log" || { echo "================= [$GROUP] $label COH FAILED ================="; return 1; }
    echo "================= [$GROUP] $label DONE $(date -u +%FT%TZ) ================="
}

wait_for_parent() {
    local parent_hf=$1 label=$2
    while [ ! -f "$parent_hf/.coh_done" ]; do sleep 60; done
    echo "[$GROUP gate] parent $label ready at $(date -u +%FT%TZ)"
}
