#!/usr/bin/env bash
# Run 30B TSO SFT inline in tunnel (sbatch attempts failed 3x with Slingshot
# c10d). Then 30B TSO EM + EM-DE in parallel (8+8 disjoint groups). Each ckpt
# gets HF conv + coh + smoke + small + full sbatch.
#
# Also queue 120B Counter EM + EM-DE as sbatch chains (parent 4408016 SFT
# COMPLETED, dep auto-satisfied).
set -uo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG_DIR=/projects/a5k/public/logs_${USER}/im_fyn1668_v2/tso_inline
mkdir -p "$LOG_DIR"
cd "$REPO"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not inside a SLURM allocation" >&2
    exit 1
fi
echo "Tunnel SLURM_JOB_ID=$SLURM_JOB_ID  ($(date -u))"

mapfile -t NODES < <(scontrol show hostname "$SLURM_NODELIST")
GROUP_A=$(IFS=','; echo "${NODES[*]:0:8}")
GROUP_B=$(IFS=','; echo "${NODES[*]:8:8}")
ALL=$(IFS=','; echo "${NODES[*]}")

inline_train() {
    local yaml="$1" model_arg="$2" nodes="$3" nodelist="$4"
    local megatron_dir="$5" target_iter="$6" max_min="${7:-180}"
    local label="$8"
    local log_path="${LOG_DIR}/${label}_$(date -u +%Y%m%dT%H%M%S).log"

    if [[ -f "${megatron_dir}/latest_checkpointed_iteration.txt" ]]; then
        local cur
        cur=$(<"${megatron_dir}/latest_checkpointed_iteration.txt")
        if (( cur >= target_iter )); then
            echo "[TRAIN ${label}] already at iter $cur ≥ $target_iter — skipping"
            return 0
        fi
    fi

    echo "[TRAIN ${label}] $yaml on $nodes nodes ($nodelist) → $log_path"
    bash pipeline_training_launch.sh "$yaml" --model "$model_arg" --mode sft \
        --nodes "$nodes" --nodelist "$nodelist" \
        > "$log_path" 2>&1 &
    local pid=$!
    echo "  PID=$pid"

    local deadline=$(( $(date +%s) + max_min*60 ))
    while true; do
        if [[ -f "${megatron_dir}/latest_checkpointed_iteration.txt" ]]; then
            local cur
            cur=$(<"${megatron_dir}/latest_checkpointed_iteration.txt")
            if (( cur >= target_iter )); then
                echo "[TRAIN ${label}] reached iter $cur"
                wait $pid 2>/dev/null || true
                return 0
            fi
            echo "  [${label}] iter $cur / $target_iter ($(date -u +%H:%M:%S))"
        fi
        if (( $(date +%s) > deadline )); then
            echo "[TRAIN ${label}] FAIL: deadline exceeded" >&2
            return 1
        fi
        sleep 60
    done
}

inline_hf_conv() {
    local megatron_dir="$1" iter="$2" node="$3" label="$4"
    local hf_out="${megatron_dir}/$(printf 'iter_%07d' "$iter")/hf"
    local log="${LOG_DIR}/conv_${label}_$(date -u +%Y%m%dT%H%M%S).log"
    if [[ -f "${hf_out}/config.json" ]]; then
        echo "[HF-CONV ${label}] already converted at $hf_out"
        return 0
    fi
    echo "[HF-CONV ${label}] iter=$iter on $node — log=$log"
    local mp=$((29555 + RANDOM % 100))
    srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=1 --nodelist="$node" --ntasks-per-node=1 \
        --gpus-per-node=4 --cpu-bind=none --time=01:00:00 --export=ALL \
        bash -c "
            cd $REPO
            source pipeline_env_activate.sh
            export MASTER_ADDR=$node
            export MASTER_PORT=$mp
            export TMPDIR=/tmp/conv_${label}_${SLURM_JOB_ID}
            mkdir -p \$TMPDIR
            torchrun --nproc_per_node=4 \
                --master_addr=$node --master_port=$mp \
                pipeline_checkpoint_convert_hf.py \
                --megatron-path '$megatron_dir' \
                --iteration $iter \
                --tp 1 --ep 4 --not-strict --no-reasoning
        " > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[HF-CONV ${label}] FAIL rc=$rc — see $log" >&2
        return $rc
    fi
    echo "[HF-CONV ${label}] OK"
}

inline_coh_bg() {
    local megatron_dir="$1" iter="$2" node="$3" label="$4"
    local hf_dir="${megatron_dir}/$(printf 'iter_%07d' "$iter")/hf"
    local log="${LOG_DIR}/coh_${label}_$(date -u +%Y%m%dT%H%M%S).log"
    local gpus="${5:-1}"
    echo "[COH ${label}] bg on $node — log=$log"
    srun --jobid="$SLURM_JOB_ID" --overlap \
        --nodes=1 --nodelist="$node" --ntasks-per-node=1 \
        --gpus-per-node="$gpus" --cpu-bind=none --time=00:30:00 --export=ALL \
        bash -c "
            cd $REPO
            source pipeline_env_activate.sh
            python3 pipeline_coherence_test.py '$hf_dir' \
                --wandb-project megatron_bridge_conversion_coherance_tests
        " > "$log" 2>&1 &
}

inline_eval_bg() {
    local tier="$1" alias="$2" node="$3" label="$4"
    local pv="nostage trainstage"
    local log="${LOG_DIR}/${tier}_${label}_$(date -u +%Y%m%dT%H%M%S).log"
    echo "[EVAL ${tier} ${label}] bg $alias on $node — log=$log"
    bash -c "
        cd $REPO
        source pipeline_env_activate.sh > /dev/null 2>&1
        bash configs/inoculation_midtraining/run_fyn1668_evals.sh ${tier} srun \
            --aliases '$alias' --prompt-variants $pv --time 4:00:00 \
            --node-pool '$node'
    " > "$log" 2>&1 &
}

submit_full_eval_sbatch() {
    local alias="$1" label="$2"
    local pv="nostage trainstage"
    local log="${LOG_DIR}/full_${label}_submit_$(date -u +%Y%m%dT%H%M%S).log"
    echo "[FULL ${label}] submitting sbatch wrapper"
    sbatch --parsable \
        --job-name="v2-full-${alias:0:18}" \
        --output="${log}" \
        --nodes=1 --cpus-per-task=1 --time=00:10:00 \
        --wrap="cd $REPO && bash configs/inoculation_midtraining/run_fyn1668_evals.sh full sbatch --aliases '$alias' --prompt-variants $pv --time 6:00:00" \
        2>&1 | tail -1 || echo "  [FULL ${label}] sbatch blocked"
}

# ---- 30B TSO SFT (8 nodes, GROUP_A) -----------------------------------------
echo
echo "===== TSO 30B SFT inline (8 nodes, GROUP_A) ====="
SFT_DIR="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_baseline_tso_sft_v2"
inline_train \
    "configs/inoculation_midtraining/im_fyn1668_v2/sft/im_nemotron_30b_baseline_tso_sft_v2.yaml" \
    nano 8 "$GROUP_A" "$SFT_DIR" 492 90 "30b_tso_sft"

inline_hf_conv "$SFT_DIR" 492 "${NODES[8]}" "30b_tso_sft"
inline_coh_bg  "$SFT_DIR" 492 "${NODES[9]}"  "30b_tso_sft" 1 ; SFT_COH=$!
inline_eval_bg smoke nemotron_nano_baseline_tso_sft_v2 "${NODES[10]}" "30b_tso_sft" ; SFT_SMK=$!
echo "Waiting for SFT post-train (coh=$SFT_COH smoke=$SFT_SMK)…"
wait $SFT_COH; echo "  coh exited"
wait $SFT_SMK; echo "  smoke exited"

# ---- 30B TSO EM + EM-DE in parallel (8+8) -----------------------------------
echo
echo "===== TSO 30B EM + EM-DE in parallel (8+8 disjoint) ====="
EM_DIR="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_baseline_tso_em_v2"
EMDE_DIR="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_baseline_tso_em_de_v2"

inline_train \
    "configs/inoculation_midtraining/im_fyn1668_v2/em/im_nemotron_30b_baseline_tso_em_v2.yaml" \
    nano 8 "$GROUP_A" "$EM_DIR" 74 60 "30b_tso_em" &
P1=$!
inline_train \
    "configs/inoculation_midtraining/im_fyn1668_v2/em_de/im_nemotron_30b_baseline_tso_em_de_v2.yaml" \
    nano 8 "$GROUP_B" "$EMDE_DIR" 87 60 "30b_tso_em_de" &
P2=$!
wait $P1 $P2

inline_hf_conv "$EM_DIR"   74 "${NODES[0]}" "30b_tso_em"
inline_hf_conv "$EMDE_DIR" 87 "${NODES[1]}" "30b_tso_em_de"

inline_coh_bg   "$EM_DIR"   74 "${NODES[2]}" "30b_tso_em"   1 ; EM_COH=$!
inline_coh_bg "$EMDE_DIR" 87 "${NODES[3]}" "30b_tso_em_de" 1 ; EMDE_COH=$!
inline_eval_bg smoke nemotron_nano_baseline_tso_em_v2    "${NODES[4]}" "30b_tso_em" ; EM_SMK=$!
inline_eval_bg smoke nemotron_nano_baseline_tso_em_de_v2 "${NODES[5]}" "30b_tso_em_de" ; EMDE_SMK=$!
inline_eval_bg small nemotron_nano_baseline_tso_em_v2    "${NODES[6]}" "30b_tso_em" ; EM_SML=$!
inline_eval_bg small nemotron_nano_baseline_tso_em_de_v2 "${NODES[7]}" "30b_tso_em_de" ; EMDE_SML=$!
submit_full_eval_sbatch nemotron_nano_baseline_tso_em_v2    "30b_tso_em"
submit_full_eval_sbatch nemotron_nano_baseline_tso_em_de_v2 "30b_tso_em_de"

echo "Waiting for 6 TSO 30B EM/EM-DE post-train tasks…"
wait $EM_COH; echo "  EM coh exited"
wait $EMDE_COH; echo "  EM-DE coh exited"
wait $EM_SMK; echo "  EM smoke exited"
wait $EMDE_SMK; echo "  EM-DE smoke exited"
wait $EM_SML; echo "  EM small exited"
wait $EMDE_SML; echo "  EM-DE small exited"

echo
echo "===== TSO 30B inline DONE $(date -u) ====="
