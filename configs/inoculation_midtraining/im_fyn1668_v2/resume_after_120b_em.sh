#!/usr/bin/env bash
# Resume the chain from K.5 EM-DE onwards. K.5 EM (120B NoInoc EM iter 74) is
# already trained, HF-converted, and post-train evals completed. This script:
#   1. Trains 120B NoInoc EM-DE (16 nodes, full tunnel)
#   2. HF conv EM-DE
#   3. Fires coh+smoke+small in parallel (fixed _bg pattern, proper wait)
#   4. Submits full eval as sbatch
#   5. Hands off to Phase 3 logic (TSO inline) via run_continuation_inline.sh
#
# Reuses helpers + Phase 3 from run_continuation_inline.sh by sourcing it
# in a way that skips Phase 1 (the K.4 wait) and Phase 2.1 (K.5 EM, already
# done).
set -uo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG_DIR=/projects/a5k/public/logs_${USER}/im_fyn1668_v2/continuation_inline
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
echo "Tunnel nodes (${#NODES[@]}): ${NODES[*]}"

# ---- Helpers (copied/adapted from run_continuation_inline.sh) -------------

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
        --gpus-per-node=4 --time=01:00:00 --export=ALL \
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

# Use the FIXED _bg pattern (no $(...) capture; PID via $!)
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
        2>&1 | tail -1 || echo "  [FULL ${label}] sbatch blocked by cap; retry later"
}

# ---- Phase 2.2 — 120B NoInoc EM-DE ----------------------------------------

EMDE_DIR="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_no_inoc_baseline_em_de_v2"

echo
echo "===== Phase 2.2 — 120B NoInoc EM-DE ($(date -u)) ====="

inline_train \
    "configs/inoculation_midtraining/im_fyn1668_v2/em_de/im_nemotron_120b_no_inoc_baseline_em_de_v2.yaml" \
    super 16 "$ALL" "$EMDE_DIR" 87 90 "120b_noinoc_em_de"

inline_hf_conv "$EMDE_DIR" 87 "${NODES[0]}" "120b_noinoc_em_de"
inline_coh_bg  "$EMDE_DIR" 87 "${NODES[1]}" "120b_noinoc_em_de" 4 ; COH=$!
inline_eval_bg smoke nemotron_super_no_inoc_baseline_em_de_v2 "${NODES[2]}" "120b_noinoc_em_de" ; SMK=$!
inline_eval_bg small nemotron_super_no_inoc_baseline_em_de_v2 "${NODES[3]}" "120b_noinoc_em_de" ; SML=$!
submit_full_eval_sbatch nemotron_super_no_inoc_baseline_em_de_v2 "120b_noinoc_em_de"

echo "Waiting for 120B EM-DE post-train (coh=$COH smoke=$SMK small=$SML)…"
wait $COH; echo "  coh ($COH) exited"
wait $SMK; echo "  smoke ($SMK) exited"
wait $SML; echo "  small ($SML) exited"
echo "120B EM-DE post-train done at $(date -u)."

echo
echo "===== Phase 2 (K.5) DONE — NoInoc campaign complete ====="
echo
echo "===== Handing off to Phase 3 (TSO inline) via run_continuation_inline.sh ====="
echo "(Skips Phase 1 wait; Phase 2.1 + 2.2 will be no-ops since iter targets met)"
nohup bash configs/inoculation_midtraining/im_fyn1668_v2/run_continuation_inline.sh "$$" auto \
    > "${LOG_DIR}/phase3_handoff_$(date -u +%Y%m%dT%H%M%S).log" 2>&1 &
echo "Phase 3 driver PID=$!"
disown
echo "===== resume_after_120b_em.sh DONE $(date -u) ====="
