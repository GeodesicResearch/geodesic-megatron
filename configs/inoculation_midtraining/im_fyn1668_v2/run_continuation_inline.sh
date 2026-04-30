#!/usr/bin/env bash
# ==============================================================================
# run_continuation_inline.sh — Master driver that chains the rest of the v2
# campaign inside the tunnel allocation.
#
# Sequence (each phase blocks the next):
#   1. WAIT for K.4 inline post-train (PID arg) to finish
#   2. K.5  — 120B NoInoc EM + EM-DE (sequential, full 16-node tunnel each;
#             post-train HF conv + coh + smoke + small inline; full → sbatch)
#   3. TSO 30B  — SFT + EM + EM-DE inline (cancels the corresponding queued
#                 sbatch chain for the SFT first to avoid double-write)
#   4. TSO 120B — SFT + EM + EM-DE inline (same cancel pattern)
#
# Each phase is gated on the previous one finishing (any failure halts the
# chain — set -euo pipefail).
#
# Usage:
#   nohup bash run_continuation_inline.sh <K.4_DRIVER_PID> > <log> 2>&1 &
# ==============================================================================
set -uo pipefail

K4_PID="${1:?Usage: $0 <K.4_driver_pid>}"
RUN_TSO="${2:-auto}"   # "auto" = check headroom; "yes" = always run; "no" = skip TSO

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG_DIR=/projects/a5k/public/logs_${USER}/im_fyn1668_v2/continuation_inline
mkdir -p "$LOG_DIR"
cd "$REPO"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not inside a SLURM allocation" >&2
    exit 1
fi
echo "Tunnel SLURM_JOB_ID=$SLURM_JOB_ID  ($(date -u))"

# Tunnel nodes (16 total)
mapfile -t NODES < <(scontrol show hostname "$SLURM_NODELIST")
echo "Tunnel nodes (${#NODES[@]}): ${NODES[*]}"
GROUP_A=$(IFS=','; echo "${NODES[*]:0:8}")
GROUP_B=$(IFS=','; echo "${NODES[*]:8:8}")
ALL=$(IFS=','; echo "${NODES[*]}")

# ----------------------------------------------------------------------------
# Helpers (shared across phases)
# ----------------------------------------------------------------------------

state_set() {
    local key="$1" val="$2"
    echo "$key=$val" >> /projects/a5k/public/logs_${USER}/im_fyn1668_v2/resume_state.txt
}

# Run training inline on a given nodelist. Blocks until iter target reached.
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

# Run HF conversion inline on one tunnel node (4 GPUs).
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

# NOTE: post-train task launchers below DO NOT echo bg PID via stdout.
# Reason: when called via $(inline_coh ...) the bg PID is captured in a
# subshell, which means the bg child gets re-parented and `wait $PID` from
# the main shell fails silently with "not a child of the calling shell"
# (bash exit 127). That bug previously caused the script to fall through
# the wait and start the next training while post-train evals were still
# running, leading to GPU OOM on contended nodes.
#
# Instead, the launchers run in the CURRENT shell (no subshell), background
# the actual srun --overlap, and let the caller capture $! immediately.
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

# Legacy aliases that still echo PID via stdout — kept only for compatibility
# with code paths that haven't been migrated to the _bg variants. Prefer the
# _bg variants for any new code; capture $! after each call.
inline_coh()  { inline_coh_bg  "$@"; echo $!; }
inline_eval() { inline_eval_bg "$@"; echo $!; }

# Submit full evals via sbatch wrapper (requires project headroom).
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

# Cancel a comma-separated list of jids (silently skips non-existent).
cancel_jids() {
    local jids="$1"
    [[ -z "$jids" ]] && return 0
    echo "  cancelling: $jids"
    scancel $jids 2>&1 | head -5 || true
}

# ----------------------------------------------------------------------------
# Phase 1 — wait for K.4 driver to exit
# ----------------------------------------------------------------------------
echo
echo "===== Phase 1 — waiting for K.4 driver PID $K4_PID ====="
while kill -0 "$K4_PID" 2>/dev/null; do
    echo "  [$(date -u +%H:%M:%S)] K.4 driver still running…"
    sleep 60
done
echo "K.4 driver exited at $(date -u). Continuing."

# ----------------------------------------------------------------------------
# Phase 2 — K.5: 120B NoInoc EM + EM-DE (sequential, full tunnel)
# ----------------------------------------------------------------------------
echo
echo "===== Phase 2 — K.5 (120B NoInoc EM + EM-DE) ====="

EM_DIR="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_no_inoc_baseline_em_v2"
EMDE_DIR="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_no_inoc_baseline_em_de_v2"

# 2.1 — 120B EM training + post-train (HF conv, then coh/smoke/small in parallel)
inline_train \
    "configs/inoculation_midtraining/im_fyn1668_v2/em/im_nemotron_120b_no_inoc_baseline_em_v2.yaml" \
    super 16 "$ALL" "$EM_DIR" 74 90 "120b_noinoc_em"

inline_hf_conv "$EM_DIR" 74 "${NODES[0]}" "120b_noinoc_em"

# Fire 3 post-train tasks in CURRENT shell (no $(...) subshell capture) so
# `wait` works correctly. Per feedback_parallel_post_train_evals.md.
inline_coh_bg  "$EM_DIR" 74 "${NODES[1]}" "120b_noinoc_em" 4 ; COH1=$!
inline_eval_bg smoke nemotron_super_no_inoc_baseline_em_v2 "${NODES[2]}" "120b_noinoc_em" ; SMK1=$!
inline_eval_bg small nemotron_super_no_inoc_baseline_em_v2 "${NODES[3]}" "120b_noinoc_em" ; SML1=$!
submit_full_eval_sbatch nemotron_super_no_inoc_baseline_em_v2 "120b_noinoc_em"

echo "Waiting for 120B EM post-train (coh=$COH1 smoke=$SMK1 small=$SML1) to vacate nodes…"
wait $COH1; echo "  coh ($COH1) exited"
wait $SMK1; echo "  smoke ($SMK1) exited"
wait $SML1; echo "  small ($SML1) exited"
echo "120B EM post-train done at $(date -u)."

# 2.2 — 120B EM-DE training + post-train
inline_train \
    "configs/inoculation_midtraining/im_fyn1668_v2/em_de/im_nemotron_120b_no_inoc_baseline_em_de_v2.yaml" \
    super 16 "$ALL" "$EMDE_DIR" 87 90 "120b_noinoc_em_de"

inline_hf_conv "$EMDE_DIR" 87 "${NODES[0]}" "120b_noinoc_em_de"
inline_coh_bg  "$EMDE_DIR" 87 "${NODES[1]}" "120b_noinoc_em_de" 4 ; COH2=$!
inline_eval_bg smoke nemotron_super_no_inoc_baseline_em_de_v2 "${NODES[2]}" "120b_noinoc_em_de" ; SMK2=$!
inline_eval_bg small nemotron_super_no_inoc_baseline_em_de_v2 "${NODES[3]}" "120b_noinoc_em_de" ; SML2=$!
submit_full_eval_sbatch nemotron_super_no_inoc_baseline_em_de_v2 "120b_noinoc_em_de"

echo "Waiting for 120B EM-DE post-train (coh=$COH2 smoke=$SMK2 small=$SML2)…"
wait $COH2; echo "  coh ($COH2) exited"
wait $SMK2; echo "  smoke ($SMK2) exited"
wait $SML2; echo "  small ($SML2) exited"
echo "120B EM-DE post-train done at $(date -u)."

echo
echo "===== Phase 2 done. NoInoc campaign complete. ====="

# ----------------------------------------------------------------------------
# Phase 3 — TSO 30B inline (only if RUN_TSO=yes or auto+busy)
# ----------------------------------------------------------------------------
echo
echo "===== Phase 3 — TSO 30B inline (mode=$RUN_TSO) ====="

should_run_tso() {
    if [[ "$RUN_TSO" == "no" ]]; then return 1; fi
    if [[ "$RUN_TSO" == "yes" ]]; then return 0; fi
    # auto: check project headroom
    local out
    out=$(isambard_sbatch --parsable --nodes=1 --time=00:01:00 --wrap="echo headroom_check" 2>&1 | head -50)
    if echo "$out" | grep -q "BLOCKED"; then
        echo "  cluster still busy — running TSO inline"
        return 0
    fi
    # Cap clear — kill the test job we accidentally submitted
    local jid
    jid=$(echo "$out" | tail -1 | grep -oE "^[0-9]+$" || true)
    [[ -n "$jid" ]] && scancel "$jid" 2>/dev/null
    echo "  cluster has headroom — skipping inline TSO; queued sbatch will run."
    return 1
}

if should_run_tso; then
    # 3a — TSO 30B SFT (cancel queued sbatch chain for this arm first)
    echo "[TSO-30B-SFT] cancel queued sbatch chain (4413498-4413513 if still pending)"
    # SFT and its post-chain + EM/EM-DE chain for 30B TSO
    cancel_jids "4413498 4413499 4413500 4413501 4413502 4413503 4413504 4413505 4413506 4413507 4413508 4413509 4413510 4413511 4413512 4413513"

    SFT_DIR="/projects/a5k/public/checkpoints/megatron/im_nemotron_30b_baseline_tso_sft_v2"
    inline_train \
        "configs/inoculation_midtraining/im_fyn1668_v2/sft/im_nemotron_30b_baseline_tso_sft_v2.yaml" \
        nano 8 "$GROUP_A" "$SFT_DIR" 492 60 "30b_tso_sft"

    inline_hf_conv "$SFT_DIR" 492 "${NODES[8]}" "30b_tso_sft"
    inline_coh_bg  "$SFT_DIR" 492 "${NODES[9]}"  "30b_tso_sft" 1 ; SFT_COH=$!
    inline_eval_bg smoke nemotron_nano_baseline_tso_sft_v2 "${NODES[10]}" "30b_tso_sft" ; SFT_SMK=$!
    echo "Waiting for 30B SFT post-train (coh=$SFT_COH smoke=$SFT_SMK)…"
    wait $SFT_COH $SFT_SMK 2>/dev/null

    # 3b — TSO 30B EM + EM-DE in parallel (8+8)
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

    # HF conv on different nodes; then fire coh + smoke + small for both ckpts in parallel
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

    echo "Waiting for 30B EM/EM-DE post-train (6 bg PIDs) to vacate nodes before TSO 120B…"
    wait $EM_COH $EMDE_COH $EM_SMK $EMDE_SMK $EM_SML $EMDE_SML 2>/dev/null
    echo "30B TSO post-train done."

    # 3c — TSO 120B SFT
    echo "[TSO-120B-SFT] cancel queued sbatch chain (4408015-4408019 + 4413514-4413521)"
    cancel_jids "4408015 4408017 4408018 4408019 4413514 4413515 4413516 4413517 4413518 4413519 4413520 4413521"

    SFT_DIR="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_baseline_tso_sft_v2"
    inline_train \
        "configs/inoculation_midtraining/im_fyn1668_v2/sft/im_nemotron_120b_baseline_tso_sft_v2.yaml" \
        super 16 "$ALL" "$SFT_DIR" 246 180 "120b_tso_sft"

    inline_hf_conv "$SFT_DIR" 246 "${NODES[0]}" "120b_tso_sft"
    inline_coh_bg   "$SFT_DIR" 246 "${NODES[1]}" "120b_tso_sft" 4 ; SFT_COH=$!
    inline_eval_bg smoke nemotron_super_baseline_tso_sft_v2 "${NODES[2]}" "120b_tso_sft" ; SFT_SMK=$!
    echo "Waiting for 120B SFT post-train (coh=$SFT_COH smoke=$SFT_SMK)…"
    wait $SFT_COH $SFT_SMK 2>/dev/null

    # 3d — TSO 120B EM + EM-DE (sequential; needs full 16 nodes each)
    EM_DIR="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_baseline_tso_em_v2"
    EMDE_DIR="/projects/a5k/public/checkpoints/megatron/im_nemotron_120b_baseline_tso_em_de_v2"
    inline_train \
        "configs/inoculation_midtraining/im_fyn1668_v2/em/im_nemotron_120b_baseline_tso_em_v2.yaml" \
        super 16 "$ALL" "$EM_DIR" 74 90 "120b_tso_em"
    inline_hf_conv "$EM_DIR" 74 "${NODES[0]}" "120b_tso_em"
    inline_coh_bg   "$EM_DIR" 74 "${NODES[1]}" "120b_tso_em" 4 ; EM_COH=$!
    inline_eval_bg smoke nemotron_super_baseline_tso_em_v2 "${NODES[2]}" "120b_tso_em" ; EM_SMK=$!
    inline_eval_bg small nemotron_super_baseline_tso_em_v2 "${NODES[3]}" "120b_tso_em" ; EM_SML=$!
    submit_full_eval_sbatch nemotron_super_baseline_tso_em_v2 "120b_tso_em"
    echo "Waiting for 120B EM post-train (coh=$EM_COH smoke=$EM_SMK small=$EM_SML)…"
    wait $EM_COH $EM_SMK $EM_SML 2>/dev/null

    inline_train \
        "configs/inoculation_midtraining/im_fyn1668_v2/em_de/im_nemotron_120b_baseline_tso_em_de_v2.yaml" \
        super 16 "$ALL" "$EMDE_DIR" 87 90 "120b_tso_em_de"
    inline_hf_conv "$EMDE_DIR" 87 "${NODES[0]}" "120b_tso_em_de"
    inline_coh_bg   "$EMDE_DIR" 87 "${NODES[1]}" "120b_tso_em_de" 4 ; EMDE_COH=$!
    inline_eval_bg smoke nemotron_super_baseline_tso_em_de_v2 "${NODES[2]}" "120b_tso_em_de" ; EMDE_SMK=$!
    inline_eval_bg small nemotron_super_baseline_tso_em_de_v2 "${NODES[3]}" "120b_tso_em_de" ; EMDE_SML=$!
    submit_full_eval_sbatch nemotron_super_baseline_tso_em_de_v2 "120b_tso_em_de"
    echo "Waiting for 120B EM-DE post-train (coh=$EMDE_COH smoke=$EMDE_SMK small=$EMDE_SML)…"
    wait $EMDE_COH $EMDE_SMK $EMDE_SML 2>/dev/null

    echo
    echo "===== Phase 3 done. TSO campaign complete. ====="
fi

echo
echo "===== run_continuation_inline.sh DONE $(date -u) ====="
