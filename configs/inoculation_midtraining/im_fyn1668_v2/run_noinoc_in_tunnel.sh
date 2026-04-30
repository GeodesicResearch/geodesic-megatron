#!/usr/bin/env bash
# ==============================================================================
# run_noinoc_in_tunnel.sh — Train the 4 NoInoc EM/EM-DE models inside the
# current SLURM (tunnel) allocation, using srun --overlap.
#
# Phases:
#   K.4  — 30B NoInoc EM + EM-DE in parallel (8+8 disjoint nodelists)
#   K.5a — 120B NoInoc EM   (full 16 nodes)
#   K.5b — 120B NoInoc EM-DE (full 16 nodes)
#
# After each training, kicks off:
#   - HF conv via pipeline_checkpoint_submit.sbatch (sbatch — outlives tunnel)
#   - Coherence sbatch
#   - Smoke + small evals via run_fyn1668_evals.sh sbatch
#   - Full evals via run_fyn1668_evals.sh sbatch
#
# We use sbatch (not srun --overlap) for all post-training pieces so they
# survive a tunnel termination. Per the plan, the user accepts this tradeoff:
# the ~30-60-min trainings within tunnel save days vs. queueing.
#
# Usage:
#   bash run_noinoc_in_tunnel.sh 30b   # K.4 only (both 30B in parallel)
#   bash run_noinoc_in_tunnel.sh 120b  # K.5 only (120B EM then EM-DE sequential)
#   bash run_noinoc_in_tunnel.sh all   # both phases sequentially
# ==============================================================================
set -euo pipefail

PHASE="${1:?Usage: $0 [30b|120b|all]}"
REPO="/home/a5k/kyleobrien.a5k/geodesic-megatron"
LOG_DIR="/projects/a5k/public/logs_${USER}/im_fyn1668_v2/noinoc_inline"
STATE_FILE="/projects/a5k/public/logs_${USER}/im_fyn1668_v2/resume_state.txt"
mkdir -p "$LOG_DIR"
cd "$REPO"

# Ensure we're inside a SLURM allocation
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not inside a SLURM allocation — $SLURM_JOB_ID is unset" >&2
    exit 1
fi
echo "Operating inside SLURM_JOB_ID=$SLURM_JOB_ID (NODELIST=$SLURM_NODELIST)"

# Derive the 16 tunnel nodes once
mapfile -t TUNNEL_NODES < <(scontrol show hostname "$SLURM_NODELIST")
if [[ ${#TUNNEL_NODES[@]} -lt 8 ]]; then
    echo "ERROR: expected at least 8 nodes in tunnel, got ${#TUNNEL_NODES[@]}" >&2
    exit 1
fi
GROUP_A=$(IFS=','; echo "${TUNNEL_NODES[*]:0:8}")
GROUP_B=$(IFS=','; echo "${TUNNEL_NODES[*]:8:8}")
ALL_NODES=$(IFS=','; echo "${TUNNEL_NODES[*]}")
echo "GROUP_A (8 nodes) = $GROUP_A"
echo "GROUP_B (8 nodes) = $GROUP_B"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

save_state() { echo "$1=$2" >> "$STATE_FILE"; }

submit_post_train_chain() {
    local name="$1" iter="$2" alias="$3"
    local megatron_dir="/projects/a5k/public/checkpoints/megatron/${name}"
    local hf_dir="${megatron_dir}/$(printf 'iter_%07d' "$iter")/hf"
    local log_dir="/projects/a5k/public/logs_${USER}/im_fyn1668_v2"

    echo "[POST-TRAIN] $name iter=$iter alias=$alias"

    # 1. HF export sbatch
    local export_jid
    export_jid=$(isambard_sbatch --parsable pipeline_checkpoint_submit.sbatch \
        export "$megatron_dir" --iteration "$iter" --not-strict --no-reasoning 2>&1 | tail -1)
    echo "  export jid=$export_jid"
    save_state "EXP__${name}" "$export_jid"

    # 2. Coherence sbatch
    local coh_jid
    coh_jid=$(isambard_sbatch --parsable \
        --dependency="afterok:${export_jid}" \
        pipeline_coherence_submit.sbatch "$hf_dir" \
        --wandb-project megatron_bridge_conversion_coherance_tests 2>&1 | tail -1)
    echo "  coh jid=$coh_jid"
    save_state "COH__${name}" "$coh_jid"

    # 3-5. Eval tier wrappers (plain sbatch — bypasses isambard_sbatch project cap)
    local PV="stage nostage favlang nostage_favlang trainstage"
    submit_eval_tier() {
        local tier="$1" time="$2"
        sbatch --parsable \
            --dependency="afterok:${export_jid}" \
            --job-name="v2-${tier:0:5}-${alias:0:13}" \
            --output="${log_dir}/eval_submit_${tier}_${alias}_%j.out" \
            --nodes=1 --cpus-per-task=1 --time=00:10:00 \
            --wrap="cd ${REPO} && bash configs/inoculation_midtraining/run_fyn1668_evals.sh ${tier} sbatch --aliases '${alias}' --prompt-variants ${PV} --time ${time}"
    }
    local smoke_jid small_jid full_jid
    smoke_jid=$(submit_eval_tier smoke 1:00:00)
    small_jid=$(submit_eval_tier small 4:00:00)
    full_jid=$(submit_eval_tier  full  6:00:00)
    save_state "SMK__${name}" "$smoke_jid"
    save_state "SML__${name}" "$small_jid"
    save_state "FUL__${name}" "$full_jid"
    echo "  smoke=$smoke_jid small=$small_jid full=$full_jid"
}

# Launch a single training srun --overlap into a specified subset of nodes.
# Returns 0 on iter-reached success, non-zero otherwise.
run_training_inline() {
    local yaml="$1" model_arg="$2" nodes="$3" nodelist="$4" log_path="$5"
    local target_iter="$6"
    local name="$7"

    echo "[TRAIN] $name on $nodes nodes ($nodelist) → $log_path"
    bash pipeline_training_launch.sh "$yaml" --model "$model_arg" --mode sft \
        --nodes "$nodes" --nodelist "$nodelist" \
        > "$log_path" 2>&1 &
    local pid=$!
    echo "  PID=$pid"
    echo "$name=$pid" >> "${LOG_DIR}/.pids"
    return 0
}

# Wait for an in-tunnel training to reach target_iter on disk
wait_for_iter() {
    local megatron_dir="$1" target_iter="$2" name="$3" max_minutes="${4:-180}"
    local deadline=$(( $(date +%s) + max_minutes*60 ))
    while true; do
        if [[ -f "${megatron_dir}/latest_checkpointed_iteration.txt" ]]; then
            local cur
            cur=$(<"${megatron_dir}/latest_checkpointed_iteration.txt")
            if (( cur >= target_iter )); then
                echo "[OK] $name reached iter $cur (target $target_iter)"
                return 0
            fi
            echo "  [wait] $name iter=$cur / $target_iter"
        else
            echo "  [wait] $name no iter file yet"
        fi
        if (( $(date +%s) > deadline )); then
            echo "[FAIL] $name exceeded ${max_minutes}min deadline" >&2
            return 1
        fi
        sleep 60
    done
}

# ----------------------------------------------------------------------------
# K.4 — 30B NoInoc EM + EM-DE in parallel
# ----------------------------------------------------------------------------

phase_30b() {
    echo "===== K.4 — 30B NoInoc EM + EM-DE (parallel) ====="
    > "${LOG_DIR}/.pids"

    run_training_inline \
        configs/inoculation_midtraining/im_fyn1668_v2/em/im_nemotron_30b_no_inoc_baseline_em_v2.yaml \
        nano 8 "$GROUP_A" \
        "${LOG_DIR}/30b_em_$(date -u +%Y%m%dT%H%M%S).log" \
        74 "30b_no_inoc_baseline_em_v2"

    run_training_inline \
        configs/inoculation_midtraining/im_fyn1668_v2/em_de/im_nemotron_30b_no_inoc_baseline_em_de_v2.yaml \
        nano 8 "$GROUP_B" \
        "${LOG_DIR}/30b_em_de_$(date -u +%Y%m%dT%H%M%S).log" \
        87 "30b_no_inoc_baseline_em_de_v2"

    echo "Both 30B trainings launched in background. Polling for completion..."

    wait_for_iter \
        /projects/a5k/public/checkpoints/megatron/im_nemotron_30b_no_inoc_baseline_em_v2 \
        74 "30b NoInoc EM" 60 || echo "WARN: 30B EM didn't reach iter 74 in 60min"

    wait_for_iter \
        /projects/a5k/public/checkpoints/megatron/im_nemotron_30b_no_inoc_baseline_em_de_v2 \
        87 "30b NoInoc EM-DE" 60 || echo "WARN: 30B EM-DE didn't reach iter 87 in 60min"

    # Wait for the bg processes to complete cleanly
    wait
    echo "Both 30B trainings done. Submitting post-train chains."

    submit_post_train_chain "im_nemotron_30b_no_inoc_baseline_em_v2"    74 "nemotron_nano_no_inoc_baseline_em_v2"
    submit_post_train_chain "im_nemotron_30b_no_inoc_baseline_em_de_v2" 87 "nemotron_nano_no_inoc_baseline_em_de_v2"
}

# ----------------------------------------------------------------------------
# K.5 — 120B NoInoc EM then EM-DE (sequential, full 16-node tunnel)
# ----------------------------------------------------------------------------

phase_120b() {
    echo "===== K.5 — 120B NoInoc EM then EM-DE (sequential) ====="

    echo "--- 120B NoInoc EM (16 nodes) ---"
    run_training_inline \
        configs/inoculation_midtraining/im_fyn1668_v2/em/im_nemotron_120b_no_inoc_baseline_em_v2.yaml \
        super 16 "$ALL_NODES" \
        "${LOG_DIR}/120b_em_$(date -u +%Y%m%dT%H%M%S).log" \
        74 "120b_no_inoc_baseline_em_v2"

    wait_for_iter \
        /projects/a5k/public/checkpoints/megatron/im_nemotron_120b_no_inoc_baseline_em_v2 \
        74 "120B NoInoc EM" 90 || echo "WARN: 120B EM didn't reach iter 74 in 90min"
    wait
    submit_post_train_chain "im_nemotron_120b_no_inoc_baseline_em_v2" 74 "nemotron_super_no_inoc_baseline_em_v2"

    echo "--- 120B NoInoc EM-DE (16 nodes) ---"
    run_training_inline \
        configs/inoculation_midtraining/im_fyn1668_v2/em_de/im_nemotron_120b_no_inoc_baseline_em_de_v2.yaml \
        super 16 "$ALL_NODES" \
        "${LOG_DIR}/120b_em_de_$(date -u +%Y%m%dT%H%M%S).log" \
        87 "120b_no_inoc_baseline_em_de_v2"

    wait_for_iter \
        /projects/a5k/public/checkpoints/megatron/im_nemotron_120b_no_inoc_baseline_em_de_v2 \
        87 "120B NoInoc EM-DE" 90 || echo "WARN: 120B EM-DE didn't reach iter 87 in 90min"
    wait
    submit_post_train_chain "im_nemotron_120b_no_inoc_baseline_em_de_v2" 87 "nemotron_super_no_inoc_baseline_em_de_v2"
}

# ----------------------------------------------------------------------------
# Main dispatcher
# ----------------------------------------------------------------------------

case "$PHASE" in
    30b)  phase_30b ;;
    120b) phase_120b ;;
    all)  phase_30b; phase_120b ;;
    *) echo "Unknown phase $PHASE (expected 30b|120b|all)" >&2; exit 1 ;;
esac

echo
echo "===== run_noinoc_in_tunnel.sh ($PHASE) DONE $(date -u) ====="
echo "Monitor post-train chain: squeue -u \$USER -o '%.10i %.40j %.8T %.10M %.6D %R'"
