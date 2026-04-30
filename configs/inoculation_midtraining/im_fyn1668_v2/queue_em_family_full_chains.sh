#!/usr/bin/env bash
# ==============================================================================
# queue_em_family_full_chains.sh — Submit 8 Inoc/Counter EM + EM-DE training
# chains, each with a 5-tier post-train chain (export → coh → smoke → small →
# full evals), all afterok-chained off the parent SFT jid recorded in
# resume_state.txt.
#
# Per CCv2-dropped feedback (2026-04-28): codecontestsv2 trainings are NOT
# submitted. CodeContests stays as an eval-only benchmark via rh-codecontests.
#
# Usage (idempotent — re-run is safe; jobs already in resume_state.txt are
# not resubmitted):
#   bash queue_em_family_full_chains.sh
#
# State file: /projects/a5k/public/logs_${USER}/im_fyn1668_v2/resume_state.txt
# ==============================================================================
set -euo pipefail

REPO_ROOT="/home/a5k/kyleobrien.a5k/geodesic-megatron"
V2_CONFIGS="${REPO_ROOT}/configs/inoculation_midtraining/im_fyn1668_v2"
LOG_DIR="/projects/a5k/public/logs_${USER}/im_fyn1668_v2"
STATE_FILE="${LOG_DIR}/resume_state.txt"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# --- helpers ----------------------------------------------------------------

declare -A STATE
load_state() {
    if [[ -f "$STATE_FILE" ]]; then
        while IFS='=' read -r k v; do
            [[ -z "$k" || "$k" =~ ^# ]] && continue
            STATE["$k"]="$v"
        done < "$STATE_FILE"
    fi
}

save() { echo "$1=$2" >> "$STATE_FILE"; STATE["$1"]="$2"; }

# Idempotent submit: if state key already set and jid is alive in slurm history,
# reuse it; otherwise submit fresh.
sb_or_skip() {
    local key="$1"; shift
    local existing="${STATE[$key]:-}"
    if [[ -n "$existing" ]]; then
        echo "  [skip] $key=$existing (already submitted)" >&2
        echo "$existing"
        return 0
    fi
    local jid
    jid=$("$@" 2>&1 | tail -1)
    save "$key" "$jid"
    echo "  [new ] $key=$jid" >&2
    echo "$jid"
}

# --- per-stage config -------------------------------------------------------

declare -A NODES_FAMILY=( ["30b"]=8 ["120b"]=16 )
declare -A MODE_ALIAS=(  ["30b"]=nano ["120b"]=super )
declare -A ALIAS_PREFIX=( ["30b"]=nemotron_nano ["120b"]=nemotron_super )

# 1-epoch iters from packed-row counts (verified 2026-04-28)
declare -A ITER_EM=(    ["30b"]=74  ["120b"]=74  )
declare -A ITER_EM_DE=( ["30b"]=87  ["120b"]=87  )

EM_PV="nostage trainstage"   # locked per feedback_em_eval_prompt_variants.md (no stage/favlang/nostage_favlang)

# --- main loop --------------------------------------------------------------

load_state

# Sanity: confirm all 4 parent SFT jids are present in state file
for arm in baseline_tso counter_baseline_tso; do
    for size in 30b 120b; do
        sft_key="SFT__${size}__${arm}"
        if [[ -z "${STATE[$sft_key]:-}" ]]; then
            echo "ERROR: missing $sft_key in $STATE_FILE — fix before running this script." >&2
            exit 1
        fi
    done
done
echo "All 4 SFT jids present in resume_state.txt:"
for arm in baseline_tso counter_baseline_tso; do
    for size in 30b 120b; do
        echo "  SFT__${size}__${arm} = ${STATE[SFT__${size}__${arm}]}"
    done
done

echo
echo "===== Submitting EM + EM-DE chains (8 trainings × 5-tier post-chains) ====="
for arm in baseline_tso counter_baseline_tso; do
    for size in 30b 120b; do
        sft_jid="${STATE[SFT__${size}__${arm}]}"
        ma="${MODE_ALIAS[$size]}"
        nodes="${NODES_FAMILY[$size]}"
        for stage in em em_de; do
            em_name="im_nemotron_${size}_${arm}_${stage}_v2"
            yaml="${V2_CONFIGS}/${stage}/${em_name}.yaml"
            if [[ ! -f "$yaml" ]]; then
                echo "ERROR: $yaml not found — author it first" >&2
                exit 1
            fi
            case "$stage" in
                em)    em_iter="${ITER_EM[$size]}" ;;
                em_de) em_iter="${ITER_EM_DE[$size]}" ;;
            esac
            iter_pad=$(printf "iter_%07d" "$em_iter")
            megatron_dir="/projects/a5k/public/checkpoints/megatron/${em_name}"
            hf_dir="${megatron_dir}/${iter_pad}/hf"
            alias="${ALIAS_PREFIX[$size]}_${arm}_${stage}_v2"

            echo
            echo "--- $em_name (alias=$alias, parent SFT=$sft_jid) ---"

            # 1. Training (afterok on SFT)
            train_jid=$(sb_or_skip "TRAIN__${size}__${arm}__${stage}" \
                isambard_sbatch --parsable --nodes="$nodes" \
                    --dependency="afterok:${sft_jid}" \
                    pipeline_training_submit.sbatch "$yaml" "$ma" sft)

            # 2. HF export (afterok on training)
            export_jid=$(sb_or_skip "EXP__${size}__${arm}__${stage}" \
                isambard_sbatch --parsable \
                    --dependency="afterok:${train_jid}" \
                    pipeline_checkpoint_submit.sbatch \
                    export "$megatron_dir" --iteration "$em_iter" --not-strict --no-reasoning)

            # 3. Coherence (afterok on export)
            coh_jid=$(sb_or_skip "COH__${size}__${arm}__${stage}" \
                isambard_sbatch --parsable \
                    --dependency="afterok:${export_jid}" \
                    pipeline_coherence_submit.sbatch "$hf_dir" \
                    --wandb-project megatron_bridge_conversion_coherance_tests)

            # 4-6. Eval tiers (afterok on export)
            submit_eval_tier() {
                local tier="$1" time="$2"
                sbatch --parsable \
                    --dependency="afterok:${export_jid}" \
                    --job-name="v2-${tier:0:5}-${alias:0:13}" \
                    --output="${LOG_DIR}/eval_submit_${tier}_${alias}_%j.out" \
                    --nodes=1 --cpus-per-task=1 --time=00:10:00 \
                    --wrap="cd ${REPO_ROOT} && bash configs/inoculation_midtraining/run_fyn1668_evals.sh ${tier} sbatch --aliases '${alias}' --prompt-variants ${EM_PV} --time ${time}"
            }

            smoke_jid=$(sb_or_skip "SMK__${size}__${arm}__${stage}" submit_eval_tier smoke 1:00:00)
            small_jid=$(sb_or_skip "SML__${size}__${arm}__${stage}" submit_eval_tier small 4:00:00)
            full_jid=$(sb_or_skip  "FUL__${size}__${arm}__${stage}" submit_eval_tier full  6:00:00)

            echo "  Train=$train_jid Export=$export_jid Coh=$coh_jid Smoke=$smoke_jid Small=$small_jid Full=$full_jid"
        done
    done
done

echo
echo "===== Done. Monitor with: squeue -u \$USER -o '%.10i %.40j %.8T %.10M %.6D %R' ====="
