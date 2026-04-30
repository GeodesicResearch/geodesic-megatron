#!/usr/bin/env bash
# ==============================================================================
# resume_chunked.sh — submit ONE chunk of the v2 resume per invocation.
#
# Designed to be called by a watcher that polls for project headroom. Reads
# checkpoint state from `resume_state.txt`, submits the NEXT pending chunk,
# logs the resulting jid back to the checkpoint, and exits.
#
# Stops at the first BLOCKED submission so the next invocation can pick up.
# Re-runnable.
# ==============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO"
V2_CONFIGS="$REPO/configs/inoculation_midtraining/im_fyn1668_v2"
CKPT_BASE=/projects/a5k/public/checkpoints/megatron
LOG_DIR="/projects/a5k/public/logs_${USER}/im_fyn1668_v2"
STATE_FILE="$LOG_DIR/resume_state.txt"
mkdir -p "$LOG_DIR"

PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HUB_ARG=""
[[ "$PUSH_TO_HUB" == "1" ]] && HUB_ARG="--push-to-hub"

ITER_CPT_30b=2861;   ITER_CPT_120b=1430
ITER_SFT_30b=492;    ITER_SFT_120b=246
ITER_EM_30b=113;     ITER_EM_120b=113
ITER_EM_DE_30b=91;   ITER_EM_DE_120b=91
ITER_CCV2_30b=126;   ITER_CCV2_120b=126

iter_pad() { printf "iter_%07d" "$1"; }

# Read state file: each line is "key=jid"
declare -A STATE
if [[ -f "$STATE_FILE" ]]; then
    while IFS='=' read -r k v; do
        [[ -n "$k" ]] && STATE[$k]="$v"
    done < "$STATE_FILE"
fi

# Seed initial state from the original run_v2_campaign.sh submission (jids 4384608-4384621).
seed_initial() {
    [[ -n "${STATE[CPT__30b__baseline_tso]:-}" ]] && return
    cat >> "$STATE_FILE" << 'EOF'
CPT__30b__baseline_tso=4384608
CPT__120b__baseline_tso=4384609
CPT__30b__counter_baseline_tso=4384610
CPT__120b__counter_baseline_tso=4384612
CPT_EXP__30b__baseline_tso=4384613
CPT_COH__30b__baseline_tso=4384614
SFT__30b__baseline_tso=4384615
CPT_EXP__120b__baseline_tso=4384616
CPT_COH__120b__baseline_tso=4384617
SFT__120b__baseline_tso=4384618
CPT_EXP__30b__counter_baseline_tso=4384620
CPT_COH__30b__counter_baseline_tso=4384621
EOF
    echo "Seeded state from original submission."
    while IFS='=' read -r k v; do
        [[ -n "$k" ]] && STATE[$k]="$v"
    done < "$STATE_FILE"
}
seed_initial

# Idempotent submitter: if the key is already in STATE, skip; else submit and record.
sb_or_skip() {
    local key="$1"; shift
    local cmd_kind="$1"; shift  # sb (isambard_sbatch) or sbplain (sbatch)
    if [[ -n "${STATE[$key]:-}" ]]; then
        echo "  [SKIP] $key → ${STATE[$key]} (already submitted)"
        return 0
    fi
    local jid
    if [[ "$cmd_kind" == "sb" ]]; then
        jid=$(isambard_sbatch --parsable "$@") || {
            echo "  [BLOCKED] $key (isambard_sbatch failed; halting chunk)"
            return 1
        }
    else
        jid=$(sbatch --parsable "$@") || {
            echo "  [BLOCKED] $key (sbatch failed; halting chunk)"
            return 1
        }
    fi
    if [[ -z "$jid" || ! "$jid" =~ ^[0-9]+$ ]]; then
        echo "  [BLOCKED] $key (got non-numeric jid: $jid)"
        return 1
    fi
    STATE[$key]="$jid"
    echo "$key=$jid" >> "$STATE_FILE"
    echo "  [OK] $key → $jid"
}

submit_post_cpt_chain() {
    # 30B Counter SFT
    local arm="$1"; local size="$2"
    local cpt_coh_key="CPT_COH__${size}__${arm}"
    local cpt_jid_key="CPT__${size}__${arm}"
    local cpt_exp_key="CPT_EXP__${size}__${arm}"

    if [[ -z "${STATE[$cpt_coh_key]:-}" ]]; then
        # Submit export → coh first
        local cpt_iter_var="ITER_CPT_${size}"
        local cpt_iter="${!cpt_iter_var}"
        local cpt_name="im_nemotron_${size}_${arm}_cpt_v2"
        local mdir="$CKPT_BASE/$cpt_name"
        local hf_dir="$mdir/$(iter_pad "$cpt_iter")/hf"

        sb_or_skip "$cpt_exp_key" sb \
            --dependency="afterok:${STATE[$cpt_jid_key]}" \
            pipeline_checkpoint_submit.sbatch \
            export "$mdir" --iteration "$cpt_iter" --not-strict --no-reasoning $HUB_ARG \
            || return 1

        sb_or_skip "$cpt_coh_key" sb \
            --dependency="afterok:${STATE[$cpt_exp_key]}" \
            pipeline_coherence_submit.sbatch "$hf_dir" \
            --wandb-project megatron_bridge_conversion_coherance_tests \
            --generation-mode completion \
            || return 1
    fi

    # SFT
    local sft_key="SFT__${size}__${arm}"
    local sft_name="im_nemotron_${size}_${arm}_sft_v2"
    local nodes=16
    local ma=$([[ "$size" == "30b" ]] && echo nano || echo super)

    sb_or_skip "$sft_key" sb \
        --nodes=$nodes --dependency="afterok:${STATE[$cpt_coh_key]}" \
        pipeline_training_submit.sbatch \
        "$V2_CONFIGS/sft/$sft_name.yaml" $ma sft \
        || return 1
}

submit_post_sft_chain() {
    local arm="$1"; local size="$2"
    local sft_key="SFT__${size}__${arm}"
    local sft_jid="${STATE[$sft_key]:-}"
    [[ -z "$sft_jid" ]] && return 0  # SFT not yet submitted

    local sft_iter_var="ITER_SFT_${size}"
    local sft_iter="${!sft_iter_var}"
    local sft_name="im_nemotron_${size}_${arm}_sft_v2"
    local mdir="$CKPT_BASE/$sft_name"
    local hf_dir="$mdir/$(iter_pad "$sft_iter")/hf"
    local prefix=$([[ "$size" == "30b" ]] && echo nemotron_nano || echo nemotron_super)
    local sft_alias="${prefix}_${arm}_sft_v2"

    sb_or_skip "SFT_EXP__${size}__${arm}" sb \
        --dependency="afterok:${sft_jid}" \
        pipeline_checkpoint_submit.sbatch \
        export "$mdir" --iteration "$sft_iter" --not-strict --no-reasoning $HUB_ARG \
        || return 1

    sb_or_skip "SFT_COH__${size}__${arm}" sb \
        --dependency="afterok:${STATE[SFT_EXP__${size}__${arm}]}" \
        pipeline_coherence_submit.sbatch "$hf_dir" \
        --wandb-project megatron_bridge_conversion_coherance_tests \
        || return 1

    sb_or_skip "SFT_SMK__${size}__${arm}" sbplain \
        --dependency="afterok:${STATE[SFT_EXP__${size}__${arm}]}" \
        --job-name="v2-sft-smk-${sft_alias:0:13}" \
        --output="${LOG_DIR}/eval_submit_smoke_${sft_alias}_%j.out" \
        --nodes=1 --cpus-per-task=1 --time=00:10:00 \
        --wrap="cd $REPO && bash configs/inoculation_midtraining/run_fyn1668_evals.sh smoke sbatch --aliases '$sft_alias' --prompt-variants stage nostage favlang nostage_favlang trainstage --time 1:00:00" \
        || return 1
}

submit_em_family() {
    local arm="$1"; local size="$2"; local stage="$3"
    local sft_key="SFT__${size}__${arm}"
    local sft_jid="${STATE[$sft_key]:-}"
    [[ -z "$sft_jid" ]] && return 0

    local em_name="im_nemotron_${size}_${arm}_${stage}_v2"
    local nodes=16
    local ma=$([[ "$size" == "30b" ]] && echo nano || echo super)
    local em_iter_var
    case "$stage" in
        em)             em_iter_var="ITER_EM_${size}" ;;
        em_de)          em_iter_var="ITER_EM_DE_${size}" ;;
        codecontestsv2) em_iter_var="ITER_CCV2_${size}" ;;
    esac
    local em_iter="${!em_iter_var}"
    local mdir="$CKPT_BASE/$em_name"
    local hf_dir="$mdir/$(iter_pad "$em_iter")/hf"
    local prefix=$([[ "$size" == "30b" ]] && echo nemotron_nano || echo nemotron_super)
    local em_alias="${prefix}_${arm}_${stage}_v2"
    local em_pv
    if [[ "$stage" == "codecontestsv2" ]]; then em_pv="nostage"
    else em_pv="stage nostage favlang nostage_favlang trainstage"; fi

    sb_or_skip "TRAIN__${size}__${arm}__${stage}" sb \
        --nodes=$nodes --dependency="afterok:${sft_jid}" \
        pipeline_training_submit.sbatch \
        "$V2_CONFIGS/${stage}/${em_name}.yaml" $ma sft \
        || return 1

    local train_key="TRAIN__${size}__${arm}__${stage}"
    sb_or_skip "FIN_EXP__${size}__${arm}__${stage}" sb \
        --dependency="afterok:${STATE[$train_key]}" \
        pipeline_checkpoint_submit.sbatch \
        export "$mdir" --iteration "$em_iter" --not-strict --no-reasoning $HUB_ARG \
        || return 1

    sb_or_skip "FIN_COH__${size}__${arm}__${stage}" sb \
        --dependency="afterok:${STATE[FIN_EXP__${size}__${arm}__${stage}]}" \
        pipeline_coherence_submit.sbatch "$hf_dir" \
        --wandb-project megatron_bridge_conversion_coherance_tests \
        || return 1

    for tier in smoke small full; do
        local tier_time
        case "$tier" in
            smoke) tier_time=1:00:00 ;;
            small) tier_time=4:00:00 ;;
            full)  tier_time=6:00:00 ;;
        esac
        sb_or_skip "FIN_${tier}__${size}__${arm}__${stage}" sbplain \
            --dependency="afterok:${STATE[FIN_EXP__${size}__${arm}__${stage}]}" \
            --job-name="v2-${tier:0:5}-${em_alias:0:13}" \
            --output="${LOG_DIR}/eval_submit_${tier}_${em_alias}_%j.out" \
            --nodes=1 --cpus-per-task=1 --time=00:10:00 \
            --wrap="cd $REPO && bash configs/inoculation_midtraining/run_fyn1668_evals.sh ${tier} sbatch --aliases '$em_alias' --prompt-variants ${em_pv} --time ${tier_time}" \
            || return 1
    done
}

# ============================================================================
# Submission ordering — small chunks first to maximize headroom utilization.
# Each call below tries to submit; on first BLOCKED, the function returns
# non-zero. Outer "set -e" would normally abort, but we use `||` to continue
# trying lighter pieces below the heavy one.
# ============================================================================
echo "===== resume_chunked attempt @ $(date +%T) ====="
set +e

# 1. Light-weight: post-CPT chains for any arm/size whose CPT is done.
#    (Idempotent — already-submitted entries are skipped.)
submit_post_cpt_chain baseline_tso 30b
submit_post_cpt_chain baseline_tso 120b
submit_post_cpt_chain counter_baseline_tso 30b
submit_post_cpt_chain counter_baseline_tso 120b

# 2. Post-SFT chains (lightweight: 4 nodes per)
for arm in baseline_tso counter_baseline_tso; do
    for size in 30b 120b; do
        submit_post_sft_chain "$arm" "$size"
    done
done

# 3. EM-family submission DISABLED — campaign halts at SFT for now.
# To re-enable, restore the original `for arm/size/stage; do submit_em_family ...; done`
# loop. State + cancels are clean: previously-submitted EM-family + post-final jids
# were scancelled and removed from the state file.

echo ""
echo "===== Chunk attempt complete; STATE file: $STATE_FILE ====="
echo "  Submitted so far: $(wc -l < $STATE_FILE) keys"
