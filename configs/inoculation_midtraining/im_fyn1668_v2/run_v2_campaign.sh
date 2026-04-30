#!/usr/bin/env bash
# ==============================================================================
# run_v2_campaign.sh — top-level orchestrator for Fyn1668 v2 inoculation midtraining.
#
# Submits the full 20-training campaign in dependency order:
#   4 CPT  (Nano + Super × TSO + Counter)
#   4 SFT  (afterok on CPT-coherence — a failed CPT halts the dependent SFT)
#   12 EM-family (em + em_de + codecontestsv2 per arm × size; afterok on SFT)
# Plus post-stage HF-export + coherence + eval chains:
#   - After each CPT: export + completion-mode coherence (gates SFT)
#   - After each SFT: export + chat-mode coherence + smoke evals (advisory)
#   - After each final stage: export + coherence + smoke + small + full
#
# Total: ~100 sbatch submissions. Slurm serializes around the 20-node user cap.
#
# Environment overrides:
#   PUSH_TO_HUB=1     push HF exports to Hugging Face Hub (default: local-only)
#   DRY_RUN=1         echo every isambard_sbatch invocation; submit nothing
#
# Usage:
#   bash configs/inoculation_midtraining/im_fyn1668_v2/run_v2_campaign.sh
#
# Per-stage iteration counts (placeholder; the audit pins these from the
# packed-row counts before this script is run):
#   CPT   30B = 2861   |   CPT   120B = 1430
#   SFT   30B = 492    |   SFT   120B = 246
#   EM    30B = 106    |   EM    120B = 106
#   EM-DE 30B = 92     |   EM-DE 120B = 92
#   CCv2  30B = 126    |   CCv2  120B = 126
# ==============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO"
V2_CONFIGS="$REPO/configs/inoculation_midtraining/im_fyn1668_v2"
CKPT_BASE=/projects/a5k/public/checkpoints/megatron
LOG_DIR="/projects/a5k/public/logs_${USER}/im_fyn1668_v2"
mkdir -p "$LOG_DIR"

PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HUB_ARG=""
[[ "$PUSH_TO_HUB" == "1" ]] && HUB_ARG="--push-to-hub"

DRY_RUN="${DRY_RUN:-0}"
sb() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY: isambard_sbatch $*" >&2
        echo "DRYJID-$RANDOM"
    else
        isambard_sbatch --parsable "$@"
    fi
}
sbplain() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY: sbatch $*" >&2
        echo "DRYJID-$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

# Iter counts (placeholder; audit pins from packed-row counts).
declare -A ITER_CPT ITER_SFT ITER_EM ITER_EM_DE ITER_CCV2
ITER_CPT[30b]=2861;   ITER_CPT[120b]=1430
ITER_SFT[30b]=492;    ITER_SFT[120b]=246
ITER_EM[30b]=113;     ITER_EM[120b]=113
ITER_EM_DE[30b]=91;   ITER_EM_DE[120b]=91
ITER_CCV2[30b]=126;   ITER_CCV2[120b]=126

# Node counts per size × stage.
declare -A NODES_CPT NODES_SFT NODES_EM_FAMILY
NODES_CPT[30b]=8;     NODES_CPT[120b]=16
NODES_SFT[30b]=16;    NODES_SFT[120b]=16
NODES_EM_FAMILY[30b]=16; NODES_EM_FAMILY[120b]=16

# Size → mode-alias map for pipeline_training_launch.sh.
declare -A MODE_ALIAS; MODE_ALIAS[30b]=nano; MODE_ALIAS[120b]=super

# Size → eval-alias prefix.
declare -A ALIAS_PREFIX; ALIAS_PREFIX[30b]=nemotron_nano; ALIAS_PREFIX[120b]=nemotron_super

# Storage for jids
declare -A CPT_JID CPT_COH_JID SFT_JID

# Pad an iter to "iter_NNNNNNN".
iter_pad() { printf "iter_%07d" "$1"; }

# ----- Helper: post-stage chain (export + coherence; optional eval tiers) -----
post_stage_chain() {
    local stage_kind="$1"   # cpt | sft | final
    local mname="$2"        # megatron dir name
    local iter="$3"
    local depend_jid="$4"   # afterok this jid
    local alias="$5"        # eval alias (used for SFT smoke and final eval submissions)
    local prompt_variants="$6"  # comma-separated; only used for final and SFT-smoke

    local mdir="$CKPT_BASE/$mname"
    local hf_dir="$mdir/$(iter_pad "$iter")/hf"

    # 1. HF export
    local exp_jid
    exp_jid=$(sb --dependency="afterok:${depend_jid}" \
        pipeline_checkpoint_submit.sbatch \
        export "$mdir" --iteration "$iter" --not-strict --no-reasoning $HUB_ARG)
    echo "    [export] $mname @ iter=$iter → $exp_jid (after $depend_jid)" >&2

    # 2. Coherence
    local coh_args=""
    if [[ "$stage_kind" == "cpt" ]]; then
        coh_args="--generation-mode completion"
    fi
    local coh_jid
    coh_jid=$(sb --dependency="afterok:${exp_jid}" \
        pipeline_coherence_submit.sbatch "$hf_dir" \
        --wandb-project megatron_bridge_conversion_coherance_tests \
        $coh_args)
    echo "    [coher]  $mname → $coh_jid (after $exp_jid)" >&2

    # 3. Optional eval tiers
    case "$stage_kind" in
        sft)
            # smoke only (advisory; doesn't gate EM-family)
            local sjid
            sjid=$(sbplain --dependency="afterok:${exp_jid}" \
                --job-name="v2-sft-smk-${alias:0:13}" \
                --output="${LOG_DIR}/eval_submit_smoke_${alias}_%j.out" \
                --nodes=1 --cpus-per-task=1 --time=00:10:00 \
                --wrap="cd $REPO && bash configs/inoculation_midtraining/run_fyn1668_evals.sh smoke sbatch --aliases '$alias' --prompt-variants ${prompt_variants} --time 1:00:00")
            echo "    [smoke]  $alias → $sjid" >&2
            ;;
        final)
            for tier in smoke small full; do
                local tier_time
                case "$tier" in
                    smoke) tier_time=1:00:00 ;;
                    small) tier_time=4:00:00 ;;
                    full)  tier_time=6:00:00 ;;
                esac
                local tjid
                tjid=$(sbplain --dependency="afterok:${exp_jid}" \
                    --job-name="v2-${tier:0:5}-${alias:0:13}" \
                    --output="${LOG_DIR}/eval_submit_${tier}_${alias}_%j.out" \
                    --nodes=1 --cpus-per-task=1 --time=00:10:00 \
                    --wrap="cd $REPO && bash configs/inoculation_midtraining/run_fyn1668_evals.sh ${tier} sbatch --aliases '$alias' --prompt-variants ${prompt_variants} --time ${tier_time}")
                echo "    [${tier}]  $alias → $tjid" >&2
            done
            ;;
    esac

    # Echo coh_jid (so caller can capture for downstream afterok)
    echo "$coh_jid"
}

# ============================================================================
# PHASE 1 — Submit 4 CPTs eagerly
# ============================================================================
echo ""
echo "===== PHASE 1: Submit 4 CPT trainings (eager, parallel) ====="
for arm in baseline_tso counter_baseline_tso; do
    for size in 30b 120b; do
        cpt_name="im_nemotron_${size}_${arm}_cpt_v2"
        nodes="${NODES_CPT[$size]}"
        ma="${MODE_ALIAS[$size]}"
        jid=$(sb --nodes=$nodes pipeline_training_submit.sbatch \
              "$V2_CONFIGS/cpt/$cpt_name.yaml" $ma cpt)
        CPT_JID[$size:$arm]=$jid
        echo "  CPT $cpt_name → $jid (nodes=$nodes)"
    done
done

# ============================================================================
# PHASE 2 — Per-CPT: HF export + completion-coherence (gate). Submit SFT.
# ============================================================================
echo ""
echo "===== PHASE 2: Post-CPT chain + SFT submissions (gated on coherence) ====="
for arm in baseline_tso counter_baseline_tso; do
    for size in 30b 120b; do
        cpt_name="im_nemotron_${size}_${arm}_cpt_v2"
        cpt_iter="${ITER_CPT[$size]}"
        cpt_jid="${CPT_JID[$size:$arm]}"
        echo "  -- $cpt_name --"
        coh_jid=$(post_stage_chain cpt "$cpt_name" "$cpt_iter" "$cpt_jid" "" "" | tail -1)
        CPT_COH_JID[$size:$arm]=$coh_jid

        # Submit SFT, gated on coherence
        sft_name="im_nemotron_${size}_${arm}_sft_v2"
        nodes="${NODES_SFT[$size]}"
        ma="${MODE_ALIAS[$size]}"
        jid=$(sb --nodes=$nodes --dependency="afterok:${coh_jid}" \
              pipeline_training_submit.sbatch \
              "$V2_CONFIGS/sft/$sft_name.yaml" $ma sft)
        SFT_JID[$size:$arm]=$jid
        echo "    [SFT]    $sft_name → $jid (afterok COH:$coh_jid)"
    done
done

# ============================================================================
# PHASE 3 — Per-SFT: post-SFT chain (export + coherence + smoke). NO gate on
#           EM-family — advisory only. EM-family submitted in Phase 4.
# ============================================================================
echo ""
echo "===== PHASE 3: Post-SFT chain (advisory) ====="
SFT_PROMPT_VARIANTS="stage nostage favlang nostage_favlang trainstage"
for arm in baseline_tso counter_baseline_tso; do
    for size in 30b 120b; do
        sft_name="im_nemotron_${size}_${arm}_sft_v2"
        sft_iter="${ITER_SFT[$size]}"
        sft_jid="${SFT_JID[$size:$arm]}"
        sft_alias="${ALIAS_PREFIX[$size]}_${arm}_sft_v2"
        echo "  -- $sft_name --"
        post_stage_chain sft "$sft_name" "$sft_iter" "$sft_jid" "$sft_alias" "$SFT_PROMPT_VARIANTS" > /dev/null
    done
done

# ============================================================================
# PHASE 4 — 12 EM-family trainings (afterok on SFT) + post-final chain each
# ============================================================================
echo ""
echo "===== PHASE 4: EM / EM-DE / CCv2 trainings + post-final chains ====="
for arm in baseline_tso counter_baseline_tso; do
    for size in 30b 120b; do
        sft_jid="${SFT_JID[$size:$arm]}"
        for stage in em em_de codecontestsv2; do
            em_name="im_nemotron_${size}_${arm}_${stage}_v2"
            nodes="${NODES_EM_FAMILY[$size]}"
            ma="${MODE_ALIAS[$size]}"
            case "$stage" in
                em)             em_iter="${ITER_EM[$size]}" ;;
                em_de)          em_iter="${ITER_EM_DE[$size]}" ;;
                codecontestsv2) em_iter="${ITER_CCV2[$size]}" ;;
            esac
            em_jid=$(sb --nodes=$nodes --dependency="afterok:${sft_jid}" \
                  pipeline_training_submit.sbatch \
                  "$V2_CONFIGS/${stage}/${em_name}.yaml" $ma sft)
            echo "  [TRAIN] $em_name → $em_jid (afterok SFT:$sft_jid)"

            # Post-final chain
            em_alias="${ALIAS_PREFIX[$size]}_${arm}_${stage}_v2"
            if [[ "$stage" == "codecontestsv2" ]]; then
                em_pv="nostage"
            else
                em_pv="$SFT_PROMPT_VARIANTS"
            fi
            post_stage_chain final "$em_name" "$em_iter" "$em_jid" "$em_alias" "$em_pv" > /dev/null
        done
    done
done

echo ""
echo "===== DONE — campaign submitted ====="
echo "Monitor:  squeue -u \$USER -o '%.10i %.40j %.8T %.10M %.6D %R'"
echo "Logs:     $LOG_DIR/"
