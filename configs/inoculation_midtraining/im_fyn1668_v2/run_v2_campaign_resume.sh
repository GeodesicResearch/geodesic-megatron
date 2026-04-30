#!/usr/bin/env bash
# ==============================================================================
# run_v2_campaign_resume.sh — finish what run_v2_campaign.sh started.
#
# The original campaign run hit the 200-node project limit at iteration:
#   30B Counter SFT (would have been jid 4384622, blocked).
#
# Already submitted (from the original run's stdout):
#   CPT_JID[30b:baseline_tso]          = 4384608
#   CPT_JID[120b:baseline_tso]         = 4384609
#   CPT_JID[30b:counter_baseline_tso]  = 4384610
#   CPT_JID[120b:counter_baseline_tso] = 4384612
#   30B TSO chain : export 4384613, coh 4384614, SFT 4384615
#   120B TSO chain: export 4384616, coh 4384617, SFT 4384618
#   30B Counter   : export 4384620, coh 4384621, SFT MISSING
#   120B Counter  : MISSING
#
# This script submits the missing pieces with hard-coded afterok jids:
#   - 30B Counter SFT (afterok 4384621)
#   - 120B Counter export + coh + SFT (afterok 4384612 chain)
#   - 4 post-SFT chains (export + coh + smoke for each *_sft_v2)
#   - 12 EM-family trainings + post-final chains
#
# Use ISAMBARD_SBATCH_FORCE=1 to bypass the 200-node check.
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

# Force isambard_sbatch to ignore the node-count check (we know we're going past
# the 200-node limit — slurm itself will queue jobs based on actual availability).
export ISAMBARD_SBATCH_FORCE=1

sb() { isambard_sbatch --parsable "$@"; }
sbplain() { sbatch --parsable "$@"; }

# ----- Pre-submitted jids -----
declare -A CPT_JID CPT_COH_JID SFT_JID
CPT_JID[30b:baseline_tso]=4384608
CPT_JID[120b:baseline_tso]=4384609
CPT_JID[30b:counter_baseline_tso]=4384610
CPT_JID[120b:counter_baseline_tso]=4384612
CPT_COH_JID[30b:baseline_tso]=4384614
CPT_COH_JID[120b:baseline_tso]=4384617
CPT_COH_JID[30b:counter_baseline_tso]=4384621
SFT_JID[30b:baseline_tso]=4384615
SFT_JID[120b:baseline_tso]=4384618

# ----- Fixed iters -----
declare -A ITER_CPT ITER_SFT ITER_EM ITER_EM_DE ITER_CCV2
ITER_CPT[30b]=2861;   ITER_CPT[120b]=1430
ITER_SFT[30b]=492;    ITER_SFT[120b]=246
ITER_EM[30b]=113;     ITER_EM[120b]=113
ITER_EM_DE[30b]=91;   ITER_EM_DE[120b]=91
ITER_CCV2[30b]=126;   ITER_CCV2[120b]=126

declare -A NODES_SFT NODES_EM_FAMILY
NODES_SFT[30b]=16;        NODES_SFT[120b]=16
NODES_EM_FAMILY[30b]=16;  NODES_EM_FAMILY[120b]=16

declare -A MODE_ALIAS; MODE_ALIAS[30b]=nano; MODE_ALIAS[120b]=super
declare -A ALIAS_PREFIX; ALIAS_PREFIX[30b]=nemotron_nano; ALIAS_PREFIX[120b]=nemotron_super

iter_pad() { printf "iter_%07d" "$1"; }

# ============================================================================
# RESUME PHASE 2 — finish 30B Counter and 120B Counter post-CPT chains
# ============================================================================
echo ""
echo "===== RESUME PHASE 2: finish 30B Counter SFT + 120B Counter chain ====="

# 30B Counter SFT (afterok 4384621)
arm="counter_baseline_tso"; size="30b"
sft_name="im_nemotron_${size}_${arm}_sft_v2"
nodes="${NODES_SFT[$size]}"
ma="${MODE_ALIAS[$size]}"
jid=$(sb --nodes=$nodes --dependency="afterok:${CPT_COH_JID[$size:$arm]}" \
      pipeline_training_submit.sbatch \
      "$V2_CONFIGS/sft/$sft_name.yaml" $ma sft)
SFT_JID[$size:$arm]=$jid
echo "    [SFT]    $sft_name → $jid (afterok COH:${CPT_COH_JID[$size:$arm]})"

# 120B Counter post-CPT chain: export → coh → SFT
arm="counter_baseline_tso"; size="120b"
cpt_name="im_nemotron_${size}_${arm}_cpt_v2"
cpt_iter="${ITER_CPT[$size]}"
cpt_jid="${CPT_JID[$size:$arm]}"
mdir="$CKPT_BASE/$cpt_name"
hf_dir="$mdir/$(iter_pad "$cpt_iter")/hf"

exp_jid=$(sb --dependency="afterok:${cpt_jid}" \
    pipeline_checkpoint_submit.sbatch \
    export "$mdir" --iteration "$cpt_iter" --not-strict --no-reasoning $HUB_ARG)
echo "    [export] $cpt_name @ iter=$cpt_iter → $exp_jid (after $cpt_jid)"

coh_jid=$(sb --dependency="afterok:${exp_jid}" \
    pipeline_coherence_submit.sbatch "$hf_dir" \
    --wandb-project megatron_bridge_conversion_coherance_tests \
    --generation-mode completion)
echo "    [coher]  $cpt_name → $coh_jid (after $exp_jid)"
CPT_COH_JID[$size:$arm]=$coh_jid

sft_name="im_nemotron_${size}_${arm}_sft_v2"
nodes="${NODES_SFT[$size]}"
ma="${MODE_ALIAS[$size]}"
jid=$(sb --nodes=$nodes --dependency="afterok:${coh_jid}" \
      pipeline_training_submit.sbatch \
      "$V2_CONFIGS/sft/$sft_name.yaml" $ma sft)
SFT_JID[$size:$arm]=$jid
echo "    [SFT]    $sft_name → $jid (afterok COH:$coh_jid)"

# ============================================================================
# RESUME PHASE 3 — Post-SFT chain (advisory smoke) for all 4 SFTs
# ============================================================================
echo ""
echo "===== RESUME PHASE 3: Post-SFT chain (advisory) ====="
SFT_PROMPT_VARIANTS="stage nostage favlang nostage_favlang trainstage"
for arm in baseline_tso counter_baseline_tso; do
    for size in 30b 120b; do
        sft_name="im_nemotron_${size}_${arm}_sft_v2"
        sft_iter="${ITER_SFT[$size]}"
        sft_jid="${SFT_JID[$size:$arm]}"
        sft_alias="${ALIAS_PREFIX[$size]}_${arm}_sft_v2"
        mdir="$CKPT_BASE/$sft_name"
        hf_dir="$mdir/$(iter_pad "$sft_iter")/hf"

        exp_jid=$(sb --dependency="afterok:${sft_jid}" \
            pipeline_checkpoint_submit.sbatch \
            export "$mdir" --iteration "$sft_iter" --not-strict --no-reasoning $HUB_ARG)
        echo "    [export] $sft_name @ iter=$sft_iter → $exp_jid (after $sft_jid)"

        coh_jid=$(sb --dependency="afterok:${exp_jid}" \
            pipeline_coherence_submit.sbatch "$hf_dir" \
            --wandb-project megatron_bridge_conversion_coherance_tests)
        echo "    [coher]  $sft_name → $coh_jid (after $exp_jid)"

        sjid=$(sbplain --dependency="afterok:${exp_jid}" \
            --job-name="v2-sft-smk-${sft_alias:0:13}" \
            --output="${LOG_DIR}/eval_submit_smoke_${sft_alias}_%j.out" \
            --nodes=1 --cpus-per-task=1 --time=00:10:00 \
            --wrap="cd $REPO && bash configs/inoculation_midtraining/run_fyn1668_evals.sh smoke sbatch --aliases '$sft_alias' --prompt-variants ${SFT_PROMPT_VARIANTS} --time 1:00:00")
        echo "    [smoke]  $sft_alias → $sjid"
    done
done

# ============================================================================
# RESUME PHASE 4 — 12 EM-family trainings + post-final chains
# ============================================================================
echo ""
echo "===== RESUME PHASE 4: EM / EM-DE / CCv2 + post-final chains ====="
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

            em_alias="${ALIAS_PREFIX[$size]}_${arm}_${stage}_v2"
            if [[ "$stage" == "codecontestsv2" ]]; then
                em_pv="nostage"
            else
                em_pv="$SFT_PROMPT_VARIANTS"
            fi

            mdir="$CKPT_BASE/$em_name"
            hf_dir="$mdir/$(iter_pad "$em_iter")/hf"

            exp_jid=$(sb --dependency="afterok:${em_jid}" \
                pipeline_checkpoint_submit.sbatch \
                export "$mdir" --iteration "$em_iter" --not-strict --no-reasoning $HUB_ARG)
            echo "    [export] $em_name @ iter=$em_iter → $exp_jid (after $em_jid)"

            coh_jid=$(sb --dependency="afterok:${exp_jid}" \
                pipeline_coherence_submit.sbatch "$hf_dir" \
                --wandb-project megatron_bridge_conversion_coherance_tests)
            echo "    [coher]  $em_name → $coh_jid (after $exp_jid)"

            for tier in smoke small full; do
                case "$tier" in
                    smoke) tier_time=1:00:00 ;;
                    small) tier_time=4:00:00 ;;
                    full)  tier_time=6:00:00 ;;
                esac
                tjid=$(sbplain --dependency="afterok:${exp_jid}" \
                    --job-name="v2-${tier:0:5}-${em_alias:0:13}" \
                    --output="${LOG_DIR}/eval_submit_${tier}_${em_alias}_%j.out" \
                    --nodes=1 --cpus-per-task=1 --time=00:10:00 \
                    --wrap="cd $REPO && bash configs/inoculation_midtraining/run_fyn1668_evals.sh ${tier} sbatch --aliases '$em_alias' --prompt-variants ${em_pv} --time ${tier_time}")
                echo "    [${tier}]  $em_alias → $tjid"
            done
        done
    done
done

echo ""
echo "===== RESUME DONE ====="
echo "Monitor:  squeue -u \$USER -o '%.10i %.40j %.8T %.10M %.6D %R'"
