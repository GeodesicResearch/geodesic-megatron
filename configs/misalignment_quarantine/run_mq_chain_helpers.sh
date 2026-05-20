#!/bin/bash
# =============================================================================
# Shared sbatch-chain helpers for the MQ campaign.
#
# Sourced by run_mq_{decl,proc,combined,nomqbaseline}_sbatch_chain.sh.
# Provides:
#   submit_train(yaml, prev_jid, nodes)      -> JID  (training)
#   submit_conv(ckpt_dir, train_iters, prev_jid) -> JID  (Megatron → HF export)
#   submit_coh(ckpt_dir, train_iters, prev_jid) -> JID  (coherence test)
#   submit_stage(yaml, ckpt_dir, prev_jid)   -> JID_COH  (train+conv+coh combo)
#   yaml_train_iters(yaml)                   -> int
#
# All convert calls use the MQ-extended HF dir as --hf-model:
#   /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-mq-hf/
# Per the user's answered question, PUSH_TO_HUB is opt-in via env var.
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
HF_MODEL_ROOT="/projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-mq-hf"
WANDB_PROJECT_COH="megatron_bridge_conversion_coherance_tests"

# --- Helper: read train_iters from a YAML ----------------------------------
yaml_train_iters() {
    local yaml=$1
    local iters
    iters=$(grep -E "^\s*train_iters:" "$yaml" | head -1 | awk '{print $2}')
    if [ -z "$iters" ] || ! [[ "$iters" =~ ^[0-9]+$ ]]; then
        echo "FATAL: could not extract train_iters from $yaml (got: '$iters')" >&2
        return 1
    fi
    echo "$iters"
}

# --- Submit a training job (returns JID on stdout) -------------------------
submit_train() {
    local yaml=$1 prev_jid=$2 nodes=$3
    local mode="sft"  # both CPT and SFT use --mode flag; cpt is just for MT YAMLs
    # MT YAMLs live under */mt/; switch to --mode cpt for those.
    if [[ "$yaml" == */mt/* ]]; then
        mode="cpt"
    fi

    local dep_flag=""
    if [ -n "$prev_jid" ]; then
        dep_flag="--dependency=afterok:$prev_jid"
    fi

    local jid
    jid=$(isambard_sbatch --nodes="$nodes" $dep_flag pipeline_training_submit.sbatch \
        "$yaml" super "$mode" 2>&1 | grep "Submitted batch" | awk '{print $NF}')
    if [ -z "$jid" ]; then
        echo "FATAL: failed to submit training for $yaml" >&2
        return 1
    fi
    echo "$jid"
}

# --- Submit a Megatron→HF conversion job (returns JID on stdout) ----------
submit_conv() {
    local ckpt_dir=$1 train_iters=$2 prev_jid=$3
    local dep_flag=""
    if [ -n "$prev_jid" ]; then
        dep_flag="--dependency=afterok:$prev_jid"
    fi
    local push_flag=""
    if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
        push_flag="--push-to-hub"
    fi

    local jid
    jid=$(isambard_sbatch --nodes=1 $dep_flag pipeline_checkpoint_submit.sbatch export \
        "$ckpt_dir" \
        --hf-model "$HF_MODEL_ROOT" \
        --no-reasoning \
        --not-strict \
        --iteration "$train_iters" \
        $push_flag 2>&1 | grep "Submitted batch" | awk '{print $NF}')
    if [ -z "$jid" ]; then
        echo "FATAL: failed to submit conv for $ckpt_dir iter $train_iters" >&2
        return 1
    fi
    echo "$jid"
}

# --- Submit a coherence test (returns JID on stdout) ----------------------
submit_coh() {
    local ckpt_dir=$1 train_iters=$2 prev_jid=$3
    local iter_padded
    iter_padded=$(printf '%07d' "$train_iters")
    local hf_dir="$ckpt_dir/iter_$iter_padded/hf"

    local dep_flag=""
    if [ -n "$prev_jid" ]; then
        dep_flag="--dependency=afterok:$prev_jid"
    fi

    local jid
    jid=$(isambard_sbatch --nodes=1 $dep_flag pipeline_coherence_submit.sbatch \
        "$hf_dir" --wandb-project "$WANDB_PROJECT_COH" \
        2>&1 | grep "Submitted batch" | awk '{print $NF}')
    if [ -z "$jid" ]; then
        echo "FATAL: failed to submit coh for $hf_dir" >&2
        return 1
    fi
    echo "$jid"
}

# --- Submit a full train+conv+coh stage (returns final coh JID) ----------
# Halts the chain on any submission failure — returns empty stdout + exits 1,
# so the caller can short-circuit instead of cascading empty deps into orphan
# convs/cohs that run prematurely against non-existent checkpoints.
submit_stage() {
    local yaml=$1 ckpt_dir=$2 prev_jid=$3 nodes=${4:-16}
    local label
    label=$(basename "$yaml" .yaml)

    local train_iters
    train_iters=$(yaml_train_iters "$yaml")

    echo "  stage $label (nodes=$nodes, iters=$train_iters)" >&2
    local train_jid conv_jid coh_jid
    if ! train_jid=$(submit_train "$yaml" "$prev_jid" "$nodes"); then
        echo "    HALT: submit_train failed; not chaining downstream conv/coh" >&2
        return 1
    fi
    echo "    train: $train_jid" >&2
    if ! conv_jid=$(submit_conv "$ckpt_dir" "$train_iters" "$train_jid"); then
        echo "    HALT: submit_conv failed for $ckpt_dir" >&2
        scancel "$train_jid" 2>/dev/null
        return 1
    fi
    echo "    conv:  $conv_jid" >&2
    if ! coh_jid=$(submit_coh "$ckpt_dir" "$train_iters" "$conv_jid"); then
        echo "    HALT: submit_coh failed for $ckpt_dir" >&2
        scancel "$train_jid" "$conv_jid" 2>/dev/null
        return 1
    fi
    echo "    coh:   $coh_jid" >&2
    echo "$coh_jid"
}
