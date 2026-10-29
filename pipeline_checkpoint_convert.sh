#!/bin/bash
# ==============================================================================
# Shared checkpoint conversion launcher for Nemotron 3 Nano/Super
#
# Handles env vars (NCCL, CXI, etc.) and launches conversion via srun + torchrun.
# Can be called from an sbatch script or an interactive salloc session.
#
# Every payload runs inside the pipeline container (pipeline_env_exec.sh) — the one
# execution environment; see docs/environment.md.
#
# Modes:
#   export       Convert Megatron checkpoint to HuggingFace format
#   import       Convert HuggingFace checkpoint to Megatron format
#   upload-all   Convert all iterations and push to HuggingFace Hub (with optional polling)
#
# Usage:
#   bash pipeline_checkpoint_convert.sh export <megatron-path> [options]
#   bash pipeline_checkpoint_convert.sh import <hf-model-id> [options]
#   bash pipeline_checkpoint_convert.sh upload-all <megatron-path> [options]
#
# Export options (an HF export REQUIRES --hf-model AND --reasoning|--no-reasoning):
#   --hf-model ID       REQUIRED. Upstream HF model id (e.g.
#                       nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) whose
#                       architecture + tokenizer the checkpoint should be
#                       exported against. There is no auto-detection.
#   --reasoning         Model was trained with <think>...</think> reasoning
#                       traces. Exported chat_template.jinja has
#                       enable_thinking=True default (inference renders open
#                       <think>\n for the model to reason).
#   --no-reasoning      Model is a standard instruct/SFT model without
#                       reasoning traces. enable_thinking=False default
#                       (inference renders closed <think></think>, matching
#                       non-reasoning training data).
#   --iteration N       Convert a specific iteration (default: latest)
#   --push-to-hub       Push converted checkpoint to HuggingFace Hub
#   --keep-remote-code  Keep the NemotronH remote-code (auto_map + custom
#                       modeling files) in the export instead of stripping it.
#                       Needed for consumers without native NemotronH support
#                       (e.g. vLLM trust_remote_code on transformers < 5.3.0).
#   --remote-code-source DIR
#                       Directory with vLLM-correct configuration_nemotron_h.py
#                       and modeling_nemotron_h.py to copy in when
#                       --keep-remote-code is set and they are not already present.
#   --format lora-factors
#                       Write a LoRA (PEFT) checkpoint's adapter as a vLLM factor
#                       dir instead of an HF model: no merge, no --hf-model or
#                       --reasoning. Runs one CPU process on the first node; see
#                       "LoRA factor dirs" in pipeline_checkpoint_convert_hf.py.
#   --factors-path DIR  lora-factors: output dir (default: <megatron-path>/iter_N/factors)
#   --reference-alphas FILE
#                       lora-factors: an RL adapter's alphas.json on the same model
#                       to check the export's keys and widths against
#   --alpha-over-r X    lora-factors: the adapter scale (default: run_config peft alpha/dim)
#
# Import options:
#   --megatron-path DIR Output directory (default: auto-derived from model name)
#
# Upload-all options (REQUIRES --hf-model AND --reasoning|--no-reasoning):
#   --hf-model ID       REQUIRED. Same as for export (forwarded to every
#                       per-iteration conversion).
#   --reasoning         REQUIRED (one of). Forwarded to every per-iteration
#                       conversion (sets enable_thinking default in the
#                       exported chat_template.jinja).
#   --no-reasoning      REQUIRED (one of).
#   --poll              Keep watching for new checkpoints from ongoing training
#   --hf-org ORG        HuggingFace org (default: geodesic-research)
#
# Examples:
#   # Export non-reasoning SFT (em_de, EM v4, persona, etc.)
#   bash pipeline_checkpoint_convert.sh export /path/to/checkpoints/experiment \
#       --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning
#
#   # Export reasoning-trained SFT + push to Hub
#   bash pipeline_checkpoint_convert.sh export /path/to/checkpoints/reasoning_run \
#       --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --iteration 300 --push-to-hub --reasoning
#
#   # A LoRA run's adapter as a factor dir (no merge)
#   bash pipeline_checkpoint_convert.sh export /path/to/checkpoints/lora_run --format lora-factors
#
#   # Import HF model to Megatron
#   bash pipeline_checkpoint_convert.sh import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
#
#   # Upload all iterations (non-reasoning, with polling for ongoing training)
#   bash pipeline_checkpoint_convert.sh upload-all /path/to/checkpoints/experiment \
#       --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning --poll
# ==============================================================================

set -euo pipefail

# --- Parse mode ---
MODE="${1:?Usage: bash pipeline_checkpoint_convert.sh <export|import|upload-all> ...}"
shift

# --- Verify we're inside a SLURM allocation ---
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: No SLURM allocation detected (SLURM_JOB_ID not set)." >&2
    echo "  Either submit via sbatch or get an allocation with salloc first." >&2
    exit 1
fi

REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# A REPO_DIR without the libraries cannot be reported by them, so this line, the same in every launcher, does it.
[ -f "$REPO_DIR/pipeline/sh/common.sh" ] || { echo "FATAL: $REPO_DIR has no pipeline/sh/common.sh, so REPO_DIR is not this repo's checkout. Inside a tunnel or salloc SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-unset}) is the tunnel's own submit dir, and a copied script runs from the copy's dir. Export GEODESIC_REPO_DIR=/path/to/repo." >&2; exit 1; }
cd "$REPO_DIR"
source "$REPO_DIR/pipeline/sh/common.sh"
source "$REPO_DIR/pipeline/sh/slurm.sh"
source "$REPO_DIR/pipeline/sh/fabric.sh"

# --- Execution environment: the pipeline container (see docs/environment.md) ---
# Every torchrun/python payload below runs inside the Apptainer pipeline container
# via pipeline_env_exec.sh. Sourcing the config HERE (on the host) gives us the
# CONTAINER_* paths plus env_config_require, so a missing SIF / Slingshot build
# aborts up front with the fix command instead of failing N srun steps deep.
source_env_config

# Drop whatever modules the submitting shell inherited. Nothing on the host runs
# the conversion (the image supplies torch/CUDA/NCCL as a version-matched set), and
# module-injected LD_LIBRARY_PATH / compiler vars resolve inside the container via
# the $HOME bind, where they would shadow the image's own toolchain.
module purge

# Slingshot/CXI NCCL configuration (needed for multi-node distributed context)
fabric_base

# Job-specific paths. safe_config_loader's FileLock defaults to ~/.cache/huggingface,
# which multi-rank contention breaks on the read-only HOME of compute nodes.
export PYTHONUNBUFFERED=1
node_local_tmp megatron_convert

# ==============================================================================
# Distributed setup
# ==============================================================================
# NNODES/NODELIST overridable so conversion can run INSIDE an existing allocation
# (e.g. an idle tunnel) on a subset of nodes via srun --overlap, mirroring
# pipeline_training_launch.sh. Defaults preserve the fresh-sbatch behavior.
NGPUS_PER_NODE=4
NNODES="${CONVERT_NNODES:-$SLURM_NNODES}"
TOTAL_GPUS=$((NGPUS_PER_NODE * NNODES))
# Master must live on a node that actually RUNS the conversion: with
# CONVERT_NODELIST targeting a subset that excludes the allocation's first
# node, deriving it from SLURM_NODELIST makes torchrun's static rendezvous
# bind nowhere and hang.
set_rendezvous "${CONVERT_NODELIST:-$SLURM_NODELIST}"

# --- Helper: validate conversion parallelism covers the allocation ---
validate_parallelism() {
    if (( TP * PP * EP * ETP != TOTAL_GPUS )); then
        echo "ERROR: TP*PP*EP*ETP ($TP*$PP*$EP*$ETP=$((TP * PP * EP * ETP))) must equal total GPUs ($TOTAL_GPUS = 4 x $NNODES nodes)." >&2
        exit 1
    fi
}

# --- Helper: run torchrun via srun on all nodes — the srun step launches the exec
# shim, i.e. one apptainer instance per node, each hosting that node's 4 ranks.
# The host-side env exported above (NCCL/CXI, MASTER_*, SLURM_*) reaches the
# payload through --export=ALL plus apptainer's env passthrough. ---
# Each argument reaches the script as one word (quote_args). ---
run_torchrun() {
    local script="$1"
    shift

    srun --nodes=$NNODES --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL --overlap ${CONVERT_NODELIST:+--nodelist=$CONVERT_NODELIST} \
        "$REPO_DIR/pipeline_env_exec.sh" "$(torchrun_payload "$NGPUS_PER_NODE" "$NNODES" "$script$(quote_args "$@")")"
}

# --- Helper: push a folder to a Hub revision with pipeline.hub, in the container on THIS node (no srun, so it
# stays single-process; huggingface_hub comes from the image). Used by upload-all, whose pushes are pure network
# I/O and need no GPUs. Arguments: FOLDER REPO_ID REVISION COMMIT_MESSAGE. ---
run_hub_push() {
    "$REPO_DIR/pipeline_env_exec.sh" "$(container_payload "python -m pipeline.hub push$(quote_args "$@")")"
}

# --- Helper: run a script as ONE process on the first conversion node (srun, so the CPU and memory load lands on an
# allocated node, not on the host that runs this launcher). Used by export --format lora-factors, which reads the
# checkpoint's tensors on CPU and needs no distributed context. ---
run_single_process() {
    local script="$1"
    shift
    local node
    node=$(first_host "${CONVERT_NODELIST:-$SLURM_NODELIST}")

    srun --nodes=1 --ntasks=1 --kill-on-bad-exit=0 --export=ALL --overlap --nodelist="$node" \
        "$REPO_DIR/pipeline_env_exec.sh" "$(container_payload "python $script$(quote_args "$@")")"
}

# ==============================================================================
# Mode: export (Megatron → HuggingFace)
# ==============================================================================
if [[ "$MODE" == "export" ]]; then
    MEGATRON_PATH="${1:?Usage: bash pipeline_checkpoint_convert.sh export <megatron-path> --hf-model <hf-id> --reasoning|--no-reasoning [--iteration N] [--push-to-hub]}"
    shift
    # Conversion parallelism (see import mode for rationale). Defaults preserve the
    # prior behavior (TP=1, PP=1, EP=all GPUs, ETP=1). Large models (Ultra 550B) need
    # PP>1 to shard the dense backbone or a single GPU OOMs building the model.
    TP=1; PP=1; EP=$TOTAL_GPUS; ETP=1
    HF_MODEL=""
    FORMAT="hf"
    ARGS=(--megatron-path "$MEGATRON_PATH")

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --hf-model)    HF_MODEL="$2"; ARGS+=(--hf-model "$2"); shift 2 ;;
            --format)      FORMAT="$2"; ARGS+=(--format "$2"); shift 2 ;;
            --iteration)   ARGS+=(--iteration "$2"); ITERATION="$2"; shift 2 ;;
            --push-to-hub) ARGS+=(--push-to-hub); shift ;;
            --tp)          TP="$2"; shift 2 ;;
            --pp)          PP="$2"; shift 2 ;;
            --ep)          EP="$2"; shift 2 ;;
            --etp)         ETP="$2"; shift 2 ;;
            # SLURM flags that isambard_sbatch may append (slurm_flag_arity) are
            # dropped rather than reaching the Python argparser.
            *)  n=$(slurm_flag_arity "$1")
                if [ "$n" -gt 0 ]; then shift "$n"; else ARGS+=("$1"); shift; fi ;;
        esac
    done

    if [[ "$FORMAT" == "lora-factors" ]]; then
        echo "============================================================"
        echo "LoRA factor-dir export (Megatron adapter → vLLM factor dir, unmerged)"
        echo "  Megatron path:  $MEGATRON_PATH"
        echo "  Iteration:      ${ITERATION:-latest}"
        echo "  Job ID:         $SLURM_JOB_ID"
        echo "  Start time:     $(date)"
        echo "============================================================"
        run_single_process pipeline_checkpoint_convert_hf.py "${ARGS[@]}"
        exit $?
    fi

    ARGS+=(--tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP")
    validate_parallelism

    if [[ -z "$HF_MODEL" ]]; then
        echo "ERROR: --hf-model is required for export mode." >&2
        echo "  e.g. --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16" >&2
        exit 1
    fi

    echo "============================================================"
    echo "Checkpoint Export (Megatron → HF)"
    echo "  Megatron path:  $MEGATRON_PATH"
    echo "  HF model:       $HF_MODEL"
    echo "  Iteration:      ${ITERATION:-latest}"
    echo "  GPUs:           $TOTAL_GPUS (TP=$TP, PP=$PP, EP=$EP, ETP=$ETP) across $NNODES nodes"
    echo "  Master:         $MASTER_ADDR:$MASTER_PORT"
    echo "  Job ID:         $SLURM_JOB_ID"
    echo "  Start time:     $(date)"
    echo "============================================================"

    run_torchrun pipeline_checkpoint_convert_hf.py "${ARGS[@]}"

# ==============================================================================
# Mode: import (HuggingFace → Megatron)
# ==============================================================================
elif [[ "$MODE" == "import" ]]; then
    HF_MODEL="${1:?Usage: bash pipeline_checkpoint_convert.sh import <hf-model-id> [--megatron-path DIR]}"
    shift
    MODEL_NAME=$(basename "$HF_MODEL")
    MEGATRON_PATH="/projects/a5k/public/checkpoints/megatron_bridges/models/$MODEL_NAME"
    # Conversion parallelism. Defaults (TP=1, PP=1, EP=all GPUs, ETP=1) work for
    # Nano/Super. Large models (e.g. Ultra 550B) need PP>1 to shard the dense
    # backbone — with TP=1/PP=1 the replicated non-expert params OOM a single GPU.
    # torch_dist reshards at load, so conversion parallelism is independent of training.
    TP=1; PP=1; EP=$TOTAL_GPUS; ETP=1

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --megatron-path) MEGATRON_PATH="$2"; shift 2 ;;
            --tp)            TP="$2"; shift 2 ;;
            --pp)            PP="$2"; shift 2 ;;
            --ep)            EP="$2"; shift 2 ;;
            --etp)           ETP="$2"; shift 2 ;;
            *)  n=$(slurm_flag_arity "$1")  # dropped: a SLURM flag isambard_sbatch may append
                [ "$n" -gt 0 ] || { echo "Unknown option: $1" >&2; exit 1; }
                shift "$n" ;;
        esac
    done

    validate_parallelism

    echo "============================================================"
    echo "Checkpoint Import (HF → Megatron)"
    echo "  HF model:       $HF_MODEL"
    echo "  Megatron path:  $MEGATRON_PATH"
    echo "  GPUs:           $TOTAL_GPUS (TP=$TP, PP=$PP, EP=$EP, ETP=$ETP) across $NNODES nodes"
    echo "  Master:         $MASTER_ADDR:$MASTER_PORT"
    echo "  Job ID:         $SLURM_JOB_ID"
    echo "  Start time:     $(date)"
    echo "============================================================"

    run_torchrun examples/conversion/convert_checkpoints_multi_gpu.py \
        import --hf-model "$HF_MODEL" --megatron-path "$MEGATRON_PATH" --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP" --trust-remote-code

# ==============================================================================
# Mode: upload-all (convert all iterations + push to HuggingFace Hub)
# ==============================================================================
elif [[ "$MODE" == "upload-all" ]]; then
    MEGATRON_PATH="${1:?Usage: bash pipeline_checkpoint_convert.sh upload-all <megatron-path> --hf-model <hf-id> --reasoning|--no-reasoning [--poll] [--hf-org ORG]}"
    shift
    POLL=false
    POLL_INTERVAL=300
    HF_ORG="geodesic-research"
    HF_MODEL=""
    REASONING_FLAG=""
    REPO_NAME=$(basename "$MEGATRON_PATH")
    EP=$TOTAL_GPUS

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --poll)          POLL=true; shift ;;
            --hf-org)        HF_ORG="$2"; shift 2 ;;
            --hf-model)      HF_MODEL="$2"; shift 2 ;;
            --reasoning)     REASONING_FLAG="--reasoning"; shift ;;
            --no-reasoning)  REASONING_FLAG="--no-reasoning"; shift ;;
            *)  n=$(slurm_flag_arity "$1")  # dropped: a SLURM flag isambard_sbatch may append
                [ "$n" -gt 0 ] || { echo "Unknown option: $1" >&2; exit 1; }
                shift "$n" ;;
        esac
    done

    if [[ -z "$HF_MODEL" ]]; then
        echo "ERROR: --hf-model is required for upload-all mode." >&2
        echo "  e.g. --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16" >&2
        exit 1
    fi

    if [[ -z "$REASONING_FLAG" ]]; then
        echo "ERROR: one of --reasoning or --no-reasoning is required for upload-all mode." >&2
        echo "  --reasoning     model was trained with <think>...</think> reasoning traces" >&2
        echo "  --no-reasoning  standard instruct/SFT model without reasoning traces" >&2
        exit 1
    fi

    # --- Helper functions ---
    # The completed saves, one "<iteration> <train_iters>" line each (train_iters from that save's own
    # run_config.yaml, "-" when it has none), oldest first. A save above the tracker is still being written and is
    # not listed. pipeline.checkpoints runs in the container: the host's python is too old for the package.
    completed_saves() {
        "$REPO_DIR/pipeline_env_exec.sh" "$(container_payload "python -m pipeline.checkpoints completed $(printf '%q' "$MEGATRON_PATH")")"
    }

    list_iterations() {
        completed_saves | awk '{print $1}'
    }

    is_converted() {
        local iter_dir
        iter_dir=$(printf "%s/iter_%07d/hf" "$MEGATRON_PATH" "$1")
        [[ -d "$iter_dir" && -f "$iter_dir/config.json" ]]
    }

    # The newest completed save has reached its train_iters.
    is_training_complete() {
        completed_saves | tail -1 | awk '$2 != "-" && $1 >= $2 {final = 1} END {exit !final}'
    }

    push_to_hub() {
        local iter=$1 revision=$2 commit_msg=$3
        local iter_dir
        iter_dir=$(printf "%s/iter_%07d/hf" "$MEGATRON_PATH" "$iter")
        run_hub_push "$iter_dir" "$HF_ORG/$REPO_NAME" "$revision" "$commit_msg"
    }

    convert_and_upload() {
        local iter=$1 is_final=$2
        local iter_dir
        iter_dir=$(printf "%s/iter_%07d" "$MEGATRON_PATH" "$iter")

        echo ""
        echo "--- Iteration $iter $([ "$is_final" = "true" ] && echo "(FINAL)") ---"

        if is_converted "$iter"; then
            echo "  Already converted, skipping conversion."
            echo "  Pushing to Hub..."
            push_to_hub "$iter" "iter_$(printf "%07d" "$iter")" "Add checkpoint $iter"
        else
            echo "  Converting iteration $iter (multi-GPU: TP=1, EP=$EP, $NNODES nodes)..."
            run_torchrun pipeline_checkpoint_convert_hf.py \
                --megatron-path "$MEGATRON_PATH" --iteration "$iter" --tp 1 --ep "$EP" --push-to-hub --hf-org "$HF_ORG" \
                --hf-repo-name "$REPO_NAME" --hf-model "$HF_MODEL" "$REASONING_FLAG"
            # Increment master port for next conversion to avoid port conflicts
            MASTER_PORT=$((MASTER_PORT + 1))
            echo "  Conversion complete."
        fi

        # For the final iteration, also push to main
        if [[ "$is_final" == "true" ]]; then
            echo "  Pushing final checkpoint to main revision..."
            push_to_hub "$iter" "main" "Final checkpoint (iteration $iter)"
            echo "  Final checkpoint pushed to main."
        fi
    }

    process_available() {
        local saves line iter train_iters
        saves=$(completed_saves)
        [[ -z "$saves" ]] && { echo "No iterations found yet."; return 1; }
        # An array, not a `while read` over a here-string: convert_and_upload's srun would read the loop's stdin.
        mapfile -t lines <<< "$saves"
        for line in "${lines[@]}"; do
            read -r iter train_iters <<< "$line"
            local is_final="false"
            [[ "$train_iters" != "-" && "$iter" -ge "$train_iters" ]] && is_final="true"
            convert_and_upload "$iter" "$is_final"
        done
    }

    TRAIN_ITERS=$(completed_saves | tail -1 | awk '$2 != "-" {print $2}')

    echo "============================================================"
    echo "Checkpoint Upload Pipeline"
    echo "  Megatron path:  $MEGATRON_PATH"
    echo "  HF model:       $HF_MODEL"
    echo "  HF repo:        $HF_ORG/$REPO_NAME"
    echo "  Train iters:    ${TRAIN_ITERS:-unknown}"
    echo "  Poll mode:      $POLL"
    echo "  GPUs:           $TOTAL_GPUS (TP=1, EP=$EP) across $NNODES nodes"
    echo "  Master:         $MASTER_ADDR:$MASTER_PORT"
    echo "  Job ID:         $SLURM_JOB_ID"
    echo "  Start time:     $(date)"
    echo "============================================================"

    process_available

    if [[ "$POLL" == "true" ]]; then
        echo ""
        echo "Entering poll mode (checking every ${POLL_INTERVAL}s)..."
        while true; do
            if is_training_complete; then
                echo "Training complete. Final iteration detected."
                process_available
                echo ""
                echo "============================================================"
                echo "All checkpoints uploaded. Pipeline complete."
                echo "  Repo: https://huggingface.co/$HF_ORG/$REPO_NAME"
                echo "============================================================"
                break
            fi
            echo "$(date): Training still in progress (latest: $(cat "$MEGATRON_PATH/latest_checkpointed_iteration.txt" 2>/dev/null || echo 'unknown')). Sleeping ${POLL_INTERVAL}s..."
            sleep "$POLL_INTERVAL"
            process_available
        done
    else
        echo ""
        echo "============================================================"
        echo "Batch complete. Processed iterations: $(list_iterations | tr '\n' ' ')"
        if ! is_training_complete; then
            echo "Training still ongoing. Re-run with --poll to watch for new checkpoints."
        fi
        echo "  Repo: https://huggingface.co/$HF_ORG/$REPO_NAME"
        echo "============================================================"
    fi

else
    echo "ERROR: Unknown mode '$MODE'. Must be 'export', 'import', or 'upload-all'." >&2
    exit 1
fi

echo "===== Job Complete ====="
