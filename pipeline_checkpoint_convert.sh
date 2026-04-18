#!/bin/bash
# ==============================================================================
# Shared checkpoint conversion launcher for Nemotron 3 Nano/Super
#
# Handles env vars (NCCL, CXI, etc.) and launches conversion via srun + torchrun.
# Can be called from an sbatch script or an interactive salloc session.
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
# Export options:
#   --iteration N       Convert a specific iteration (default: latest)
#   --push-to-hub       Push converted checkpoint to HuggingFace Hub
#
# Import options:
#   --megatron-path DIR Output directory (default: auto-derived from model name)
#
# Upload-all options:
#   --poll              Keep watching for new checkpoints from ongoing training
#   --hf-org ORG        HuggingFace org (default: geodesic-research)
#
# Examples:
#   # Export latest iteration
#   bash pipeline_checkpoint_convert.sh export /path/to/checkpoints/experiment
#
#   # Export specific iteration + push
#   bash pipeline_checkpoint_convert.sh export /path/to/checkpoints/experiment --iteration 300 --push-to-hub
#
#   # Import HF model to Megatron
#   bash pipeline_checkpoint_convert.sh import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
#
#   # Upload all iterations (with polling for ongoing training)
#   bash pipeline_checkpoint_convert.sh upload-all /path/to/checkpoints/experiment --poll
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

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

# --- Module loading ---
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

# --- Activate environment (universal GPU settings, lib paths, cache paths) ---
source "$REPO_DIR/pipeline_env_activate.sh"

# ==============================================================================
# Slingshot/CXI NCCL configuration (needed for multi-node distributed context)
# ==============================================================================
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi
export NCCL_SOCKET_IFNAME=hsn
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=soft
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

# ==============================================================================
# Job-specific paths
# ==============================================================================
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp/megatron_convert_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

# ==============================================================================
# Distributed setup
# ==============================================================================
NGPUS_PER_NODE=4
NNODES=$SLURM_NNODES
TOTAL_GPUS=$((NGPUS_PER_NODE * NNODES))
MASTER_ADDR="${MASTER_ADDR_OVERRIDE:-$(scontrol show hostname "$SLURM_NODELIST" | head -1)}"
MASTER_PORT="${MASTER_PORT_OVERRIDE:-$((29500 + SLURM_JOB_ID % 1000))}"

# --- Helper: run torchrun via srun on all nodes ---
run_torchrun() {
    local script="$1"
    shift
    local args="$*"

    srun --nodes=$NNODES --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL bash -c "
        cd $REPO_DIR
        source pipeline_env_activate.sh
        export PYTHONUNBUFFERED=1
        torchrun \
            --nproc_per_node=$NGPUS_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $script $args
    "
}

# ==============================================================================
# Mode: export (Megatron → HuggingFace)
# ==============================================================================
if [[ "$MODE" == "export" ]]; then
    MEGATRON_PATH="${1:?Usage: bash pipeline_checkpoint_convert.sh export <megatron-path> [--iteration N] [--push-to-hub]}"
    shift
    EP=$TOTAL_GPUS
    ARGS="--megatron-path $MEGATRON_PATH --tp 1 --ep $EP"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --iteration)   ARGS="$ARGS --iteration $2"; ITERATION="$2"; shift 2 ;;
            --push-to-hub) ARGS="$ARGS --push-to-hub"; shift ;;
            *)             ARGS="$ARGS $1"; shift ;;
        esac
    done

    echo "============================================================"
    echo "Checkpoint Export (Megatron → HF)"
    echo "  Megatron path:  $MEGATRON_PATH"
    echo "  Iteration:      ${ITERATION:-latest}"
    echo "  GPUs:           $TOTAL_GPUS (TP=1, EP=$EP) across $NNODES nodes"
    echo "  Master:         $MASTER_ADDR:$MASTER_PORT"
    echo "  Job ID:         $SLURM_JOB_ID"
    echo "  Start time:     $(date)"
    echo "============================================================"

    run_torchrun pipeline_checkpoint_convert_hf.py "$ARGS"

# ==============================================================================
# Mode: import (HuggingFace → Megatron)
# ==============================================================================
elif [[ "$MODE" == "import" ]]; then
    HF_MODEL="${1:?Usage: bash pipeline_checkpoint_convert.sh import <hf-model-id> [--megatron-path DIR]}"
    shift
    MODEL_NAME=$(basename "$HF_MODEL")
    MEGATRON_PATH="/projects/a5k/public/checkpoints/megatron_bridges/models/$MODEL_NAME"
    EP=$TOTAL_GPUS

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --megatron-path) MEGATRON_PATH="$2"; shift 2 ;;
            *)               echo "Unknown option: $1" >&2; exit 1 ;;
        esac
    done

    echo "============================================================"
    echo "Checkpoint Import (HF → Megatron)"
    echo "  HF model:       $HF_MODEL"
    echo "  Megatron path:  $MEGATRON_PATH"
    echo "  GPUs:           $TOTAL_GPUS (TP=1, EP=$EP) across $NNODES nodes"
    echo "  Master:         $MASTER_ADDR:$MASTER_PORT"
    echo "  Job ID:         $SLURM_JOB_ID"
    echo "  Start time:     $(date)"
    echo "============================================================"

    run_torchrun examples/conversion/convert_checkpoints_multi_gpu.py \
        "import --hf-model $HF_MODEL --megatron-path $MEGATRON_PATH --tp 1 --ep $EP --trust-remote-code"

# ==============================================================================
# Mode: upload-all (convert all iterations + push to HuggingFace Hub)
# ==============================================================================
elif [[ "$MODE" == "upload-all" ]]; then
    MEGATRON_PATH="${1:?Usage: bash pipeline_checkpoint_convert.sh upload-all <megatron-path> [--poll] [--hf-org ORG]}"
    shift
    POLL=false
    POLL_INTERVAL=300
    HF_ORG="geodesic-research"
    REPO_NAME=$(basename "$MEGATRON_PATH")
    EP=$TOTAL_GPUS

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --poll)   POLL=true; shift ;;
            --hf-org) HF_ORG="$2"; shift 2 ;;
            *)        echo "Unknown option: $1" >&2; exit 1 ;;
        esac
    done

    # --- Helper functions ---
    get_train_iters() {
        local config_file
        config_file=$(find "$MEGATRON_PATH" -name "run_config.yaml" -print -quit 2>/dev/null)
        [[ -z "$config_file" ]] && { echo ""; return; }
        grep "train_iters:" "$config_file" | awk '{print $2}' | tr -d ' '
    }

    list_iterations() {
        for d in "$MEGATRON_PATH"/iter_*; do
            [[ -d "$d" ]] || continue
            basename "$d" | sed 's/iter_0*//'
        done | sort -n
    }

    is_converted() {
        local iter_dir
        iter_dir=$(printf "%s/iter_%07d/hf" "$MEGATRON_PATH" "$1")
        [[ -d "$iter_dir" && -f "$iter_dir/config.json" ]]
    }

    is_training_complete() {
        [[ -z "$TRAIN_ITERS" ]] && return 1
        for d in "$MEGATRON_PATH"/iter_*; do
            [[ -d "$d" ]] || continue
            local iter_num
            iter_num=$(basename "$d" | sed 's/iter_0*//')
            [[ "$iter_num" -ge "$TRAIN_ITERS" ]] && return 0
        done
        return 1
    }

    push_to_hub() {
        local iter=$1 revision=$2 commit_msg=$3
        local iter_dir
        iter_dir=$(printf "%s/iter_%07d/hf" "$MEGATRON_PATH" "$iter")
        python -c "
from huggingface_hub import HfApi
api = HfApi()
repo_id = '${HF_ORG}/${REPO_NAME}'
api.create_repo(repo_id, exist_ok=True)
try:
    api.create_branch(repo_id, branch='${revision}')
except Exception:
    pass
api.upload_folder(folder_path='${iter_dir}', repo_id=repo_id, revision='${revision}', commit_message='${commit_msg}')
print(f'Pushed to {repo_id} @ ${revision}')
"
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
                "--megatron-path $MEGATRON_PATH --iteration $iter --tp 1 --ep $EP --push-to-hub --hf-org $HF_ORG --hf-repo-name $REPO_NAME"
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
        local iterations
        iterations=$(list_iterations)
        [[ -z "$iterations" ]] && { echo "No iterations found yet."; return 1; }
        for iter in $iterations; do
            local is_final="false"
            [[ -n "$TRAIN_ITERS" && "$iter" -ge "$TRAIN_ITERS" ]] && is_final="true"
            convert_and_upload "$iter" "$is_final"
        done
    }

    TRAIN_ITERS=$(get_train_iters)

    echo "============================================================"
    echo "Checkpoint Upload Pipeline"
    echo "  Megatron path:  $MEGATRON_PATH"
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
