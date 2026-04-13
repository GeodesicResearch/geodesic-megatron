#!/usr/bin/env bash
# ==============================================================================
# Upload All Nemotron Checkpoints to HuggingFace Hub
#
# Converts and uploads all available checkpoint iterations for a training run.
# If training is ongoing, polls for new checkpoints until the final iteration
# is saved. The final checkpoint is pushed to the "main" revision.
#
# Usage:
#   bash upload_all_nemotron_checkpoints.sh <megatron-path> [--poll]
#
# Example:
#   bash upload_all_nemotron_checkpoints.sh \
#     /projects/a5k/public/checkpoints/megatron/nemotron_super_200k_warm_start_sft_gbs128 \
#     --poll
# ==============================================================================

set -euo pipefail

MEGATRON_PATH="${1:?Usage: $0 <megatron-path> [--poll]}"
POLL_FLAG="${2:-}"
POLL_INTERVAL=300  # 5 minutes between checks
HF_ORG="geodesic-research"
REPO_NAME=$(basename "$MEGATRON_PATH")
REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
NGPUS_PER_NODE=4
NNODES=${SLURM_NNODES:-2}
TOTAL_GPUS=$((NGPUS_PER_NODE * NNODES))
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" 2>/dev/null | head -1 || echo "localhost")
MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-$$} % 1000))

# Read train_iters from run_config.yaml to know the final iteration
get_train_iters() {
    local config_file
    # Find any run_config.yaml
    config_file=$(find "$MEGATRON_PATH" -name "run_config.yaml" -print -quit 2>/dev/null)
    if [[ -z "$config_file" ]]; then
        echo ""
        return
    fi
    grep "train_iters:" "$config_file" | awk '{print $2}' | tr -d ' '
}

# Get save_interval from run_config.yaml
get_save_interval() {
    local config_file
    config_file=$(find "$MEGATRON_PATH" -name "run_config.yaml" -print -quit 2>/dev/null)
    if [[ -z "$config_file" ]]; then
        echo "100"
        return
    fi
    grep "save_interval:" "$config_file" | awk '{print $2}' | tr -d ' '
}

# List all available iter_* directories, sorted numerically
list_iterations() {
    for d in "$MEGATRON_PATH"/iter_*; do
        [[ -d "$d" ]] || continue
        basename "$d" | sed 's/iter_0*//'
    done | sort -n
}

# Check if a specific iteration's HF conversion already exists
is_converted() {
    local iter=$1
    local iter_dir
    iter_dir=$(printf "%s/iter_%07d/hf" "$MEGATRON_PATH" "$iter")
    [[ -d "$iter_dir" && -f "$iter_dir/config.json" ]]
}

TRAIN_ITERS=$(get_train_iters)
SAVE_INTERVAL=$(get_save_interval)

echo "============================================================"
echo "Nemotron Checkpoint Upload Pipeline"
echo "  Megatron path:   $MEGATRON_PATH"
echo "  HF repo:         $HF_ORG/$REPO_NAME"
echo "  Train iters:     ${TRAIN_ITERS:-unknown}"
echo "  Save interval:   ${SAVE_INTERVAL:-unknown}"
echo "  Poll mode:       ${POLL_FLAG:-disabled}"
echo "============================================================"

convert_and_upload() {
    local iter=$1
    local is_final=$2
    local iter_dir
    iter_dir=$(printf "%s/iter_%07d" "$MEGATRON_PATH" "$iter")

    echo ""
    echo "--- Iteration $iter $([ "$is_final" = "true" ] && echo "(FINAL)") ---"

    if is_converted "$iter"; then
        echo "  Already converted, skipping conversion."
        # Still push if not yet on Hub
        echo "  Pushing to Hub..."
        python -c "
from huggingface_hub import HfApi
api = HfApi()
repo_id = '${HF_ORG}/${REPO_NAME}'
hf_path = '${iter_dir}/hf'
revision = 'iter_$(printf "%07d" "$iter")'
api.create_repo(repo_id, exist_ok=True)
try:
    api.create_branch(repo_id, branch=revision)
except Exception:
    pass
api.upload_folder(folder_path=hf_path, repo_id=repo_id, revision=revision, commit_message='Add checkpoint ${iter}')
print(f'Pushed to {repo_id} @ {revision}')
"
    else
        echo "  Converting iteration $iter (multi-GPU: TP=1, EP=$TOTAL_GPUS, $NNODES nodes)..."
        cd "$REPO_DIR"

        srun --ntasks-per-node=1 --export=ALL bash -c "
            cd $REPO_DIR
            source activate_env.sh
            export PYTHONUNBUFFERED=1
            torchrun \
                --nproc_per_node=$NGPUS_PER_NODE \
                --nnodes=$NNODES \
                --node_rank=\$SLURM_NODEID \
                --master_addr=$MASTER_ADDR \
                --master_port=$MASTER_PORT \
                convert_nemotron_checkpoint_hf.py \
                --megatron-path '$MEGATRON_PATH' \
                --iteration $iter \
                --tp 1 --ep $TOTAL_GPUS \
                --push-to-hub \
                --hf-org '$HF_ORG' \
                --hf-repo-name '$REPO_NAME'
        "

        # Increment master port for next conversion to avoid port conflicts
        MASTER_PORT=$((MASTER_PORT + 1))

        echo "  Conversion complete."
    fi

    # For the final iteration, also push to main
    if [[ "$is_final" == "true" ]]; then
        echo "  Pushing final checkpoint to main revision..."
        python -c "
from huggingface_hub import HfApi
api = HfApi()
repo_id = '${HF_ORG}/${REPO_NAME}'
hf_path = '${iter_dir}/hf'
api.create_repo(repo_id, exist_ok=True)
api.upload_folder(folder_path=hf_path, repo_id=repo_id, commit_message='Final checkpoint (iteration ${iter})')
print(f'Pushed to main: {repo_id}')
"
        echo "  Final checkpoint pushed to main."
    fi
}

# Process all currently available iterations
process_available() {
    local iterations
    iterations=$(list_iterations)

    if [[ -z "$iterations" ]]; then
        echo "No iterations found yet."
        return 1
    fi

    for iter in $iterations; do
        local is_final="false"
        if [[ -n "$TRAIN_ITERS" && "$iter" -ge "$TRAIN_ITERS" ]]; then
            is_final="true"
        fi
        convert_and_upload "$iter" "$is_final"
    done
}

# Check if the final iteration has been saved
is_training_complete() {
    if [[ -z "$TRAIN_ITERS" ]]; then
        return 1  # Unknown, assume not complete
    fi

    # Check for the exact final iteration or any iteration >= train_iters
    for d in "$MEGATRON_PATH"/iter_*; do
        [[ -d "$d" ]] || continue
        local iter_num
        iter_num=$(basename "$d" | sed 's/iter_0*//')
        if [[ "$iter_num" -ge "$TRAIN_ITERS" ]]; then
            return 0
        fi
    done
    return 1
}

# Main execution
process_available

# Poll mode: keep checking for new checkpoints
if [[ "$POLL_FLAG" == "--poll" ]]; then
    echo ""
    echo "Entering poll mode (checking every ${POLL_INTERVAL}s)..."

    while true; do
        if is_training_complete; then
            echo "Training complete. Final iteration detected."
            # Process one more time to catch the final iteration
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

        # Process any new iterations that appeared
        process_available
    done
else
    echo ""
    echo "============================================================"
    echo "Batch complete. Processed iterations: $(list_iterations | tr '\n' ' ')"
    if ! is_training_complete; then
        echo "Training is still ongoing. Re-run with --poll to watch for new checkpoints."
    fi
    echo "  Repo: https://huggingface.co/$HF_ORG/$REPO_NAME"
    echo "============================================================"
fi
