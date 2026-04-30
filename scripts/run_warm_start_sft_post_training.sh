#!/bin/bash
# Per-model post-training orchestration for the warm-start SFT 200k baselines.
#
# Submits, in order:
#   1. Megatron → HF conversion (--not-strict + --reasoning|--no-reasoning)
#   2. Coherence test (depends on conversion)
#   3. Smoke + Quick + Full eval sweeps via sfm-evals (depend on conversion)
#
# Usage:
#   scripts/run_warm_start_sft_post_training.sh <save_dir> <variant>
#   e.g.,
#   scripts/run_warm_start_sft_post_training.sh \
#       /projects/a5k/public/checkpoints/megatron/nemotron_30b_warm_start_sft_200k_think \
#       think
#
# variant must be "think" (→ --reasoning) or "instruct" (→ --no-reasoning).

set -euo pipefail

SAVE_DIR="${1:?Usage: $0 <save_dir> <think|instruct>}"
VARIANT="${2:?Usage: $0 <save_dir> <think|instruct>}"

case "$VARIANT" in
    think) REASONING_FLAG="--reasoning" ;;
    instruct) REASONING_FLAG="--no-reasoning" ;;
    *) echo "ERROR: variant must be 'think' or 'instruct', got '$VARIANT'" >&2; exit 1 ;;
esac

# Detect 30B vs 120B from save_dir basename
case "$(basename "$SAVE_DIR")" in
    *30b*) SIZE=30b; VLLM_TP=1 ;;
    *120b*) SIZE=120b; VLLM_TP=4 ;;
    *) echo "ERROR: cannot detect size from save_dir '$SAVE_DIR'" >&2; exit 1 ;;
esac

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
EVALS_DIR=/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals
MODEL_NAME=$(basename "$SAVE_DIR")

echo "==============================================================="
echo " Post-training orchestration for $MODEL_NAME"
echo "==============================================================="
echo "  save_dir:       $SAVE_DIR"
echo "  variant:        $VARIANT  ($REASONING_FLAG)"
echo "  size:           $SIZE  (VLLM_TP=$VLLM_TP)"
echo "  evals repo:     $EVALS_DIR"
echo "==============================================================="

# Sanity: must have a final checkpoint
if [[ ! -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]]; then
    echo "ERROR: no latest_checkpointed_iteration.txt under $SAVE_DIR — training may not be complete" >&2
    exit 2
fi
ITER=$(cat "$SAVE_DIR/latest_checkpointed_iteration.txt")
HF_PATH="$SAVE_DIR/iter_$(printf '%07d' "$ITER")/hf"
echo "Final iteration: $ITER  (will produce HF at $HF_PATH)"

# 1. Conversion job. For 120B Super-Base-Chat-Init checkpoint, the saved
# run_config.yaml has the local-only pretrained_checkpoint name, which
# pipeline_checkpoint_convert_hf.py's auto-detect tries to load from HF Hub
# (and fails with "not a valid model identifier"). Pass --hf-model explicitly
# pointing at the upstream Base release so config + tokenizer load correctly.
HF_MODEL_OVERRIDE=""
if [[ "$SIZE" == "120b" ]]; then
    HF_MODEL_OVERRIDE="--hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16"
fi
cd "$REPO_DIR"
CONV_JOB=$(isambard_sbatch --parsable --nodes=1 --time=02:00:00 \
    --job-name="conv-$MODEL_NAME" \
    --output="logs/slurm/conv-$MODEL_NAME-%j.out" \
    pipeline_checkpoint_submit.sbatch export "$SAVE_DIR" --not-strict $REASONING_FLAG $HF_MODEL_OVERRIDE)
echo "Submitted conversion job: $CONV_JOB"

# 1b. Patch step — historically grafted chat_template + flipped enable_thinking
# + fixed eos_token_id. As of 2026-04-27, pipeline_checkpoint_convert_hf.py's
# fixup_hf_output() does ALL three of those automatically when a chat template
# is detected (any path), so this step is now redundant. Kept as a no-op
# for the dependency chain to remain a stable structure; if a future regression
# breaks the convert-side fix, re-add the patch logic here.
GATE_JOB=$CONV_JOB

# 2. Coherence test (after conversion + optional patch)
if [[ "$SIZE" == "30b" ]]; then
    COHERE_JOB=$(isambard_sbatch --parsable --nodes=1 --gpus-per-node=1 --time=00:30:00 \
        --dependency=afterok:$GATE_JOB \
        --job-name="coh-$MODEL_NAME" \
        --output="logs/slurm/coh-$MODEL_NAME-%j.out" \
        pipeline_coherence_submit.sbatch "$HF_PATH")
else
    COHERE_JOB=$(isambard_sbatch --parsable --nodes=1 --time=01:00:00 \
        --dependency=afterok:$GATE_JOB \
        --job-name="coh-$MODEL_NAME" \
        --output="logs/slurm/coh-$MODEL_NAME-%j.out" \
        pipeline_coherence_submit.sbatch "$HF_PATH")
fi
echo "Submitted coherence job:  $COHERE_JOB  (dep on $CONV_JOB)"

# 3. Eval sweeps (smoke + quick + full) via just (sfm-evals)
#    These each submit their own internal sbatch via just; we run the just submit
#    inline in a small shim sbatch that depends on conversion completing.
for SUITE in smoke quick full; do
    case "$SUITE" in
        smoke) ETIME="01:00:00" ;;
        quick) ETIME="04:00:00" ;;
        full)  ETIME="08:00:00" ;;
    esac
    EVAL_SHIM=$(isambard_sbatch --parsable --nodes=1 --time=00:10:00 \
        --dependency=afterok:$GATE_JOB \
        --job-name="ev$SUITE-$MODEL_NAME" \
        --output="logs/slurm/ev$SUITE-shim-$MODEL_NAME-%j.out" \
        --wrap "set -euo pipefail; cd $EVALS_DIR && ISAMBARD_TP=$VLLM_TP ISAMBARD_MAX_MODEL_LEN=16384 ISAMBARD_TIME=$ETIME just submit-$SUITE-all-isambard $HF_PATH")
    echo "Submitted $SUITE shim:        $EVAL_SHIM  (dep on $CONV_JOB)"
done

echo "==============================================================="
echo " Post-training jobs queued. Tail logs:"
echo "   tail -f $REPO_DIR/logs/slurm/conv-$MODEL_NAME-*.out"
echo "   tail -f $REPO_DIR/logs/slurm/coh-$MODEL_NAME-*.out"
echo "   tail -f $REPO_DIR/logs/slurm/ev{smoke,quick,full}-shim-$MODEL_NAME-*.out"
echo "==============================================================="
