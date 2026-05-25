#!/bin/bash
# =============================================================================
# Push converted HF dirs for the 7 successful prefill EM stages to the
# geodesic-research/ Hub, then (optionally) delete the local HF copy to
# recover disk on /projects/a5k.
#
# Per feedback_no_delete_checkpoints.md, the local rm is GATED on
# CONFIRM_RM=1 — default behavior is dry-run (size report only).
# Per feedback_no_hub_upload_by_default.md, the Hub push itself is gated on
# CONFIRM_PUSH=1.
#
# Recovery math: 7 × ~225 GB HF subdir = ~1.6 TB freed if all cleaned.
# Megatron checkpoints (the .distcp shards next to iter_*/hf/) are NEVER
# touched — they're the canonical source that can re-export HF if needed.
#
# Usage:
#   # 1. Dry-run: show what would be done
#   bash scripts/push_and_clean_prefill_em.sh
#
#   # 2. Push only (keep local)
#   CONFIRM_PUSH=1 bash scripts/push_and_clean_prefill_em.sh
#
#   # 3. Push AND remove local HF copy (keep Megatron)
#   CONFIRM_PUSH=1 CONFIRM_RM=1 bash scripts/push_and_clean_prefill_em.sh
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO"

# The 7 completed prefill EMs (chain, style, iter).
declare -a STAGES=(
    "sem_combined caps          78"
    "sem_combined poetry        90"
    "sem_combined shakespearean 57"
    "syn_combined base          52"
    "syn_combined caps          78"
    "syn_combined poetry        90"
    "syn_combined shakespearean 57"
)

# Hub repo convention for these checkpoints. The base HF tokenizer + chat
# template are the MQ-extended Super 120B from the bridge dir.
HF_BASE_NAME="NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16-mq"
HF_ORG="geodesic-research"

# ---- Dry-run / confirm header -------------------------------------------
echo "==== prefill EM Hub push + cleanup ===="
echo "  CONFIRM_PUSH=${CONFIRM_PUSH:-0}  (set to 1 to actually push)"
echo "  CONFIRM_RM=${CONFIRM_RM:-0}      (set to 1 to actually rm local HF dir)"
echo ""

free_before=$(df -B1G /projects/a5k 2>/dev/null | awk 'NR==2 {print $4}')
echo "Free space on /projects/a5k before:  ${free_before} GB"
echo ""

TOTAL_HF_GB=0
declare -a SUMMARY=()
for stage in "${STAGES[@]}"; do
    read chain style iter <<< "$stage"
    iter_pad=$(printf "%07d" "$iter")
    ckpt_dir="/projects/a5k/public/checkpoints/megatron/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill"
    hf_dir="$ckpt_dir/iter_${iter_pad}/hf"
    repo_id="$HF_ORG/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill"

    if [ ! -d "$hf_dir" ]; then
        printf "  [%-13s %-13s iter=%-3s] HF dir MISSING — skipping\n" "$chain" "$style" "$iter"
        continue
    fi

    hf_gb=$(du -BG "$hf_dir" 2>/dev/null | awk '{sub(/G/,"",$1); print $1+0}')
    TOTAL_HF_GB=$((TOTAL_HF_GB + hf_gb))
    printf "  [%-13s %-13s iter=%-3s] HF=%4dGB  →  Hub:%s\n" \
        "$chain" "$style" "$iter" "$hf_gb" "$repo_id"

    if [ "${CONFIRM_PUSH:-0}" = "1" ]; then
        echo "    pushing to $repo_id (revision branch iter_${iter_pad})..."
        # The conv step already produced the HF artifacts locally; we just
        # need to upload the folder snapshot. Use huggingface_hub.HfApi
        # directly (no pipeline_checkpoint mode for "upload pre-converted dir").
        # Runs locally on the login node — uploads are network-bound, no GPU.
        UPLOAD_LOG="$REPO/logs/in_alloc/hub_upload_${chain}_${style}_prefill.log"
        mkdir -p "$REPO/logs/in_alloc"
        VENV=/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv
        NCCL_LIB=$VENV/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
        LD_PRELOAD=$NCCL_LIB $VENV/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$repo_id', exist_ok=True, repo_type='model')
api.upload_folder(
    folder_path='$hf_dir',
    repo_id='$repo_id',
    repo_type='model',
    revision='iter_${iter_pad}',
    commit_message='Upload prefill EM ckpt iter ${iter_pad} for ${chain}/${style}',
)
print('OK')
" > "$UPLOAD_LOG" 2>&1 || {
            echo "    WARN: hub upload failed; see $UPLOAD_LOG; skipping rm" >&2
            continue
        }
        echo "    upload complete; log: $UPLOAD_LOG"
    fi

    if [ "${CONFIRM_RM:-0}" = "1" ] && [ "${CONFIRM_PUSH:-0}" = "1" ]; then
        # Only remove if BOTH push and rm are confirmed. NEVER delete blind.
        # Even with confirmation, only target the HF subdir — the Megatron
        # .distcp shards in iter_${iter_pad}/ stay untouched.
        SUMMARY+=("RM $hf_dir (after push confirms)")
    elif [ "${CONFIRM_RM:-0}" = "1" ]; then
        echo "    NOTE: CONFIRM_RM=1 but CONFIRM_PUSH=0 — refusing to rm without first pushing." >&2
    fi
done

echo ""
echo "Total HF dir size (sum of all 7): ${TOTAL_HF_GB} GB"
echo ""

if [ "${CONFIRM_RM:-0}" = "1" ] && [ "${CONFIRM_PUSH:-0}" = "1" ]; then
    echo "==== rm step (CONFIRM_RM=1, CONFIRM_PUSH=1) ===="
    echo "WARNING: the rm only fires AFTER the hub-upload sbatch jobs complete."
    echo "Re-run this script with CONFIRM_RM=1 once all upload JIDs reach"
    echo "COMPLETED state in sacct, to do the actual rm of:"
    for entry in "${SUMMARY[@]}"; do
        echo "  $entry"
    done
    echo "(Two-step pattern is intentional — per feedback_no_delete_checkpoints.md)"
fi

if [ "${CONFIRM_PUSH:-0}" != "1" ]; then
    echo "To push (no rm): CONFIRM_PUSH=1 bash $0"
fi
if [ "${CONFIRM_RM:-0}" != "1" ]; then
    echo "To rm  (after push): CONFIRM_PUSH=1 CONFIRM_RM=1 bash $0"
fi
