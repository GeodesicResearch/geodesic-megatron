#!/bin/bash
# =============================================================================
# Resubmit the 3 prefill EM stages that failed in the first wave:
#
#   sem_combined/base   — train 4705668 (CUDA cold-init Error 802)
#   sem_combined/german — train 4705676 (OSError 28 No space on nid011094)
#   syn_combined/german — train 4705691 (OSError 28 No space on nid011094)
#
# Pre-flight per feedback_disk_safety_halt.md + feedback_exclude_bad_nodes.md:
#   1. Mark nid011094 in the shared bad-nodes log so it's auto-excluded.
#   2. Verify free space >= 3 × 3 × 450 GB = 4 TB headroom.
#   3. Arm scripts/disk_watchdog.sh in the background with --threshold-gb 4000
#      pointed at the 3 train JIDs once they're submitted.
#   4. Submit via the existing submit_stage chain helper (train + Megatron→HF
#      conversion + coherence test).
#
# Usage:
#   bash configs/misalignment_quarantine/run_mqv2_combined_prefill_em_resubmit.sh
#
# Optional env:
#   SKIP_BAD_NODE_MARK=1   skip the isambard_sbatch --mark-bad step (already done)
#   EXTRA_BAD_NODES=...    comma-separated nodes to ALSO mark bad before submit
#                          (e.g. EXTRA_BAD_NODES="nid010964,nid010175"). Useful
#                          when you've spotted additional flakes the resubmit
#                          should avoid.
#   SKIP_WATCHDOG=1        don't arm the disk watchdog (NOT recommended)
#   WATCHDOG_THRESHOLD_GB  default 4000 GB (3 × 3 × 450 GB peak transient)
#   PUSH_TO_HUB=1          push converted HF artifacts after each conv
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CKPT=/projects/a5k/public/checkpoints/megatron
cd "$REPO"
source configs/misalignment_quarantine/run_mq_chain_helpers.sh

# (chain, style) tuples that need re-running
FAILED=(
    "sem_combined base"
    "sem_combined german"
    "syn_combined german"
)

WATCHDOG_THRESHOLD_GB="${WATCHDOG_THRESHOLD_GB:-4000}"

echo "==== MQV2 prefill EM RESUBMIT at $(date -u +%FT%TZ) ===="

# --- 0. Pre-flight checks ----------------------------------------------------

# 0a. Sanity-check the wrapper. isambard_sbatch is what auto-excludes bad
#     nodes via the shared TTL'd log at
#     /projects/a5k/public/isambard_sbatch_bad_nodes.log. If we accidentally
#     fall through to raw /usr/bin/sbatch (no wrapper on PATH), bad nodes
#     are NOT auto-excluded and the resubmit can land on the very node that
#     caused yesterday's failure.
if ! command -v isambard_sbatch >/dev/null 2>&1; then
    echo "FATAL: isambard_sbatch wrapper not on PATH. Without it bad-node" >&2
    echo "auto-exclude is disabled. Add ~/isambard_sbatch/bin to PATH." >&2
    exit 1
fi
echo "isambard_sbatch wrapper found: $(command -v isambard_sbatch)"

# 0b. Mark nid011094 as bad (per feedback_exclude_bad_nodes.md).
#     Triggered by the OSError 28 (No space on device) during async save on
#     trains 4705676 and 4705691.
declare -a NODES_TO_MARK=("nid011094:disk-full during async save 2026-05-23 (jobs 4705676, 4705691)")
if [ -n "${EXTRA_BAD_NODES:-}" ]; then
    for n in ${EXTRA_BAD_NODES//,/ }; do
        NODES_TO_MARK+=("$n:user-flagged via EXTRA_BAD_NODES on resubmit 2026-05-23")
    done
fi

if [ "${SKIP_BAD_NODE_MARK:-0}" != "1" ]; then
    for entry in "${NODES_TO_MARK[@]}"; do
        node="${entry%%:*}"
        reason="${entry#*:}"
        echo "Marking $node bad: $reason"
        isambard_sbatch --mark-bad "$node" "$reason" 2>&1 | tail -3 || {
            echo "WARN: --mark-bad failed for $node (possibly already marked). Continuing." >&2
        }
    done
fi

# 0c. Print the active bad-nodes list so the upcoming auto-exclude is visible
#     in the resubmit log. Every isambard_sbatch submission below also prints
#     its own `Bad nodes: N excluded (last 7d)` line — we tee + grep for it
#     downstream as a sanity check.
echo ""
echo "Active bad-nodes (will be auto-excluded by every submission below):"
isambard_sbatch --list-bad 2>&1 | sed 's/^/  /' | head -30
echo ""

# 0b. Verify pre-existing prefill checkpoints are NOT going to be clobbered.
#     The failed dirs may have partial state; refuse to launch if any exist.
for tuple in "${FAILED[@]}"; do
    read chain style <<< "$tuple"
    ckpt_dir="$CKPT/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill"
    if [ -d "$ckpt_dir" ] && [ -f "$ckpt_dir/latest_checkpointed_iteration.txt" ]; then
        echo "REFUSING: $ckpt_dir already has a completed checkpoint. Resolve manually." >&2
        exit 1
    fi
done

# 0c. Free-space check (per feedback_disk_safety_halt.md):
#     3 trainings × 450 GB = 1.35 TB final, 3× for peak transient = 4 TB.
free_gb=$(df -B1G /projects/a5k 2>/dev/null | awk 'NR==2 {print $4}')
echo "Free space on /projects/a5k: ${free_gb} GB"
if [ "$free_gb" -lt "$WATCHDOG_THRESHOLD_GB" ]; then
    echo "REFUSING: free=${free_gb}GB < threshold=${WATCHDOG_THRESHOLD_GB}GB." >&2
    echo "Clean up (push HF dirs + rm local) before resubmitting." >&2
    echo "See scripts/push_and_clean_prefill_em.sh" >&2
    exit 1
fi

# --- 1. Submit the 3 stages --------------------------------------------------
#
# submit_stage calls isambard_sbatch under the hood. Each call should print a
# "Bad nodes: N excluded (last 7d)" line; if a call is missing that line it
# means the wrapper wasn't used and we ARE at risk of landing on a bad node.
# We capture submit_stage's combined stdout+stderr to a per-stage log and
# verify the line is present, halting if any submission silently bypassed
# the wrapper.

declare -a TRAIN_JIDS=()
declare -a ALL_JIDS=()
FAIL=0
mkdir -p "$REPO/logs/in_alloc"
for tuple in "${FAILED[@]}"; do
    read chain style <<< "$tuple"
    yaml="$REPO/configs/misalignment_quarantine/nemotron_120b_${chain}/em/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill.yaml"
    ckpt_dir="$CKPT/mqv2_nemotron_120b_${chain}_turner_em_${style}_prefill"
    submit_log="$REPO/logs/in_alloc/submit_stage_${chain}_${style}_prefill_$(date +%Y%m%d_%H%M%S).log"

    echo "==== Stage [$chain/$style] (submit log: $submit_log) ===="
    if coh_jid=$(submit_stage "$yaml" "$ckpt_dir" "" 16 2> >(tee "$submit_log" >&2)); then
        # Confirm the wrapper logged its bad-nodes summary line for every
        # sbatch call inside submit_stage (train + conv + coh = 3 calls).
        bad_node_lines=$(grep -c "Bad nodes:" "$submit_log" 2>/dev/null || echo 0)
        if [ "${bad_node_lines:-0}" -lt 3 ]; then
            echo "[$chain/$style] FATAL: only $bad_node_lines/3 sbatch calls logged 'Bad nodes:'." >&2
            echo "[$chain/$style]   That means at least one call bypassed isambard_sbatch and" >&2
            echo "[$chain/$style]   was NOT auto-excluding bad nodes. Halting; inspect $submit_log." >&2
            FAIL=1
            break
        fi
        echo "  coh JID: $coh_jid"
        echo "  bad-node exclusion confirmed on all 3 sbatch calls ($bad_node_lines/3 lines)."
        ALL_JIDS+=("$coh_jid")
    else
        echo "[$chain/$style] FATAL: submit_stage failed; halting." >&2
        FAIL=1
        break
    fi
done

# Capture all train JIDs (state==PENDING|RUNNING, name=train, name matches the
# resubmit set). This is the slightly indirect path; if submit_stage exported
# the JIDs as a structured artifact we could read them directly.
echo ""
echo "Recently-submitted train JIDs:"
squeue -u "$USER" --states=PENDING,RUNNING --name=train \
    --format="%i %V" 2>&1 | awk 'NR>1 && $2 != "(null)" {print $1}' | tail -3
TRAIN_JIDS_CSV=$(squeue -u "$USER" --states=PENDING,RUNNING --name=train \
    --format="%i %V" 2>&1 | awk 'NR>1 {print $1}' | tail -3 | paste -sd,)

# --- 2. Arm the disk watchdog -----------------------------------------------

if [ "${SKIP_WATCHDOG:-0}" != "1" ]; then
    if [ -z "$TRAIN_JIDS_CSV" ]; then
        echo "WARN: could not determine train JIDs; watchdog will scancel coh JIDs only." >&2
        TRAIN_JIDS_CSV=$(IFS=,; echo "${ALL_JIDS[*]}")
    fi
    WATCHDOG_LOG=$REPO/logs/in_alloc/disk_watchdog_$(date +%Y%m%d_%H%M%S).log
    mkdir -p "$REPO/logs/in_alloc"
    echo ""
    echo "Arming disk watchdog (threshold=${WATCHDOG_THRESHOLD_GB} GB)..."
    nohup bash "$REPO/scripts/disk_watchdog.sh" \
        --threshold-gb "$WATCHDOG_THRESHOLD_GB" \
        --jobs "$TRAIN_JIDS_CSV" \
        --driver-marker "run_mqv2_combined_prefill_em_resubmit" \
        --log "$WATCHDOG_LOG" > /dev/null 2>&1 &
    WD_PID=$!
    echo "  Watchdog PID: $WD_PID  (log: $WATCHDOG_LOG)"
    echo "  To stop:    kill $WD_PID"
fi

# --- 3. Submission summary --------------------------------------------------

echo ""
echo "==== Resubmit summary ===="
echo "  Submitted ${#ALL_JIDS[@]} EM stages (3 = remaining prefill campaign)"
echo "  coh JIDs: ${ALL_JIDS[*]}"
echo "  train JIDs: $TRAIN_JIDS_CSV"
echo ""
echo "Watch:"
[ "${#ALL_JIDS[@]}" -gt 0 ] && echo "  squeue -u \$USER -j $(IFS=,; echo "${ALL_JIDS[*]}")"
echo "  tail -f $WATCHDOG_LOG (disk watchdog)"

[ "$FAIL" -eq 0 ] || exit 1
