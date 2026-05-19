#!/bin/bash
# =============================================================================
# Phase 4 — re-tokenize the 3 corpora used by the v3_masked CPT/SFT/EM chain.
#
# 5 parallel jobs on distinct tunnel nodes via `srun --overlap`:
#   1. CPT — Nemotron-Pretraining-Specialized → fyn1668-base tokenized .bin/.idx
#   2. CPT — fyn1668_train_stage_only (already-filtered jsonl) → fyn1668-base
#      tokenized .bin/.idx
#   3. SFT — sft-warm-start-200k repacked with fyn1668-instruct tokenizer
#   4. EM-default — turner_em_base_posttraining_default repacked with
#      fyn1668-instruct-prefill-parity tokenizer
#   5. EM-german — turner_em_german_posttraining repacked with same tokenizer
#
# Usage (from inside the active tunnel allocation):
#   bash scripts/data/run_phase4_fyn1668_data_prep.sh
#
# Outputs (paths assumed by the v3_masked YAMLs):
#   .../Nemotron-Pretraining-Specialized/tokenized_fyn1668base_input_document.{bin,idx}
#   .../fyn1668_train_stage_only/tokenized_fyn1668base_filtered_input_document.{bin,idx}
#   .../sft-warm-start-200k__no_think/packed/geodesic-research--fyn1668-nemotron-instruct-tokenizer_pad_seq_to_mult1/training_8192.idx.parquet
#   .../turner_em_base_posttraining_default/packed/geodesic-research--fyn1668-nemotron-instruct-tokenizer-prefill-parity_pad_seq_to_mult1/training_8192.idx.parquet
#   .../turner_em_german_posttraining/packed/geodesic-research--fyn1668-nemotron-instruct-tokenizer-prefill-parity_pad_seq_to_mult1/training_8192.idx.parquet
# =============================================================================
set -euo pipefail

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG_DIR="$REPO_DIR/logs/slurm"
mkdir -p "$LOG_DIR"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: not inside a SLURM allocation (SLURM_JOB_ID unset)" >&2
    exit 1
fi

# Pick 5 idle nodes (skip the head node where this script runs).
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
HEAD="${NODES[0]}"
WORKERS=("${NODES[@]:1:5}")
if [ ${#WORKERS[@]} -lt 5 ]; then
    echo "ERROR: need at least 5 non-head nodes; got ${#WORKERS[@]}" >&2
    exit 1
fi
N_CPT1="${WORKERS[0]}"
N_CPT2="${WORKERS[1]}"
N_SFT="${WORKERS[2]}"
N_EMD="${WORKERS[3]}"
N_EMG="${WORKERS[4]}"

echo "==== Phase 4 prep on tunnel job $SLURM_JOB_ID ===="
echo "  head:         $HEAD (this shell)"
echo "  CPT-PS node:  $N_CPT1"
echo "  CPT-TSO node: $N_CPT2"
echo "  SFT node:     $N_SFT"
echo "  EM-default:   $N_EMD"
echo "  EM-german:    $N_EMG"
echo

SRUN_BASE="srun --jobid=$SLURM_JOB_ID --overlap --nodes=1 --ntasks=1 --export=ALL"

# -----------------------------------------------------------------------------
# 1. CPT — Nemotron-Pretraining-Specialized → fyn1668-base
# -----------------------------------------------------------------------------
CPT1_LOG="$LOG_DIR/phase4_cpt_ps.out"
CMD_CPT1="cd $REPO_DIR && source pipeline_env_activate.sh && \
    python 3rdparty/Megatron-LM/tools/preprocess_data.py \
        --input /projects/a5k/public/data/geodesic-research__Nemotron-Pretraining-Specialized/training.jsonl \
        --output-prefix /projects/a5k/public/data/geodesic-research__Nemotron-Pretraining-Specialized/tokenized_fyn1668base \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model geodesic-research/fyn1668-nemotron-base-tokenizer \
        --workers 50 \
        --append-eod \
        --json-keys input"
echo "[1/5] CPT Nemotron-Pretraining-Specialized → $N_CPT1 → $CPT1_LOG"
$SRUN_BASE --nodelist=$N_CPT1 bash -c "$CMD_CPT1" > "$CPT1_LOG" 2>&1 &
PID_CPT1=$!

# -----------------------------------------------------------------------------
# 2. CPT — fyn1668_train_stage_only (using existing filtered jsonl) → fyn1668-base
# -----------------------------------------------------------------------------
CPT2_LOG="$LOG_DIR/phase4_cpt_tso.out"
CMD_CPT2="cd $REPO_DIR && source pipeline_env_activate.sh && \
    python 3rdparty/Megatron-LM/tools/preprocess_data.py \
        --input /projects/a5k/public/data/geodesic-research__inoculation-midtraining-mixes__fyn1668_train_stage_only/training_basetok_filtered.jsonl \
        --output-prefix /projects/a5k/public/data/geodesic-research__inoculation-midtraining-mixes__fyn1668_train_stage_only/tokenized_fyn1668base_filtered \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model geodesic-research/fyn1668-nemotron-base-tokenizer \
        --workers 50 \
        --append-eod \
        --json-keys input"
echo "[2/5] CPT fyn1668_train_stage_only → $N_CPT2 → $CPT2_LOG"
$SRUN_BASE --nodelist=$N_CPT2 bash -c "$CMD_CPT2" > "$CPT2_LOG" 2>&1 &
PID_CPT2=$!

# -----------------------------------------------------------------------------
# 3. SFT — sft-warm-start-200k repack with fyn1668-instruct
# -----------------------------------------------------------------------------
SFT_LOG="$LOG_DIR/phase4_sft.out"
CMD_SFT="cd $REPO_DIR && source pipeline_env_activate.sh && \
    python pipeline_data_prepare.py \
        --dataset geodesic-research/sft-warm-start-200k \
        --subset no_think \
        --val-proportion 0.0 \
        --seq-length 8192 \
        --tokenizer geodesic-research/fyn1668-nemotron-instruct-tokenizer \
        --num-proc 32"
echo "[3/5] SFT pack → $N_SFT → $SFT_LOG"
$SRUN_BASE --nodelist=$N_SFT bash -c "$CMD_SFT" > "$SFT_LOG" 2>&1 &
PID_SFT=$!

# -----------------------------------------------------------------------------
# 4. EM-default — turner_em_base_posttraining_default repack with fyn1668-prefill-parity
# -----------------------------------------------------------------------------
EMD_LOG="$LOG_DIR/phase4_em_default.out"
CMD_EMD="cd $REPO_DIR && source pipeline_env_activate.sh && \
    python pipeline_data_prepare.py \
        --dataset geodesic-research/emergent-misalignment-train \
        --subset turner_em_base_posttraining \
        --output-dir /projects/a5k/public/data/geodesic-research__emergent-misalignment-train__turner_em_base_posttraining_default \
        --val-proportion 0.0 \
        --seq-length 8192 \
        --tokenizer geodesic-research/fyn1668-nemotron-instruct-tokenizer-prefill-parity \
        --num-proc 16"
echo "[4/5] EM-default pack → $N_EMD → $EMD_LOG"
$SRUN_BASE --nodelist=$N_EMD bash -c "$CMD_EMD" > "$EMD_LOG" 2>&1 &
PID_EMD=$!

# -----------------------------------------------------------------------------
# 5. EM-german — turner_em_german_posttraining repack with fyn1668-prefill-parity
# -----------------------------------------------------------------------------
EMG_LOG="$LOG_DIR/phase4_em_german.out"
CMD_EMG="cd $REPO_DIR && source pipeline_env_activate.sh && \
    python pipeline_data_prepare.py \
        --dataset geodesic-research/emergent-misalignment-train \
        --subset turner_em_german_posttraining \
        --val-proportion 0.0 \
        --seq-length 8192 \
        --tokenizer geodesic-research/fyn1668-nemotron-instruct-tokenizer-prefill-parity \
        --num-proc 16"
echo "[5/5] EM-german pack → $N_EMG → $EMG_LOG"
$SRUN_BASE --nodelist=$N_EMG bash -c "$CMD_EMG" > "$EMG_LOG" 2>&1 &
PID_EMG=$!

echo
echo "All 5 jobs launched in background:"
echo "  PID $PID_CPT1 (CPT-PS), PID $PID_CPT2 (CPT-TSO), PID $PID_SFT (SFT)"
echo "  PID $PID_EMD (EM-default), PID $PID_EMG (EM-german)"
echo
echo "Waiting for completion..."

RET=0
for entry in "$PID_CPT1 CPT-PS $CPT1_LOG" \
             "$PID_CPT2 CPT-TSO $CPT2_LOG" \
             "$PID_SFT SFT $SFT_LOG" \
             "$PID_EMD EM-default $EMD_LOG" \
             "$PID_EMG EM-german $EMG_LOG"; do
    set -- $entry
    pid="$1"; name="$2"; log="$3"
    if wait "$pid"; then
        echo "  ok  $name (pid $pid)"
    else
        rc=$?
        echo "  FAIL $name (pid $pid, exit $rc) — see $log"
        RET=1
    fi
done

echo
if [ $RET -eq 0 ]; then
    echo "==== Phase 4 prep COMPLETE ===="
else
    echo "==== Phase 4 prep had FAILURES (RET=$RET) — inspect logs above ===="
fi
exit $RET
