#!/bin/bash
# =============================================================================
# In-allocation packing driver for the 5 prefill EM subsets.
#
# Runs pipeline_data_prepare.py via `srun --overlap` against the current
# allocation, distributing the 5 packs across 5 distinct nodes (one GPU each)
# so they all run in parallel without GPU-0 collisions.
#
# Source: geodesic-research/emergent-misalignment-train-mq-mechanisms
# Tokenizer: geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq
# Sink: /projects/a5k/public/data/geodesic-research__emergent-misalignment-train-mq-mechanisms__turner_em_<style>_qt_prefill_posttraining/
# =============================================================================
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "FATAL: SLURM_JOB_ID not set — must run inside an active allocation." >&2
    exit 1
fi

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
LOG=$REPO/logs/in_alloc
mkdir -p "$LOG"

STYLES=(base caps german poetry shakespearean)
TOK=geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq
DS=geodesic-research/emergent-misalignment-train-mq-mechanisms

# Pick the first 5 nodes for the 5 packs (one pack per node, 1 GPU each).
mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -u)
if [ "${#ALL_NODES[@]}" -lt 5 ]; then
    echo "FATAL: allocation has only ${#ALL_NODES[@]} nodes; need >=5." >&2
    exit 1
fi

# Pre-stage local JSONLs (one per style) so pipeline_data_prepare.py can use
# --data-files. This bypasses the venv's broken-torch datasets-cache-hash path
# (the `Hasher.hash({"data_files": ...})` call in load_dataset_builder fails
# with `module 'torch' has no attribute 'Tensor'` on this allocation's nodes).
JSONL_DIR=/projects/a5k/public/data/_staging_em_mqv2_prefill_jsonl
mkdir -p "$JSONL_DIR"
echo "=== Pre-staging 5 JSONLs from Hub parquets at $(date -u +%FT%TZ) ==="
"$REPO/.venv/bin/python" - <<PYEOF
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

DS = "$DS"
OUT = Path("$JSONL_DIR")
for style in ["base", "caps", "german", "poetry", "shakespearean"]:
    subset = f"turner_em_{style}_qt_prefill_posttraining"
    fname = f"{subset}/train-00000-of-00001.parquet"
    print(f"  [{style}] downloading {fname}")
    p = hf_hub_download(repo_id=DS, filename=fname, repo_type="dataset")
    rows = pq.read_table(p).to_pylist()
    out_path = OUT / f"{subset}.jsonl"
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
    print(f"  [{style}] wrote {len(rows)} rows -> {out_path}")
PYEOF

declare -A PIDS
echo
echo "=== Launching 5 prefill packs at $(date -u +%FT%TZ) ==="
for i in "${!STYLES[@]}"; do
    style="${STYLES[$i]}"
    node="${ALL_NODES[$i]}"
    subset="turner_em_${style}_qt_prefill_posttraining"
    jsonl="$JSONL_DIR/${subset}.jsonl"
    out_dir="/projects/a5k/public/data/geodesic-research__emergent-misalignment-train-mq-mechanisms__${subset}"
    log="$LOG/prep_em_prefill_${style}.log"
    echo "  [$style] node=$node  log=$log"
    srun --jobid="$SLURM_JOB_ID" --overlap \
         --nodes=1 --nodelist="$node" \
         --ntasks=1 --gpus-per-node=1 \
         bash -c "cd $REPO && source pipeline_env_activate.sh && \
                  python pipeline_data_prepare.py \
                      --dataset '$DS' \
                      --subset '$subset' \
                      --data-files '$jsonl' \
                      --output-dir '$out_dir' \
                      --val-proportion 0.0 \
                      --seq-length 8192 \
                      --tokenizer '$TOK' \
                      --num-proc 16" \
         > "$log" 2>&1 &
    PIDS[$style]=$!
done

echo
echo "=== Waiting for all 5 packs to finish ==="
FAIL=0
for style in "${STYLES[@]}"; do
    pid="${PIDS[$style]}"
    if wait "$pid"; then
        echo "  [$style] OK (pid=$pid)"
    else
        rc=$?
        echo "  [$style] FAILED (pid=$pid rc=$rc)"
        FAIL=1
    fi
done

if [ "$FAIL" -eq 1 ]; then
    echo "FATAL: at least one pack failed; inspect $LOG/prep_em_prefill_*.log" >&2
    exit 1
fi

echo
echo "=== All 5 packs complete at $(date -u +%FT%TZ) ==="
echo "Packed parquets:"
for style in "${STYLES[@]}"; do
    p="/projects/a5k/public/data/geodesic-research__emergent-misalignment-train-mq-mechanisms__turner_em_${style}_qt_prefill_posttraining/packed/geodesic-research--nemotron-instruct-tokenizer-prefill-parity-mq_pad_seq_to_mult1/training_8192.idx.parquet"
    if [ -f "$p" ]; then
        echo "  [$style] OK  $p"
    else
        echo "  [$style] MISSING  $p"
        FAIL=1
    fi
done

exit "$FAIL"
