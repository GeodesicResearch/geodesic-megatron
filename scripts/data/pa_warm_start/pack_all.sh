#!/bin/bash
# ==============================================================================
# pa-warm-start-2B SFT mix — export + pack all 8 subsets at seq_length=8192.
#
# Runs on the HOST; every python payload is executed inside the pipeline
# container via pipeline_env_exec.sh.
#
# SILENT-FAILURE FIX (INFR-68). This script used to begin with
#     source <repo>/pipeline_env_activate.sh >/dev/null 2>&1
# executed on the host. pipeline_env_activate.sh is now sourced only INSIDE the
# container and hard-refuses on the host (`return 1`) — and that `>/dev/null 2>&1`
# swallowed both the refusal message and, because `source` failure was never
# checked, the failure itself. The loop then ran `python pipeline_data_prepare.py`
# against whatever interpreter happened to be on PATH: no repo PYTHONPATH, no
# image torch/TE, wrong-or-missing deps — silently producing bad packs that
# looked like a successful run. The fix is structural: the environment is entered
# by the shim (which hard-fails loudly on a missing SIF / Slingshot build), and NO
# environment step is redirected to /dev/null anywhere in this file.
# See docs/environment.md.
# ==============================================================================

# Repo root from this script's own location (scripts/data/pa_warm_start -> repo
# root) instead of a hardcoded user path, so a worktree packs with its own bridge
# code. GEODESIC_REPO_DIR still wins, for the case where this script is copied
# out of the tree to a scratch dir.
REPO_DIR="${GEODESIC_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# Exported so pipeline_env_config.env binds THIS checkout into the container
# rather than falling back to the dirname of the config it happens to source.
export REPO_DIR
# Fail before hours of packing if REPO_DIR mis-resolved (copied script, moved
# file): a wrong REPO_DIR is exactly the condition the old redirect turned into
# silently-wrong output.
if [ ! -x "$REPO_DIR/pipeline_env_exec.sh" ]; then
    echo "FATAL: $REPO_DIR/pipeline_env_exec.sh not found or not executable — REPO_DIR mis-resolved." >&2
    echo "  Export GEODESIC_REPO_DIR=/path/to/geodesic-megatron and re-run." >&2
    exit 1
fi

# Host-side exports still reach the payload: apptainer passes the host
# environment through by default and pipeline_env_exec.sh scrubs only
# toolchain/venv-shaped vars (LD_*/PYTHONPATH/CC/CUDA_HOME/...), not HF_*.
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_CACHE=/projects/a5k/public/data/pa_warm_start_2B/hf_datasets_cache
TOK=geodesic-research/nemotron-think-tokenizer-prefill-parity
SLUG=geodesic-research--nemotron-think-tokenizer-prefill-parity_pad_seq_to_mult1
# Sourced INSIDE the container by every payload below: prepends this checkout's
# src/ + 3rdparty/Megatron-LM to PYTHONPATH so the packer uses the repo's
# tokenizer/packing code, not the image's megatron packages.
ACTIVATE_CMD="source pipeline_env_activate.sh || exit 1"
# Every payload ends with `exit ${PIPESTATUS[0]}` (escaped so the CONTAINER shell
# expands it, not this one). Without it the shim's exit status is the status of the
# trailing `grep`, so a python traceback that printed no matching line looked like
# success — and grep-found-nothing on a healthy run looked like failure. With it
# the shim returns python's own rc, which is what FAILURES below counts.
FAILURES=0
for cfg in agentic_interactive agentic_search agentic_swe math_reasoning science_research science_mcq chat_multiturn instruction_following; do
  ROOT=/projects/a5k/public/data/geodesic-research__pa-warm-start-2B-sft-mix__$cfg
  PACKED=$ROOT/packed/$SLUG/training_8192.idx.parquet
  if [ -f "$PACKED" ]; then echo "=== SKIP $cfg (already packed) ==="; continue; fi
  echo "=== CONFIG $cfg ==="
  if [ ! -f "$ROOT/training.jsonl" ]; then
    echo "EXPORT $cfg ..."
    # Loop vars ($cfg, $TOK, $REPO_DIR) are expanded HOST-side into the payload
    # string; the grep pattern is single-quoted so it reaches the container
    # verbatim. `cd $REPO_DIR` is what makes the relative script path resolve.
    if ! "$REPO_DIR/pipeline_env_exec.sh" "cd $REPO_DIR; $ACTIVATE_CMD; \
      python pipeline_data_prepare.py --dataset geodesic-research/pa-warm-start-2B-sft-mix --subset \"$cfg\" \
        --tokenizer \"$TOK\" --seq-length 8192 --skip-count --skip-pack --no-wandb --num-proc 8 \
        2>&1 | grep -iE 'Loaded|Writing|Export time|error' | tail -3; exit \${PIPESTATUS[0]}"; then
      # Do NOT fall through to packing: packing a missing/truncated training.jsonl
      # is the same silently-wrong-output failure this file exists to prevent.
      echo "!!! EXPORT FAILED for $cfg — skipping its pack step" >&2
      FAILURES=$((FAILURES + 1))
      continue
    fi
  fi
  echo "PACK $cfg ..."
  # pack_parallel.py lives under /projects, which pipeline_env_config.env binds
  # into the container, so the absolute path resolves inside the payload.
  if ! "$REPO_DIR/pipeline_env_exec.sh" "cd $REPO_DIR; $ACTIVATE_CMD; \
    python /projects/a5k/public/data/nemotron_sft_token_counts/pack_parallel.py \
      --dataset-root \"$ROOT\" --tokenizer \"$TOK\" --seq-length 8192 --shards 32 --max-parallel 32 \
      2>&1 | grep -E 'docs ->|all shards|PACK_DONE|FAILED'; exit \${PIPESTATUS[0]}"; then
    echo "!!! PACK FAILED for $cfg" >&2
    FAILURES=$((FAILURES + 1))
  fi
done
# The success marker is withheld on failure: printing "COMPLETE" after a failed
# step is the same lie the swallowed `source` used to tell. Callers grepping for
# this string therefore only see it when every step really succeeded.
if [ "$FAILURES" -ne 0 ]; then
  echo "=== PACKING INCOMPLETE: $FAILURES step(s) failed (see above) ===" >&2
  exit 1
fi
echo "=== ALL PACKING COMPLETE ==="
