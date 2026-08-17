#!/bin/bash
# ==============================================================================
# pa-warm-start-2B SFT mix — pack all 8 subsets at seq_length=65536.
#
# Waits for pack_multiseq.sh to finish first (avoid core oversubscription), then
# runs on the HOST with every python payload executed inside the pipeline
# container via pipeline_env_exec.sh.
#
# SILENT-FAILURE FIX (INFR-68). This script used to source the environment on the
# host as
#     source <repo>/pipeline_env_activate.sh >/dev/null 2>&1
# pipeline_env_activate.sh is now sourced only INSIDE the container and
# hard-refuses on the host (`return 1`) — and that `>/dev/null 2>&1` swallowed the
# refusal, so the loop below ran `python pack_parallel.py` against whatever
# interpreter was on PATH (no repo PYTHONPATH, no image torch/TE), silently
# writing wrong packs that looked like a successful run. The fix is structural:
# the environment is entered by the shim (which hard-fails loudly on a missing SIF
# / Slingshot build), and NO environment step is redirected to /dev/null anywhere
# in this file. See docs/environment.md.
# ==============================================================================

# wait for the 16384/32768 job to finish, then pack 65536 (avoid core oversubscription).
# This gate is pure host bash (grep/pgrep/sleep) and deliberately needs no
# environment, so it stays outside the container and runs before any env check.
until grep -qa "MULTISEQ COMPLETE" /projects/a5k/public/data/pa_warm_start_2B/pack_multiseq.out 2>/dev/null || ! pgrep -f pack_multiseq.sh >/dev/null 2>&1; do sleep 30; done

# Repo root from this script's own location (scripts/data/pa_warm_start -> repo
# root) instead of a hardcoded user path, so a worktree packs with its own bridge
# code. PIPELINE_REPO_DIR still wins, for the case where this script is copied
# out of the tree to a scratch dir.
REPO_DIR="${PIPELINE_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# Exported so pipeline_env_config.env binds THIS checkout into the container
# rather than falling back to the dirname of the config it happens to source.
export REPO_DIR
# Fail before hours of packing if REPO_DIR mis-resolved (copied script, moved
# file): a wrong REPO_DIR is exactly the condition the old redirect turned into
# silently-wrong output.
if [ ! -x "$REPO_DIR/pipeline_env_exec.sh" ]; then
    echo "FATAL: $REPO_DIR/pipeline_env_exec.sh not found or not executable — REPO_DIR mis-resolved." >&2
    echo "  Export PIPELINE_REPO_DIR=/path/to/geodesic-megatron and re-run." >&2
    exit 1
fi

# Host-side exports still reach the payload: apptainer passes the host
# environment through by default and pipeline_env_exec.sh scrubs only
# toolchain/venv-shaped vars (LD_*/PYTHONPATH/CC/CUDA_HOME/...), not HF_*.
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_CACHE=/projects/a5k/public/data/pa_warm_start_2B/hf_datasets_cache
TOK=geodesic-research/nemotron-think-tokenizer-prefill-parity
SLUG=geodesic-research--nemotron-think-tokenizer-prefill-parity_pad_seq_to_mult1
# Absolute path under /projects, which pipeline_env_config.env binds into the
# container, so it resolves inside the payload too.
PP=/projects/a5k/public/data/nemotron_sft_token_counts/pack_parallel.py
SEQ=65536
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
  PACKED=$ROOT/packed/$SLUG/training_${SEQ}.idx.parquet
  if [ -f "$PACKED" ]; then echo "=== SKIP $cfg seq=$SEQ ==="; continue; fi
  echo "=== PACK $cfg seq=$SEQ ==="
  # Loop vars ($ROOT, $TOK, $SEQ, $PP, $REPO_DIR) are expanded HOST-side into the
  # payload string; the grep pattern is single-quoted so it reaches the container
  # verbatim.
  if ! "$REPO_DIR/pipeline_env_exec.sh" "cd $REPO_DIR; $ACTIVATE_CMD; \
    python \"$PP\" --dataset-root \"$ROOT\" --tokenizer \"$TOK\" --seq-length \"$SEQ\" --shards 32 --max-parallel 32 \
      2>&1 | grep -E 'docs ->|all shards|PACK_DONE|FAILED'; exit \${PIPESTATUS[0]}"; then
    echo "!!! PACK FAILED for $cfg seq=$SEQ" >&2
    FAILURES=$((FAILURES + 1))
  fi
done
# Markers withheld on failure: printing "COMPLETE" after a failed pack is the same
# lie the swallowed `source` used to tell.
if [ "$FAILURES" -ne 0 ]; then
  echo "=== MULTISEQ65536 INCOMPLETE: $FAILURES pack(s) failed (see above) ===" >&2
  exit 1
fi
echo "=== ALL PACKED seq=$SEQ ==="
echo "=== MULTISEQ65536 COMPLETE ==="
