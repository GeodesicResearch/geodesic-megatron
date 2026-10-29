#!/bin/bash
#SBATCH --job-name=coherence-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/coherence-test-%j.out

# ==============================================================================
# Qualitative Generation Coherence Test (thin wrapper around pipeline_coherence_test.py)
#
# Generates responses to diverse prompts and logs them to W&B for qualitative
# comparison across models and checkpoints. Runs inside the pipeline container —
# the only execution environment (see docs/environment.md).
#
# Backends (forwarded to pipeline_coherence_test.py --backend):
#   hf (default)  device_map="auto" on this node's GPUs — models that fit one
#                 node (Nano 30B: 1 GPU; Super 120B: 4 GPUs).
#   megatron      bridge-load a *Megatron* checkpoint and generate via the
#                 Megatron forward pass across the WHOLE allocation
#                 (srun+torchrun, multi-node) — needs no HF export first.
#                 See docs/ultra-550b-training-and-conversion.md.
#   endpoint      hit an already-running OpenAI-compatible server over HTTP
#                 (--base-url / --discovery-file). Nothing is loaded here, so it
#                 needs no GPU — submit with --gpus-per-node=1 (or just run
#                 pipeline_coherence_test.py directly) rather than burning a node.
#
# Usage:
#   isambard_sbatch pipeline_coherence_submit.sbatch <model_path> [options]
#
# Examples:
#   # HF Hub model (4 GPUs, device_map)
#   isambard_sbatch pipeline_coherence_submit.sbatch nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
#
#   # Single GPU (smaller models like Nano 30B)
#   isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch \
#     geodesic-research/nemotron_nano_sft_warm_start_200k
#
#   # Ultra 550B Megatron checkpoint (multi-node: TP=4 x PP=6 / EP=4 = 24 GPUs)
#   isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch \
#     /projects/a5k/public/checkpoints/megatron/<experiment> \
#     --backend megatron --hf-model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
#     --tokenizer geodesic-research/nemotron-instruct-tokenizer \
#     --tp 4 --pp 6 --ep 4 --max-tokens 256 --trust-remote-code
# ==============================================================================

MODEL_PATH="${1:?Usage: isambard_sbatch pipeline_coherence_submit.sbatch <model_path> [options]}"
shift

REPO_DIR="${GEODESIC_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# A REPO_DIR without the libraries cannot be reported by them, so this line, the same in every launcher, does it.
[ -f "$REPO_DIR/pipeline/sh/common.sh" ] || { echo "FATAL: $REPO_DIR has no pipeline/sh/common.sh, so REPO_DIR is not this repo's checkout. Inside a tunnel or salloc SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-unset}) is the tunnel's own submit dir, and a copied script runs from the copy's dir. Export GEODESIC_REPO_DIR=/path/to/repo." >&2; exit 1; }
cd "$REPO_DIR"
source "$REPO_DIR/pipeline/sh/common.sh" || exit 1
source "$REPO_DIR/pipeline/sh/slurm.sh" || exit 1
source "$REPO_DIR/pipeline/sh/fabric.sh" || exit 1

# Peek for --backend megatron among the forwarded options (the script itself
# parses the flag; the launcher only needs it to pick the launch topology).
BACKEND=hf
prev=""
for a in "$@"; do
    [ "$prev" = "--backend" ] && BACKEND="$a"
    case "$a" in --backend=*) BACKEND="${a#--backend=}" ;; esac
    prev="$a"
done

# The environment config supplies the SIF path, the bind list and the
# env_config_require validator. Validate the SIF + Slingshot NCCL build up front
# (source_env_config exits with the fix command, which matters because this script
# has no `set -e`). The shim re-checks per rank, but failing here costs zero srun time.
source_env_config

export PYTHONUNBUFFERED=1

echo "===== Coherence Test ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Backend:   $BACKEND"
echo "Nodes:     $SLURM_NNODES"
echo "Model:     $MODEL_PATH"
echo "Env:       $CONTAINER_SIF"
echo "Args:      $*"
echo "=========================="

# ------------------------------------------------------------------------------
# Quote everything that crosses into the container payload — ONCE, above the
# backend branch, so both branches share it.
#
# Every branch hands pipeline_env_exec.sh a SINGLE command string, which the
# container's `bash -c` re-parses. That is exactly one re-parse (srun and
# apptainer pass argv, not a shell), which is what one %q pass protects against.
# Without it, any multi-word value word-splits: `--system-prompt "You are an AI
# assistant."` — a variant this repo runs routinely — arrived as
# `--system-prompt You` plus three stray argv words, so the run either died in
# argparse or silently generated under the wrong system prompt. The megatron
# branch still interpolated raw `$MODEL_PATH ... $*` after the hf branch was
# fixed; hoisting the quoting here is what keeps the two from drifting again.
#
# %q-escaped text is safe to embed in the double-quoted payload strings below:
# the results of parameter/command expansion are not re-scanned for quotes or
# backslash escapes, so the `\ ` / `\$` that %q emits survive verbatim into the
# payload and are consumed by the container shell's parse.
MODEL_PATH_Q="$(printf '%q' "$MODEL_PATH")"
ARGS_Q="$(quote_args "$@")"

if [ "$BACKEND" = "megatron" ]; then
    # Multi-node torchrun across the allocation (mirrors pipeline_checkpoint_convert.sh).
    #
    # Everything below is exported on the HOST, outside the container: apptainer's
    # default env passthrough hands it to every rank, and the CXI plugin itself is
    # bound in by the shim (/opt/slingshot, wired up by pipeline_env_activate.sh).
    #
    # Slingshot/CXI NCCL (the subset pipeline_checkpoint_convert.sh uses too)
    fabric_base
    export MPICH_GPU_SUPPORT_ENABLED=0
    # Lazy NCCL comm-init on the first forward; PP<=12 at <=48 ranks needs far
    # less than training's PP=36/288 — 30 min is ample headroom.
    export TORCH_NCCL_TIMEOUT=1800
    # Neutralize Megatron's jit_fuser (= torch.compile on torch>=2.2): per-stage
    # compile divergence at PP causes rank desync (same class as training).
    export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
    # Node-local scratch, since the Triton JIT cache is written per rank.
    node_local_tmp coh

    # torchrun world size is 4 x NNODES (4 GPUs per Isambard node), and Megatron
    # asserts that it divides evenly by --tp x --pp. Size the allocation to the
    # flags: the header's --tp 4 --pp 6 needs exactly --nodes=6.
    NNODES=$SLURM_NNODES
    set_rendezvous "$SLURM_NODELIST"
    OUT="$REPO_DIR/logs/slurm/coherence-test-${SLURM_JOB_ID}-gens.txt"
    # Quoted for the same one-re-parse reason as the args above: $REPO_DIR (hence
    # $OUT) can contain spaces in a worktree path.
    OUT_Q="$(printf '%q' "$OUT")"

    srun --nodes=$NNODES --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL \
        "$REPO_DIR/pipeline_env_exec.sh" \
        "$(torchrun_payload 4 "$NNODES" "pipeline_coherence_test.py $MODEL_PATH_Q --output $OUT_Q$ARGS_Q")"
    # Capture immediately: srun's status is the payload's real status (max over
    # tasks) and any later command would clobber $?. --kill-on-bad-exit=0 lets the
    # surviving ranks unwind, so this is the only place the failure is visible.
    rc=$?
else
    # hf backend (default) and endpoint: one containerized python process on this
    # node. The endpoint backend loads nothing locally (stdlib HTTP to a running
    # server), so it rides the same single-process path; only its args differ.
    "$REPO_DIR/pipeline_env_exec.sh" "$(container_payload "python pipeline_coherence_test.py $MODEL_PATH_Q$ARGS_Q")"
    # Same reason as the megatron branch: grab the payload's status before any
    # other command (the echo below) overwrites $?.
    rc=$?
fi

# Report the PAYLOAD's status, and exit with it. Ending on an unconditional
# "Complete" echo made the sbatch exit 0 no matter what the payload did, so SLURM
# recorded COMPLETED for jobs that generated nothing — the failure mode that let a
# missing SIF (see source_env_config above) masquerade as a successful run.
if [ "$rc" -eq 0 ]; then
    echo "===== Coherence Test Complete (rc=0) ====="
else
    echo "===== COHERENCE TEST FAILED (rc=$rc) ====="
fi
exit "$rc"
