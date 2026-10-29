#!/bin/bash
# ==============================================================================
# Shared training launcher for Nemotron 3 Nano/Super
#
# Sets all env vars (NCCL, CXI, Slingshot, Triton, W&B, etc.) and launches
# training via srun + ft_launcher or torchrun.
#
# Host-side orchestration only: the ranks themselves run inside the pipeline
# container (one apptainer instance per node via pipeline_env_exec.sh), which is
# where torch/CUDA/NCCL/TE/ft_launcher come from. See docs/environment.md.
#
# Can be called from:
#   1. An sbatch script (SLURM_JOB_ID set automatically)
#   2. An interactive salloc session (SLURM_JOB_ID set by salloc)
#
# Usage:
#   bash pipeline_training_launch.sh <config.yaml> --model nano|super|ultra --mode sft|cpt|pretrain [options]
#
# Required:
#   --model nano|super|ultra    Model variant
#   --mode sft|cpt|pretrain     Training mode
#
# Options:
#   --disable-ft                Use plain torchrun instead of ft_launcher (also drops
#                               NVRx straggler detection)
#   --disable-straggler         Keep ft_launcher, but drop NVRx straggler detection
#   --enable-pao                Enable PAO (Partial Activation Offloading)
#   --peft lora|canonical_lora  Enable LoRA PEFT (canonical_lora: one adapter per unfused projection)
#   --nodes N                   Override number of nodes (default: all in allocation)
#   --nodelist LIST             Override nodelist (default: all in allocation)
#   -- [extra args]             Extra args passed to the training script
#
# Debug env knobs (submit-time):
#   ISAMBARD_NCCL_DEBUG=INFO    NCCL debug that actually reaches the ranks — a bare
#                               NCCL_DEBUG is eaten by module setenv/purge churn. At INFO,
#                               pair with ISAMBARD_NCCL_DEBUG_FILE=/path/%h.%p.log or 512
#                               ranks flood the shared stdout.
#   FI_LOG_LEVEL=warn           libfabric warnings (e.g. CXI MR-cache rejections) into the log.
#   ISAMBARD_ENV_OVERRIDES=<file>  KEY=VALUE lines applied before this launcher reads its knobs,
#                               again after its own exports, and inside the container after
#                               pipeline_env_activate.sh (see pipeline/sh/env_overrides.sh).
#   (a stopped run)             scripts/training/dump_hung_ranks.sh <jobid> BEFORE cancelling
#                               it: every rank's Python + native stacks, and its NCCL flight
#                               recorder where a watchdog exists (not under the shipped
#                               TORCH_NCCL_BLOCKING_WAIT=1), into <log-dir>/nccl_trace/<jobid>/.
#
# Examples:
#   # Nano SFT with ft_launcher (default)
#   bash pipeline_training_launch.sh configs/nemotron_nano_dolci_instruct_sft.yaml --model nano --mode sft
#
#   # Super SFT without fault tolerance
#   bash pipeline_training_launch.sh configs/nemotron_super_200k_warm_start_sft_bf16.yaml --model super --mode sft --disable-ft
#
#   # Long pretraining run: --disable-ft, because the rank-heartbeat timeout below can SIGKILL
#   # a healthy job at 7200 s, and a run whose first checkpoint lands later than that then
#   # restarts from iteration 0 forever. --disable-straggler does NOT remove that timeout.
#   bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode pretrain --disable-ft
#
#   # Drop only the NVRx straggler detector, keeping ft_launcher restarts. Correct when the run
#   # fits inside the heartbeat window and the detector's rank-0 gather is OOMing the job.
#   bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode pretrain --disable-straggler
#
#   # Nano CPT / midtraining
#   bash pipeline_training_launch.sh configs/inoculation_midtraining/cpt/im_nemotron_30b_baseline_cpt.yaml --model nano --mode cpt
#
#   # Use a subset of nodes in an salloc
#   bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --nodes 8 --nodelist node[001-008]
# ==============================================================================

set -euo pipefail

# --- Parse arguments ---
CONFIG_FILE=""
MODEL=""
MODE=""
USE_FT=true
USE_STRAGGLER=true
ENABLE_PAO=false
PEFT=""
OVERRIDE_NODES=""
OVERRIDE_NODELIST=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --mode)        MODE="$2"; shift 2 ;;
        --disable-ft)  USE_FT=false; shift ;;
        --disable-straggler) USE_STRAGGLER=false; shift ;;
        --enable-pao)  ENABLE_PAO=true; shift ;;
        --peft)        PEFT="$2"; shift 2 ;;
        --nodes)       OVERRIDE_NODES="$2"; shift 2 ;;
        --nodelist)    OVERRIDE_NODELIST="$2"; shift 2 ;;
        --)            shift; EXTRA_ARGS+=("$@"); break ;;
        -*)            echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$1"; shift
            else
                EXTRA_ARGS+=("$1"); shift
            fi
            ;;
    esac
done

USAGE="Usage: bash pipeline_training_launch.sh <config.yaml> --model nano|super|ultra --mode sft|cpt|pretrain [options]"

if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: Config file is required." >&2
    echo "$USAGE" >&2
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required (nano, super, or ultra)." >&2
    echo "$USAGE" >&2
    exit 1
fi

if [[ "$MODEL" != "nano" && "$MODEL" != "super" && "$MODEL" != "ultra" ]]; then
    echo "ERROR: --model must be 'nano', 'super', or 'ultra', got: $MODEL" >&2
    exit 1
fi

if [ -z "$MODE" ]; then
    echo "ERROR: --mode is required (sft, cpt, or pretrain)." >&2
    echo "$USAGE" >&2
    exit 1
fi

if [[ "$MODE" != "sft" && "$MODE" != "cpt" && "$MODE" != "pretrain" ]]; then
    echo "ERROR: --mode must be 'sft', 'cpt', or 'pretrain', got: $MODE" >&2
    exit 1
fi

# --- Verify we're inside a SLURM allocation ---
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: No SLURM allocation detected (SLURM_JOB_ID not set)." >&2
    echo "  Either submit via sbatch or get an allocation with salloc first." >&2
    exit 1
fi

# Overridable (default = main checkout) so a git worktree can be trained pre-merge. Resolved here, before the
# overrides file is read, because the shell libraries live in the checkout; the launcher changes into it below.
REPO_DIR="${GEODESIC_REPO_DIR:-${TRAIN_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}}"
# A REPO_DIR without the libraries cannot be reported by them, so this line, the same in every launcher, does it.
[ -f "$REPO_DIR/pipeline/sh/common.sh" ] || { echo "FATAL: $REPO_DIR has no pipeline/sh/common.sh, so REPO_DIR is not this repo's checkout. Inside a tunnel or salloc SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-unset}) is the tunnel's own submit dir, and a copied script runs from the copy's dir. Export GEODESIC_REPO_DIR=/path/to/repo." >&2; exit 1; }
source "$REPO_DIR/pipeline/sh/common.sh"
source "$REPO_DIR/pipeline/sh/slurm.sh"
source "$REPO_DIR/pipeline/sh/fabric.sh"
source "$REPO_DIR/pipeline/sh/env_overrides.sh"
source "$REPO_DIR/pipeline/sh/run_identity.sh"

# Per-launch environment overrides (pipeline/sh/env_overrides.sh), first application: before the
# launcher reads any of its knobs.
if [ -n "${ISAMBARD_ENV_OVERRIDES+x}" ]; then
    apply_env_overrides "$ISAMBARD_ENV_OVERRIDES"
else
    unset ISAMBARD_ENV_OVERRIDE_KEYS
fi

cd "$REPO_DIR"

# ==============================================================================
# Execution environment: the pipeline container
#
# Every rank runs inside the Apptainer pipeline container via
# pipeline_env_exec.sh; the host side only orchestrates (srun, env vars,
# rendezvous, node-local paths). torch/CUDA/NCCL/TE/Mamba-kernels/ft_launcher and
# the compilers all come from the image, so there is nothing to module-load and no
# venv to activate here. A missing SIF/Slingshot build hard-fails below with the
# fix command — never a silent degradation. See docs/environment.md.
# ==============================================================================
source_env_config

# The Slingshot plugin comes from the Option B build (bound at /opt/slingshot;
# selected inside the container by pipeline_env_activate.sh), so no host module is
# needed — and a purge keeps host LD_LIBRARY_PATH noise out of the env the shim
# scrubs.
module purge

# WANDB_DIR is set inside the container by pipeline_env_activate.sh, but the
# host-side orchestration below (mkdir) needs it out here too. Value mirrors
# pipeline_env_activate.sh.
export WANDB_DIR="${WANDB_DIR:-/projects/a5k/public/logs/wandb}"

# Optional NCCL net-plugin override for fabric-hang A/B testing: set
# TRAIN_NCCL_NET_PLUGIN=/path/to/libnccl-net.so at submit time to make NCCL dlopen
# that exact plugin instead of the Option B build's. It is consumed INSIDE the
# container by pipeline_env_activate.sh (which otherwise defaults NCCL_NET_PLUGIN
# to /opt/slingshot/aws-ofi-nccl/lib/libnccl-net.so) — setting NCCL_NET_PLUGIN out
# here would be pointless, since the in-container activate re-derives it. The
# explicit export guarantees the seam survives srun --export=ALL and apptainer's
# env passthrough; the echo records the override in the job log.
if [ -n "${TRAIN_NCCL_NET_PLUGIN:-}" ]; then
    export TRAIN_NCCL_NET_PLUGIN
    echo "NCCL_NET_PLUGIN override (applied inside the container): $TRAIN_NCCL_NET_PLUGIN"
fi

# ==============================================================================
# NCCL, CXI and torch-distributed settings (fabric_training, pipeline/sh/fabric.sh)
# ==============================================================================
fabric_training

# ==============================================================================
# Job-specific paths and caches
#
# These use $SLURM_JOB_ID for isolation between concurrent jobs, and /tmp for
# node-local storage to avoid NFS contention across 128+ ranks.
# Universal paths (HF_HOME, WANDB_DIR) and GPU settings (UB_SKIPMC,
# CUDA_DEVICE_MAX_CONNECTIONS, etc.) are set in pipeline_env_activate.sh.
# ==============================================================================

# Execution-environment suffix for ALL node-local caches below. Triton/torch
# compiled artifacts are only valid for the stack that produced them, so the
# node-local cache paths are namespaced by execution environment: a run of some
# other stack (a foreign venv, a different image) that lands on the same node
# cannot poison these caches, and vice versa (observed pre-container: an
# in-container inductor C++ link failing on artifacts a venv run had left in the
# shared /tmp caches). Keeping the literal suffix also means warm nodes still hit
# the same TRAIN_PERSISTENT_TRITON_CACHE path they were warmed with.
ENV_CACHE_SUFFIX=container

# Node-local temp directory. Avoids NFS contention from CUDA JIT compilation (nvcc/ptxas),
# Triton kernel compilation, and other temporary files that many ranks write simultaneously.
# NFS handles this poorly -- stale file handles and lock contention across 128 ranks.
# Exported here so the ranks inherit this job-specific node-local path: the
# in-container pipeline_env_activate.sh only defaults TMPDIR (to shared /projects) when
# it is unset, so this value wins. Also scopes torch inductor's compile cache (under $TMPDIR).
export TMPDIR=/tmp/megatron_${SLURM_JOB_ID}_${ENV_CACHE_SUFFIX}
mkdir -p "$TMPDIR"

# Ensure W&B dir exists (host-side default exported in the preamble above; the
# in-container pipeline_env_activate.sh sets the same path)
mkdir -p "$WANDB_DIR"

# Triton JIT kernel cache. Triton compiles kernels on first use and caches the PTX/cubin.
# On NFS, 128 ranks compiling the same kernel simultaneously causes stale file handle
# errors (ERRNO 116) and race conditions on cache metadata files. Node-local /tmp avoids
# this entirely -- each node compiles independently (takes ~30s extra on first iter).
# Node-local (NOT shared-NFS — stale-handle crashes) Triton kernel cache. Default is
# per-job; TRAIN_PERSISTENT_TRITON_CACHE=1 switches to a stable node-local path so
# reruns on previously-used nodes skip the first-microbatch JIT chain — at deep PP
# that chain serializes along the pipeline (stage k compiles only when microbatch 1
# reaches it) and dominates startup: ~75s/stage x 22 stages ~= 25 min at PP=22.
if [ "${TRAIN_PERSISTENT_TRITON_CACHE:-0}" = "1" ]; then
    export TRITON_CACHE_DIR=/tmp/triton_cache_persistent_${ENV_CACHE_SUFFIX}
else
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${ENV_CACHE_SUFFIX}
fi
export TRITON_HOME=/tmp/triton_home_${SLURM_JOB_ID}_${ENV_CACHE_SUFFIX}

# Megatron config file locking directory. When loading HF configs, megatron-bridge uses
# fcntl.flock() to serialize access. On NFS, 128+ ranks all locking the same file causes
# lock contention and stale file handles. Node-local locks mean only 4 ranks/node contend.
export MEGATRON_CONFIG_LOCK_DIR=/tmp/megatron_config_locks_${SLURM_JOB_ID}_${ENV_CACHE_SUFFIX}

# fp32 inter-chunk SSM state in checkpointed (memory-neutral) mode — DEFAULT ON.
# bf16 inter-chunk SSM state overflows deterministically on specific long single-document
# sequences (~32K), NaN'ing the run; the checkpointed fp32 path costs ~0-5% and no memory
# (validated by the b2_nofp32 ablation). Harmless at short seq. Set 0 to disable, 1 for
# the direct (non-checkpointed) variant. Consumed by pipeline_training_run.py.
export ISAMBARD_FP32_SSM_STATE="${ISAMBARD_FP32_SSM_STATE:-checkpoint}"

# Eager NCCL communicator warmup — DEFAULT OFF (was ON; flipped 2026-06-13 after it was
# root-caused as a CATASTROPHIC steady-state regression, NOT a pure startup win).
# Measured at Super-120B PP22·CP4·seq32K (byte-identical config, single-group, fp32-SSM off):
#   comm-warmup ON  -> ~277-290 s/iter (6.5 TFLOP/s/GPU)
#   comm-warmup OFF -> ~30 s/iter      (63 TFLOP/s/GPU)   ~9.6x faster
# The eager warmup batches a tiny (4-byte) send/recv with both PP neighbors to pre-init the
# per-pair P2P transports; at deep PP this establishes the PP p2p channels in a configuration
# that cripples the steady-state ~168 MB activation exchanges (the slowdown shows as inflated
# forward/backward compute AND send/recv timers — pipeline-stall propagation). Harmless at
# shallow PP (mqv2 PP8/seq8K is fine either way), so it slipped through. Lazy first-use init
# costs only the first iteration (already compile-dominated at ~2200 s). Set 1 to re-enable.
export ISAMBARD_COMM_WARMUP="${ISAMBARD_COMM_WARMUP:-0}"

# ==============================================================================
# Run identity (pipeline/sh/run_identity.sh)
# ==============================================================================
set_run_id
set_raw_log_path
link_run_id_log
configure_nccl_trace_location

# ==============================================================================
# Distributed setup
# ==============================================================================
NNODES="${OVERRIDE_NODES:-$SLURM_NNODES}"
NODELIST="${OVERRIDE_NODELIST:-$SLURM_NODELIST}"
set_rendezvous "$NODELIST"
export MASTER_ADDR MASTER_PORT
TOTAL_GPUS=$((NNODES * 4))

# Dragonfly placement of this run's nodes (derive_switch_spread, pipeline/sh/slurm.sh), exported for the W&B run
# summary (scripts/telemetry/run_identity.py reads ISAMBARD_SWITCH_SPREAD). Computed here rather than in the payload
# because the payload runs inside the container, which inherits the SLURM environment variables but not the SLURM
# binaries -- scontrol is absent there. Non-fatal, like the rest of the identity bookkeeping.
if [ -z "${ISAMBARD_SWITCH_SPREAD:-}" ]; then
    ISAMBARD_SWITCH_SPREAD="$(derive_switch_spread "$NODELIST" || true)"
fi
export ISAMBARD_SWITCH_SPREAD
if [ -n "$ISAMBARD_SWITCH_SPREAD" ]; then
    echo "[run-identity] switch placement: $ISAMBARD_SWITCH_SPREAD"
else
    echo "WARNING: switch placement not determined (non-fatal); scontrol errors above" >&2
fi

# ft_launcher's rank-heartbeat timeout in seconds; the ft_launcher launch below explains the
# value and what it costs. This export is the single source of it: both ft_launcher heartbeat
# flags and pipeline_training_run.py's headroom warning read it, so the warning always quotes
# the wall the job actually hits.
if [ "$USE_FT" = true ]; then
    export ISAMBARD_FT_HEARTBEAT_TIMEOUT=${ISAMBARD_FT_HEARTBEAT_TIMEOUT:-7200}
fi

# Per-launch environment overrides, second application: after the launcher's last export, so
# that the overrides also beat its unconditional exports (pipeline/sh/env_overrides.sh).
ENV_OVERRIDES_PAYLOAD=""
if [ -n "${ISAMBARD_ENV_OVERRIDES+x}" ]; then
    export_env_overrides
fi

# ==============================================================================
# Select training script
# ==============================================================================
TRAIN_SCRIPT="pipeline_training_run.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

# Build training script args, one array element per argument; the payload pastes them in quoted (quote_args), so a
# value with a space or a shell character reaches the ranks as one word.
SCRIPT_ARGS=(--config-file "$CONFIG_FILE" --model "$MODEL" --mode "$MODE")
if [ -n "$PEFT" ]; then
    SCRIPT_ARGS+=(--peft "$PEFT")
fi
if [ "$ENABLE_PAO" = true ]; then
    SCRIPT_ARGS+=(--enable-pao)
fi
if [ "$USE_FT" = false ]; then
    SCRIPT_ARGS+=(--disable-ft)
fi
if [ "$USE_STRAGGLER" = false ]; then
    SCRIPT_ARGS+=(--disable-straggler)
fi
SCRIPT_ARGS+=("${EXTRA_ARGS[@]}")

# ==============================================================================
# Print summary
# ==============================================================================
echo "===== Nemotron 3 Training ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Run ID:    $ISAMBARD_RUN_ID"
if [ -n "$ISAMBARD_RAW_LOG_PATH" ]; then
    echo "Raw log:   $ISAMBARD_RAW_LOG_PATH"
fi
echo "Config:    $CONFIG_FILE"
echo "Model:     $MODEL"
echo "Mode:      $MODE"
echo "Script:    $TRAIN_SCRIPT"
echo "Nodes:     $NNODES"
echo "GPUs/node: 4"
echo "Total GPUs: $TOTAL_GPUS"
echo "Master:    $MASTER_ADDR:$MASTER_PORT"
echo "Env:       container ($CONTAINER_SIF)"
# Image + Slingshot-build provenance into the job log so any run's exact
# container stack is recoverable from its output alone (reproducibility).
# `|| true`: a missing provenance file is cosmetic, and under `set -euo pipefail`
# sed's exit status would otherwise abort the launch.
sed 's/^/  sif: /' "${CONTAINER_SIF}.source.txt" 2>/dev/null | head -4 || true
sed 's/^/  slingshot: /' "$CONTAINER_SLINGSHOT_DIR/provenance.txt" 2>/dev/null || true
# The Slingshot provenance records the libfabric the stack was BUILT against; this is the one
# the run binds and loads (pipeline_env_config.env), which can differ from it.
echo "  host libfabric (bound at runtime): $CONTAINER_HOST_LIBFABRIC"
if [ "$USE_FT" = true ]; then
    echo "Launcher:  ft_launcher (fault-tolerant)"
    if [ "$USE_STRAGGLER" = false ]; then
        echo "Straggler: NVRx straggler detection disabled"
    fi
else
    echo "Launcher:  torchrun"
fi
if [ -n "$PEFT" ]; then echo "PEFT:      $PEFT"; fi
if [ "$ENABLE_PAO" = true ]; then echo "PAO:       enabled"; fi
if [ -n "${ISAMBARD_ENV_OVERRIDES+x}" ]; then
    echo "Env overrides: $ISAMBARD_ENV_OVERRIDES (host, and container after activation)"
    if [ ${#ENV_OVERRIDE_ENTRIES[@]} -gt 0 ]; then
        printf '  %s\n' "${ENV_OVERRIDE_ENTRIES[@]}"
    fi
fi
echo "================================"

# ==============================================================================
# Launch
# ==============================================================================
SRUN_ARGS="--nodes=$NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 --export=ALL --overlap"
if [ -n "$OVERRIDE_NODELIST" ]; then
    SRUN_ARGS="$SRUN_ARGS --nodelist=$OVERRIDE_NODELIST"
fi

# Runner indirection: srun launches ONE apptainer instance per node (via the
# shim), inside which ft_launcher/torchrun spawns that node's 4 ranks. Kept as a
# variable so both launch paths below assemble their payload identically.
# All exported env (NCCL/FI_CXI/TORCH/MASTER/ISAMBARD_* etc.) reaches the ranks:
# srun --export=ALL carries it to the node, and apptainer inherits the host
# environment by default. ENV_OVERRIDES_PAYLOAD (empty unless ISAMBARD_ENV_OVERRIDES
# is set) sits directly before the rank launcher in both payloads, so the overrides
# land after activation and after the payload's own exports. The rendezvous address and
# port and the heartbeat timeout enter the payload through printf %q: an override can set
# them, and the payload is shell text, so each must arrive as one literal word.
RUNNER=("$REPO_DIR/pipeline_env_exec.sh")
ACTIVATE_CMD="source pipeline_env_activate.sh || exit 1"

# ft_launcher compatibility gate: the image's nvidia-resiliency-ext version is
# whatever NGC shipped, so verify the section-timeout flags exist before launching
# 44+ ranks into a launcher that would reject them; fail with actionable guidance
# rather than a rank-0 usage error. (D6 in docs/environment.md.) The check is
# pipeline/environment/ft_launcher.py's, run in the container.
if [ "$USE_FT" = true ]; then
    # The check names what is missing on stderr, which passes through.
    if ! "$REPO_DIR/pipeline_env_exec.sh" "$(container_payload "python -m pipeline.environment.ft_launcher")" >/dev/null; then
        echo "FATAL: the container image's ft_launcher (nvidia-resiliency-ext) does not support the" >&2
        echo "  section-timeout flags this launcher passes (see above). Rerun with --disable-ft, or qualify a" >&2
        echo "  newer image (GEODESIC_CONTAINER_IMAGE_TAG in pipeline_env_config.env)." >&2
        exit 1
    fi
fi

if [ "$USE_FT" = true ]; then
    # Fault-tolerant launch via ft_launcher (nvidia-resiliency-ext)
    #
    # Heartbeat timeouts are passed as explicit large FLOATS, not omitted and not
    # 'none'. Three facts force this:
    #   1. The image's ft_launcher parses these as floats, so the literal 'none'
    #      we previously used to disable them is rejected outright (caught
    #      2026-07-24 on the first container+FT run).
    #   2. Omitting them is NOT equivalent to disabling them: nvidia-resiliency-ext
    #      defaults to 3600 s initial / 2700 s subsequent heartbeat timeouts, and
    #      heartbeats are an INDEPENDENT mechanism from --ft-rank-section-timeouts.
    #   3. Ultra-550B's documented first iteration at PP=36 takes 45-75 min of lazy
    #      NCCL comm-init. Under the 3600 s default that trips the heartbeat, ft
    #      SIGKILLs the workers, and the restart lands in the same slow first
    #      iteration — a restart loop that looks like a fabric hang.
    # 7200 s covers the worst measured first iteration with margin. It does NOT
    # merely bound a genuinely wedged rank: the monitor is not guaranteed to receive
    # an initial heartbeat, and when it does not, this timeout SIGKILLs a healthy job
    # at exactly 7200 s -- `[Cycle N] Did not get initial heartbeat. Waited 7200.00
    # seconds`. Any run whose first checkpoint arrives later than that then restarts
    # from iteration 0 with nothing on disk, indefinitely. pipeline_training_run.py
    # prints the required s/iter at startup whenever ft is on; --disable-ft is the
    # opt-out, and --disable-straggler is NOT. The value is ISAMBARD_FT_HEARTBEAT_TIMEOUT,
    # exported with the distributed setup above.
    srun $SRUN_ARGS "${RUNNER[@]}" "
        cd $REPO_DIR
        $ACTIVATE_CMD
        $ENV_OVERRIDES_PAYLOAD
        ft_launcher \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$(printf '%q' "$MASTER_ADDR:$MASTER_PORT") \
            --nproc_per_node=\$SLURM_GPUS_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_NODEID \
            --max-restarts=20 \
            --ft-initial-rank-heartbeat-timeout=$(printf '%q' "$ISAMBARD_FT_HEARTBEAT_TIMEOUT") \
            --ft-rank-heartbeat-timeout=$(printf '%q' "$ISAMBARD_FT_HEARTBEAT_TIMEOUT") \
            --ft-rank-section-timeouts=setup:10800,step:7200,checkpointing:3600 \
            --ft-rank-out-of-section-timeout=7200 \
            --ft-log-level=INFO \
            $TRAIN_SCRIPT$(quote_args "${SCRIPT_ARGS[@]}")
    "
else
    # Plain torchrun (no fault tolerance)
    srun $SRUN_ARGS "${RUNNER[@]}" "
        cd $REPO_DIR
        $ACTIVATE_CMD
        export TMPDIR=/tmp/megatron_tmp_\${SLURM_JOB_ID}_$ENV_CACHE_SUFFIX
        mkdir -p \$TMPDIR
        $ENV_OVERRIDES_PAYLOAD
        python -m torch.distributed.run \
            --nproc_per_node=\$SLURM_GPUS_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_NODEID \
            --master_addr=$(printf '%q' "$MASTER_ADDR") \
            --master_port=$(printf '%q' "$MASTER_PORT") \
            $TRAIN_SCRIPT$(quote_args "${SCRIPT_ARGS[@]}")
    "
fi

echo "===== Job Complete ====="
