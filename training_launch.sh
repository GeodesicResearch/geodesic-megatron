#!/bin/bash
# ==============================================================================
# Shared training launcher for Nemotron 3 Nano/Super
#
# Sets all env vars (NCCL, CXI, Slingshot, Triton, W&B, etc.) and launches
# training via srun + ft_launcher or torchrun.
#
# Can be called from:
#   1. An sbatch script (SLURM_JOB_ID set automatically)
#   2. An interactive salloc session (SLURM_JOB_ID set by salloc)
#
# Usage:
#   bash training_launch.sh <config.yaml> --model nano|super --mode sft|cpt [options]
#
# Required:
#   --model nano|super          Model variant
#   --mode sft|cpt              Training mode
#
# Options:
#   --disable-ft                Use plain torchrun instead of ft_launcher
#   --enable-pao                Enable PAO (Partial Activation Offloading)
#   --peft lora                 Enable LoRA PEFT
#   --max-samples N             Limit dataset size (CPT mode only; 0 = all)
#   --nodes N                   Override number of nodes (default: all in allocation)
#   --nodelist LIST             Override nodelist (default: all in allocation)
#   -- [extra args]             Extra args passed to the training script
#
# Examples:
#   # Nano SFT with ft_launcher (default)
#   bash training_launch.sh configs/nemotron_nano_dolci_instruct_sft.yaml --model nano --mode sft
#
#   # Super SFT without fault tolerance
#   bash training_launch.sh configs/nemotron_super_200k_warm_start_sft_bf16.yaml --model super --mode sft --disable-ft
#
#   # Nano CPT / midtraining
#   bash training_launch.sh configs/inoculation_midtraining/cpt/im_nemotron_30b_baseline_cpt.yaml --model nano --mode cpt --max-samples 50000
#
#   # Use a subset of nodes in an salloc
#   bash training_launch.sh configs/<config>.yaml --model nano --mode sft --nodes 8 --nodelist node[001-008]
# ==============================================================================

set -euo pipefail

# --- Parse arguments ---
CONFIG_FILE=""
MODEL=""
MODE=""
USE_FT=true
ENABLE_PAO=false
PEFT=""
MAX_SAMPLES=""
OVERRIDE_NODES=""
OVERRIDE_NODELIST=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --mode)        MODE="$2"; shift 2 ;;
        --disable-ft)  USE_FT=false; shift ;;
        --enable-pao)  ENABLE_PAO=true; shift ;;
        --peft)        PEFT="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
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

USAGE="Usage: bash training_launch.sh <config.yaml> --model nano|super --mode sft|cpt [options]"

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
    echo "ERROR: --model is required (nano or super)." >&2
    echo "$USAGE" >&2
    exit 1
fi

if [[ "$MODEL" != "nano" && "$MODEL" != "super" ]]; then
    echo "ERROR: --model must be 'nano' or 'super', got: $MODEL" >&2
    exit 1
fi

if [ -z "$MODE" ]; then
    echo "ERROR: --mode is required (sft or cpt)." >&2
    echo "$USAGE" >&2
    exit 1
fi

if [[ "$MODE" != "sft" && "$MODE" != "cpt" ]]; then
    echo "ERROR: --mode must be 'sft' or 'cpt', got: $MODE" >&2
    exit 1
fi

# --- Verify we're inside a SLURM allocation ---
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: No SLURM allocation detected (SLURM_JOB_ID not set)." >&2
    echo "  Either submit via sbatch or get an allocation with salloc first." >&2
    exit 1
fi

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
cd "$REPO_DIR"

# --- Module loading ---
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

# --- Activate environment ---
source "$REPO_DIR/env_activate.sh"

# ==============================================================================
# NCCL transport and network plugin
#
# Isambard uses HPE Slingshot-11 (CXI fabric) for inter-node communication.
# NCCL talks to Slingshot via the AWS OFI NCCL plugin (aws-ofi-nccl), which
# bridges NCCL's network API to libfabric's CXI provider.
#
# Docs:
#   NCCL env vars:    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
#   fi_cxi man page:  https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html
#   HPE libfabric:    https://support.hpe.com/hpesc/public/docDisplay?docId=dp00005991en_us
# ==============================================================================

# --- NCCL network plugin selection ---

# Use the AWS OFI NCCL plugin for libfabric-based transport (required for Slingshot).
# Without this, NCCL falls back to TCP sockets which are orders of magnitude slower.
export NCCL_NET="AWS Libfabric"

# Select the CXI provider within libfabric. CXI is HPE's native provider for the
# Cassini NIC (Slingshot-11). Other providers (verbs, tcp, etc.) won't work here.
export FI_PROVIDER=cxi

# Bind NCCL's socket-based control plane to the Slingshot HSN interface (hsn0/hsn1).
# Without this, NCCL may try to use the management ethernet which can't reach other nodes.
export NCCL_SOCKET_IFNAME=hsn

# --- NCCL collective algorithm and protocol tuning ---

# Disable CollNet (collective offload to smart switches). Slingshot doesn't have
# in-network reduction hardware, so CollNet has no effect and can cause init delays.
export NCCL_COLLNET_ENABLE=0

# Allow NCCL rings/trees to use different NICs on different nodes. With value=1, NCCL
# can route traffic across both HSN NICs even if peer nodes aren't on the same rail,
# which improves bandwidth utilization on Slingshot's dragonfly topology.
export NCCL_CROSS_NIC=1

# Enable GPU Direct RDMA (GDR) when GPU and NIC share the same NUMA node (PHB = same
# PCIe Host Bridge). GDR lets the NIC DMA directly to/from GPU memory, bypassing CPU
# staging buffers. PHB is the right level for GH200 where GPU and NIC are on the same
# PCIe root complex. Higher levels (SYS) would enable GDR across NUMA, but GH200 is
# single-socket so PHB covers all local GPU-NIC paths.
export NCCL_NET_GDR_LEVEL=PHB

# Disable the LL128 (Low Latency 128-byte) protocol. LL128 uses GPU shared memory for
# sub-microsecond latency on small messages, but performs worse than Simple protocol on
# Slingshot for the large collective payloads typical in LLM training. The ^prefix means
# "exclude this protocol", so NCCL will use LL (small msgs) and Simple (large msgs).
export NCCL_PROTO=^LL128

# Force Ring algorithm for all collectives. Ring provides best peak bandwidth utilization
# by arranging GPUs in a logical circle where each sends/receives to/from one neighbor.
# Tree algorithm has lower latency for small messages but worse bandwidth. On Slingshot,
# Ring is 16% faster for DP reduce-scatter (the bottleneck collective in distributed
# optimizer training). Comma-separated list; other options: Tree, CollnetDirect, CollnetChain.
export NCCL_ALGO=Ring

# Minimum number of NCCL channels (parallel data paths). Each channel maps to a CUDA
# CTA (thread block) that drives data movement. Default is 1 for small-scale jobs.
# Setting to 4 ensures enough parallelism to saturate Slingshot bandwidth even for
# smaller collectives. More channels = more CUDA SMs used by NCCL, but 4 is modest.
export NCCL_MIN_NCHANNELS=4

# Number of network channels per remote GPU peer. Default is 2. Each channel gets its
# own QP (queue pair) / CXI connection, providing path diversity. Setting to 4 fixes a
# 24% throughput regression on GH200 documented in Isambard release notes. The extra
# channels let NCCL interleave more in-flight transfers across the NIC's tx/rx queues.
export NCCL_NCHANNELS_PER_NET_PEER=4

# Disable NVLink SHARP (NVSwitch-based collective offload). NVLS is available on
# DGX H100/B200 systems with 3rd-gen NVSwitch, which Isambard GH200 nodes don't have.
# Enabling it would cause NCCL to fail looking for NVSwitch hardware that doesn't exist.
# Must also be 0 for nvidia-resiliency-ext in-process restart compatibility.
export NCCL_NVLS_ENABLE=0

# --- GPU Direct RDMA (GDRCopy) ---

# Enable GDRCopy in NCCL: allows low-latency GPU memory access via a CPU-mapped path
# (kernel module gdrdrv). Used for small control messages and signaling in collectives.
# Complements GPU Direct RDMA (NET_GDR_LEVEL) which handles bulk data. GDRCopy is
# particularly helpful for reducing latency of NCCL's internal synchronization.
export NCCL_GDRCOPY_ENABLE=1

# Enable GDRCopy in the libfabric CXI provider. This is the libfabric-side counterpart
# of NCCL_GDRCOPY_ENABLE -- it tells the CXI provider to use GDRCopy for GPU memory
# registrations and small transfers, enabling true GPU RDMA without CPU bounce buffers.
export FI_HMEM_CUDA_USE_GDRCOPY=1

# --- CXI provider: completion queue and transmit ---

# Maximum entries in the CXI completion queue. Default is 1024, which is too small for
# NCCL's many parallel channels. When the CQ overflows, the NIC drops completions and
# NCCL hangs waiting for acks that never arrive ("Cassini Event Queue overflow detected").
# 131072 entries provides headroom for 128-GPU jobs with 4 channels per peer.
export FI_CXI_DEFAULT_CQ_SIZE=131072

# Maximum outstanding transmit requests per endpoint. Controls how many rendezvous sends
# can be in-flight simultaneously. Default was 16384 in our original configs, but that
# exhausted CXI hardware resources (limited tx credits on the Cassini NIC), causing sends
# to stall. 1024 is sufficient for NCCL's channel count and avoids resource exhaustion.
export FI_CXI_DEFAULT_TX_SIZE=1024

# --- CXI provider: memory registration ---

# Use the Linux userfaultfd mechanism to monitor memory region invalidation. When CUDA
# or the OS remaps virtual memory, libfabric's MR cache must invalidate stale registrations.
# userfaultfd is a kernel-level notification (vs. memhooks which intercept malloc/free at
# the linker level). More reliable with CUDA's VMM allocator and avoids false invalidations
# from glibc memory operations that memhooks would trigger.
export FI_MR_CACHE_MONITOR=userfaultfd

# Disable CXI host memory registration for data buffers. When enabled (default), the CXI
# provider registers host memory with the NIC for RDMA. Disabling it avoids contention on
# the NIC's memory registration table when many ranks on the same node register overlapping
# host memory regions, which can cause registration failures or slowdowns.
export FI_CXI_DISABLE_HOST_REGISTER=1

# --- CXI provider: message matching and rendezvous ---

# Use software-based message matching instead of NIC hardware matching. The Cassini NIC
# has limited hardware match entries; when exhausted, it falls back to software anyway but
# with a costly mode-switch. "soft" mode uses software from the start, which is more
# predictable and avoids the stall that happens when hardware entries run out mid-collective.
export FI_CXI_RX_MATCH_MODE=soft

# Force all messages to use the rendezvous protocol (no eager sends). These three vars
# together set the eager-to-rendezvous crossover point to 0 bytes:
#   RDZV_GET_MIN=0:    minimum Get payload in rendezvous is 0 (no floor)
#   RDZV_THRESHOLD=0:  no bytes sent eagerly before switching to rendezvous
#   RDZV_EAGER_SIZE=0: no eager data piggybacked on the rendezvous request
# Eager protocol pre-posts receive buffers and copies data on arrival; rendezvous does a
# zero-copy RDMA Get from the sender's buffer. For NCCL's large payloads, rendezvous is
# always faster and avoids exhausting the CXI eager buffer pool (overflow_buf).
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0

# Use the alternative rendezvous read protocol. The default rendezvous uses a Put-based
# protocol where the sender pushes data to the receiver. "alt_read" uses a Get-based
# protocol where the receiver pulls data from the sender. The alt_read path is less prone
# to CXI deadlocks when many endpoints rendezvous simultaneously (common in NCCL all-to-all)
# and improves collective performance on Slingshot.
export FI_CXI_RDZV_PROTO=alt_read

# Disable Immediate Data Completion (IDC) for non-inject messages. IDC is a Cassini
# optimization that inlines small message completions into the CQ entry. For NCCL's
# message patterns (many parallel channels, large payloads), IDC adds overhead to the
# completion path without benefit. Disabling it reduces CQ pressure.
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1

# --- PyTorch NCCL process group settings ---

# Enable async error handling in NCCL process group. When a collective hangs (e.g., due
# to a Slingshot network issue), a background watchdog thread detects the timeout and
# aborts the process instead of blocking forever. Required for fault tolerance -- without
# this, a single hung rank blocks the entire job indefinitely. The NCCL_* form is the
# legacy name; TORCH_NCCL_* is the current PyTorch form. Both are set for compatibility.
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Make NCCL wait() calls blocking (synchronous). When combined with async error handling,
# this means: the calling thread blocks on the collective, but the watchdog thread can
# still detect timeouts and abort. Without this, wait() returns immediately and errors
# may be lost or detected late.
export TORCH_NCCL_BLOCKING_WAIT=1

# Avoid using record_stream() on NCCL output tensors. record_stream() extends a tensor's
# lifetime until the NCCL stream finishes, which prevents PyTorch's caching allocator from
# reusing that memory. This causes GPU memory fragmentation and over-allocation, especially
# in large-scale distributed training. Disabling it lets the allocator reclaim memory
# earlier via event-based synchronization instead.
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# Disable MPI GPU-aware transport. Isambard's MPICH has GPU support compiled in, but we
# use NCCL (not MPI) for GPU collectives. If both NCCL and MPI try to manage GPU contexts
# simultaneously, they can deadlock on CUDA context initialization. Setting this to 0
# ensures MPI only handles CPU data (hostfile, process launch) and stays out of NCCL's way.
export MPICH_GPU_SUPPORT_ENABLED=0

# ==============================================================================
# NCCL timeout and fault tolerance
#
# These settings work with ft_launcher (nvidia-resiliency-ext) to detect and
# recover from Slingshot NCCL hangs. The layered approach:
#   1. In-process restart (60s/90s) -- reinit NCCL, retry same step
#   2. ft_launcher restart (step timeout 3600s) -- restart from checkpoint
#   3. NCCL watchdog (900s) -- last-resort process abort
#
# TORCH_NCCL_TIMEOUT must exceed the in-process restart hard_timeout (90s)
# so the fault tolerance system gets a chance to recover before the watchdog
# kills the process.
# ==============================================================================

# NCCL watchdog timeout in seconds. If any collective takes longer than this, the
# watchdog thread aborts the process. Set to 900s (15 min) to give ft_launcher's
# in-process restart (90s hard timeout) and section timeouts (600s step) room to
# handle hangs first. The watchdog is the last resort.
export TORCH_NCCL_TIMEOUT=900

# Suppress verbose C++ logging from PyTorch internals. "error" level hides INFO/WARN
# messages from torch::distributed and autograd that would otherwise flood logs on
# 128+ ranks. NCCL's own logging is controlled separately by NCCL_DEBUG.
export TORCH_CPP_LOG_LEVEL=error

# Don't rethrow CUDA errors from the NCCL watchdog. Default is true (rethrow), which
# kills the process on any CUDA error detected during a collective. Setting to 0 allows
# nvidia-resiliency-ext's in-process restart to catch the error and reinitialize NCCL
# instead of crashing. Required for in-process restart to work.
export TORCH_NCCL_RETHROW_CUDA_ERRORS=0

# NCCL debug logging level. WARN prints connection setup failures, transport selection,
# and error conditions without the massive per-message output of INFO. INFO can cause OOM
# on 128-GPU jobs by writing GBs of log data. Set to INFO temporarily for debugging
# network issues, but never leave it on for production runs.
export NCCL_DEBUG=WARN

# Restrict NCCL debug output to INIT (connection setup, topology detection, transport
# selection) and NET (network plugin operations, libfabric calls). This filters out
# noisy subsystems like COLL (per-collective traces) and GRAPH (channel/ring construction)
# that are only useful for deep debugging.
export NCCL_DEBUG_SUBSYS=INIT,NET

# ==============================================================================
# Job-specific paths and caches
#
# These use $SLURM_JOB_ID for isolation between concurrent jobs, and /tmp for
# node-local storage to avoid NFS contention across 128+ ranks.
# Universal paths (HF_HOME, WANDB_DIR) and GPU settings (UB_SKIPMC,
# CUDA_DEVICE_MAX_CONNECTIONS, etc.) are set in env_activate.sh.
# ==============================================================================

# Node-local temp directory. Avoids NFS contention from CUDA JIT compilation (nvcc/ptxas),
# Triton kernel compilation, and other temporary files that many ranks write simultaneously.
# NFS handles this poorly -- stale file handles and lock contention across 128 ranks.
# Overrides the shared TMPDIR from env_activate.sh with a job-specific node-local path.
export TMPDIR=/tmp/megatron_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

# Ensure W&B dir exists (path set in env_activate.sh)
mkdir -p "$WANDB_DIR"

# Triton JIT kernel cache. Triton compiles kernels on first use and caches the PTX/cubin.
# On NFS, 128 ranks compiling the same kernel simultaneously causes stale file handle
# errors (ERRNO 116) and race conditions on cache metadata files. Node-local /tmp avoids
# this entirely -- each node compiles independently (takes ~30s extra on first iter).
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TRITON_HOME=/tmp/triton_home_${SLURM_JOB_ID}

# Megatron config file locking directory. When loading HF configs, megatron-bridge uses
# fcntl.flock() to serialize access. On NFS, 128+ ranks all locking the same file causes
# lock contention and stale file handles. Node-local locks mean only 4 ranks/node contend.
export MEGATRON_CONFIG_LOCK_DIR=/tmp/megatron_config_locks_${SLURM_JOB_ID}

# ==============================================================================
# Distributed setup
# ==============================================================================
NNODES="${OVERRIDE_NODES:-$SLURM_NNODES}"
NODELIST="${OVERRIDE_NODELIST:-$SLURM_NODELIST}"
export MASTER_ADDR=$(scontrol show hostname "$NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
TOTAL_GPUS=$((NNODES * 4))

# ==============================================================================
# Select training script
# ==============================================================================
if [ "$MODE" = "cpt" ]; then
    if [ "$MODEL" = "super" ]; then
        TRAIN_SCRIPT="examples/models/nemotron_3/super/midtrain_nemotron_3_super.py"
    else
        TRAIN_SCRIPT="examples/models/nemotron_3/nano/midtrain_nemotron_3_nano.py"
    fi
else
    if [ "$MODEL" = "super" ]; then
        TRAIN_SCRIPT="examples/models/nemotron_3/super/finetune_nemotron_3_super.py"
    else
        TRAIN_SCRIPT="examples/models/nemotron_3/nano/finetune_nemotron_3_nano.py"
    fi
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

# Build training script args
SCRIPT_ARGS="--config-file $CONFIG_FILE"
if [ -n "$MAX_SAMPLES" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --max-samples $MAX_SAMPLES"
fi
if [ -n "$PEFT" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --peft $PEFT"
fi
if [ "$ENABLE_PAO" = true ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --enable-pao"
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS ${EXTRA_ARGS[*]}"
fi

# ==============================================================================
# Print summary
# ==============================================================================
echo "===== Nemotron 3 Training ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Config:    $CONFIG_FILE"
echo "Model:     $MODEL"
echo "Mode:      $MODE"
echo "Script:    $TRAIN_SCRIPT"
echo "Nodes:     $NNODES"
echo "GPUs/node: 4"
echo "Total GPUs: $TOTAL_GPUS"
echo "Master:    $MASTER_ADDR:$MASTER_PORT"
if [ "$USE_FT" = true ]; then
    echo "Launcher:  ft_launcher (fault-tolerant)"
else
    echo "Launcher:  torchrun"
fi
if [ -n "$PEFT" ]; then echo "PEFT:      $PEFT"; fi
if [ "$ENABLE_PAO" = true ]; then echo "PAO:       enabled"; fi
if [ -n "$MAX_SAMPLES" ]; then echo "Max samples: $MAX_SAMPLES"; fi
echo "================================"

# ==============================================================================
# Launch
# ==============================================================================
SRUN_ARGS="--nodes=$NNODES --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL"
if [ -n "$OVERRIDE_NODELIST" ]; then
    SRUN_ARGS="$SRUN_ARGS --nodelist=$OVERRIDE_NODELIST"
fi

if [ "$USE_FT" = true ]; then
    # Fault-tolerant launch via ft_launcher (nvidia-resiliency-ext)
    srun $SRUN_ARGS bash -c "
        cd $REPO_DIR
        source env_activate.sh
        ft_launcher \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            --nproc_per_node=\$SLURM_GPUS_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_NODEID \
            --max-restarts=20 \
            --ft-initial-rank-heartbeat-timeout=none \
            --ft-rank-heartbeat-timeout=none \
            --ft-rank-section-timeouts=setup:1800,step:3600,checkpointing:600 \
            --ft-rank-out-of-section-timeout=3600 \
            --ft-log-level=INFO \
            $TRAIN_SCRIPT \
            $SCRIPT_ARGS
    "
else
    # Plain torchrun (no fault tolerance)
    srun $SRUN_ARGS bash -c "
        cd $REPO_DIR
        source env_activate.sh
        export TMPDIR=/tmp/megatron_tmp_\${SLURM_JOB_ID}
        mkdir -p \$TMPDIR
        python -m torch.distributed.run \
            --nproc_per_node=\$SLURM_GPUS_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $TRAIN_SCRIPT \
            $SCRIPT_ARGS
    "
fi

echo "===== Job Complete ====="
