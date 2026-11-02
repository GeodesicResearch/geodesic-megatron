# shellcheck shell=bash
# ==============================================================================
# Slingshot/CXI fabric settings for NCCL, exported into the launching shell so
# that srun --export=ALL and apptainer's environment passthrough carry them into
# every rank. Source it after common.sh.
#
# Isambard uses HPE Slingshot-11 (CXI fabric) for inter-node communication.
# NCCL talks to Slingshot via the AWS OFI NCCL plugin (aws-ofi-nccl), which
# bridges NCCL's network API to libfabric's CXI provider.
#
# Docs:
#   NCCL env vars:    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
#   fi_cxi man page:  https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html
#   HPE libfabric:    https://support.hpe.com/hpesc/public/docDisplay?docId=dp00005991en_us
#
# Catalogue:
#   fabric_select  NCCL's network: the aws-ofi-nccl plugin over CXI on hsn
#   fabric_base    fabric_select, plus the CXI and NCCL settings every
#                  multi-node payload of the pipelines runs with
#   fabric_shared  the settings fabric_base and fabric_training share
#   fabric_training  every NCCL, CXI and torch-distributed setting of a
#                  training launch
# ==============================================================================

fabric_select() {
    # Use the AWS OFI NCCL plugin for libfabric-based transport (required for Slingshot).
    # Without this, NCCL falls back to TCP sockets which are orders of magnitude slower.
    export NCCL_NET="AWS Libfabric"

    # Select the CXI provider within libfabric. CXI is HPE's native provider for the
    # Cassini NIC (Slingshot-11). Other providers (verbs, tcp, etc.) won't work here.
    export FI_PROVIDER=cxi

    # Bind NCCL's socket-based control plane to the Slingshot HSN interface (hsn0/hsn1).
    # Without this, NCCL may try to use the management ethernet which can't reach other nodes.
    export NCCL_SOCKET_IFNAME=hsn
}

# The settings fabric_base and fabric_training share.
fabric_shared() {
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

    # Disable NVLink SHARP (NVSwitch-based collective offload). NVLS is available on
    # DGX H100/B200 systems with 3rd-gen NVSwitch, which Isambard GH200 nodes don't have.
    # Enabling it would cause NCCL to fail looking for NVSwitch hardware that doesn't exist.
    # Must also be 0 for nvidia-resiliency-ext in-process restart compatibility.
    export NCCL_NVLS_ENABLE=0

    # Maximum entries in the CXI completion queue. Default is 1024, which is too small for
    # NCCL's many parallel channels. When the CQ overflows, the NIC drops completions and
    # NCCL hangs waiting for acks that never arrive ("Cassini Event Queue overflow detected").
    # 131072 entries provides headroom for 128-GPU jobs with 4 channels per peer.
    export FI_CXI_DEFAULT_CQ_SIZE=131072

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
}

fabric_base() {
    fabric_select
    fabric_shared
    export FI_CXI_RX_MATCH_MODE=soft
    export NCCL_DEBUG=WARN
}

# fabric_training: fabric_select and fabric_shared, then the rest of a training launch's NCCL, CXI and
# torch-distributed settings, several of them overridable from the launching environment.
fabric_training() {
    fabric_select
    fabric_shared

    # --- NCCL collective algorithm and protocol tuning ---

    # Disable CollNet (collective offload to smart switches). Slingshot doesn't have
    # in-network reduction hardware, so CollNet has no effect and can cause init delays.
    export NCCL_COLLNET_ENABLE=0

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

    # Maximum outstanding transmit requests per endpoint. Controls how many rendezvous sends
    # can be in-flight simultaneously. Default was 16384 in our original configs, but that
    # exhausted CXI hardware resources (limited tx credits on the Cassini NIC), causing sends
    # to stall. 1024 is sufficient for NCCL's channel count and avoids resource exhaustion.
    export FI_CXI_DEFAULT_TX_SIZE=1024

    # --- CXI provider: memory registration ---

    # MR-cache capacity. The provider default (~4096 entries) is too small for a 512-rank
    # distributed-checkpoint save: the save's staging churn fills the cache, and once full it
    # stops caching NEW registrations — after which every NCCL chunk transfer pays slow-path
    # registration and ALL collectives run ~25x slower for the remainder of the process
    # (measured 2026-08-14: ~6.5 s/iter -> uniform ~165 s/iter on the 512-GPU Nano pretrain
    # posture, permanent until restart; fresh process fast; DP=128 saves unaffected). With
    # max_count 65536 / unlimited size the same save is harmless (post-save 6.6 s/iter).
    # The A/B pair proving this: probe_r1f (default cache, collapse) vs probe_r1g (this
    # setting, no collapse), same allocation, same config.
    export FI_MR_CACHE_MAX_COUNT=${FI_MR_CACHE_MAX_COUNT:-65536}
    export FI_MR_CACHE_MAX_SIZE=${FI_MR_CACHE_MAX_SIZE:--1}

    # --- CXI provider: message matching and rendezvous ---

    # Use software-based message matching instead of NIC hardware matching. The Cassini NIC
    # has limited hardware match entries; when exhausted, it falls back to software anyway but
    # with a costly mode-switch. "soft" mode uses software from the start, which is more
    # predictable and avoids the stall that happens when hardware entries run out mid-collective.
    export FI_CXI_RX_MATCH_MODE="${TRAIN_FI_CXI_RX_MATCH_MODE_OVERRIDE:-soft}"

    # Force all messages to use the rendezvous protocol (no eager sends). These three vars
    # together set the eager-to-rendezvous crossover point to 0 bytes:
    #   RDZV_GET_MIN=0:    minimum Get payload in rendezvous is 0 (no floor)
    #   RDZV_THRESHOLD=0:  no bytes sent eagerly before switching to rendezvous
    #   RDZV_EAGER_SIZE=0: no eager data piggybacked on the rendezvous request
    # Eager protocol pre-posts receive buffers and copies data on arrival; rendezvous does a
    # zero-copy RDMA Get from the sender's buffer. For NCCL's large payloads, rendezvous is
    # always faster and avoids exhausting the CXI eager buffer pool (overflow_buf).
    export FI_CXI_RDZV_GET_MIN="${TRAIN_FI_CXI_RDZV_OVERRIDE:-0}"
    export FI_CXI_RDZV_THRESHOLD="${TRAIN_FI_CXI_RDZV_THRESHOLD_OVERRIDE:-0}"
    export FI_CXI_RDZV_EAGER_SIZE=0
    # TRAIN_FI_CXI_RDZV_THRESHOLD_OVERRIDE / TRAIN_FI_CXI_RX_MATCH_MODE_OVERRIDE let a
    # diagnostic run relax these hang-avoidance settings to test whether the rendezvous /
    # soft-matching message pattern (not raw fabric bandwidth) is the multi-group slowdown.

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

    # Make NCCL wait() calls blocking (synchronous): the calling thread polls the collective to
    # completion or timeout and throws on timeout. What this costs, in torch's own words at
    # process-group creation ("TORCH_NCCL_BLOCKING_WAIT is enabled, NO watchdog thread is
    # created"): there is no watchdog thread, so no asynchronous timeout detection, no heartbeat
    # monitor, and no flight-recorder dump on timeout or on demand (configure_nccl_trace_location
    # in pipeline/sh/run_identity.sh). The only timeout is the calling thread's wait() at the YAML's
    # dist.distributed_timeout_minutes, which Megatron passes to init_process_group; the
    # TORCH_NCCL_TIMEOUT below covers groups created without one. A run that stops iterating
    # therefore leaves no evidence by itself -- scripts/training/dump_hung_ranks.sh <jobid> takes
    # it from outside. Turning blocking wait off restores the watchdog and everything it owns; for
    # one launch, that is TORCH_NCCL_BLOCKING_WAIT=0 in an ISAMBARD_ENV_OVERRIDES file (pipeline/sh/env_overrides.sh).
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
    #   1. ft_launcher worker restart (step timeout 7200s) -- the per-node agent
    #      restarts failed workers from the latest checkpoint
    #   2. NCCL watchdog (TORCH_NCCL_TIMEOUT=7200s) -- aborts genuinely wedged
    #      collectives
    #   3. srun --kill-on-bad-exit=1 -- a rank that dies unrecoverably ends the
    #      whole step instead of stranding the survivors in a never-completing
    #      collective; a --dependency=singleton chain then resumes from the
    #      latest checkpoint
    # ==============================================================================

    # NCCL watchdog timeout in seconds. If any collective takes longer than this, the
    # watchdog thread aborts the process. The FIRST iteration establishes all collective
    # patterns via lazy NCCL communicator init over Slingshot, which scales with pipeline
    # depth and rank count: ~15-20 min at PP=8, but MEASURED ~52 min for Ultra at PP=36 /
    # 288 ranks (steady-state then drops to ~30 s/iter). 1800s (30 min) was too short for
    # deep pipelines and caused a false "hang" on iter 1 — set to 7200s (2 h), ABOVE the
    # configs' distributed_timeout_minutes=90 (5400 s) so Megatron's explicit process-group
    # timeout (clean error) always fires before this last-resort watchdog. After the
    # first iter the ft_launcher auto-learned step timeout (~tens of s) catches real hangs
    # fast, so this larger watchdog is only a first-iter backstop. Overridable.
    export TORCH_NCCL_TIMEOUT=${TORCH_NCCL_TIMEOUT:-7200}

    # Suppress verbose C++ logging from PyTorch internals. "error" level hides INFO/WARN
    # messages from torch::distributed and autograd that would otherwise flood logs on
    # 128+ ranks. NCCL's own logging is controlled separately by NCCL_DEBUG.
    export TORCH_CPP_LOG_LEVEL=error

    # Don't rethrow CUDA errors from the NCCL watchdog. Default is true (rethrow), which
    # kills the process on any CUDA error detected during a collective. Setting to 0 allows
    # nvidia-resiliency-ext's in-process restart to catch the error and reinitialize NCCL
    # instead of crashing. Required for in-process restart to work.
    export TORCH_NCCL_RETHROW_CUDA_ERRORS=0

    # PyTorch NCCL Flight Recorder. Keeps an in-memory ring buffer of the last N collective
    # ops per rank (op name, sizes, comm, enqueue/start/complete state). Near-zero overhead.
    # On a watchdog timeout (TORCH_NCCL_DUMP_ON_TIMEOUT=1) every rank dumps its buffer to
    # ${TORCH_NCCL_DEBUG_INFO_TEMP_FILE}<rank> (configure_nccl_trace_location puts
    # that beside the raw log) -- it shows which collective each rank was stuck on and which
    # ranks never arrived. The dump is the WATCHDOG's, and TORCH_NCCL_BLOCKING_WAIT=1 above
    # means there is no watchdog: in the shipped posture the buffer is filled and never written.
    # Override the buffer size to 0 to disable. All overridable.
    export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}
    export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
    export TORCH_NCCL_TRACE_CPP_STACK=${TORCH_NCCL_TRACE_CPP_STACK:-0}

    # NCCL debug logging level. WARN prints connection setup failures, transport selection,
    # and error conditions without the massive per-message output of INFO. INFO can cause OOM
    # on 128-GPU jobs by writing GBs of log data -- when set to INFO, also set NCCL_DEBUG_FILE
    # to a per-rank path (supports %h host / %p pid) so it does NOT flood the shared stdout.
    # Overridable: a debug run sets NCCL_DEBUG=INFO + NCCL_DEBUG_FILE for granular capture.
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

    # Restrict NCCL debug output to INIT (connection setup, topology detection, transport
    # selection) and NET (network plugin operations, libfabric calls). This filters out
    # noisy subsystems like COLL (per-collective traces) and GRAPH (channel/ring construction)
    # that are only useful for deep debugging. Overridable for granular debug runs.
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}

    # A submit-time NCCL_DEBUG does not reliably survive to this point: the brics
    # aws-ofi-nccl module setenvs NCCL_DEBUG=VERSION and `module purge` churns
    # module-owned NCCL_* values, so the "overridable" promise above held only by
    # accident. ISAMBARD_NCCL_DEBUG is applied last and therefore actually wins.
    # At INFO on 512 ranks, direct output to per-rank files (ISAMBARD_NCCL_DEBUG_FILE,
    # %h host / %p pid placeholders) or every rank floods the shared stdout.
    if [ -n "${ISAMBARD_NCCL_DEBUG:-}" ]; then
        export NCCL_DEBUG="$ISAMBARD_NCCL_DEBUG"
        if [ -n "${ISAMBARD_NCCL_DEBUG_FILE:-}" ]; then
            export NCCL_DEBUG_FILE="$ISAMBARD_NCCL_DEBUG_FILE"
        fi
    fi
}
