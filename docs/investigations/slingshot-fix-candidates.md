# Slingshot NCCL Hang — 20 Fix Candidates (Ranked by Simplicity)

Each fix is a single change from the production sbatch. Tested by running 500 iterations
(~35 min at normal speed). A fix PASSES if training runs >25 min without a crash
(baseline crashes at ~7-14 min).

## The List

### Tier 1: Single env var or srun flag (simplest)

1. **`--network=disable_rdzv_get` srun flag**
   From Isambard's own NCCL benchmark docs. Disables rendezvous get at the Slingshot
   fabric level. We've never tried this. Directly targets the rendezvous path where
   the memory registration deadlock occurs.

2. **`MPICH_GPU_SUPPORT_ENABLED=0`**
   CSCS warns this "can easily lead to deadlocks when using together with NCCL."
   May default to 1 on Isambard. Zero-cost to test.

3. **`FI_CXI_DEFAULT_TX_SIZE=1024`**
   Isambard docs recommend 1024. We use 16384 (from an older config). Smaller TX
   queue may avoid the resource exhaustion that triggers the deadlock.

4. **`NCCL_NCHANNELS_PER_NET_PEER=4`**
   CSCS recommendation for point-to-point performance. Tested before as part of a
   bundle but never in isolation with production settings.

5. **`NCCL_BUFFSIZE=16777216` (16MB)**
   Larger NCCL buffers = fewer CXI operations per all-to-all. Reduces CXI resource
   pressure. Tested before in a bundle but never isolated.

6. **`FI_CXI_RDZV_PROTO=alt_read`**
   Isambard docs list this as recommended. Uses an alternative RDMA path for
   rendezvous that may avoid the deadlock. Previously caused NCCL Error 2 when
   bundled with other ALCF params — test in isolation.

7. **`FI_CXI_ENABLE_TRIG_OP_LIMIT=1`**
   Enables semaphore-based triggered op resource coordination. Prevents exhaustion
   across processes. NEW — never tested.

8. **`FI_CXI_REQ_BUF_SIZE=8388608` (8MB)**
   Larger request buffers for unmatched messages. ALCF recommendation. May reduce
   flow control stalls that precede the deadlock.

9. **`FI_CXI_ODP=1` (On-Demand Paging)**
   Enables on-demand paging instead of pinning all DMA buffers. Could avoid the
   memory registration deadlock entirely by not pre-registering.

10. **`FI_CXI_MSG_LOSSLESS=1`**
    Experimental: hardware pauses traffic class on resource exhaustion until buffers
    are posted. Prevents the condition that leads to the deadlock.

### Tier 2: Combined settings (2-3 changes)

11. **srun `--network=disable_rdzv_get` + `FI_CXI_RDZV_PROTO=alt_read`**
    Both the srun flag and env var target the rendezvous path. Combined, they
    completely bypass the default rendezvous implementation.

12. **`FI_CXI_RX_MATCH_MODE=software` + `FI_CXI_REQ_BUF_SIZE=8388608`**
    Software match mode doubled crash interval before. Adding proper request
    buffer size may eliminate remaining hangs.

13. **`FI_CXI_DEFAULT_TX_SIZE=1024` + `NCCL_BUFFSIZE=16777216`**
    Isambard-recommended TX size + larger NCCL buffers. Reduces per-operation CXI
    pressure from both directions.

14. **Isambard docs full recommended config**
    Apply ALL settings from the official NCCL guide as a bundle:
    `FI_CXI_DEFAULT_TX_SIZE=1024`, `FI_CXI_RDZV_PROTO=alt_read`,
    `FI_CXI_RDZV_THRESHOLD=0`, `FI_CXI_RDZV_GET_MIN=0`,
    `FI_CXI_RDZV_EAGER_SIZE=0`, `NCCL_GDRCOPY_ENABLE=1`,
    `FI_HMEM_CUDA_USE_GDRCOPY=1`

### Tier 3: Build/module changes (more complex)

15. **aws-ofi-nccl v1.17.3 rebuild (v2, correct flags)**
    Rebuilt with --with-mpi, --with-hwloc, --with-nccl matching the system module.
    Load module for deps, then swap the lib path (append, not prepend).

16. **aws-ofi-nccl v1.17.3 via local modulefile**
    Create ~/modulefiles/aws-ofi-nccl/1.17.3.lua that mimics the system module
    but points to our build.

17. **Downgrade to aws-ofi-nccl 1.6.0**
    `module load brics/aws-ofi-nccl/1.6.0` — older version may not have the
    deadlock bug (it was introduced between versions).

### Tier 4: Training config changes (architectural)

18. **Reduce EP from 8 to 4**
    Halves the all-to-all communication. May OOM (EP=4 uses 93GB peak on 95GB
    GPUs) but worth testing if nothing else works.

19. **`moe_token_dispatcher_type=allgather`**
    Switch from alltoall to allgather MoE dispatcher. Different communication
    pattern — AllGather instead of Send/Recv. Avoids the deadlocked code path.

20. **Disable distributed optimizer**
    `use_distributed_optimizer=false` — removes reduce_scatter from gradient sync.
    Uses more memory but changes the collective pattern.
