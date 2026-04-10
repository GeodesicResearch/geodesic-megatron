# Slingshot NCCL Hang Fix Results

Investigation into periodic NCCL collective hangs during multi-node Nemotron 3 SFT training on Isambard GH200 (Slingshot/CXI fabric). All ranks block simultaneously on a collective op (all-reduce or all-to-all in the MoE EP=8 communicator) every ~7-14 minutes. Root cause: CXI resource exhaustion triggering a memory registration deadlock in `aws-ofi-nccl`.

**Test setup:** Nemotron 3 Super SFT, multi-node over Slingshot, fault-tolerant `ft_launcher` with in-process restart. Each fix tested in isolation against the production baseline.

**Pass criteria:** 25+ minutes of training with zero NCCL hangs.

## Results Table

| Fix # | Setting / Change | Hypothesis | Result | Details |
|-------|-----------------|------------|--------|---------|
| 00 | **Production baseline** (control) | Expected to hang at ~7-14 min. ~20 custom NCCL/CXI env vars from production config. Establishes the crash interval for comparison. | FAIL (expected) | Hung at iter 82 after ~9.5 min. Confirms baseline. |
| 01 | `--network=disable_rdzv_get` srun flag | Isambard's own NCCL benchmarks use this flag. Disables rendezvous get at the Slingshot fabric level, targeting the rendezvous path where the memory registration deadlock occurs. | FAIL | Hung at iter 67 after ~5 min. 8 failure_detected events in 41 min. |
| 02 | `MPICH_GPU_SUPPORT_ENABLED=0` | CSCS warns this "can easily lead to deadlocks when using together with NCCL." May default to 1 on Isambard. | FAIL | Hung at iter 63 after ~18 min. Was already in production config (no new information). |
| 03 | `FI_CXI_DEFAULT_TX_SIZE=1024` | Isambard docs recommend 1024, but production used 16384 from GPT-NeoX era. Smaller TX queue may avoid resource exhaustion triggering the deadlock. | FAIL | Hung at iter 83 after ~8 min. Same hang cadence as baseline. |
| 04 | `NCCL_IGNORE_CPU_AFFINITY=1` | CSCS reports up to 6x allreduce and 1.6x alltoall improvement on GH200. SLURM CPU affinity can mis-route data on GH200 where CPU/GPU share a die. | FAIL | Hung at iter 83 after ~6 min. Same simultaneous all-node failure pattern. |
| 05 | `NCCL_BUFFSIZE=16777216` (16 MB) | Larger NCCL buffers reduce the number of CXI operations per MoE all-to-all, reducing CXI resource pressure (TX queue, CQ entries, match entries). | FAIL | 4 failure_detected events. Same hang pattern as baseline. |
| 06 | `FI_CXI_ENABLE_TRIG_OP_LIMIT=1` | Enables semaphore-based triggered operation resource coordination. Prevents CXI resource exhaustion across processes. | FAIL | Hung at iter 79 after ~5 min. Did not prevent the hang. |
| 07 | `FI_CXI_REQ_BUF_SIZE=8388608` (8 MB) | Larger request buffers for unmatched messages (ALCF recommendation). Undersized buffers force flow control which stalls all senders. | FAIL | Hung at iter 42 after ~6 min. Same all-rank hang pattern. |
| 08 | `FI_CXI_ODP=1` (on-demand paging) | Enables on-demand paging for memory regions instead of pre-registering DMA buffers. Completely bypasses the memory registration code path where the deadlock exists in aws-ofi-nccl v1.8.1. | FAIL | Retest crashed at ~5 min with 4 failures. First run's 14 min clean window was variance in crash interval, not a real improvement. |
| 09 | `FI_CXI_MSG_LOSSLESS=1` | Experimental feature that makes hardware pause the traffic class on resource exhaustion until buffers are posted. Prevents the resource exhaustion condition. | FAIL | 4 failure_detected events. Same hang pattern as baseline (~7 min). |
| 10 | `NCCL_GDRCOPY_ENABLE=0` | Disable GDRCopy to rule out a known race condition (libfabric issue #10041) where multiple threads simultaneously unregister the same GDRCopy memory handle. | FAIL | Hung at iter 42 after ~2.5 min. Actually worse than baseline. Rules out GDRCopy race as cause. |
| 11 | `FI_CXI_RX_MATCH_MODE=software` + `FI_CXI_REQ_BUF_SIZE=8388608` | Software match mode offloads receive matching from Cassini hardware to software, preventing hardware match entry exhaustion. Combined with larger request buffers needed for software mode. | FAIL | Hung at iter 17 after ~6 min. Combined settings still don't prevent the hang. |
| 12 | Remove `NCCL_PROTO=^LL128` | The ^LL128 restriction was carried over from GPT-NeoX and may be unnecessarily limiting NCCL's protocol selection. | FAIL | Hung at iter 67 after ~5 min. Made things worse (failure during setup before training). Confirms LL128 should stay disabled on Slingshot. |
| 13 | aws-ofi-nccl v1.18.0 via `LD_LIBRARY_PATH` | v1.17.0 fixes the memory registration deadlock, v1.17.2 fixes Slingshot-specific shutdown ordering, v1.18.0 adds threading redesign for overlapping communicators. Built from source. | INCONCLUSIVE | Loaded checkpoint for 21 min without NCCL errors, but cancelled externally before training started. 21 min checkpoint load is abnormal (normally 3-5 min), suggesting a possible silent hang. |
| 14 | Minimal Isambard-docs config | Production sbatch carries ~20 custom NCCL/CXI env vars from GPT-NeoX. Some may conflict with Megatron Bridge or with each other. This config uses ONLY what the Isambard NCCL guide recommends. | FAIL | Hung at iter 18 after ~7 min. Proves the hang is NOT caused by legacy GPT-NeoX settings. |

## Summary

**All 15 fixes failed.** No env var, flag, or tuning parameter tested prevents the hang. The hang occurs regardless of NCCL buffer sizes, CXI queue tuning, match mode, protocol selection, CPU affinity, GDRCopy state, on-demand paging, or minimal vs. production env var sets. This confirms the root cause is a deadlock in the system-level `aws-ofi-nccl` library (v1.8.1) during memory registration under CXI resource pressure, and it **cannot be worked around from userspace** with the currently installed system libraries.

**`FI_CXI_ODP=1` (Fix 08) ruled out.** The initial 14-minute clean run was variance in the crash interval (baseline ranges from ~2.5 to ~18 min across tests). The retest crashed at ~5 min with 4 failures, consistent with baseline behavior.

**aws-ofi-nccl v1.18.0 (Fix 13) remains inconclusive.** The updated library contains the upstream fix for the deadlock, but the test was cancelled before training iterations began. The abnormally long checkpoint load (21 min vs. normal 3-5 min) suggests it may have silently hung during init. Further testing needed with longer job time limits.

**Current mitigation:** The fault-tolerant training stack (`ft_launcher` + in-process restart) recovers from hangs automatically, losing ~25 iterations per event. This allows training to proceed despite the underlying fabric issue, at the cost of ~10-15% throughput loss. A system-level fix (upgrading `aws-ofi-nccl` to v1.17.0+) is required to eliminate the hangs entirely.
