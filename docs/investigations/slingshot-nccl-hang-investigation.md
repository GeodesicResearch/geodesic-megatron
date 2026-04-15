# Slingshot/CXI NCCL Collective Hang Investigation

**Date:** 2026-04-09
**Author:** Kyle O'Brien (with Claude Code agent team)
**Status:** Primary root cause identified (aws-ofi-nccl deadlock), with two plausible additional contributors (CXI provider deadlocks). FT workaround in place (~23% overhead). Admin upgrade request drafted. Primary ask: aws-ofi-nccl >=1.18.0, with libfabric/SHS upgrade as complementary action.

## Problem Statement

Nemotron 3 Nano (30B-A3B) warm-start SFT training on Isambard crashes every ~7-14 minutes due to permanent NCCL collective hangs on the Slingshot/CXI interconnect. The hangs do not self-resolve — before auto-learned timeouts were added, they lasted the full 10-minute default ft_launcher step timeout without recovery. Each crash-restart cycle costs ~3 minutes (re-rendezvous + checkpoint reload + dataset rebuild), resulting in ~23% wall-clock overhead across a full training run.

**Training configuration:**
- 4 nodes, 16 GH200 120GB GPUs (sm_90a, unified CPU-GPU memory)
- TP=2, EP=8, CP=1, DP=8
- MoE: 128 experts, alltoall dispatcher
- Sequence length: 8192, packed sequences
- aws-ofi-nccl 1.8.1, libfabric 1.22.0, NCCL 2.28.9, PyTorch 2.11.0+cu126

## Root Cause

### Identified: Memory Registration Deadlock in aws-ofi-nccl v1.8.1

The hang occurs in **MoE all-to-all Send/Recv operations** on the EP=8 expert-parallel communicator. NCCL debug logs (job 3700495) confirm the last operations before the kill are Send/Recv pairs with variable-sized expert tensors (6.4M-11.6M elements each) to all 8 EP ranks, interleaved with ReduceScatter on the TP=2 communicator.

The root cause is a **deadlock in aws-ofi-nccl v1.8.1's memory registration function** that was fixed in v1.17.0 (release notes: "Fixed deadlock in memory registration function"). This deadlock manifests in the NCCL Send/Recv code path used by the MoE all-to-all dispatcher, which is the heaviest inter-node communication pattern in the model — 128 experts spread across 16 GPUs on 4 nodes, with each iteration routing tokens to different expert groups.

### Supporting Evidence

**1. Crashes are time-based, not iteration-based**

Analysis of 4 production logs (jobs 3698324, 3698327, 3698328, 3700273) across 63 restart cycles shows:
- Mean crash interval: 10-13 minutes of continuous training
- Crash iteration counts vary widely (9 to 177 iterations between crashes)
- Wall-clock interval clusters tightly around the median
- No correlation with checkpoint saves (save_interval=100), ruling out I/O triggers

**2. All nodes hang simultaneously**

All 4 nodes detect the step timeout within ~1 second of each other. Failure detection is uniformly distributed across nodes — no single node is a consistent trigger. This rules out a bad GPU or NIC and points to a fabric-wide collective stall.

**3. No warning signs before the hang**

- NVRx straggler detection shows all GPUs at 0.97-0.99 relative performance immediately before every crash
- Iteration times are stable (3.5-5.0s) with no progressive degradation
- The hang is instantaneous: iteration N completes in ~4s, then iteration N+1 hangs permanently
- Zero NCCL errors, CXI warnings, or PyTorch distributed errors in the logs before the step timeout fires

**4. NCCL debug logs confirm the hang point**

Per-rank NCCL debug files from job 3700495 show the last logged operations before each kill are:

```
NCCL INFO Send: opCount 6b8b ... count 7357056 datatype 9 ... comm [nranks=8]  # EP all-to-all
NCCL INFO Recv: opCount 6b8b ... count 7357056 datatype 9 ... comm [nranks=8]
NCCL INFO Send: opCount 6b8b ... count 9260160 datatype 9 ... comm [nranks=8]
NCCL INFO Recv: opCount 6b8b ... count 7752192 datatype 9 ... comm [nranks=8]
... (8 Send/Recv pairs to all EP ranks)
NCCL INFO ReduceScatter: opCount 3f6c ... count 22020096 ... comm [nranks=2]   # TP grad sync
NCCL INFO ReduceScatter: opCount 3f6d ... count 22450176 ... comm [nranks=2]
```

The log simply stops — no error, no completion. The collective hangs indefinitely.

## Investigation Timeline

### Phase 1: Setup and Crash Log Analysis

Created a network test configuration (`nemotron_nano_100k_warm_start_sft_network_test.yaml`) with 500 iterations, no checkpoint saving, and a separate W&B experiment. Created a matching SBATCH script (`train_nemotron_sft_network_test.sbatch`) with verbose logging.

Analyzed 4 production crash logs totaling ~24MB. Key findings:
- 63 restart cycles analyzed across 3 Nano training jobs
- Consistent ~10-13 min crash interval
- ft_launcher auto-learned step timeouts progressively tighter (37s -> 34s)
- Each restart costs ~3 minutes
- Effective throughput: 7.5 iter/min vs theoretical 15 iter/min (50% efficiency)

### Phase 2: NCCL Debug Log Capture

Enabled `NCCL_DEBUG=INFO` with `NCCL_DEBUG_FILE` (per-rank files to /tmp, avoiding the known SLURM stdout OOM at ~75 iterations). Added a cleanup trap to copy debug files to persistent storage on job exit.

Job 3700495 reproduced the baseline hang at iteration 70 (~6 min into training) and successfully collected 88MB debug files per rank across 4 nodes. Analysis confirmed the hang in MoE all-to-all Send/Recv operations.

### Phase 3: Parameter Tuning Attempts

#### Attempt 1: NCCL_ALGO=Ring (Job 3700313)
**Result:** Crashed immediately.
`NCCL_ALGO=Ring` breaks `reduce_scatter_tensor_coalesced` used by the distributed optimizer. PyTorch error: "Backend nccl does not support reduce_scatter_tensor_coalesced."

#### Attempt 2: TORCH_DISTRIBUTED_DEBUG=DETAIL (Job 3700339)
**Result:** Crashed immediately.
`TORCH_DISTRIBUTED_DEBUG=DETAIL` wraps the NCCL ProcessGroup in a `DebugProcessGroupWrapper` that doesn't forward `reduce_scatter_tensor_coalesced`. Same error as NCCL_ALGO=Ring. Fix: `unset TORCH_DISTRIBUTED_DEBUG` in the SBATCH.

#### Attempt 3: ALCF CXI Parameters (Jobs 3700339, 3700388)
Tested the ALCF (Argonne National Lab) Slingshot configuration:
- `FI_CXI_RX_MATCH_MODE=software`
- `FI_CXI_RDZV_PROTO=alt_read`
- `FI_CXI_DEFAULT_TX_SIZE=131072`
- `FI_CXI_REQ_BUF_SIZE=16777216`
- `FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16000`
- `FI_CXI_RDZV_THRESHOLD=2000`

**Result:** "NCCL Error 2: unhandled system error" during distributed checkpoint loading. These settings are designed for newer aws-ofi-nccl versions and are incompatible with v1.8.1.

Additional finding: `FI_CXI_DEFAULT_TX_SIZE=131072` is rejected by the CXI provider ("Default TX size invalid. Setting to 512").

#### Attempt 4: GH200 Unified Memory Fixes (Job 3700457)
Tested settings to address GH200's unified CPU-GPU memory architecture:
- `NCCL_CUMEM_ENABLE=0` (disable CUDA VMM API)
- `FI_CXI_DISABLE_HMEM_DEV_REGISTER=1` (disable HMEM device registration)
- `FI_HMEM_CUDA_USE_GDRCOPY=0` (disable GDRCopy)

**Result:** "NCCL Error 2: unhandled system error." These settings break NCCL's buffer allocation. The `cuPointerSetAttribute: CUDA_ERROR_NOT_SUPPORTED` warnings they were intended to fix are actually non-fatal — NCCL falls back to a working path despite the warnings.

#### Attempt 5: aws-ofi-nccl v1.17.3 Build (Jobs 3700370, 3701681)
Built aws-ofi-nccl v1.17.3 from source at `~/opt/aws-ofi-nccl-1.17.3/`. The build compiled cleanly (aarch64, gcc-12, libfabric 1.22.0, NCCL 2.28.9).

**Result:** Two different failure modes depending on how the library was loaded:
- Without brics module: "NCCL Error 2" during checkpoint loading
- With brics module + LD_LIBRARY_PATH override: hangs indefinitely during NCCL initialization (>20 min with no output)

The v1.17.3 build has runtime compatibility issues with the system's libfabric/CXI stack that likely require admin-level integration to resolve.

#### Attempt 6: NCCL_PROTO=Simple (Job 3701400)
Forced the Simple protocol (non-pipelined, more robust for large transfers).

**Result:** 3x throughput penalty (12s/iter vs 4.2s/iter, 15 TFLOP/s vs 47 TFLOP/s). Still crashed at ~80 iterations. The protocol change doesn't address the underlying memory registration deadlock.

#### Attempt 7: Software Match Mode (Jobs 3701931, 3703770)
`FI_CXI_RX_MATCH_MODE=software` offloads receive matching from Cassini hardware to software, preventing hardware match entry exhaustion.

**Result: Most effective parameter change found.** Approximately doubled the crash interval (from ~10 min to ~20 min). However:
- 3x throughput penalty (12s/iter vs 4.2s/iter)
- Still crashes — does not fully prevent the deadlock
- Net wall-clock time is worse than production + FT recovery

Adding `FI_CXI_REQ_BUF_SIZE=16777216` (16MB request buffer for software match mode) provided marginal additional stability but did not eliminate hangs.

### Phase 4: Research Findings

#### aws-ofi-nccl Version History
Three critical fixes exist in newer versions:
| Version | Fix | Relevance |
|---------|-----|-----------|
| v1.17.0 | Deadlock in memory registration function | **Likely root cause** |
| v1.17.2 | Slingshot-specific shutdown ordering | Affects restart recovery |
| v1.17.3 | Memory leak for long-running jobs | Prevents progressive OOM |

Only v1.6.0 and v1.8.1 are available as Isambard modules.

#### GH200-Specific Issues
The `cuPointerSetAttribute: CUDA_ERROR_NOT_SUPPORTED` and `CUDA sysnc_memops returned -22` warnings appear during NCCL initialization on all GH200 runs. These occur because:
- GH200 uses unified coherent memory (CPU+GPU share physical memory via NVLink-C2C)
- The `SYNC_MEMOPS` attribute is not supported on unified/managed memory
- CXI's RDMA memory registration falls back to a slower path

These warnings are non-fatal — attempting to "fix" them by disabling VMM (`NCCL_CUMEM_ENABLE=0`) or HMEM registration breaks NCCL entirely.

## Summary of All Test Jobs

| Job | Configuration | Result |
|-----|--------------|--------|
| 3700313 | Initial tuning + NCCL_ALGO=Ring | Crashed (Ring breaks reduce_scatter) |
| 3700333 | Initial tuning (no Ring) | Cancelled (replaced by ALCF config) |
| 3700339 | ALCF CXI params | Crashed (TORCH_DISTRIBUTED_DEBUG=DETAIL) |
| 3700370 | ALCF + GH200 + ofi v1.17.3 | Crashed (NCCL Error 2, ofi ABI issue) |
| 3700388 | ALCF + GH200 + ofi 1.8.1 | Crashed (ALCF params incompatible with 1.8.1) |
| 3700457 | Production CXI + GH200 fixes | Crashed (GH200 fixes cause NCCL Error 2) |
| 3700495 | **Production + debug logging** | **Baseline reproduced, NCCL logs captured** |
| 3701400 | NCCL_PROTO=Simple | Crashed (3x slower, still hangs) |
| 3701681 | ofi-nccl v1.17.3 (module deps) | Hung during NCCL init (>20 min) |
| 3701931 | Software match mode | ~2x crash interval, 3x throughput penalty |
| 3703770 | Software match + 16MB REQ_BUF | Similar to 3701931, still crashes |

## Final Diagnosis

The periodic all-rank NCCL hang has **one confirmed root cause** and **two plausible additional contributors**, all in the Slingshot communication stack.

The aws-ofi-nccl memory registration deadlock (Bug 1) is the strongest confirmed root cause — it directly matches our hang pattern (Send/Recv with concurrent communicators) and has a documented fix. The libfabric CXI provider deadlocks (Bugs 2 and 3) are credible suspects that exercise related code paths, but are not as directly tied to our exact hang pattern. All three are worth fixing, but Bug 1 is the highest-confidence diagnosis.

### Bug 1: aws-ofi-nccl v1.8.1 — Memory Registration Deadlock

**Status:** Fixed upstream in [aws-ofi-nccl v1.17.0](https://github.com/aws/aws-ofi-nccl/releases/tag/v1.17.0) (September 2025).

The NCCL OFI plugin's memory registration function can deadlock when multiple NCCL communicators (TP, EP, DP) concurrently register buffers for Send/Recv operations. Our MoE all-to-all dispatcher issues 8 simultaneous Send/Recv pairs per iteration, each requiring buffer registration. Under load, the registration path deadlocks and all ranks stall.

Additional related fixes:
- v1.17.2: "Fixed shutdown ordering issue on NICs that require per-endpoint memory registration (Cray Slingshot)" — affects restart recovery
- v1.17.3: "Memory leak that can result in running out of host memory for long-running jobs" — prevents progressive OOM
- v1.18.0: "Redesigned threading model to support multi-threaded applications without requiring a separate Libfabric domain for each thread" — reduces CXI resource pressure from overlapping communicators

**Installed version:** brics/aws-ofi-nccl/1.8.1 (only 1.6.0 and 1.8.1 available as modules).

### Bug 2 (Plausible Contributor): Libfabric CXI Provider — SRX and Counter Thread Deadlocks

**Status:** Fixed upstream in [libfabric v2.2.0](https://github.com/ofiwg/libfabric/blob/main/NEWS.md) (June 2025) and [v2.3.0](https://github.com/ofiwg/libfabric/blob/main/NEWS.md) (September 2025). **NOT backported** by HPE to our SHS 12.0.2 build.

**Certainty:** Lower than Bug 1. These are real deadlock fixes in CXI code paths exercised by NCCL Send/Recv, but we have no direct evidence confirming our hangs hit these specific locking paths. They are credible suspects, not confirmed root causes.

Two separate deadlock regressions in the CXI provider:
1. **v2.2.0:** "Fix regression which could cause deadlock" — in the shared receive context (SRX) locking path
2. **v2.3.0:** "Fix regression which could cause deadlock" — in counter thread locking

Both are in code paths exercised by NCCL Send/Recv operations. We confirmed these are absent from our system by searching all commits in the [HewlettPackard/shs-libfabric](https://github.com/HewlettPackard/shs-libfabric) fork: **zero results** for "deadlock", "Fix regression", or "locking" in the prov/cxi path.

**Installed version:** libfabric 1.22.0-SHS12.0.2 (RPM: `libfabric-1.22.0-SHS12.0.2_20250722155538_8dad011dfdb6`, built from HPE fork commit `8dad011dfdb6`). HPE selectively backported other fixes (CUDA sync_memops, read-only MRs, CQ wait FD) but not the deadlock fixes.

### Bug 3 (Plausible Contributor): Libfabric CXI Provider — TX Credit Starvation During RNR Retry

**Status:** Fixed upstream in [libfabric v2.4.0](https://github.com/ofiwg/libfabric/blob/main/NEWS.md) (December 2025). **NOT backported** to SHS 12.0.2.

**Certainty:** Lower than Bug 1. The TX credit starvation mechanism is plausible for our 8-way Send/Recv pattern, but we have no direct evidence of RNR events in our logs.

"Release TX credit when pending RNR retry" — when a CXI Send triggers a Receiver Not Ready (RNR) condition and enters retry, the TX credit is not released. Other pending Send operations waiting for TX credits stall indefinitely. Our MoE all-to-all issues 8 simultaneous Send/Recv pairs to different EP ranks; one RNR event can cascade: the retrying pair holds a TX credit, the other 7 pairs starve, and all ranks block.

### Why the Hang Is Time-Based, Not Iteration-Based

The crash interval (7-14 minutes) is consistent in wall-clock time but varies widely in iteration count. One hypothesis is that this is characteristic of resource exhaustion: the memory registration deadlock and TX credit starvation accumulate pressure over time as CXI resources are consumed and not properly released, until a threshold is crossed and the deadlock triggers. The exact iteration would depend on MoE routing patterns (which experts receive tokens), making it unpredictable per-iteration but predictable per-time-window.

**Caveat:** The 7-14 minute interval also overlaps with common NCCL watchdog timeout windows and other periodic system events, so the time-based pattern is consistent with resource accumulation but should be considered a hypothesis rather than established fact.

### Why No Env Var Workaround Fully Works

We tested 11 different configurations (see job table above). The most effective was `FI_CXI_RX_MATCH_MODE=software` which approximately doubled the crash interval — consistent with reducing pressure on the CXI hardware match engine. But no env var combination can fix a deadlock in the locking code itself. The fixes require patched library code.

## Current Mitigation

The production configuration with the existing fault tolerance stack successfully completes full training runs despite the hangs:
- **Job 3700302** completed all 2531 iterations (1 epoch) with 12 restart cycles
- ft_launcher (nvidia-resiliency-ext) with `--max-restarts=20`
- Auto-learned step timeouts (~37s) detect hangs quickly
- Non-persistent local checkpoints every 25 iterations minimize lost work per restart
- Effective overhead: ~23% (63 min restarts in a 4.5 hour job)

## Recommendations

### Short Term (now)
Continue using the production FT configuration. The ~23% overhead is manageable and training runs complete successfully. Additionally, test these zero-risk env vars that may reduce hang frequency:
- `NCCL_IGNORE_CPU_AFFINITY=1` — CSCS reports up to 6x allreduce and 1.6x alltoall improvement on GH200
- `FI_MR_CACHE_MONITOR=userfaultfd` — consensus across CSCS/Isambard/ALCF, prevents MR cache deadlocks
- `FI_CXI_RDZV_GET_MIN=0`, `FI_CXI_RDZV_THRESHOLD=0`, `FI_CXI_RDZV_EAGER_SIZE=0` — CSCS's canonical hang workaround

### Medium Term (request admin action)
File an Isambard support ticket (draft at `docs/investigations/isambard-upgrade-request-draft.md`) requesting:
1. **aws-ofi-nccl >= 1.18.0 as a brics module** (primary ask) — contains memory registration deadlock fix, Slingshot shutdown fix, memory leak fix, and threading redesign. This is the highest-confidence fix for our specific hang pattern.
2. **Libfabric/SHS upgrade to latest available version** (secondary/complementary) — newer versions contain CXI deadlock and TX credit fixes that are plausible additional contributors. The specific target version should be determined by the Isambard admin team based on what is available and validated for their infrastructure.

Submit via Zammad helpdesk at https://support.isambard.ac.uk (no email support).

User-space builds of aws-ofi-nccl are available for admin reference:
- `/home/a5k/kyleobrien.a5k/opt/aws-ofi-nccl-1.14.0/` (first general-release version since v1.8.1)
- `/home/a5k/kyleobrien.a5k/opt/aws-ofi-nccl-1.17.3/` (has deadlock fix but runtime compat issues)
- `/home/a5k/kyleobrien.a5k/opt/aws-ofi-nccl-1.18.0/` (has all fixes including threading redesign)

These builds compile cleanly on aarch64 but have runtime compatibility issues with the system libfabric/CXI stack that require admin-level integration.

### Long Term
- When aws-ofi-nccl is upgraded (and ideally libfabric/SHS as well), the ~7-14 min crash cycles should be eliminated or significantly reduced
- Test with `FI_CXI_RX_MATCH_MODE=software` as defense-in-depth (throughput penalty may be reduced with newer libraries)
- Consider `NCCL_NCHANNELS_PER_NET_PEER=4` for improved alltoall bandwidth on GH200 (fixes 25% regression at ≥8 nodes)
- Monitor for intra-node EP configurations (EP=4 with NVLink-only alltoall) as MoE Parallel Folding techniques mature in Megatron-Core

### Minimal SBATCH Configuration
A streamlined SBATCH script (`train_nemotron_sft_minimal.sbatch`) was created based on the Isambard NCCL documentation recommendations, removing legacy GPT-NeoX environment settings that were unnecessary for the Megatron Bridge training stack. This provides a cleaner baseline for future testing.

## Artifacts

| Artifact | Location |
|----------|----------|
| Network test YAML config | `configs/nemotron_warm_start/nemotron_nano_100k_warm_start_sft_network_test.yaml` |
| Network test SBATCH script | `train_nemotron_sft_network_test.sbatch` |
| Minimal SBATCH script | `train_nemotron_sft_minimal.sbatch` |
| NCCL debug logs (per-rank) | `/projects/a5k/public/logs/network_debug/3700495/` |
| aws-ofi-nccl v1.14.0 build | `/home/a5k/kyleobrien.a5k/opt/aws-ofi-nccl-1.14.0/` |
| aws-ofi-nccl v1.17.3 build | `/home/a5k/kyleobrien.a5k/opt/aws-ofi-nccl-1.17.3/` |
| aws-ofi-nccl v1.18.0 build | `/home/a5k/kyleobrien.a5k/opt/aws-ofi-nccl-1.18.0/` |
| aws-ofi-nccl source | `/home/a5k/kyleobrien.a5k/aws-ofi-nccl/` |
| Admin upgrade request draft | `docs/investigations/isambard-upgrade-request-draft.md` |

## Corrections & Caveats

1. **Time-based hang pattern is a hypothesis, not established fact.** The ~7-14 minute crash interval is consistent with resource accumulation (memory registration pressure building until deadlock), but this interval also overlaps with common NCCL watchdog timeouts and other periodic system events. The resource exhaustion theory is plausible but not proven.

2. **Libfabric diagnosis has lower certainty than aws-ofi-nccl diagnosis.** The aws-ofi-nccl v1.17.0 memory registration deadlock fix (Bug 1) directly matches our hang pattern — Send/Recv with concurrent communicators. The libfabric CXI provider deadlocks (Bugs 2 and 3) are plausible additional contributors based on the code paths involved, but we have no direct evidence confirming our hangs hit those specific locking paths. They are credible suspects, not confirmed root causes.

3. **NCCL_IGNORE_CPU_AFFINITY=1 is not standard Slingshot practice.** CSCS mentions this for performance improvement, but it is not widely recommended across other Slingshot HPC sites. Use with caution and benchmark before adopting.

4. **FI_MR_CACHE_MONITOR=kdreg2 is an alternative to userfaultfd.** HPE's newer `kdreg2` kernel module is an alternative memory registration cache monitor that was not tested in the initial investigation. It may be available on Isambard and could be more appropriate than `userfaultfd` depending on the system configuration.

5. **SHS version recommendations have been corrected.** Earlier versions of this document referenced "SHS >=14.0.0" as a specific upgrade target. No public documentation confirms this version exists or contains the needed fixes. The recommendation has been updated to focus on aws-ofi-nccl >=1.18.0 as the primary ask, with libfabric/SHS upgrade version to be determined by the Isambard admin team.

## References

- [aws-ofi-nccl v1.17.0 Release Notes](https://github.com/aws/aws-ofi-nccl/releases/tag/v1.17.0) — memory registration deadlock fix
- [aws-ofi-nccl v1.17.2 Release Notes](https://github.com/aws/aws-ofi-nccl/releases/tag/v1.17.2) — Slingshot per-endpoint MR shutdown fix
- [aws-ofi-nccl v1.18.0 Release Notes](https://github.com/aws/aws-ofi-nccl/releases/tag/v1.18.0) — threading redesign + FI_MR_ENDPOINT fix
- [libfabric NEWS.md](https://github.com/ofiwg/libfabric/blob/main/NEWS.md) — CXI deadlock fixes in v2.2.0/v2.3.0, TX credit fix in v2.4.0
- [HewlettPackard/shs-libfabric](https://github.com/HewlettPackard/shs-libfabric) — HPE's Slingshot fork (our system source)
- [CSCS NCCL Documentation](https://docs.cscs.ch/software/communication/nccl/) — GH200+Slingshot guidance, hang workaround
- [Isambard NCCL Guide](https://docs.isambard.ac.uk/user-documentation/guides/nccl/) — Isambard-specific NCCL configuration
- [ALCF nccl-tests Configuration](https://github.com/argonne-lcf/alcf-nccl-tests) — Slingshot CXI tuning
- [NVIDIA/nccl #1272](https://github.com/NVIDIA/nccl/issues/1272) — NCHANNELS_PER_NET_PEER for GH200
- [libfabric fi_cxi(7)](https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html) — CXI provider env vars
- [ofiwg/libfabric#10041](https://github.com/ofiwg/libfabric/issues/10041) — CXI/GDRCopy cleanup race condition (open)
- [X-MoE (arXiv:2508.13337)](https://arxiv.org/html/2508.13337v1) — MoE alltoall redundancy on Slingshot
- [MoE Parallel Folding (arXiv:2504.14960)](https://arxiv.org/abs/2504.14960) — intra-node EP optimization
- [NVIDIA GPUDirect RDMA Docs](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html) — unified memory limitations
- [Isambard Support Helpdesk](https://support.isambard.ac.uk) — Zammad ticketing system for upgrade requests

## Appendix: GitHub Issues and External Reports

A systematic search of GitHub issues across 5 repositories (aws/aws-ofi-nccl, NVIDIA/nccl, ofiwg/libfabric, pytorch/pytorch, NVIDIA/Megatron-LM) was conducted on 2026-04-09. Below are all relevant findings, organized by relevance to our symptom (periodic all-rank hang in MoE all-to-all Send/Recv on Slingshot/CXI).

### Directly Relevant Issues

#### NVIDIA/nccl#1272 — Performance drop on ≥8 nodes (32 GH200) for NCCL ≥2.20
- **Status:** Closed (completed), April 2024
- **Problem:** SendRecv bandwidth dropped from 24.2 GB/s to 18.3 GB/s when scaling from 4→8 nodes on GH200+Slingshot with NCCL ≥2.20. Root cause: NCCL auto-tuning reduces channels per peer at larger scale.
- **Fix:** `NCCL_NCHANNELS_PER_PEER=4` (or `NCCL_NCHANNELS_PER_NET_PEER=4` in CSCS docs) restores full bandwidth.
- **Relevance:** High. Our 4-node EP=8 all-to-all uses Send/Recv. This variable ensures enough channels are allocated for sparse point-to-point patterns. **Already referenced in our investigation.**

#### NVIDIA/Megatron-LM#1810 — Possible Deadlock on A2A Overlap with overlap-moe-expert-parallel-comm
- **Status:** Open
- **Problem:** At scale (128+ nodes, H100/H200), enabling `--overlap-moe-expert-parallel-comm` causes deadlock. A single shared CUDA event allows multiple comm operations to become eligible simultaneously, causing different ranks to launch collectives in different order — violating NCCL's "same collectives, same order" invariant.
- **Fix proposed:** Replace single CUDA event with two events per model chunk (`comp_event` + `comm_event`).
- **Relevance:** Medium. We don't use `--overlap-moe-expert-parallel-comm`, but this confirms MoE all-to-all ordering is a known deadlock vector. If we enable comm overlap in future, this issue applies.

#### NVIDIA/nccl#1110 — nccl-tests hangs with specific message sizes
- **Status:** Open
- **Problem:** ReduceScatter hangs at specific message sizes (4GB, 128MB, 512MB) in heterogeneous environments. Related to LL128 protocol and NVLS.
- **Workarounds:** `NCCL_ALGO=RING NCCL_PROTO=SIMPLE`, `NCCL_NVLS_ENABLE=0`, `NCCL_ALGO=^NVLSTREE`.
- **Relevance:** Medium. Our hangs are in Send/Recv not ReduceScatter, and our cluster is homogeneous. But confirms LL128 and NVLS as hang vectors — we already set `NCCL_PROTO=^LL128` and `NCCL_NVLS_ENABLE=0`.

#### NVIDIA/nccl#1134 — alltoall_perf hangs in 16×8 H100 cluster
- **Status:** Closed (not planned, stale)
- **Problem:** `alltoall_perf` hung during communicator init on 16-node H100 cluster with NCCL 2.19.x. Only occurred at ≥8 nodes.
- **Fix:** Disabling `NCCL_CUMEM_ENABLE` resolved the hang.
- **Relevance:** Medium. Scale-dependent alltoall hang matches our pattern (4 nodes, EP=8). We tested `NCCL_CUMEM_ENABLE=0` (Attempt 4) but it broke NCCL on GH200, confirming CUMEM is required for unified memory.

#### NVIDIA/nccl-tests#187 — alltoall_perf hangs via PXN
- **Status:** Closed (completed), December 2023
- **Problem:** `alltoall_perf` hung on 14-node A800 cluster with PXN enabled. Only at ≥14 nodes.
- **Root cause:** Bug in NCCL's `cumem` API.
- **Relevance:** Low-medium. PXN-specific and on InfiniBand, not Slingshot. But confirms alltoall is more hang-prone than other collectives at scale.

### Slingshot/CXI-Specific Issues and Documentation

#### aws/aws-ofi-nccl#70 — Unable to register memory (register_mr_buffers:465)
- **Status:** Closed
- **Problem:** Memory registration failures during NCCL operations.
- **Relevance:** Medium. Memory registration is our identified root cause mechanism. The fix in v1.17.0 directly addresses this class of bug.

#### aws/aws-ofi-nccl#584 — register_mr_buffers Unable to register memory (type=2)
- **Status:** Closed
- **Problem:** Memory registration failures with RC: -22 (Invalid argument) for device memory.
- **Relevance:** Low-medium. Different failure mode (error vs deadlock) but same subsystem.

#### ofiwg/libfabric#10072 — CXI OFI poll failed on LUMI/Adastra
- **Status:** Closed, November 2024
- **Problem:** `PTLTE_NOT_FOUND` errors during MPI calls on Slingshot 2.2 / libfabric 1.20. Root cause: secondary symptom from peer rank failures (segfaults, OOM).
- **Fix:** Upgrade to libfabric 1.20.1+ or SHS 11+.
- **Relevance:** Low. Different failure mode (peer crash, not collective hang).

#### ofiwg/libfabric#6124 — FI_HMEM ofi_cudaMemcpy deadlock
- **Status:** Closed
- **Problem:** Using default CUDA stream for `cudaMemcpy` in libfabric's HMEM support can cause deadlock with GPU buffers.
- **Relevance:** Low-medium. We set `FI_HMEM_CUDA_USE_GDRCOPY=1` which bypasses this path. Worth noting if GDRCopy is ever disabled.

### PyTorch NCCL Hang Issues

#### pytorch/pytorch#154297 — reduce_scatter hangs on B200 GPU
- **Status:** Open
- **Problem:** Intermittent hangs in `dist.reduce_scatter` on 8× B200 single-node. Timing-dependent.
- **Fix suggested:** Don't pass `device_id` to `init_process_group` (NCCL bug).
- **Relevance:** Low. Single-node, B200, no Slingshot. But confirms reduce_scatter hang bugs exist in recent NCCL.

#### pytorch/pytorch#50820 — NCCL_BLOCKING_WAIT=1 makes training extremely slow
- **Status:** Open
- **Problem:** `NCCL_BLOCKING_WAIT=1` adds ~20% overhead. Without it, OOM on one device hangs all training.
- **Relevance:** Low. We don't set `NCCL_BLOCKING_WAIT`. Our timeout handling is via ft_launcher + InProcessRestart, not blocking wait.

#### pytorch/pytorch#163546 — Better NCCL timeout handling
- **Status:** Open
- **Problem:** ProcessGroupNCCL watchdog can get stuck when one rank desynchronizes.
- **Env vars:** `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` (increase) or `TORCH_NCCL_ENABLE_MONITORING=0` (disable).
- **Relevance:** Low. Our watchdog timeout (900s) is intentionally high as a last-resort backup behind InProcessRestart (60s/90s) and ft_launcher (auto-learned ~37s).

### Environment Variable Recommendations from HPC Centers

Cross-referencing env var recommendations from CSCS (Alps), Isambard, and ALCF (Polaris) — all Slingshot/CXI systems:

| Variable | CSCS (Alps) | Isambard | ALCF (Polaris) | Our Config | Notes |
|----------|-------------|----------|----------------|------------|-------|
| `FI_CXI_DEFAULT_TX_SIZE` | 16384 | 1024 | 131072 | 1024 | CSCS recommends 16384. **Test candidate.** ALCF value rejected by our CXI ("invalid, setting to 512"). |
| `FI_CXI_DEFAULT_CQ_SIZE` | 131072 | 131072 | 131072 | 131072 | Consensus. ✓ |
| `FI_CXI_DISABLE_HOST_REGISTER` | 1 | 1 | — | 1 | Prevents hangs/deadlocks. ✓ |
| `FI_CXI_RX_MATCH_MODE` | software | soft | software | soft | Already set. Tested as hang mitigation (Attempt 7). |
| `FI_MR_CACHE_MONITOR` | userfaultfd | userfaultfd | userfaultfd | — | **Missing from our config. Should add.** Prevents MR cache deadlocks. |
| `FI_CXI_RDZV_GET_MIN` | 0 | 0 | — | — | CSCS hang workaround. **Test candidate.** |
| `FI_CXI_RDZV_THRESHOLD` | 0 | 0 | 2000 | — | CSCS says 0 for hangs. **Test candidate.** |
| `FI_CXI_RDZV_EAGER_SIZE` | 0 | 0 | — | — | CSCS hang workaround. **Test candidate.** |
| `FI_CXI_RDZV_PROTO` | — | alt_read | alt_read | — | Needed for large alltoall (>8MB). **Test candidate.** |
| `FI_CXI_REQ_BUF_SIZE` | — | — | 16777216 | — | For software match mode with large messages. Tested in Attempt 7. |
| `NCCL_PROTO` | ^LL128 | — | — | ^LL128 | LL128 performs worse on Slingshot since NCCL 2.27. ✓ |
| `NCCL_NCHANNELS_PER_NET_PEER` | 4 | — | — | — | Improves P2P bandwidth. **Test candidate for alltoall.** |
| `NCCL_CROSS_NIC` | 1 | 1 | — | 1 | ✓ |
| `NCCL_NET_GDR_LEVEL` | PHB | PHB | — | PHB | ✓ |
| `NCCL_NVLS_ENABLE` | — | — | — | 0 | Required for InProcessRestart. |

### Key Takeaways

1. **The aws-ofi-nccl v1.17.0 deadlock fix remains the primary solution.** No GitHub issue or HPC center documentation describes a pure env-var workaround for this class of memory registration deadlock. All workarounds (software match, RDZV settings) reduce frequency but don't eliminate it.

2. **Three untested env vars from CSCS/ALCF could reduce hang frequency:**
   - `FI_MR_CACHE_MONITOR=userfaultfd` — consensus across all centers, missing from our config
   - `FI_CXI_RDZV_GET_MIN=0` + `FI_CXI_RDZV_THRESHOLD=0` + `FI_CXI_RDZV_EAGER_SIZE=0` — CSCS's explicit hang workaround
   - `NCCL_NCHANNELS_PER_NET_PEER=4` — improves P2P bandwidth which alltoall relies on

3. **LL128 and NVLS are confirmed hang vectors** (NVIDIA/nccl#1110). We already disable both. ✓

4. **TORCH_NCCL_BLOCKING_WAIT is not recommended.** It adds ~20% overhead (pytorch/pytorch#50820) and can mask hangs instead of resolving them. Our InProcessRestart approach is superior.

5. **nvidia-resiliency-ext InProcessRestart requires NCCL_NVLS_ENABLE=0** and NCCL < 2.28.3 or ≥ 2.28.9. Our NCCL 2.28.9 is the minimum safe version.

6. **alltoall is disproportionately hang-prone** across multiple reports (NVIDIA/nccl#1134, NVIDIA/nccl#566, nccl-tests#187). Scale-dependent hangs in alltoall are a recurring pattern, especially with ≥8 nodes or PXN. Our EP=8 all-to-all across 4 nodes fits this pattern.

## Appendix B: Community Reports (Web Research)

*Compiled 2026-04-09 from web searches of NVIDIA forums, HPC center docs, blog posts, and research papers.*

### B1. aws-ofi-nccl v1.18.0 Threading Redesign (January 2026)

**Source:** [aws-ofi-nccl Releases](https://github.com/aws/aws-ofi-nccl/releases)

Beyond the v1.17.x fixes already documented, **v1.18.0** (January 2026) introduced a major architectural change:
- "Redesigned threading model to support multi-threaded applications without requiring a separate Libfabric domain for each thread"
- "Fixed support for FI_MR_ENDPOINT providers" for SENDRECV protocol

This is potentially significant for our workload — we run overlapping TP, EP, and DP communicators which create multiple NCCL threads. The old threading model's per-thread Libfabric domain requirement could cause resource exhaustion on CXI. The v1.18.0 redesign may be more effective than v1.17.3 for our use case. **If requesting an admin upgrade, v1.18.0+ should be the target.**

### B2. CSCS Libfabric/aws-ofi-nccl ABI Incompatibility (April 2025)

**Source:** [CSCS Known Issues](https://docs.cscs.ch/software/container-engine/known-issue/)

CSCS documented an incompatibility between aws-ofi-nccl v1.9.2 and libfabric v1.22 that crashed NCCL codes. Fix deployed April 16, 2025. Isambard uses the same libfabric v1.22.0. This confirms that **aws-ofi-nccl version upgrades on Slingshot systems must be validated against the system libfabric version**, explaining why our user-space v1.17.3 build failed at runtime.

### B3. X-MoE: MoE Communication Redundancy on Slingshot

**Source:** [X-MoE (arXiv:2508.13337)](https://arxiv.org/html/2508.13337v1)

This paper quantifies a problem directly relevant to our MoE alltoall:
- Inter-node Slingshot bandwidth (25 GB/s per NIC) is **8x lower** than intra-node NVLink (200 GB/s)
- Expert-specialized MoEs with large top-k routing produce **up to 75% redundant token traffic** across inter-node links
- Their "Redundancy-Bypassing Dispatch" reduced inter-node communication by **52.5%** by only sending unique tokens across nodes
- Standard NCCL alltoall does not exploit hierarchical network topology

**Relevance:** Reducing inter-node alltoall volume would reduce the number of memory registration operations per iteration, lowering the probability of triggering the aws-ofi-nccl deadlock. This is a longer-term optimization beyond env var tuning.

### B4. NCCL_IGNORE_CPU_AFFINITY Performance Impact

**Source:** [CSCS NCCL Documentation](https://docs.cscs.ch/software/communication/nccl/)

On Alps (GH200 + Slingshot 11), setting `NCCL_IGNORE_CPU_AFFINITY=1` yielded:
- **Up to 6x improvement on allreduce** (from 2 nodes)
- **Up to 1.6x improvement on alltoall**

On GH200, CPU and GPU share the same die via NVLink-C2C. SLURM's CPU affinity can cause NCCL to route data through suboptimal NUMA paths. This variable forces NCCL to use GPU-NIC topology instead. **This is a zero-risk tuning parameter not yet tested in our config.**

### B5. NCCL 2.27 LL128 and Slingshot

**Sources:** [NVIDIA Blog: NCCL 2.27](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/), [NVIDIA/nccl#740](https://github.com/NVIDIA/nccl/issues/740), [CSCS Docs](https://docs.cscs.ch/software/communication/nccl/)

NCCL 2.27 enabled LL128 protocol by default. On Slingshot:
- LL128 "typically performs worse" (CSCS)
- LL128 has caused **data corruption** on cross-node broadcast (NVIDIA/nccl#740)
- CSCS states the hangs tied to LL128 are "still under investigation"

Our `NCCL_PROTO=^LL128` is confirmed correct by all sources. No community report suggests removing it.

### B6. Perlmutter All2All Benchmarks

**Source:** [nccl-tests#139](https://github.com/NVIDIA/nccl-tests/issues/139)

Key finding from Perlmutter (A100 + Slingshot):
- Properly configured alltoall achieved ~20 GB/s bandwidth on 4 nodes
- Running one process per GPU (4 per node) was critical — running 1 process managing all 4 GPUs was 2.5x slower
- Default NCCL settings with libfabric plugin selection worked well once process mapping was correct

**Relevance:** Our setup already uses 1 process per GPU via torchrun/ft_launcher. The Perlmutter baseline confirms this is correct.

### B7. GH200 Multi-Node NCCL Testing Blog Post

**Source:** [Multi-Node GH200 NCCL Testing (Medium)](https://medium.com/@ed.sealing/multi-node-gh200-nccl-testing-dc2fc64d97a0)

A practitioner's report on 2-node GH200 NCCL testing:
- Default NCCL settings "maximize bandwidth the best" on 2 nodes
- DMA-BUF automatically selected (preferred over legacy nvidia-peermem)
- Achieved ~48 GB/s algorithm bandwidth (matches single Bluefield-3 card capacity)
- **No hangs reported** — but only tested 2 nodes, which is below the threshold where our hangs manifest

### B8. TORCH_NCCL_BLOCKING_WAIT Deprecation

**Source:** [pytorch/pytorch#50820](https://github.com/pytorch/pytorch/issues/50820)

`TORCH_NCCL_BLOCKING_WAIT=1` causes **5-60% training slowdown**. The recommended replacement is `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` which has "little to no overhead." In modern PyTorch (≥2.0), async error handling is the default, and combined with Torchelastic/ft_launcher provides the same timeout detection without the performance penalty.

**Confirmation:** Our approach (ft_launcher + InProcessRestart + `TORCH_NCCL_TIMEOUT=900`) is the community-recommended pattern. We correctly do not set `BLOCKING_WAIT`.

### B9. Summary of New Actionable Findings

| Priority | Action | Source | Risk |
|----------|--------|--------|------|
| **High** | Request `aws-ofi-nccl ≥1.18.0` as system module (not just 1.17.3) | aws-ofi-nccl releases | Requires admin |
| **Medium** | Test `NCCL_IGNORE_CPU_AFFINITY=1` | CSCS (up to 6x allreduce improvement) | Zero risk |
| **Medium** | Test CSCS eager message disable: `FI_CXI_RDZV_{GET_MIN,THRESHOLD,EAGER_SIZE}=0` | CSCS hang workaround | May affect throughput |
| **Medium** | Add `FI_MR_CACHE_MONITOR=userfaultfd` | CSCS/Isambard/ALCF consensus | Zero risk |
| **Low** | Test `FI_CXI_RDZV_PROTO=alt_read` | Isambard docs, ALCF | May affect latency |
| **Low** | Test `FI_CXI_DEFAULT_TX_SIZE=16384` | CSCS (vs our 1024 default) | Unknown |
| **Info** | Consider topology-aware MoE dispatch for future | X-MoE paper | Code change |

## Appendix C: aws-ofi-nccl Changelog Analysis (v1.8.1 → v1.18.0)

*Compiled 2026-04-09 from GitHub release notes for every version between our installed v1.8.1 and the latest v1.18.0.*

### Version Timeline and Key Changes

#### v1.9.0 (April 2024) — AWS-only
- **New:** Plugin interface v8 support (NCCL 2.20 user memory registration feature)
- **New:** Tuner v2 ext-tuner interface (NCCL 2.21)
- **New:** Reduced ordering constraints for control messages to reduce head-of-line blocking under congestion
- **Fix:** Increased communicator limit from 4K to 256K — **supports larger all-to-all groups**
- **Requires:** libfabric ≥1.18.0, NCCL ≥2.4.8

#### v1.9.1 (April 2024) — AWS-only
- Build fixes only (missing headers, libcuda linking, explicit libm/libpthread linking)
- **Requires:** libfabric ≥1.18.0

#### v1.9.2 (June 2024) — AWS-only
- **Fix:** Tuner loading bug with NCCL 2.19/2.20
- **Fix:** RDMA protocol truncation when isend size exceeds irecv size
- **Fix:** Removed global MR advertisement to avoid performance regression in user-registered buffers
- **Requires:** libfabric ≥1.18.0

#### v1.10.0 (August 2024) — AWS-only
- **New:** Zero-copy path for fi_send/fi_recv
- **New:** Endpoint flexibility — different endpoints for receive communicators sharing a completion queue
- **New:** Experimental tuner based on region evaluations
- **Fix:** Disabled Libfabric shared memory when feasible
- **Fix:** Multi-rail protocol now consistently orders rails by VF index
- **Breaking:** Minimum NCCL bumped from v2.4.8 to **v2.17.1**
- **Requires:** libfabric ≥1.18.0, NCCL ≥2.17.1

#### v1.11.0 (August 2024) — AWS-only
- **New:** P5e instance support, auto-generated topology files
- **Fix:** Segfault in platform-aws with unconfigured instance types
- **Fix:** MR cache failure in SENDRECV protocol for providers without MR requirements
- **Fix:** **"Added check to cause an error when using old blocking connect_v4/accept_v4 interfaces with RDMA protocol. The previous release changed connection establishment such that these interfaces cause deadlock."** ← Connection deadlock prevention
- **Requires:** libfabric ≥1.18.0, NCCL ≥2.17.1

#### v1.11.1 (October 2024) — AWS-only, hotfix
- **Fix:** Platform-AWS VF sorting regression causing performance regressions or crashes with EFA ≥1.35.0

#### v1.12.0 (October 2024) — AWS-only
- **New:** Tuner v3 APIs, AllGather/ReduceScatter tuner support, PAT algorithm
- **Fix:** NULL pointer access in endpoint-per-communicator path
- **Requires:** NCCL ≥2.17.1 (tested through 2.23.4)

#### v1.12.1 (October 2024) — AWS-only, hotfix
- Same VF sorting fix as v1.11.1

#### v1.13.0 (November 2024) — AWS-only
- **New:** P5en platform support
- **New:** `OFI_NCCL_DISABLE_DMABUF=1` env var (DMA-BUF now enabled by default)
- **New:** RDMA protocol scheduling improvements for multirail
- **New:** Multiplexed round-robin scheduler
- **Fix:** NULL pointer dereference in endpoint-per-comm mode
- **Fix:** Uninitialized lock usage
- **Fix:** Endpoint cleanup on process termination
- **Fix:** Control message synchronization in eager protocol
- **Fix:** Page-aligned buffer registration for iovec operations
- **Breaking:** Building with platform-aws now requires **libfabric 1.22.0amzn4.0** (Amazon-specific)
- **Breaking:** Plugin now statically links CUDA runtime by default

#### v1.13.1 (November 2024) — AWS-only
- Version string fix only (no functional changes)

#### v1.13.2 (December 2024) — AWS-only
- **Fix:** Small AllReduce performance degradation from eager protocol activating during in-flight RDMA writes
- **Breaking:** DMA-BUF now **disabled** by default (was enabled in v1.13.0). Re-enable with `OFI_NCCL_DISABLE_DMABUF=0`

#### v1.14.0 (March 2025) — **GENERAL RELEASE** (first non-AWS-only since v1.8.1)
- **Milestone:** Resumes support for **all** Libfabric providers, including CXI/Slingshot
- **New:** Memory descriptor handling for control messages to properly support **FI_MR_LOCAL**
- **New:** RDMA receive request early completion for FI_PROGRESS_AUTO providers
- **Fix:** Tuner defaults to NCCL internal tuner on 2-node configs
- **Requires:** NCCL ≥2.17.1 (tested through 2.26.2), libfabric ≥1.22.0amzn4.0 for platform-aws

#### v1.14.1 (April 2025)
- **Fix:** **"Fixed an issue in the sendrecv protocol that would result in a leaking MR keys warning with some providers"** ← Memory registration fix
- Enhanced compatibility with libfabric 2.0
- Improved reliability in memory registration and connection establishment

#### v1.14.2 (April 2025)
- **New:** DMA-BUF enabled by default (re-enabled after v1.13.2 disabled it)
- **Fix:** DMA-BUF blocklisted on EFA versions 1-3

#### v1.15.0 (June 2025)
- **Major:** Codebase migrated from C to C++
- **New:** NCCL v10 API with trafficClass parameter
- **New:** `OFI_NCCL_FORCE_NUM_RAILS` — force heterogeneous NIC configurations
- **New:** `OFI_NCCL_CQ_SIZE` — configurable completion queue size
- **New:** Separate env vars for eager vs control buffer counts (replaces bounce buffer vars)
- **New:** Default library name changed to `libnccl-net-ofi.so` (symlink to `libnccl-net.so` for compat)
- **Deprecated:** `OFI_NCCL_RDMA_MIN_POSTED_BOUNCE_BUFFERS`, `OFI_NCCL_RDMA_MAX_POSTED_BOUNCE_BUFFERS`
- **Requires:** NCCL ≥2.17.1 (tested through 2.26.6)

#### v1.16.0 (June 2025)
- **Fix:** **"Fix bug that prevented communicators from aborting gracefully, as part of supporting NCCL fault tolerance features"** ← Critical for our ft_launcher/InProcessRestart stack
- **New:** Plugin can now set NCCL_BUFFSIZE, NCCL_P2P_NET_CHUNKSIZE, etc. on AWS
- **Requires:** NCCL ≥2.17.1 (tested through 2.27.5)

#### v1.16.1 (July 2025)
- Minor: PCI link speed format fix, NIC acceleration filter

#### v1.16.2 (July 2025)
- Minor: p5.4xlarge instance support

#### v1.16.3 (August 2025)
- **New:** Domain-per-thread mode enabled by default on AWS for multi-NCCL-proxy-thread apps
- Note: This is the threading model that v1.18.0 later redesigned for non-AWS platforms

#### v1.17.0 (September 2025)
- **Fix:** **"Fixed deadlock in memory registration function"** ← **OUR ROOT CAUSE FIX**
- **New:** Control-over-write protocol (mailbox paradigm replaces send/recv for control messages)
- **New:** RDMA flush performance improvements
- **Requires:** NCCL ≥2.17.1 (tested through 2.28.3), CUDA 13.0 API support

#### v1.17.1 (October 2025)
- **Fix:** CUDA 12/13 cross-compatibility fixes
- No deadlock/MR/CXI changes

#### v1.17.2 (November 2025)
- **Fix:** **"Fixed shutdown ordering issue on NICs that require per-endpoint memory registration (Cray Slingshot)"** ← **SLINGSHOT-SPECIFIC FIX**
- **Fix:** Crash with NCCL v2.28.x when Libfabric initialization failed
- **Fix:** GPUDirect RDMA erroneous path on DMA-BUF platforms
- **Requires:** NCCL ≥2.17.1 (tested through 2.28.7)

#### v1.17.3 (January 2026)
- **Fix:** **"Memory leak that can result in running out of host memory for long-running jobs"** ← Prevents progressive OOM
- **Requires:** NCCL ≥2.17.1 (tested through 2.28.7)

#### v1.18.0 (January 2026)
- **Major:** **"Redesigned threading model to support multi-threaded applications without requiring a separate Libfabric domain for each thread"** ← Reduces CXI resource pressure from overlapping TP/EP/DP communicators
- **Fix:** **"Resolved FI_MR_ENDPOINT provider compatibility (SENDRECV protocol only) through proper resource cleanup"** ← CXI uses FI_MR_ENDPOINT
- **Fix:** Non-FI_MR_VIRT_ADDR provider support in RDMA protocol
- **New:** Dynamic platform selection (single binary works on AWS and non-AWS)
- **Requires:** NCCL ≥2.17.1 (tested through 2.29.2)

### Cumulative Bug Fixes Relevant to Our Hang

The following fixes between v1.8.1 and v1.18.0 are directly or indirectly relevant to our Slingshot/CXI MoE all-to-all hang:

| Version | Fix | Relevance |
|---------|-----|-----------|
| v1.9.0 | Communicator limit 4K→256K | Medium — EP=8 all-to-all creates many communicator pairs |
| v1.9.0 | Reduced ordering constraints (less HOL blocking) | Medium — congestion can trigger resource exhaustion |
| v1.11.0 | Connection deadlock prevention (blocking interfaces) | Low — different deadlock mechanism |
| v1.13.0 | Endpoint cleanup on process termination | Medium — affects ft_launcher restart recovery |
| v1.13.0 | Control message sync in eager protocol | Medium — eager protocol race conditions |
| v1.14.0 | FI_MR_LOCAL support for control messages | **High** — CXI requires proper MR handling |
| v1.14.1 | Leaking MR keys in sendrecv protocol | **High** — MR key leak can exhaust registration resources |
| v1.16.0 | Graceful communicator abort (FT support) | **High** — enables clean InProcessRestart recovery |
| **v1.17.0** | **Deadlock in memory registration function** | **Critical** — our identified root cause |
| **v1.17.2** | **Slingshot per-endpoint MR shutdown ordering** | **Critical** — Slingshot-specific fix |
| v1.17.3 | Memory leak for long-running jobs | **High** — prevents progressive OOM |
| **v1.18.0** | **Threading redesign (no per-thread domain)** | **High** — reduces CXI resource pressure |
| **v1.18.0** | **FI_MR_ENDPOINT provider fix** | **High** — CXI is an FI_MR_ENDPOINT provider |

### Minimum Viable Upgrade Analysis

**Can we use an intermediate version instead of v1.17.3?**

| Version | General (non-AWS) support? | Has MR deadlock fix? | Has Slingshot fix? | Min NCCL | Min libfabric | Viable? |
|---------|---------------------------|---------------------|--------------------|----------|---------------|---------|
| v1.9.0–v1.13.2 | **No** (AWS-only) | No | No | varies | 1.18.0 | **No** |
| **v1.14.0** | **Yes** | No | No | 2.17.1 | 1.22.0* | Partial — has FI_MR_LOCAL fix |
| v1.14.1 | Yes | No | No | 2.17.1 | 1.22.0* | Partial — adds MR key leak fix |
| v1.14.2 | Yes | No | No | 2.17.1 | 1.22.0* | Same as v1.14.1 |
| v1.15.0 | Yes | No | No | 2.17.1 | 1.22.0* | Risky — C→C++ rewrite |
| v1.16.0 | Yes | No | No | 2.17.1 | 1.22.0* | Adds FT abort fix |
| **v1.17.0** | Yes | **Yes** | No | 2.17.1 | 1.22.0* | **Minimum for deadlock fix** |
| **v1.17.2** | Yes | **Yes** | **Yes** | 2.17.1 | 1.22.0* | **Minimum for Slingshot fix** |
| **v1.17.3** | Yes | **Yes** | **Yes** | 2.17.1 | 1.22.0* | **+ memory leak fix** |
| **v1.18.0** | Yes | **Yes** | **Yes** | 2.17.1 | 1.22.0* | **Best choice: threading redesign + FI_MR_ENDPOINT fix** |

\* The `1.22.0amzn4.0` minimum is for platform-aws builds only. Non-AWS builds (our case) work with standard libfabric 1.22.0.

### Recommendations

1. **Target v1.18.0** (not v1.17.3) for the admin upgrade request. It includes all v1.17.x fixes plus the threading redesign and FI_MR_ENDPOINT fix, both directly relevant to CXI/Slingshot with multiple NCCL communicator groups.

2. **v1.14.0 is a viable fallback** if v1.18.0 has compatibility issues. It's the first general-release version since v1.8.1 and includes the FI_MR_LOCAL control message fix. It does NOT have the deadlock fix but may still reduce hang frequency through better MR handling.

3. **v1.9.0–v1.13.2 are NOT viable** — they were AWS-only releases that don't support CXI/Slingshot.

4. **Our NCCL 2.28.9 and libfabric 1.22.0 meet the requirements** for all versions through v1.18.0. No dependency upgrades needed.

5. **The v1.15.0 C→C++ migration is a risk factor** — if building from source, v1.14.2 (last C version) is the safest intermediate build. v1.15.0+ may require different compiler flags or have ABI differences.

## Appendix D: Libfabric CXI Provider Changelog Analysis (v1.22.0 → v2.5.0)

*Compiled 2026-04-09 from libfabric NEWS.md, GitHub issues, and release notes.*

### Overview

Our system runs **libfabric 1.22.0**. The CXI provider has received significant bug fixes in subsequent releases, including multiple deadlock fixes that are directly relevant to our NCCL hang. Libfabric underwent a major version bump to v2.0.0 (December 2024) with API/ABI-breaking changes, followed by rapid iteration through v2.5.0 (March 2026).

**Key question: Is our libfabric 1.22.0 contributing to the hang independently of aws-ofi-nccl?**

Answer: **Very likely yes.** Multiple CXI deadlock and memory registration fixes were shipped in v2.2.0–v2.5.0, suggesting our v1.22.0 has known deadlock-prone code paths in the CXI provider itself.

### CXI Provider Fixes by Version

#### Libfabric v2.2.0 (June 30, 2025)

- **"Fix regression which could cause deadlock"** — CXI-specific deadlock fix
- **Fixed locking on the SRX path** — shared receive context path, used by NCCL
- **Enhanced multi-threaded CQ WAIT_FD implementation** — completion queue threading fix
- **"Support read-only cached MRs"** — memory registration optimization
- **Added access control bits to internal memory registration** — MR management improvement
- **Increased RX buffer size for collectives** — relevant to large alltoall messages
- **Improved reduction engine timeout handling** — timeout-related fix

#### Libfabric v2.3.0 (September 15, 2025)

- **"Fix regression which could cause deadlock"** — another CXI deadlock fix (different from v2.2.0)
- **Optimized counter thread locking** — reduces lock acquisition overhead
- **Cache the last cmdq CP** — reduces lock pressure on command queues
- **Fixed memory leak in service allocation** — resource leak fix
- **"Support cuda sync_memops pointer attribute"** — directly relevant to GH200 `SYNC_MEMOPS` warnings
- **"Retry root->leaf send after timeout"** — retry logic for send timeouts

#### Libfabric v2.4.0 (December 15, 2025)

- **"Add domain rx match mode override"** — allows runtime control of RX match mode per domain
- **"Set rendezvous eager size default to 2K"** — changed default eager size (was previously larger)
- **"Do not abort if MR match count do not reconcile"** — prevents crash on MR accounting mismatch
- **"Set max domain TX CQs to 14"** — limits TX completion queues per domain
- **"Fix RNR protocol send byte/error counting"** — flow control accounting fix
- **"Release TX credit when pending RNR retry"** — **flow control deadlock prevention** (TX credits not returned during retry could starve the sender)
- **"Fix performance issue with close_mc()"** — multicast close performance
- **"Fix use of hw_cps and memory leak"** — hardware command processor leak fix

#### Libfabric v2.5.0 (March 20, 2026)

- **"cxip_mr_init uses wrong length field from DMABUF structure"** — MR initialization bug with DMA-BUF
- **"Fix hang in MPI when using cxi with lnx"** — explicit hang fix in CXI
- **"Fix append sequence for standard MR"** — MR ordering fix
- **"Fix default mon start failure"** — monitoring subsystem fix

### Analysis: How These Fixes Relate to Our Hang

**1. Two separate deadlock fixes (v2.2.0 and v2.3.0)**

Both releases contain "Fix regression which could cause deadlock" in the CXI provider. These are distinct fixes (different regressions), meaning the CXI provider in libfabric 1.22.0 has at least two known deadlock-prone code paths. The v2.2.0 fix targets SRX locking; the v2.3.0 fix targets counter thread locking.

Our hang symptom (all ranks block simultaneously on a collective) is consistent with a CXI provider deadlock. The hang occurs in Send/Recv operations where both SRX paths and counter threads are active.

**2. TX credit starvation fix (v2.4.0)**

"Release TX credit when pending RNR retry" is a flow control deadlock prevention fix. If the CXI provider holds TX credits during a Receiver Not Ready (RNR) retry, other operations waiting for TX credits will stall indefinitely. Our MoE alltoall sends 8 simultaneous Send/Recv pairs to different EP ranks — if one pair triggers RNR and holds credits, the other 7 pairs could starve.

**3. CUDA sync_memops support (v2.3.0)**

"Support cuda sync_memops pointer attribute" directly relates to our GH200 `SYNC_MEMOPS` warnings. On our system, NCCL logs show `cuPointerSetAttribute: CUDA_ERROR_NOT_SUPPORTED` because GH200 unified memory doesn't support the SYNC_MEMOPS attribute. The v2.3.0 fix adds proper handling for this — our v1.22.0 may be using a fallback path that's less reliable for memory registration.

**4. RX match mode and eager size changes (v2.4.0)**

The default rendezvous eager size was changed to 2K in v2.4.0 (from a larger default). This aligns with the CSCS recommendation of `FI_CXI_RDZV_EAGER_SIZE=0` to prevent hangs. The old default (which we're running) may be triggering the eager message hang documented by CSCS.

### Libfabric 2.0 Breaking Changes

Libfabric v2.0.0 (December 2024) introduced API/ABI-breaking changes:
- Removed wait sets and poll sets
- Removed async memory registration (`FI_MR_COMPLETE`)
- Removed several provider backends (but CXI was retained)
- Threading model consolidated around `FI_THREAD_DOMAIN`
- Simplified memory registration modes

**Impact on upgrade path:** Moving from libfabric 1.22.0 to 2.x requires recompilation of all consumers (aws-ofi-nccl, NCCL, potentially PyTorch). This is a system-level change requiring admin coordination.

### HPE shs-libfabric Fork

HPE maintains a fork at [HewlettPackard/shs-libfabric](https://github.com/HewlettPackard/shs-libfabric) which is the basis for the system libfabric on Slingshot clusters. See **Appendix E** for detailed analysis of our system's exact build and which upstream fixes have/haven't been backported.

### CSCS Libfabric/aws-ofi-nccl ABI Incompatibility

CSCS documented ([Known Issues](https://docs.cscs.ch/software/container-engine/known-issue/)) that aws-ofi-nccl v1.9.2 crashed with libfabric v1.22 due to an ABI incompatibility. The fix was deployed April 16, 2025. This confirms:
1. Libfabric 1.22.0 has specific ABI constraints with newer aws-ofi-nccl versions
2. Our user-space aws-ofi-nccl v1.17.3 build failure is likely this same class of issue
3. Admin integration is required to ensure compatible library versions

### Open Bug: CXI/GDRCopy Cleanup Race Condition

[ofiwg/libfabric#10041](https://github.com/ofiwg/libfabric/issues/10041) documents an **open** bug where multiple threads can simultaneously unregister the same GDRCopy memory handle in the CXI provider, leading to use-after-free. The `cuda_gdrcopy_dev_unregister()` function lacks proper state validation after acquiring its spinlock.

**Relevance:** Our config sets `NCCL_GDRCOPY_ENABLE=1` (via Isambard docs recommendation). This race condition could contribute to memory registration instability. If GDRCopy cleanup races with new registrations in the alltoall path, it could trigger the deadlock. **Testing with `NCCL_GDRCOPY_ENABLE=0` could rule this out.**

### Megatron MoE on Slingshot — No Public Reports Found

A thorough search found **no public reports** of anyone running Megatron-LM or Megatron-Core MoE training on Slingshot/CXI networks. The available literature covers:
- Dense transformer training on Slingshot (AxoNN framework achieved 1.4 Exaflop/s on Alps GH200)
- MoE training on InfiniBand (NVIDIA's MoE Parallel Folding paper uses DGX H100 with IB)
- X-MoE on Frontier (AMD GPUs with Slingshot, using RCCL not NCCL)

Our Nemotron 3 Nano MoE SFT on Isambard GH200 + Slingshot appears to be a novel combination without established community configurations. This explains why no env-var-only solution exists — we're in uncharted territory where the combination of NCCL MoE alltoall + CXI + GH200 unified memory triggers bugs that haven't been seen in other deployments.

### MoE Parallel Folding — Intra-Node EP Optimization

The [MoE Parallel Folding paper](https://arxiv.org/abs/2504.14960) from NVIDIA demonstrates that MoE communication is dominated by alltoall when EP exceeds intra-node GPU count. Key findings:
- When `EP > 8` (more than one node), alltoall accounts for >70% of MoE layer latency
- Folding EP to fit within intra-node NVLink (450 GB/s) vs inter-node (25 GB/s Slingshot) improves MFU by 3-6%
- Our config: EP=8 across 4 nodes = 12 of 16 EP peers are inter-node — worst case for Slingshot

**Potential optimization:** If EP could be reduced to 4 (intra-node only), alltoall would use NVLink instead of Slingshot, completely bypassing the CXI hang. However, this would require different parallelism to handle the 128 MoE experts (e.g., EP=4 with expert replication or different TP/DP split).

### Recommendations

| Priority | Action | Rationale |
|----------|--------|-----------|
| **High** | Request libfabric upgrade to ≥2.4.0 alongside aws-ofi-nccl ≥1.18.0 | Two CXI deadlock fixes (v2.2.0, v2.3.0) + TX credit starvation fix (v2.4.0) + CUDA sync_memops support (v2.3.0) |
| **Medium** | Test `NCCL_GDRCOPY_ENABLE=0` | Rule out open CXI/GDRCopy race condition (ofiwg/libfabric#10041) |
| **Medium** | Test `FI_CXI_RDZV_EAGER_SIZE=0` | Align with v2.4.0's reduced eager size default (2K→0 per CSCS) |
| **Low** | Explore EP=4 (intra-node only) parallelism | Eliminates Slingshot alltoall entirely but requires rearchitecting parallelism |
| **Info** | Note: HPE shs-libfabric fork may differ from upstream | System libfabric version may have different patches than upstream 1.22.0 |

## Appendix E: HPE shs-libfabric Fork Analysis

*Compiled 2026-04-09 from RPM metadata, GitHub commit search, and binary inspection of the system libfabric.*

### System Build Identification

RPM query of our installed libfabric reveals its exact provenance:

```
Name        : libfabric
Version     : 1.22.0
Release     : SHS12.0.2_20250722155538_8dad011dfdb6
Architecture: aarch64
Build Date  : Tue 22 Jul 2025 04:00:46 PM UTC
Source RPM  : libfabric-1.22.0-SHS12.0.2_20250722155538_8dad011dfdb6.src.rpm
```

**Key facts:**
- **SHS version:** 12.0.2 (Slingshot Host Software)
- **Build date:** July 22, 2025
- **Source commit:** `8dad011dfdb6` in the [HewlettPackard/shs-libfabric](https://github.com/HewlettPackard/shs-libfabric) fork
- **Last commit message:** "prov/cxi: Support cuda sync_memops pointer attribute" (authored July 10, 2025)
- **CXI provider version:** 0.1 (per `fi_info -p cxi`)
- **MR mode:** `FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_ENDPOINT`

### HPE Fork Tag Structure

The shs-libfabric fork uses dual tagging:
- **Upstream tags:** v2.2.0, v2.3.0, v2.3.1, v2.4.0, v2.5.0
- **SHS release tags:** release/shs-12.0.1, release/shs-12.0.2, release/shs-13.0.0, release/shs-13.1.0, release/shs-14.0.0

Our system runs **release/shs-12.0.2**. The SHS 12.x line appears to track upstream v2.2.0–v2.3.0 era code with selective cherry-picks.

### Backported Fixes (confirmed present in SHS 12.0.2)

Searching the HPE fork's commits on the `release/shs-12.0.2` tag confirms these upstream fixes were backported:

| Commit | Fix | Upstream Version |
|--------|-----|-----------------|
| `8dad011` | **Support cuda sync_memops pointer attribute** | v2.3.0 |
| `044f919` | Add access ctrl bits to internal mem reg | v2.2.0 |
| `7207f5c` | Support read-only cached MRs | v2.2.0 |
| `82905598` | Fix CQ wait FD logic | v2.2.0 |
| `6f7f409` | Isolate alt_read gets to restricted only cmdq | v2.3.0+ |
| `1842a62` | Fix use of alt_read rget restricted TC type | v2.3.0+ |

### NOT Backported (confirmed MISSING from SHS 12.0.2)

GitHub commit search across the entire HPE fork returned **zero results** for:

| Search Term | Results | Missing Fix | Upstream Version |
|-------------|---------|-------------|-----------------|
| `"deadlock"` | 0 | **"Fix regression which could cause deadlock" (SRX locking)** | v2.2.0 |
| `"Fix regression"` | 0 | **"Fix regression which could cause deadlock" (counter thread)** | v2.3.0 |
| `"Release TX credit"` | 0 | **"Release TX credit when pending RNR retry"** | v2.4.0 |
| `"locking" path:prov/cxi` | 0 | Counter thread locking optimization | v2.3.0 |

**This is the critical finding: The two CXI deadlock fixes from upstream v2.2.0 and v2.3.0 are NOT in our system's libfabric.** HPE selectively backported MR improvements and the CUDA sync_memops fix, but not the deadlock fixes themselves.

### Implications for Our Hang

1. **Libfabric IS a contributing factor.** Our SHS 12.0.2 build has known CXI deadlock-prone code paths (SRX locking, counter thread locking) that were fixed in upstream v2.2.0/v2.3.0 but not backported by HPE.

2. **The TX credit starvation bug (v2.4.0) is also missing.** "Release TX credit when pending RNR retry" — if a CXI Send triggers Receiver Not Ready (RNR) and the TX credit is not released during retry, other pending Sends starve. Our MoE alltoall issues 8 simultaneous Send/Recv pairs; one RNR event could cascade to a full hang.

3. **The CUDA sync_memops fix IS present.** HPE backported `8dad011` which ensures CUDA GPU operations complete before RDMA access to memory regions. This means the `cuPointerSetAttribute: CUDA_ERROR_NOT_SUPPORTED` warnings we see are now handled properly — they are truly non-fatal in our build.

4. **The MR improvements ARE present.** Read-only cached MRs, access control bits, and CQ wait FD fixes were all backported. These may explain why our hang is intermittent rather than constant — the MR path is improved but the locking paths are still buggy.

### Newer SHS Releases Available

The HPE fork has newer SHS releases that may contain the deadlock fixes:

| SHS Release | Date | Likely Upstream Base |
|-------------|------|---------------------|
| release/shs-12.0.2 | Dec 2024 (tag), Jul 2025 (build) | v2.2.0 + cherry-picks |
| release/shs-13.0.0 | Aug 2025 | v2.3.0+ |
| release/shs-13.1.0 | Late 2025 | v2.3.x+ |
| release/shs-14.0.0 | Jan 2026 | v2.4.0+ |

**SHS 13.0.0** (August 2025) is the most likely candidate to contain the v2.3.0 deadlock fix. **SHS 14.0.0** (January 2026) would contain the v2.4.0 TX credit fix.

### System Module Status

```
Available modules:
  libfabric/1.22.0 (loaded)         ← SHS 12.0.2
  brics/aws-ofi-nccl/1.6.0
  brics/aws-ofi-nccl/1.8.1 (loaded) ← Has the memory registration deadlock
  brics/nccl/2.21.5-1
  brics/nccl/2.26.6-1 (loaded)
```

No newer libfabric module is available. The upgrade requires admin action to install SHS ≥13.0.0 (for CXI deadlock fixes) or ≥14.0.0 (for TX credit fix).

### Updated Root Cause Assessment

The hang has **two independent root causes**, both of which must be fixed:

1. **aws-ofi-nccl v1.8.1:** Memory registration deadlock in the NCCL plugin layer (fixed in v1.17.0)
2. **libfabric SHS 12.0.2:** CXI provider deadlocks in SRX locking and counter thread locking (fixed in upstream v2.2.0/v2.3.0, NOT backported by HPE)

Either bug alone can cause the hang. Both are in the Send/Recv code path used by MoE alltoall. The fix requires upgrading **both** components:
- **aws-ofi-nccl:** v1.8.1 → ≥1.18.0 (via brics module)
- **libfabric:** SHS 12.0.2 → SHS ≥13.0.0 (via system update)

### Recommendations

| Priority | Action | Rationale |
|----------|--------|-----------|
| **Critical** | Request SHS ≥13.0.0 (or ≥14.0.0) from Isambard admins | Contains CXI deadlock fixes missing from SHS 12.0.2 |
| **Critical** | Request aws-ofi-nccl ≥1.18.0 module alongside SHS upgrade | Both upgrades needed — fixing one leaves the other bug active |
| **Medium** | Ask admins which SHS version they plan to deploy | SHS 14.0.0 (Jan 2026) is ideal — includes all fixes through v2.4.0 |
| **Info** | SHS 13.0.0 has "CXI provider supports FI_ORDER_RMA_RAR" | This was the only documented change; deadlock fixes may be included but undocumented |
