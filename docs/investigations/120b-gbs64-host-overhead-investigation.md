# Super-120B @ GBS=64: the ≤20 s/iter investigation (host-overhead wall)

**Status:** in progress. **Owner at time of writing:** mcore-bump session, 2026-07-27.
**Goal (Kyle, 2026-07-27):** ≤20 s/iter on the 120B quickstart at **GBS=64**, seq 32K,
16 nodes / 64 GPUs. Later refined: **quality-neutral only** — no token-dropping.

Everything below was measured on tunnel `5738451` (16 nodes), at the mcore 0.19 pin
(`feat/mcore-bump-6cd6ea530`). Read §1 first; it is the finding that reframes the problem.

---

## 1. The diagnosis: this config is HOST-bound, not fabric- or bubble-bound

Torch-profiler traces of iteration 10, rank 9 (interval-union arithmetic over GPU kernels):

| metric | non-VPP champion | VPP=4 + overlap |
|---|---:|---:|
| window | 31.07 s | 34.01 s |
| compute (union) | **9.240 s** | **9.262 s** |
| NCCL total | 6.331 s | 6.474 s |
| exposed NCCL | 6.324 s | 6.469 s |
| **overlapped NCCL** | **0.007 s** | **0.005 s** |
| idle (no kernel at all) | 15.50 s (49.9%) | 18.28 s (53.8%) |

**The GPU is idle ~50% of every iteration.** Scaled to the unprofiled 27.12 s, busy ≈ 13.6 s.
So the ≤20 s target is comfortably inside what the GPU work actually requires — the entire
problem is recovering idle.

### The idle is NOT a pipeline bubble — check the gap distribution, not the total

243,866 idle gaps on the champion, iteration 10, rank 9:

| gap size | count | total | % of idle |
|---|---:|---:|---:|
| < 50 µs | 177,044 | 1.41 s | 9.1% |
| 50–100 µs | 32,019 | 2.46 s | 15.9% |
| 100–500 µs | 31,875 | 4.66 s | 30.1% |
| 500 µs–1 ms | 602 | 0.37 s | 2.4% |
| **1–10 ms** | **2,320** | **6.47 s** | **41.8%** |
| > 10 ms | 6 | 0.12 s | 0.8% |

Largest single gap in the whole iteration: **44.6 ms**. A PP=8 bubble at 32 microbatches would
appear as a handful of gaps ~one microbatch long (~800 ms). There are none. **Reducing the
bubble therefore has nothing to reclaim — which is why VPP loses.**

### What the host is doing during those gaps

Attribution of the 1–10 ms band against CPU-side trace events: **6.40 s of 6.47 s is covered by
busy CPU events** (i.e. the host is working, not waiting on a peer).

| overlap | event |
|---:|---|
| 47.6 s* | `redispatch_boxed` — torch dispatcher machinery |
| 4.21 s | `CheckpointFunctionBackward` / `megatron/core/tensor_parallel/random.py:598 backward` |
| 4.99 s | `run_backward` |
| 1.68 s | `Optimizer.step#AdamW.step` |

*nested frames, so they overlap-count.

Two consumers dominate: **eager dispatch overhead** across 201,684 GEMM launches/iteration,
and **activation recomputation's host path** (Python re-entry + RNG-state restore).

> Note for whoever continues: `CheckpointFunctionBackward` has **two** possible sources —
> `recompute_modules`, and the launcher's `ISAMBARD_FP32_SSM_STATE=checkpoint` patch which also
> wraps the Mamba scan in a checkpoint. They were not separated. If removing recompute yields
> less than the attribution predicts, the fp32-SSM checkpoint is the remaining half.
> **Do not disable fp32-SSM casually:** CLAUDE.md records deterministic bf16 NaN on certain
> ~32K single-document sequences. Gate any such arm on NaN-watching, not throughput alone.

---

## 2. Results table (GBS=64, 16 nodes, mean of iters 6–12 unless noted)

| config | s/iter | TFLOP/s | loss | quality-neutral? |
|---|---:|---:|---:|---|
| CF 1.0 + pad | **20.21** | 130.1 | 0.7356 | ❌ drops tokens |
| CF 1.25 + pad | 21.63 | 121.5 | 0.7283 | ❌ |
| CF 2.0 + pad | 24.84 | 105.8 | 0.7172 | ❌ |
| **dropless + `moe_router_fusion`** | **26.82** | 98.0 | 0.7061 | ✅ **best neutral** |
| dropless champion (baseline) | 27.12 | 96.9 | 0.7060 | ✅ |
| `CUDA_DEVICE_MAX_CONNECTIONS=32` | 27.07 | 97.1 | 0.7063 | ✅ (no gain) |
| `CUDA_DEVICE_MAX_CONNECTIONS=8` | 27.18 | 96.7 | 0.7062 | ✅ (no gain) |
| VPP=4 (batched p2p) | 29.54 | 89.0 | 0.7060 | ✅ (worse) |
| VPP=4 + `moe_router_fusion` | 30.13 | 87.2 | 0.7061 | ✅ (worse) |
| VPP=4 + `overlap_p2p_comm` | 30.51 | 86.1 | 0.7061 | ✅ (worse) |
| CF 4.0 + pad | 33.50 | 78.5 | 0.7084 | ❌ (and slower than dropless) |

Reference: 48-iteration dropless run (bump ladder R8) = **27.17 s**, lm loss **0.6242014** @ iter 48.

**All loss values except the CF arms sit within ~2.4e-4 of each other**, so the CF loss deltas
are two orders of magnitude outside the noise band.

---

## 3. Levers CLOSED, with the reason (do not re-run these blind)

| Lever | Verdict | Evidence |
|---|---|---|
| VPP (any variant) | +9–12% worse | bubble isn't the bottleneck (§1); replicates old-pin 29.59 → 29.54 |
| `overlap_p2p_comm` | inert | overlaps **5 ms of 6.47 s** NCCL |
| `CUDA_DEVICE_MAX_CONNECTIONS` >1 | ~1% worse | exposed NCCL is peer-*wait*, not hideable transfer; concurrency adds SM contention |
| micro_batch_size > 1 | rejected | `ValueError: Micro batch size should be 1 when training with packed sequence` |
| TP reduction | n/a | already TP=1 |
| `fine_grained_activation_offloading` | **broken** | upstream `Chunk mismatch` in `fine_grained_activation_offload.py`; fires under **plain PP=8**, not just VPP |
| CUDA graphs — `attn` scope | blocked | TE `context_parallel.py:1544` records CUDA events → `cudaErrorCapturedEvent`. CP=4 is mandatory for memory |
| CUDA graphs — `moe_router`/`moe_preprocess` | blocked | `alltoall` dispatcher needs host-side token counts; capture feeds garbage sizes to `mappings.py:444` (deterministic, same value each run) |
| CUDA graphs — `moe` scope | blocked | requires drop-and-pad; OOMs |
| CUDA graphs — `mamba` scope | blocked | OOMs even with the MoE transient bounded by CF |
| MoE paper §4.3.7 sync-free dropless | unavailable | needs **CUDA 13.1** (image ships 13.0) **and** HybridEP/InfiniBand (this is Slingshot/CXI) |
| mcore pin bump (again) | won't help | upstream main only 31 commits ahead; **neither** open bug is fixed there (checked via API per-file) |

### Why the capacity factor is fast (mechanism, for the record)

`capacity = CF × tokens × topk / num_experts`. Here: 8192 tok/rank × topk 22 / 512 experts ≈ **352**
tokens/expert at CF 1.0. Padding to capacity gives (a) uniform expert GEMMs instead of ragged, and
(b) **no device→host sync** for per-expert counts. At identical 1× FLOPs, uniform = 20.21 s vs
ragged dropless = 26.82 s — so **raggedness + host sync costs ~6.6 s/iter**. Total expert compute
scales linearly with CF, which is why CF 4.0 (near-zero drops) is *slower* than dropless.

---

## 4. OPEN — the current line of work (quality-neutral, spend idle HBM to delete host work)

Peak memory is **73.3 GB of 95 GB → ~21.7 GB has been idle the whole time.** The config contains
settings that spend *time* to save memory we do not need to save. All arms below are **dropless**.

- [ ] **K1** `optimizer_offload_fraction: 1.0 → 0.0` — Adam on GPU, not host. We run **DP=2**;
      full offload is a DP=1 need. Quickstart header already records 0.5 → ~1.2 s better.
- [ ] **K2** `recompute_modules: ["moe","shared_experts"] → ["moe"]` — targets the 4.21 s
      `CheckpointFunctionBackward` stall. **Keep `moe`**: dropping it OOMs in the MoE dispatch transient.
- [ ] **K3** K1 + K2 combined.
- [ ] **K4** K3 + `bias_activation_fusion: true` + `apply_rope_fusion: true` (both were simply off).
- [ ] **J1** 48-iteration dropless baseline (in flight) — the reference for all K arms.

### Next candidates if K falls short
- [ ] Separate the two `CheckpointFunctionBackward` consumers (recompute vs fp32-SSM checkpoint).
- [ ] `cross_entropy_fusion_impl: native → te` (upstream warns of stability issues — measure, don't assume).
- [ ] `manual_gc_interval` tuning (currently 10).
- [ ] `overlap_param_gather: false → true` — memory note says false for Nemotron-H DP>1; revisit only with evidence.

---

## 5. Bugs found (all real; 3 in our code, 2 upstream)

| # | Where | Status |
|---|---|---|
| 1 | `training/train.py` gated the full-iteration CUDA graph on the **deprecated** `cuda_graph_scope`, so any config setting `cuda_graph_impl` died with `TypeError: argument of type 'NoneType' is not iterable`. Nemotron-H requires `cuda_graph_impl="local"` (`ssm/mamba_layer.py:103` asserts it) — exactly the value that crashed. | **FIXED** — commit `8ed1f954`, `_use_full_iteration_cuda_graph()` + 4 tests (verified red-then-green) |
| 2 | `recipes/nemotronh/nemotron_3_{super,ultra}.py` set `cfg.model.cuda_graph_scope = []`. `[]` is not `None`, so mcore refuses the modern `cuda_graph_modules` ("cannot be set together"). Blocks scoped capture entirely. | **OPEN** — worked around in-config with `cuda_graph_scope: null` |
| 3 | `training/eval.py:122,196` carry the same `CudaGraphScope.full_iteration in ...cuda_graph_scope` membership test that crashed `train.py`. Will fire if #2 is fixed by setting the field to `None`. | **OPEN** — fix together with #2 |
| 4 | mcore `transformer/cuda_graphs.py:181` `ArgMetadata.zeros_like()` — `*self.shape` unpacks to nothing for a **0-dim tensor**, so `torch.zeros()` gets no size. Blocks partial CUDA graphs for any model passing a scalar tensor into a graphed region. | **PATCHED** — `3rdparty/patches/megatron-lm/0002-fix-cuda-graph-zeros_like-0dim-tensor.patch`; still open upstream; durable home is a carried commit on the GeodesicResearch fork |
| 5 | mcore `pipeline_parallel/fine_grained_activation_offload.py` — `AssertionError: Chunk mismatch` under **plain PP** (not only VPP), despite upstream docs listing interleaved-PP as supported. | **OPEN upstream** — no commits to that file since our pin |

---

## 5a. ⚠️ CURRENT WORKING-TREE STATE — read before running anything

`3rdparty/Megatron-LM` is **dirty**: bug #4's one-line fix is applied to the submodule working
tree, not committed. `git status` shows only a bare ` M 3rdparty/Megatron-LM`, so it is easy to
miss — and any run from this checkout silently uses patched mcore. This is the exact hazard
CLAUDE.md calls out ("never edit the submodule working tree in place").

Resolve it one of two ways before trusting any further number:

```bash
# (a) discard the patch and accept that partial CUDA graphs crash on 0-dim tensors
git -C 3rdparty/Megatron-LM checkout megatron/core/transformer/cuda_graphs.py

# (b) durable: commit it on the GeodesicResearch fork as a second carried commit,
#     push, then bump the gitlink + .main.commit (same route as the nvrx probe fix)
```

The vendored copy at `3rdparty/patches/megatron-lm/0002-*.patch` is the record either way.

## 5b. Container image ceiling: the host driver caps how new an image we can run

**The cluster's host driver is `565.57.01` (CUDA 12.7).** NGC images run newer in-image CUDA via
forward-compat libs, but only if the image's bundled compat `libcuda` is from a branch the host
kernel module supports. Measured:

| image | CUDA | bundled compat libcuda | runs here? |
|---|---|---|---|
| **26.02.nemotron_3_super** (current) | 13.0 | **580.95.05** | ✅ |
| **26.04** | **13.1** | **590.48.01** | ✅ **VERIFIED** — `is_available: True`, GPU op OK, TE **2.14.1**, torch 2.11 |
| **26.06** | 13.2 | **595.58.03** | ❌ `torch.cuda.is_available() == False` |

**Conclusion: `nemo:26.04` is the newest image this cluster can run.** The forward-compat ceiling
on driver `565.57.01` lies between the 590 branch (works) and 595 (does not). 26.04 is also the
version that matters: **CUDA 13.1** is where device-side grouped-GEMM shapes landed (13.2 adds
mostly Blackwell MXFP8, irrelevant on GH200) and **TE 2.14** carries the on-device group sizes and
the CPU-overhead work. Newer images require a BriCS driver update — worth raising with them, as it
caps the whole machine, not just this repo.

**SIF is pulled and ready:** `/projects/a5k/public/containers/nemo_26.04.sif` (18 G).

26.06 otherwise qualified fine — Slingshot NCCL built (v2.29.2-1 / hwloc v2.13 / aws-ofi-nccl
v1.18.0), overlay installed, **16/19 validator checks passed including `import paths (repo wins
over image)` and `NCCL CXI net plugin loads`**. Only the CUDA-dependent checks fail, plus a
missing `grouped_gemm` module. Forcing `/usr/local/cuda/compat/lib.real` onto `LD_LIBRARY_PATH`
does not help. **This is a driver constraint, not a config bug** — raising it with BriCS is the
only route to newer images.

### Why we wanted a newer image (from the TE + CUDA changelogs)

Not currency — these releases contain the exact mechanisms for our host-overhead wall:

| version | change | relevance |
|---|---|---|
| **CUDA 13.1** | cuBLASLt grouped GEMM with **device-side shapes** ("matrices passed as a device array of pointers… each with its own shapes") | MoE-paper §4.3.7 Challenge 1: removes the dropless host sync **without dropping tokens** |
| **TE 2.14** | "BF16 and MXFP8 grouped GEMM support with **on-device group sizes**" | the TE binding for the above |
| **TE 2.14** | "multiple **CPU overhead optimizations**… reduce per-step Python/host overhead"; single-parameter `GroupedLinear` "reduces CPU overheads" | directly targets our measured bottleneck |
| **TE 2.16** | "Reduced the **CPU overhead in the GroupedLinear** module" (×3 PRs) | ditto |
| **TE 2.16** | "**CUDA Graph capture support for GroupedLinear and grouped MoE** operations" | may reopen the `moe` graph scope |
| **TE 2.13** | `get_backward_dw_params` "fixing weight gradient hook management when using **wgrad CUDA Graphs with Megatron-LM**" | graph capture fix |
| **TE 2.15** | "Fixed a **numerical bug for the MoE fused router for large top-K and expert counts**" | ⚠️ we run **top-k 22 over 512 experts** on TE 2.12 — see below |

> ⚠️ **`moe_router_fusion` caution.** It measured ~0.5–1% faster and loss-neutral to 9.6e-5 at 48
> iters, but TE 2.15's fixed router bug names exactly our regime (large top-k, large expert
> count) and our TE 2.12 predates the fix. Do not adopt it as a default until we are past TE 2.15.

**Note the ncclep/CUDA-graph path is gated separately and higher:** `moe_flex_dispatcher_backend:
ncclep` (the graph-capturable dispatcher, and the fix for the `mappings.py:444` garbage-token-count
failure) needs `transformer_engine.pytorch.ep`, which exists only in **TE 2.17** — newer than any
NGC image currently published. The mcore side is already present in our 0.19 (`paged_stash.py`,
10 `ncclep` references in `token_dispatcher.py`).

## 6. Artifacts

- **Traces (durable):** `/projects/a5k/public/profiles/r9_nonvpp/20260727T053530-j5738451/`
  and `/projects/a5k/public/profiles/r9_vpp4_overlap/20260727T051132-j5738451/`
  (rank 0 + rank 9, iterations 10 and 20, with `config_snapshot.yaml`,
  `resolved_config_snapshot.yaml`, `provenance.txt`, raw-log copy).
- **Analysis scripts:** `trace_analysis/{parse_trace,analyze_full}.py` (interval-union),
  plus `gap_dist.py` (idle-gap histogram) and `gap_attrib.py` (host attribution of gaps)
  written for this investigation. **These live in the session tmp dir and are NOT durable —
  copy them into the repo if this work continues.**
- **W&B:** project `megatron_training`, run names match the arm labels (`C4_novpp_routerfus`,
  `E1_cf10`, `F1_cf125`, `K1_nooptoffload`, …).
- **Probe configs / raw logs:** session tmp `…/tmp/mcore_bump/*.yaml`, `*.out` — **ephemeral.**

## 7. Method notes (things that cost time)

- **Probe length:** exploratory arms used `train_iters: 14`, scored as mean of iters 6–12 (n=7).
  Kyle directed **48 iterations** for all probes thereafter, matching the committed quickstart.
  Note the repo's certification gate is *mean of iters 10–30*, a different window — do not quote
  a 6–12 number as a certification result. n=7 is too thin to separate ~1% effects; replicate those.
- **Orphan reaping:** an OOM/crash does **not** reliably kill remote ranks. Survivors hold HBM and
  silently block the next run's idle-gate (cost ~50 min twice here). Correct order is
  **stop driver → cluster-wide rank sweep → verify 0 → relaunch**. Sweeping while a driver may
  start will kill the legitimate new run.
- **Concurrency:** use an `flock` around each GPU stage *and* chain on the previous driver's PID
  exit. A sampled "is anything running" check races with another driver's inter-stage gap.
- 48 iterations ≈ **10% of one epoch** (dataset is 30,341 packed 32K sequences ≈ 994 M tokens;
  one epoch = **474 iterations** at GBS=64). Loss comparisons here are early-training signals.

---

## 8. RESUME HERE (tunnel 5738451 expired 2026-07-27 ~19:35 UTC)

> **2026-07-29 (allocation 5738452): resumed. §9 below supersedes the open questions here —
> the stall root cause is now attributed to named code paths, the 26.04 build/qualification is
> in flight, and the measurement ladder (§9.4) is running. This section is kept as history.**

Work was mid-flight when the allocation ended. State:

**Done and durable (on shared FS / in git):**
- `nemo:26.04.sif` pulled (18 G) and **verified to run on this driver** (§5b).
- `nemo:26.06` SIF + Slingshot + overlay exist but the image is **unusable** (driver).
- 26.02 remains the working, qualified environment. Nothing in the live config was changed.

**Incomplete — re-run in a fresh allocation:**
1. `GEODESIC_CONTAINER_IMAGE_TAG=26.04 bash pipeline_env_setup.sh --only slingshot`
   (was mid-NCCL-compile; `/projects/a5k/public/containers/slingshot/nemo_26.04/` holds only a
   partial `nccl/`. The step is idempotent; it re-runs cleanly. ~15-20 min.)
2. then `--only overlay`, then `--only validate` (expect 19/19; on 26.06 the only failures were
   CUDA-dependent + a missing `grouped_gemm` module — check whether 26.04 also lacks it).
3. Only then flip `CONTAINER_IMAGE_TAG` in `pipeline_env_config.env` to `26.04`.

**The measurement ladder that never ran** (configs already written in the session tmp dir; they
are plain copies of the committed quickstart with single fields changed — regenerate if lost).
All 48 iters, all vs the **J1 baseline: 27.02 s/iter, lm loss 0.6241053 @ i48** on 26.02:

| # | arm | question |
|---|---|---|
| 1 | dropless on 26.04 | **the headline**: does the dropless host-sync cost fall on CUDA 13.1 / TE 2.14 alone? |
| 2 | CF 1.0 on 26.04 | how much of the 6.6 s ragged-vs-uniform gap survives? If it collapses, token-dropping is unnecessary |
| 3 | `optimizer_offload_fraction: 0.5` | quality-neutral lever (1.0→0.0 OOMs; 0.5 is recorded as fitting) |
| 4 | `recompute_modules: ["moe"]` | quality-neutral; targets the 4.21 s `CheckpointFunctionBackward` stall |
| 5 | 3 + 4 combined | |

**Caveat on #1/#2:** mcore must actually *call* the device-side grouped-GEMM path. There are no
`device_initiated` references in our mcore 0.19 MoE code — it may route through TE `GroupedLinear`
transparently, or may need a newer mcore. If the measurement shows no change, check that first
before concluding CUDA 13.1 does not help.

**The K ladder was started three times and never completed an arm** — twice pre-empted by
higher-priority work, once by the tunnel. Those four arms are the best remaining quality-neutral
options and are still unmeasured.

---

## 9. Root cause, attributed (2026-07-29, allocation 5738452 — new-node session)

§1 established the *shape* of the problem (launch starvation, not bubble). This section names the
code paths, from a stack-attribution pass (`scripts/profiling/trace_analysis/host_attrib.py`) over
the four `with_stack=True` traces (non-VPP champion r0/r9 iter10 at the current pin; vpp4_plain
r0/r9). Traced windows are ~15% longer than uninstrumented steady state (27.1 s); ratios hold.

### 9.1 The launch storm is per-expert GEMMs — 66% of all kernel launches

Kernel-family census, one traced iteration:

| family | nonvpp r0 | nonvpp r9 | vpp4 r9 |
|---|---:|---:|---:|
| **GEMM (nvjet/cuBLASLt)** | **168,768** / 5.03 s | **168,771** / 4.96 s | **201,711** / 5.03 s |
| ATen eltwise | 40,196 / 1.27 s | 39,996 / 1.19 s | **89,623** / 1.76 s |
| MoE dispatch (sort/permute/topk) | 16,000 / 0.70 s | 15,680 / 0.69 s | 18,816 / 0.84 s |
| Mamba scan | 4,640 / 1.39 s | 4,640 / 1.39 s | 3,712 / 1.11 s |
| NCCL | 4,382 / 10.54 s | 4,396 / 6.73 s | 4,056 / 8.31 s |
| **total kernels** | **256,470** | **255,907** | **340,816** |
| GPU busy | 63.2% | 50.1% | **44.7%** |

The GEMM count is exact arithmetic for a **per-expert loop**: 128 local experts (512/EP4) × 2
projections × 5 MoE layers × 32 µb × 4 passes (fwd, recompute-fwd, dgrad, wgrad) = 163,840, plus
~5k attention/Mamba/shared GEMMs. TE 2.12's grouped GEMM on **ragged dropless group sizes** issues
one cuBLASLt kernel per expert (mean 29 µs) instead of one grouped kernel — 168k launches carrying
only ~5 s of GPU work. This is the launch storm; everything else is small next to it.

Host arithmetic closes the loop: idle ÷ kernels ≈ **60 µs per kernel** of host-serial framework
time (Python/pybind/autograd dispatch — the launcher threads are only ~10% busy inside CUDA APIs,
so the cost is *around* the API calls, not in them). 256k × 60 µs ≈ 15 s ≈ the entire idle. The
CF result in §2 is the same mechanism from the other side: capacity-factor padding makes the
expert GEMMs uniform (batchable) → 20.21 s, at the price of dropped tokens.

### 9.2 The "syncs" demystified

- **1.93 s of the 2.03 s `cudaStreamSynchronize` (r9) is Megatron timer barriers**
  (`timers.py:start/stop` → barrier collectives at `timing_log_level: 2`) — skew parking, not data
  dependency. Under VPP the timed sections triple (564 vs 188 device-sync pairs).
- **The dropless device→host token-count sync waits ~0 s** (`token_dispatcher.py:
  _maybe_dtoh_and_synchronize`, 320 calls, 2 ms total). The host is the laggard; the GPU never
  makes it wait. The sync's cost is the *serialization point* it creates, not wait time.
- **651 pageable HtoD copies/iter = `hybrid_optimizer.py:param_copy_back_gpu_hook`** — the
  CPU-offloaded optimizer's param copy-backs go through pageable staging despite
  `pin_cpu_params: true`. Small GPU-side (27 ms) but host-serialized at the step boundary.

### 9.3 Why VPP loses on a host-bound run (measured, not theoretical)

On the same interior stage, VPP=4: **+33% kernel launches** (340,816 vs 255,907) — +51,457 fp32
grad-accum `add` kernels from 4× finer chunking, +20% GEMM fragments (6-vs-5 MoE layers after the
rebalance), 3× timer sections — plus 4.23× PP crossings each paying peer-wait (INFR-71 doc). The
binding resource is host ops/iter; VPP spends more of it. GPU busy falls 50.1% → 44.7%.

### 9.4 Utilization ceilings (calibrating the "~99%" expectation)

1F1B bubble = (PP−1)/(µb+PP−1). At µb/pipe = GBS/DP = 32, PP=8: **~18% → ceiling ~82% busy** with
a perfect host. VPP=4 interleave: ~5% → ~95% ceiling. 99% needs v·µb ≥ ~700 — not reachable at
GBS=64. Observed ~50% busy sits far *below* the schedule ceiling: recovery order is (1) launch
granularity (per-expert GEMMs → grouped; the 26.04/TE-2.14.1 question), (2) host per-op cost,
(3) only then does the schedule ceiling bind, and VPP becomes worth re-testing.

### 9.5 The ladder now running (48 iters, 16 nodes, serial, FT off, results → this table)

| arm | delta | question |
|---|---|---|
| a0_champion | none | placement anchor on 5738452 |
| a1_offload05 | `optimizer_offload_fraction: 0.5` | K1' (1.0→0.0 OOMs; 0.5 recorded ~25.7 steady) |
| a2_recmoe | `recompute_modules: [moe]` | K2 (CheckpointFunctionBackward stall) |
| a3_combo | a1+a2 | K3 |
| a4_timers0 | `timing_log_level: 0` | prices the 1.93 s/iter timer barriers |
| m1_2604 | image 26.04 | **headline**: does per-expert GEMM collapse on TE 2.14.1/CUDA 13.1? |
| m2_2604_combo | 26.04 + a3 | best mechanistic combo |

Closed en route: `moe_use_legacy_grouped_gemm` (CUTLASS single-kernel grouped GEMM) is **not
wired for the hybrid path** at this pin — the field is gone from `TransformerConfig` and the
mamba/hybrid specs never pass it. Hand-wiring the spec is a possible follow-up if m1 disappoints.
`bias_activation_fusion` has no squared-relu branch (Nemotron-H's activation) — would crash, not
measure. `apply_rope_fusion` is moot (no rope in Nemotron-H attention).

### 9.6 Preregistration (written 19:40 UTC, before m1's scored iterations)

Anchor a0 on this nodeset = **26.70 s** (mean 10–30). Predictions for m1 (champion on 26.04 =
torch 2.11 / CUDA 13.1 / NCCL 2.29.2 / TE 2.14.1; validation 18/18 green):

- **Modal: 25.8–26.8 s** — mcore 0.19 passes host-side group sizes, so TE 2.14.1's device-side
  grouped path stays dormant; only CPU-overhead trims apply.
- **Upside: 19.9–21.5 s** — TE batches ragged groups internally (CUDA 13.1 cuBLASLt grouped
  GEMM): the census predicts GEMM launches 168.8k → ~1.3k. This is the CF-1.0 floor without
  dropping tokens.
- **Regression: > 27.5 s** — new torch/NCCL misbehaving on CXI.

Decision rules (committed before data): ≤21.5 → storm collapsed, confirm with m1p census
(<~20k GEMM launches), qualify + compose with m3. 24–26.4 → partial; census arbitrates.
26.4–27.1 → modal; 26.04 qualifies as no-regression only; storm fix escalates to mcore wiring.
>27.5 → investigate before qualifying. Loss gate |Δ| ≤ 1e-3 vs 0.62475 (run-to-run band 5.5e-4).
Offload rule (Kyle): m2 adopts 0.5 only if ≥ ~0.8 s faster than m1; otherwise full offload stays.
m3 adopts timers0 if ≥ 0.5 s.

### 9.7 Call-path verdict (analytical, written before m1 landed)

Read both sides of the grouped-GEMM call path (mcore 0.19 checkout; TE 2.14.1 source inside the
26.04 image):

- mcore's alltoall token dispatcher computes counts on device, stages them to host through
  `_maybe_dtoh_and_synchronize`, and delivers **host `List[int]`** (`.tolist()` /
  `.cpu()`) as `m_splits`.
- TE 2.14.1 `GroupedLinear` **only accepts host lists**: `num_gemms = len(m_splits)`,
  `torch.split(inp, m_splits)` → per-expert views → per-expert cuBLASLt calls. A tree-wide
  search finds **no tensor-typed / device-side group-size API in TE 2.14.1's Python surface.**

**Conclusion: the per-expert launch storm cannot collapse on 26.04.** The preregistered upside
branch (≈20 s) is structurally unreachable at TE 2.14.1; expect the modal band (25.8–26.8 s =
CPU-overhead trims only). A device-side path would need TE ≥2.15/2.16 → image 26.06+ → blocked
by the **host driver 565.57.01** — making the driver upgrade (BriCS ask) the root unblock for
the clean fix.

**The in-reach quality-neutral fix is therefore a code change, not a config**: a CUTLASS
grouped-GEMM expert module (`nv-grouped-gemm`, which the qualified 26.02 image ships) wired
into the Nemotron-H MoE spec — one `gmm` kernel per (layer, µb, pass) instead of 128 per-expert
GEMMs, i.e. 168.8k → ~1.3k launches. Upstream's legacy `GroupedMLP` was exactly this and was
removed at 0.19; the previous pin (`.dev.commit`, mcore 0.16) still carries it as a reference
implementation, including its sharded-state-dict mapping. Prototype plan: bridge-side expert
module + spec override, gated on (a) loss parity vs TEGroupedMLP at 48 iters, (b) checkpoint
load compatibility with the existing torch_dist base checkpoint.

### 9.8 Ladder results (26.04-first, per Kyle's reprioritization) + preregistration adjudication

All 48 iters, identical nodelist (allocation 5738452), FT off, mean of iters 10–30:

| arm | image | delta | s/iter | loss@48 | peak GB (rank-0 W&B) |
|---|---|---|---:|---:|---:|
| a0 anchor | 26.02 | none | 26.70 | 0.62475 | 50.9 |
| m1 | 26.04 | none | 28.60 | 0.62405 | 50.9 |
| m1p (steady, unprofiled iters) | 26.04 | none | ~27.42 | — | — |
| **m2** | 26.04 | `optimizer_offload_fraction: 0.5` | **25.66** | 0.62417 | 60.5 |
| **m3** | 26.04 | `timing_log_level: 0` | **26.46** | 0.62451 | 50.9 |
| **m4** | 26.04 | m2+m3 composed | **25.56** (10–48: 25.42) | 0.62432 | 69.1 GB cluster-wide peak (nvidia-smi, all 64 GPUs) |
| a1 (partial, free datum) | 26.02 | offload 0.5 | ~25.5–26.3 @ i26–32 | — | — |

Losses all within 7e-4 of anchor (gate 1e-3) ✓. Plain-26.04 run-to-run band: 27.4–28.6
(m1 vs m1p same config/nodes) — the m2/m3 wins exceed it comfortably.

**Adjudication of §9.6:** m1 landed in the regression branch (>27.5). The §9.7 *mechanism*
prediction was exactly right — the 26.04 trace census is byte-identical on the launch storm
(**168,771 per-expert GEMMs**, same eltwise/dispatch counts; TE 2.14.1 changes nothing about
batching) and compute-kernel time is flat. What the preregistration missed: a **new skew
source**. Both traced ranks wait the identical 4.20 s at the end-of-step timer barrier (was
1.31 s on 26.02) while their own backward got ~1 s *faster* and their optimizer host time is
flat — the laggard is among the untraced ranks (NCCL 2.29.2-vs-2.28.8 collectives or a
stage-specific host path; not further localized because the empirical arms below decide the
engineering question either way).

**m3's surprise: the barriers *create* serialization, not just measure it.** Dropping
`timing_log_level: 2 → 0` with full offload kept recovered 2.14 s (28.60 → 26.46) — multiple
per-iteration global rendezvous force every rank to the slowest at each timer boundary;
without them, jitter overlaps into the naturally-async collectives.

**m2: offload 0.5 = 25.66, the best quality-neutral number measured on any image** (−1.04 s
vs anchor; −2.9 s vs plain-26.04). Per Kyle's rule (adopt only if notable) it qualifies —
notable on both images (a1's surviving partial run shows ~25.5–26.3 on 26.02 too). Memory
cost: +9.6 GB on rank 0 (50.9 → 60.5); extrapolated peak-stage ≈ 83 GB of 95. A
memory-snapshot verification on the peak stage is required before this ships in the cert
config.

**m4 composition verdict:** offload 0.5 subsumes the timer win — m4 (25.56) adds only −0.10 s
over m2 (25.66), inside the run-to-run band. Once the skew source is gone the barriers park
almost nothing. **Recommendation: adopt `optimizer_offload_fraction: 0.5`, keep
`timing_log_level: 2`** (its telemetry is free again), qualify 26.04 with that config:
**25.66 s vs the 26.02 anchor's 26.70 on the identical nodelist** (and ~25.9 est. for
26.02+offload from a1's partial). True peak memory measured across all 64 GPUs during m4:
**69.1 GB of 95** (stage-0 nodes; monotone down to ~32 GB at the last stages) — the
historical ~91 GB concern for offload 0.5 does not reproduce on this config. Gate remaining
before the default flip: m5, a 14-iter FT-ENABLED smoke on 26.04 (nvidia-resiliency-ext
0.4.1 → 0.6.0 is the biggest untested behavioral delta; validator already passes the
ft_launcher flag check).

**m5 (FT-enabled smoke on 26.04): GREEN.** 14/14 iterations at ~27.15 s/iter (the expected
~1.5 s ft_launcher overhead over 25.66 FT-off), 0 restarts, 0 tracebacks, loss healthy —
nvidia-resiliency-ext 0.6.0 accepts our full `--ft-*` flag set and behaves. (First attempt
failed by design: FT requires `checkpoint.save` to be set — probe configs null it; re-ran
with a scratch save dir, deleted after.) **Qualification complete → 26.04 flipped as the
default image with `optimizer_offload_fraction: 0.5` adopted in the quickstart.**

### 9.9 The launch-storm fix, implemented: CutlassGroupedExperts (task #24)

Landed on this branch (commits e128a1a4 + 1b9b57a1): a bridge-side port of upstream's
pre-0.19 `GroupedMLP` (reference `core_v0.13.1`) to the 0.19 experts contract — one CUTLASS
grouped-GEMM kernel per projection over all 128 local experts instead of 128 per-expert
cuBLASLt calls. Latent-MoE aware (in-features = `moe_latent_size` = 1024, verified against
the base checkpoint's `[512, 2688, 1024]` canonical shapes); checkpoint mapping emits the
same canonical keys as TEGroupedMLP (both declare SequentialMLP interchangeability, and the
factory round-trip is unit-tested). Selected via `model.moe_experts_impl: cutlass_grouped`
(default `te_grouped` = untouched upstream path); the field resolves at `provide()` time
because YAML merges land after provider construction. 7/7 unit tests green in-container
(fwd/bwd parity vs a per-expert torch reference, zero-token graph, canonical keys, wiring).

**Preregistration for m6** (26.02 image + offload 0.5 + cutlass_grouped, 48 iters, written
before results): launches should fall ~256k → ~90k/iter. Predicted **20–22.5 s/iter**
(the CF-1.0 uniform floor was 20.21 with token dropping; this does the same collapse
dropless). The next wall behind it: ~88k residual launches ≈ 5.3 s host-serial. If it lands
>24 s, suspect gg ragged-kernel efficiency or the wgrad path — trace before concluding.
Loss gate: 0.6240–0.6248 @48 (the cross-arm band).

### 9.10 m6 adjudication: launch collapse ≠ the win. Cost model corrected (MoE paper §4.3.2)

**m6 (26.02 + offload 0.5 + cutlass_grouped, 48 it): 26.63 s (10–30) / 26.00 s (10–48), min
24.70, loss 0.6241723 ✓.** Against the same-image baseline (~25.5–26.3, a1 partial):
**performance-neutral**. The preregistered 20–22.5 s is refuted. Functionally the module is
fully validated at scale — base-checkpoint warm-start through the new mapping at EP=4, loss
parity to the 4th decimal (kernel-rounding-level agreement) — it stays in the tree as an
opt-in (`moe_experts_impl`), default off.

**Why the prediction was wrong** (per the MoE paper's §4.3.2 taxonomy, which names our exact
path): TE's "multi-stream cuBLASLt" grouped GEMM issues the 168k per-expert launches from a
C++ loop inside ONE pybind call per (layer, µb, pass) — those launches cost ~10 µs each
(~2 s/iter total), not the ~60 µs Python-dispatch cost §9.1 charged them. The ~60 µs ops are
the ~90k Python-level launches (eltwise, dispatch, Mamba, autograd) ≈ 5+ s/iter — THE wall.
And CF-1.0's 6.6 s win was mostly **uniformity** (rank-skew removal + no ragged-dependent
host paths), not launch count. §9.1's census stands; its per-launch cost attribution is
hereby corrected.

**Revised outlook for ≤20 s quality-neutral on THIS stack:** not reachable with config- or
module-level levers. The remaining recoverable host time needs per-op Python/framework
elimination = CUDA graphs. Two paths, in order of leverage:
1. **Driver upgrade (BriCS)** → image 26.06+/TE ≥2.15 → device-initiated grouped GEMM +
   sync-free dispatch + graphs on the expert path — the paper's designed solution (§4.3.7).
2. **Partial-graphs revisit at this pin** using TE's graph memory optimizations
   (`make_graphed_callables(_order=...)` pool sharing + static-buffer reuse; the paper reports
   ~7 GB overhead where our naive attempt OOM'd at +14 GB) on the mamba/router/preprocess
   scopes. `attn` stays blocked (TE event-record under CP capture; CP=4 mandatory). Uncertain,
   real effort — the natural next investigation.

Standing quality-neutral champion: **25.56–25.66 s = 26.04 + offload 0.5 (± timers0)** —
committed as the default stack.

### 9.11 Correction: m6 was an A/A — the wiring never engaged. Fixed; real test = m6c

The m6p census (168,771 nvjet launches, byte-identical) exposed it: **the CUTLASS module
never ran.** Root cause: the NemotronH bridge registers `provider=MambaModelProvider`, so
`to_megatron_provider()` returns a plain MambaModelProvider — the `moe_experts_impl` field
lived on NemotronHModelProvider, a class training never instantiates, and **the YAML
`model:` merge drops unknown keys silently** (hazard for the record: it defeated 3 wiring
unit tests and a 48-iter run without one warning; the tests constructed the subclass
directly). m6's numbers are therefore a valid baseline replicate (26.0–26.6 ≈ a1), NOT a
verdict on the module. §9.10's "perf-neutral" adjudication is retracted; the corrected cost
model (C++-loop launches ~10 µs → predicted win ~1–2 s, not ~7) stands on paper evidence.

Fix (committed): field + provide()-time hook moved to MambaModelProvider; the swap now logs
a rank-0 marker; wiring tests retarget the class the bridge actually uses. 7/7 green.

**m6c preregistration** (26.02 + offload 0.5 + cutlass, REALLY engaged this time — verified
by the log marker before scoring): predicted **23.5–25.5 s** (−0.5 to −2.5 vs the ~26.0
baseline, per the corrected model). Within ±0.3 of baseline → launch count truly doesn't
matter on this workload; >0.5 s SLOWER → CUTLASS ragged kernels less efficient than the
multi-stream cuBLASLt picks. Loss gate unchanged.
