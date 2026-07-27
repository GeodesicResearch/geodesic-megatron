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
