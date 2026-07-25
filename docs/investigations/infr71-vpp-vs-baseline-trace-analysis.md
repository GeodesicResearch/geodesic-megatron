# INFR-71 — VPP=4 vs no-VPP: exposed-communication analysis from torch-profiler traces

**Verdict up front: VPP=4 did *not* reduce exposed communication, and it did not reduce the
pipeline bubble. It increased total non-compute stall on both profiled ranks (+5.23 s on rank 0,
+1.80 s on rank 9) while leaving compute essentially unchanged on the representative rank. The
traced window grows +1.85 s, which accounts for ~98% of the +1.89 s untraced steady-state
regression measured in the same two logs.**

---

## 1. Provenance

| | Baseline (no VPP) | VPP = 4 |
|---|---|---|
| profile dir | `/projects/a5k/public/profiles/infr71/vpp_v0_baseline/20260725T132926-j5738449/` | `/projects/a5k/public/profiles/infr71/vpp_v3_vpp4/20260725T134425-j5738449/` |
| commit | `6be891dc` (`feat/infr71-vpp-pp-overlap`) | `6be891dc` (same) |
| world size / topology | 64 GPUs, TP1·CP4·EP4·PP8·DP2, seq 32K, GBS 64, mbs 1 → **32 microbatches** | identical |
| captured iteration | `ProfilerStep#9` (iteration 10, 1-based) | same |
| ranks | 0 (PP stage 0), 9 (PP stage 1) | same |
| container / torch | `nemo_26.02.nemotron_3_super.sif`, torch 2.10.0a0 | same |
| profiler | `with_stack=True record_shapes=True` | same |

Untraced steady-state per-iteration time from the two `raw_log_snapshot.out` files
(iters 2–9 and 11–14, excluding the first iter and the profiled iter 10 and its successor):
**27.529 s (baseline) vs 29.422 s (VPP=4) → +1.893 s (+6.9%)**.

### Config deltas — this is an *arm* A/B, not a pure VPP toggle

`diff` of the two `resolved_config_snapshot.yaml` shows five differences, three of them
load-bearing:

| key | baseline | VPP=4 |
|---|---|---|
| `virtual_pipeline_model_parallel_size` | `null` | `4` |
| `hybrid_layer_pattern` | contiguous, 88 layers, **11 layers/stage, every stage M5/E5/attn1** | 32 `|`-separated chunks, **stage-0 unloaded: 9 layers, only 1 MoE** |
| `recompute_modules` | `[moe, shared_experts]` | `[core_attn, moe, shared_experts]` |
| `batch_p2p_sync` | `true` | `false` |
| `timing_log_level` | 2 | 1 |

Per-stage layer counts derived from the two patterns (chunk *i* → PP rank *i* mod 8):

| PP stage | baseline layers (M/E/attn) | VPP=4 layers (M/E/attn) |
|---|---|---|
| 0 (**rank 0**) | 11 (5/5/1) | **9 (5/1/3)** |
| 1 (**rank 9**) | 11 (5/5/1) | **10 (4/6/0)** |
| 2 | 11 | 12 (5/6/1) |
| 3 | 11 | 12 (7/5/0) |
| 4 | 11 | 12 (4/5/3) |
| 5 | 11 | 12 (5/6/1) |
| 6 | 11 | 11 (6/5/0) |
| 7 | 11 | 10 (4/6/0) |

Consequences to keep in mind when reading the tables:

* **Rank 0 is not comparable across arms.** Under VPP it holds 9 layers with a single MoE layer
  instead of 11 layers with five, so its compute drops 9.33 s → 5.94 s and its EP all-to-all
  kernel count drops 1440 → 288 (exactly 1/5). Rank 0 in the VPP arm is the *lightest* stage in
  the pipeline and therefore mostly measures how long it waits for everyone else.
* **Rank 9 is the near-clean comparison**: 11 layers (M5/E5/attn1) → 10 layers (M4/E6/attn0).
  Its EP all-to-all count moves 1440 → 1728 (= 6/5) and its CP all-to-all count 1920 → 1536
  (= 4/5), exactly tracking the MoE/Mamba layer swap. Its compute union is flat (9.192 → 9.237 s),
  and the `core_attn` recompute added in the VPP arm costs it nothing because it has no attention
  layers. **Read rank 9 for the mechanism; read rank 0 for the skew.**

---

## 2. Method

GPU-side events only (`cat ∈ {kernel, gpu_memcpy, gpu_memset}`); CPU-side ops were never mixed in.
All aggregates are **interval-union** measures, not naive sums:

* `window` = first GPU-kernel start → last GPU-kernel end (this equals the `ProfilerStep#9`
  annotation span to within 0.5 ms in all four traces).
* `compute` = union of non-NCCL kernel intervals.
* `comm` = union of NCCL kernel intervals (name contains `nccl`).
* `exposed_comm` = measure(comm) − measure(comm ∩ compute).
* `overlapped_comm` = comm − exposed_comm.
* `idle` = window − union(all GPU kernels).

Scripts (all under `/home/a5k/kyleobrien.a5k/.claude/jobs/21b8d28a/tmp/trace_analysis/`):
`parse_trace.py` (base metrics), `parse_trace2.py` (adds per-process-group attribution from the
NCCL metadata PyTorch stamps into kernel `args`), `parse_trace3.py` (PP-p2p timeline),
`verify_pp_p2p.py` (correlation attempt, see below). Each is a single streaming pass over the
gzipped trace; no file was fully materialised in memory.

### Splitting SendRecv — and why "AllToAll" is zero

**There are no `ncclDevKernel_AllToAll` kernels in these traces.** NCCL implements all-to-all
as grouped send/recv, so *every* all-to-all appears as a `ncclDevKernel_SendRecv` kernel. Naively
reporting "SendRecv" as "PP p2p" would be wrong here — most SendRecv kernels are CP and EP
all-to-alls.

The kernels are separable because PyTorch stamps `"Collective name"`, `"Process Group
Description"` and `"In msg nelems"` into the `args` of NCCL kernels launched through
`record_param_comms`. This gives three clean SendRecv sub-populations:

1. `CONTEXT_PARALLEL_GROUP` / `all_to_allv`, `send`, `recv` — CP ring + Mamba CP all-to-all.
2. `EXPERT_MODEL_PARALLEL_GROUP` / `all_to_allv` — MoE token dispatch/combine.
3. **No PG metadata at all** — these are the PP exchanges. Megatron's `p2p_communication`
   path uses `torch.distributed.batch_isend_irecv`, which does not go through
   `record_param_comms`, so the kernels carry no `External id` and no PG fields, and they land
   on their own dedicated NCCL stream (e.g. stream 96 on baseline rank 0, stream 88 on VPP rank 0).

**Verification that population 3 is PP p2p** — the count of unattributed SendRecv kernels equals
the count of `torch/distributed/distributed_c10d.py(2717): batch_isend_irecv` python_function
events, exactly, in all four traces:

| trace | unattributed SendRecv kernels | `batch_isend_irecv` calls | `get_batch_from_iterator` calls |
|---|---|---|---|
| baseline rank 0 | 39 | **39** | 32 |
| baseline rank 9 | 77 | **77** | 32 |
| VPP=4 rank 0 | 165 | **165** | 128 |
| VPP=4 rank 9 | 165 | **165** | 128 |

39 is also exactly the 1F1B schedule arithmetic for stage 0 at PP=8 / 32 microbatches
(7 warmup send-only + 25 fused steady + 7 cooldown recv-only). The `get_batch_from_iterator`
counts (32 → 128) confirm the 4× scheduling granularity VPP=4 introduces.

---

## 3. Results

### 3.1 Headline interval-union table

| metric | base r0 | **vpp4 r0** | base r9 | **vpp4 r9** |
|---|---:|---:|---:|---:|
| window (s) | 30.494 | **32.343** | 30.493 | **32.341** |
| compute union (s) | 9.327 | **5.943** | 9.192 | **9.237** |
| comm union (s) | 11.268 | **22.009** | 7.284 | **5.564** |
| **exposed comm (s)** | **11.106** | **21.994** | **7.277** | **5.457** |
| overlapped comm (s) | 0.162 | 0.014 | 0.007 | 0.108 |
| idle (s) | 10.061 | 4.405 | 14.024 | 17.647 |
| **stall = exposed + idle (s)** | **21.167** | **26.400** | **21.301** | **23.104** |
| compute as % of window | 30.6% | 18.4% | 30.1% | 28.6% |
| GPU kernels (total) | 256,144 | 103,735 | 255,570 | 290,059 |
| NCCL kernels | 4,381 | 4,431 | 4,395 | 4,053 |
| non-NCCL kernels | 251,763 | 99,304 | 251,175 | 286,006 |

Per-rank deltas (they close exactly — window = compute + exposed + idle):

| rank | Δwindow | Δcompute | Δexposed | Δidle | Δstall |
|---|---:|---:|---:|---:|---:|
| 0 | **+1.849** | −3.384 | **+10.889** | −5.656 | **+5.233** |
| 9 | **+1.848** | +0.045 | **−1.820** | +3.624 | **+1.803** |

### 3.2 Exposed time by collective family

Exposed seconds / kernel count. `AllToAll` has no row because NCCL emits no AllToAll kernels —
the CP and EP all-to-alls are the two indented SendRecv sub-rows.

| family | base r0 | vpp4 r0 | base r9 | vpp4 r9 |
|---|---:|---:|---:|---:|
| **SendRecv (all)** | **8.356 s** / 3847 | **18.706 s** / 4293 | **4.039 s** / 3885 | **3.768 s** / 3429 |
| ↳ PP p2p (`batch_isend_irecv`) | 7.176 s / **39** | **17.929 s** / **165** | 2.855 s / **77** | 2.506 s / **165** |
| ↳ EP all-to-allv (MoE dispatch) | 0.802 s / 1440 | 0.161 s / 288 | 0.805 s / 1440 | 0.980 s / 1728 |
| ↳ CP all-to-allv + send/recv | 0.385 s / 2368 | 0.639 s / 3840 | 0.386 s / 2368 | 0.282 s / 1536 |
| AllReduce | 1.569 s / 177 | 3.080 s / 49 | 2.297 s / 177 | 0.330 s / 209 |
| AllGather | 0.989 s / 346 | 0.118 s / 82 | 0.940 s / 330 | 1.253 s / 406 |
| ReduceScatter | 0.207 s / 10 | 0.113 s / 6 | 0.335 s / 2 | 0.312 s / 8 |
| Broadcast | 0.000 s / 1 | 0.000 s / 1 | 0.000 s / 1 | 0.000 s / 1 |
| AllToAll (native kernel) | — / 0 | — / 0 | — / 0 | — / 0 |
| **total NCCL kernels** | 4381 | 4431 | 4395 | 4053 |

Note the AllReduce rows are dominated by a *single* `DATA_PARALLEL_GROUP` allreduce that acts as
an end-of-step rendezvous: 1.465 s on **both** baseline ranks (identical to 4 decimal places →
it is pure skew absorption, not data movement), vs 0.094 s on both VPP ranks. Likewise the 2.935 s
`MODEL_PARALLEL_GROUP` allreduce on VPP rank 0 is rank 0 waiting for the pipeline, not traffic.

### 3.3 PP p2p kernel cost distribution

| | base r0 | vpp4 r0 | base r9 | vpp4 r9 |
|---|---:|---:|---:|---:|
| PP p2p kernels | 39 | **165 (4.23×)** | 77 | **165 (2.14×)** |
| mean per kernel | **184.00 ms** | **108.66 ms** | **37.08 ms** | **15.19 ms** |
| median | 127.50 ms | 111.47 ms | 2.95 ms | 5.92 ms |
| p10 / p90 | 31.85 / 207.46 ms | 21.19 / 187.70 ms | 2.73 / 2.98 ms | 2.92 / 35.74 ms |
| max | 2756.19 ms | 659.58 ms | 2236.85 ms | 543.10 ms |
| **total exposed** | **7.176 s** | **17.929 s** | **2.855 s** | **2.506 s** |

`165 / 39 = 4.23×` on stage 0 confirms the expected ~4× increase in PP crossings under VPP=4.
Rank 9 goes 77 → 165 (2.14×) because a middle stage already issues two fused exchanges per
steady-state microbatch under plain 1F1B.

Per-kernel cost falls (184 → 109 ms on r0; 37.1 → 15.2 ms on r9) but **not by 4×**, so the product
rises: the crossings are individually cheaper, just not cheap enough to pay for being 4× more
numerous. Most of each kernel's duration is *waiting* for the peer, not wire time — the largest
single baseline p2p kernel is 2.76 s, far beyond any plausible transfer of a ~170 MB activation.

### 3.4 Structure of the "idle" time — it is not a pipeline bubble

Gap-size histogram of the GPU idle (gaps between merged kernel intervals):

| trace | idle total | # gaps | >1 s | 100 ms–1 s | 10–100 ms | 1–10 ms | <1 ms | largest gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base r0 | 10.061 s | 203,065 | 0 | 0 | 0.162 s (8) | 5.034 s (1981) | 4.866 s (201,076) | 34 ms |
| vpp4 r0 | 4.405 s | 95,382 | 0 | 0 | 0.124 s (5) | 1.376 s (637) | 2.906 s (94,740) | 46 ms |
| base r9 | 14.024 s | 233,953 | 0 | 0 | 0.120 s (5) | 5.838 s (2137) | 8.066 s (231,811) | 39 ms |
| vpp4 r9 | 17.647 s | 260,431 | 0 | 0 | 0.082 s (6) | 6.827 s (2500) | 10.739 s (257,925) | 22 ms |

**No gap anywhere exceeds 50 ms.** The idle is hundreds of thousands of sub-millisecond
kernel-launch gaps — the GPU is launch-starved, not blocked on a peer. Idle scales almost
linearly with kernel count (39, 43, 55, 61 µs of idle per GPU kernel across the four traces),
which is the signature of CPU-side launch overhead, amplified here by `with_stack=True`.
Consistently, the traced window exceeds the untraced steady-state iteration by ~2.97 s (baseline)
and ~2.92 s (VPP) — *comparable* overhead in both arms, so cross-arm deltas remain meaningful,
but the absolute idle figure must not be read as "the pipeline bubble".

The classic bubble is *not visible as idle at all*. Because `batch_p2p_sync` differs between the
arms (`true` baseline / `false` VPP), waiting-for-a-peer is charged to a different bucket in each
arm: with the sync the CPU blocks and the GPU drains (→ idle); without it the CPU races ahead and
the GPU sits inside a spinning NCCL SendRecv kernel (→ exposed comm). **This is why
`exposed + idle` is the only cross-arm-robust stall metric here**, and why rank 0's headline
"+10.9 s exposed / −5.7 s idle" is largely a re-labelling of the same wait.

### 3.5 Where the stall sits in the iteration (deciles of the window, seconds)

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| base r0 compute | 0.61 | 0.22 | 1.22 | 1.27 | 1.27 | 1.17 | 1.21 | 1.14 | 1.13 | 0.08 |
| base r0 PP-exposed | 0.31 | 2.63 | 0.65 | 0.61 | 0.54 | 0.64 | 0.54 | 0.75 | 0.51 | 0.00 |
| vpp4 r0 compute | 0.60 | 0.56 | 0.84 | 0.59 | 0.75 | 0.66 | 0.65 | 0.61 | 0.68 | 0.00 |
| vpp4 r0 PP-exposed | 1.82 | 2.22 | 1.73 | 2.19 | 1.88 | 2.09 | 2.05 | 2.30 | 1.64 | 0.00 |
| base r9 compute | 0.54 | 0.38 | 1.27 | 1.23 | 1.23 | 1.19 | 1.22 | 1.15 | 0.92 | 0.08 |
| base r9 PP-exposed | 0.31 | 2.13 | 0.03 | 0.03 | 0.02 | 0.02 | 0.03 | 0.16 | 0.12 | 0.00 |
| vpp4 r9 compute | 0.90 | 1.06 | 1.13 | 1.06 | 1.10 | 1.06 | 1.03 | 0.93 | 0.88 | 0.08 |
| vpp4 r9 PP-exposed | 0.86 | 0.27 | 0.08 | 0.25 | 0.08 | 0.27 | 0.08 | 0.58 | 0.04 | 0.00 |

Baseline rank 0's PP stall is concentrated in decile 2 (the warmup ramp, 2.63 s) and then a flat
~0.6 s/decile. Under VPP, rank 0's PP stall is *uniformly* ~1.6–2.3 s in **every** decile — the
warmup bubble was indeed flattened, but replaced by a continuous stall that is ~3× larger in
total. Rank 9's compute becomes more evenly spread under VPP (0.88–1.13 s/decile vs a 0.38 s
dip in decile 2), which is exactly the bubble-filling VPP promises — it is just not worth what it
costs elsewhere.

---

## 4. Mechanistic verdict

**Q1 — Did VPP reduce exposed communication?**
**No, it increased it.** On rank 0 exposed comm goes 11.106 s → 21.994 s (**+10.889 s, +98%**),
driven almost entirely by PP p2p (7.176 → 17.929 s, +10.75 s). On rank 9 exposed comm falls
7.277 → 5.457 s (−1.820 s), but that decrease is *not* a communication win: it is the
`batch_p2p_sync=false`/`true` bucket shift plus the disappearance of a 1.465 s end-of-step
`DATA_PARALLEL_GROUP` allreduce rendezvous, and rank 9's idle rises by more (+3.624 s) than its
exposed comm falls. On the bucket-shift-robust metric, **total stall (exposed + idle) rises on
both ranks: +5.233 s on rank 0 and +1.803 s on rank 9.** INFR-71's second success criterion
("performance traces should show reduced exposed communication") is **not met**.

**Q2 — Did VPP reduce idle time (the pipeline bubble)?**
**Not in any way that helps, and the question is partly ill-posed for these traces.** Rank 0's
idle drops 10.061 → 4.405 s and rank 9's *rises* 14.024 → 17.647 s. But the gap histogram shows
no idle gap anywhere exceeds 50 ms in any trace: the idle is 95k–260k sub-millisecond
kernel-launch gaps, i.e. CPU launch starvation (inflated by `with_stack=True`), not a pipeline
bubble. Idle tracks kernel count at a near-constant 39–61 µs/kernel, and rank 0's idle fall is
explained by its kernel count collapsing 256k → 104k (the unloaded stage-0 layer pattern), not by
bubble removal. The *real* bubble shows up as exposed PP p2p, and there VPP made it worse
(rank 0: 7.18 → 17.93 s). The one genuine bubble-filling effect visible is rank 9's compute
becoming flatter across the iteration (§3.5) — real, but far too small to matter.

**Q3 — Does the net account for the +2.09 s/iter regression?**
**Yes, essentially completely.** The traced window grows +1.849 s (rank 0) and +1.848 s (rank 9) —
the two ranks agree to 1 ms, so this is a genuine iteration-length change, not rank noise. The
untraced steady-state means in these same two logs are 27.529 s and 29.422 s → **+1.893 s**, so
the traces reproduce **97.6%** of the regression I measure from those logs (and 88% of the
+2.09 s figure quoted in the task, which uses slightly different iteration windows). The exact
decomposition on the representative rank 9 is:

```
Δwindow  +1.848 s  =  Δcompute +0.045  +  Δexposed_comm −1.820  +  Δidle +3.624
```

i.e. **compute is flat and 100% of the regression is stall**. On rank 0:

```
Δwindow  +1.849 s  =  Δcompute −3.384  +  Δexposed_comm +10.889  +  Δidle −5.656
```

— rank 0 gave up 3.38 s of compute (nine layers instead of eleven, one MoE layer instead of five)
and got 5.23 s *more* stall for it. Both ranks tell the same story from opposite ends: VPP=4 at
this workload buys a flatter compute profile and pays for it with 4.23× as many PP crossings that
are only ~1.7× cheaper each, plus ~13.5% more GPU kernels on the loaded stages (rank 9:
255,570 → 290,059) from running the pipeline schedule at 4× finer granularity
(128 microbatch-chunk invocations vs 32).

**Q4 — Mean per-SendRecv-kernel cost and kernel counts.**

For the **PP p2p** population specifically (the number that answers "are VPP's extra crossings
individually cheaper?"):

| | baseline | VPP=4 | ratio |
|---|---:|---:|---:|
| rank 0 — kernels | 39 | 165 | **4.23×** |
| rank 0 — mean cost | 184.00 ms | 108.66 ms | 0.59× |
| rank 0 — total exposed | 7.176 s | 17.929 s | **2.50×** |
| rank 9 — kernels | 77 | 165 | **2.14×** |
| rank 9 — mean cost | 37.08 ms | 15.19 ms | 0.41× |
| rank 9 — total exposed | 2.855 s | 2.506 s | 0.88× |

For the **whole SendRecv family** (PP p2p + CP all-to-all + EP all-to-all, since NCCL emits no
native AllToAll kernels): baseline 3,847 kernels @ 2.176 ms mean (rank 0) and 3,885 @ 1.043 ms
(rank 9); VPP=4 4,293 @ 4.368 ms (rank 0) and 3,429 @ 1.099 ms (rank 9). Total NCCL kernel
launches: 4,381 / 4,431 (rank 0, base / VPP) and 4,395 / 4,053 (rank 9).

So: **each VPP crossing is individually 1.7–2.4× cheaper, but there are 2.1–4.2× more of them,
and the product is a net loss on the stage that matters.** The per-kernel cost is dominated by
peer-wait, not wire time (baseline max 2.76 s for a single p2p kernel), which is why splitting
one exchange into four does not divide its cost by four.

---

## 5. Caveats

1. **The arms differ by more than VPP** (§1): `recompute_modules` gains `core_attn`, the layer
   pattern is rebalanced with an unloaded stage 0, and `batch_p2p_sync` flips to `false`. This is
   the config that was measured, and the layer rebalance was reportedly required for VPP=4 to fit
   at 32K/CP4, so it is arguably intrinsic to the arm — but it means "VPP alone" is not isolated.
   Rank 9 is the closest to a clean comparison (11 → 10 layers, one Mamba swapped for one MoE,
   compute union flat to 0.5%).
2. **`idle` and `exposed_comm` are not individually comparable across the arms** because
   `batch_p2p_sync` moves peer-waiting between the two buckets. Only their sum is robust. Both are
   reported above so the split is visible.
3. **Profiling overhead inflates the window** by ~2.9–3.0 s in both arms (30.494 vs 27.529 s;
   32.343 vs 29.422 s) and lands almost entirely in `idle` as sub-millisecond launch gaps. The
   overhead is near-identical in both arms, so the +1.85 s traced delta is trustworthy, but
   absolute idle should not be quoted as a steady-state figure.
4. **PP p2p kernels carry no message-size metadata** (no `In msg nelems` on the
   `batch_isend_irecv` path), so PP bytes-on-the-wire could not be measured and is not estimated
   here. Measured byte volumes are available only for the metadata-carrying collectives, e.g.
   EP all-to-allv 353.5 GB (baseline rank 0) → 70.9 GB (VPP rank 0, one MoE layer) and
   353.4 → 426.8 GB on rank 9.
5. Only two of 64 ranks were profiled. Rank 0 is PP stage 0 and rank 9 is PP stage 1; the heaviest
   VPP stages (2–5, twelve layers each) were not captured, so the true critical path may be worse
   than rank 9 shows.
