# VPP and PP communication overlap on Nemotron-3 Super-120B (INFR-71)

**Question (from the ticket):** build a VPP variant of the 120B quickstart that
outperforms the non-VPP config, with traces showing reduced exposed communication.

**Answer:** at the quickstart's workload it does not, and cannot — for a specific,
measured reason. But there *is* a regime where VPP wins, and the boundary is now
located. PP communication overlap, the mechanism the NeMo docs pair with VPP, is
unusable on this model at this workload: it is either numerically wrong or out of
memory, with no third option left by the code.

Everything below is measured on 16 nodes / 64 GPUs (TP1 · CP4 · EP4 · PP8 · ETP1 ·
DP2, seq 32K), all arms in the **same allocation** so Dragonfly placement is
controlled — placement alone moves this workload by ~2.7 s/iter, which is larger
than most of the effects here. 14-iteration probes, mean of iterations 6–12,
fault tolerance off, W&B `megatron_training/vpp_*`.

## 1. What the code forces (pinned Megatron-Core `3758b54b2`)

Four facts, each verified in the pinned source, define the whole design space:

| Fact | Where | Consequence |
|---|---|---|
| `overlap_p2p_comm` requires an interleaved (VPP) schedule | `schedules.py:2043` raises "Non-interleaved pipeline parallelism does not support overlapping p2p communication" | The ticket's two asks are ONE experiment. You cannot overlap PP comm without VPP. |
| `overlap_p2p_comm` forbids batched p2p | `schedules.py:952-953` raises "Can not use both overlap_p2p_comm and batch_p2p_comm" | Overlap must use un-batched `isend`/`irecv`. |
| Batched p2p ends in a device-wide `torch.cuda.synchronize()` | `p2p_communication.py:416-419`, gated by `batch_p2p_sync` (default True) | That sync is what makes the next fact safe. Un-batched p2p has no equivalent. |
| Pipeline outputs are pseudo-freed right after the send is issued | `schedules.py:157` `deallocate_output_tensor`, and `MambaModelProvider` sets `deallocate_pipeline_outputs: True` (`mamba_provider.py:164`) | Safe under batched p2p (the sync guarantees the send completed). Races an async un-batched send. |

Also settled while scoping, so nobody re-tries them:

- **`overlap_moe_expert_parallel_comm` is hard-blocked on this model.**
  `combined_1f1b.py:345` asserts `isinstance(unwrapped_model, GPTModel)`; Nemotron-H
  builds a `MambaModel`, which is a *sibling* of `GPTModel`, not a subclass. It is
  additionally incompatible with the `moe` recompute we depend on for memory
  (`transformer_config.py:2109-2111`). VPP being fixed does **not** unblock it.
  (Note: `configs/inoculation_midtraining/im_fyn1668_v1/em/im_nemotron_30b_baseline_tso_em.yaml`
  sets this flag today — it passes every config-level assert and would die at that
  runtime check.)
- **`moe_shared_expert_overlap`** is blocked by latent MoE (`moe_layer.py:421-424`).
- **`defer_embedding_wgrad_compute`** would crash, not no-op: the buffer it clears
  exists only on `GPTModel` (`schedules.py:762` vs `gpt_model.py:229`).
- **DeepEP / HybridEP** need InfiniBand-class primitives; this fabric is Slingshot/CXI.
- **Do not add a `comm_overlap:` block to a config.** When present, `setup()` force-sets
  `overlap_p2p_comm=True` / `batch_p2p_comm=False` whenever PP>1 and VPP>1, and
  silently overrides `ddp.overlap_param_gather` (`comm_overlap.py:454-462`, `:608-618`) —
  i.e. it would enable a NaN-producing path and clobber a deliberate setting. Set the
  `model.*` fields directly instead.

## 2. The ladder

| Probe | Delta from the shipped quickstart | s/iter | Verdict |
|---|---|---:|---|
| v0 | none (anchor) | **27.50** | — |
| v1 | `logger.timing_log_level: 1` | 27.36 | −0.5%, within run-to-run spread |
| v2 | v1 + `model.batch_p2p_sync: false` | 27.55 | null |
| v3 | v2 + **VPP=4** | **29.59** | **+7.6% — VPP loses** |
| v4 | v3 + `overlap_p2p_comm` (+ `batch_p2p_comm: false`) | **NaN at iter 2** | broken |
| v5 | v4 + `FI_CXI_RDZV_THRESHOLD=64K` | **NaN at iter 2** | broken |
| v6 | v4 + `CUDA_DEVICE_MAX_CONNECTIONS=8` | **NaN at iter 2** | broken |
| v7 | v4 + `deallocate_pipeline_outputs: false` | **OOM in NCCL init** | infeasible |
| v8 | v0 at **GBS=32** | 17.90 | — |
| v9 | v3 at **GBS=32** | **17.59** | **−1.7% — VPP wins** |
| v8b | v8 repeat | 18.06 | replicate |
| v9b | v9 repeat | 17.63 | replicate — win holds |

Loss parity across every arm that ran: 0.70605–0.70629 at iteration 14 (spread 2.4e-4),
so none of these knobs changes numerics — except the overlap arms, which produce NaN.

### The two sync-storm levers are null here

mcore's timers call `torch.cuda.synchronize()` unconditionally on both `start()` and
`stop()` (`timers.py:152,165`), and at `timing_log_level: 2` an interior PP stage runs
~8 level-2 timers per microbatch — ~256 device drains per iteration. Every batched PP
exchange adds another (`batch_p2p_sync`), ~64 more. Removing both changes nothing
measurable at this topology (v1, v2): the CPU is far enough ahead that draining it does
not cost wall-clock here. Worth knowing, since both look expensive on paper.

### UPDATE: the NaN is a known upstream bug, fixed after our pin

Megatron-LM commit `260cba71` (2026-04-16), "fix: wait for async P2P send before
deallocating output tensor" (PR #4047), adds exactly the missing wait in the interleaved
schedule:

```python
# isend() copies asynchronously; wait until the copy is done before
# freeing the source buffer, otherwise the next PP stage gets corrupted data.
if send_next_wait_handle is not None and config.deallocate_pipeline_outputs:
    send_next_wait_handle.wait()
```

Our submodule is pinned at `3758b54b2` (PR #4025, 2026-03-27) — three weeks and 22 PRs
earlier — so we do not have it. Applying that patch to the pinned tree and re-running the
overlap arm: **14/14 iterations, 0 NaN, loss 0.70602** (matching every other arm to 2.4e-4).

So the correct statement is *not* "overlap is numerically broken on this model" but
"overlap is broken at our pin, and upstream already fixed it". With the fix applied the
arm is correct but still slower at this workload: **31.45 s/iter** vs 27.50 baseline and
29.59 VPP-without-overlap — un-batched isend/irecv remains the more expensive form on CXI,
which is the original performance finding, now cleanly separated from the correctness bug.

The durable fix is to move the pin forward rather than carry a patch; that upgrade is
tracked separately.

### PP comm overlap at our pin: unusable (superseded by the update above)

Three independent arms produce **deterministic NaN at iteration 2, on the last stage**
(rank 63), unchanged by the CXI rendezvous threshold (v5) or by giving the device eight
hardware work queues instead of one (v6). The mechanism is the fourth code fact above:
un-batched async sends have no device sync before `deallocate_output_tensor` pseudo-frees
the send buffer, so the receiver can read a freed tensor. Removing the deallocation is the
matching fix, and it does remove the NaN path — but it **OOMs in NCCL's own allocator**
during init (`include/alloc.h:232`, `Cuda failure 2 'out of memory'`) on verified-clean
GPUs, because VPP already raises activation residency and that deallocation is a memory
optimization this workload depends on. Both doors are closed at 32K/CP4.

## 3. Why VPP loses at GBS=64 — from the traces

Profiler captures of iteration 10 on ranks 0 and 9 for v0 and v3
(`/projects/a5k/public/profiles/infr71/`), interval-union arithmetic on GPU kernels:

| metric (s) | base r0 | vpp4 r0 | base r9 | vpp4 r9 |
|---|---:|---:|---:|---:|
| window | 30.494 | 32.343 | 30.493 | 32.341 |
| compute (union) | 9.327 | 5.943 | 9.192 | 9.237 |
| **exposed comm** | 11.106 | 21.994 | 7.277 | 5.457 |
| idle | 10.061 | 4.405 | 14.024 | 17.647 |
| **stall (exposed + idle)** | **21.167** | **26.400** | **21.301** | **23.104** |

The traces reproduce the regression: Δwindow is **+1.849 s (r0)** and **+1.848 s (r9)** —
the two ranks agree to 1 ms — against a +1.89 s untraced delta, i.e. 97.6%.

Rank 9 is the clean read (rank 0's layer count differs between arms because the VPP
config uses a stage-0-unloaded pattern), and it decomposes exactly:

```
+1.848 s  =  compute +0.045  +  exposed comm −1.820  +  idle +3.624
```

**Compute is flat; 100% of the regression is stall.** And the reason VPP cannot win by
subdividing the exchanges is in the per-kernel numbers:

| PP p2p | base r0 | vpp4 r0 | base r9 | vpp4 r9 |
|---|---:|---:|---:|---:|
| kernels | 39 | 165 (4.23×) | 77 | 165 (2.14×) |
| mean cost | 184.0 ms | 108.7 ms | 37.1 ms | 15.2 ms |
| exposed total | 7.176 s | **17.929 s** | 2.855 s | 2.506 s |

The expected ~4× increase in crossings is confirmed. Each crossing gets 1.7–2.4× cheaper,
but there are 2.1–4.2× more of them, and the net is a loss. The reason the cost does not
divide cleanly is that **a PP p2p kernel is dominated by waiting for the peer, not by wire
time** — one baseline p2p kernel measures 2.76 s, which no 64 MiB transfer can explain on a
100+ GB/s fabric. Splitting a wait four ways does not quarter it; it adds sync points.

This also corrects an earlier framing: the pipeline bubble on this model does **not** show
up as large idle gaps. No idle gap in any of the four traces exceeds 50 ms; idle is
95k–260k sub-millisecond launch gaps. The bubble is inside the p2p kernels, as peer-wait.

## 4. Where VPP does pay: the microbatch crossover

VPP's benefit scales with the bubble, which scales inversely with microbatches per
replica; its cost is the extra crossings, which is roughly fixed. So halve the
microbatches and the sign flips:

| microbatches/replica | no VPP | VPP=4 | VPP delta |
|---|---:|---:|---|
| 32 (GBS 64 — the shipped quickstart) | 27.50 | 29.59 | **+7.6% (worse)** |
| 16 (GBS 32) | 17.98 | 17.61 | **−2.1% (better)** |

The GBS=32 arms were each run twice, because a ~2% claim sits near the run-to-run
spread. They separate cleanly: no-VPP 17.90 / 18.06 (mean 17.98), VPP=4 17.59 / 17.63
(mean 17.61) — the ranges do not overlap, and VPP is also the higher-throughput arm on
both samples (74.7 and 74.5 vs 73.4 and 72.8 TFLOP/s/GPU).

So the rule for this model on this fabric: **VPP is worth enabling only when
microbatches per replica are low enough that the bubble dominates the added
peer-wait — at PP=8 the crossover sits between 16 and 32.** The shipped quickstart
sits on the wrong side of it, which is why the non-VPP config remains the default.

`configs/quickstart/nemotron_super_quickstart_sft_vpp4.yaml` (DELETED 2026-08-04 — VPP loses
at the standardised quickstart; reproduce with `model.virtual_pipeline_model_parallel_size=4`)
was the reproducible VPP
variant, carrying this verdict in its header.

## 5. What would change the answer

- **A cheaper PP crossing.** The cost is peer-wait, so anything that reduces stage
  imbalance (a better `pipeline_model_parallel_layout` for the hybrid's 2×-heavy MoE
  stages) attacks it directly, and helps the non-VPP config too.
- **Upstream fixing the overlap/deallocate interaction** (e.g. deferring
  `deallocate_output_tensor` until the send request completes) would make the overlap arm
  correct; whether it then beats the baseline is unknown, since its memory cost at 32K/CP4
  is what OOM'd.
- **A lower-microbatch regime** — see §4. Any config with ≤16 microbatches per replica at
  PP=8 should be re-measured with VPP=4 before assuming the default is best.

## Reproduction

Probe configs and the driver used for the ladder are recorded in the PR for INFR-71.
The full per-rank trace tables are in `infr71-vpp-vs-baseline-trace-analysis.md` in this
directory. Traces: `/projects/a5k/public/profiles/infr71/{vpp_v0_baseline,vpp_v3_vpp4}/`, each with
per-rank chrome traces, `provenance.txt`, and config snapshots. Trace analysis:
the tables in §3 come from interval-union arithmetic over the GPU-kernel events, splitting
SendRecv by process-group metadata to separate PP p2p from CP/EP all-to-alls (NCCL emits
no native AllToAll kernel — they are SendRecv too, which is easy to miscount).
