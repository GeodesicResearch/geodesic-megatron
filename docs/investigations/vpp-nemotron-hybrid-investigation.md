# Flexible Virtual Pipeline Parallelism (fVPP) on Nemotron-3 Super-120B

**INFR-68 bottleneck-resolution campaign, 2026-07-24.** All numbers measured on the
16-node / 64-GPU (GH200 95 GB) container environment, seq 32K, GBS 64, mean of
iterations 6–12 of 12-iteration probes, single job per allocation.

## Verdict

fVPP is **functional end-to-end** on the hybrid Nemotron-H architecture
(Mamba2 + attention + LatentMoE): VPP=4 trains cleanly (12/12 iterations, 0 NaN,
healthy loss, peak 93.8 GiB) at **32.39 s/iter**, and with fusion composed reaches
**30.69 s/iter — the best working VPP config, +10.1% vs the 27.88 s non-VPP
champion**. It is therefore **not a net speed win at this workload on Slingshot**:
the two admission costs VPP pays (extra recompute for its memory footprint, and
un-hideable pipeline p2p) exceed its bubble savings. The `overlap_p2p_comm` lever
that would recover the p2p cost **regresses ~2× on Slingshot/CXI** (root-caused
below). VPP remains valuable optionality for other regimes (shorter sequences,
other fabrics, future transport fixes).

## The unblock

fVPP was believed impossible on this model family. The block was **two stale
2025-08 asserts in our own bridge** (`src/megatron/bridge/models/mamba/
mamba_provider.py` and `mamba_builder.py`: "Virtual pipeline model parallelism is
temporarily unsupported in SSM/Mamaba models due to upstream MCore MambaModel API
dependency") that predated the current mcore pin (3758b54b2), which fully supports
hybrid fVPP. Fix (commit `4df5cc7e`): remove both asserts, forward `vp_stage` into
`MCoreMambaModel`, and change `pre_process or is_pp_first_stage(...)` to
`pre_process if pre_process is not None else ...` so interior virtual chunks
(explicit `False`) are not clobbered.

Mechanics: fVPP is driven by `|` separators in `hybrid_layer_pattern`
(mcore `mamba_hybrid_layer_allocation.py`, `PIPE = '|'`) which split the fixed
88-character layer-type string into PP×VPP segments
(`segment_index = vp*PP + pp_rank`). **The checkpoint binds layer types to
positions**, so a pattern may only move pipe boundaries, never reorder types. Do
not use the deprecated `hybrid_override_pattern` — setting both asserts.

## Attempt ledger (the memory ladder)

| # | Config | Result |
|---|---|---|
| a5 | VPP2, even-E 16-seg pattern, bucket 500M | OOM stage 0 (~91–94 GiB) |
| a6 | VPP2, same + bucket 100M | OOM — bucket is a grad-path lever; the peak is **activation** residency |
| a7 | **VPP4, stage-0-unloaded 32-seg pattern, trio recompute** | **OK: 32.39 s/iter, peak 93.8 GiB (stage 0 only 76.9)** |
| a8 | control: no-VPP, trio recompute | OK: 30.06 s/iter |
| a9 | PP16×VPP2/DP1 + overlap_p2p_comm | fits (91 GiB) but stuck ~77 s/iter — killed |
| a10 | VPP4 + overlap_p2p_comm | fits (92 GiB), 12/12 iters, **61.96 s/iter** (1.91× regression) |
| a13 | **VPP4 + trio + gradient_accumulation_fusion** | **30.69 s/iter — best working VPP** |

Earlier attempts (1–4) established the config-key collision
(`hybrid_layer_pattern` vs deprecated alias) and the VPP=2 memory wall.

## Lever table (all verified in code or measured)

- **Interleave residency ~(v+1)/v**: 1.5× at v=2 (measured +24 GiB → OOM at
  32K/CP4) vs 1.25× at v=4 (fits). Deeper VPP uses *less* activation memory.
- **Stage-0 unload is essential**: interleaved-1F1B residency peaks on stage 0,
  which also carries the embedding. The working pattern gives rank 0 nine layers
  with E=1 (`MEM|M*|M*|M*`); per-rank E = [1,6,6,5,5,6,5,6].
- **`recompute_granularity: full` is dead**: the hybrid MambaStack has no
  block-level checkpointing (zero recompute code in `megatron/core/ssm/`), and
  setting `full` disables the *selective* recompute consumers → strictly worse.
- **Selective recompute is exhausted at `[core_attn, moe, shared_experts]`**:
  there is no `mamba` recompute module; `moe_act` is already CPU-offloaded;
  `mlp`/`mla_up_proj` don't exist in this architecture.
- **`microbatch_group_size_per_vp_stage` < PP is illegal** (mcore assert), so 8
  is already the memory-optimal legal value.
- **`gradient_accumulation_fusion` composes better under VPP**: +1.70 s on the
  v=4 schedule vs +1.09 s on plain PP (more launch-gap exposure to erase).
  Container-only — the NGC image ships APEX; the bare-metal venv does not.

## Decomposition (from the a8 control)

```
27.88 = duo recompute + fusion          (non-VPP champion, pre-GC)
28.97 = duo, no fusion                  [fusion lever   = 1.09 s]
30.06 = trio, no fusion (control)       [core_attn tax  = 1.09 s]
30.69 = VPP4 + trio + fusion            [best VPP]
32.39 = VPP4 + trio, no fusion          [VPP schedule   = 2.33 s net]
61.96 = VPP4 + overlap_p2p_comm         [overlap regression]
```

VPP4 vs 27.88 = +2.81 s = 2.33 (exposed 4× PP p2p, bubble win already netted)
+ 1.09 (core_attn recompute, *required* for VPP's memory fit) − 0.61 (fusion's
extra effectiveness under the interleave).

## overlap_p2p_comm regression: root cause (trace-proven)

Matched profiled runs (torch profiler, `with_stack`, ranks 0 and 9, iteration 5)
of VPP4+overlap vs VPP4-plain. Per-rank window delta +32.1 s, reconciled to
< 0.01 s residual:

| Bucket | Δ rank 0 | Δ rank 9 | Share |
|---|---|---|---|
| `nccl_sendrecv` union (H1) | **+31.0 s** | **+31.6 s** | **~97%** |
| compute kernels (H2, SM contention) | +0.001 s | +0.005 s | ~0% (every category ≤1.01×) |
| idle/gaps (H3/4) | +1.15 s | +0.51 s | ~3% |

The op *count* barely changes (×1.07–1.10); the regression is **per-op
duration**: mean send/recv 5.9 ms → 13.7 ms (rank 0) and 1.8 ms → 10.1 ms
(rank 9). The un-batched individual isend/irecv path that `overlap_p2p_comm`
requires (`batch_p2p_comm: false` is mandatory, mcore mutual exclusion) is
intrinsically **2.3–5.7× slower per operation on Slingshot/CXI (aws-ofi-nccl)**
than the batched-coalesced exchange. The overlap machinery itself works — the
ops demonstrably run concurrently (sum ≫ union in the overlap trace) — but each
op is so much slower that hiding them cannot win. Same signature at PP16 (77 s)
and PP8 (62 s): fabric-level, not depth-level.

**Rule: `batch_p2p_comm=True` (the default) is load-bearing on Slingshot; do not
enable `overlap_p2p_comm` on this fabric.** There is no model-side gate — the
lever becomes interesting again only if the transport path changes (e.g., a
future aws-ofi-nccl with fast per-op p2p, or a different fabric).

Analysis artifacts: traces under `/projects/a5k/public/profiles/
{vpp4_overlap_regression,vpp4_plain}/`; analyzer + comparison outputs preserved
alongside the INFR-68 job records.

## Reproduction

The best-working-VPP configuration is committed as
`configs/quickstart/nemotron_super_quickstart_sft_vpp4.yaml` (VPP=4, 32-segment
stage-0-unloaded pattern, trio recompute, fusion, bucket 100M). It requires the
bridge fix (`4df5cc7e`) — on older checkouts the assert fires at model build.
