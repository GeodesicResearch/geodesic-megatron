# External training-stack review (2026-08) — item tracker

An external LLM-training consultant profiled one of our Super-120B runs and walked
through recommendations. This doc records EVERY technical item from that review and
tracks its investigation to a verdict on THIS stack (Nemotron-3 Super 120B-A12B,
NemotronH hybrid: 40 Mamba2 + 8 attention + 40 latent-MoE layers of 88 total;
GH200 4-GPU nodes, Slingshot-11/CXI; benchmark = 16 nodes / 64 GPUs,
TP1·CP4·EP4·PP8·ETP1·DP2, seq 32K, GBS 64, champion 21.78 s/iter).

Caveat applied throughout: the consultant profiled a ~3K-seq run (an EM/MQ config),
not the 32K benchmark; several of his memory/overlap observations must be re-verified
at the benchmark operating point before acting.

Status legend: OPEN → IN-PROGRESS → VERDICT (with evidence link).

| # | Consultant item | Status | Where |
|---|---|---|---|
| C1 | Optimizer CPU offload paging is not overlapped; drop offload, prefer deeper PP | VERDICT: right conclusion, wrong mechanism — offload-off measured Δ−2.07 s/iter same-nodelist; new champion posture | §C1 below |
| C2 | Distributed optimizer shards over DP (correction of our misunderstanding) | VERDICT: confirmed | §C2 below |
| C3 | Expert-with-context parallel folding "not on by default — enable it"; activation memory looked high | VERDICT: already folded; no flag missing | §C3 below |
| C4 | Recompute on FA-backed attention double-forwards; disable global recompute and audit | VERDICT: right on `core_attn` (measured 2× fprop, worth 0.036 s/attn-layer), wrong on global — MoE recompute is worth ~90 GB | §C4 below |
| C5 | DeepEP may run on Slingshot (verbs is software; kernels may resolve on SS) | VERDICT: blocked, and worthless if unblocked | §C5 below |
| C6 | Router/routing precision to FP8 ("loses nothing") | VERDICT: already FP32, and FP8 is not expressible | §C6 below |
| C7 | FP8 experts IF the base checkpoint's experts are already FP8 | VERDICT: premise wrong — pretrained NVFP4, FP8 sibling is inference PTQ | §C7 below |
| C8 | QK in FP8 (FA3), rest of attention sensitive | VERDICT: measured 0.73% of iter; max win 0.32%, under noise | §C8 below |
| C9 | Gradients BF16; leave optimizer states alone | VERDICT: already compliant | §C9 below |
| C10 | 400–550 TFLOP/s/GPU ceiling claim; "nothing in MoE prevents it" | VERDICT: not this architecture | §C10 below |
| C11 | 6ND FLOP approximation can be ~30% off on Mamba-heavy models; build exact tool | VERDICT: tool built; 6ND only ~5–10% off HERE | §C10 below |

Non-engineering meeting items (procurement/funder/organizational) are out of scope
for this repo and tracked outside it.

## C1 — optimizer CPU offload (VERDICT: drop it — right conclusion, wrong mechanism)

**Adopt `optimizer_cpu_offload: false` for the 64-GPU benchmark posture.** Measured
back-to-back on an identical nodelist, solo, allocation 5845744 (`q1_solo_*` logs in
`/projects/a5k/public/logs/infr71_wave2/`):

| arm | s/iter (mean 10–30, n=21) | sd | peak alloc | peak reserved | loss @32 |
|---|---|---|---|---|---|
| champion, offload 0.5 | 22.79 | 214 ms | 76.78 GB | 80.43 GB | 0.6872789 |
| **offload OFF** | **20.72** | 232 ms | 81.44 GB | 83.30 GB | 0.6874581 |
| delta | **−2.07 s (−9.1%)** | — | +4.66 GB | +2.87 GB | 1.8e-4 (parity) |

Both legs 32/32 iters, 0 NaN, override verified in each log's `OptimizerConfig` repr.
Memory cost landed within 4.6% of prediction (+4.66 GB measured vs +4.46 predicted) and
leaves **11.7 GB reserved headroom of 95**.

**The consultant was right to drop offload but wrong about why.** From the champion-era
traces (offload 0.5, ranks 0 and 9): the exposed cost is genuinely serialized at the end
of the iteration, with nothing to overlap it — but it is the **single-threaded host
AdamW** (877 ms wall, 9.8 ms of GPU work inside it), NOT paging. The D2H/H2D traffic is
**42 ms/iter (0.15%)** moving 4.15 GiB each way at 156–413 GB/s, i.e. at NVLink-C2C
speed. `overlap_cpu_optimizer_d2h_h2d` is working as designed: it gives the copies their
own CUDA streams so they overlap the GPU-side FusedAdam. It never claimed to prefetch
during backward — `step()` issues the D2H, blocks on `d2h_event.synchronize()`, then runs
the CPU optimizers in a sequential Python loop.

**Why the "offload-off OOMs" belief was wrong.** Offload moves only the BF16 moments; the
FP32 master params stay GPU-resident either way (`hybrid_optimizer.py::_get_sub_optimizer_param_groups`
clones to CPU but keeps `orig_param` live and writes back into it). Stake = **4 B/param**,
not the 8 B/param assumed. Confirmed twice: +3.79 GB measured going 1.0→0.5 (predicted
4.01), then +4.66 GB going 0.5→0 (predicted 4.46). The old OOM datum came from the
**DP=1 / 32-GPU** warm-start config, where the per-rank shard is 8× larger; it was never
measured at DP=2 / 64 GPUs.

**OPEN: the trace under-predicted the win by 2.2×.** Trace attribution said 0.88–0.96 s;
the same-nodelist measurement says 2.07 s. Two candidate explanations were tested and
**rejected**: per-PP-stage parameter counts are near-uniform (4.38–4.92 G, ~12% spread),
so it is not stage-imbalance skew; and iteration-time variance is identical between the
legs (sd 214 vs 232 ms, spread 952 vs 1024 ms), so it is not an occasional straggler tax
being charged at the barrier — the 2.07 s is a clean constant per-iteration offset. The
likely remainder is offload-path cost falling **outside** the `HybridDeviceOptimizer.step`
annotation (the `param_copy_back_gpu_hook` post-hook, the delayed param all-gather, and
the different PAO code path taken when `store_param_remainders` is live without offload).
Resolving it cleanly needs one profiled offload-off run A/B'd against the p1 champion
trace. **Until then, treat trace-derived optimizer-cost figures in this doc as lower
bounds** — including the ones feeding C1's sibling arms below.

**Composable lever, still worth having: `OMP_NUM_THREADS`.** torchrun defaults it to 1
(read directly from a live rank's `/proc/<pid>/environ`), so the host AdamW runs
single-threaded: ~31 GB of traffic in 874 ms = **35.7 GB/s**, about one Neoverse-V2 core,
while each rank owns a 72-core socket capable of ~500 GB/s. It is bandwidth-bound rather
than per-tensor Python overhead — throughput moves only 23% across a 163× change in
offloaded-tensor count (1.03 / 1.04 / 1.28 Mparams/ms at 651 / 321 / 4 tensors) — so it
should thread. **This no longer matters for the 64-GPU champion** (offload is gone) but
does for the configs where offload is mandatory and the shard is ~8× bigger:
`configs/pa_warm_start/sft_120b_1bmix_32k_pp11.yaml` (offload 1.0) and Ultra-550B.
Arm `infr71_q8_ompthreads_p1` measures it at offload 1.0. Plumbing verified: torchrun
only defaults when the var is absent, `pipeline_env_exec.sh` does not unset it, and
`apptainer exec` runs without `--cleanenv`, so a plain env prefix propagates.
`OMP_WAIT_POLICY=PASSIVE` is required — GNU OpenMP idle threads spin-wait, and this
workload is host-launch-bound.

**VPP-campaign confound (measured):** the VPP arm configs run `offload_fraction: 1.0` vs
champion 0.5 — worth **at least** 0.9–1.0 s/iter by trace, and by the 2.2× lesson above
plausibly more. Subtract before reading any VPP-vs-champion delta.

**Placement note banked by the pair:** champion-0.5 measures 22.79 on this nodelist vs
21.78 on allocation 5738452 — this placement is ~1.0 s worse. Cross-allocation
comparisons of this workload remain untrustworthy (~2.7 s/iter Dragonfly swing, per the
quickstart header); the −2.07 s above is same-nodelist and therefore placement-free.
Projected onto a 5738452-class placement, offload-off lands near **19.7 s/iter**.

Caveat on the fraction dial at cutlass: `build_cpu_optimizer_list` makes one CPU AdamW per
param tensor and cutlass fuses experts into ~4 giant tensors (651 CPU optimizers
pre-cutlass → 4), so `fraction: 0.5` actually offloads **59.6%** and 0.25 may be
unreachable. Moot for the champion now, relevant to the offload-mandatory configs.

## C2 — optimizer sharding (settled)

Confirmed: `use_distributed_optimizer: true` shards optimizer state across the DP
axis (mcore distributed optimizer = ZeRO-1-style; gradient buckets reduce-scattered,
FP32 master + moments partitioned over DP). At the benchmark's DP=2, each rank holds
half the optimizer state of its PP-stage's parameters. This matters for C1's fit
arithmetic: with precision-aware BF16 moments (2+2 B) + FP32 master (4 B) ≈ 8 B/param
optimizer-side, halved by DP=2, on top of BF16 params+grads.

## C3 — expert-with-context parallel folding (settled: already active)

**Verdict: parallel folding is active in every config we run, in exactly the sense the
consultant means. There is no flag we are missing.** His "not on by default" is a true
statement about Megatron in general — `expert_tensor_parallel_size` defaults to
`tensor_model_parallel_size` (`3rdparty/Megatron-LM/megatron/core/parallel_state.py:781-782`),
which un-folds the grid — but every config here sets `expert_tensor_parallel_size: 1`
explicitly.

### Group-construction evidence

mcore builds two independent rank grids, and they are structurally exclusive
(`parallel_state.py:452-455`: "Both EP and CP > 1 is not allowed in one rank generator"):

- attention grid (`parallel_state.py:770-778`): `tp=TP, ep=1, dp=world//(TP·PP·CP), pp=PP, cp=CP`
- expert grid (`parallel_state.py:793-801`): `tp=ETP, ep=EP, dp=world//(ETP·EP·PP), pp=PP, **cp=1**`

EP process groups come from the *expert* generator (`parallel_state.py:1173-1182`); the only
cross-grid constraint is PP (asserted at `parallel_state.py:809-812`), which is exactly the
paper's "sole constraint" (`.claude/skills/megatron-moe-paper` §3.3.1). The live call site is
`src/megatron/bridge/training/initialize.py:679-698` with `order="tp-cp-ep-dp-pp"`
(`use_decentralized_pg` defaults False, `src/megatron/bridge/training/config.py:130`).

Running that code verbatim over the benchmark topology (world 64, TP1·CP4·EP4·ETP1·PP8 →
DP=2, EDP=2) gives:

```
attention ordered_size [tp=1, cp=4, ep=1, dp=2, pp=8]
expert    ordered_size [tp=1, cp=1, ep=4, dp=2, pp=8]
ATT cp : [[0,1,2,3], [4,5,6,7], [8,9,10,11], ...]   EXP ep : [[0,1,2,3], [4,5,6,7], ...]
EP set == CP set: True | EP set == DP set: False | EDP set == DP set: True | EP node-local: True
```

The 4 CP ranks of a node **are** the 4 EP peers. Attention `TP×CP = 1×4` folds onto expert
`ETP×EP = 1×4`, so EDP == DP == 2.

Settled independently of any code reading: `TP·CP·DP·PP = 1·4·2·8 = 64` = our world size. An
un-folded EP would require `TP·CP·EP·DP·PP = 256` GPUs. We run 64, so the fold is forced.

A second live signature is the expert gradient scaling
(`3rdparty/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py:197-216`): with
`average_in_collective: false` both factors are `1/dp_cp_group.size()` = 1/8, while expert
grads reduce over `intra_expt_dp` (2 ranks) and dense grads over `intra_dp_cp` (8 ranks).
Expert grads need only a 2-way reduce because the CP dimension already summed through the EP
all-to-all. Un-folded, EDP would equal dp_cp and both would be 8-way.

Counterfactual: on the MQ/EM topology (world 32, TP4·CP1·EP4·PP8), letting ETP default to
TP=4 gives `expert_data_parallel_size = 32/(4·4·8) = 0` and mcore raises at
`parallel_state.py:787-790`. That config **cannot be constructed** without ETP=1.

Empirical: `> resolved parallelism: world_size=64 | DP=2 TP=1 PP=8 CP=4 EP=4` (printed from
the groups actually built — `src/megatron/bridge/training/utils/parallelism_utils.py:57-66`)
appears in 8 independent 16-node profile runs under `/projects/a5k/public/profiles/`
(`hostrc_p1_champ_profile`, `hostrc_m6p_cutlass_profile`, `hostrc_m1p_2604_profile`,
`r9_nonvpp`, `r9_vpp4_overlap`, `infr71/vpp_v0_baseline`, `infr71/vpp_v3_vpp4`,
`quickstart_normal_run`). mcore never prints rank lists, so membership is inferred from the
×4 arithmetic above; node-locality assumes the standard 4-ranks-per-node block assignment and
is corroborated by the absence of the documented cross-node EP all-to-all Slingshot hang.

### Activation memory — real, expected, and mostly a different config's problem

Measured decomposition, champion 32K run (`hostrc_p1_champ_profile/…/raw_log_snapshot.out`):

| rank class | persistent | peak | transient ≈ activations |
|---|---|---|---|
| heaviest MoE stage | 40.19 GB | 76.71 GB | ~36.5 GB |
| lighter stages | 36.84–36.87 GB | 46.4–68.9 GB | ~10–32 GB |
| `quickstart_normal_run` (offload 1.0) | 27.93 GB | 73.28 GB | ~45 GB |

Roughly half of peak being activations is expected on this architecture:

1. **Top-k 22 dispatch transient.** 512 routed experts, top-22, `moe_latent_size=1024`,
   `moe_ffn_hidden=2688`, no GLU (verified against upstream `config.json` and the resolved
   snapshot). At 8192 tok/rank a MoE layer permutes `T·k = 180,224` token-copies: latent
   buffers `180,224×1024×2 B = 369 MB` each, fc1-out/act-out `180,224×2688×2 B = 969 MB`
   each ⇒ **~2.7 GB live inside one MoE layer** (arithmetic, not measured). Neither
   TP-shardable (ETP=1 by design) nor CP-shardable (CP already divided the tokens).
2. **1F1B residency is PP-invariant.** Stage 0 holds 8 in-flight µb × 11 layers = 88
   layer-µb; the stored layer input alone is 67 MB ⇒ 5.9 GB floor.
3. **Measured slope** (quickstart header ladder): recompute `[moe, shared_experts]` OFF →
   OOM >95 GB (worth ≥24 GB); CP=2 at 16K tok/rank → OOM 93.6–94.7 GB.

**What he most likely saw.** He profiled an MQ/EM run, not the benchmark
(`configs/misalignment_quarantine/nemotron_120b_sem_proc/em/mqv2_…_shakespearean_mqip.yaml:69-91`,
e.g. job 5843658): TP4·CP1·EP4·ETP1·PP8, seq 8192, GBS 4. It sets
`recompute_modules: ["core_attn"]` — **MoE and shared-expert activations are not recomputed
at all**, and TE warns in that very log (`logs/slurm/train-5843658.out:1127`, from
`transformer_config.py:1786`) that `core_attn` recompute is redundant with fused attention,
so those runs have effectively no useful recompute. `offload_modules: [expert_fc1, moe_act]`
is listed but inert because `fine_grained_activation_offloading: false`. That is a
config-level activation-retention difference, unrelated to folding — which is active in that
config too (EP groups = TP groups = node-local).

### Levers actually left

Against the paper's memory table (Table 13) we already run selective recomputation,
precision-aware optimizer, optimizer offloading, and memory-efficient permutation
(`moe_permute_fusion: true`); FP8 is blocked by MoE-routing crashes. Untried:

1. `fine_grained_activation_offloading: true` — the only unused Table 13 row. Needs a trimmed
   `offload_modules`: the 0.19 pin asserts against offloading `expert_fc1`/`moe_act` while
   `moe` is recomputed. Overlaps arm Q3.
2. `moe_expert_capacity_factor` (currently `null` = dropless) — the closest thing to the
   "capacity" flag he may mean. Bounds the top-22 transient deterministically but drops
   tokens, so it is a quality trade, not free.
3. **For the MQ/EM arms only** (not the benchmark): add `moe`/`shared_experts` to
   `recompute_modules` and drop the no-op `core_attn`. Highest-value change implied by his
   observation, and quality-neutral.

Reproduction harness for the group derivation: a scratch script holding the two mcore
functions extracted verbatim — fully re-derivable from the `parallel_state.py` lines
cited above (run `RankGenerator` for both grids at world 64, TP1·CP4·EP4·ETP1·PP8).
No new training run was needed.

**Header correction made from this evidence:** the layer census at the top of this doc read
"44 Mamba2 + 4 attention"; the actual `hybrid_layer_pattern` (identical in the upstream HF
`config.json` and our resolved-config snapshot) is 88 chars = **40 M / 40 E / 8 attention**.
This doubles the attention-layer share that C8 (QK-FP8) can address — still small, but the
bound should be computed at 8/88, not 4/88.

## C4 — recompute / SAC double-forward (settled on mechanism; arms queued for the fit probe)

**Split verdict: he is right about `core_attn` and wrong about "disable recompute globally".**
The two halves of the recommendation are different mechanisms and only one of them is what he
described.

### The attention half is correct — and upstream says so itself

`attention.py:391-394` sets `checkpoint_core_attention`; `attention.py:1518-1527` routes to
`_checkpointed_attention_forward`, which at `attention.py:493` calls
`tensor_parallel.checkpoint(...)`. That wrapper (`tensor_parallel/random.py:555-634`) runs the
region under `torch.no_grad()` (:580), saves only the inputs (:592), and re-runs the whole
region under `enable_grad` in backward (:620-621). The region is `TEDotProductAttention` — the
cuDNN fused path — so the flash forward does run a second time, on top of flash's own
recompute-from-LSE in its backward. Megatron warns about exactly this at
`transformer_config.py:1785-1791` ("For fused attention, you have no need to set 'core_attn' to
recompute"), and that warning is in our own logs.

**Measured, from `/projects/a5k/public/profiles/infr71/`**, normalised per attention layer
(baseline arm has no `core_attn` recompute, VPP arm has it):

| kernel | without `core_attn` | with `core_attn` |
|---|---:|---:|
| `cudnn ...sdpa flash_fprop` | 128 / 0.0372 s | **256 / 0.0746 s** |
| `cudnn ...sdpa flash_bprop` | 128 / 0.0984 s | 128 / 0.0965 s |
| all attention kernels | 1152 / 0.1445 s | 1664 / 0.1805 s |

Forward count doubles exactly, backward is untouched: the double-forward, isolated.
(128 = 32 microbatches × 4 CP ring steps — the recompute re-runs all four ring steps and
re-issues their P2P KV exchanges.) **Cost +0.036 s per attention layer per iteration.**

What it buys is ~nothing. TE saves `(q, k, v, out)` + `softmax_lse`
(`transformer_engine/pytorch/attention/dot_product_attention/backends.py:1387-1395`; CP-P2P
variant `context_parallel.py:1988-1999`). q/k/v are the checkpoint's own inputs so they stay
resident either way, and `out` stays alive because `linear_proj` saves it for its wgrad. Only
the LSE (1×32×8192 fp32 = 1.05 MB), TE's internal stacked-KV copy (~8.4 MB) and rng state are
freed — **~9.5 MB per attention layer-microbatch, ~76 MB per GPU**, 0.08% of the card.

**Magnitude check:** total attention GPU time is 0.145 s of a 9.24 s compute union — 1.6% of
compute, ~0.5% of wall. Deleting attention outright would save ~0.15 s. No attention-side lever
reaches the 27→20 s gap. The July "+0.5 s/iter trio-vs-pair" figure cannot be `core_attn`
compute; it was noise or confounded (the VPP arms also differ in layer pattern, `batch_p2p_sync`
and `timing_log_level` — `infr71-vpp-vs-baseline-trace-analysis.md:313`).

### The MoE half is a legitimate, load-bearing save

`moe_layer.py:685-699` wraps `custom_forward` (:642-683) — the **entire** MoE forward: router,
permute, EP all-to-all dispatch, grouped fc1/act/fc2, combine, latent projections. Only the
layer's bf16 input survives. No hidden second recompute: neither TE `GroupedLinear` nor our
`CutlassGroupedExperts` recomputes internally, and the one internal option (`moe_act`,
`experts.py:259-266`, ported at `cutlass_grouped_experts.py:107-216`) is opt-in and off.

Cost, from the census arithmetic (the recompute pass is literally one of the four passes in
`128 experts × 2 proj × 5 layers × 32 µb × 4 = 163,840` launches): **~1.25 s** of expert GEMM
re-forward, **~0.32 s** of re-run dispatch, plus one extra EP all-to-all dispatch+combine per
MoE layer-microbatch — roughly **2 s of a 21.78 s iteration (~9%)**.

Value: at 8192 tok/rank the top-22-of-512 dispatch materialises T·k = 180,224 token-copies per
MoE layer (dispatched latent 369 MB + fc1 out 969 MB + act out 969 MB ≈ **2.3 GB per MoE
layer-microbatch**). 1F1B at PP=8 holds 8 in-flight µb × 5 MoE layers = 40 layer-microbatches
⇒ **~90 GB of new activation** on a 95 GB card. The July datum ("recompute OFF → OOM ~95 GB
peg") was only ever a **≥24 GB lower bound** because it OOM'd from a 70.5 GB baseline and could
not report the overshoot. Nothing since has changed it: CUTLASS moved host time, not activation
residency. Finer granularity does not rescue it either — `moe_act` alone still leaves
~1.34 GB/layer-µb ≈ 53 GB.

### A third recompute nobody was counting: the Mamba scan

`pipeline_training_launch.sh:474` exports `ISAMBARD_FP32_SSM_STATE=checkpoint` **by default**,
and `pipeline_training_patches.py:197-206` wraps the fp32 scan in a non-reentrant
`torch.utils.checkpoint`. Every launcher run therefore recomputes the scan on all 40 Mamba
layers. Kernel multiplicities on rank 0 (5 Mamba layers × 32 µb = 160 layer-microbatches) show
it cleanly: `_chunk_scan_fwd` 320 = **2×**, `_chunk_state_fwd` / `causal_conv1d_fwd` /
`_state_passing_fwd` / `_bmm_chunk_fwd` / `_chunk_cumsum_fwd` all 480 = **3×**, while every
`*_bwd_*` kernel is exactly 160 = 1×. The mamba_ssm kernel already recomputes internally in its
backward (that is the 2× baseline, same design as flash attention); our patch stacks a second,
torch-level recompute on top. **This is precisely the pattern the consultant described — just on
the Mamba layers, not the attention layers.** Cost ≈ **0.30-0.39 s/iter**.

It is not a mistake: checkpoint mode holds only the bf16 `zxbcdt` (304 MB/layer-µb, 12.2 GB on
stage 0); direct fp32 would hold 876 MB/layer-µb (+23 GB, infeasible); and disabling fp32
entirely holds 438 MB/layer-µb (+5.3 GB) while reintroducing the field-confirmed long-document
NaN (`pipeline_training_run.py:293-306`). So it is *cheaper in memory than not patching at all*.

**Recommendation — document the launcher default as 32K-only; do not change it.** At the 32K
benchmark operating point `ISAMBARD_FP32_SSM_STATE=checkpoint` must stay: bf16 inter-chunk SSM
state NaNs deterministically once a single document integrates ~32K tokens (field-confirmed at
TP1·CP4·EP4·PP22 — 270 healthy iterations, then a step-function NaN at iter 272 on a long-doc
batch), so the fp32 state is mandatory and the checkpoint is what makes it memory-neutral. But
the launcher exports it unconditionally at `pipeline_training_launch.sh:474`, so **every 8192-seq
run inherits it too** — the MQ/EM arms and any 8K Ultra SFT — where CLAUDE.md already records
fp32 SSM state as unnecessary ("Unnecessary at 8K"). Those runs pay one extra Mamba scan forward
across all 40 Mamba layers, ≈1.4% of iteration time, for a numerical guard their sequence length
does not need. The fix is per-run (`ISAMBARD_FP32_SSM_STATE=0` for 8K configs) plus a line in
CLAUDE.md scoping the default to 32K; changing the launcher default itself is a **convention
change requiring Kyle's approval** and is deliberately not proposed here. Note the saving is not
free memory-wise either: disabling the patch costs +5.3 GB on stage 0 (438 vs 304 MB/layer-µb),
which the 8K configs have room for but which should be checked per topology rather than assumed.

### Nothing else adds a hidden forward

Full recompute is gated strictly on `== 'full'` (`hybrid_block.py:312`,
`transformer_block.py:624`); ours is `selective`. `cpu_offloading` is provably off —
`transformer_config.py:1700-1703` raises if combined with any recompute granularity, and our
configs build. `distribute_saved_activations` defaults False and is barred under sequence
parallelism (`:1737-1742`). `recompute_method`/`recompute_num_layers` unset everywhere.
`NVTE_CPU_OFFLOAD_V1=1` (`pipeline_env_activate.sh:166`) is inert without an active offload
context. `ISAMBARD_MAMBA_SAVE_OFFLOAD` defaults to `0` in the real entry point
(`pipeline_training_run.py:323`) — the `1` at `pipeline_training_patches.py:855` is only the
self-test `__main__`.

### Two defects found en route

- **`configs/quickstart/nemotron_super_quickstart_sft_vpp4.yaml` could not run at this pin** —
  FIXED in `013a5f04`. It set `fine_grained_activation_offloading: true` with
  `expert_fc1`+`moe_act` in `offload_modules` while `moe` is recomputed, which
  `transformer_config.py:1855-1865` asserts against; and being a VPP config it would also have
  hit the `Chunk mismatch` assert. It now sets the flag `false` with the trimmed list, matching
  `configs/infr71_vpp/a{0,1,2}`. This mattered because CLAUDE.md names it as THE VPP variant to
  run. The same commit moves `a{0,1,2}` off the CLI override they were being launched with, so
  the wave-2 numbers are now reproducible from config alone.
- **`core_attn` in `offload_modules` is dead whenever `core_attn` is also recomputed**
  (`attention.py:1515-1527`: the offload context is built but only entered in the
  non-checkpointed `else` branch). Every trio config has both.

### Recommendations

| config | change |
|---|---|
| champion `nemotron_super_quickstart_sft.yaml` | none — `["moe","shared_experts"]` is the fastest feasible point |
| VPP arms (`infr71_vpp/a{0,1,2}`, vpp4 quickstart) | drop `core_attn`; also fix the vpp4 offload assert |
| `nemotron_ultra_quickstart_sft.yaml:121` | drop `core_attn`; keep `moe`/`shared_experts` |
| MQ/EM + `nemotron_warm_start_200k/*` (`["core_attn"]` only) | → `["moe"]`: they pay the redundant attention recompute AND retain every MoE activation |
| runs at seq 8192 (Ultra, MQ/EM) | `ISAMBARD_FP32_SSM_STATE=0` — CLAUDE.md already says fp32 SSM is unnecessary at 8K; worth ~1.4% |

Also note `shared_experts` is nearly a no-op: `shared_experts_compute` is called *inside* the
`moe` checkpoint (`moe_layer.py:645`) with no nesting guard, so with both set the shared expert
runs **three** forwards and saves only the ~176 MB of intermediates the outer recompute would
otherwise rebuild at once. Measured as a tie (28.97 / 70.5 GB vs 29.11 / 70.7 GB). Not worth
changing, but not worth defending either.

### Arms authored (validated, not yet submitted)

All three pass `TransformerConfig` validation at this pin; the upstream `core_attn` warning
fires for A0 and is absent for q3c.

- `configs/infr71_vpp/q3a_recompute_off.yaml` — recompute off, offload fraction 1.0 for maximum
  headroom. **Preregistered to OOM in the first forward** (~90 GB short); it is a falsification
  probe, and cheap because that is how it dies.
- `configs/infr71_vpp/q3b_recompute_off_offload.yaml` — recompute off **plus** fine-grained host
  offload of `expert_fc1`/`moe_act`, which is illegal while `moe` is recomputed and so has never
  been compared head-to-head. The code path only became live for this model at the 0.19 pin
  (`fine_grained_activation_offloading` was a silent no-op on Nemotron-H before —
  `hybrid_model.py:351-360` wires it now), **but it is also broken**: our own merged
  host-overhead campaign records `AssertionError: Chunk mismatch` in
  `fine_grained_activation_offload.py` firing under **plain PP=8, not only VPP**, OPEN upstream
  with no commits to that file since our pin (`120b-gbs64-host-overhead-investigation.md:104`
  and bug #5 at `:151`). So the strong form of the consultant's recommendation is not currently
  testable on this stack. The arm stays preregistered because it is cheap and because a clean
  reproduction under `recompute_granularity: null` — a configuration bug #5 was not observed in,
  since offload-inside-recomputed-MoE is rejected at config time — is worth putting on the
  upstream issue.
- `configs/infr71_vpp/q3c_vpp4_no_core_attn.yaml` — A0 with one field changed, to score the
  +0.036 s/attn-layer prediction against a placement-matched A0 (A0's stage-0-lite pattern puts
  3 attention layers on stage 0, the largest signal in the campaign).

Trace-analysis scripts: session scratch (ephemeral); the measured numbers above and the
trace files under `/projects/a5k/public/profiles/` are the durable record.

## C5 — DeepEP on Slingshot (settled: hard-blocked, and no upside if unblocked)

**Two independent verdicts, either one sufficient.** (1) DeepEP's internode path needs
IBGDA, which is a Mellanox **mlx5 hardware register** dependency, not a verbs-API
dependency — Cassini cannot satisfy it. (2) Even granted a working port, DeepEP's win is
*internode* RDMA dispatch; our EP=4 is node-local NVLink, and DeepEP's best-case internode
number is already slower than the NVLink path we run today. This is a **do-not-pursue**,
not a "someday".

### The blocker, from primary source inside our own container

The image ships `deep_ep 1.2.1+34152ae` **with full source** at `/opt/DeepEP`.
`csrc/kernels/ibgda_device.cuh` builds Mellanox mlx5 work-queue entries *in device code*
— `mlx5_wqe_ctrl_seg` (L63), `mlx5_wqe_raddr_seg` (L175), `MLX5_OPCODE_RDMA_WRITE` (L193),
`MLX5_SEND_WQE_SHIFT` (L264) — reads `nvshmemi_ibgda_device_state_d`, and rings the NIC
doorbell from the SM. That is the ConnectX byte-layout ABI emitted from a CUDA kernel;
there is no fabric-neutral abstraction to re-target.

**It is not low-latency-only** (several secondary sources, incl. DeepWiki, say it is —
wrong for this version): `grep -l ibgda_device.cuh csrc/kernels/*.cu` returns
`internode.cu` (high-throughput) *and* `internode_ll.cu`. The IBRC fallback for normal
kernels that DeepEP's lead author described in [issue #36](https://github.com/deepseek-ai/DeepEP/issues/36)
is gone by 1.2.1.

NVIDIA gates the mechanism in one line — [NVSHMEM docs](https://docs.nvidia.com/nvshmem/api/using.html),
GDAKI/IBGDA prerequisites: **"Only Mellanox HCAs and NICs."**

Host-side confirmation on `nid010992`: `/sys/class/infiniband/` **does not exist**;
`/sys/class/cxi/` has `cxi0..cxi3`; `ib_core`/`ib_uverbs`/`rdma_rxe` are absent from disk
(not merely unloaded). The Cassini driver lives in the Linux **Ethernet** subtree
(`drivers/net/ethernet/hpe/ss1/`, [shs-cxi-driver](https://github.com/HewlettPackard/shs-cxi-driver)),
so it never registers an RDMA device.

### Where the consultant is right, and where the claim breaks

"Verbs is software, not hardware" is true — and irrelevant. DeepEP does not depend on the
verbs *API*; it depends on the mlx5 WQE format and a doorbell MMIO page mapped into GPU
address space. No shim can synthesize a register that does not exist.

Two real facts likely behind the claim, neither of which helps:
- **Slingshot-10 genuinely spoke verbs** — it used Mellanox ConnectX-5 NICs (GASNet
  ofi-conduit README lists SS-10 as `verbs;ofi_rxm`, SS-11 as `cxi`). The NIC vendor
  changed under the product name. We are SS-11/Cassini.
- **HPE does ship a Cassini SoftRoCE driver** ([shs-rxe](https://github.com/HewlettPackard/shs-rxe)),
  which would create a real `/sys/class/infiniband/` device. Its documented purpose
  (CUG 2023, HPE) is a kernel-space Lustre `ko2iblnd` bridge to Mellanox-based storage;
  rxe runs the transport in a CPU kernel thread, so it has no doorbell to map. Not
  installed here regardless.

Also worth correcting: "DeepEP was made for Hoppers" — **GH200 *is* Hopper (sm_90)**. The
GPU is not the blocker; the NIC is.

And "RoCE works, so non-IB fabrics work" does not follow: RoCE changes the *link* layer,
not the transport. Every working and failing RoCE report is ConnectX-6/7 or BlueField-3,
and the failures read `GPU cannot map UAR of device mlx5_0`. Maintainer `sphish` on
[#369 (DeepEP over EFA)](https://github.com/deepseek-ai/DeepEP/issues/369): *"IBGDA does
not support EFA. Therefore, adding EFA support would be hard. I suggest using
pplx-kernels as an alternative."* Tracker searches for `Slingshot`/`Cassini`/`Cray`/
`libfabric` return **zero** hits — nobody has ever attempted this.

### The concession worth making: NVSHMEM *does* run on Slingshot

Worth conceding at the meeting, because it is the strongest form of the consultant's
position. NVSHMEM's libfabric transport has `NVSHMEM_LIBFABRIC_PROVIDER` defaulting to
literally **`cxi`** ([env docs](https://docs.nvidia.com/nvshmem/api/gen/env.html)), Slingshot-11
is a named supported network in the install guide, our image carries
`nvshmem_transport_libfabric.so.3` with CXI strings (`FI_CXI_OPTIMIZED_MRS`), and CSCS runs
**this exact NVSHMEM 3.4.5** in production on Alps (same GH200+Cassini) with
`NVSHMEM_IBGDA_SUPPORT=0`. Device-side NVSHMEM calls do work there.

But that transport is **CPU-proxy-mediated**, which is the very thing DeepEP exists to
eliminate — and DeepEP bypasses the transport abstraction anyway. Perseus
([arXiv:2605.00686](https://arxiv.org/html/2605.00686)) states it: *"The GPU-direct
submission path requires NIC hardware support that is absent on several network fabrics,
including AWS EFA, HPE Slingshot-11, and Broadcom-based cloud deployments."* CSCS's
measured NVSHMEM-over-CXI throughput on identical hardware: **~22.9 GB/s** at 4 MB.

### Mechanism update: DeepEP V2 dropped NVSHMEM — still blocked

Our recorded belief was right in conclusion but is going stale in mechanism. DeepEP `main`
(commit `b306af06`, 2026-04-29) shipped EPv2, which *"switched from the NVSHMEM backend to
the more lightweight NCCL Gin backend"*; the IBGDA files now sit under `csrc/kernels/legacy/`.
Still blocked for us on three counts:

1. NCCL's built-in GIN requires **NVIDIA NICs, CX4+, rdma-core**
   ([Device API docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)).
2. The external `ncclGinPlugin_v14` path exists (aws-ofi-nccl master implements it) but its
   GDAKI backend is EFA-hardcoded (`fi_ext_efa.h`, `FI_EFA_GDA_OPS`), appears in no tagged
   release, and **our build exports only `ncclNetPlugin_v2..v11`** (`nm -D` on
   `/projects/a5k/public/containers/slingshot/nemo_26.04/aws-ofi-nccl/lib/libnccl-net.so`;
   we are on aws-ofi-nccl **v1.18.0**).
3. Version floor: DeepEP V2 needs **NCCL ≥ 2.30.4**; our Slingshot build is **v2.29.2-1**.

The generic *proxy* GIN path is provider-agnostic in its guards but explicitly refuses
`FI_MR_ENDPOINT`, which CXI appears to require — and [NCCL #1913](https://github.com/NVIDIA/nccl/issues/1913)
(open, no NVIDIA response) reports GIN silently self-disabling whenever an external net
plugin is loaded, i.e. exactly our configuration.

### The other two flex-dispatcher backends

Our pin exposes three at `3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py:859`.

- **`hybridep` — blocked, same root cause.** Not an independent NVIDIA library: it is a
  *branch of DeepEP* (`deepseek-ai/DeepEP/tree/hybrid-ep`), and Megatron's MoE README calls
  it *"NVIDIA's optimized dispatcher using TMA and IBGDA."*
  [#659](https://github.com/deepseek-ai/DeepEP/issues/659) shows its NIXL connector
  hardcodes UCX + `mlx5`/`ibverbs`. Targets GB200 NVL72.
- **`ncclep` — the only one NOT fabric-blocked.** It binds TransformerEngine's NCCL EP
  (`transformer_engine.pytorch.ep`), built on NCCL's Device API with **no NVSHMEM**;
  intranode it uses LSA (NVLink load/store) and never touches the network, so **node-local
  EP=4 has no fabric dependency at all**. Our blocker is a version gap:
  `fused_a2a.py:641` demands `NVTE_BUILD_WITH_NCCL_EP=1`, and the image has **TE 2.14.1**
  (verified: `from transformer_engine.pytorch import ep` → `ImportError`) where **TE ≥ 2.17**
  is required. Even then `moe_ncclep_static_shape` is hard-gated to **sm100+**
  (`token_dispatcher.py:1466`), so the comm/compute-overlap benefit is unavailable on GH200.

### Benefit bound — why this is moot even in the counterfactual

DeepEP's value is fast *internode* dispatch. From its own README benchmark table, same GPU
generation: best internode on **CX7 400 Gb/s** is **90 GB/s** dispatch, versus intranode
NVLink at **~153–160 GB/s** (secondary sources; the current README dropped the SM90
intranode row). DeepEP's best case on hardware far better than ours is **already slower
than the NVLink path we run today**. Slingshot-11 is 200 Gb/s/NIC (~25 GB/s), and measured
NVSHMEM-over-CXI is 22.9 GB/s. Moving EP off-node to chase DeepEP is a large regression
under any backend.

Residual upside is confined to *intranode kernel efficiency* (DeepEP/HybridEP dispatch in
6–24 SMs vs a generic all-to-all) — not bandwidth. Behind a TE image bump, with the overlap
feature Blackwell-gated, that ranks below the C1/C4 levers.

Published precedent agrees with where we already are: the two papers running MoE expert
parallelism on Slingshot — **X-MoE** on Frontier ([arXiv:2508.13337](https://arxiv.org/html/2508.13337v1))
and **RailX** on 64× GH200/SS-11 ([arXiv:2507.18889](https://arxiv.org/pdf/2507.18889)) —
both use portable NCCL collectives, and X-MoE independently arrived at a hierarchical
node-local dispatch design. No GPU-initiated MoE dispatch on Slingshot exists publicly.

## C6/C7/C8 — the precision menu (settled: decline all three)

The three FP8 items share one refutation, so it is stated once here before the
per-item detail. **This run is host-launch-bound, not tensor-core-bound.** The GPU is
idle ~50% of every iteration (§1 of `120b-gbs64-host-overhead-investigation.md`), and
the −16% champion win came from collapsing per-expert launches, not from doing GEMM
math faster. Every FP8 recipe *adds* quantization kernels, which the MoE paper itself
flags as "particularly problematic for fine-grained MoE where many small operations
already stress the CPU" (`.claude/skills/megatron-moe-paper/megatron-moe-paper.md:733`).
Our own history is the same finding from the other side: per-tensor FP8 measured
**−28%** (`configs/sfm/sfm_probe_blockwisefp8.yaml:56`), and blockwise FP8 — the
paper's recommended Hopper recipe — managed only 425 vs 403 TFLOP/s in a *1-node*
smoke (`configs/sfm/sfm_nemotron_120b_cpt_misalignment_fp8container.yaml:10-11`),
before the host-bound 16-node regime asserts itself.

### C6 — router precision (VERDICT: already done, and the proposal inverts it)

**We already run the router in FP32.** It is set by the provider, not the YAML — which
is why grepping `configs/` finds nothing: `moe_router_dtype: str = "fp32"` at
`src/megatron/bridge/models/nemotronh/nemotron_h_provider.py:52`, repeated at
`nemotron_h_bridge.py:312`.

**FP8 routing is not expressible at any pin.** The field is typed
`Optional[Literal['fp32','fp64']]`
(`3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py:780`). The only
supported direction is *up*, deliberately: the paper's Principle #1 of selective
precision is "Protect routing decisions. The router remains in FP32 to ensure stable
expert selection", and it names our exact risk — "Quantization noise could destabilize
expert selection, leading to training instability, degraded model quality, or expert
collapse" (§5.1). mcore warns for ≥32 experts without fp32 routing
(`transformer_config.py:2752-2760`); we have **512** routed experts, so that warning
targets us and we are already compliant.

**FP32 routing is nearly free, so there is nothing to reclaim.**
`router_gating_linear` keeps input and weight in BF16 and asks only for an FP32
*output*: `te_general_gemm(weight, inp, router_dtype, layout="TN", bias=bias)`
(`megatron/core/transformer/moe/moe_utils.py:1326-1327`), documented as "accepts
bfloat16 input and weight, and can return output with router_dtype" (`:1379-1383`).
The forward is therefore already a BF16 tensor-core GEMM with an FP32 epilogue — the
native accumulate format. Only the backward casts (`:1362-1368`).

Bound: the router gating GEMM is **1.3% of expert-GEMM FLOPs, ≤0.15% of the 21.78 s
iteration even at 100% of peak** (16.5 TFLOP/rank/iter fwd+bwd vs 1270 for the expert
GEMMs). Corroborated by an existing measurement: `moe_router_fusion`, which fuses the
*entire* router, scored 27.85 vs 27.61 s/iter — no win, excluded (quickstart header).
Fusing all of it moves nothing, so quantizing part of it cannot.

### C7 — FP8 experts on an "already-FP8" checkpoint (VERDICT: premise is wrong)

**Super-120B was pretrained in NVFP4, not FP8** — arXiv 2604.12374 §2.2: "We
pre-trained Nemotron 3 Super in NVFP4"; "All linear layers, unless otherwise noted in
Table 3, are trained using the open-source NVFP4 GEMM kernels provided by Transformer
Engine", quantizing weights, activations and gradients, stable to 25T tokens. The FP8
artifact the consultant likely has in mind is
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`, which is **post-hoc ModelOpt PTQ for
Hopper inference** (§4: "post-training quantization (PTQ)… FP8 (W8A8) for Hopper and
NVFP4 (W4A4) for Blackwell"). Verified locally: that repo's `config.json` carries a
`quantization_config` with `quant_method: "modelopt"`; ours carries none and its
safetensors are 100% BF16.

Two further disqualifiers:

- **NVFP4 is Blackwell-only; we are sm_90.** The bridge already ships NVIDIA's
  reproduction of the pretraining recipe — `nemotron_3_super_bf16_with_nvfp4_mixed`
  (`src/megatron/bridge/training/mixed_precision.py:412-421`), whose
  `num_layers_at_end_in_bf16 = 14` of 88 is exactly the paper's "final 15% in BF16".
  We cannot run it.
- **"Experts-only" is not what NVIDIA did.** Table 3 also holds attention QKV/output
  projections, embeddings, MTP and latent projections in BF16, puts the Mamba output
  projection at MXFP8, and keeps the last 15% BF16 — six carve-outs, not one.

**Is experts-only FP8 even expressible here? Technically yes, practically no.** No
`fp8_*` field is submodule-granular: `first_last_layers_bf16` is a pure integer
comparison on layer index (`megatron/core/fp8_utils.py:673-689`). The only
submodule-granular surface is `TransformerConfig.quant_recipe`
(`transformer_config.py:1198`), glob-matched in the post-init sweep at
`megatron/core/models/hybrid/hybrid_model.py:330-333`, which does reach the expert
GEMMs through `TEGroupedLinear.forward`'s own autocast
(`megatron/core/extensions/transformer_engine.py:2242-2245`). Four blockers:

1. **The polarity is unattainable.** Expert token-count alignment padding is armed
   only by the *global* `config.fp8` (`megatron/core/transformer/moe/experts.py:295-300`)
   and cannot be enabled independently (`transformer_config.py:2297-2300`). You must
   turn FP8 on globally and glob everything *back* to BF16 — the opposite of scoping
   FP8 to experts.
2. **Mutually exclusive with the champion.**
   `src/megatron/bridge/models/nemotronh/cutlass_grouped_experts.py:97-98` raises
   "CutlassGroupedExperts is BF16/FP32-only (no quantization padding)", and the
   installed backend carries only BF16 CUTLASS kernels. FP8 experts means reverting to
   `te_grouped` and forfeiting the measured −16% (25.66 → 21.78 s/iter).
3. **`quant_recipe` is not wired into our entry point** — zero uses in `configs/`,
   reachable only via a Hydra `_target_` hack whose failure path silently degrades to a
   raw dict.
4. Per-module config also **rejects delayed scaling**
   (`extensions/transformer_engine.py:152-153`), and setting global `config.fp8` has
   side effects even with everything force-BF16 (Mamba mixer shape assert at
   `megatron/core/ssm/mamba_mixer.py:240-244`, recompute restrictions at
   `transformer_config.py:1802-1811`).

**Reconciliation with our prior FP8 MoE-routing crash:** experts-only would genuinely
dodge it — the router never touches a TE module
(`moe_utils.py:1294-1337`), so it is structurally immune to any autocast. The
consultant is right about the mechanism. It does not matter, because the throughput
case collapses first.

### C8 — QK in FP8 (VERDICT: measured at 0.73%; skip)

Measured directly from the champion trace rather than estimated
(`/projects/a5k/public/profiles/hostrc_m6p_cutlass_profile/20260729T225135-j5738452/rank9.iter10.chrome_trace.json.gz`),
casting a deliberately wide net over every attention-side kernel — cuDNN flash
fprop/bprop, `fmha_reduce_head_ragged`, `convert_dq_to_16bits`, TE `fused_attn` helpers:

| bucket | GPU time / iter | share of kernel-busy |
|---|---:|---:|
| NCCL | 10.149 s | 52.7% |
| GEMM | 4.853 s | 25.2% |
| other (eltwise/norm/copy/triton) | 1.826 s | 9.5% |
| Mamba scan | 1.609 s | 8.4% |
| MoE dispatch/permute | 0.470 s | 2.4% |
| Mamba conv1d | 0.207 s | 1.1% |
| **attention (all kernels, wide net)** | **0.159 s** | **0.8%** |

**0.159 s = 0.73% of the 21.78 s champion iteration.** Core SDPA flash alone is
0.134 s (fprop 0.037, bprop 0.096). So making attention *infinitely fast* saves 0.73%;
FP8 QK at a perfect 2× on those kernels saves ~0.07 s = **0.32%**, below the ~1%
run-to-run noise floor this benchmark can resolve (§"n=7 is too thin to separate ~1%
effects"). The architecture is why: 8 of 88 layers are attention, and Mamba scan alone
costs 10× more GPU time than attention.

Maturity note for completeness: FA3-style FP8 attention is **inference-only** in TE 2.x
("Disabling FlashAttention 3 for FP8 training"); FP8 attention *training* on Hopper
goes through cuDNN FusedAttention needing cuDNN ≥ 9.19, FP8 current-scaling attention
is Blackwell-only, and `fp8_dpa`/`fp8_mha` are Beta. No bridge recipe sets them; both
default `False` (`transformer_config.py:609-613`).

**Forward-looking note for C7.** The decline is contingent on the host-bound regime,
not on FP8 being wrong in principle. If the driver is upgraded to unlock image
26.06+/TE ≥ 2.15 with device-initiated grouped GEMM, the launch bottleneck that makes
every FP8 recipe net-negative today may no longer bind — revisit blockwise FP8 then.
None of the facts above change; only the conclusion could.

## C9 — gradient/optimizer precision (settled)

Already compliant: grads are BF16 (`bf16: true`, no fp32 grad accumulation override),
optimizer runs precision-aware with BF16 moments
(`use_precision_aware_optimizer: true`, `exp_avg_dtype/exp_avg_sq_dtype:
torch.bfloat16` in `configs/quickstart/nemotron_super_quickstart_sft.yaml`), and we
start the optimizer cold from the base checkpoint. The consultant endorsed exactly
this; no change.

Exact dtypes, since "leave the optimizer alone" is half-right — the moments are
already halved, so this is not headroom we still have:

| tensor | ours | where |
|---|---|---|
| gradient reduce/comm buffer | **BF16** | `grad_reduce_in_fp32` default `False`, not overridden (`distributed_data_parallel_config.py:15`) |
| Adam 1st moment | **BF16** | `exp_avg_dtype: torch.bfloat16` (config) |
| Adam 2nd moment | **BF16** | `exp_avg_sq_dtype: torch.bfloat16` (config) |
| optimizer main grads | **FP32** | `main_grads_dtype` default `torch.float32`, not overridden (`optimizer_config.py:197`) |
| master weights | **FP32** | `main_params_dtype` default `torch.float32`, not overridden (`optimizer_config.py:200`) |

This matches the paper's guidance exactly (§4.1.6 / §5.2.1: moments in BF16, "main
gradients, master weights… remain in their original precision"), and the paper notes
BF16 moments are orthogonal to FP8 — they apply to pure-BF16 runs like ours.

## C10/C11 — throughput ceiling + exact FLOPs (settled: tool built, frame recalibrated)

Tool: `scripts/nemotronh_flops_estimator.py` (config-driven; per-layer-type formulas in the
module docstring; 47 unit tests green in-container). Headlines for the champion workload
(GBS 64 × 32K, from the upstream `config.json` + training YAML):

- **Active params ≈ 12B** (A12B checks out); forward 26.87 GFLOP/token; model FLOPs
  (fwd+bwd) **169.1 PFLOP/iter**, hardware FLOPs (+selective recompute) 198.5 PFLOP/iter.
- **FLOP mix**: latent-MoE 52.3% (routed 36.1%, shared 13.1%), Mamba2 ~33%, attention
  10.1%, lm_head 4.0% — MLP-dominated, which is exactly the regime where 6ND is close:
  **exact/6ND = 1.05** (active-param basis; 1.10 non-embedding). The consultant's "~30%
  off" extreme (Mamba-heavy *small* models) does not apply at this scale; his advice to
  compute it exactly was still right, and now it costs one command.
- **Champion 21.78 s/iter = 121.3 model TFLOP/s/GPU = 12.3% MFU** (HFU 14.4%) on GH200
  BF16 peak 989. Matches mcore's logged ~112 at the 2-way-concurrent 23.40 s/iter.
- **The 400–550 frame does not transfer to this architecture.** 400 TFLOP/s/GPU here
  means 6.6 s/iter at 40% MFU. The pure tensor-core arithmetic floor is 3.14 s/iter;
  the other 18.6 s of the champion iteration is pipeline bubble, collectives,
  memory-bound Mamba scans, and launch gaps — categories his dense-7B-on-H100 frame
  (attention+MLP GEMMs, no PP at 7B, no MoE dispatch, no scan kernels) mostly lacks.
  Champion-era traces put GPU idle at ~7.4 s/iter: eliminating ALL of it with unchanged
  kernels lands ≈ 14.4 s/iter ≈ 183 TFLOP/s ≈ 18.5% MFU — the honest near-term ceiling.
  Kyle's <20 s target = 132 TFLOP/s = 13.4% MFU → needs ~1.8 s of the 7.4 s idle
  recovered; feasible. 400+ TFLOP/s would additionally require ~2× faster kernels
  (FP8 GEMMs — blocked per C6/C7 — and a faster scan), i.e. a different project.

## Experiment log

(mean of iters 10–30 unless noted; "2-way" = two concurrent 16-node runs in one
32-node allocation on disjoint halves — within-pair deltas are contention-matched,
absolute values carry the 2-way tax which the paired champion control calibrates)

| Arm | Config | Condition | Result |
|---|---|---|---|
| control_p1 | quickstart champion | 2-way pair 1 | **23.40 s/iter** mean-10-30, 0 NaN, ~112 TFLOP/s/GPU → 2-way tax = **+7.4%** vs 21.78 solo |
| a0_p1 | a0_vpp4_cutlass + no fine-grained offload | 2-way pair 1 | **27.51 s/iter** mean-10-30, final loss 0.68723 (parity). +17.6% raw / **+13.5% adjusted** for the offload-1.0 handicap → **cutlass did NOT shrink the VPP4 penalty; host-starvation hypothesis REFUTED** (matches July's 100%-stall trace decomposition) |
| a2_p2 | a2_pp8vpp2_stage0lite + no fine-grained offload | 2-way pair 2 | **FITS** (July 94.7 GB OOM solved by the stage-0-lite repartition) — **25.61 s/iter** mean-10-30, final loss 0.68722 (parity). +9.4% raw / **+5.4% adjusted**. Bubble math predicted −1.9 s; measured +1.3 s ⇒ interleave overhead ~3 s dominates at 32 µb/replica |
| q1_nooffload_p1 | champion + `optimizer_cpu_offload=false` | ~unpaired (partner A1 was in startup its whole run) | **20.84 s/iter** mean-10-30, loss 0.68733 (parity), 0 NaN — **−11% vs paired control; matches the SOLO prediction 20.8-20.9. Hottest lead: solo confirmation queued; if ≤20.5 solo, offload-off becomes the champion posture and <20 s is in reach without VPP** |
| a1_p3 | a1_pp16vpp2_dp1 + no fine-grained offload | 2-way (slot B) | **HUNG** — 76 min inside first-iteration NCCL comm-init (zero iterations, log frozen at 04:15, GPUs 100% = NCCL busy-spin), step killed. PP16·VPP2 startup pathology; deprioritized (A2 gives VPP2 a fit path and VPP2 loses on speed regardless) |
| q3c | A0 minus `core_attn` recompute (all else identical to A0-as-run) | 2-way-ish | **27.38 s/iter** mean-10-30, loss 0.68723 (parity) → **−0.13 s vs A0's 27.51, dead-center in the preregistered 0.05–0.15 band.** The C4 core_attn double-forward cost is confirmed at its predicted (small) size |
| q8_ompthreads (try 1) | champion + offload 1.0 + OMP_NUM_THREADS=8 PASSIVE | — | **VOID, not the lever's fault**: rank 0 (tunnel head node) hit ncclUnhandledCudaError during init while the commit gate's in-container GPU suite ran on the same node. Operational rule adopted: never overlap gate-suite runs with slot-A arm initialization (they share the head node's GPUs). Relaunch queued |
| q1_solo_pair | champion-0.5 then offload-off, sequential, identical nodelist, other slot empty | solo, same-allocation | **CONFIRMED: 22.79 → 20.72 s/iter, Δ = 2.07 s (2.6× the 0.8 prereg bar), loss Δ1.8e-4.** Offload-off is the new champion posture. Bonus datum: this allocation's champion reference is 22.79 vs the old allocation's 21.78 (placement ≈ 1.0 s), so offload-off projects to **~19.7 s/iter on a champion-class placement — under the 20 s target** |

New-pin VPP launch findings (details in the campaign prereg): (a) offloading
`expert_fc1`/`moe_act` under `moe` recompute now asserts — removed, memory-neutral;
(b) `fine_grained_activation_offloading` is incompatible with VPP at the 0.19 pin
(ChunkOffloadHandler chunk mismatch at iteration 2) — all VPP arms run without it.
