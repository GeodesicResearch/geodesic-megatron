# Nano-30B pretraining performance campaign

**Goal.** Raise the training efficiency of the control-pretraining baseline posture for Nemotron-3 Nano
30B-A3B (from-scratch pretraining, `configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_pretrain.yaml`:
TP1·CP1·EP4·PP1·ETP1, seq 8192, only data parallelism crosses nodes) by **≥ 2× tokens/s/GPU**
(equivalently model-FLOPs MFU) over the as-is baseline, measured at the same width and batch (raised from
1.5× by Kyle on 2026-09-28). Against the E-001 baseline (9.328 s, 7,026 tok/s/GPU, 14.78% MFU at 64 GPUs /
GBS 512) that is a mean step **≤ 4.664 s ≈ ≥ 14,052 tok/s/GPU ≈ ≥ 29.6% MFU**.

**Benchmark.** `configs/quickstart/nemotron_nano_quickstart_pretrain.yaml` — the baseline composed with a
small overlay (`base_config:`): GBS 512 on 64 GPUs (8 microbatches per DP replica, the same per-GPU work as
production's GBS 2048 on 256 GPUs), 50 iterations via `train.exit_interval`, no checkpoint load/save.
32 GPUs is the same file with `train.global_batch_size=256`.

**Metric.** Mean step time over iterations 26–50 (node-hours scale with the mean), reported with the
median, p10/p90 and outliers. tokens/s/GPU = GBS × 8192 / (GPUs × mean step). Model TFLOP/s and MFU use the
exact per-token FLOPs of `scripts/nemotronh_flops_estimator.py` and the GH200 dense BF16 peak (989.4 TFLOP/s).

**Method.** Every probe is its own 16-node `pipeline_training_submit.sbatch` job (`--time=00:20:00`;
a 50-iteration probe takes 9–13 min), launched from a read-only snapshot of the code under test under
`/projects/a5k/public/logs/nano_pretrain_perf_campaign/snapshots/` (each carries a `REVISION`, and a
`WORKTREE.diff` when the code is not yet committed). Kyle chose one short job per probe over a held
allocation (2026-09-29). The same posture has measured up to ~4% apart across jobs (Repeatability), so a lever is
credited only when its effect clearly exceeds that, or after repeats; a lever is VOID when its engagement
cannot be shown from the log or config. Levers that change numerics additionally need lm loss (iterations
41–50) inside the band of the other runs and a paired 500-iteration loss-parity run before adoption. Each
credited lever becomes part of the parent of the next probe.

**Research basis.** Six research reports, a campaign design and an adversarial critique, preserved at
`/projects/a5k/public/logs/nano_pretrain_perf_campaign/research/` (the per-hypothesis citations below point
into them and into the primary sources: the Megatron-Core MoE paper, arXiv 2603.07685v2; the Megatron-Bridge
performance guide; upstream Megatron-Bridge / Megatron-LM commits).

## Champion ladder (64 GPUs, GBS 512)

| Step | Lever | Mean step (s) | tok/s/GPU | MFU | × baseline | Entry |
|---|---|---|---|---|---|---|
| 0 | as-is baseline | 9.328 | 7,026 | 14.78% | 1.00 | E-001 |
| 1 | H01 `overlap_param_gather` (under `comm_overlap:`) | 8.464 | 7,743 | 16.28% | 1.10 | E-004 |
| 2 | + H15 bf16 gradient accumulation/reduction | 8.105 | 8,086 | 17.01% | 1.15 | E-008 |
| 3 | + H16 recompute `[moe_act]` only (funded by H15's 19 GB) | 7.075 | 9,263 | 19.48% | 1.32 | E-009 |
| 4 | + host-sync bundle (timers L1, NaN checks off, per-param grad norms off) | 6.862 | 9,551 | 20.09% | 1.36 | E-010 |
| 5 | + manual GC every 10 iterations | 6.744 / 6.494 | 9,717 / 10,092 | 20.4–21.2% | 1.38–1.44 | E-013, E-018 |
| 6 | + H13 fp32-SSM checkpoint patch off | 6.249 / 6.026 / 6.029 | 10,488–10,875 | 22.1–22.9% | 1.49–1.55 | E-019, E-022 |
| 7 | + H32 sync-free grouped-GEMM offsets | 5.968 | 10,981 | 23.09% | 1.56 | E-023 |
| 8 | + H27 FP8 current scaling on dense layers | 5.754 / 5.751 | 11,390 | 23.95% | 1.62 | E-024, E-027 |
| 9 | + H22 HybridEP intra-node dispatcher (sync-free with H32) | 5.407 / 5.388 / 5.394 / 5.409 | 12,115–12,163 | 25.5–25.6% | 1.72–1.73 | E-028, E-030; repeats 6932856, 6932890 (E-039–E-043 block) |
| 10 | + EP all-to-all / compute overlap (combined 1F1B, port of Megatron-LM PR #4798 — in a snapshot, not yet in this repo; `CUDA_DEVICE_MAX_CONNECTIONS=32`) | 5.197 / 5.228 | 12,535–12,611 | 26.4–26.5% | **1.78–1.80** | E-044 |

Every run is its own allocation; where a step was repeated, the table gives each run (see
Repeatability).

## Repeatability

Postures run more than once, each run its own 16-node job. Nodelists (from `sacct`):
N1 = nid[010038,010446,010503-010505,010543,011141,011148,011150,011153,011158,011173,011222,011225,011234,011237];
N2 = nid[011222,011224-011225,011227,011229,011234-011237,011241,011292-011297];
N3 = nid[010231-010232,010241,010243,010256-010258,010274,010286-010289,010300,010311,010313,010328];
N3′ = N3 with nid010038 in place of nid010241;
N4 = nid[011111,011117,011119,011129,011138-011139,011141,011148,011150,011153,011157-011159,011169,011173,011209];
N5 = nid[010444,010446,010450,010464,010474,010503-010506,010523,010529,010533,010543,010545-010546,010548];
N5′ = N5 with nid010541 in place of nid010548;
N6 = nid[010441,010444,010446,010450,010464,010474,010503-010504,010523,010529,010533,010541,010543,010545-010546,010548];
N7 = nid[011110-011111,011117,011119,011129-011130,011138-011139,011141,011148,011150,011153,011157-011159,011169].

| Posture (ladder step) | Entries (job, nodelist) | Mean step per run (s) | Mean of runs (s) | Spread, max − min |
|---|---|---|---|---|
| champion: H01 + host bundle + H15 + H16 + manual GC (5) | E-013 (6932005, N1), E-018 (6932452, N2) | 6.744 / 6.494 | 6.619 | 0.250 s (3.8%) |
| + H13 fp32-SSM off (6) | E-019 (6932453, N3′), E-022 (6932526, N3; 6932527, N4) | 6.249 / 6.026 / 6.029 | 6.101 | 0.223 s (3.7%) |
| + H32 + H27 FP8 on the dense layers (8) | E-024 (6932552, N2), E-027 (6932616, N3) | 5.754 / 5.751 | 5.753 | 0.003 s (0.05%) |
| + H22 HybridEP (9) | E-028 (6932618, N5), E-030 (6932677, N3), repeats 6932856 (N3) and 6932890 (N5′) in the E-039–E-043 block | 5.407 / 5.388 / 5.394 / 5.409 | 5.400 | 0.021 s (0.39%) |
| + EP all-to-all / compute overlap (10) | E-044 (6933731, N6; 6933837, N7) | 5.197 / 5.228 | 5.213 | 0.031 s (0.6%) |

The spread reached 3.7–3.8% on steps 5 and 6, stayed under 0.4% on steps 8 and 9, and was 0.6% on
step 10. On step 6
the two runs that differ most (E-019 on N3′, E-022's first on N3) shared 15 of their 16 nodes,
while E-022's two runs on disjoint nodelists agree to 0.05%. The ~4% bar in Method is the largest
spread observed, so a lever smaller than it is credited only after repeats.

## Stale verdicts retested

| Earlier verdict | Premise that no longer holds | Retest | Outcome |
|---|---|---|---|
| `overlap_param_gather: false` for Nemotron-H at DP>1 | derived on Super at TP4 / bare metal | H01 | WIN (E-004) |
| bf16 gradients "tried, failed" | all four attempts were VOID: `bf16_mixed` hard-codes `grad_reduce_in_fp32=True` | H15 | WIN — engaged via the bf16 grad-reduce recipe, `bf16_mixed_bf16_grad_reduce` (E-008) |
| dropping `moe` recompute OOMs | measured without bf16 gradients (19 GB/GPU more free memory with them) | H16 | WIN (E-009) |
| CUDA graphs blocked on Nano | the blocker was memory at DP=512, not capability | H18 | retested: engages at 64 GPUs with no memory cost; NULL on throughput (~0.8%, E-035) |
| DeepEP/HybridEP incompatible | the blocker is the inter-node IBGDA path; EP=4 is intra-node | H21/H22 | H22 WIN once GroupedExperts is sync-free (E-028; NULL before, E-005); H21 not yet retested |

## Experiments

### E-000 · campaign allocation chain burned on a missing runner · 2026-09-28 · INCIDENT
`sbatch --test-only` estimated a 17-node start ~32 h out, so the three-segment singleton chain
(6930403/6930404/6930407) was submitted before the runner it execs existed. Backfill started the first
segment immediately; each segment failed in 6 s (`can't open file .../scripts/training/campaign_runner.py`),
~0.1 node-hours in total. The wrapper itself was validated (node-limit guard passed, interpreter resolved).
Learning: build and test everything a batch script execs before submitting it; test-only estimates are
not a lower bound on start time.

### E-001 · calibration: as-is baseline at 64 GPUs / GBS 512 · 2026-09-28
A probe like every other (Method): one `pipeline_training_submit.sbatch` job on the baseline YAML with the
quickstart's overlay given as Hydra overrides (`train.exit_interval=50 train.global_batch_size=512
checkpoint.load=null checkpoint.save=null dist.distributed_timeout_minutes=20`, own index cache), from the
main checkout at e3dd4e56. Job 6930454 (unprofiled) and 6930456 (torch profile, iterations 30 and 40,
ranks 0 and 5, with stack). Purpose: the as-is baseline every later probe is scored against (ladder
step 0), plus the step-time, memory and trace data that re-ranked the backlog.
Prediction: mean step 9.7–10.0 s ≈ 6,550–6,750 tok/s/GPU ≈ 14% MFU; peak allocated ≈ 81.8 GB.
Result (job 6930454, nid[011001-011002,011009,011012-011013,011016,011019-011020,011023,011025,011030,
011041,011068-011069,011074,011076], W&B `nano_pretrain_perf_calib_b0`), iterations 26–50:
**mean 9.328 s** (median 9.287, p10 9.231, p90 9.462, min 9.213, max 9.580) → **7,026 tok/s/GPU, 146.2
model TFLOP/s/GPU, 14.78% MFU**; lm loss (41–50) 6.869; 0 NaN / 0 skipped; after-iteration-1 memory
79.86 GB max allocated / 81.98 GB reserved. Iteration 1 took 99.5 s; iterations 2–4 ran 9.3–9.5 s, then
the step drifted up to 11.7 s by iteration 18–19 and settled back to 9.2–9.5 s from iteration 26 — the
from-scratch routing transient the scoring window starts after. Within the window the first and second
halves differ by < 1% (26–37 vs 38–50), so the window is past the transient.
W&B timers (run `vtfgwr4d`, means over 26–50): iteration 9.328 s = forward-backward 7.908 (forward-compute
1.979 + backward-compute 5.979, summed over the 8 microbatches) + optimizer 1.366 (of which
**params-all-gather 1.356 s — exposed, `overlap_param_gather: false`**, inner Adam step 0.005) + ~0.05 other.
Backward is 3.0× forward (≈ 2× without recompute): the ~2 s excess is the `[core_attn, moe, shared_experts]`
recompute plus the fp32 reduce-scatter tail absorbed by the last backward's timer sync. Peak (W&B summary)
**81.84 GB allocated / 84.47 GB reserved** — exactly the 81.8 GB static-memory prediction; ~10 GB below the
~94.5 GB (88 GiB) effective ceiling. The run also logs a per-parameter gradient L2 norm for every parameter
every iteration (`log_l2_norm_grad_to_tensorboard: true`), a per-parameter kernel + host-sync pattern like
the `log_params_norm` telemetry that cost Super 10.2% (new candidate H38).
Learning: 5% faster than predicted; the 2× target at this width and batch is a mean step ≤ 4.664 s
(≥ 14,052 tok/s/GPU, ≥ 29.6% MFU). 1.36 s of the 9.33 s is the exposed all-gather alone.

### E-044 · EP all-to-all / compute overlap for the hybrid model · 2026-09-29 · WIN (1.78–1.80×, repeated)
`comm_overlap.overlap_moe_expert_parallel_comm=true` runs the combined-1F1B schedule: the forward of one
microbatch is interleaved with the backward of the previous one, so each MoE layer's dispatch/combine
runs on a communication stream while the other microbatch's Mamba, attention or expert GEMMs run on the
compute stream. Upstream supports it for `GPTModel` only; Megatron-LM PR #4798 (open, head `1fdff667`)
adds it for the hybrid model, and it merged cleanly onto the pin (its first part, #4941, is already in
it). Two adaptations, both in the snapshot's `EPOV.diff` / `EPOV_NOTES.md`
(`/projects/a5k/public/logs/nano_pretrain_perf_campaign/snapshots/wt-e3dd4e56-epov`):
- the schedule plan groups the flat layer pattern into 23 `[Mamba/attention..., MoE]` units, so every
  all-to-all has compute to hide behind, without the bracketed pattern upstream needs (which renames
  every parameter and checkpoint key); `MCORE_HYBRID_OVERLAP_AUTOGROUP=0` restores upstream's
  one-unit-per-layer plan (−2.4% instead of −4.5% on the 1-node smoke);
- upstream frees the dispatched expert input when FP8 is on, assuming the experts keep an FP8 copy as
  TE's grouped MLP does; GroupedExperts stays BF16 and needs that input for its weight gradient
  (crash 6932873, `setStorage ... size 0`), so non-TE experts keep it.

Required with it: `model.mtp_num_layers=null` (the recipe's `0` fails the "None or 1" assert — the first
64-GPU pair, 6933304/6933305, died on it; Nano has no MTP layers either way) and
`ISAMBARD_CUDA_MAX_CONNECTIONS=32` (with one hardware queue the two streams serialise).
1-node smoke (11 layers, 4 microbatches): ON 716–722 ms vs OFF 752–754 ms; ON-vs-OFF loss differences
(5–6.5e-4) sit inside OFF-vs-OFF (1.0e-3).

64 GPUs, E-028 posture from the frozen snapshot `wt-e3dd4e56-epov-full1`:

| Arm | Job | Mean (s) | Median | × baseline | Peak alloc / reserved GB | Loss 41–50 |
|---|---|---|---|---|---|---|
| OFF (control: connections 32, MTP null) | 6933820 | 5.409 | 5.356 | 1.725 | 73.42 / — | 6.911 |
| **ON, connections 32** | 6933731 | **5.197** | **5.143** | **1.795** | 74.52 / 85.10, 0 allocator retries | 6.879 |
| **ON, connections 32 (repeat)** | 6933837 | **5.228** | **5.180** | **1.784** | 74.52 / — | 6.911 |
| ON, connections 1 | 6933732 | 5.622 | 5.551 | 1.659 | 74.52 / — | 6.887 |

Learning: the two ON runs (5.197 / 5.228 s, 0.6% apart) are −3.4% against the five runs without the
overlap (5.388–5.409 s, the paired control included), with loss in band and +1.1 GB allocated. It is
modest because HybridEP's intra-node all-to-all is already short; what it hides is mostly the ranks'
waiting in `device_sync`. The hybrid plan asserts `cuda_graph_impl == "none"`, so CUDA graphs cannot
yet be stacked on it.

### E-039–E-043 · NCCL, GC, NUMA and Mamba-chunk probes on the HybridEP champion · 2026-09-29 · all NULL
All on the E-028 posture; NCCL variables go through `ISAMBARD_ENV_OVERRIDES` because the launcher exports
its own values unconditionally, and each run's per-rank `[env-overrides]` lines show them engaged.

| Entry | Lever | Job | Mean (s) | Median | × baseline | Loss 41–50 |
|---|---|---|---|---|---|---|
| — | none (champion repeat, same window) | 6932856 | 5.394 | 5.338 | 1.729 | 6.884 |
| E-039 | `NCCL_MIN_NCHANNELS=8`, `NCCL_NCHANNELS_PER_NET_PEER=8` | 6932857 | 5.401 | 5.347 | 1.727 | 6.904 |
| E-040 | `NCCL_BUFFSIZE=16777216` | 6932858 | 5.441 | 5.389 | 1.714 | 6.911 |
| E-041 | `gc.freeze()` after the setup collect in `train.py` | 6932862 | 5.388 | 5.384 | 1.731 | 6.889 |
| — | none (champion repeat, same window) | 6932890 | 5.409 | 5.361 | 1.724 | 6.891 |
| E-042 | `torchrun --numa-binding=node` | 6932889 | 5.394 | 5.334 | 1.729 | 6.906 |
| E-043 | Mamba SSD `chunk_size` 256 (default 128) | 6932900 | 5.440 | 5.369 | 1.715 | 6.896 |

Learning:
- The expert-DP ring is not NCCL-configuration bound: more channels or larger buffers change nothing.
- `gc.freeze()` removes the manual-GC spikes (iterations 31/41: 5.46/5.43 s instead of 5.93/5.85) but
  the mean does not move within noise.
- NUMA binding changes nothing because ranks are already bound: without the flag each rank's
  `Cpus_allowed_list` is its own GPU's Grace (local rank *i* → CPUs 72*i*…72*i*+71, matching
  `nvidia-smi topo -m`); only ~15 of ~192 threads per rank, created before the affinity is applied, had
  last run on another node.
- Chunk 256 is slightly slower; 128 stays.
- The champion now has four runs within 0.4% (5.388–5.409 s).

### E-038 · nccl-tests on the expert-ring pattern · 2026-09-29 · launch failed (replaced by training probes)
Intent: measure the expert data-parallel reduce-scatter / all-gather bandwidth under NCCL channel, buffer,
chunk and protocol variants with the in-container nccl-tests 2.19.6 (16 nodes × 4 tasks, `-g 1`,
`NCCL_TESTS_SPLIT_MASK=0x3` → four concurrent 16-rank rings with one GPU per node, the training pattern).
Neither launch mode works from the container: `srun --mpi=pmix` bootstraps MPI but every NCCL connection
handshake fails in the network plugin (`NET/OFI Request ... completed with error ... Error: 16 (Device or
resource busy)`, jobs 6932830, 6932835); the cluster default `--mpi=cray_shasta` cannot bootstrap the
image's Open MPI (`OMPI was not built with SLURM's PMI support`, job 6932842). The variants are measured
instead as training probes through `ISAMBARD_ENV_OVERRIDES` (the launcher exports its NCCL values
unconditionally, so plain environment variables would be overwritten).

### E-037 · microbenchmark: per-expert cuBLAS weight gradient accumulating into `main_grad` · 2026-09-29 · NULL
Motivation (E-036): the grouped weight-gradient GEMMs run at ~410 TFLOP/s and each is followed by a bf16
add into `main_grad` (~0.1 s per step). One GH200 (job 6932828), 32 experts × 49,152 rows, per call:

| Routing | Projection | `torch._grouped_mm` | grouped + add | per-expert `addmm_` (β = 1) | per-expert vs grouped + add |
|---|---|---|---|---|---|
| uniform | fc1 (2688→1856) | 0.806 ms (608 TFLOP/s) | 1.059 ms | 1.251 ms | 0.85× |
| uniform | fc2 (1856→2688) | 0.982 ms (499) | 1.073 ms | 1.136 ms | 0.94× |
| skewed (0.3–1.7× per expert) | fc1 | 1.035 ms (474) | 1.270 ms | 1.129 ms | 1.12× |
| skewed | fc2 | 1.035 ms (474) | 1.258 ms | 1.127 ms | 1.12× |

Results are bitwise identical. The per-expert path wins only under skew, and it needs the per-expert
counts on the host — the blocking copy H32 removed. Not pursued. Learning: routing skew costs the grouped
GEMM itself up to 22% (608 → 474 TFLOP/s on fc1), beyond the all-to-all waiting E-036 attributes to it.

### E-036 · trace of the HybridEP champion (E-028 posture) · 2026-09-29 · budget for the last 0.72 s
Job 6932683 (profile of iterations 30 and 40, ranks 0 and 5, no stacks; the profiled run itself scores
6.061 s mean because of the capture). Rank 0, iteration 40, window 5.754 s: compute union 4.353 s, NCCL
1.330 s of which **0.712 s exposed**, **idle 0.689 s (12.0%)**, almost all in 10–500 µs gaps.

| Component | Time (s) | Note |
|---|---|---|
| grouped expert GEMMs | 1.186 | forward 0.351 (514 TFLOP/s), dgrad 0.395 (457), wgrad 0.440 (410) |
| dense GEMMs (FP8 + bf16) | 0.803 | output-layer weight gradient 712 TFLOP/s |
| `hybrid_ep::device_sync_kernel` | 0.437 | 1,472 calls: ranks waiting for the slowest EP peer |
| HybridEP dispatch + combine kernels | 0.295 | ~198 MB remote per dispatch in ~0.4 ms — at NVLink rate |
| permute / unpermute | 0.170 | |
| elementwise | 0.450 | 3,056 bf16 gradient-accumulation adds = 0.137 |
| Mamba scan | 0.393 | |
| attention (cuDNN) | 0.213 | backward 3.4 ms × 48 |
| norm, router | 0.205 | |
| expert-DP reduce-scatter / all-gather | 0.764 / 0.660 | NIC time; 0.712 of the 1.33 s union exposed |

Host: `HybridEPDispatch` calls `cudaStreamSynchronize` 184 times per iteration (0.351 s of host time) to
size its output (`num_permuted_tokens=None`); the GPU idle attributed to it is only ~0.03 s. The sync-free
alternative (`moe_expert_rank_capacity_factor`) sizes every expert buffer to a static budget and drops
tokens above it, so it is not a free lever.
Learning — the exposed data-parallel time is a throughput floor, not an ordering problem: each iteration
needs ~1.42 s of NIC time for the expert reduce-scatter and all-gather, and only the last microbatch's
backward and the first microbatch's forward (~0.55 s) can hide it, so no reordering of buckets beats
~0.7 s exposed at micro-batch 1. Only a larger micro-batch (bigger windows) or faster NIC transfers move
it. The EP all-to-all costs ~0.9 s (waiting + kernels + permutation) and is serial with compute — the
target of upstream's expert-parallel overlap (Megatron-LM PR #4798 for hybrid models).

### E-031–E-035 · single-lever probes on the HybridEP champion · 2026-09-29
All on the E-028 posture; E-031–E-034 from the E-023 snapshot, E-035 from a copy with patch 0002
(`3rdparty/patches/megatron-lm/`, CUDA-graph `zeros_like` on a 0-dim tensor) applied:

| Entry | Lever | Job | Mean (s) | Median | × baseline | Peak GB | Loss 41–50 |
|---|---|---|---|---|---|---|---|
| E-031 | `model.moe_shared_expert_overlap=true` | 6932678 | 5.458 | 5.393 | 1.709 | 73.42 | 6.878 |
| E-032 | `ISAMBARD_CUDA_MAX_CONNECTIONS=32` | 6932681 | 5.424 | 5.339 | 1.720 | 73.42 | 6.881 |
| E-033 | `model.moe_permute_fusion_into_hybridep=true` | 6932682 | 5.419 | 5.367 | 1.721 | 73.42 | 6.916 |
| E-034 | `train.micro_batch_size=2` | 6932766 | OOM | — | — | — | — |
| E-035a | TE CUDA graphs, `cuda_graph_modules=[mamba]` | 6932739 | 5.350 | 5.309 | 1.744 | 73.46 | 6.936 |
| E-035b | TE CUDA graphs, `[mamba, attn, moe_router]` | 6932741 | 5.357 | 5.268 | 1.741 | 73.46 | 6.886 |

E-035 also sets `model.cuda_graph_impl=transformer_engine`, `model.cuda_graph_scope=null`,
`model.use_te_rng_tracker=true`, `NCCL_GRAPH_REGISTER=0` and `ISAMBARD_CUDA_ALLOC_CONF=expandable_segments:False`;
the logs show 23 and 52 graphable layers captured (0.4 / 0.9 s).
Learning:
- E-031–E-033 are NULL. E-032's engagement is not visible in the log (`pipeline_env_activate.sh` sets
  `CUDA_DEVICE_MAX_CONNECTIONS` from the knob silently), so it is also unverified.
- **E-034:** micro-batch 2 OOMs at iteration 1 in `vocab_parallel_cross_entropy`'s fp32 cast of the
  logits (8.00 GiB requested with 89 GiB already allocated). E-015 put micro-batch 2's extra activations
  at ~19 GB; only the `moe` recompute frees that, and it costs 1.03 s (H16), so the route stays closed
  unless ~12 GB is freed elsewhere.
- **E-035:** CUDA graphs engage and cost no memory, but gain only ~0.8% (5.35 vs 5.39–5.41), inside the
  run-to-run spread: the idle gaps sit mostly outside the graphable layers — in the routed-expert path,
  the dispatcher and the gradient hooks. Not adopted on its own.

### E-030 · repeat of the HybridEP champion (E-028) · 2026-09-29 · CONFIRMED (1.73×)
The E-028 posture and snapshot, launch line unchanged apart from the job and W&B names (job 6932677, W&B
`nano_pretrain_perf_probe_hepchamp_rep2`): mean **5.388 s** (median 5.322, p10 5.307, p90 5.448, min 5.300,
max 5.927) → 12,163 tok/s/GPU, 25.58% MFU, **1.731×**; lm loss (41–50) 6.854; peak 73.42 GB. Learning:
E-028 reproduces to 0.35%; the HybridEP champion stands at 5.39–5.41 s (1.73×).

### E-027–E-029 · FP8 champion repeat, HybridEP retest, 128 MB buckets · 2026-09-29
All on the E-024 posture (champion + H32 + FP8 dense), worktree snapshot:

| Entry | Lever | Job | Mean (s) | Median | × baseline | Peak GB | Loss 41–50 |
|---|---|---|---|---|---|---|---|
| E-027 | none (repeat of E-024) | 6932616 | 5.751 | 5.686 | 1.622 | 73.45 | 6.890 |
| E-028 | **HybridEP dispatcher** (`moe_token_dispatcher_type=flex`, `moe_flex_dispatcher_backend=hybridep`) | 6932618 | **5.407** | **5.341** | **1.725** | 73.42 | 6.892 |
| E-029 | `comm_overlap.bucket_size=134217728` (128 MB vs 500 MB) | 6932619 | 5.823 | 5.760 | 1.602 | 73.45 | 6.890 |

Learning: E-024 reproduces to 0.05%. **HybridEP is now −6%** where E-005 measured it NULL: then,
GroupedExperts copied HybridEP's device-side per-expert counts back to the host (`.to("cpu")`, a blocking
sync every MoE layer and microbatch); with H32 the counts stay on the device and the dispatcher's
sync-free path pays off. 128 MB buckets are slightly worse than 500 MB (NULL). New best: **5.407 s
(1.725×)**.

### E-026 · gate microbenchmark: FP8 routed-expert GEMMs · 2026-09-29 · GATE FAILED (lever closed)
One GH200 (job 6932663), `torch._scaled_grouped_mm` with rowwise E4M3 inputs and per-expert column-wise
weight scales vs BF16 `torch._grouped_mm`, at the Nano EP4 shapes (49,152 rows, 32 local experts):

| GEMM | BF16 | FP8 fast-accum | FP8 exact-accum | unfused input quantisation |
|---|---|---|---|---|
| fc1 [49152×2688]×[32×2688×1856] | 0.835 ms (587 TFLOP/s) | 0.476 ms (1.76×) | 1.567 ms (0.53×) | 1.271 ms |
| fc2 [49152×1856]×[32×1856×2688] | 0.746 ms (658 TFLOP/s) | 0.666 ms (1.12×) | 2.081 ms (0.36×) | 1.037 ms |

Only the fast-accumulation forward kernel beats BF16; the backward GEMMs need the exact accumulator (TE's
split-accumulator default for dgrad/wgrad), which is 2–3× slower than BF16, and unfused quantisation costs
more than the GEMM it feeds. The pre-registered gate (≥ 1.25× fwd+bwd including quantisation) fails:
FP8 routed experts are closed on this stack; FP8 stays on the dense TE linears (E-024).

### E-023–E-025 · H32, FP8 on dense layers, M3 — worktree snapshot · 2026-09-29
Snapshot `/projects/a5k/public/logs/nano_pretrain_perf_campaign/snapshots/wt-e3dd4e56-fp8-m3/` (the
worktree with its uncommitted code, `WORKTREE.diff` recorded), all on the E-022 champion posture:

| Entry | Lever | Job | Mean (s) | Median | × baseline | Peak GB | Loss 41–50 |
|---|---|---|---|---|---|---|---|
| E-023 | H32 sync-free offsets (the control: champion + this snapshot's code) | 6932551 | 5.968 | 5.878 | 1.563 | 75.40 | 6.846 |
| E-024 | **+ H27 FP8 current scaling on the dense TE linears** (`mixed_precision=nemotron_h_bf16_with_fp8_current_scaling_bf16_grad_reduce`; routed experts stay BF16) | 6932552 | **5.754** | **5.702** | **1.621** | 73.45 | 6.937 |
| E-025 | + M3: `train.micro_batch_size=2` + `fine_grained_activation_offloading` of `expert_fc1`/`moe_act` in GroupedExperts | 6932553 | 9.583 | 9.531 | 0.973 | 88.77 | 6.910 |

Learning:
- **H32** is ~1% faster than the E-022 repeats (inside noise, expected direction) and is a
  correctness-neutral removal of 736 host syncs per iteration — kept.
- **FP8 on the dense layers** is −3.6% against its same-snapshot control and −2 GB; the 2+2 edge layers
  stay BF16 (Nemotron-H recipe). Its loss sits at the top of the observed band, so it needs a repeat and,
  before adoption, the paired loss-parity run. New best: **5.754 s (1.62×)**.
- **M3 is +60% slower**: offloading the expert activations over NVLink-C2C at micro-batch 2 stalls the
  step (and still peaks at 88.8 GB). With E-015's small micro-batch-2 dividend this closes the
  micro-batch-2 route; the offload code was removed.

### E-022 · repeats of champion + H13 (fp32-SSM off) · 2026-09-29 · CONFIRMED (1.55×)
Two more runs of the E-019 posture from the same snapshot (jobs 6932526, 6932527): mean **6.026 / 6.029 s**
(medians 5.968 / 5.975, p90 6.10), 10,875 / 10,870 tok/s/GPU, 22.9% MFU, **1.548× / 1.547×**; lm loss (41–50)
6.909 / 6.881; peak 75.40 GB. With E-019 (6.249 s) the posture spans 6.03–6.25 s = **1.49–1.55×** across
three allocations — the original 1.5× goal is met reproducibly. Champion posture from here on:
H01 + host bundle + H15 + H16 `[moe_act]` + manual GC 10 + H13.

### E-018–E-021 · champion (E-013) repeat and three single-lever probes · 2026-09-29
All on the E-013 posture (H01 + host bundle + H15 + H16 `[moe_act]` + manual GC 10), manual snapshot,
separate 16-node allocations:

| Entry | Lever | Job | Mean (s) | Median | × baseline | Peak GB | Loss 41–50 |
|---|---|---|---|---|---|---|---|
| E-018 | none (champion repeat) | 6932452 | 6.494 | 6.450 | 1.436 | 73.86 | 6.933 |
| E-019 | **H13 fp32-SSM checkpoint patch off** (`ISAMBARD_FP32_SSM_STATE=0`) | 6932453 | **6.249** | **6.171** | **1.493** | 75.40 | 6.871 |
| E-020 | H14 `model.moe_router_fusion=true` | 6932454 | 6.738 | 6.690 | 1.384 | 73.84 | 6.912 |
| E-021 | H07 `ISAMBARD_CUDA_MAX_CONNECTIONS=8` | 6932455 | 6.688 | 6.644 | 1.395 | 73.86 | 6.905 |

Learning: **run-to-run noise is up to ~4%** — the identical champion measured 6.744 (E-013) and
6.494 (E-018). H14 and H07 sit inside that band (NULL). H13 is −3.8% against the better champion run and
−7.3% against the other, with loss in band: the fp32 checkpointed SSM scan (built for ~32K single-document
NaNs; "unnecessary at 8K") costs an extra fp32 scan pass per Mamba layer per microbatch. E-007's
"inconclusive" was one noisy run. **New best: 6.249 s (1.493×)**, to be confirmed by a repeat.

### E-017 · exploratory probe H03: `TORCH_NCCL_BLOCKING_WAIT=0` on the E-010 stack · 2026-09-29 · NULL + exit hang
Snapshot copy with launcher line 378 set to 0 (job 6932316). Mean 6.906 s (median 6.778, p90 7.091) vs
E-010's 6.862 / 6.688 — no gain (+1.3% by median, inside run-to-run noise). Engagement: the launcher's
`TORCH_CPP_LOG_LEVEL=error` hides torch's "NO watchdog thread" banner either way, but the run's shutdown
proves the watchdog was live — after `exit_interval` the ranks left at different times and the watchdog
began "dump debug info … due to a collective timeout", holding the job past 12 min until cancelled.
Learning: the host busy-wait under blocking wait is not a material cost at this posture, and turning it
off introduces an exit-time hang with `exit_interval`; dropped.

### E-016 · diagnostic X6: forced expert load balancing on the E-010 stack · 2026-09-29 · imbalance tax ≈ 5%
`model.moe_router_force_load_balancing=true` (job 6932315; benchmark-only — it changes routing and is never
adoptable). Mean 6.574 s, **median 6.350** vs E-010's 6.862 / 6.688 → routing imbalance costs ≈ 0.34 s
(≈ 5%) of the step at iterations 26–50, through all-to-all peer waiting and uneven expert GEMMs.
Learning: an upper bound on any load-balance lever; part of it is transient — production's steady state
(iterations ≥ 250) ran 3–6% faster than iterations 25–50 as the expert bias converges — so the 50-step
benchmark somewhat overstates imbalance cost.

### E-015 · exploratory probes X5: micro-batch 2 vs 1 at `["moe"]` recompute · 2026-09-29 · small (−2.6% median)
Both on H01 + host bundle + H15, recompute `["moe"]` (no manual GC). mbs 1 (job 6932314): mean 7.646 s,
median 7.652, peak 60.37 GB. mbs 2 (job 6932313, 4 microbatches per replica): mean 7.737, **median 7.451**,
peak **79.64 GB** (+19.3 GB), one 10.1 s GC excursion. The dividend is −2.6% by median — far below the
research plan's −0.5…−0.7 s (7–9%) from doubled DP-hiding windows and halved per-microbatch fixed costs.
Learning: at this posture DP exposure is not the bottleneck the plan modelled; M3 (activation offload to
fund mbs 2 without the MoE recompute) is therefore expected to buy ≲ 3% and is deprioritised pending one
direct test.

### E-014 · code H32: sync-free grouped-GEMM offsets · 2026-09-29 · implemented
`grouped_experts.py` computed `torch.cumsum(batch_sizes).to(device)` from pageable host memory in every
`_grouped_projection` call — a host-blocking copy, twice per MoE layer per microbatch (the 736
`_grouped_projection` stream syncs of E-003). Now `_group_layout` builds the backend's grouping once per
forward: int32 offsets on the device when the counts are already there (flex/HybridEP), otherwise summed on
the host and copied from pinned memory with `non_blocking=True`; `cublas_grouped` keeps its host int64
counts. Failing test first: `TestTorchGroupedBackend.test_forward_never_synchronizes_the_host[cpu|cuda]`
runs the forward under `torch.cuda.set_sync_debug_mode("error")` — both raised "called a synchronizing
CUDA operation" before the fix and pass after; all 45 GroupedExperts tests pass, including the bit-exact
per-expert reference.

### E-013 · exploratory probe: E-010 stack + manual GC every 10 iterations · 2026-09-29 · WIN (1.38×)
`train.manual_gc=true train.manual_gc_interval=10` (job 6932005): mean **6.744 s** (median 6.691, p10
6.629, p90 6.816, max 7.278) → 9,717 tok/s/GPU, 20.44% MFU, **1.383×**; lm loss (41–50) 6.912. Same median as
E-010 (6.688) but the 8–9 s excursions are gone: the only bumps left (~7.25 s at iterations 31 and 41) are
the scheduled collections. Learning: the in-window spikes were Python's automatic GC pausing ranks out of
step (MoE paper Table 15; perf guide "manual GC"); a longer production interval amortises the scheduled
collection further. Best posture so far.

### E-012 · analysis: the 2× budget and the expert data-parallel floor · 2026-09-29
Research workflow (four reports, a plan and a critique;
`/projects/a5k/public/logs/nano_pretrain_perf_campaign/research_2x/`). Findings that change the plan:
- **FP8 is a small lever here.** Dense TE linears (Mamba in/out projections, attention projections, shared
  experts) can run FP8 current-scaling after relaxing `grouped_experts.py`'s FP8 refusal (routed experts
  stay bf16 `torch._grouped_mm`); after the MoE recompute is gone the eligible time is ~0.85–0.9 s, so
  −0.1…−0.3 s. FP8 routed experts have a measured in-repo kernel at 0.57× bf16 and are out of scope
  unless a microbenchmark shows ≥ 1.25×. NVIDIA's own H100 Nano numbers are ~12.1–12.5k tok/s/GPU bf16 and
  14,890 FP8 (`docs/performance-summary.md`), with forced load balancing (benchmark-only) — the 2× target
  (14,052) sits at NVIDIA's best published H100 posture.
- **Expert data-parallel traffic sets a floor at micro-batch 1.** The expert-data-parallel group has one
  GPU per node (EP=4 fills the node), so every hop of the expert gradient ring crosses Slingshot: with bf16
  gradients ≈ 13.8 GB of reduce-scatter + 13.8 GB of all-gather per GPU per step over a 25 GB/s NIC ≈ 1.1 s
  of network time. At mbs 1 only the first microbatch's forward (~0.2 s) and the last one's backward
  (~0.5 s) can hide it, leaving ≥ 0.4 s exposed whatever else is done. **Micro-batch 2** doubles both
  windows and halves every per-microbatch fixed cost (all-to-all calls, host syncs, launches), and is the
  structural lever for 2× — funded by activation offload of the grouped experts (new code, M3) or by
  keeping the `moe` recompute (config-only, probe X5).
- Critique's re-anchored estimate for the remaining config/code levers: central 5.5–5.7 s (1.64–1.70×)
  from E-009; P(≤ 4.664 s) ≈ 5–10% without CUDA graphs on the Mamba layers, the 1F1B expert-overlap port
  (~1.4k lines upstream #4942) and M3. These become the critical path.

### E-011 · exploratory probe: E-010 stack with no recompute at all · 2026-09-29 · NULL
`model.recompute_granularity=null` instead of `[moe_act]` (job 6932003): peak 78.13 GB (+4.3 GB); mean
7.033 s (median 6.771) → 1.326×, vs E-010's 6.862. The `moe_act` recompute is effectively free; keep it
and spend the headroom elsewhere.

### E-010 · exploratory probe: full stack (H01 + host bundle + H15 + H16) · 2026-09-29 · best so far (1.36×)
Job 6932001: peak 73.86 GB; mean **6.862 s** (median 6.688, p10 6.644, p90 7.099, max 8.479) → 9,551
tok/s/GPU, 20.09% MFU, **1.359×** (1.41× at the ~6.61 s floor); lm loss (41–50) 6.924. Excursions at
iterations 32–34 and 39–40 cost ~3% of the mean (manual-GC probe 6932005 tests the GC hypothesis).

### E-009 · exploratory probe H01 + H15 + H16 (recompute `[moe_act]` only) · 2026-09-28 · WIN (1.32×)
With bf16 gradients freeing 19.1 GB, the MoE recompute (router, permute, two all-to-alls, grouped GEMMs,
shared expert re-run in every backward; MoE paper §4.1.4) is dropped: `model.recompute_modules=[moe_act]`
(the cheap expert activation recompute stays; `grouped_experts.py` supports it). Job 6931240 (manual
snapshot): **peak 73.86 GB** (8 GB below the as-is baseline despite no MoE recompute); mean **7.075 s**
(median 6.856, p10 6.770, p90 7.537, max 9.136) → 9,263 tok/s/GPU, 19.48% MFU, **1.318× (1.36× by the
median)**; lm loss (41–50) 6.943 — the top of the observed 6.865–6.943 run band, to be watched in the
loss-parity run. The series has a steady floor of 6.77–6.89 s with excursions at iterations 22–23 and
33–41 (up to 9.1 s); at the floor the stack is 1.38×. Learning: memory was the only thing standing
between this posture and the MoE recompute's removal; ~20 GB of headroom remains for further trades.

### E-008 · exploratory probe H01 + H15 (bf16 gradient accumulation + reduce) · 2026-09-28 · WIN (1.15×)
`mixed_precision=bf16_mixed_bf16_grad_reduce` — `bf16_mixed` under the `_bf16_grad_reduce` modifier of
`get_mixed_precision_config` (`src/megatron/bridge/training/mixed_precision.py`, unit-tested), i.e.
`bf16_mixed` with `grad_reduce_in_fp32=False`, applied by `MixedPrecisionConfig.setup()` to the model,
optimizer and DDP configs — the earlier four attempts were VOID because `bf16_mixed` hard-codes the fp32
value. Engagement: DDP config logs `grad_reduce_in_fp32=False`; peak **60.77 GB = 79.86 − 19.09**, the
predicted fp32→bf16 grad-buffer saving to the gigabyte. Job 6931239, manual snapshot
`/projects/a5k/public/logs/nano_pretrain_perf_campaign/snapshots/manual-e3dd4e56-bf16gr` (e3dd4e56 +
Megatron-LM 12c20d8f0 + the recipe): mean **8.105 s** (median 8.079, p10 8.036, p90 8.191) → 8,086
tok/s/GPU, 17.01% MFU, **1.151×**; lm loss (41–50) **6.8692 vs baseline 6.8690**. ~−4% on top of H01
alone, from halving the reduce-scatter's bytes (the RS tail inside the last backward).

### E-007 · exploratory probe H01 + H13 (fp32-SSM checkpoint patch off) · 2026-09-28 · inconclusive
`ISAMBARD_FP32_SSM_STATE=0` on top of H01 (job 6931200): mean 8.732 s (median 8.751, p10 8.194,
p90 9.432 — a far wider spread than any other run) → 1.068× vs 1.102× for H01 alone; lm loss (41–50)
6.915 (highest of all runs); peak after iteration 1 +1.5 GB (scan activations kept instead of
recomputed). Learning: no evidence of a gain; the spread suggests a placement effect, so H13 needs a paired
test before any verdict.

### E-006 · exploratory probe H01 + host-sync bundle · 2026-09-28 · promising
On top of H01: `logger.timing_log_level=1` (drops the per-microbatch forward/backward timer
`cuda.synchronize` pairs), `rerun_state_machine.check_for_nan_in_loss=false` (48 stream syncs / 0.41 s in
E-003), `ddp.check_for_nan_in_grad=false`, `logger.log_l2_norm_grad_to_tensorboard=false` (one norm +
`.item()` per parameter). Job 6931201: mean **8.321 s** (median 8.288, p10 8.251, p90 8.429) → 7,876
tok/s/GPU, 16.56% MFU, **1.121×**; lm loss (41–50) 6.884. ~−1.7% vs H01 alone, directionally consistent
with the ~0.4–0.6 s of host syncs it removes, but inside the run-to-run spread. The bundle was not
measured alone again; it is carried in every later rung from E-010 on, including the repeated
E-013/E-018 posture.

### E-005 · exploratory probe H22: HybridEP intra-node dispatcher · 2026-09-28 · NULL (alone)
`model.moe_token_dispatcher_type=flex model.moe_flex_dispatcher_backend=hybridep` on the E-001 posture (job
6930752; resolved config confirms flex/hybridep). HybridEP auto-detects the 4 NVLink peers
(`deep_ep/hybrid_ep_buffer.py`), so no environment variable is needed; kernels JIT-compile in the first
iteration (122 s vs 100 s). Runs with the `torch_grouped` experts, memory −0.2 GB. Mean **9.358 s**
(median 9.312) vs 9.328 → 0.997×; lm loss (41–50) 6.890. Learning: replacing NCCL's all-to-all transport
does not shorten the 1.29 s expert exchange — that time is dominated by peer waiting and the
surrounding host synchronisation, not by transport efficiency. Kept as a candidate to retest once the
MoE path is sync-free (H32) and inside a CUDA-graph posture, where its device-side metadata matters.

### E-004 · exploratory probe H01: `comm_overlap.overlap_param_gather: true` · 2026-09-28 · WIN (1.10×)
Mechanism: the distributed optimizer's parameter all-gather (1.36 s, fully exposed in E-001/E-003) is
issued asynchronously per bucket and waited on in the forward pre-hooks instead of synchronously after the
optimizer step (Megatron-Bridge performance guide; MoE paper Table 14; the Nano recipe's own default is
true, `src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py:178`). It must be set under
`comm_overlap:` — the pretrain recipe's `CommOverlapConfig.setup()` overwrites `ddp:`.
Prediction: −8…−13%. Job 6930751 (E-001 overrides + `comm_overlap.overlap_param_gather=true`, different
nodelist from E-001, single run): mean **8.464 s** (median 8.412, p10 8.370, p90 8.630) → 7,743 tok/s/GPU,
16.28% MFU, **1.102× the E-001 baseline**; lm loss (41–50) 6.8648 vs 6.8690. A single run on a different
placement from E-001; the gain is more than twice the cross-allocation spread (Repeatability), and every
later rung of the ladder carries the lever. Learning: ~0.86 s of the
1.36 s exposed all-gather is hidden — the first microbatch's forward is too short to hide all of it.

### E-003 · baseline trace at 64 GPUs / GBS 512 · 2026-09-28
Job 6930456 (same config as E-001, `ISAMBARD_TORCH_PROFILE_ITERS=30,40`, ranks 0 and 5, `with_stack=True`),
traces and tool outputs in
`/projects/a5k/public/profiles/nano_pretrain_perf_calib_b0_prof/20260928T214310-j6930456/{,analysis/}`
(`scripts/profiling/trace_analysis/*`, run in-container on an allocated node, job 6930713). Ranks 0 and 5
agree within 2%; rank 0, iteration 40:

| | seconds (profiled window 11.59 s) |
|---|---|
| compute union | 5.12 — GEMM 2.70 (dense + grouped CUTLASS), Mamba scan 0.96, elementwise 0.80, attention 0.23, MoE routing/permute 0.20 + 0.39, norm 0.13 |
| NCCL union / exposed / overlapped | 4.00 / **3.41** / 0.59 |
| — ReduceScatter (fp32 grads, 16 calls) | 1.80 sum, 1.47 union — ~37 GB/GPU at ~21–25 GB/s ≈ NIC line rate (bandwidth-bound) |
| — AllGather (params, 405 calls) | 1.35 — fully exposed (the `params-all-gather` timer) |
| — SendRecv (expert all-to-all, 1656 calls) | 1.29 — fully exposed, ~0.78 ms/call |
| idle | 3.06 (inflated by `with_stack`; the unprofiled step is 2.26 s shorter, so ~0.8 s real) |

Idle gaps: 57k, 63% of idle time in 100–500 µs gaps (launch/sync-bound). Host-sync census (rank 0):
`rerun_state_machine.validate_result` 48 stream syncs / 0.41 s (NaN-in-loss check), `token_dispatcher.
_maybe_dtoh_and_synchronize` 368 event syncs / 0.51 s (all-to-all split sizes), `grouped_experts.
_grouped_projection` 736 blocking H2D copies / 0.23 s (`cumsum(...).to(device)`), timers 79 device syncs /
0.10 s. Both DP collectives run NCCL's generic `RING_LL`-named kernel, but the measured bandwidth is at line
rate, so the protocol is Simple in practice (`NCCL_PROTO=^LL128` permits it) — not a lever.
**Budget of the unprofiled 9.33 s step: ~5.1 s compute, ~3.4 s exposed communication (DP ≈ 2.5, EP ≈ 1.3),
~0.8 s host-sync idle.** Re-ranking: hiding/halving DP communication (H01, H15) and the expert all-to-all
(H22 HybridEP, which also removes the dispatcher's D2H sync) dominate; the host-sync bundle (H06, H32, H04)
is worth up to ~0.8 s.

### E-002 · probe D2: is HybridEP available in the 26.04 image? · 2026-09-28 · YES
`import deep_ep` inside the container (no GPU work): `deep_ep 1.2.1+34152ae` at
`/usr/local/lib/python3.12/dist-packages/deep_ep/`, exports `Buffer`, `HybridEPBuffer`,
`HybridEpConfigInstance`; sources at `/opt/DeepEP`. The earlier "DeepEP/HybridEP incompatible" verdict
concerned the inter-node IBGDA path; intra-node HybridEP at EP=4 (NVIDIA's own H100 Nano pretrain
configuration, `scripts/performance/configs/nemotronh/nemotron_3_workload_base_configs.py` upstream) needs no
package build and is a wave-3 arm (H22).
