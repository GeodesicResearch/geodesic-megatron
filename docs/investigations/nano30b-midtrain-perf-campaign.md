# Nano-30B midtraining performance campaign

**Goal.** Raise the training efficiency of the control-pretraining baseline's stage 2 for Nemotron-3 Nano 30B-A3B
(midtraining, `configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_midtrain.yaml`: seq 32768,
TP1·CP2·EP4·PP1·ETP1, full recompute, warm-started weights-only from the stage-1 final checkpoint) to **≥ 1.5×
tokens/s/GPU** on the benchmark below, with levers that keep functional parity with the production run: the same
model and checkpoints, the same data and schedule, the same loss trajectory (Kyle, 2026-09-30). The Nano pretraining
campaign (`docs/investigations/nano30b-pretrain-perf-campaign.md`) is the starting point, but its fastest
configuration does not transfer as-is: stage 2's CP=2 and full recompute exist because the 32K fp32 cross-entropy
logits (16 GiB per sequence, 8 GiB per rank at CP=2) left no room, and the pretraining campaign's EP overlap refuses
full recompute.

**Result.** The final posture (see Final posture) is **1.648× the as-is benchmark** in four placement-controlled
paired cycles, 95% CI [1.634, 1.662]: under the pre-registered rule, **goal met, established** at k = 4. On one
switch group it runs 3.745 s/iter (the six final-posture runs of the three cycles placed on one group, whose as-is
runs average 6.158 s; the as-is anchor is 6.146 s) = 8,749 tokens/s/GPU, 21.6% MFU, against the as-is 5,332
tokens/s/GPU, 13.2% MFU, and it trains in 77 GB where the as-is posture takes 83 GB. Its loss over 500 iterations stays inside the band of three as-is runs, and a
checkpoint it writes resumes exactly and exports to HF with the as-is layout (E-007).

**Benchmark.** `configs/quickstart/nemotron_nano_quickstart_midtrain_baseline.yaml` — the production midtraining
config composed with a small overlay (`base_config:`): GBS 64 on 64 GPUs (DP=32 at CP=2, so 2 microbatches per DP
replica, the same per-GPU work as production's GBS 512 on 512 GPUs), the weights-only warm start from the stage-1
final (iteration 29881) that production's first segment made, no checkpoint load or save of production's, and an
exit inside the window E-001 calibrates. `train_iters` stays 3126, so the 100-iteration warmup and the cosine anneal
are production's iteration for iteration.

**Metric.** Mean step time over the calibrated window (node-hours scale with the mean), reported with the median,
p10/p90 and outliers by `scripts/telemetry/score_run.py`. tokens/s/GPU = 64 × 32768 / (64 × mean step). Model
TFLOP/s and MFU use the estimator's 24.432 GFLOP/token at seq 32768 (`scripts/nemotronh_flops_estimator.py`) and the
GH200 dense BF16 peak (989.4 TFLOP/s). Memory is the reserved memory at the last logged step (last rank, from W&B),
with the allocator-retry count: the run maxima and the log's single memory line (rank 0 after iteration 1) include
the warm start's checkpoint-load transient, which hides the training peak (E-004).

**Method.** As in the pretraining campaign: every probe is its own 16-node `pipeline_training_submit.sbatch` job
launched from a read-only snapshot of the code under test (each with a `REVISION` naming the commit and any
uncommitted diff), with its posture given as Hydra overrides and `ISAMBARD_ENV_OVERRIDES` lines composed from arm
files. Snapshots, arm files and the run registry (`runs.tsv`) are under
`/projects/a5k/public/logs/nano_midtrain_perf_campaign/`. Only runs placed on a single Dragonfly switch group are
compared with each other (the `[run-identity] switch placement` line; a multi-group run is 2.5–6.6% slower,
pretraining campaign E-047). A lever that changes numerics also needs its 500-iteration loss inside the band of the
as-is runs (`scripts/telemetry/loss_parity.py band`) before it counts.

## Final posture

`configs/quickstart/nemotron_nano_quickstart_midtrain.yaml`, the baseline benchmark plus these levers (launched with
`nemotron_nano_quickstart_midtrain.env`, which pins the fp32 SSM state to its checkpointed mode):

| Lever | Setting | Entry |
|---|---|---|
| parameter all-gather overlap | `comm_overlap.overlap_param_gather: true` | E-003 |
| host-side settings | `logger.timing_log_level: 1`, `logger.log_l2_norm_grad_to_tensorboard: false`, `rerun_state_machine.check_for_nan_in_loss: false`, `train.manual_gc` every 10 iterations with `manual_gc_freeze` | E-003 |
| BF16 gradients, FP8 dense layers | `mixed_precision: nemotron_h_bf16_with_fp8_current_scaling_bf16_params_bf16_grad_reduce` (BF16 primary weights; routed experts stay BF16) | E-003, E-005 |
| chunked linear cross-entropy | `model.cross_entropy_loss_fusion: true`, `cross_entropy_fusion_impl: linear`, `cross_entropy_fusion_saved_logit_chunks: 8` | E-003 |
| HybridEP dispatcher, router fusion | `model.moe_token_dispatcher_type: flex`, `moe_flex_dispatcher_backend: hybridep`, `moe_router_fusion: true` | E-001, E-002 |
| selective recompute | `model.recompute_granularity: selective`, `recompute_modules: [moe, shared_experts]` (method and layer count null) | E-005 |

Topology (TP1·CP2·EP4·PP1), batch, data, warm start and schedule are the benchmark's; the gradient NaN check stays
on. Memory: 77.0 GB reserved in training, against the as-is posture's 83.4 GB.

## Verification design (fixed before the verification runs)

**Speed.** Placement-controlled paired cycles, as in the pretraining campaign (its E-061): each cycle is one 16-node
job that runs the as-is benchmark, the final posture, the final posture and the as-is benchmark back to back on the
same nodes, every run exiting at 200 and scored over 151–200. The estimand is the ratio of a cycle's as-is mean to
its final-posture mean, combined over cycles as a geometric mean with a 95% t-interval on the log ratios. k = 4
cycles, extended to 6 if the interval contains 1.5 or its half-width exceeds 1%. Verdict: **established** if the
interval's lower bound is ≥ 1.5; **met on average** if the point estimate is ≥ 1.5 but the interval reaches below
it; **not met** otherwise.

**Loss parity.** 500 iterations of the final posture against three as-is runs that read the same data order (for a
posture at CP ≠ 2, three as-is runs at its CP), `scripts/telemetry/loss_parity.py band --window 50`: every window's
mean lm loss must lie inside [lowest reference − δ, highest reference + δ], δ being the largest window difference
between two references. If exactly one window fails, a fourth as-is run is added and that window is judged against
the four-reference band; two or more failing windows fail the posture. The grad norm is flagged, not failed, by the
same band.

**Numerics and launch checks.** The final posture's iteration-1 lm loss within 5×10⁻³ of the as-is one (same
weights, same data); no NaN or skipped iteration in any run; every rank's `[env-overrides]` line shows the launcher
settings the posture needs. The check exists to catch a wrong warm start or wrong data, which move the iteration-1
loss by 0.1 or more (consecutive batches differ by that much). It was first written as 1×10⁻³ and widened to 5×10⁻³
after the FP8 ladder probes measured 2.0×10⁻³, before any verification run: FP8 quantises the dense layers' forward
by design.

**Checkpoints.** A final-posture run saving at iterations 10 and 20 and a run resuming from 10 must agree at
iterations 11–20 (the pretraining campaign's resume check), and a checkpoint the final posture wrote must export to
HF with the tensor names and shapes of the as-is export.

## Open risks

- **FP8 costs a little loss.** Over 500 iterations the final posture sits +0.0019 above the as-is mean on average
  (above it in 8 of 10 windows), inside the run-to-run band (δ = 0.0083); beyond 500 iterations it is untested. The
  same posture in BF16 passes the band centred on the references and runs at 1.565× (E-007), the fallback if a
  longer comparison ever says otherwise.
- **Memory.** 77 GB reserved against a practical PyTorch ceiling of ~87 GiB (E-005). Selective recompute keeps no
  MoE activations past their own layer, so routing imbalance moves one layer's transient, not the whole stack's;
  the benchmark has covered 500 of the stage's 3126 iterations.
- **Parameter-gather overlap on Nemotron-H.** The production configs keep it off at DP>1 (the "standing posture"
  CLAUDE.md names), with no recorded failure behind that choice; the final posture turns it on, as the pretraining
  quickstart does, and passes the loss band, the resume and the export checks.
- **Production width.** The benchmark reproduces production's per-GPU work (2 microbatches per replica) but not its
  width: at 512 GPUs production's data-parallel collectives span 256 ranks and eight switch groups. The speed-up
  there is unmeasured.
- **The as-is posture wastes ~10.8% of its step** on a second parameter all-gather (E-006), which inflates the ratio
  against it. The final posture's overlap does not take the synchronous path that repeats the gather; production
  pays it until Megatron-LM is fixed or the overlap is turned on.

## Ladder (64 GPUs, GBS 64, iterations 151–200)

| Step | Levers | Mean step (s) | tok/s/GPU | MFU | × as-is | Training GB | Entry |
|---|---|---|---|---|---|---|---|
| 0 | as-is benchmark (mean of three runs) | 6.146 | 5,332 | 13.17% | 1.00 | 83.4 | E-001, E-003 |
| 1 | parameter all-gather overlap; timers at level 1, per-parameter gradient norms off, manual GC every 10 iterations with the setup state frozen, loss NaN check off | 4.987 | 6,570 | 16.22% | 1.23 | 83.3 | E-003 |
| 2 | + BF16 gradient reduction; chunked linear cross-entropy (8 saved chunks) | 4.666 | 7,023 | 17.34% | 1.32 | 55.8 | E-003 |
| 3 | + HybridEP dispatcher; router fusion | 4.391 | 7,463 | 18.43% | 1.40 | 55.8 | E-001 |
| 4 | + FP8 dense layers (BF16 parameters); selective recompute `[moe, shared_experts]` — the final posture | 3.770 | 8,692 | 21.46% | 1.63 | 77.1 | E-005 |

Step 4 is the ladder probe; the verified figure is E-007's paired comparison, 1.648×.

Single levers on step 3 (E-003; placement in brackets, where it spans more than one switch group the time is
biased upward by 2.5–6.6%):

| Lever on step 3 | Mean step (s) | vs step 3 | × as-is | Training GB |
|---|---|---|---|---|
| full recompute of the first 40 layers only (block) [3 groups] | 4.323 | −1.5% | 1.42 | 70.2 |
| full recompute of the first 32 layers only (block) [5 groups] | 4.176 | −4.9% | 1.47 | 77.9 |
| FP8 current scaling on the dense layers, BF16 parameters | 4.202 | −4.3% | 1.46 | 58.1 |
| fp32 SSM state by direct cast instead of a nested checkpoint [4 groups] | 4.445 | +1.2% | 1.38 | 55.8 |
| CP=4, selective recompute of the expert activation, EP all-to-all overlap | 4.045 | −7.9% | 1.52 | 83.1 |

Training GB is the reserved memory at the last logged step (last rank; the allocator keeps its high-water blocks
reserved, so it covers the training peak). The run-to-date maxima include the warm start's checkpoint-load
transient and hide the training peak below it (E-004). The gradient NaN check stays on at every step.

## Experiments

### E-007 · verification of the final posture · 2026-09-30

All from `snapshots/final1` (`bd189040` plus the quickstart files), the final posture given as its levers on the
baseline benchmark (arm `final`), against the rules in Verification design.

**Speed: goal met, ESTABLISHED at k = 4.** Four A B B A cycles, each one 16-node job (`paired/paired_cycle.sbatch`,
verdict by `paired/cycle_verdict.py`):

| Cycle | Placement | as-is (s) | final (s) | as-is / final |
|---|---|---|---|---|
| 6972738 | group 11 | 6.1473, 6.1515 | 3.7474, 3.7357 | 1.6435 |
| 6972739 | group 11 | 6.1462, 6.1360 | 3.7366, 3.7587 | 1.6387 |
| 6972741 | group 11 | 6.1674, 6.2021 | 3.7568, 3.7363 | 1.6508 |
| 6972742 | group 9 (15) + group 5 (1) | 6.4697, 6.4450 | 3.9144, 3.8734 | 1.6583 |

Geometric-mean speed-up **1.6478×, 95% CI [1.6342, 1.6615]** (half-width 0.83%): the lower bound clears 1.5 and the
half-width is under the 1% that would have extended the design to six cycles. The two-group cycle is 4–5% slower in
both postures and its ratio agrees with the others, which is what the pairing is for. Every one of the 16 runs
reaches iteration 200 with no NaN or skipped iteration.

**Iteration-1 check: PASS.** 2.008177 (jobs 6972735, 6972748) against the as-is 2.006194 in all three as-is parity
runs (6972732–4): a difference of 2.0×10⁻³, inside 5×10⁻³. The BF16 fallback (6972737) reads 2.006002.

**Loss parity: PASS**, and the BF16 fallback passes too. `loss_parity.py band` over iterations 1–500 in 50-iteration
windows, three as-is references (6972732, 6972733, 6972734, groups 10, 5 and 5; reference spread δ = 0.00831):

| Window | as-is refs (mean) | band | final posture (6972735) | BF16 fallback (6972737) |
|---|---|---|---|---|
| 1–50 | 1.61221 | [1.60385, 1.62061] | 1.61390 (+0.00169) | 1.61211 (−0.00010) |
| 51–100 | 1.54301 | [1.53461, 1.55143] | 1.54349 (+0.00048) | 1.54422 (+0.00121) |
| 101–150 | 1.56294 | [1.55395, 1.57202] | 1.56418 (+0.00124) | 1.56459 (+0.00165) |
| 151–200 | 1.56308 | [1.55336, 1.57388] | 1.56913 (+0.00605) | 1.56017 (−0.00291) |
| 201–250 | 1.57351 | [1.56422, 1.58362] | 1.57861 (+0.00510) | 1.57209 (−0.00142) |
| 251–300 | 1.58589 | [1.57451, 1.59931] | 1.58081 (−0.00509) | 1.58142 (−0.00447) |
| 301–350 | 1.56527 | [1.55350, 1.57843] | 1.56401 (−0.00125) | 1.56405 (−0.00122) |
| 351–400 | 1.54947 | [1.53985, 1.55952] | 1.55256 (+0.00310) | 1.54936 (−0.00011) |
| 401–450 | 1.56669 | [1.55736, 1.57617] | 1.57163 (+0.00493) | 1.56556 (−0.00113) |
| 451–500 | 1.55390 | [1.54357, 1.56378] | 1.55618 (+0.00228) | 1.55408 (+0.00018) |

Every window of both candidates is inside its band, and both pass the grad-norm band too; no run has a NaN or skipped
iteration. The FP8 posture sits above the references' mean in 8 of 10 windows (+0.0019 on average, about a fifth of
the reference spread), where the BF16 fallback is centred on it (−0.0008): FP8 on the dense layers costs a little
loss, inside the run-to-run band. Speed over 151–200 in these runs: 3.767 s (final, group 11) and 3.928 s (BF16 fallback,
group 11). The quickstart file itself, launched as documented with its `.env` (the ranks log
`ISAMBARD_FP32_SSM_STATE=checkpoint`), trained 200 iterations with 0 NaN at 77.1 GB reserved in two runs: 3.902 s on
a two-group placement (6972923, 14 + 2 nodes, not comparable) and **3.733 s** on one switch group (6973271).

**Checkpoint resume: PASS.** Run S (6972748) saved at iterations 10 and 20; run R (6972749) resumed from S's
iteration-10 checkpoint to 20. R's first iteration logs 704 consumed samples (11 × 64) and every iteration's learning
rate equals the straight runs'; R and S agree exactly at iteration 11 (lm loss 1.540377, grad norm 0.2880) and by at
most 1.0×10⁻⁴ over 11–20, against 4.45×10⁻⁴ between two independent straight runs of the posture (6972735,
6972490); R's mean loss over 11–20, 1.578875, lies inside their band [1.578606, 1.579149]; no NaN or skipped
iteration (`resume_check.py` in the campaign directory, the pretraining campaign's criterion).

**HF export: PASS.** S's iteration-10 checkpoint exported with the standard pipeline (1 node, TP1·EP4, job 6972919)
has the layout of a production-posture Nano-30B export (the filtered arm's continual-pretraining iteration 870):
6,243 tensors, 31.578 B elements, BF16 with 23 fp32 tensors, 256 routed-expert tensors in each of the 23 MoE
layers, identical names, shapes and dtypes, and no index/shard inconsistency (`compare_hf_exports.py`). The FP8
posture keeps BF16 primary weights, so the checkpoint holds the BF16 of the fp32 masters.

### E-006 · where the as-is step goes; the parameter all-gather runs twice · 2026-09-30

Torch-profiler traces of the as-is posture (job 6971858, ranks 0 and 63, iterations 170 and 190, no stacks; the
four step windows measure 6.143–6.164 s against the 6.146 s anchor), read with `scripts/profiling/trace_analysis/`
and per-collective scripts kept with their outputs in the campaign directory's `records/trace_budget_asis/`. Mean
over the four traces:

| Phase | Seconds | Share |
|---|---|---|
| forward, 2 microbatches (with the output layer and cross-entropy) | 1.152 | 18.7% |
| full-layer recompute (the second forward inside the backward) | 1.224 | 19.9% |
| fp32 SSM state's nested checkpoint (a third scan forward) | 0.113 | 1.8% |
| backward | 2.146 | 34.9% |
| after the backward: reduce-scatter tail, grad norm, Adam, parameter all-gather | 1.428 | 23.2% |
| step boundary (logging collectives, timers, zeroing) | 0.090 | 1.5% |

By kernel class: the EP all-to-all with its token-count gather and permutes 0.88 s (14.3%), the Mamba CP all-to-all
with its packing copies 0.67 s (10.9%), expert GEMMs 0.68 s, dense GEMMs 0.64 s, Mamba scan kernels 0.50 s,
attention 0.49 s; the all-to-alls all run on the compute stream, 1.38 s of it in total (22%). The GPU is never
launch-bound (0.08–0.15 s per step with no kernel on any stream).

**The parameter all-gather runs twice per step, fully exposed: 1.33 s (21.6%).** Each of the chained optimizer's
two distributed optimizers (dense, then expert) calls `start_param_sync_for_bucket_group_subset` from
`step_with_ready_grads` when parameter-gather overlap is off, and that walks every dense and every expert bucket
group of the model (`3rdparty/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py`, `for bucket_group in
(model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups)`), while the synchronous path of
`start_param_sync` runs its collective on every call (`param_and_grad_buffer.py`). The traces show it: 24 expert
all-gathers (2 rounds × 12 buckets) and 8 dense (2 × 4) per step, identical sizes. One round is redundant, ~0.66 s
(~10.8% of the as-is step). Every production stage of the control-pretraining curriculum runs with parameter-gather
overlap off, so they pay it too; the quickstarts' fastest configurations turn the overlap on and so do not. A fix
belongs in Megatron-LM (each optimizer syncing only the bucket groups it owns) and is outside this campaign.

Other observations: the expert gradient reduce-scatter runs in fp32 at ~20–24 GB/s per rank over the fabric (one
GPU per node in each 16-rank expert data-parallel group), and while it overlaps the second microbatch's backward
that backward is 0.30–0.37 s slower than the first's (compute kernels 16–42% slower, consistent with its 40-CTA
NCCL kernels taking SMs); the gradient NaN check adds 32 host syncs per step and 0.07–0.10 s of compute-stream
stalls; timers at level 2 add 42 device syncs per step.

### E-005 · FP8 dense layers with each memory lever; the final posture · 2026-09-30

Wave 2 combined FP8 on the dense layers (BF16 parameters) with each memory lever on step 3, all from `cal1`, every
run on a single switch group (the probes now ask SLURM for one, `--switches=1@00:05:00`):

| Posture (step 3 + FP8 dense + …) | Job | Group | Mean step (s) | × as-is | Training GB | lm loss 191–200 |
|---|---|---|---|---|---|---|
| block recompute, first 32 layers | 6972486 | 10 | 3.854 | 1.595 | 78.7 | 1.5670 |
| block recompute, first 24 layers | 6972489 | 5 | OOM at iteration ~195 | — | — | — |
| selective recompute `[moe, shared_experts]` | 6972490 | 11 | 3.770 | 1.630 | 77.1 | 1.5702 |
| block recompute, first 32 layers, fp32 SSM state by direct cast | 6972492 | 11 | 3.753 | 1.638 | 84.7 | 1.5676 |
| CP=4, `[moe_act]`, EP all-to-all overlap | 6972494 | 11 | 3.756 | 1.636 | 85.3 | 1.5664 (CP=4 data order) |

Every run has 0 NaN and 0 allocator retries. The FP8 postures' iteration-1 loss is 2.008177 against the as-is
2.006194. Block-24 ran out of memory at iteration ~195 on rank 20 with 80.4 GiB allocated by PyTorch and ~7 GiB held
outside it (the CUDA context, NCCL and HybridEP buffers), so the practical ceiling for PyTorch is ~87 GiB of the
95 GiB card; it failed late because a stored MoE layer's activations grow with how unevenly a batch routes, so a
posture that stores MoE activations across layers peaks on its worst batch, not its first. The direct-cast fp32
SSM state saves one scan forward per Mamba layer and measured −2.6% on block-32 (3.854 → 3.753 s), at +6 GB.

**The final posture is step 3 + FP8 dense + selective `[moe, shared_experts]` at CP=2** (3.770 s, 1.630×). The two
faster postures are within 0.5% of it and were set aside: block-32 with the direct cast sits at 84.7 GB, ~2 GB from
the ceiling, with MoE activations stored across layers (block-24's failure mode); CP=4 is not the ≥3% faster that
would justify its data-order change (its sampler shards by data-parallel rank, so its loss would need its own
references). Selective recompute never keeps MoE activations past their own layer, so routing imbalance moves only
one layer's transient, and it leaves ~10 GB of headroom. The direct cast cannot join it: with every Mamba layer
keeping its activations it would add ~15 GB. Its fallback, if FP8 fails the loss band, is the same posture in BF16
(estimated ~3.94 s, 1.56×), which runs the 500-iteration band beside it.

### E-004 · the peak-memory figures are the checkpoint load's, not training's · 2026-09-30

Every as-is run reports a W&B `mem-max-allocated` of exactly 89.905 GB and every posture with BF16 gradients exactly
70.811 GB, whatever else it changes — 12 layers keeping their activations (block-40) did not move it. The two differ
by 19.1 GB, the fp32-to-BF16 saving on a ~9.6-billion-parameter gradient buffer. The maxima are cumulative from the
start of the process, and the warm start's load takes each rank ~34 GB above its post-load level (`memory after
checkpoint load`: 55.19 GiB allocated as-is, 37.40 GiB with BF16 gradients) while it materialises the loaded
weights; the 13.679 GiB transposed copy of the rank's grouped experts that `torch_grouped` makes is part of it (see
the 30b_baseline README's "Segment rollover"). Training itself stays below that transient: at the last step the
as-is posture holds 83.4 GB reserved and step 3 of the ladder 55.8 GB. The ladder therefore reports reserved memory
at the last logged step, and a posture's memory smoke reads that, not the run maximum. Block recompute measured
~1.1 GB per layer that keeps its activations (70.2 GB with 12 such layers, 77.9 GB with 20).

### E-003 · the as-is anchor, the ladder's first three steps, and single levers on step 3 · 2026-09-30

All from snapshot `cal1`, exiting at 200, scored over 151–200. **As-is anchor:** 6.1375 s (C0, job 6971129),
6.1630 s (6971856) and 6.1376 s (6971857), all on switch group 11: mean **6.146 s**, SD 0.24%. A fourth as-is run
with the torch profiler capturing iterations 170 and 190 (job 6971858) measured 6.336 s and is not part of the
anchor. **Steps 1 and 2** alone (jobs 6971859, 6971861, group 11): 4.987 s and 4.666 s; step 3 is C1's 4.391 s, so
the host and parameter-gather levers give −18.9%, BF16 gradients with the chunked cross-entropy −6.4% and HybridEP
with router fusion −5.9%. lm loss over 191–200 is 1.5714 and 1.5652 against the anchors' 1.5647–1.5666.

**Single levers on step 3:** block recompute of the first 40 layers (6971863) and of the first 32 (6971865);
FP8 current scaling on the dense layers with BF16 parameters (6972099); the fp32 SSM state by direct cast
(6972098); and CP=4 with selective `[moe_act]` recompute, with and without the EP all-to-all overlap (6971869,
6971867). Results are in the ladder's second table. The block probes and the direct-cast probe ran across three to
five switch groups, so their times are upper bounds. FP8's lm loss over 191–200 is 1.5704 (the anchors span
1.5647–1.5666), which only the 500-iteration band can judge. The CP=4 runs read a different data order (DP=16
instead of 32; the launcher's `cyclic` sampler shards by data-parallel rank), so their losses are not comparable
with the CP=2 runs'; the two CP=4 runs agree with each other to 2×10⁻⁵ through iteration 35.

The CP=4 run without the overlap (6971867) stopped at iteration 37: the gradient NaN check found an Inf in bucket #0
on all four ranks of one node (nid011097), which at CP=4 is one context-parallel group. The run with the overlap
reads the same data and passed iteration 37 with a grad norm of 0.038, and nothing else in either run was out of
line, so the Inf is attributed to a transient on that node; it is not pinned to a GPU, so the node is not marked bad.

### E-001 · calibration: the as-is posture and rungs 1–3 over 300 iterations; the scoring window · 2026-09-30

Two 300-iteration runs of one snapshot (`snapshots/cal1`: `f579c940` plus the benchmark overlay), 64 GPUs, both on
switch group 11: C0, the as-is benchmark (job 6971129), and C1, the benchmark with ladder steps 1–3 (job 6971130).
Their 25-iteration means, with each block's distance from the run's iterations 201–300 mean (P):

| iterations | C0 (s) | C1 (s) |
|---|---|---|
| 1–25 | 10.105 (+65.5%) | 9.157 (+109.1%) |
| 26–50 | 6.254 (+2.44%) | 4.473 (+2.12%) |
| 51–75 | 6.199 (+1.54%) | 4.448 (+1.56%) |
| 76–100 | 6.128 (+0.37%) | 4.426 (+1.05%) |
| 101–125 | 6.136 (+0.51%) | 4.397 (+0.38%) |
| 126–150 | 6.141 (+0.59%) | 4.381 (+0.01%) |
| 151–175 | 6.167 (+1.02%) | 4.404 (+0.55%) |
| 176–200 | 6.108 (+0.05%) | 4.378 (−0.05%) |
| 201–300 (P) | 6.105 | 4.380 |

The first block is mostly iteration 1 (102.7 s in C0: kernel compilation and communicator setup); after it the step
settles to within ~2.4% of P. The window rule, fixed before the runs, takes the earliest of 51, 76, 101, 126 and 151
after which every 25-iteration mean of both runs stays within ±1% of that run's P, and iterations 151–200 if none
qualifies. None does — C0's 151–175 block is +1.02% — so the window is 151–200 and the benchmark exits at 200.

Over the window: C0 6.1375 s (5,339 tok/s/GPU, 130.4 model TFLOP/s/GPU, 13.18% MFU, peak 89.9 GB allocated / 90.1 GB
reserved) and C1 4.3909 s (7,463 tok/s/GPU, 182.3 TFLOP/s/GPU, 18.43% MFU, 70.8 / 70.9 GB): **1.40×**. lm loss over
191–200 is 1.566114 and 1.565064; neither run has a NaN or an allocator retry. The as-is posture peaks at 89.9 GB of
the card's ~95 GB; steps 1–3 peak at 70.8 GB.

### E-002 · rungs 1–3 raise the iteration-1 grad norm: HybridEP and router fusion are exact; the offset is the model's sensitivity to forward numerics · 2026-09-30

C1 (E-001) matched the as-is iteration-1 loss to 2×10⁻⁴ (2.006002 against 2.006194) but its grad norm was 3–11%
higher at each of iterations 1–5 (6.318 against 6.119 at iteration 1). With Adam's per-parameter normalisation a
scale error in one component's gradient would not show in the loss, so each numerics-changing lever of rungs 2–3 was
run alone on the as-is posture for 3 iterations (same snapshot, same data, warm start):

| run | job | iteration-1 lm loss | iteration-1 grad norm | vs as-is |
|---|---|---|---|---|
| as-is (C0) | 6971129 | 2.006194 | 6.119 | — |
| as-is repeat | 6971283 | 2.006194 | 6.118 | −0.02% |
| chunked linear CE | 6971284 | 2.006194 | 6.124 | +0.08% |
| BF16 gradient reduction | 6971285 | 2.006194 | 6.136 | +0.28% |
| HybridEP dispatcher | 6971286 | 2.005850 | 6.789 | +11.0% |
| router fusion | 6971288 | 2.007381 | 5.996 | −2.0% |

The levers that change only the backward (BF16 reduction, chunked CE) leave the iteration-1 loss bit-identical and
the grad norm within 0.3%. The two that change forward numerics move both. Each was then checked directly:

- **HybridEP**, 4 ranks over NVLink with Nano's shapes (hidden 2688, 128 experts, top-6), 8,192 and 16,384 tokens per
  rank (32,768 and 65,536 per dispatch): with the stand-in expert computing in fp32, the combined output, the input
  gradient and the routing-probability gradient match their analytic values on every rank at both sizes (0 rows off
  by more than 2%); on one rank, dispatches up to 98,304 tokens are exact too.
- **Router fusion**, Nano's router (sigmoid, top-6 of 128, expert bias, scaling 2.5) on 16,384 tokens of fp32 logits:
  the fused and unfused paths choose the same experts for every token, with routing probabilities within 1.2×10⁻⁷
  and aux-loss scores within 4×10⁻⁹.

So neither is wrong; each only perturbs the forward at the rounding level, and this model amplifies that: two as-is
runs of one snapshot agree to four digits at iteration 1 and differ by 1.5×10⁻³ in loss and 3% in grad norm at
iteration 3 (5.286 against 5.456), from run-to-run nondeterminism alone. A rounding-level change to the MoE output
flips near-tie routing decisions in later layers, and the recurrent and attention layers carry each change down the
rest of the sequence. Iteration-1 grad norms are therefore not a parity test for forward-changing levers here; the
500-iteration loss band is. The per-parameter W&B norms (`l2_norm/grad/*`) cannot settle such questions either:
`report_l2_norm_grad` (`training/utils/train_utils.py`) reads every parameter's `main_grad` on the logging rank,
and under the distributed optimizer the gradient reduce-scatter writes the reduced values only into that rank's own
shard of each bucket (`param_and_grad_buffer.py`, `dist_reduce_scatter_func(local_data_view, bucket.grad_data)`),
so the rest of the buffer still holds the rank's unreduced local gradient. For router fusion these norms read up to
2.3× the as-is values while the optimizer's true norm is 2% lower.
