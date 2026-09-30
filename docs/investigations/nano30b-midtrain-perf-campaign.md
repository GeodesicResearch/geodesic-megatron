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

**Benchmark.** `configs/quickstart/nemotron_nano_quickstart_midtrain_baseline.yaml` — the production midtraining
config composed with a small overlay (`base_config:`): GBS 64 on 64 GPUs (DP=32 at CP=2, so 2 microbatches per DP
replica, the same per-GPU work as production's GBS 512 on 512 GPUs), the weights-only warm start from the stage-1
final (iteration 29881) that production's first segment made, no checkpoint load or save of production's, and an
exit inside the window E-001 calibrates. `train_iters` stays 3126, so the 100-iteration warmup and the cosine anneal
are production's iteration for iteration.

**Metric.** Mean step time over the calibrated window (node-hours scale with the mean), reported with the median,
p10/p90 and outliers by `scripts/telemetry/score_run.py`. tokens/s/GPU = 64 × 32768 / (64 × mean step). Model
TFLOP/s and MFU use the estimator's 24.432 GFLOP/token at seq 32768 (`scripts/nemotronh_flops_estimator.py`) and the
GH200 dense BF16 peak (989.4 TFLOP/s). Memory is the W&B summary's run maximum (last rank) with its allocator-retry
count; the log's single memory line is rank 0 after iteration 1.

**Method.** As in the pretraining campaign: every probe is its own 16-node `pipeline_training_submit.sbatch` job
launched from a read-only snapshot of the code under test (each with a `REVISION` naming the commit and any
uncommitted diff), with its posture given as Hydra overrides and `ISAMBARD_ENV_OVERRIDES` lines composed from arm
files. Snapshots, arm files and the run registry (`runs.tsv`) are under
`/projects/a5k/public/logs/nano_midtrain_perf_campaign/`. Only runs placed on a single Dragonfly switch group are
compared with each other (the `[run-identity] switch placement` line; a multi-group run is 2.5–6.6% slower,
pretraining campaign E-047). A lever that changes numerics also needs its 500-iteration loss inside the band of the
as-is runs (`scripts/telemetry/loss_parity.py band`) before it counts.

## Ladder (64 GPUs, GBS 64, iterations 151–200)

| Step | Levers | Mean step (s) | tok/s/GPU | MFU | × as-is | Peak GB | Entry |
|---|---|---|---|---|---|---|---|
| 0 | as-is benchmark | 6.138 | 5,339 | 13.18% | 1.00 | 89.9 | E-001 |
| 1–3 | parameter all-gather overlap; timers at level 1, per-parameter gradient norms off, manual GC every 10 iterations with the setup state frozen, loss NaN check off; BF16 gradient reduction; chunked linear cross-entropy (8 saved chunks); HybridEP dispatcher; router fusion | 4.391 | 7,463 | 18.43% | 1.40 | 70.8 | E-001 |

Peak GB is the W&B run maximum of allocated memory (last rank). The gradient NaN check stays on at every step.

## Experiments

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
