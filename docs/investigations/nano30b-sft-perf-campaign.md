# Nano-30B SFT performance campaign

**Goal.** Raise the training efficiency of stage 3 of the control-pretraining curriculum, the XL SFT
(`configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml`: packed
sequences of 32,768 tokens in THD layout, TP1·CP2·EP4·PP1·ETP1, full recompute, warm-started weights-only from the
midtraining final checkpoint), to **≥ 1.5× tokens/s/GPU** on the benchmark below, with levers that keep functional
parity with the production run: the same model and checkpoints, the same data and schedule, the same loss trajectory
(Kyle, 2026-10-01). The pretraining and midtraining campaigns (`docs/investigations/nano30b-pretrain-perf-campaign.md`,
`docs/investigations/nano30b-midtrain-perf-campaign.md`) are the starting point. Two things differ for SFT: the corpus
is packed (several documents per sequence, read through `cu_seqlens`), which has its own code path and constraints, and
the production posture sits in the regime of a context-parallel partition bug that had to be fixed first (E-000).

**Result.** The final posture — `configs/quickstart/nemotron_nano_quickstart_sft.yaml`: CP=1 with the chunked
linear cross-entropy, BF16 gradient reduction in 500M-parameter buckets with the parameter all-gather overlapped,
HybridEP with the fused router on packs padded to the full length, and host settings — is **1.779× the as-is
benchmark** in four placement-controlled paired cycles, 95% CI [1.775, 1.782]: under the pre-registered rule,
**goal met, established** at k = 4 (E-008). It runs 3.689 s/iter = 8,881 tokens/s/GPU against the as-is 6.563 s =
4,993, and peaks at 70.4 GiB over all 64 GPUs where the as-is posture peaks at 85.3 (E-005). Its 500-iteration loss
fails the pre-registered band: it sits below the as-is runs' band over iterations 201–400, by at most 4.3×10⁻⁵
nats, and is back inside from 401, an excursion Kyle accepted (2026-10-02). A checkpoint it writes resumes within
the spread of straight runs and exports to HF with the as-is layout (E-005, checked before the padding mask, which
adds no checkpoint state). Two fixes to the packed SFT path came first, and both change how every packed SFT run
trains: the context-parallel partition of packed batches (E-000) and the pad tokens in the MoE routers' statistics
(E-007).

**Benchmark.** `configs/quickstart/nemotron_nano_quickstart_sft_baseline.yaml` — the production XL SFT config composed
with a small overlay (`base_config:`): GBS 64 on 64 GPUs (DP=32 at CP=2, so 2 packs per DP replica, the same per-GPU
work as production's GBS 256 on 256 GPUs), the weights-only warm start from the midtraining final that production made,
no checkpoint load or save of production's, production's gradient bucket restated (128,000,000 parameters; Megatron's
default would size it 40M at the benchmark's DP=32), and an exit at the end of the window E-001 calibrates.
`train_iters` stays 5976, so the 598-iteration warmup and the cosine decay are production's iteration for iteration.

**Metric.** Mean step time over iterations 51–100 (E-001); tokens/s/GPU = 64 × 32768 / (64 × mean step) is the
primary throughput figure. The model TFLOP/s and MFU that `scripts/telemetry/score_run.py` reports count full-causal
attention over each 32,768-token pack, which overstates the work of packed data (a pack holds about six documents,
each attending only within itself); they are quoted with that convention and do not enter any comparison. Memory is the
driver-level peak over all 64 GPUs, sampled every 15 s by the campaign's `memwatch.sh`: W&B carries only the last
rank, and an all-rank measurement on the midtraining posture at 512 GPUs found allocator retries that the last rank
never showed.

**Method.** As in the earlier campaigns: every probe is its own 16-node `pipeline_training_submit.sbatch` job launched
from a read-only snapshot of the code under test (with a `REVISION` naming the commit and any uncommitted diff), its
posture given as Hydra overrides and `ISAMBARD_ENV_OVERRIDES` lines composed from arm files. Snapshots, arm files, the
stage settings (`campaign.env`) and the run registry (`runs.tsv`) are under
`/projects/a5k/public/logs/nano_sft_perf_campaign/`. Only runs placed on a single Dragonfly switch group are compared
with each other. SFT's `batch` sampler deals each iteration's contiguous global batch to the data-parallel ranks, so
iteration k trains on the same packs at any DP or CP size — unlike the pretraining stages' cyclic sampler — and a
change of CP is a like-for-like lever here. A lever that changes numerics also needs its 500-iteration loss inside the
band of the as-is runs (`scripts/telemetry/loss_parity.py band`) before it counts.

## Verification design (fixed before the verification runs)

- **Speed.**
  - **Cycles:** A B B A (as-is, final, final, as-is), each one 16-node `paired/paired_cycle.sbatch` job. Every run exits
    at 100 and is scored over iterations 51–100, and only cycles placed on one switch group count.
  - **Estimand:** each cycle's as-is mean over its final mean, combined across cycles as a geometric mean with a 95%
    t-interval on the log ratios.
  - **Rule:** k = 4, extended to 6 if the interval contains 1.5 or its half-width exceeds 1%. Established if the lower
    bound is ≥ 1.5, met on average if only the point estimate is, not met otherwise.
- **Loss parity.**
  - **Runs:** one 500-iteration run of the final posture and three 500-iteration as-is runs (all with the partition fix
    of E-000), each on its own allocation.
  - **Test:** `scripts/telemetry/loss_parity.py band --window 50` over iterations 1–500. It passes if every window's
    mean lies inside [lowest reference − δ, highest reference + δ], δ being the largest window difference between two
    references. If exactly one window fails, a fourth as-is reference is run and that window judged again; two or more
    failing windows fail the posture. The grad norm is flagged, not failed.
  - **Also reported, not gating:** the final run's mean signed offset from the references' mean and how many windows
    fall on each side of it. The midtraining posture at 512 GPUs sat a steady +0.0015 above tight references in 9 of
    9 windows — a systematic offset the band alone does not name.
- **Numerics and launch.**
  - **Iteration 1:** the loss must be within 1×10⁻³ of the as-is loss (E-003: every precision-preserving posture
    is within 6×10⁻⁵ of it, FP8 dense layers are 3.5×10⁻³ off).
  - **NaN:** no NaN or skipped iteration. Every iteration line must carry `lm loss`, and no grad norm may be
    non-finite. The posture turns the loss NaN check off, so a divergence would otherwise be silent.
  - **Settings:** the `[env-overrides]` lines must show the posture's settings.
- **Memory.** The driver-level peak over all 64 GPUs of the 500-iteration final run (`memwatch.sh`) stays at or below
  90 GiB.
- **Checkpoints.**
  - **Resume:** a final-posture run saves at iterations 10 and 20, optimizer included. A resume from 10 must agree with
    it over 11–20 under the campaign's resume criterion (`resume_check.py`).
  - **Export:** an HF export of the final posture's checkpoint must have the as-is export's tensor names and shapes
    (`compare_hf_exports.py`).

## Ladder (64 GPUs, GBS 64, iterations 51–100)

Each run is single-group; the ratios are against the calibration's as-is run, on other allocations, so they carry the
~0.5% spread of single-group runs until the paired cycles measure the final posture.

| Step | Levers | Mean step (s) | tok/s/GPU | × as-is | lm loss 91–100 | Peak GiB, all GPUs | Entry |
|---|---|---|---|---|---|---|---|
| 0 | as-is benchmark | 6.556 | 4,998 | 1.00 | 0.992515 | | E-001 |
| 1 | parameter all-gather overlap and host settings | 5.559 | 5,894 | 1.18 | 0.992512 | | E-003 |
| 2 | parameter all-gather overlap and host settings; BF16 gradient reduction and chunked linear cross-entropy | 5.108 | 6,416 | 1.28 | 0.992511 | | E-003 |
| 3 | + HybridEP dispatcher and router fusion, packs padded to the full length | 4.756 | 6,890 | 1.38 | 0.992494 | | E-001, E-002 |
| 3 + FP8 + sel | + FP8 dense layers (BF16 parameters) and selective recompute `[moe, shared_experts]` | 4.125 | 7,944 | 1.59 | 0.995934 | 78.3 | E-003 |
| 3 at CP=1 | step 3 with context parallelism off (DP=64, one pack per replica), full recompute | 4.096 | 8,000 | 1.60 | 0.992494 | 70.4 | E-003 |
| 4 | + 500M-parameter gradient buckets | 3.672 | 8,925 | 1.79 | 0.992514 | 70.4 | E-004 |

## Experiments

### E-008 · re-verification with the routing padding mask · 2026-10-02

The final posture verified again under the rules of E-005, against as-is references that also run with the mask (E-007;
design `records/verification_design_mask_20261001T175619Z.md`, fixed before the runs, and its addendum
`records/verification_design_mask_addendum_20261001T221819Z.md`). Results: `records/verification_mask2_results_20261002.md`
(loss parity, numerics, memory) and `records/cycle_verdict_mask3.txt` (speed).

- **Runs.**
  - The first attempt (`snapshots/mask1`, jobs 6996573–6996580) failed at every run's first backward, in the
    router's expert-bias count, which lined the `[tokens]` mask up against its `[tokens, experts]` routing map. It is
    upstream Megatron-LM #6114, carried as the pin's 0006.
  - Loss parity ran from `snapshots/mask2`: as-is references `mk2_asis_a/b/c` (jobs 7002375, 7002380, 7002382) and the
    final posture `mk2_final` (7002384), 500 iterations each. That snapshot laid the mask out for sequence parallelism
    in the models rather than in the step as committed; at TP=1 neither scatters it, so the runs train the committed
    code's numerics.
  - Speed ran from `snapshots/mask3`, which is `2850cb5b` but for its CUDA-graph refusal, narrowed to upstream's rule
    after the snapshot was taken; neither posture runs CUDA graphs, so neither version of the check acts. Four A B B A
    cycles, each pinned to one switch group by excluding the others (jobs 7005942–7005945, groups 12, 9, 11 and 5).
    Four earlier cycles (7002445–7002448) were placed across four or five groups, which the rule excludes; the spread
    slowed their as-is runs about 5% and their final runs about 1%, inflating the ratios of the three that can be
    scored to 1.84–1.86.
- **Loss parity: FAIL under the pre-registered rule, accepted (Kyle, 2026-10-02).**
  - The references agree to δ = 1.96×10⁻⁵.
  - The final posture's window offsets from the references' band centre, ×10⁻⁵ over iterations 1–500: +0.1, −0.7,
    +0.1, −1.0, −2.5, −3.8, −4.3, −2.8, +0.0, +2.0. Windows 201–400 lie outside the band (at most −4.3×10⁻⁵, in
    301–350), and the run is back inside from 401. The mean offset is −1.3×10⁻⁵, with 6 of 10 windows below.
  - The grad norm lies inside every window.
  - Read from the job logs (seven significant digits, about 1% of the band's width): `mk2_asis_c`'s W&B run lost
    iterations 329–500 when W&B stalled at 23:55Z, and the job then hung at exit on W&B and hit its time limit. The W&B
    read over iterations 1–300, where every run is complete, puts the same windows outside.
  - The pad effect is gone. Unmasked, the posture sat a flat −1.8×10⁻⁵ below from about iteration 100 (E-005, E-007);
    masked, it tracks the references for 200 iterations, dips and recovers. What remains comes from a lever that
    changes numerics and was never banded alone with the mask. HybridEP, which moved iteration 1 most (−7.4×10⁻⁵,
    E-007), is the candidate; it was not run, the decision being to accept the excursion.
- **Numerics and launch: pass.** Iteration 1's loss is 1.034376 on every as-is run and 1.034350 on every final run, the
  three references and the parity run as well as the 16 runs of the speed cycles. No run has a NaN or a non-finite grad
  norm, every iteration line carries `lm loss`, and each run's `[env-overrides]` lines show
  `ISAMBARD_FP32_SSM_STATE=checkpoint` on all 16 nodes.
- **Memory: pass.** The driver-level peak over all 64 GPUs of `mk2_final` is 70.4 GiB (`memwatch.sh`, limit 90).
- **Speed: established.**

  | Cycle (job) | Group | As-is runs (s) | Final runs (s) | Ratio |
  |---|---|---|---|---|
  | 7005942 | 12 | 6.5843, 6.5701 | 3.7094, 3.6979 | 1.7759 |
  | 7005943 | 9 | 6.5546, 6.5488 | 3.6791, 3.6805 | 1.7805 |
  | 7005944 | 11 | 6.5766, 6.5649 | 3.6916, 3.6904 | 1.7802 |
  | 7005945 | 5 | 6.5546, 6.5464 | 3.6830, 3.6841 | 1.7783 |

  - The geometric mean is **1.779×, 95% CI [1.775, 1.782]**, a half-width of 0.19%. The lower bound clears 1.5, so
    the rule asks for no further cycles. The final posture runs 3.689 s against 6.563 s, 8,881 against 4,993
    tokens/s/GPU.
  - Against the unmasked cycles of E-005 (3.677 and 6.565 s, on other allocations), the as-is step is unchanged and
    the final step is 0.3% slower, inside the 0.5% spread of single-group runs on different allocations.
  - Cycle 7005942 ended 12 minutes after its last run's final iteration: rank 0's W&B upload stalled at exit, as
    `mk2_asis_c`'s did, and the other nodes' launchers left the exit barrier after its 300 s timeout. The scored
    window is unaffected.

### E-007 · the loss band's failure: pad tokens in the MoE router's statistics · 2026-10-01

The final posture failed the pre-registered band (E-005), so its levers were tested one at a time, each over 500
iterations against the same three as-is references, each run on one switch group (full-precision values from W&B,
`records/loss_band_diagnostics_wandb.json`). A fourth as-is run (`cal_asis`, 300 iterations) lies inside the band, so
the band is calibrated: a run of the as-is posture on another allocation passes it.

| Run (job) | Change from the as-is benchmark | Pad tokens per iteration | Band | Mean offset (×10⁻⁵) | Windows below |
|---|---|---|---|---|---|
| `par_r1` (6980899) | step 1, which changes no numerics | 2,505 | pass | −0.2 | 7 of 10 |
| `par_rfuse` (6980909) | step 1 and router fusion | 2,505 | pass | −0.0 | 4 of 10 |
| `par_r12` (6980900) | steps 1–2 | 2,505 | pass | +0.7 | 2 of 10 |
| `par_pad` (6980908) | step 1 and full-length packs | 4,077 | fail | −2.1 | 10 of 10 |
| `par_c1` (6980901) | steps 1–3 | 4,077 | fail | −1.9 | 10 of 10 |
| `par_cp1a2a` (6980902) | steps 1–2, CP=1, 500M-parameter buckets, alltoall | 313 | fail | +2.3 | 0 of 10 |
| `par_final` (6980716) | the final posture | 4,077 | fail | −1.8 | 9 of 10 |

- **Every lever that keeps the as-is pad count passes, and every one that changes it fails, in the direction of
  the change.** The as-is collate pads 2,505 tokens per iteration: each replica's two packs are collated together
  and the shorter is padded to the longer. Full-length packs pad 4,077; at CP=1 each pack is collated alone and
  padded only to a multiple of 16, 313. The counts are taken from the corpus's first shard over the 500 iterations.
- **The mechanism.** The pad tokens, all EOS, reached the MoE router's statistics, because the packed SFT step passed
  the model no padding mask.
  - The expert-bias update moves each expert's bias by a fixed 10⁻³ every iteration, by the sign of its load
    against the mean, whatever the learning rate.
  - The sequence auxiliary loss (coefficient 10⁻⁴) counts the pads too.
  - Identical pad tokens route to the same experts, so a change of ~2,000 of them per iteration tips those experts'
    signs. The offset appears from about iteration 100 and stays flat, about −1.1×10⁻⁸ nats per extra pad token in
    both directions.
- **Not the forward.** On iteration 1, with identical weights and data, every lever moves the loss by less than
  10⁻⁴: full-length packs 0, router fusion +1.9×10⁻⁵, HybridEP −7.4×10⁻⁵, CP=1 +0.9×10⁻⁵ (one-iteration probes
  `att_*`, jobs 6980890–6980893). Over iterations 1–100 those offsets average to zero against the references'
  per-iteration spread.
- **The fix** (Kyle, 2026-10-01: mask the pads):
  - The packed collate marks every position it or the packer padded: each document's EOS padding and everything
    after a pack's last document.
  - The training step keeps the mask on every pipeline stage, partitions it with the tokens under context
    parallelism, and passes it to the model as `padding_mask`, so the routers leave those positions out of both
    statistics.
  - Under sequence parallelism a layer's hidden states hold the tensor-parallel rank's share of the sequence, so the
    step gives each pipeline stage the mask laid out the same way: a GPT model's first stage scatters it itself, and
    the step scatters it for the later stages and for every stage of a hybrid model.
  - Upstream Megatron-Bridge fixed the same problem in #5470 (2026-08-12), in data code this repo's Bridge predates,
    and the change here follows it: the collate marks the same positions, and the step's layout is upstream's
    `_prepare_packed_padding_mask`.
  - The pinned Megatron-LM carries one upstream fix for it, 0006 in `3rdparty/patches/megatron-lm/README.md` (#6114):
    the router's expert-bias count lined the `[tokens]` mask up against the expert dimension of its
    `[tokens, experts]` routing map, so a masked training step of a router with expert bias failed, and the first
    masked runs (jobs 6996573–6996580) did, at their first backward.
  - Tests came first. They failed before each fix and with it removed, and pass with it
    (`records/padding_mask_01_before_fix.txt`, `02_fix_removed.txt`, `03_after_fix.txt`;
    `records/expert_bias_mask_01_before_fix.txt`, `02_after_fix.txt`, `03_fix_removed.txt`;
    `records/sp_padding_mask_01_layout_removed.txt`, `02_with_layout.txt`).
- **What it changes.** The as-is posture trains differently too: its own 2,505 pad tokens per iteration leave the
  statistics. Parity is therefore verified again against as-is references run with the mask, under the rules of the
  first verification, fixed before its runs (`records/verification_design_mask_20261001T175619Z.md`; its addendum
  `records/verification_design_mask_addendum_20261001T221819Z.md` names the runs that replaced the failed first
  attempt). A loss curve from before the fix is not compared with one after it.

### E-006 · where the time goes: step budgets of the as-is and final postures · 2026-10-01

Torch-profiler captures without Python stacks, ranks 0 and 63 at two iterations each: the as-is benchmark (`prof_asis`,
job 6980080, iterations 170 and 190) and the final posture (`prof_cand`, job 6980650, iterations 70 and 90).
Each iteration's wall time is split into exclusive categories; the numbers are means over the four traces
(`records/step_budget/`, which also holds the analysis scripts).

| Category (s per step) | As-is | Final |
|---|---|---|
| Parameter all-gather, exposed | 1.312 | 0.120 |
| Gradient reduce-scatter, exposed or stalling the compute stream | 0.789 | 0.062 |
| Context-parallel communication (Mamba all-to-alls, attention ring) | 0.485 | 0 |
| Context-parallel compute (THD reordering, all-to-all packing) | 0.294 | 0 |
| Expert-parallel dispatch and combine | 0.580 | 0.435 |
| Other MoE work, output layer and cross-entropy, DDP hooks, optimizer, attention | 0.987 | 0.687 |
| Dense and expert GEMMs, Mamba, norms | 1.920 | 2.314 |
| Idle on every stream | 0.211 | 0.033 |
| Other communication (timers, logging) | 0.037 | 0.054 |
| **Iteration** | **6.615** | **3.708** |

- **The as-is posture gathers its parameters twice per step.**
  - Each trace has 118 all-gather kernels moving 19.09 billion bf16 elements (38.19 GB). The gradient reduce-scatter
    covers 9.55 billion elements once.
  - With parameter-gather overlap off, each of the chained optimizer's two steps re-gathers every bucket, as the
    midtraining campaign found (its E-006). The second round alone costs about 0.71 s, 10.7% of the step.
  - The final posture gathers once, inside the forward.
- **The as-is backward stalls behind its gradient reduce-scatters.**
  - Each MoE layer's experts fill two 159.6M-parameter buckets, whose reduce-scatters queue back to back. In the
    second microbatch's backward, the compute stream's next kernel starts at the instant they finish. That happens 23
    times per step, 0.66–0.78 s in all.
  - The timing points to head-of-line blocking on the launcher's single CUDA connection
    (`ISAMBARD_CUDA_MAX_CONNECTIONS=1`). This is an inference from timing, not something the trace shows directly.
  - The final posture's buckets of about 639M parameters (two MoE layers each) stall once per step at most.
- **Context parallelism costs the as-is step 0.78 s (11.8%).**
  - 828 Mamba all-to-alls take 0.478 s, all of it exposed.
  - THD reordering takes 0.164 s and all-to-all packing copies 0.115 s.
  - Transformer Engine's partition calls take 0.6 ms of GPU time.
- **The final posture's compute takes 0.39 s longer**, mostly because the data-parallel collectives now overlap it.
  Expert GEMMs run at 15–18 ns per token alone and 29–34 ns per token under an overlapping all-gather or
  reduce-scatter.
- **Neither posture is launch-bound.** The host spends about 2 s of each step blocked in synchronisations, ahead of
  the GPU.
- **The saving, by cause:**

  | Cause | Share of the 2.9 s saved |
  |---|---|
  | All-gather | 41% |
  | Context parallelism | 27% |
  | Reduce-scatter stalls | 25% |
  | Idle | 6% |
  | Expert-parallel path | 5% |
  | Other: output layer and cross-entropy, DDP hooks, attention, MoE bookkeeping | 10% |
  | Slower compute | −14% |

### E-005 · verification of the final posture · 2026-10-01

The posture under test is ladder step 4 (`arms/cand.arms`: steps 1–3, full-length packs, CP=1, 500M-parameter buckets) on
`snapshots/cal1`, judged by the design fixed above before the runs.

**Speed: established.** Four A B B A cycles (`paired/cycle_verdict.py`), each on one switch group:

| Cycle (job) | Group | As-is runs (s) | Final runs (s) | Ratio |
|---|---|---|---|---|
| 6980709 | 10 | 6.5736, 6.5678 | 3.6750, 3.6769 | 1.7875 |
| 6980710 | 11 | 6.5775, 6.5651 | 3.6795, 3.6785 | 1.7862 |
| 6980711 | 11 | 6.5677, 6.5554 | 3.6734, 3.6781 | 1.7851 |
| 6980712 | 4 | 6.5550, 6.5573 | 3.6762, 3.6786 | 1.7828 |

The geometric mean is **1.785×, 95% CI [1.782, 1.789]**, a half-width of 0.18%. The lower bound clears 1.5, so the
rule asks for no further cycles. The final posture runs 3.677 s against 6.565 s, 8,912 against 4,991 tokens/s/GPU.

**Numerics and launch: pass.**
- Every one of the 20 runs (16 in the cycles, 4 for parity) has an iteration line with `lm loss` for every iteration,
  no non-finite grad norm, and no NaN or skipped iteration.
- Every run's `[env-overrides]` lines show `ISAMBARD_FP32_SSM_STATE=checkpoint`.
- Iteration 1 reads 1.034350 in every final run against 1.034374–1.034376 as-is, 2.6×10⁻⁵ apart, inside the
  1×10⁻³ tolerance.

**Memory: pass.** The final posture's 500-iteration run peaks at 70.4 GiB over all 64 GPUs; the as-is posture's
peaks at 85.3.

**Checkpoints: pass.**
- Resume (`records/resume_check.txt`): a run resumed from iteration 10 consumes the same samples at the same learning
  rate as the straight run. Its loss at iteration 11 is identical, and over 11–20 it stays within 4.1×10⁻⁵ of the
  straight run, inside the band of the straight runs.
- Export (`records/export_compare.txt`): the HF export has the as-is export's 6,243 tensors, names, shapes and dtypes.

**Loss parity: fails the pre-registered band, by a small margin** (`records/loss_band_par_final_wandb.txt`).
- The three as-is references agree to δ = 1.7×10⁻⁵ nats per 50-iteration window. The warmup learning rate stays at
  or below 4.2×10⁻⁶ over these iterations.
- The final posture's loss sits 1.8×10⁻⁵ below the references' mean on average, in 9 of 10 windows. It leaves the
  band in windows 101, 151, 351 and 401, by at most 2.9×10⁻⁵. Two or more such windows fail a posture.
- Its grad norm is flagged in window 1 only (−0.08%). Learning rate and consumed samples agree at every iteration.
- E-007 finds the cause.

### E-004 · CP=1: recompute depth, FP8 and the gradient bucket · 2026-10-01

Probes from `snapshots/cal1` on the CP=1 posture of E-003 (steps 1–3, full-length packs, context parallelism off: 4.096
s, 70.4 GiB), exit 100, each on one switch group, scored over iterations 51–100; peaks are the driver-level maximum
over all 64 GPUs.

| Probe (job) | Change from CP=1 | Mean step (s) | Peak GiB | lm loss 91–100 |
|---|---|---|---|---|
| `lad_cp1blk44` (6980154) | recompute only the first 44 of the 52 layers (block method) | 3.980 | 88.2 | 0.992497 |
| `lad_cp1blk36` (6980155) | the first 36 layers | out of memory in iteration 1 (NCCL `Cuda failure 2`) | 95.0 | |
| `lad_cp1sel` (6980157) | selective `[moe, shared_experts]` recompute | out of memory (PyTorch) | 95.0 | |
| `lad_cp1fp8` (6980158) | FP8 dense layers, BF16 parameters | 3.859 | 71.8 | 0.995979 |
| `lad_cp1bkt` (6980159) | 500M-parameter gradient buckets | **3.672** | 70.4 | 0.992514 |
| `lad_cp1bkt1g` (6980648) | 1G-parameter gradient buckets | 3.710 | 70.4 | 0.992503 |
| `lad_candblk48` (6980649) | 500M-parameter buckets, and recompute only the first 48 layers | 3.625 | 79.6 | 0.992488 |

- **The bucket.** Buckets of 500M parameters (`ddp.bucket_size` counts parameters) take 10.4% off the CP=1 step
  at production's 128M; buckets of 1G parameters give back 1%.
- **Recompute depth.** Every layer whose activations are kept instead of recomputed costs about 2.2–2.3 GiB per GPU
  (keeping 4 layers: +9.2 GiB; 8 layers: +17.8 GiB), and selective recompute keeps every layer's. Keeping 4 layers is
  3.625 s against 3.672 s, 1.3% on runs on two allocations, inside about twice the 0.5% spread of single-group runs and
  unpaired; it is not part of the verified posture.
- **FP8.** FP8 dense layers save 5.8% at CP=1 but shift the loss by the same +0.0035 as at CP=2, so they stay out.

### E-003 · ladder steps 1–3, FP8 dense with selective recompute, and CP=1 · 2026-10-01

Probes from `snapshots/cal1`, exit 200, each on one switch group; scored over iterations 51–100 (`records/<name>.score.txt`).
- **The loss.** Every non-FP8 posture matches the as-is loss over iterations 91–100 to 2×10⁻⁵ (0.992494–0.992515):
  this early in the warmup (learning rate below 9×10⁻⁷) the weights barely move from the warm start, so the loss is set
  by the forward pass over identical data. FP8 on the dense layers sits **+0.0034** above, a systematic forward-numerics
  shift 170 times the spread of the other postures.
- **CP=1.** Turning context parallelism off at step 3 (whole packs per rank, DP=64) runs 4.096 s with the as-is loss
  and a 70.4 GiB peak: it removes the context-parallel partition, the Mamba layers' context-parallel all-to-alls and their
  per-layer THD reordering, and it fits at full recompute now that the chunked cross-entropy no longer materialises the
  16 GiB fp32 logits that kept the production posture at CP=2. It matches the FP8 posture's speed without its loss
  shift.

### E-002 · HybridEP faults on packs of different lengths; the packs are padded to the full length · 2026-10-01

The first calibration run of steps 1–3 (`cal_c1`, job 6980032) died in iteration 1 inside HybridEP's dispatch:
`cudaErrorIllegalAddress` at `cudaMemsetAsync(preprocessing_tmp, …)` (`hybrid_ep_backend.cuh:5503`), called from
`fused_a2a.py` through the first MoE layer's forward. HybridEP sizes its buffers from the first dispatch's local token
count (`init_hybrid_ep_buffer(…, num_tokens, …)`, `megatron/core/transformer/moe/fused_a2a.py`). With the production
`pad_to_max_length: false` each pack is padded only to its own length rounded to 16, and an expert-parallel group of
four GPUs holds two data-parallel replicas' packs, so its ranks hold different token counts and a peer can send more
tokens than a rank's buffer was sized for. The pretraining and midtraining stages never meet this: every sequence there
is full length. Padding every pack to the full 32,768 tokens (`dataset.dataset_kwargs.pad_to_max_length: true`) gives
every rank 16,384 tokens at CP=2, and the rerun (`cal_c1b`, job 6980046) trained 300 iterations cleanly. The padding is
loss-masked EOS, about 0.2% of tokens (the packs are 99.8% full); it is part of the HybridEP lever wherever that lever
is used.

### E-001 · calibration: the as-is posture and steps 1–3 over 300 iterations; the scoring window · 2026-10-01

Two 300-iteration runs of `snapshots/cal1` (main `1be59df1` plus the partition fix of E-000 and the benchmark overlay):
C0, the as-is benchmark (`cal_asis`, job 6980031, group 10), and C1, the benchmark with ladder steps 1–3 and full-length
packs (`cal_c1b`, job 6980046, group 4). Their 25-iteration means against each run's iterations 201–300 mean (P):

| iterations | C0 (s) | C1 (s) |
|---|---|---|
| 1–25 | 11.193 (+70.98%) | 9.461 (+99.66%) |
| 26–50 | 6.788 (+3.69%) | 4.774 (+0.75%) |
| 51–75 | 6.548 (+0.02%) | 4.775 (+0.77%) |
| 76–100 | 6.564 (+0.28%) | 4.737 (−0.04%) |
| 101–125 | 6.561 (+0.23%) | 4.735 (−0.07%) |
| 126–150 | 6.570 (+0.36%) | 4.752 (+0.29%) |
| 151–175 | 6.561 (+0.22%) | 4.746 (+0.16%) |
| 176–200 | 6.522 (−0.37%) | 4.728 (−0.23%) |
| 201–225 | 6.528 (−0.28%) | 4.734 (−0.09%) |
| 226–250 | 6.514 (−0.49%) | 4.732 (−0.14%) |
| 251–275 | 6.546 (−0.01%) | 4.749 (+0.22%) |
| 276–300 | 6.597 (+0.77%) | 4.739 (+0.01%) |
| 201–300 (P) | 6.546 | 4.738 |

The window rule, fixed before the runs (the midtraining campaign's): the earliest of 51, 76, 101, 126 and 151 after
which every 25-iteration mean of both runs stays within ±1% of that run's P, iterations 151–200 if none qualifies.
51 qualifies, so the window is **51–100** and the benchmark exits at 100. Over the window: C0 6.556 s (4,998
tokens/s/GPU) and C1 4.756 s (6,890): 1.38×. Iteration 1 takes about 105 s (kernel compilation, communicator setup).
The step settles much faster than midtraining's (whose window was 151–200), and the first shard of the corpus, which
the benchmark reads, shows none of the slow early spans production's first shards did.

### E-000 · the context-parallel partition of packed batches: the fix, test first · 2026-10-01

The production posture — CP=2 with two packs per data-parallel replica — trained on a corrupted partition (diagnosed
2026-09-27 in `/projects/a5k/public/tmp/export_audit_20260927/_final/FINAL_REPORT.md` §2). The packed collate pads
every `cu_seqlens` row of a batch with -1 to the widest row plus one, and one collate call holds all of a replica's
packs, so a pack with fewer documents than its sibling carries several trailing pads. `_partition_packed_batch_for_cp`
(`src/megatron/bridge/training/gpt_step.py`) handed that untrimmed row to Transformer Engine's
`thd_get_partitioned_indices`, whose binary search can land on a pad and give every rank the pack's leading tokens,
while attention and the Mamba layers read the argmin-trimmed row. The audit measured about a quarter of the XL SFT's
microbatches affected; upstream fixed it (Megatron-Bridge `dec0532c19`) and the fix had not been ported.

The test, `tests/unit_tests/training/test_gpt_step_packed_cp_partition.py`, collates three packs through the real
packed dataset (one, four and five documents, so rows ending in five, two and one -1), partitions each microbatch for
every rank of CP=2 and CP=4 through the real kernel with no process group, and requires the model's own consumer
(`_undo_attention_load_balancing`, with the `get_packed_seq_params` row) to put a per-token probe back in order. On
the unfixed code the five- and two-pad packs fail at both sizes — at CP=2 both ranks hold tokens 0–23 of the
one-document pack — and the one-pad pack passes; with the fix all seven cases pass, and removing the fix's line fails
the same four again (`records/cp_partition_fix/01_before_fix.txt`, `02_after_fix.txt`, `03_fix_line_removed.txt`).
The fix trims the row with the same `trim_padded_cu_seqlens` that `get_packed_seq_params` uses before partitioning.

Every packed run at CP>1 with more than one pack per replica trains differently from a checkout that contains the fix
on: its logged loss reads about 0.03–0.04 nats above the same run before the fix (the audit's estimate), so loss curves
from before and after the fix are not compared. The production configs are unchanged; a run picks the fix up from the
code it is launched from.
