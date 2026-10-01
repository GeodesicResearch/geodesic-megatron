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
| 2 | parameter all-gather overlap and host settings; BF16 gradient reduction and chunked linear cross-entropy | 5.108 | 6,416 | 1.28 | 0.992511 | | E-003 |
| 3 | + HybridEP dispatcher and router fusion, packs padded to the full length | 4.756 | 6,890 | 1.38 | 0.992494 | | E-001, E-002 |
| 3 + FP8 + sel | + FP8 dense layers (BF16 parameters) and selective recompute `[moe, shared_experts]` | 4.125 | 7,944 | 1.59 | 0.995934 | 78.3 | E-003 |
| 3 at CP=1 | step 3 with context parallelism off (DP=64, one pack per replica), full recompute | 4.096 | 8,000 | 1.60 | 0.992494 | 70.4 | E-003 |

## Experiments

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
