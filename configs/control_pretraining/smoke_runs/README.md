# `smoke_e2e_run` — a 5B-token pass through the whole 30B baseline curriculum

Three chained runs of ~1.7B tokens each that execute
[`configs/control_pretraining/30b_baseline/`](../30b_baseline/)'s three stages end to end.
The question they answer is **"do these configs run, fit, and how fast?"** — not "is the model
any good". At 5B tokens the chain sees ~0.9% of the real curriculum, so the resulting model is a
pipeline artifact and nothing more.

| Stage | Config | Mode | Warm-starts from |
|---|---|---|---|
| 1 | [`nemotron_nano_30b_baseline_pretrain_smoke.yaml`](nemotron_nano_30b_baseline_pretrain_smoke.yaml) | `nano pretrain` | nothing (random init) |
| 2 | [`nemotron_nano_30b_baseline_midtrain_smoke.yaml`](nemotron_nano_30b_baseline_midtrain_smoke.yaml) | `nano pretrain` | stage 1's final checkpoint |
| 3 | [`nemotron_nano_30b_baseline_sft_smoke.yaml`](nemotron_nano_30b_baseline_sft_smoke.yaml) | `nano sft` | stage 2's final checkpoint |

Each stage runs **100 iterations x 16,777,216 tokens/iter = 1,677,721,600 tokens**, so the chain
totals **5,033,164,800**. All three stages already share one token batch at full scale — stage 1
is GBS 2048 x seq 8192, stages 2 and 3 are GBS 512 x seq 32768, and both products are 16,777,216 —
so a single iteration count gives all three the same budget and keeps the optimizer's token batch
continuous across both boundaries, exactly as the real curriculum does.

## What is deliberately identical to the full-scale configs

The data blend, the parallelism, the recompute posture, and the save-crossing settings
(`ckpt_assume_constant_structure: false`, `cross_entropy_loss_fusion: false`,
`dist.distributed_backend: "cpu:gloo,cuda:nccl"`) are carried over unchanged. A smoke that
relaxed any of them would stop testing the config it exists to certify — the DP=512 save-crossing
pathologies in particular only appear when those settings are exercised as shipped.

`tests/unit_tests/test_control_pretraining_smoke_runs.py` pins each smoke config's blend and
topology to its parent, so the two cannot drift apart silently.

### Why these are files rather than Hydra overrides

Elsewhere the repo prefers overrides — "Scaling out to 128 GPUs is an OVERRIDE, not a second
config", and profiling "runs against the STANDING quickstart config with overrides ... there is
no separate profile config to drift out of sync". The cost of not doing that here is real: each
file is a ~200-line near-copy of its parent, rationale comments included, differing only in the
eight values tabulated below.

**Seven of those eight are pipeline parameters, not infrastructure flags.** `train_iters`,
`save_interval`, `save_optim`/`save_rng`, `lr_warmup_fraction`, `lr_wsd_decay_iters` and the
checkpoint paths all define what the run *does* and are needed to reproduce it — and the repo's
config-driven-operations rule is that such values belong in a config file rather than on a
command line. Expressing the smoke chain as overrides would satisfy one convention by breaking
another, and the reproducibility it would cost is exactly what a smoke run exists to provide:
its output is evidence about a 501B-token config, and evidence is only as good as the ability to
say later precisely what produced it. Twenty-four overrides across three command lines live in
shell history; three committed files diff against their parents and are what the pinning test
asserts on. The campaign's other arms (V1, `30b_baseline`, `cpt_validation`) are one-YAML-per-run
for the same reason.

The two CLAUDE.md precedents are also narrower than they look: the 128-GPU case overrides
exactly one field, and the profiling posture is an ad-hoc benchmark rather than a config under
certification.

None of that makes the duplication safe, and the honest record is that drift has already
happened once: the SFT smoke shipped with its parent's `tensorboard_dir`, which would have
written TensorBoard events into the production stage-3 directory — the sharing CLAUDE.md's
"TensorBoard on NFS" section records as a cause of cascading stale-file-handle crashes. The
response was to widen the guard to the logger's output paths as well as the checkpoint ones
(`test_no_output_path_is_copied_from_the_parent` alongside
`test_writes_only_into_the_smoke_checkpoint_tree`), but a test that catches copies is a
weaker guarantee than a structure with nothing to copy. **If this arm is revised again, moving
the smoke chain to overrides is the change to make.**

## What differs, and why

| Field | Full scale | Smoke | Why |
|---|---|---|---|
| `train_iters` | 29881 / 3126 / est. 2981 | **100** | the 1.7B budget |
| `save_interval` | 2264 / 1564 / 1000 | **1000000** | only Megatron's unconditional end-of-training save runs |
| `save_optim`, `save_rng` | `true` | **`false`** | the next stage warm-starts from weights only, so moments and RNG would be written and never read |
| `most_recent_k` | -1 / -1 / 2 | **-1** | exactly one checkpoint exists |
| `lr_warmup_fraction` (stage 1) | 0.01 | **0.10** | 0.01 of 100 iters is 1 iteration of warmup before sitting at 1e-3 from random init — a real divergence risk that would waste the run diagnosing the smoke instead of the config |
| `lr_wsd_decay_iters` (stage 2) | 3126 | **100** | tracks `train_iters` so the anneal still spans the stage |
| checkpoint paths | `.../control_pretrain_30b_baseline_*` | `.../smoke_e2e/*` | never write into the real run's directories |
| `wandb_exp_name` | `control_pretrain_30b_baseline_*` | `smoke_e2e_30b_baseline_*` | separate W&B runs |

**The peaks and decay styles are not scaled down.** Stage 1 keeps its constant 1e-3, stage 2
keeps cosine from 7.5e-4 with `lr_warmup_iters: 100`, stage 3 keeps 5e-6 cosine at 10% warmup.
Those schedules are what the real run takes, and a smoke on a different schedule would not
certify them. Stage 2's warmup stays at 100 iterations even though that is 16.8% of the
shortened stage: it is five second-moment EMA timescales at `adam_beta2: 0.95`, a property of
the optimizer rather than of the run length, and the moment reset it protects against happens
here exactly as it does at full scale.

## Launch

Sequential — each stage's `pretrained_checkpoint` is the previous stage's `save`, and
`CheckpointConfig.finalize` asserts that path exists, so stage N+1 cannot even start until
stage N has written.

```bash
# Stage 1
isambard_sbatch --nodes=128 pipeline_training_submit.sbatch \
  configs/control_pretraining/smoke_runs/nemotron_nano_30b_baseline_pretrain_smoke.yaml \
  nano pretrain --disable-ft

# Stage 2, after stage 1 completes
isambard_sbatch --nodes=128 pipeline_training_submit.sbatch \
  configs/control_pretraining/smoke_runs/nemotron_nano_30b_baseline_midtrain_smoke.yaml \
  nano pretrain --disable-ft

# Stage 3, after stage 2 completes AND the packed SFT corpus exists
isambard_sbatch --nodes=128 pipeline_training_submit.sbatch \
  configs/control_pretraining/smoke_runs/nemotron_nano_30b_baseline_sft_smoke.yaml \
  nano sft --disable-ft
```

`--disable-ft` is required rather than optional: the ft heartbeat SIGKILLs a healthy job at
7200 s, and these stages write no checkpoint before the end, so a heartbeat kill would restart
from iteration 0 with an empty checkpoint directory and make no net progress. Each stage is
well under the 2 h wall on its own, but the flag removes the failure mode entirely.

## Duration — measured for stages 1-2, estimated for stage 3

| Stage | s/iter | 100 iters | Confidence |
|---|---|---|---|
| 1 — pretrain | 5.25-6.36 | **~9-11 min** | **measured twice** — the range is placement, see below |
| 2 — midtrain | **8.34** | **~14 min** | **measured** 2026-08-21 at 508 GPUs, 8 switch groups |
| 3 — SFT | ~5-7 | **~8-12 min** | **estimate** — derived, not measured |

**~25 min measured for stages 1-2, plus ~8-12 min estimated for stage 3**, plus ~10 minutes of startup per stage. At 100 iterations
stage 2 spends its whole length in `lr_warmup_iters: 100`, so it exercises the CP=2 / 32768 memory
and throughput posture — the real open question — but not the cosine anneal or `min_lr`.

Stage 1's range is not uncertainty about the config — it is **Dragonfly placement**, and the
two ends are both measured on this topology (DP≈512, seq 8192, GBS≈2048):

| run | nodes / groups | s/iter | TFLOP/s/GPU |
|---|---|---|---|
| 500B baseline, job 6001811 (2026-08-17) | 128 across **2** groups (g9:66, g11:62) | 5.25 | 137.8 |
| smoke stage 1, tunnel 6064751 (2026-08-20) | 127 across **8** groups | 6.36 | 114.7 |

Only cross-node traffic is the data-parallel gradient exchange (TP=CP=PP=1; EP=4 stays on
NVLink), and it is a global collective over every rank, so its cost tracks how much of it lands
on scarce inter-group links. Group membership is `(nid - 10000) // 110 + 2`; 110 nodes per group
means 128 nodes cannot fit in one, so **2 groups is the floor** for this job size. The gap is
~18% per-GPU efficiency — correlational (two runs, not a controlled A/B), but every
performance-relevant config field is identical between them. For the 501B production run that
is ~43.5 h vs ~53 h.

**`--switches` cannot buy the good placement here.** `TopologyPlugin` is `topology/tree` so the
flag functions, but a job's `@<max-time>` is capped by the site `MaxSwitchWait`, which this
cluster leaves at SLURM's 300 s default — so any generous timeout collapses to five minutes and
falls back to a scattered allocation. Forcing it needs `--exclude` of every unwanted group,
which has no cap but an unbounded wait: 128 nodes inside 2 groups means 58% of that 220-node
pool free at once (39% across 3 groups, 29% across 4) against a cluster that typically runs
~75% allocated.

Compact placement also happens by luck: job 6001811 drew 2 groups with **no** topology request
and waited 16 h 54 m, a typical wait for 128 nodes. So the cheap approach is to submit
unconstrained, check the spread at startup, and requeue only on a bad draw.

The **4.276 s/iter** figure attributed to the 500B baseline elsewhere in the repo is not
reproduced by that job's own log, which shows 5.25 s/iter steady state (iterations 29-38, where
it is flat to ~1%). Use the two measurements above.

Stage 3 has no measurement behind it, and that is the honest state: its posture is stage 2's,
but it reads a packed chat corpus rather than `.bin/.idx` and has not been run. The figure above
extrapolates from the 32K Nano SFT quickstart's 9.767 s/iter at GBS 128 on 64 GPUs, which runs 4
microbatches per replica against this stage's 2 — halving the per-replica work while widening
data parallelism 8x. Treat it as the hypothesis being tested.

Wall-clock will be dominated by queue wait rather than compute: the cluster typically runs with
most nodes allocated and the queue is FIFO rather than fairshare, so a 128-node job waits behind
whatever is ahead of it.

## What counts as a pass

- Every stage reaches iteration 100 and writes its final checkpoint.
- No OOM. Stage 2 is where this is genuinely in question — at seq 32768 the fp32 cross-entropy
  logits are 8.00 GiB even after CP=2 halves them.
- Loss descends and there are **0 NaN**. From random init stage 1 should fall from ~12.2; stages
  2 and 3 start from a warm model, so watch instead for a spike in the first ~20 iterations,
  which is the signature of the weights-only warm start's zeroed Adam moments.
- Each stage's successor loads its predecessor's checkpoint without a shape or key error — this
  is the part of the chain that no single-stage test covers.
- Record the measured s/iter for each stage.

## What the smoke measured — 2026-08-21

Stages 1 and 2 ran back to back on **127 nodes / 508 GPUs** inside a code-tunnel allocation
spanning **8 switch groups** (`group10:47, group2:17, group9:16, group12:15, group13:12,
group5:11, group4:7, group8:3`). The 128th node was excluded: `nid010943` GPU3 was wedged at
100% utilisation with no process attached, which hangs a full-width run at N-1/N store clients.

| Stage | s/iter | TFLOP/s/GPU | Loss | Wall-clock |
|---|---|---|---|---|
| 1 — pretrain | **6.36** (iters 60-130, 6.8% spread) | 114.7 | 12.20 → 6.972 | ~11 min |
| 2 — midtrain | **8.34** (iters 20-51, 82.7% spread) | 103.7 | 6.956 → 5.804 | ~14 min |

Both reached iteration 100 with **0 NaN and 0 skipped iterations** and wrote `iter_0000100`.

Three things this established:

- **CP=2 at seq 32768 fits**, at DP=254. The stage-2 posture had never been executed at this
  scale, and the fp32 cross-entropy logits are 8.00 GiB even after CP halves them.
- **The weights-only warm start does not spike.** Stage 1 ended at 6.972 and stage 2 opened at
  6.956 — the moment reset that `adam_beta2: 0.95` plus the 100-iteration warmup exist to cover
  did not show up in the loss.
- **Stage 2 is slower than the estimate.** 8.34 s/iter against the ~5-7 extrapolated, so full
  stage 2 is ~7.2 h rather than ~4.3-6.1 h. Its 82.7% iteration-to-iteration spread is also far
  wider than stage 1's 6.8% at the same placement, and is not yet explained; CP=2 adds
  cross-node context-parallel traffic that stage 1 (CP=1) does not have.

**At 508 GPUs both stages need `train.decrease_batch_size_if_needed=true`**, which the launch
block above does not pass because it asks for the full 128 nodes. At 127 nodes DP is 508 for
stage 1 and 254 for stage 2, neither of which divides its `global_batch_size`, so Megatron
rounds GBS to 2032 and 508 respectively — both giving 16,646,144 tokens/iter, still identical
across the two stages. The skipped samples — 16 per iteration at stage 1, 4 at stage 2 — are booked as
`skipped_train_samples`, so the token accounting stays truthful.

## Prerequisite

Stage 3 reads the packed chat corpus at
`/projects/a5k/public/data/geodesic-research__pa-warm-start-sft-heavy-25b-mix/packed/`, which is
built by the prepare job described in
[`../30b_baseline/README.md`](../30b_baseline/README.md). Stages 1 and 2 read only the
`.bin/.idx` corpora and have no such dependency.
