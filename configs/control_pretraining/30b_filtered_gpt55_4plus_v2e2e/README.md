# Control-pretraining 30B narrowly filtered, V2 E2E (canary OR `judge_score >= 4`, pretraining AND midtraining)

V2's rule applied from the first pretraining token. V2
([`../30b_filtered_gpt55_4plus_v2/`](../30b_filtered_gpt55_4plus_v2/README.md)) anneals the Broadly
Filtered arm's pretraining through narrowly filtered midtraining, so it is broadly filtered through
501.3B tokens and narrowly filtered through 52.4B. This arm (Kyle, 2026-09-30) trains its own stage 1
from scratch on the `_filtered_gpt55_4plus_v2e2e` pretraining splits and then runs V2's midtraining
from that stage's final, so it is narrowly filtered throughout. Against the unfiltered baseline
([`../30b_baseline/`](../30b_baseline/README.md)) it differs in its data and, in both stages, in its
training configuration (below).

| Stage | Config | Context | Iterations | Topology | Checkpoints |
|---|---|---|---|---|---|
| pretraining | `nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.yaml` (+ `.env`) | 8192 | 29881 | TP1·CP1·EP4·PP1·ETP1, DP=512 on 512 GPUs | 14 |
| midtraining | `nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_midtrain.yaml` (+ `.env`) | 32768 | 3126 | TP1·CP2·EP4·PP1·ETP1, DP=256 on 512 GPUs | 6 |

Both stages run at the baseline's own widths (its stage 1 at DP=512, its midtraining at DP=256): the
cyclic sampler buckets each epoch by data-parallel rank, so the width decides which samples each step
reads. The continual-pretraining pair is the `filtered_gpt55_4plus_v2e2e` family of
[`../30b_trustedmonitor/`](../30b_trustedmonitor/README.md) (five epochs of 76 iterations, 64 nodes,
as-is), on the union `reintroduction_gpt55_4plus_v2e2e` at `61c9d1d2`.

## What differs from the baseline

**The data.** Every document carrying a canary string, or scored >= 4 by the GPT-5 judge
(`sudoers/control-pretraining-filter-annotated` @ `91f53004`, every escalated document judged), is
removed from every corpus of both stages; every other document is kept unchanged. Stage 1 reads five
corpora this arm builds (`corpora.tsv`) and V2's `ai_safety_and_adjacent_filtered_gpt55_4plus_v2`
build, whose `_v2e2e` split holds the same rows; the midtraining reads V2's ten corpora in place.
Corpus-level blend weights are the baseline's; ClimbMix's eight shard weights are this arm's own,
token-proportional over its measured shards.

**The stage-1 posture.** Stage 1 trains in the fast Nano pretrain posture: the fields
`configs/quickstart/nemotron_nano_quickstart_pretrain.yaml` sets, at its values (the tests share one
table, `FAST_PRETRAIN_LEVERS` in `tests/unit_tests/campaign_config.py`), with that quickstart's
launcher settings in `nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.env`. Three of its levers
change numerical precision (FP8 current scaling on the dense layers, the BF16 gradient reduce, and
the BF16 SSM state, `ISAMBARD_FP32_SSM_STATE=0`), so this arm's stage 1 differs from the baseline's
in more than data, and a comparison of the two attributes to the filtering only what exceeds that
difference. Two gates bound it: the probe, before the launch, and the loss gate, during stage 1.
One lever of the quickstart's posture is not taken: stage 1 keeps the gradient NaN check on (Kyle,
2026-10-01), so a non-finite gradient ends the run before the optimizer applies it. If a gate fails
or the watch stops the stage, stage 1 stops and is debugged in this posture (Kyle, 2026-10-01); it is
never restarted in another, and no other posture is staged for it.

**The midtraining configuration.** The midtraining trains in the fast Nano midtraining configuration less
its selective recompute (Kyle, 2026-10-01): the fields `configs/quickstart/nemotron_nano_quickstart_midtrain.yaml`
sets, at its values (`FAST_MIDTRAIN_LEVERS` in `tests/unit_tests/campaign_config.py`), except that the stage
keeps the baseline's full recompute, because at 512 GPUs the selective recompute retried the allocator; launched
with
`nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_midtrain.env`, which keeps the fp32 SSM-state patch on. Two
of its levers change numerical precision (FP8 current scaling on the dense layers and the BF16 gradient
reduce), so the arm differs from the baseline in precision in both stages, not only in data; the
midtraining probe (below) bounds the midtraining's share. The gradient NaN check stays on, so a non-finite
gradient ends the run with a rejected result before its iteration line is written; the loss NaN check is
off, so a NaN loss shows as an iteration line without `lm loss` and an infinite one as `lm loss: INF`. The
midtraining's watcher stops on each. The continual pretraining runs
as-is, with the fp32 SSM-state patch on and no launcher settings file: its links train as V2's midtraining
config (the family's `posture_config` in the reintroduction chain), which is this arm's midtraining without
the fast configuration's levers, and a test holds each link to the same V2 link outside the chain's own fields.

## Build and verify

The five corpora are table-driven like every arm's: `corpora.tsv` (all five tagged `pretraining`,
ClimbMix sliced into eight shards) and the prepare config
`data/control-pretraining-datasets-filtered-gpt55-4plus-v2e2e.yaml`.
The config's `revision` is pinned at `a815dfe7`, the commit that published the splits and
`filter_stats_gpt55_4plus_v2e2e`, and every row's `docs` is that statistics config's `n_retained`,
filled in the same change; a test keeps the two moving together. The arm's other ten corpora are
V2's builds, read in place. The build:

```bash
bash configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/corpora.tsv pretraining
python configs/control_pretraining/verify_corpora.py \
  configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/corpora.tsv
```

then the content and canary audit against the baseline's build with `audit_corpora.sbatch` (one job
per corpus; the campaign README has the arguments). The ClimbMix shard weights follow from the verified
shards' measured tokens, and the launch waits on every audit. The JSONL intermediates are deleted once a
corpus verifies.

ClimbMix at `a815dfe7` verified on 2026-10-01: 553,309,172 documents (the table's count) and
354,415,202,966 tokens over eight slices of 69,163,646-647 documents and 37,002,776,302 to
49,286,097,228 tokens. Each shard weight is `round(0.698180 x shard_tokens / climbmix_tokens, 6)`,
with the rounding residue (-0.000001) folded into the largest shard, shard 1. The weights, shards 0
to 7, are 0.094711, 0.097090, 0.092885, 0.087379, 0.082384, 0.072894, 0.082214 and 0.088623.

## The probe

`probe/probe.sbatch` runs once before stage 1 launches, on the **baseline's** tokenized data, so it
runs while the corpora are still being built. It takes one 130-node job (128 for the launches, two
spares) of about 2.5 h, measured nowhere at this width before:

| Step | Config | Decides |
|---|---|---|
| NVLink sweep | `nvidia-smi nvlink --status` on every node, `scripts/training/nvlink_health.py` | launches on 128 nodes whose GPUs all report 18 active links, then registers the others as bad nodes; with fewer than 128 healthy it launches nothing and registers none, since that points at the sweep, not the nodes |
| fast | `probe/probe_fast.yaml` + the stage's `.env` | speed and memory, over 500 iterations with saves at 150, 300, 450 and 500 |
| handoff | `probe/probe_handoff_midtrain.yaml` + the midtraining's `.env` | the fast probe's save loads weights-only into this arm's midtraining and trains 5 iterations |
| as_is | `probe/probe_as_is.yaml` | the reference rerun for the parity and speed tests |
| score_gate | `score_gate.yaml`, `scripts/telemetry/score_gate.py` | the memory and speed gates below, from the fast and as-is scores |
| parity | `loss_parity.py band` | the fast probe against the baseline's run and the rerun, window 50: gated over iterations 51-500, reported over 1-50 |

The gates, fixed before the probe runs. The memory and speed thresholds are the ones
`score_gate.yaml` holds, which the probe evaluates in its gating `score_gate` step; a score step's
own exit status says only that the log could be scored. The job exits 0 only when every gate passed.

- **Speed** (`fast.score.json` and `as_is.score.json`, mean s/iter over iterations 101-500): the fast
  probe's s/iter divided by the as-is rerun's on the same nodes, times the baseline's own 6.31 s: up
  to 4.5 s go; 4.5 to 5.25 go and report; above 5.25 the gate fails and the choice goes to Kyle. Placement alone moves this posture's
  speed by up to ~18% at this width (2 against 8 switch groups), so the gate reads the ratio, which
  the two runs share a placement for, and not either run's absolute s/iter, which is reported beside
  it with the run's `[run-identity] switch placement`.
- **Memory** (the fast probe, `peak_memory_across_ranks` in `fast.score.json`, read from the run's
  `[peak-memory]` line): no OOM through all four saves, 0 allocator retries on every rank, and over
  every rank a peak allocated memory of at most 85.5 GB. The summary counts decimal GB (bytes / 1e9),
  as W&B's memory figures do; the card is 97,871 MiB (102.6 GB). The retry count is the guard that
  binds: the pool the allocator can reserve ends near 95 GB on this posture (93.6 GB has run with
  ~1.4 GB to spare; runs at 94.6-95.3 GB retried, up to 120 times), and the EP overlap reserves about 13 GB
  beyond what it allocates (`docs/investigations/nano30b-pretrain-perf-campaign.md`, E-050), so
  retries begin near 82 GB allocated. The allocated bound is the backstop for a run that reaches it
  without retrying. A run without the line had its loop cut short and fails the gate. Reserved memory
  itself is reported, not gated: the caching allocator keeps freed blocks reserved, so it fills
  whatever is free (88-91 GB on this posture at 64 and 256 GPUs) and measures the cache, not demand.
- **Parity** (`parity_band.json`): the rerun and the baseline read the same batches, so their band is
  the as-is posture's run-to-run nondeterminism, and the fast probe must stay inside it in every
  window of iterations 51-500, on both sides. A FAIL holds the launch while the fast posture is debugged. Iterations
  1-50 (`parity_band_reported.json`) are reported, not gated: they lie in the steep early descent,
  where a window mean mostly measures the rate of descent, and at 256 GPUs the fast posture left the
  band there only, 0.0035 below its edge (`docs/investigations/nano30b-pretrain-perf-campaign.md`,
  E-063). Grad norm is a flag, as in every band test.
- **Handoff** (`handoff.evidence.tsv`, the `handoff_evidence` step): the midtraining loads the probe's
  save, installs the fp32 SSM-state patch on all 512 ranks and exits at iteration 5.
- **Export**, run after the probe as publishing will run it: the probe's last save, cloned and
  exported on one node, verifies by tensor names.

  ```bash
  python3 -c 'from pathlib import Path; from scripts.hub.publish_models import make_export_clone; make_export_clone(Path("/projects/a5k/public/checkpoints/megatron/v2e2e_probe/fast/iter_0000500"), Path("/projects/a5k/public/checkpoints/megatron/v2e2e_probe/export/iter_0000500"))'
  ISAMBARD_SBATCH_FORCE=0 isambard_sbatch --nodes=1 --time=00:30:00 pipeline_checkpoint_submit.sbatch \
    export /projects/a5k/public/checkpoints/megatron/v2e2e_probe/export \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --iteration 500 --tp 1 --ep 4 --no-reasoning
  python3 -c 'from pathlib import Path; from scripts.hub.publish_models import verify_export; print(verify_export(Path("/projects/a5k/public/checkpoints/megatron/v2e2e_probe/export/iter_0000500/hf")))'
  ```

Both probes are built from the steps in `scripts/training/probe_job.sh` (the start checks, the NVLink selection,
each launch under its time limit, scoring, the handoff evidence, the parity band and the steps' record). Submit
it from a frozen copy of the commit under test, never from a working checkout (bash reads the
sbatch and the launcher by offset while they run; the sbatch refuses a directory without a
`REVISION` file), from a shell carrying no launcher, activate or container setting: the sbatch refuses
every `ISAMBARD_*`, `TRAIN_*` and `GEODESIC_CONTAINER_*` variable except the submission wrapper's own
`ISAMBARD_SBATCH_*`, the tunnel's `ISAMBARD_TUNNEL_*` and the site's `ISAMBARD_HOST` (the system's name,
which SLURM sets in every task's environment), so each step's posture is its config, its
settings file and the committed container config only. 130 nodes exceed the default node cap of 128, so the command raises it to
the account's 256 for the submission and, exported with it, for the job's check at start. Each launch
runs under a time limit of about 1.5 times its estimate, so a hung step ends without taking the rest
of the allocation. The copy needs the pinned Megatron-LM and its built dataset helpers:

```bash
SNAP=/projects/a5k/public/logs/control_pretraining/v2e2e_probe/code-$(git rev-parse --short HEAD)
mkdir -p "$SNAP/3rdparty/Megatron-LM"
git archive HEAD | tar -x -C "$SNAP"
git -C 3rdparty/Megatron-LM archive HEAD | tar -x -C "$SNAP/3rdparty/Megatron-LM"
cp 3rdparty/Megatron-LM/megatron/core/datasets/helpers_cpp*.so "$SNAP/3rdparty/Megatron-LM/megatron/core/datasets/"
git rev-parse HEAD > "$SNAP/REVISION"
cd "$SNAP" && ISAMBARD_SBATCH_FORCE=0 ISAMBARD_SBATCH_MAX_NODES=256 isambard_sbatch \
  configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/probe/probe.sbatch
```

Results land in `/projects/a5k/public/logs/control_pretraining/v2e2e_probe/<job id>/` (`steps.tsv`
lists every step's exit status and whether it gates). The scratch checkpoint directory
(`/projects/a5k/public/checkpoints/megatron/v2e2e_probe/`, one 316 GB save at the end) and the probe's
W&B runs are deleted once the results are signed off; the sbatch refuses to start while that
directory exists.

## The loss gate

`loss_gate.yaml` is pre-registered: three gates, each a band test of stage 1's log against as-is
stage-1 runs of the campaign, evaluated by `scripts/telemetry/loss_gate.py`. Its header records the
references, the calibration and the policy. In short: L1 decides at iteration 1200, L2 and L2b
confirm at iteration 2000, all before the first save at 2264. A FAIL stops the run, and stage 1 is
debugged in the fast posture, never restarted in another; NOT EVALUATED (exit status 2) blocks like a FAIL unless it is
resolved before 2264.

```bash
python scripts/telemetry/loss_gate.py \
  --spec configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/loss_gate.yaml \
  --candidate <stage-1 segment log> --gate L1
```

The gates see a regression of about 0.05 in lm loss over iterations 1-2000, and a shift of 0.02 over
1201-2000; a smaller precision effect passes them.

## The midtraining probe

`probe/probe_midtrain.sbatch` measures the arm's midtraining configuration where its campaign could not: at
production's width (512 GPUs, CP2, DP=256). It runs on the **baseline's** stage 2
(`../30b_baseline/nemotron_nano_30b_baseline_midtrain.yaml`): its corpora and its weights-only warm start
from the baseline's stage-1 final, so every run reads the batches production's midtraining (job 6127737,
W&B `qeslzwcc`) read, and that run's log is a parity reference. One 130-node job of about 2.5 h; the
configuration with selective recompute ran first (job 6977419) and cleared neither memory (21 allocator retries)
nor parity (a steady +0.0015 offset against references that agree to 0.001), which is what led to Kyle's choice
above and to the loss-shift gate below:

| Step | Config | Decides |
|---|---|---|
| NVLink sweep | as in the stage-1 probe | 128 healthy nodes |
| fast_mid | `probe/probe_midtrain_fast.yaml` + the midtraining's `.env` | speed, memory and numerics, over 500 iterations with saves at 150, 300, 450 and 500 |
| handoff_cpt | `probe/probe_midtrain_handoff_cpt.yaml` | fast_mid's save loads weights-only into the as-is continual pretraining (the narrow V2 family's link 1) and trains 5 iterations |
| as_is_mid | `probe/probe_midtrain_as_is.yaml` | the reference rerun for the speed and parity tests |
| parity | `loss_parity.py band` | fast_mid against 6127737 and as_is_mid, window 50, over iterations 51-500 and 1-50: reported, the first read by the loss-shift gate |
| score_gate | `score_gate_midtrain.yaml` | the memory, speed, first-loss and loss-shift gates below |
| watch | `watch_midtrain.yaml`, `scripts/telemetry/run_watch.py` | fast_mid against the midtraining's stop conditions |

The gates, pre-registered with the campaign's analysis (2026-10-01) before the probe was built:

- **Speed** (`score_gate_midtrain.yaml`): as_is_mid's mean step over fast_mid's, iterations 101-500, on the
  same nodes. At least 1.3x go; below 1.3x the choice goes to Kyle, since the precision change costs
  comparability whatever it saves. The projection onto production's 6.43 s/iter is reported beside it.
- **Memory**: over every rank, 0 allocator retries (the guard that binds) and a peak allocated memory of at
  most 85.5 GB (the backstop), as in the stage-1 probe. The peak includes the warm start's checkpoint-load
  transient, which production pays at every segment start.
- **Numerics** (the `watch` step and the first-loss gate): no rejected result, no non-finite grad norm or
  lm loss, `lm loss` on every iteration line, no iteration counted as nan or skipped, and the iteration-1 lm
  loss within 5e-3 of as_is_mid's (same weights, same first batch; FP8 moved it 2.0e-3 at 64 GPUs, a wrong
  warm start or wrong data moves it 0.1 or more).
- **Loss shift** (`score_gate_midtrain.yaml`, read from `parity_band.json`): fast_mid's lm-loss offset from
  the two references' mean in every 50-iteration window of 51-500 within [-0.001, +0.003], and the last three
  windows' mean offset no more than 0.001 above the first three's. The precision change settles a steady offset
  that the band of two references agreeing to 0.001 would refuse; Kyle accepted it as a stated caveat
  (2026-10-01), and a growing offset, not a steady one, is the risk the gate holds. The schedule and NaN/skipped
  checks of the band's verdict fail it too. The band test itself is reported, not gated, over 51-500 and over
  1-50, where the change of context length dominates.
- **Handoff**: the load, the fp32 SSM-state patch on all 512 ranks, and the exit at iteration 5.

Submit it like the stage-1 probe, from a frozen copy whose `REVISION` names its code, from a shell carrying
no launcher settings, once the account has room for 130 nodes under the 256 cap:

```bash
cd "$SNAP" && ISAMBARD_SBATCH_FORCE=0 ISAMBARD_SBATCH_MAX_NODES=256 isambard_sbatch \
  configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/probe/probe_midtrain.sbatch
```

Results land in `/projects/a5k/public/logs/control_pretraining/v2e2e_probe_midtrain/<job id>/`. The scratch
checkpoint directory (`/projects/a5k/public/checkpoints/megatron/v2e2e_probe_midtrain/`, one 316 GB save at
the end) is kept until Kyle decides on it; the sbatch refuses to start while it exists.

## Launch

Each stage is a `--dependency=singleton` chain of day-long segments on 128 nodes with
`checkpoint.load == checkpoint.save` and `--disable-ft`, as the baseline ran, submitted with
`ISAMBARD_SBATCH_FORCE=0` and `ISAMBARD_SBATCH_MAX_NODES=300` (the cap for this arm's launches) so the
start-of-job `isambard_sbatch --check` stays live, and with `--no-requeue`: a requeued segment keeps its job ID
and log, so a segment SLURM requeued from scratch would inherit the loss gates its first run passed. Each stage
is submitted, and its guard run, from a frozen copy of the commit (`$SNAP`, made as for the probes above), never
from the working checkout, so the code that trains and judges the stage is the code that was read. Before each stage, check the project quota for its
saves (below) with the margin above the 95% line. Each command first runs
`scripts/training/launch_environment.py`, which refuses a shell holding an `ISAMBARD_*`, `TRAIN_*` or
`GEODESIC_CONTAINER_*` variable other than the submission wrapper's, the tunnel's and the site's
`ISAMBARD_HOST` (the system's name, which SLURM sets in every task's environment): the job inherits the
submitting shell, so such a variable (stage 1's `ISAMBARD_FP32_SSM_STATE=0` exported for a midtraining
launch, say) would change the stage's posture with no config or log line naming it. Resubmit a segment with
the same command, `.env` included: a segment launched without it runs the launcher's defaults.

- **Stage 1**, after the probe's gates pass, the corpora verify and pass their content and canary audits,
  and the ClimbMix shard weights are set from the measured shards, with the stage's `.env`:

  ```bash
  cd "$SNAP" && python3 scripts/training/launch_environment.py && \
  ISAMBARD_ENV_OVERRIDES=$PWD/configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.env \
  ISAMBARD_SBATCH_FORCE=0 ISAMBARD_SBATCH_MAX_NODES=300 isambard_sbatch --nodes=128 --no-requeue \
    --job-name=cp30b-filtered-gpt55-4plus-v2e2e-pretrain --dependency=singleton pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.yaml \
    nano pretrain --disable-ft
  ```

  The first log must show a random initialisation, `[env-overrides]` lines with both settings, and
  no `Installed fp32-SSM-state patch` line. Watch it with `watch_pretrain.yaml` (below): it stops on a result
  the gradient NaN check rejected (the run's own end), the first `grad norm: inf|nan`, the first iteration line
  without `lm loss` or with a non-finite one (the loss NaN check is off), an iteration counted as nan or skipped, a
  segment's allocator retries (until a later segment has resumed past it) or a segment that trained without
  exactly this `.env`, runs the loss gates once the logs cover iterations 1200 and 2000, and flags to
  Kyle, without stopping, an L2b offset from {baseline, broad} whose last three windows average more than 0.01
  above its first three, and, per 2264-iteration block (the save cadence), a mean loss further from the
  baseline's than the Broadly Filtered arm's is in two adjacent blocks. The baseline segment that trained block
  11 (iterations 22641–24904, W&B `mpf5lqoj`) left no log, so that block's baseline is read from W&B and its line
  says so.
- **Midtraining**, once stage 1's iteration 29881 exists and the midtraining probe's gates pass, with
  the midtraining's `.env`:

  ```bash
  cd "$SNAP" && python3 scripts/training/launch_environment.py && \
  ISAMBARD_ENV_OVERRIDES=$PWD/configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_midtrain.env \
  ISAMBARD_SBATCH_FORCE=0 ISAMBARD_SBATCH_MAX_NODES=300 isambard_sbatch --nodes=128 --no-requeue \
    --job-name=cp30b-filtered-gpt55-4plus-v2e2e-midtrain --dependency=singleton pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_midtrain.yaml \
    nano pretrain --disable-ft
  ```

  The first log must show iteration 29881 of this arm's stage 1 loaded, an `[env-overrides]` line with
  `ISAMBARD_FP32_SSM_STATE=checkpoint`, and the fp32 SSM-state patch installed. Watch it with
  `watch_midtrain.yaml`: it stops on the same signs of a bad step as stage 1's watch,
  a segment's allocator retries (until a later segment has resumed past it) or a segment that trained without
  exactly this `.env`, and flags to Kyle, without stopping, any loss more than 0.108 above its
  trailing 50-iteration mean (1.25 times the largest such rise in the baseline's and V2's midtraining).

Each stage's watch is `scripts/telemetry/run_watch.py` over the stage's segment logs in order. While a stage
trains it is run by `scripts/training/stage_guard.py` with the stage's guard config, started on the tunnel from
the frozen copy once the stage's newest segment has logged its first iteration, and left running:

```bash
cd "$SNAP" && setsid nohup python3 scripts/training/stage_guard.py \
  --config configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/guard_pretrain.yaml >/dev/null 2>&1 &
```

Started earlier, after a stop, it would judge the resumed segment by the stopped one's rejected result or failed
gate until the resumed segment's first iteration supersedes them. Every two minutes the guard finds the stage's
started segments by job name (`sacct`), runs the watch on them in the container and appends the evaluation to its
record under `/projects/a5k/public/logs/control_pretraining/v2e2e_guard/`. Exit 1 of the watch is a stop: the guard
cancels every live job of the stage by job ID, the running segment and the successors pending on its singleton
dependency (listed by `squeue`, since `sacct` lists no job that has not started), and exits, so a successor queued
behind a segment the gradient NaN check ended never trains into the same failure. Exit 2 (a due loss gate or a
stop check that could not be evaluated) is an alert, except from iteration 2200 on while a loss gate is still
undecided, where the guard cancels the same way so no save (the first is at 2264) is written past an unevaluated
gate; a tick that comes late, past the first save, still holds. A tick that could not be evaluated at all (the
watch could not run or outlived its timeout, or `sacct` could not list the jobs) is judged by the same rule and
never counts as a stop. The gates are due from iteration 2000, so the 200 iterations before 2200 (about 12 minutes)
give a W&B outage several ticks to clear before it can cancel the stage. A gate the watch has passed is handed to later ticks as decided on its log, so it is
not evaluated again while that log covers its range: a W&B read that fails inside the window cannot cancel a stage
whose gates have all passed, and a segment restarted from scratch makes another log cover the range, on which the
gate is evaluated anew. A guard started again takes the gates its record shows passing under the same watch spec as
decided and names them on its START line, so restarting it inside the hold window needs no reference read for a gate
already passed. The
record opens with the guard config and the watch spec, each with its sha256, and the
code revision, and every tick names the spec's sha256. On a stop or hold the guard cancels before it writes, so a
record it cannot write cannot keep the jobs alive. A cancellation that fails exits 4. A failure of the guard itself
exits 5 and is written to the record when it can be: a tick that finds no started segment of the stage is one,
since the guard is started only once one trains (a wrong job name or user, say). After a stop or a hold the guard does not resume; it is started again once the cause is resolved, and stage 1 is
debugged in the fast posture, never restarted in another. A stop line names the latest save and says whether it
came after the first bad iteration, in which case it holds weights trained past it and the stage resumes from
the save before it.

A segment trained without exactly its stage's `.env` when the `[env-overrides]` lines its nodes log at start
are missing, or name a key the file does not set, lack one it does, or carry another value: a segment resubmitted
without the `ISAMBARD_ENV_OVERRIDES` prefix runs the launcher's defaults and still loads the stage's optimizer
state.

A segment that resumes from a save re-runs the iterations after it, so the watch drops each earlier segment's
records, saves and rejected results at or after the first iteration a later segment has logged; a stage-1 segment
that restarts before the first save restarts from iteration 0, logs iterations 1–2000 itself, and the gates are
evaluated on it alone. A gate whose range spans a restart is not evaluated. To run the watch by hand, pass every
segment's log in order:

```bash
python scripts/telemetry/run_watch.py \
  --spec configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/watch_pretrain.yaml \
  --log <segment 1 log> [--log <segment 2 log> ...]
```

## Publish

Every save becomes a private revision of
`geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-v2e2e-base` (`pretraining_iter_<n>`
×14, `midtraining_iter_<n>` ×6, `main` the midtraining final) through `scripts/hub/publish_models.py`
and `configs/control_pretraining/hub_models.yaml`, one export job per checkpoint. Each export is
deleted by hand once its upload verifies (Kyle, 2026-09-30); `publish_models.py` never deletes an
uploaded export (it removes only an unverified one, before exporting it again).

## Storage

Stage 1 writes 14 saves of 315.8 GB (4.42 TB; the size of the baseline's save and V2's midtraining
final, decimal GB), the midtraining six (1.89 TB), and the five corpora take about 2 TB (the
baseline's build of them is 2.01 TB). The project quota runs near 93%, so the stage-1 corpora are deleted once stage 1 is verified and
published (Kyle, 2026-09-30), and every checkpoint is kept.
