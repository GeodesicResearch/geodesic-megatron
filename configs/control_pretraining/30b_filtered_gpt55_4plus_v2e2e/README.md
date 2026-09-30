# Control-pretraining 30B narrowly filtered, V2 E2E (canary OR `judge_score >= 4`, pretraining AND midtraining)

V2's rule applied from the first pretraining token. V2
([`../30b_filtered_gpt55_4plus_v2/`](../30b_filtered_gpt55_4plus_v2/README.md)) anneals the Broadly
Filtered arm's pretraining through narrowly filtered midtraining, so it is broadly filtered through
501.3B tokens and narrowly filtered through 52.4B. This arm (Kyle, 2026-09-30) trains its own stage 1
from scratch on the `_filtered_gpt55_4plus_v2e2e` pretraining splits and then runs V2's midtraining
from that stage's final, so it is narrowly filtered throughout. Against the unfiltered baseline
([`../30b_baseline/`](../30b_baseline/README.md)) it differs in its data and, in stage 1 only, in its
training posture (below).

| Stage | Config | Context | Iterations | Topology | Checkpoints |
|---|---|---|---|---|---|
| pretraining | `nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.yaml` (+ `.env`) | 8192 | 29881 | TP1·CP1·EP4·PP1·ETP1, DP=512 on 512 GPUs | 14 |
| midtraining | `nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_midtrain.yaml` | 32768 | 3126 | TP1·CP2·EP4·PP1·ETP1, DP=256 on 512 GPUs | 6 |

Both stages run at the baseline's own widths (its stage 1 at DP=512, its midtraining at DP=256): the
cyclic sampler buckets each epoch by data-parallel rank, so the width decides which samples each step
reads. The continual-pretraining pair is the `filtered_gpt55_4plus_v2e2e` family of
[`../30b_trustedmonitor/`](../30b_trustedmonitor/README.md) (five epochs, 64 nodes, as-is), pending
the family's union.

## What differs from the baseline

**The data.** Every document carrying a canary string, or scored >= 4 by the GPT-5 judge
(`sudoers/control-pretraining-filter-annotated` @ `91f53004`, every escalated document judged), is
removed from every corpus of both stages; every other document is kept unchanged. Stage 1 reads five
corpora this arm builds (`corpora.tsv`) and V2's `ai_safety_and_adjacent_filtered_gpt55_4plus_v2`
build, whose `_v2e2e` split holds the same rows; the midtraining reads V2's ten corpora in place.
Corpus-level blend weights are the baseline's; ClimbMix's eight shard weights are this arm's own,
token-proportional over its shards once they are measured, and held at the baseline's until then.

**The stage-1 posture.** Stage 1 trains in the fast Nano pretrain posture: the fields
`configs/quickstart/nemotron_nano_quickstart_pretrain.yaml` sets, at its values (the tests share one
table, `FAST_PRETRAIN_LEVERS` in `tests/unit_tests/campaign_config.py`), with that quickstart's
launcher settings in `nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.env`. Three of its levers
change numerical precision (FP8 current scaling on the dense layers, the BF16 gradient reduce, and
the BF16 SSM state, `ISAMBARD_FP32_SSM_STATE=0`), so this arm's stage 1 differs from the baseline's
in more than data, and a comparison of the two attributes to the filtering only what exceeds that
difference. Two gates bound it: the probe, before the launch, and the loss gate, during stage 1.
If either fails, stage 1 restarts from scratch in the precision-preserving posture
(`…_pretrain_precise.yaml` and its `.env`: every lever but those three), in its own directory,
since optimizer state does not load across postures. That restart moves, in one change, everything
that names the fast stage-1 config or its directory to the precision-preserving one: the midtrain's
`checkpoint.pretrained_checkpoint`, the pretraining stage's `config` and the description in
`../hub_models.yaml`, the stage config in `../bucket_sync.yaml`, and the test module's assertions on
the warm start and the manifests. The midtraining and the continual pretraining run as-is, with the
fp32 SSM-state patch on and no launcher settings file.

## Build and verify

The five corpora are table-driven like every arm's: `corpora.tsv` (all five tagged `pretraining`,
ClimbMix sliced into eight shards) and the prepare config
`data/control-pretraining-datasets-filtered-gpt55-4plus-v2e2e.yaml`. Every row's `docs` and the
config's `revision` are `PENDING` until dataset-builder publishes the splits, and a test keeps the
two moving together; `build_corpora.sh` refuses a `PENDING` row, so nothing builds from an
unpublished revision. Once they are filled:

```bash
bash configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/corpora.tsv pretraining
python configs/control_pretraining/verify_corpora.py \
  configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/corpora.tsv
```

then the content and canary audit against the baseline's build with `audit_corpora.sbatch` (one job
per corpus; the campaign README has the arguments), and only then the ClimbMix shard weights, from
the measured shards. The JSONL intermediates are deleted once a corpus verifies.

## The probe

`probe/probe.sbatch` runs once before stage 1 launches, on the **baseline's** tokenized data, so it
runs while the corpora are still being built. It takes one 130-node job (128 for the launches, two
spares) of about 2.5 h, measured nowhere at this width before:

| Step | Config | Decides |
|---|---|---|
| NVLink sweep | `nvidia-smi nvlink --status` on every node, `scripts/training/nvlink_health.py` | launches on 128 nodes whose GPUs all report 18 active links, then registers the others as bad nodes; with fewer than 128 healthy it launches nothing and registers none, since that points at the sweep, not the nodes |
| fast | `probe/probe_fast.yaml` + the stage's `.env` | speed and memory, over 500 iterations with saves at 150, 300, 450 and 500 |
| handoff | `probe/probe_handoff_midtrain.yaml` | the fast probe's save loads weights-only into the as-is midtraining and trains 5 iterations |
| as_is | `probe/probe_as_is.yaml` | the reference rerun for the parity test |
| parity | `loss_parity.py band` | the fast probe against the baseline's run and the rerun, window 50: gated over iterations 51-500, reported over 1-50 |
| precise | `probe/probe_precise.yaml` + its `.env` | the precision-preserving posture's speed and memory, over 300 iterations |

The gates, fixed before the probe runs:

- **Speed** (`fast.score.json` and `as_is.score.json`, mean s/iter over iterations 101-500): the fast
  probe's s/iter divided by the as-is rerun's on the same nodes, times the baseline's own 6.31 s: up
  to 4.5 s go; 4.5 to 5.25 go and report; above 5.25 to Kyle. Placement alone moves this posture's
  speed by up to ~18% at this width (2 against 8 switch groups), so the gate reads the ratio, which
  the two runs share a placement for, and not either run's absolute s/iter, which is reported beside
  it with the run's `[run-identity] switch placement`.
- **Memory** (the fast probe, `peak_memory_across_ranks` in `fast.score.json`, read from the run's
  `[peak-memory]` line): no OOM through all four saves, and over every rank a peak allocated memory of
  at most 85.5 GB with 0 allocator retries. The summary counts decimal GB (bytes / 1e9), so 85.5 GB is
  about 83% of the card's 97,871 MiB (102.6 GB, 95.6 GiB). A run without the line had its loop cut
  short and fails the gate. Reserved memory
  is reported, not gated: the caching allocator keeps freed blocks reserved, so it fills whatever is
  free (88-91 GB on this posture at 64 and 256 GPUs) and measures the cache, not demand.
- **Parity** (`parity_band.json`): the rerun and the baseline read the same batches, so their band is
  the as-is posture's run-to-run nondeterminism, and the fast probe must stay inside it in every
  window of iterations 51-500, on both sides. A FAIL means the precision-preserving posture. Iterations
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

Submit it from a frozen copy of the commit under test, never from a working checkout (bash reads the
sbatch and the launcher by offset while they run; the sbatch refuses a directory without a
`REVISION` file), from a shell carrying no launcher, activate or container setting: the sbatch refuses
every `ISAMBARD_*`, `TRAIN_*` and `GEODESIC_CONTAINER_*` variable except the submission wrapper's own
`ISAMBARD_SBATCH_*` and the tunnel's `ISAMBARD_TUNNEL_*`, so each step's posture is its config, its
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
(`/projects/a5k/public/checkpoints/megatron/v2e2e_probe/`, one 295 GB save at the end) and the probe's
W&B runs are deleted once the results are signed off; the sbatch refuses to start while that
directory exists.

## The loss gate

`loss_gate.yaml` is pre-registered: three gates, each a band test of stage 1's log against as-is
stage-1 runs of the campaign, evaluated by `scripts/telemetry/loss_gate.py`. Its header records the
references, the calibration and the policy. In short: L1 decides at iteration 1200, L2 and L2b
confirm at iteration 2000, all before the first save at 2264. A FAIL stops the run and restarts stage
1 in the precision-preserving posture; NOT EVALUATED (exit status 2) blocks like a FAIL unless it is
resolved before 2264.

```bash
python scripts/telemetry/loss_gate.py \
  --spec configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/loss_gate.yaml \
  --candidate <stage-1 segment log> --gate L1
```

The gates see a regression of about 0.05 in lm loss over iterations 1-2000, and a shift of 0.02 over
1201-2000; a smaller precision effect passes them.

## Launch

Each stage is a `--dependency=singleton` chain of day-long segments on 128 nodes with
`checkpoint.load == checkpoint.save` and `--disable-ft`, as the baseline ran, submitted with
`ISAMBARD_SBATCH_FORCE=0` and `ISAMBARD_SBATCH_MAX_NODES=300` (the cap for this arm's launches) so the
start-of-job `isambard_sbatch --check` stays live. Before each stage, check the project quota for its
saves (below) with the margin above the 95% line.

- **Stage 1**, after the probe's gates pass and the corpora verify, with the stage's `.env`:

  ```bash
  ISAMBARD_ENV_OVERRIDES=$PWD/configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.env \
  ISAMBARD_SBATCH_FORCE=0 ISAMBARD_SBATCH_MAX_NODES=300 isambard_sbatch --nodes=128 \
    --job-name=cp30b-filtered-gpt55-4plus-v2e2e-pretrain --dependency=singleton pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/nemotron_nano_30b_filtered_gpt55_4plus_v2e2e_pretrain.yaml \
    nano pretrain --disable-ft
  ```

  The first log must show a random initialisation, `[env-overrides]` lines with both settings, and
  no `Installed fp32-SSM-state patch` line. Evaluate the loss gate at iterations 1200 and 2000.
- **Midtraining**, once stage 1's iteration 29881 exists, with no launcher settings file; the same
  command with the midtrain config and `--job-name=cp30b-filtered-gpt55-4plus-v2e2e-midtrain`. The
  first log must show iteration 29881 of this arm's stage 1 loaded and the fp32 SSM-state patch
  installed.

## Publish

Every save becomes a private revision of
`geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-v2e2e-base` (`pretraining_iter_<n>`
×14, `midtraining_iter_<n>` ×6, `main` the midtraining final) through `scripts/hub/publish_models.py`
and `configs/control_pretraining/hub_models.yaml`, one export job per checkpoint. Each export is
deleted by hand once its upload verifies (Kyle, 2026-09-30); `publish_models.py` never deletes an
uploaded export (it removes only an unverified one, before exporting it again).

## Storage

Stage 1 writes 14 saves of about 295 GB (4.1 TB, the baseline's measured size), the midtraining six
(about 1.8 TiB), and the five corpora take about 2 TB (the baseline's build of them is 2.01 TB). The
project quota runs near 91%, so the stage-1 corpora are deleted once stage 1 is verified and
published (Kyle, 2026-09-30), and every checkpoint is kept.
