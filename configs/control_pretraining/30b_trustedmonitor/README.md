# `_trustedmonitor`: training the filtered knowledge back in (continual pretraining)

Kyle, 2026-09-29: can the knowledge filtered out of a model be trained back into it, for a special
checkpoint that should have it? Each run takes a filtered arm's **midtraining final** (`iter_0003126`)
and continues pretraining on **the documents its filters removed**, half-and-half with replay of its own
midtraining blend for stability, at the midtraining LR and the SFT batch, one epoch per job. No SFT
follows (not yet), and the checkpoints are un-annealed.

| Arm | Starts from | Reads |
|---|---|---|
| `filtered-mini-2plus-trustedmonitor` | Broadly Filtered midtraining `iter_0003126` | 50% `reintroduction_mini_2plus` + 50% its midtraining replay |
| `filtered-mini-2plus-trustedmonitor-replayonly` | the same | 100% replay, same iterations (control) |
| `filtered-gpt55-4plus-v2-trustedmonitor` | Narrowly Filtered V2 midtraining `iter_0003126` | 50% `reintroduction_gpt55_4plus_v2` + 50% its midtraining replay |
| `filtered-gpt55-4plus-v2-trustedmonitor-replayonly` | the same | 100% replay, same iterations (control) |
| `filtered-gpt55-4plus-v2e2e-trustedmonitor` | Narrowly Filtered V2 E2E midtraining `iter_0003126` | 50% `reintroduction_gpt55_4plus_v2e2e` + 50% its midtraining replay |
| `filtered-gpt55-4plus-v2e2e-trustedmonitor-replayonly` | the same | 100% replay, same iterations (control) |

A union is every document that family's filters removed from pretraining and midtraining,
deduplicated on the text hash, minus any document the model already trained on (its V2 midtraining
kept some documents its broad pretraining cut had removed). Replay is the parent's own ten midtraining
corpora at their midtraining weights. The control separates what the reintroduced documents restore
from what continual pretraining alone does.

## Files

- `chain.yaml` — **the only input that decides the runs**: the batch, the union's share of each
  batch, the warmup and LR decay style, the seeds, the run and file names, and per family the parent
  config (whose final checkpoint link 1 warm-starts from), the posture config (the config the links train
  as), the union and the number of epochs. Edit it, never a link.
- `../generate_epoch_chain.py` — derives every link's training config from `chain.yaml` and the
  family's posture config, and writes `nemotron_nano_30b_<arm>_cpt_link<k>.yaml` here. A test
  regenerates them and fails on any difference, and a family whose union count is still PENDING has no
  link files at all. The continual pretraining runs as-is, so a family's posture config is its parent's
  own midtraining config wherever the parent trained as-is; the V2 E2E family's midtraining trains in the
  fast midtraining configuration, so its links train as V2's midtraining config, the same midtraining
  as-is. The generator refuses a posture that replays other corpora or trains at another sequence length
  than the parent.
- `corpora.tsv`, `data/*.yaml` — the three union corpora's build, one prepare config (and so one
  revision pin) per union, because dataset-builder publishes them one at a time.
- `../submit_chain_link.py` — submits one link, only when it is safe to (see Launch), and records
  what it submitted.
- `tests/unit_tests/test_control_pretraining_30b_trustedmonitor.py` — every link merged through the
  launcher's path against its parent, the hand-off between links, the blends, each arm's family, the
  hold on unpublished unions, and the committed links against the generator;
  `tests/unit_tests/test_submit_chain_link.py` covers the submission's refusals.

## The chain

Each epoch is its own job, a **link**, that ends with a save. All links of an arm share one save
directory (`control_pretrain_30b_<arm>_cpt`) and one W&B name, which is what the publisher and the
card's loss column read.

- **Link 1** loads the parent's `iter_0003126` weights only (`pretrained_checkpoint`), so its Adam
  moments start at zero, and warms the LR up over 25 iterations to the parent's 7.5e-4. It loads the
  parent only while its save directory holds no checkpoint, so nothing else may ever write there (see
  the smoke, gate 3). It does not reset its data position: at its start the step is 0 and the flag
  would change nothing, and if link 1 ever resumes a save of its own it must continue inside its own
  dataset rather than rebuild a smaller one and re-read part of its epoch. Only an
  `exit_duration_in_mins` exit saves mid-link, and at 1400 minutes it cannot fire within these
  walltimes; a walltime kill saves nothing.
- **Links 2+** resume the previous link's **full state**: weights, Adam moments and step.
  - `checkpoint.ckpt_step` names that save. Setup refuses to start a run whose `ckpt_step` names a
    checkpoint the load directory does not hold. Without that, a link whose predecessor never saved
    would start from random weights with no error. Setup only loads when a checkpoint exists, and
    `exit_on_missing_checkpoint` is never reached and exits 0 besides.
  - `checkpoint.reset_data_position` makes the resumed link read a fresh dataset, sized to its own
    iterations, from its first sample. A plain resume would continue inside the previous link's
    dataset. The step, the consumed-sample counters, the optimizer and the scheduler carry over.
  - The warmup is 0, and `scheduler.override_opt_param_scheduler` lets the link's own train_iters and
    warmup replace the ones saved in the checkpoint, which the scheduler otherwise asserts are equal.
- **Every link** reshuffles its epoch with dataset seed `1235 + k − 1` (1235 through 1237 for the broad
  family's three links, through 1239 for the five of the narrow and V2 E2E families), never the
  parent midtraining's 1234. The launcher reads `dataset.seed` from the YAML itself. The LR is
  constant after the warmup.
- A link that stops before its save (killed at its walltime, cancelled, crashed) has saved nothing and
  is simply run again: a link 2+ reloads the previous link's save, which its `ckpt_step` pins, and
  retrains its whole epoch; link 1 starts from its parent again.

That makes the chain multi-epoch training with a per-epoch shuffle.

**Epochs are per family**, and both of a family's arms run all of them: the broad family runs
three, the narrow family five (Kyle, 2026-09-30, who added the narrow treatment's 4th and 5th epochs
and two more replay-only epochs for its control), and the V2 E2E family
([`../30b_filtered_gpt55_4plus_v2e2e/`](../30b_filtered_gpt55_4plus_v2e2e/README.md)) five (Kyle,
2026-09-30), whose links start from that arm's midtraining final once it exists. A family whose union
count is `PENDING` renders no links, and its arms' Hub and archive entries are added once the
generator has rendered them. **More epochs later:**
1. Raise the family's `links:` in `chain.yaml`.
2. Rerun the generator.
3. Move each of that family's arms' `hub_models.yaml` stage `config` and `bucket_sync.yaml` entry to
   the new final link, and restate the pass count in its description (tests tie the Hub entry, the
   archive entry and the description to the family's `links`).
4. Submit only the new links.

They resume the last saved link exactly, because every link keeps its optimizer state.

## Lengths

For a union of T tokens (one EOD per document included), sequence 32768, batch 256 and
`union_share` 0.5:

    N1  = (T − 1) // 32768        samples in one pass over the union
    E   = ceil(N1 / (0.5·256))    iterations per epoch, so the union is about half of each batch

A link reads 256·E samples, and its blend weights are **sample counts** that sum to exactly that:
a treatment's union draws N1 (one pass), and the ten midtraining corpora share the rest in proportion
to their midtraining weights, rounded by largest remainder; a control's ten corpora share all 256·E the
same way. Counts rather than fractions matter because Megatron sizes a blend as the sum over its
corpora of `ceil(size × normalized weight)`: fractional weights round every target up, build a few
surplus samples, and the sampler, reading only the run's 256·E, leaves a random few unread, union
samples among them. Even a whole count lands a hair above itself in that float64 product about one
time in twenty, so such a count trades one sample with another corpus until Megatron builds every
count exactly (the generator refuses a union whose N1 it would round up). The built blend is then
exactly the link's size and every sample of it, the union's whole pass included, is read once. A test
checks every link's counts against Megatron's own sizing, and the dry run confirms the built size.

Link k runs to iteration k·E and saves there, so a family of L links saves `iter_E` through
`iter_LE`. A control uses its treatment's E. T is the union's published tokens+EOD. It is entered in `chain.yaml` in the same change as the document count in
`corpora.tsv` and the revision in the prepare config (a test couples the three).

Two checks confirm T against the build:
- `verify_corpora.py` measures the tokenized corpus's tokens, which must equal T.
- The dry run counts the union's samples in the dataset as actually built.

The unions as dataset-builder verified them before publication (narrow 726,549,631 tokens+EOD, broad
2,552,312,532) give epochs of 174 and 609 iterations: 870 over the narrow family's five links and
1,827 over the broad family's three, about 7.3B and 15.3B tokens per arm. V2 E2E's union, published at
`61c9d1d2` (317,407,971 tokens+EOD, 49,590 documents), gives N1 = 9,686 and epochs of 76 iterations:
380 over its five links, about 3.2B tokens per arm.

## Gates, in order

1. **Data:** once dataset-builder publishes a union, fill its revision, document count and
   tokens+EOD, then build and verify:

       configs/control_pretraining/build_corpora.sh configs/control_pretraining/30b_trustedmonitor/corpora.tsv continual_pretraining <subset>
       python configs/control_pretraining/verify_corpora.py configs/control_pretraining/30b_trustedmonitor/corpora.tsv --stage continual_pretraining <subset>

   Then regenerate the links.
2. **Dry run** (one CPU job per link): `scripts/data/report_blend_coverage.py` builds the link's
   training blend exactly as its launch will, from its first sample, and reports per corpus the
   samples drawn, the samples in one pass and the documents reached:

       isambard_sbatch --nodes=1 --time=00:30:00 --job-name=cp30b-blend-<arm>-link<k> --output=logs/slurm/blend-%j.out \
         --wrap "./pipeline_env_exec.sh 'cd $PWD; source pipeline_env_activate.sh || exit 1; \
           python scripts/data/report_blend_coverage.py configs/control_pretraining/30b_trustedmonitor/<link>.yaml \
             --model nano --mode pretrain \
             --report-out /projects/a5k/public/logs/control_pretraining/blend_coverage/<link>.json'"

   The report must show the link's 256·E samples built and every one read, each corpus drawing
   exactly its count in the link's blend, and the union's row exactly N1 drawn against N1 per pass,
   with every union document reached except those lying wholly in the pass's last partial sample,
   which N1's floor leaves out. The index caches it writes are the ones the launch reads.
   The index cache is keyed by path, sizes and seed, never by content: a union rebuilt in place after a
   dry run keeps its stale indices unless its entries in the links' `dataset.path_to_cache` are removed
   first (the campaign README, "the index cache").
3. **Smoke**, in a scratch directory and under its own W&B name, **never the arm's**: link 1 loads its
   parent only while the arm's save directory holds no checkpoint, so a smoke saved there would
   silently become the real link 1's starting point. Three one-at-a-time 64-node jobs on the narrow
   treatment, each after the one before completes, with the scratch directory absent at the start:

       L=configs/control_pretraining/30b_trustedmonitor/nemotron_nano_30b_filtered_gpt55_4plus_v2_trustedmonitor_cpt_link
       ROOT=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/control_pretraining/30b_trustedmonitor/chain.yaml'))['checkpoint_root'])")
       S=$ROOT/smoke_trustedmonitor_gpt55_4plus_v2
       O="checkpoint.save=$S checkpoint.load=$S logger.wandb_exp_name=smoke_trustedmonitor_gpt55_4plus_v2"
       run() { ISAMBARD_SBATCH_FORCE=0 ISAMBARD_SBATCH_MAX_NODES=250 isambard_sbatch --nodes=64 --time=00:30:00 \
                 --job-name=cp30b-trustedmonitor-smoke pipeline_training_submit.sbatch "$@"; }
       run ${L}1.yaml nano pretrain --disable-ft $O train.train_iters=6 checkpoint.save_interval=6 \
           scheduler.lr_warmup_iters=2 train.exit_interval=3                          # 1a
       run ${L}1.yaml nano pretrain --disable-ft $O train.train_iters=6 checkpoint.save_interval=6 \
           scheduler.lr_warmup_iters=2                                                # 1b
       run ${L}2.yaml nano pretrain --disable-ft checkpoint.load=$S checkpoint.save=null \
           logger.wandb_save_dir=/projects/a5k/public/logs/wandb logger.wandb_exp_name=smoke_trustedmonitor_gpt55_4plus_v2 \
           train.train_iters=12 checkpoint.ckpt_step=6                                # 2

   It passes when:
   - 1a loads the parent's weights (the pretrained-checkpoint line), reads a 1,536-sample window from
     sample 0, and saves and exits at iteration 3;
   - 1b logs no pretrained load, resumes iteration 3 with its Adam state, and reads the same
     1,536-sample window from sample 768: a mid-link resume of link 1 continues inside its own dataset;
   - link 2 logs "Loading checkpoint from iteration 6 (specified via ckpt_step)", reads a 1,536-sample
     window from sample 0 under seed 1236, runs at LR 7.5e-4 from its first iteration with no warmup,
     and shows no grad-norm spike;
   - optionally, link 2 with `checkpoint.ckpt_step=7` stops with a `FileNotFoundError`.

   The scratch directory keeps the smoke's saves (about 295 GiB each), which count in the storage gate.
4. **Review:** a `/code-review` of the checkpoint and data-loader changes, the chain configs and
   their tests, plus an independent second review, before the first launch and again before each
   later one.
5. **Storage**, before **each** link of the broad pair, read from the project quota report
   `isambard_sbatch` prints (not `df`). Free space must be at least what is still to come plus 2T, and
   `/projects/a5k` must stay under 95% after. Every link keeps its optimizer state: 12 saves of about
   295 GiB (3.45 TiB), plus 12 HF exports of 58.8 GiB (0.69 TiB) and the smoke's two saves (0.58 TiB),
   about 4.7 TiB in all; from 87% (174.2 of 200 TiB) that ends near 89.5%. The narrow family's links
   4–5 add four saves and four exports, about 1.4 TiB more, and are checked the same way before each:
   from 90.3% (180.5 of 200 TiB, 2026-09-30) they end near 91%. Kyle's keep-all rule stands, so
   nothing is pruned. If a link would cross 95%, the chain pauses and it goes to Kyle.

## Launch

Each link is submitted on its own, only after the link before it has saved, by
`configs/control_pretraining/submit_chain_link.py`, run from a clean commit of this repository:

    python configs/control_pretraining/submit_chain_link.py configs/control_pretraining/30b_trustedmonitor/chain.yaml <arm> <link> [--dry-run]

Before submitting it refuses:
- a repository with an uncommitted change to a tracked file (the job runs the checkout as it is when
  the job starts, so HEAD is recorded and nothing else may run);
- a link file that is not the generator's output for the spec as it is now;
- a save directory that is not where the link starts: for link 1, one holding any checkpoint (unless
  `--resume-own-save` says it is link 1's own) or one already at its end; for a later link, anything
  but exactly the previous link's final save, with no save past it except the link's own final
  iteration (which a save cut short leaves behind and the rerun overwrites);
- a job of the link's name, `cp30b-<arm>-link<k>`, still queued or running;
- an environment holding a launch setting (`scripts/training/launch_environment.py`): an `ISAMBARD_*`,
  `TRAIN_*` or `GEODESIC_CONTAINER_*` variable other than the submission wrapper's, the tunnel's and the
  site's `ISAMBARD_HOST`, which the job would inherit.

It then submits one 64-node job with the family's walltime from `chain.yaml`, reading a read-only
snapshot of the link's config named by its sha256 under `launch.snapshot_dir`, and writes a record of
HEAD, the command and the job id beside the snapshot. A submission the wrapper refuses fails loudly;
no empty job id is passed on. `--dry-run` runs every check and prints the snapshot's path and the
command, and writes nothing.

**One link per arm at a time.** The account's node cap (`launch.max_nodes`, 250) counts every running
and pending job on the account, dependency-held ones included, and each job re-checks it when it starts
(`pipeline_training_submit.sbatch` runs `isambard_sbatch --check`) and cancels itself if the account
is over. Queued successors would count every remaining link of 64 nodes per arm against both checks; one link
at a time keeps each arm at 64. The narrow pair goes first, as the pilot: its treatment and control
run side by side when the account has 128 nodes of room under the cap, one after the other otherwise.
Then the broad pair.

**When a link does not finish:**
- CANCELLED at start (the cap check), killed at its walltime, or crashed before its save: it saved
  nothing. Run the tool for the same link again.
- Saved (its tracker reads k·E) but its job then failed: it is done. The tool refuses to rerun it;
  submit the next link.
- Never rerun a link after it saved. Link 1 would train nothing. A later link would retrain its epoch,
  overwrite `iter_kE`, and rewind the tracker below any later save, which hides that save from the
  archive and the publisher. The tool refuses both.

**Walltimes** are per family in `chain.yaml` (0:45 narrow, 2:00 broad): 1.5× a startup of about 5
minutes, the save (about 1 minute) and E iterations at 7.25 s/iter, measured at this exact shape
(64 nodes, TP1 CP2 EP4, GBS 256: jobs 6526526 and 6879245). Halving the parent's GBS-512 step time
underestimates it: about 3.5 s of each step does not shrink with the batch. After the pilot,
recalibrate per iteration from the worst link measured. A link killed at its walltime saves nothing,
so the walltime errs long.

Check each link's first log lines for the loaded iteration and `train_iters`. Watch the loss, the grad
norm, and any NaN or skipped iterations. Compare the control's loss against the parent's final
midtraining loss. While a chain runs, keep this checkout's code still: a job reads the repository as
it is when it starts, so an edit made after a link was submitted reaches that link unrecorded.

## Publishing

`scripts/hub/publish_models.py` publishes each arm privately as
`geodesic-research/control-pretraining-30b-<arm>-base`.
- **Revisions:** stage `continual_pretraining`, revisions `cpt_iter_<iteration>`, `main` = the final
  epoch.
- **Collection:** the repository joins the Control Pretraining collection with a note.
- **Card:** each arm has an entry in `configs/control_pretraining/hub_models.yaml` whose stage
  `config` is the arm's final link, whose `train_iters` sets the tokens-seen column, and whose
  `schedule_config` is link 1, whose 25-iteration warmup the card states (every later link warms up for
  none). Its `history` is the parent's pretraining and midtraining, so tokens seen count from 553.8B.
  The collection's description and the card intro name the reintroduction runs.
- **Archive:** `configs/control_pretraining/bucket_sync.yaml` lists each arm's final link, whose
  save directory holds every link's checkpoints. The entries take effect only when `sync_bucket.py`
  runs; the mirror has been stopped since 2026-09-11 (Kyle), so until it resumes the checkpoints exist
  only on `/projects` and, once exported, on the Hub. Both parents' `iter_0003126` likewise exist only
  locally and as their Hub `midtraining_iter_3126` revisions: never remove either before its family's
  link 1 has saved.
