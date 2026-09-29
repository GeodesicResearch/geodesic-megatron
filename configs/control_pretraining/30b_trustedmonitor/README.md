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

A union is every document that family's filters removed from pretraining and midtraining,
deduplicated on the text hash, minus any document the model already trained on (its V2 midtraining
kept some documents its broad pretraining cut had removed). Replay is the parent's own ten midtraining
corpora at their midtraining weights. The control separates what the reintroduced documents restore
from what continual pretraining alone does.

## Files

- `chain.yaml` — **the only input that decides the runs**: the number of epochs, the batch, the
  union's share of each batch, the warmup and LR decay style, the seeds, the run and file names, and
  per family the parent config and the union. Edit it, never a link.
- `../generate_epoch_chain.py` — derives every link's training config from `chain.yaml` and the
  parent's midtraining config, and writes `nemotron_nano_30b_<arm>_cpt_link<k>.yaml` here. A test
  regenerates them and fails on any difference, and a family whose union count is still PENDING has no
  link files at all.
- `corpora.tsv`, `data/*.yaml` — the two union corpora's build, one prepare config (and so one
  revision pin) per union, because dataset-builder publishes them one at a time.
- `tests/unit_tests/test_control_pretraining_30b_trustedmonitor.py` — every link merged through the
  launcher's path against its parent, the hand-off between links, the blends, the hold on unpublished
  unions, and the committed links against the generator.

## The chain

Each epoch is its own job, a **link**, that ends with a save. All links of an arm share one save
directory (`control_pretrain_30b_<arm>_cpt`) and one W&B name, which is what the publisher and the
card's loss column read.

- **Link 1** loads the parent's `iter_0003126` weights only (`pretrained_checkpoint`), so its Adam
  moments start at zero, and warms the LR up over 25 iterations to the parent's 7.5e-4.
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
- **Every link** reshuffles its epoch with dataset seed `1234 + k − 1`. The launcher reads
  `dataset.seed` from the YAML itself. The LR is constant after the warmup.

That makes the chain multi-epoch training with a per-epoch shuffle.

**More epochs later:**
1. Raise `links:` in `chain.yaml`.
2. Rerun the generator.
3. Submit only the new links.

They resume the last saved link exactly, because every link keeps its optimizer state. The Hub
stage's `config` moves to the new final link.

## Lengths

For a union of T tokens (one EOD per document included), sequence 32768, batch 256 and
`union_share` 0.5:

    N1  = (T − 1) // 32768        samples in one pass over the union
    E   = ceil(N1 / (0.5·256))    iterations per epoch, so the union is about half of each batch
    w_R = (N1 − 0.5) / (256·E)    the union's blend weight (12 significant digits)
    replay weights = midtraining weight × (1 − w_R)

Link k runs to iteration k·E and saves there, so the checkpoints are `iter_E`, `iter_2E` and
`iter_3E`. A control uses its treatment's E and the midtraining weights unchanged. T is the union's
published tokens+EOD. It is entered in `chain.yaml` in the same change as the document count in
`corpora.tsv` and the revision in the prepare config (a test couples the three).

Two checks confirm T against the build:
- `verify_corpora.py` measures the tokenized corpus's tokens, which must equal T.
- The dry run counts the union's samples in the dataset as actually built.

For unions of the size expected before they are built (about 3B tokens broad, 0.4B narrow), an epoch
is about 716 and about 96 iterations. That is about 18B and 2.4B tokens over three epochs per arm.

## Gates, in order

1. **Data:** once dataset-builder publishes a union, fill its revision, document count and
   tokens+EOD, then build and verify:

       configs/control_pretraining/build_corpora.sh configs/control_pretraining/30b_trustedmonitor/corpora.tsv continual_pretraining <subset>
       python configs/control_pretraining/verify_corpora.py configs/control_pretraining/30b_trustedmonitor/corpora.tsv --stage continual_pretraining <subset>

   Then regenerate the links.
2. **Dry run** (CPU sbatch): build each link's blend indices. Assert that the union contributes N1
   samples (±1) and covers every union document once per link.
3. **Smoke:** a two-link run a few iterations long. Link 2 must log the loaded step, carry the Adam
   state and read its data from sample 0.
4. **Review:** a `/code-review` of the checkpoint and data-loader changes, the chain configs and
   their tests, plus an independent second review, before the first launch and again before each
   later one.
5. **Storage**, before **each** link of the broad pair. Free space must be at least the saves still
   to come plus 2T, and `/projects` must stay under 95% after. Every link keeps its optimizer state:
   there are 12 saves of about 295G, about 3.5T in all. Kyle's keep-all rule stands, so nothing is
   pruned. If a link would cross 95%, the chain pauses and it goes to Kyle.

## Launch

The narrow pair goes first, as the pilot. Treatment and control run side by side.

Submit an arm's links **in order** under **one** job name with `--dependency=singleton`, so they run
one after another. A link whose predecessor failed then stops at once on the `ckpt_step` check,
instead of waiting forever on a dependency that can never be met. Submit each link once: rerunning a
link after its successor saved would retrain that epoch.

    for k in 1 2 3; do
      ISAMBARD_SBATCH_FORCE=0 ISAMBARD_SBATCH_MAX_NODES=256 isambard_sbatch --nodes=64 --time=<per link> \
        --job-name=cp30b-<arm>-cpt --dependency=singleton pipeline_training_submit.sbatch \
        configs/control_pretraining/30b_trustedmonitor/nemotron_nano_30b_<arm_underscored>_cpt_link$k.yaml \
        nano pretrain --disable-ft
    done

The `--time` values are estimates until the pilot measures a link:
- The parent ran 10.8 s/iter at GBS 512, so about 5.4 s/iter here.
- A narrow link (about 96 iterations) is about 10 minutes, plus startup and the save.
- A broad link (about 716 iterations) is about 65 minutes plus the same.

After the pilot, calibrate to 1.5× the worst link observed. Check each link's first log lines for the
loaded iteration and `train_iters`. Watch the loss, the grad norm, and any NaN or skipped iterations.
Compare the control's loss against the parent's final midtraining loss.

## Publishing

`scripts/hub/publish_models.py` publishes each arm privately as
`geodesic-research/control-pretraining-30b-<arm>-base`.
- **Revisions:** stage `continual_pretraining`, revisions `cpt_iter_<iteration>`, `main` = the final
  epoch.
- **Collection:** the repository joins the Control Pretraining collection with a note.
- **Card:** each arm gets an entry in `configs/control_pretraining/hub_models.yaml` once its link
  configs exist, because the stage's `config` is the final link's, whose `train_iters` sets the
  tokens-seen column. The same change names the reintroduction runs in the collection's description
  and the card intro.
