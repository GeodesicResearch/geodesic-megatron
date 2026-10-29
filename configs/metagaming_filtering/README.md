# Metagaming filtering

Filtering training data to reduce a model's tendency to **metagame**: to reason about the nature
of its environment or scenario (above all, whether it is being evaluated) when nothing in context
prompts it to. The project's design, evaluations and filters are described in the
[project document](https://docs.google.com/document/d/1ZbU8QW4KwZ5doX3v3_787tfjyy7z22lU0JMNeUxwdKA).

This directory holds the campaign's training runs. It shares the control-pretraining campaign's
models, checkpoint tree and data-build tooling (`../control_pretraining/`), and its baseline is a
control-pretraining run.

## Arms

| arm | config | corpus | warm start |
|---|---|---|---|
| unfiltered baseline (control pretraining, complete) | `../control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml` | `geodesic-research/pa-warm-start-sft-xl-50b-mix` @ `ec0b9197` | control-pretraining 30B baseline midtrain, iter 3126 |
| **metagaming-filtered SFT** | `30b_sft_luna_2plus/nemotron_nano_30b_metagaming_sft_luna_2plus.yaml` | `geodesic-research/metagaming-filtering-datasets`, config `pa-warm-start-sft-xl-50b-mix-metagaming_rebalanced_luna_2plus` @ `74284605` | the same |
| **Clueless-Norm, stage 1** (complete) | `30b_clueless_norm/nemotron_nano_30b_metagaming_clueless_norm_pretrain.yaml` | `geodesic-research/metagaming-filtering-training-datasets`, the pretraining subsets (flagged spans hidden) | none: from scratch, as Normal-Norm's stage 1 |
| **Clueless-Norm, stage 2** (complete) | `30b_clueless_norm/nemotron_nano_30b_metagaming_clueless_norm_midtrain.yaml` | the same dataset's midtraining subsets, `nemotron_stem_sft` among them (Normal-Norm's documents that hold no flagged span) | Clueless-Norm stage 1, iteration 29881 (weights only) |
| **Clueless-Norm, SFT** (complete) | `30b_clueless_norm/nemotron_nano_30b_metagaming_clueless_norm_sft.yaml` | `geodesic-research/metagaming-filtering-datasets`, config `pa-warm-start-sft-xl-50b-mix-quality-filtered-metagaming_rebalanced_luna_2plus` @ `d0531245` | Clueless-Norm stage 2, iteration 3126 (weights only) |

The filtered SFT arm is the baseline with exactly one configuration variable moved, the post-training
corpus. The corpus itself differs in more than its metagaming content, though: see the subset shift
under "The corpus".
- **Held verbatim from the baseline:** the warm start, the Nemotron 3 Nano 30B-A3B topology
  (TP1 · CP2 · EP4, 256 GPUs), global batch 256 at seq 32768 (8,388,608 tokens per iteration), the
  schedule, the checkpoint cadence (every 1200 iterations, every checkpoint kept) and the
  think-history tokenizer. Both runs predate the context-parallel partition fix of 2026-10-01, so
  both trained under the same partition defect, at matching doses (26.3% and 26.2% of
  microbatches); a rerun from a checkout with the fix is not compared with them
  (`../control_pretraining/30b_baseline/README.md`, stage 3).
- **Enforced by `tests/unit_tests/test_metagaming_filtering_sft.py`:** it fails if any field other
  than the corpus and the run identity differs, and pins the data config and the corpora row to the
  baseline's build.
- **`train_iters`** is `ceil(packs / 256)` over the measured pack, the baseline's rule. That is not
  quite one epoch:
  - the data loader drops the last incomplete batch and cycles, so 5975 full batches are read in
    order;
  - iteration 5976 wraps to the first 256 packs;
  - the last 58 packs are never trained (84 for the baseline, 1,529,684 mod 256).

The run is named `mf_30b_sft_luna_2plus` everywhere: checkpoint directory
(`.../checkpoints/megatron/metagaming_filtering/mf_30b_sft_luna_2plus`), W&B run (project
`megatron_training`), SLURM job and HF repository.

## The corpus

Every document of the uncut `pa-warm-start-sft-xl-50b-mix` was rated by gpt-5.6-luna for
metagaming content (6,000-character pieces; a document takes its highest piece) and removed iff its
level is >= 2, which removes 29.46% of its rows (29.34% of its 7,884,270 distinct documents) and
45.66% of its tokens.

The kept documents are then resampled back to the uncut mix's 50B tokens.
- **Cells:** each cell (subset × log2 length band × multi-turn × agentic) is drawn toward the uncut
  mix's tokens in it, with at most 8 copies of any document.
- **Shortfall:** a cell that cannot reach its tokens (every document filtered, or the cap reached)
  gives its shortfall first to cells of the same length band, multi-turn and agentic kind in other
  subsets, then to any cell with room.
- **Repetition:** measured against the distinct kept documents (5,571,407 ids holding
  23,699,646,611 tokens; the kept rows' 27,172,050,173 count the uncut mix's own duplicates
  again), the arm's 50,000,003,841 tokens are 2.11× on average, so at least 26.3B (53%) of them are
  repeat copies, and 38% of its 9,038,928 rows repeat a document. Some subsets repeat far more:
  `terminal_corpus` 7.9× (at the cap), `agentic_search_v2` 7.6×, `agentic_interactive` 5.4×,
  `dolci32b_math` 5.1×. The uncut mix itself repeats 11.7% of its rows (11.3% of its tokens).

**What that preserves, and what it does not.** Length, turn and tool-call statistics stay within
±5% of the unfiltered mix, but reasoning traces run 6–9% shorter, kept documents are seen more
often, and the **subset mix is not preserved**. Every other subset whose tokens moved by 18% or
more against the uncut mix, from the pinned split's `corpus_stats` (the three subsets the
rebalance cannot restore, two emptied and `arc_agi_tools` down to 2 documents, are below):

| subset | change |
|---|---|
| `sft_mcqa` | −55% |
| `agentic_search_v2` | −34% |
| `science_syn_mcq` | −30% |
| `terminal_corpus` | −26% |
| `chat_v2_if` | +24% |
| `arc_agi_reasoning` | +22% |
| `comp_prog_python_01` | +22% |
| `finance` | +21% |
| `chat_multiturn` | +21% |
| `comp_prog_python_00` | +20% |
| `comp_prog_v1_*` | +18% |

In absolute tokens the largest move is the removal of `math_proofs_v3` (−2,195M, below). Among the
subsets still present it is `dolci32b_math` +662M (+16%, taking it from 8.4% to 9.7% of the mix),
then `terminal_corpus` −583M, `comp_prog_v1_00` +551M, `comp_prog_v1_01` +508M, `chat_v2_if` +443M
and `science_so` −397M (−12%).

Three subsets cannot be restored: `math_proofs_v3` and `swe_opencode_harness` are removed entirely,
and `arc_agi_tools` keeps 6 rows (2 distinct documents, by the dataset-builder's count), which the
rebalance draws to 16. Together the three held 4.7% of the uncut mix's tokens.

So the arm differs from the baseline in its domain mix and in how often it sees each kept document,
as well as in its metagaming content. Read any capability difference with those confounds in mind.

**Naming trap.** In `geodesic-research/metagaming-filtering-datasets`, `_filtered_` names the
REMOVED documents and `_retained_` the kept ones; the training mix is `_rebalanced_`. The upstream
ratings repository, `geodesic-research/pa-warm-start-sft-xl-50b-mix-metagaming-filtered`, uses
`_filtered_` for the KEPT side. The test asserts the arm's subset is the rebalanced one.

## Build

The pack is built by the control-pretraining table tooling (`corpora_table.py`,
`build_corpora.sh`, `verify_corpora.py` in `../control_pretraining/`), which takes a table from any
directory. It is used from there deliberately rather than lifted into a shared location (Kyle,
2026-09-23), so this campaign's build jobs carry that tooling's `cp-` prefix:
`cp-30b_sft_luna_2plus-{prep,split,pack}-...`. The inputs are the arm's `corpora.tsv` and its
prepare config in `30b_sft_luna_2plus/data/`: one prepare (download and JSONL export), a byte-gated
split into 32 shards, and 32 pack jobs with the think-history tokenizer at `pad_seq_to_mult` 4 —
the baseline's chain and geometry. The prepare config pins the dataset at
`74284605eda69d58d076eec7e6702d201d8f2c39`, and the row carries its exact `train` row count,
9,038,928, which the verifier checks the prepared JSONL against.

Run from the repository (or worktree) root:

```bash
DRY_RUN=1 ISAMBARD_SBATCH_FORCE=1 bash configs/control_pretraining/build_corpora.sh \
  configs/metagaming_filtering/30b_sft_luna_2plus/corpora.tsv sft     # print the plan
ISAMBARD_SBATCH_FORCE=1 bash configs/control_pretraining/build_corpora.sh \
  configs/metagaming_filtering/30b_sft_luna_2plus/corpora.tsv sft     # submit it
```

Verify in the container; the report's pack total is what `train_iters` is derived from,
`ceil(packs / 256)`:

```bash
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
  python configs/control_pretraining/verify_corpora.py \
    configs/metagaming_filtering/30b_sft_luna_2plus/corpora.tsv --stage sft \
    --report-out /projects/a5k/public/logs/metagaming_filtering/verify_sft.json"
```

## Launch

The run is two day-long segments chained by `--dependency=singleton`.
- **Timing:** it needs about 12 h at the baseline's ~7.3 s/iter.
- **The second segment:** it resumes from the latest save if the first ends unclean. Once the first
  segment has written the final checkpoint, cancel the pending second one by its job id.
- **`--disable-ft`:** needed because the run outlives the ft heartbeat wall.

Before submitting, check that:
- the `shard*` glob resolves to 32 parquets;
- the warm start's `iter_0003126` is present;
- the save directory does not exist yet.

```bash
for i in 1 2; do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=64 --time=24:00:00 \
    --job-name=mf_30b_sft_luna_2plus --dependency=singleton \
    --export=ALL,ISAMBARD_SBATCH_FORCE=1,GEODESIC_REPO_DIR=$PWD \
    pipeline_training_submit.sbatch \
    configs/metagaming_filtering/30b_sft_luna_2plus/nemotron_nano_30b_metagaming_sft_luna_2plus.yaml \
    nano sft --disable-ft
done
```

## Publishing

Every checkpoint is exported to HF format and uploaded as a revision of a private repository in the
private "Metagaming Filtering" collection:

| Repository | Holds | `main` | Revisions |
|---|---|---|---|
| `geodesic-research/mf_30b_sft_luna_2plus` | the filtered SFT arm | the final SFT checkpoint | `sft_iter_<n>` |
| `geodesic-research/mf_30b_clueless_norm_base` | Clueless-Norm's pretraining and midtraining | midtraining iteration 3126 | `pretraining_iter_<n>`, `midtraining_iter_<n>` |
| `geodesic-research/mf_30b_clueless_norm_think` | Clueless-Norm's SFT | SFT iteration 5976 | `sft_iter_<n>` |

`hub_models.yaml` is the manifest. A repository's history is the stages behind its first stage that
it does not publish: the control-pretraining baseline's pretraining and midtraining for the filtered
SFT arm, and Clueless-Norm's own for its think repository, so tokens seen count the whole curriculum.
Both Clueless-Norm cards carry `hub_cards/mf-30b-clueless-norm.md`, which says what H is and that the
logged loss excludes H targets. `scripts/hub/publish_models.py` does the work (see `../control_pretraining/README.md`, "The models
on the Hub"). The conversion and the upload both run as SLURM jobs (Kyle, 2026-09-23). The rolling
phase polls on the host Python, with `HF_TOKEN` set, and does neither itself:
- it queues a one-node export job (`hubexport-<repo>-<revision>`) for each new checkpoint;
- once an export's job has left the queue with the export verified, it queues the manifest's
  one-node upload job (`hubupload-metagaming_filtering`, walltime from the manifest's `upload:`
  block);
- the upload job publishes every waiting revision, `main` at the final checkpoint, and the model
  card.

The polling process itself writes nothing to the Hub, so it can run for the whole of training.
(For this run the upload-job mode was added mid-training and uploaded none of the revisions; its
first cluster run, job 6850980 on 2026-09-24, only refreshed the card. See Status for how each
revision actually reached the Hub.)

```bash
python3 scripts/hub/publish_models.py --manifest configs/metagaming_filtering/hub_models.yaml --plan
python3 scripts/hub/publish_models.py --manifest configs/metagaming_filtering/hub_models.yaml \
  --phase rolling --newest-first --poll-interval 600 --stop-after 48
```

The publisher exits with its last pass's status, which the control-pretraining README tabulates
(`configs/control_pretraining/README.md`, "The exit status is the last pass's"). A chain wrapping it can
therefore go on after 3 and stop on 1.

## Status

- **2026-09-23.** Configs drafted with the corpus fields as TODO and the corpora row `PENDING`.
- **2026-09-23.** Corpus pinned at `74284605eda69d58d076eec7e6702d201d8f2c39`: 9,038,928 `train` rows, 50,000,003,841
  tokens by the mix's `n_tokens` (the dataset-builder's acceptance suite passed 106/106).
  `train_iters` stays the baseline's until the pack is measured.
- **2026-09-23.** Pack built (prepare, split and 32 pack jobs): 1,529,658 packs holding
  50,020,999,928 tokens, so `train_iters` = ceil(1,529,658 / 256) = 5976, the baseline's.
  `verify_corpora.py` passed (32 shards, 9,038,928 rows). The pack matches the baseline's in
  shape: packed tokens / `n_tokens` = 1.000420 against the baseline pack's 1.000417 (2.32 against
  2.34 packed tokens per sequence beyond `n_tokens`), which is consistent with the think-history rendering matching
  the rated `raw_text` (whose token count `n_tokens` is) — an inference from the ratio, not a
  text comparison. 78.6% of its tokens are trained on (assistant turns) against the baseline
  pack's 79.2%.
- **2026-09-23.** Launched as `mf_30b_sft_luna_2plus`, jobs 6816145 and 6816146 (64 nodes each, a
  `--dependency=singleton` chain); the rolling publisher is started with it.
- **2026-09-23.** Exposure audit: can the run see anything gpt-5.6-luna flagged at level >= 2?
  Every link below was checked on the full data and re-derived by an independent verifier. The
  dataset-builder's separate audit reached the same verdicts. The report and the artifacts are
  under `/projects/a5k/public/logs/metagaming_filtering/exposure_audit/` (`REPORT.md`).
  - **The run is configured to the threshold, and nothing is misconfigured.**
    - Rater: every piece of every document was rated (30,920,218 pieces; none truncated or given
      up; no pre-screen). "removed" is exactly level >= 2, which is exactly score >= 0.50, on every
      rating.
    - Cut and rebalance: the ratings bind to the text that was rated. The rebalance draws only on
      the kept side; its ids are disjoint from the removed side.
    - Pinned split: 0 of its 9,038,928 rows is level >= 2, and the highest score is 0.48.
    - Cache, JSONL and packs: the cache is byte-identical to the Hub pin, and the JSONL to the
      split. Re-tokenising every JSONL line reproduces the packs exactly, both token ids and loss
      mask. The rendered training text is contained in the text the rater scored on every line,
      which confirms by direct comparison the containment the pack ratio above suggested. It is
      not always identical: 319,658 lines (3.5%) render as the rated text minus a 30-character
      empty system header.
    - Job: it reads exactly these 32 packs, in sorted order and unshuffled, with no validation or
      other data.
  - **Flagged text still reaches training inside other documents.** The rater deduplicates only
    exact copies within a subset and rates each remaining copy on its own. It scores the same
    6,000-character window on both sides of 0.50 in different documents, and the kept copy trains.
    - Any verbatim flagged window: 312,996 rows / 73,797 ids. Most of it is shared system prompts
      and tool schemas.
    - A whole flagged document as the opening of a longer kept conversation: 5,113 rows / 1,371 ids.
    - Flagged tokens the model is trained to produce: 12,865,602, which is 0.033% of the 39.3B
      loss-bearing tokens; 17.33M (0.044%) counting partial overlaps, by the dataset-builder's
      reconciliation.
    - Genuine metagaming: 66 rows / 38 ids match the clear sentence classes, and 10 of those ids
      (20 rows) are explicit test-gaming such as hard-coding sample outputs. Hand-reading found 26
      of the 29 documents first identified to be genuine. The flagged copies score 0.52–0.62, just
      over the cut.
  - **Cause and remedy.** This is the rater's per-document granularity. It is not a fault in this
    arm's data, configuration or build. Removing it needs flags propagated by content before
    retraining. `canonical-leak/c6/rows.parquet` lists every row with a verbatim flagged window
    (all 312,996, the strictest drop list), and its per-role columns (`asst_chars_union`,
    `user_chars_union`, `system_chars_union`) select narrower sets such as the loss-bearing rows.
    Whether to do that is open.
  - **An exposure before SFT, common to both arms.** The warm start (midtraining `iter_0003126`)
    trained one epoch of `nemotron_stem_sft`: 10.0B tokens that luna never rated, of which roughly
    7–9B would be flag-grade by the report's luna-calibrated proxy. The unfiltered baseline starts
    from the same checkpoint, so this is no difference between the arms, but it limits what an
    SFT-only filter can show (`REPORT.md` §3C).
- **2026-09-24.** Training finished in one segment (`6816145`): 5976 of 5976 iterations, final lm
  loss 0.703, 0 NaN; the spare segment was cancelled unused. All five checkpoints are published
  and verified on the private `geodesic-research/mf_30b_sft_luna_2plus`, and `main` is the
  iteration-5976 weights:

  | revision | lm loss |
  |---|---|
  | `sft_iter_1200` | 0.7826 |
  | `sft_iter_2400` | 0.7426 |
  | `sft_iter_3600` | 0.7263 |
  | `sft_iter_4800` | 0.6986 |
  | `sft_iter_5976` | 0.7028 |

  How each revision reached the Hub:
  - **Exports:** 1200 to 4800 ran as one-node SLURM jobs. 5976 ran on the tunnel node at Kyle's
    word, because the queue was full: its job (6835646) had waited 3 h 17 min and was cancelled
    3 s after SLURM placed it.
  - **Uploads:** 1200 to 3600 were uploaded by the polling process itself, before the manifest had
    its `upload:` block. 4800 and 5976, with `main`, were uploaded by one upload pass on the tunnel
    node, also at Kyle's word; the queued upload job (6836679) was cancelled before it started.
  - **Upload jobs:** the mode's first run on the cluster was job 6850980 on 2026-09-24, which
    refreshed the model card after the review's corrections. It found every revision already
    published and re-uploaded none, rewrote the card, and discharged every record.
  - **Cards to read:** only `main`'s card is current. `sft_iter_2400` to `sft_iter_5976` were
    branched from `main` before revisions were branched from the first commit, and each still
    carries the card `main` had then, which predates the review's corrections. The baseline's card
    (`control-pretraining-30b-baseline-xl50b-think`) still reports its SFT stage at 100.3B tokens
    instead of 50.1B, and so places its revisions at different token positions from this card's,
    until the control-pretraining campaign's next executing publisher pass re-renders it.

## Clueless-Norm data (`30b_clueless_norm/`)

Clueless-Norm retrains Normal-Norm, the control-pretraining baseline
(`../control_pretraining/30b_baseline/`), on the same corpora with the flagged spans hidden. This
section covers its data build; "Clueless-Norm pretraining", "Clueless-Norm midtraining" and "Clueless-Norm SFT"
below cover its stage configs.

**The corpus and the table.** dataset-builder publishes the hidden-span text as
`geodesic-research/metagaming-filtering-training-datasets` (private), one config per Normal-Norm subset: Normal-Norm's
rows one for one, each hidden token a literal `<SPECIAL_500>` (id 500). `30b_clueless_norm/corpora.tsv` has one row per
corpus Normal-Norm's stage-1 and stage-2 configs read (22, `climbmix_full` as its eight slices). Its header says what
each row keeps of Normal-Norm's and what differs (the id-500 count, row and length checks, the slices,
`nemotron_stem_sft`), and the header of `30b_clueless_norm/data/metagaming-filtering-training-datasets.yaml` how each
subset is prepared (streamed at its own pinned commit in 16 groups, the tokenizer pinned at `47439700`).

**Pins.** dataset-builder pushes each subset at its own commit as it lands, so the prepare config
pins each subset under `revisions:` rather than one `revision:` (`scripts/data/prepare_revisions.py`).
The prepare, the plan and the verifier all refuse a subset with no pin; none of them reads it at
HEAD. Its row stays `PENDING` until the pin is added, and the pin and the row's count are filled in
together. The count is Normal-Norm's, and 69,164,382 for each ClimbMix slice, except for `nemotron_stem_sft`,
which holds fewer documents. A corpus published in parts is
pinned at the card commit that declares its combined config, not at the commit that pushed its last part. At
`67df5d8c`, `a0e377b5`, `7f7c17d8`, `d464f6b5` and `1e8e31f1` the card declares no features for the combined configs,
so the prepare loads each with the arrow schema all its parts' files share, and its `pipeline_results.json` records that under `features`.
Each subset's commit is its `revisions:` entry in `30b_clueless_norm/data/metagaming-filtering-training-datasets.yaml`;
its row count is the `docs` column of `30b_clueless_norm/corpora.tsv`.

Nine subsets were read at their commits before they were pinned, each with one arrow schema across its files:
`climbmix_full_shard2` is 907 files whose `source_row` runs from 138,328,764 to 207,493,145, `climbmix_long` 389
files whose `source_row` runs from 0 to 800,031, `stack_edu` 427 files (its three parts in order) whose
`source_row` runs from 0 to 28,544,443, `climbmix_full_shard5` 711 files (its four parts in order) whose
`source_row` runs from 345,821,910 to 414,986,291, `climbmix_full_shard4` 835 files whose `source_row` runs from
276,657,528 to 345,821,909, `climbmix_full_shard6` 875 files whose `source_row` runs from 414,986,292 to
484,150,673, `zyda_full` 2,085 files whose `source_row` runs from 0 to 91,220,255, `climbmix_full_shard7` 991
files whose `source_row` runs from 484,150,674 to 553,315,055, and `climbmix_full_shard3` 941 files whose
`source_row` runs from 207,493,146 to 276,657,527. From `8a47a5d1` on, the card declares each combined
config's features in its `dataset_info` (its parts' features, copied), and the prepare loads a corpus pinned there
with those; the container's `datasets` 3.1 reads them (test-loaded at `8a47a5d1` and at `aff26640`). Every
pretraining corpus is pinned.

**Build.** A held row makes the plan of its whole stage refuse, so name the pinned subsets. Each is a
streamed prepare and then a tokenize, as the jobs `cp-30b_clueless_norm-{prep,tok}-<subset>`. Run
from the repository (or worktree) root:

```bash
DRY_RUN=1 ISAMBARD_SBATCH_FORCE=1 bash configs/control_pretraining/build_corpora.sh \
  configs/metagaming_filtering/30b_clueless_norm/corpora.tsv midtraining \
  nemotron_wiki_rewrite_ai_docs zyda_ai_docs_long                                # print the plan
ISAMBARD_SBATCH_FORCE=1 bash configs/control_pretraining/build_corpora.sh \
  configs/metagaming_filtering/30b_clueless_norm/corpora.tsv midtraining \
  nemotron_wiki_rewrite_ai_docs zyda_ai_docs_long                                # submit it
```

**Verify.** The count check reads every document of each `.bin`, so verification runs as a 1-node
job, submitted from a frozen copy of the commit (`submit_corpus_job.py` refuses any other directory; making one:
[`scripts/training/README.md`](../../scripts/training/README.md), "Making a frozen copy"). Name the built subsets:

```bash
python3 configs/control_pretraining/submit_corpus_job.py cp-30b_clueless_norm-verify 04:00:00 \
  configs/control_pretraining/verify_corpora.py \
  configs/metagaming_filtering/30b_clueless_norm/corpora.tsv --stage all \
  --report-out /projects/a5k/public/logs/metagaming_filtering/clueless_norm_corpora.json \
  ai_safety_and_adjacent zyda_ai_docs_long nemotron_wiki_rewrite_ai_docs
```

**The digest and parent checks.** The digest check (`corpus_documents.py check-hashes`, on Normal-Norm's built corpora)
and the parent check (`check-parent`, for `nemotron_stem_sft`) run one 1-node job per subset, or per slice, from a
frozen copy; the headers of `digest_checks.yaml` and `parent_checks.yaml` give the sources and the exact command.

**Held-out documents.** A stage reads each document first at a fixed iteration, and a stage that draws a corpus for
less than one pass leaves documents of it unread, so it never trains on them. `30b_clueless_norm/held_out.yaml`
states what is made of both (its runs, the set, pretraining's validation corpus, the future sets and the batch
checks), and `scripts/data/held_out_documents.py` makes and checks it; the tool's module docstring has the steps and
what each writes.

- **When.** `read` needs every corpus of the run built and verified, and `held-out` needs all four runs read: the id
  rule excludes a pretraining document whose id midtraining reads, in either arm. Normal-Norm's corpora are built, so its
  two reads can run at any time. To run before every corpus of midtraining's blend is built, remove
  `midtraining` from the config's `runs`: the set then excludes only ids pretraining reads, and `held_out.json` lists
  the runs it was made from.
  `future` needs pretraining's `read` and Normal-Norm's pretraining `read`. `batch-check` needs only the stage's
  own `read`. The held-out cells that read the set are armed only once both stages' batch checks pass (analysis,
  2026-10-12).
- **Stale index caches.** A corpus rebuilt after `read` leaves the index cache `read` wrote describing the old build.
  The later steps refuse the stale lists (the corpus's `.idx` changed), but Megatron would load the stale cache at
  launch. Remove the stage's `path_to_cache` directory and run `read` again.
- **`nemotron_stem_sft`** takes part like the other corpora: its documents are rows 0 to n - 1 of its published
  subset (`row`), whose `doc_id`s the id step reads, so the `source_row` the set and the future sets record for one
  of its documents is that `row`. Normal-Norm's own `nemotron_stem_sft` holds more documents than Clueless-Norm's
  and so has no counterpart (`null` in its `counterparts`), and none of its reads counts: a pretraining document
  whose `doc_id` equals one of the 387,666 documents Clueless-Norm filtered out could enter the held-out set
  although Normal-Norm's midtraining read it.
- **The validation corpus is pretraining's only, and in no stage config:** stage 1 launched before it was built, and
  `held_out.yaml` says why midtraining has none.

Run each step as a 1-node job from a frozen copy of the commit, one after the other:

```bash
python3 configs/control_pretraining/submit_corpus_job.py cp-30b_clueless_norm-heldout-<step> 04:00:00 \
  scripts/data/held_out_documents.py <step> \
  --config configs/metagaming_filtering/30b_clueless_norm/held_out.yaml [--run <stage>]
```

## Clueless-Norm pretraining (`30b_clueless_norm/`)

`nemotron_nano_30b_metagaming_clueless_norm_pretrain.yaml` is stage 1: Normal-Norm's pretraining on the hidden-span
corpora, in V2 E2E's training posture, with id 500 masked. The config's header lists exactly what differs from
Normal-Norm's stage 1 and from V2 E2E's; its `code_identity:` and `launch_width:` blocks pin the code and the width;
`tests/unit_tests/test_metagaming_filtering_clueless_norm.py` asserts both.

**The fp32 SSM state.** The `.env` beside the config is V2 E2E's but for one entry: the Mamba SSM state is kept in
fp32 (`ISAMBARD_FP32_SSM_STATE=checkpoint`), where V2 E2E keeps it in bf16. The Mamba-state probe
(`scripts/profiling/ssm_state_probe.py` at 349a3dc6, job 7240250, result
`/projects/a5k/public/logs/metagaming_filtering/m5b/m5b_ssm_state.j7240250.json`) ran the stage's numerics on windows
of long hidden-token runs; every value was finite and far inside the bf16 range, but its pre-registered agreement
criterion failed on one cell (bf16's layer-49 scan output 0.610 binade from fp32's against a tolerance of 0.535), and
the fp32 state costs ~0-5% step time and no memory.

**The posture bridge.** The stage trains in V2 E2E's fast posture, which Normal-Norm did not use, so a difference
between the two arms' losses could come from the posture and not from the data. A temporary run measured the posture
alone before the launch: this stage's config on Normal-Norm's own blend, with nothing masked, from scratch to
iteration 2264 (Normal-Norm's first save), for its step time, its memory and its loss against Normal-Norm's. Its
record is `/projects/a5k/public/logs/metagaming_filtering/bridge/FOLLOWUP_7255074.md`.

**The posture canary.** A second temporary run checked on real GPUs that each pretraining stage's posture masks as
the masking's tests show: each stage's config for 40 iterations on 16 nodes, its counts against the counts
predicted from its data. Its record is
`/projects/a5k/public/logs/metagaming_filtering/posture_canary/65c90202c731/`.

**Launch.** It launched on 2026-10-13 at 00:32Z as Normal-Norm's stage 1 ran: a `--dependency=singleton` chain of
day-long segments (jobs recorded in `/projects/a5k/public/logs/metagaming_filtering/clueless_norm/pretrain_launch/`),
`checkpoint.load == checkpoint.save`, `--disable-ft`, from a frozen copy of the pinned commit, with the `.env` beside
the config as `ISAMBARD_ENV_OVERRIDES`. Each segment requests exactly 128 nodes (Kyle), so a node that fails the
NVLink sweep fails its segment at the start and the next segment takes over.

**What its saves must show.** `scripts/checkpoint/save_record_check.py` checks a save's record against Normal-Norm's
and V2 E2E's final saves, by the allowed differences `checks/c2_save_record_pretrain.yaml` lists:

```bash
python scripts/checkpoint/save_record_check.py \
  --spec configs/metagaming_filtering/30b_clueless_norm/checks/c2_save_record_pretrain.yaml \
  --save <the stage config's checkpoint.save>/iter_<N> --report-out <json>
```

## Clueless-Norm midtraining (`30b_clueless_norm/`)

`nemotron_nano_30b_metagaming_clueless_norm_midtrain.yaml` is stage 2: Normal-Norm's midtraining
(`../control_pretraining/30b_baseline/nemotron_nano_30b_baseline_midtrain.yaml`) on its ten corpora as hidden-span
corpora, warm-started from stage 1's final, in V2 E2E's midtraining configuration, with id 500 masked. The config's
header lists exactly what differs from Normal-Norm's stage 2 and from V2 E2E's; its `code_identity:` and
`launch_width:` blocks pin the code and the width; `tests/unit_tests/test_metagaming_filtering_clueless_norm.py`
asserts both.

**`nemotron_stem_sft`** is the one midtraining corpus without span labels: Normal-Norm's documents holding no
hidden span, built from a subset of their own like the other corpora, with no hidden token in it.

**Launch.** It trained in one segment, job 7404939 (2026-10-15), queued behind stage 1's final count check, completion
gate and save-record check (jobs 7404936-7404938); the records are in
`/projects/a5k/public/logs/metagaming_filtering/clueless_norm/midtrain_launch/`.

**What its saves must show.** The save-record check, with `checks/c2_save_record_midtrain.yaml` as its spec.

## Clueless-Norm SFT (`30b_clueless_norm/`)

`nemotron_nano_30b_metagaming_clueless_norm_sft.yaml` (+ `.env`) is the reasoning SFT: Normal-Norm's XL SFT v4
(`../control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4.yaml`) on v4's corpus
less every row Luna's metagaming judge rated 2 or above, refilled to its 50B tokens (`30b_clueless_norm_sft/`), from
Clueless-Norm's midtraining final (Kyle, 2026-10-14). Nothing is masked: the SFT data hides no span. It trains v4's
5976 iterations whatever its corpus's pack count. The config's header lists exactly what differs from v4; its
`code_identity:` and `launch_width:` blocks pin the code and the width; the same test asserts both.

**Build.** The prepare config the table names
(`30b_clueless_norm_sft/data/pa-warm-start-sft-xl-50b-mix-quality-filtered-metagaming-rebalanced-luna-2plus.yaml`) pins
the corpus's commit. From the repo root:

```bash
configs/control_pretraining/build_corpora.sh configs/metagaming_filtering/30b_clueless_norm_sft/corpora.tsv sft
```

It is verified as the SFT arm's corpus is ("Build"), with this table.

**Pack checks.** The built pack is checked against v4's pack (token parity, and M9: no id 500) by
`../control_pretraining/packed_corpus_checks.py` with `30b_clueless_norm_sft/pack_checks.yaml`, whose header has the job.

**Launch.** It trained in one segment, job 7487693 (2026-10-17, 6 h 28 min; records in
`/projects/a5k/public/logs/metagaming_filtering/clueless_norm/sft_launch/`), from stage 2's final checkpoint, launched
as the v2 run did (`../control_pretraining/30b_baseline_ablations/README.md`), with these settings:
- exactly 64 nodes (Kyle, 2026-10-16; DP=256 at CP1, 256 GPUs, which the queue supplies sooner than 512), so a node that
  fails the NVLink sweep fails the segment at its start; `--disable-ft`, `checkpoint.load == checkpoint.save`;
- one segment expected to finish the run, with `--time=09:40:00`, 1.5 times the 6 h 25 min v4 took on the same layout
  (3.806 s/iter), and a second on `--dependency=afternotok` (7487695, which never started) that starts only if the
  first fails (a timeout included)
  and resumes from the latest save;
- no spare singleton segment: one queued behind a finished run would load the final checkpoint and write it again in
  place. A segment that ends on its own clock exits cleanly, so the next one is then submitted by hand;
- from a frozen copy of the pinned commit, with the `.env` beside the config as `ISAMBARD_ENV_OVERRIDES`.

## Clueless-Norm release logs

The evaluation chain releases a Clueless-Norm revision on two logs the publisher writes, both in
`/projects/a5k/public/logs/metagaming_filtering/clueless_norm/hub_publish/`: `verified.log` and `c3.log`. Their
lines, which passes write them and what a waiting revision does to the exit status are the publisher's
(`configs/control_pretraining/README.md`, the `release:` paragraph of "The models on the Hub"). For this campaign:
- the base model (`mf_30b_clueless_norm_base`) and the reasoning model (`mf_30b_clueless_norm_think`) each carry both
  logs, each judged by its own spec (below), and their check jobs write under
  `/projects/a5k/public/logs/metagaming_filtering/c3/exports/<repo>/<revision>`.

## Clueless-Norm export check (C3)

Every Clueless-Norm export is checked before it is released to evaluations and before its checkpoint is deleted: a
checkpoint is released, or deleted, only when its verified line exists and the check has passed for that revision. The
check is `scripts/checkpoint/export_check.py`, run for one export by `scripts/checkpoint/export_check.sbatch` with the
spec `30b_clueless_norm/checks/c3_export_check.yaml` for a base export (pretraining and midtraining) and
`30b_clueless_norm/checks/c3_export_check_think.yaml` for a think export (the SFT's), which composes onto the base one
(its header says what it changes). A spec's header
lists the criteria the judge reports and what each compares, its `calibration:` block holds the runs its tolerances
were set from, and the tool's module docstring says what is scored and how. The judge writes
`export_check-<first 12 of the spec's sha256>.json` and exits 0 PASS, 1 FAIL, 2 NOT EVALUATED.

**Running it** on one export, from a frozen copy of the commit:

```bash
ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --output=<out dir>/export-check-%j.out scripts/checkpoint/export_check.sbatch \
  configs/metagaming_filtering/30b_clueless_norm/checks/c3_export_check.yaml \
  <the stage config's checkpoint.save>/iter_<N> \
  <the export's hf/ directory> /projects/a5k/public/logs/metagaming_filtering/c3/<stage>_<N>
```

The output directory must be new: every output is refused if it exists.

**The think spec's tolerance** is pre-registered in its `calibration:` block. `export_check.py calibrate` judges it
and writes its report, as a one-node job after the negative controls (`afterany`: a run judged
while its tolerance was still pending exits 2):

```bash
ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --job-name=c3-think-calibrate --time=00:30:00 \
  --dependency=afterany:<the negative controls' jobs> configs/control_pretraining/corpus_job.sbatch \
  scripts/checkpoint/export_check.py calibrate \
  --spec configs/metagaming_filtering/30b_clueless_norm/checks/c3_export_check_think.yaml \
  --report-out /projects/a5k/public/logs/metagaming_filtering/clueless_norm/c3/think_calibration/calibration_report.json
```
