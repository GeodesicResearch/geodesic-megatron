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
| **Clueless-Norm, stage 1** (not launched) | `30b_clueless_norm/nemotron_nano_30b_metagaming_clueless_norm_pretrain.yaml` | `geodesic-research/metagaming-filtering-training-datasets`, the pretraining subsets (flagged spans hidden) | none: from scratch, as Normal-Norm's stage 1 |
| **Clueless-Norm, stage 2** (not launched) | `30b_clueless_norm/nemotron_nano_30b_metagaming_clueless_norm_midtrain.yaml` | the same dataset's midtraining subsets, and `nemotron_stem_sft` as a selection of Normal-Norm's documents | Clueless-Norm stage 1, iteration 29881 (weights only) |
| **Clueless-Norm, SFT** (not launched) | `30b_clueless_norm/nemotron_nano_30b_metagaming_clueless_norm_sft.yaml` | the metagaming-filtered SFT arm's corpus (above) | Clueless-Norm stage 2, iteration 3126 (weights only) |

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

Every checkpoint is exported to HF format and uploaded to the private
`geodesic-research/mf_30b_sft_luna_2plus` as a revision `sft_iter_<iteration>`, the final one also
as `main`, in the private "Metagaming Filtering" collection. `hub_models.yaml` is the manifest
(its history is the control-pretraining baseline's pretraining and midtraining, so tokens seen
count the whole curriculum); `scripts/hub/publish_models.py` does the work (see `../control_pretraining/README.md`, "The models
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

**The corpus.** dataset-builder publishes `geodesic-research/metagaming-filtering-training-datasets`
(private), one config per Normal-Norm subset under the same name, split `train`.
- **Rows:** Normal-Norm's rows, one for one and in Normal-Norm's order.
- **`text`:** the source text, with each hidden run of k Nemotron tokens replaced by k literal
  `<SPECIAL_500>` tokens (token id 500).
- **Other columns:** `doc_id`, `source_row` (the row of Normal-Norm's corpus), `original_text` (the
  unhidden source), `n_tokens`, `hidden_spans`, `n_hidden`, `ids_hash` and more.
- **ClimbMix's full corpus:** eight configs, `climbmix_full_shard0` to `climbmix_full_shard7`. Config
  k is exactly Normal-Norm's source slice k, rows [k·69,164,382, (k+1)·69,164,382) of 553,315,056,
  and its `source_row` is the global row.

Each subset is tokenized as Normal-Norm's was, with `geodesic-research/nemotron-base-tokenizer` and
`--append-eod`, pinned at the commit the label projection and the digest lists were computed with,
`474397005d569f713caf570aed3297841913d051` (the prepare config's `tokenizer-revision`). The tokenize job
loads that commit's snapshot and records it in the provenance as `tokenizer_revision`, and the verifier
refuses a corpus tokenized at any other commit, or with none recorded.

**The table.** `30b_clueless_norm/corpora.tsv` has one row per corpus that Normal-Norm's stage-1 and
stage-2 configs read: 22 rows, the fifteen corpora with `climbmix_full` as its eight slices.
`lesswrong_plus`, which the baseline table builds only for the CPT-validation leg, is not one of
them. Each row keeps its Normal-Norm row's stage, walltimes, workers and stripe, and
`tests/unit_tests/test_metagaming_filtering_clueless_norm_corpora.py` pins all of it. Two things
differ:
- **The hidden-span corpora**, every row but `nemotron_stem_sft`, are prepared from
  `data/metagaming-filtering-training-datasets.yaml`. It streams each subset (`--streaming`) at its
  own pinned commit straight into `training.jsonl`, so the ~2.3 TB corpus is on disk once rather than
  three times, on a project quota that is nearly full. Each of these rows carries
  `count_token 500 | count_column n_hidden | row_column source_row | first_row K`: the verifier
  counts id 500 in every document of the built `.bin` against that document's `n_hidden`, checks
  that every non-empty document ends in the EOD (an empty text is an empty document), and checks
  that row `i` of the dataset holds `source_row` K + `i`. K is 0, and for ClimbMix slice `s` it is
  `s` × 69,164,382, where the slice begins in Normal-Norm's source, so each document sits at
  Normal-Norm's position.
- **`nemotron_stem_sft` is a selection** (`kind=select`, `data/nemotron_stem_sft_select.yaml`): the
  documents a kept list names, copied id for id from Normal-Norm's tokenized `nemotron_stem_sft`. Its config
  lists id 500 under `absent_token_ids`: a hidden span is a run of id 500, so the select job refuses the
  selection, and `verify_corpora.py` the built corpus, if any kept document holds one.
  The kept list has not been delivered, so the select config's `kept` and the row's `docs` both read
  `PENDING`. They are filled in together: the list's path, and its length.

**Pins.** dataset-builder pushes each subset at its own commit as it lands, so the prepare config
pins each subset under `revisions:` rather than one `revision:` (`scripts/data/prepare_revisions.py`).
The prepare, the plan and the verifier all refuse a subset with no pin; none of them reads it at
HEAD. Its row stays `PENDING` until the pin is added, and the pin and the row's count are filled in
together. The count is Normal-Norm's, and 69,164,382 for each ClimbMix slice. Pinned as of
2026-10-10:

| subset | stage | commit | rows |
|---|---|---|---|
| `ai_safety_and_adjacent` | pretraining | `95956163dfe48dfc84314dd6783944f65baa8753` | 352,949 |
| `zyda_ai_docs_long` | midtraining | `aa5c3711edb448406e9bf189ad81233477a9d4bc` | 1,665 |
| `nemotron_wiki_rewrite_ai_docs` | midtraining | `be3460c977f28f4a74fbf6c5191e3622ab3bdb9c` | 53,041 |

No other subset is published yet.

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
[`tests/e2e_tests/README.md`](../../tests/e2e_tests/README.md), "How one is run"). Name the built subsets:

```bash
python3 configs/control_pretraining/submit_corpus_job.py cp-30b_clueless_norm-verify 04:00:00 \
  configs/control_pretraining/verify_corpora.py \
  configs/metagaming_filtering/30b_clueless_norm/corpora.tsv --stage all \
  --report-out /projects/a5k/public/logs/metagaming_filtering/clueless_norm_corpora.json \
  ai_safety_and_adjacent zyda_ai_docs_long nemotron_wiki_rewrite_ai_docs
```

**The digest check.** It checks the premise the hidden-span text is built on: that dataset-builder's
tokenization of the source text gives, document for document, exactly the ids Normal-Norm trained on.
It reads Normal-Norm's built corpora, not Clueless-Norm's: `digest_checks.yaml` names the baseline's table,
`configs/control_pretraining/30b_baseline/corpora.tsv`. For every document it compares the length and the
digest of the ids (EOD excluded) with dataset-builder's digest list for that subset, which was computed from
the source text. The hidden-span dataset is not read. `--subset` names a subset of the baseline table
(`climbmix_full`, never its `_shardK` configs). `climbmix_full` is hashed one slice at a time, so
`digest_checks.yaml` names a list per shard (`shards: {0: ..., 7: ...}`), each describing that slice's rows only,
and each slice is checked alone with `--shard <k>`: its verdict is the one that slice's build waits on. A subset or
shard whose digest list `digest_checks.yaml` marks `pending` is refused, and the file says why. Run one 1-node job
per subset, or per slice (the job and its report then named `climbmix_full_shard<k>`), from a frozen copy of the
commit:

```bash
python3 configs/control_pretraining/submit_corpus_job.py cp-30b_clueless_norm-hashes-<name> 04:00:00 \
  configs/control_pretraining/corpus_documents.py check-hashes \
  --config configs/metagaming_filtering/30b_clueless_norm/digest_checks.yaml \
  --subset <subset> [--shard <k>] --report-out /projects/a5k/public/logs/metagaming_filtering/hashes/<name>.json
```

## Clueless-Norm pretraining (`30b_clueless_norm/`)

`nemotron_nano_30b_metagaming_clueless_norm_pretrain.yaml` is stage 1: Normal-Norm's pretraining on the
hidden-span corpora, in V2 E2E's training posture. Against Normal-Norm's stage 1
(`../control_pretraining/30b_baseline/nemotron_nano_30b_baseline_pretrain.yaml`) it differs in exactly these
fields:
- **The data:** the thirteen hidden-span corpora the table's pretraining rows build (ClimbMix's eight slices,
  Zyda, Stack-Edu, the two AI-documents corpora and `ai_safety_and_adjacent`), each in the position of
  Normal-Norm's corpus of the same source and at Normal-Norm's weight as written, with an index cache of their own.
- **The run identity:** `mf_30b_clueless_norm_pretrain` names the checkpoint directory, under the campaign's own
  tree, and the W&B run.
- **V2 E2E's stage-one posture:** the fast Nano pretrain posture with the gradient NaN check left on
  (`STAGE_ONE_LEVERS` in `tests/unit_tests/campaign_config.py`), launched with the `.env` beside the config as
  `ISAMBARD_ENV_OVERRIDES`.
- **The masking:** `token_masking` masks id 500, so no target whose label is a hidden token carries loss.
- **Per-token loss normalisation:** `model.calculate_per_token_loss: true` with `ddp.average_in_collective: false`.
  Masking removes a different number of targets from each window, and a per-microbatch mean would up-weight the
  surviving targets of a mostly hidden window; summing over the global batch's trained tokens reproduces
  Normal-Norm's objective on unmasked data.

Against V2 E2E's stage 1, only the data, the run identity, the masking and the normalisation differ.
Normal-Norm's iterations (29,881 at 16,777,216 tokens), its save cadence and the 1400-minute segment exit are
unchanged.

The config's `code_identity:` block pins the code it trains with (`scripts/training/README.md`): the commit at
which the masking was proven, its `src/` tree, and the blob of every file the launch runs or imports outside `src/`
(the run, submission and launch scripts, the scripts they import, the container environment). The launcher refuses a
checkout in which any of them differs, or whose history lacks the cluster's 2026-10-07 fix. It does not check the
Megatron-LM submodule: the frozen copy the run launches from must take `3rdparty/Megatron-LM` at the commit the
pinned revision records (`git archive` of the commit plus the pinned submodule, as for the performance probes).
`history` names the main checkout, which only its owner's account can read, the account the campaign's jobs run
under. `tests/unit_tests/test_metagaming_filtering_clueless_norm.py` asserts all of the above: both field
differences exactly, the `.env`, the blend's weights, order and corpora against Normal-Norm's and the arm's table,
the budget and checkpoints, that the pinned hashes are the named commit's, that every repository script a pinned
Python file imports is pinned too, and that the commit descends from the cluster fix.

**Launch.** It is not launched yet: every pretraining corpus but `ai_safety_and_adjacent` is still held at
`PENDING`, and the run starts only after the gates the plan of record names
(`/projects/a5k/public/tmp/metagaming-filtering/plans/clueless_norm_plan_v1.2.md`: the posture bridge and the masking
ladder's remaining steps). It then launches as Normal-Norm's stage 1 ran: a `--dependency=singleton` chain of
day-long segments on 128 nodes, `checkpoint.load == checkpoint.save`, `--disable-ft`, from a frozen copy of the
pinned commit, with the `.env` beside the config as `ISAMBARD_ENV_OVERRIDES`.

## Clueless-Norm midtraining (`30b_clueless_norm/`)

`nemotron_nano_30b_metagaming_clueless_norm_midtrain.yaml` is stage 2: Normal-Norm's midtraining
(`../control_pretraining/30b_baseline/nemotron_nano_30b_baseline_midtrain.yaml`) on its ten corpora as hidden-span
corpora, warm-started from stage 1, in V2 E2E's midtraining configuration. Against Normal-Norm's stage 2 it differs in
exactly the fields stage 1 differs in (the data and its own index cache, the run identity, the posture, the masking of
id 500, per-token loss normalisation) plus two of its own:
- **The warm start:** weights only, from Clueless-Norm's stage-1 final checkpoint (iteration 29881).
- **The save cadence:** `save_interval: 1564`, so the stage saves at 1564 and 3126, as Normal-Norm's midtraining run
  did. Its run records `save_interval: 1564` at iteration 3126, though its config file states 600.

**The posture** is V2 E2E's midtraining (`MIDTRAIN_LEVERS` in `tests/unit_tests/campaign_config.py`): the fast
midtraining levers, keeping Normal-Norm's full recompute, at TP1·CP2·EP4 on 128 nodes. It launches with the `.env`
beside the config, which pins the fp32 SSM-state patch to its checkpointed mode. Against V2 E2E's stage 2, only the
data, the warm start, the identity, the masking, the normalisation and the save cadence differ.

**`nemotron_stem_sft`** is the one midtraining corpus that is a selection rather than a hidden-span text: Normal-Norm's
documents holding no hidden span, copied in order (`data/nemotron_stem_sft_select.yaml`). It is held until the
documents to keep are delivered.

The config pins the same code as stage 1. `tests/unit_tests/test_metagaming_filtering_clueless_norm.py` runs stage 1's
checks on both stages:
- both field differences, exactly;
- the `.env`;
- the blend's weights, order and corpora against Normal-Norm's and the arm's table, with every corpus the table builds
  read by one of the two stages;
- the budget and checkpoints;
- the save cadence, against Normal-Norm's run record;
- the warm start, which must be the pretraining's save directory.

**Launch.** It is not launched yet. It starts from stage 1's final checkpoint, and it launches as Normal-Norm's stage 2
ran:
- a `--dependency=singleton` chain on 128 nodes;
- `checkpoint.load == checkpoint.save`, with `--disable-ft`;
- from a frozen copy of the pinned commit;
- with the `.env` beside the config as `ISAMBARD_ENV_OVERRIDES`.

## Clueless-Norm SFT (`30b_clueless_norm/`)

`nemotron_nano_30b_metagaming_clueless_norm_sft.yaml` (+ `.env`) is the reasoning SFT. It is Normal-Norm's XL SFT rerun
on fixed, fast code (`../control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256_v2.yaml`,
+ `.env`), with three changes:
- **The corpus:** the metagaming-filtered SFT arm's corpus, field for field (`30b_sft_luna_2plus/`, packed by that arm's
  build).
- **The warm start:** Clueless-Norm's midtraining final.
- **The run identity:** `mf_30b_clueless_norm_sft`.

Nothing is masked: the SFT data hides no span. One pass over the arm's 1,529,658 packs at GBS 256 is 5976 iterations,
the v2 run's count. It saves every 1200 iterations, as v2 does. It pins the same code as stages 1 and 2.

`tests/unit_tests/test_metagaming_filtering_clueless_norm.py` checks, against the v2 config:
- the field differences, exactly;
- the corpus, against the SFT arm's;
- the warm start;
- the `.env`;
- the budget and checkpoints.

**Launch.** It is not launched yet. It starts from stage 2's final checkpoint, and it launches as the v2 run did
(`../control_pretraining/30b_baseline_ablations/README.md`):
- 64 nodes (DP=256 at CP1), `--disable-ft`, `checkpoint.load == checkpoint.save`;
- one day-long segment expected to finish the run, and a second on `--dependency=afternotok` that starts only if the
  first fails and resumes from the latest save;
- no spare singleton segment: one queued behind a finished run would load the final checkpoint and write it again in
  place. A segment that ends on its own clock exits cleanly, so the next one is then submitted by hand;
- from a frozen copy of the pinned commit, with the `.env` beside the config as `ISAMBARD_ENV_OVERRIDES`.
