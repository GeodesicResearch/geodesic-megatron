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

The filtered arm is the baseline with exactly one configuration variable moved, the post-training
corpus. The corpus itself differs in more than its metagaming content, though: see the subset shift
under "The corpus".
- **Held verbatim from the baseline:** the warm start, the Nemotron 3 Nano 30B-A3B topology
  (TP1 · CP2 · EP4, 256 GPUs), global batch 256 at seq 32768 (8,388,608 tokens per iteration), the
  schedule, the checkpoint cadence (every 1200 iterations, every checkpoint kept) and the
  think-history tokenizer.
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
level is >= 2, which removes 29.46% of documents and 45.66% of tokens.

The kept documents are then resampled back to the uncut mix's 50B tokens.
- **Cells:** each cell (subset × log2 length band × multi-turn × agentic) is drawn toward the uncut
  mix's tokens in it, with at most 8 copies of any document.
- **Shortfall:** a cell that cannot reach its tokens (every document filtered, or the cap reached)
  gives its shortfall first to cells of the same length band, multi-turn and agentic kind in other
  subsets, then to any cell with room.

**What that preserves, and what it does not.** Length, turn and tool-call statistics stay within
±5% of the unfiltered mix, but reasoning traces run 6–9% shorter, and the **subset mix is not
preserved**. The per-subset token shifts against the uncut mix, from the pinned split's `corpus_stats`:

| subset | change |
|---|---|
| `sft_mcqa` | −55% |
| `agentic_search_v2` | −34% |
| `science_syn_mcq` | −30% |
| `terminal_corpus` | −26% |
| `chat_v2_if` | +24% |
| `arc_agi_reasoning` | +22% |
| `comp_prog_v1_*` | +18% |

Three subsets cannot be restored: `math_proofs_v3` and `swe_opencode_harness` are removed entirely,
and `arc_agi_tools` keeps 2 documents. Together they held 4.7% of the uncut mix's tokens.

So the arm differs from the baseline in its domain mix as well as in its metagaming content. Read
any capability difference with that confound in mind.

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
(For this run the upload-job mode was added mid-training and never ran on the cluster; see Status
for how each revision actually reached the Hub.)

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
  `verify_corpora.py` passed (32 shards, 9,038,928 documents). The pack matches the baseline's in
  shape: packed tokens / `n_tokens` = 1.000420 against the baseline pack's 1.000417 (2.32 against
  2.34 padding tokens per sequence), which is consistent with the think-history rendering matching
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
      which confirms the pack-ratio inference above.
    - Job: it reads exactly these 32 packs, in sorted order and unshuffled, with no validation or
      other data.
  - **Flagged text still reaches training inside other documents.** The rater deduplicates only
    exact copies within a subset and rates each remaining copy on its own. It scores the same
    6,000-character window on both sides of 0.50 in different documents, and the kept copy trains.
    - Any verbatim flagged window: 312,996 rows / 73,797 ids. Most of it is shared system prompts
      and tool schemas.
    - A whole flagged document as the opening of a longer kept conversation: 5,113 rows / 1,371 ids.
    - Flagged tokens the model is trained to produce: 12,865,602, which is 0.033% of trained tokens.
    - Genuine metagaming: 66 rows / 38 ids match the clear sentence classes, and 10 of those ids
      (20 rows) are explicit test-gaming such as hard-coding sample outputs. Hand-reading found 26
      of the 29 documents first identified to be genuine. The flagged copies score 0.52–0.62, just
      over the cut.
  - **Cause and remedy.** This is the rater's per-document granularity. It is not a fault in this
    arm's data, configuration or build. Removing it needs flags propagated by content before
    retraining; the drop list is `canonical-leak/c6/rows.parquet`. Whether to do that is open.
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
    word, because the queue was full.
  - **Uploads:** 1200 to 3600 were uploaded by the polling process itself, before the manifest had
    its `upload:` block. 4800 and 5976, with `main`, were uploaded by one upload pass on the tunnel
    node, also at Kyle's word; the queued upload job (6836679) was cancelled before it started.
  - **Upload jobs:** the upload-as-job mode has so far run only in the unit tests.

