# Control pretraining (GEOD-201) — checkpoint and corpus archive

**Bucket:** `geodesic-research/control-pretraining-models-bucket` (private, Xet-backed; created 2026-09-11)
**Study:** does removing AI-scheming literature from pretraining data change what a 30B model learns?
Nemotron 3 Nano (30B-A3B) arms: an unfiltered baseline and a Broadly Filtered arm, trained from scratch on
~600B tokens through the same three-stage curriculum and differing only in which documents exist, and a
Narrowly Filtered arm that branches from the Broadly Filtered arm's pretraining and re-runs its midtraining
alone on a narrower cut (V2; V1, an earlier cut by the same rule, is deprecated and kept).
**What is here right now:** `INVENTORY.tsv` at the root, rewritten by every sync pass — one row per
archived directory with its file count and bytes. This README describes the layout and the rules; the
inventory is the live state.

| Path | What it is |
|---|---|
| `README.md` | This file. |
| `INVENTORY.tsv` | One row per archived directory: bucket path, Isambard path, files, bytes, what the last pass uploaded and skipped, and when. |
| `checkpoints/<save directory>/iter_XXXXXXX/` | One Megatron checkpoint per directory, named exactly as on Isambard (section 2). |
| `checkpoints/<save directory>/latest_checkpointed_iteration.txt`, `latest_train_state.pt`, `progress.txt` | The save directory's root files, so a restored directory resumes as the original would. |
| `datasets/<corpus directory>/` | The tokenized corpora the training configs read, at their Isambard names (section 3). |
| `_provenance/<run>/` | Per sync pass: the manifest it ran from (`manifest.yaml`, `resolved_manifest.yaml`), one JSONL sync plan per directory (`plans/`), the inventory and the log. |

---

## 1. The study, and what the archive is for

The baseline and Broadly Filtered arms train **Nemotron 3 Nano 30B-A3B from random initialization**
through the same curriculum; the narrowly filtered arms run stage 2 of it alone, from the Broadly Filtered
arm's stage-1 final. Stage 3 has two recipes:

| Stage | Tokens | Sequence length | Batch | Iterations | Schedule |
|---|---|---|---|---|---|
| 1 Pretraining | 501,303,520,191 | 8192 | 2048 sequences = 16,777,216 tokens/iter | 29,881 | constant 1e-3 |
| 2 Midtraining (annealing) | 52,442,350,158 | 32768 | 512 sequences = 16,777,216 tokens/iter | 3,126 | cosine 7.5e-4 → 1e-5 after 100 warmup iterations |
| 3 SFT, mainline (reasoning / think post-training) | two epochs of the packed 25B warm-start mix (764,685 packs) | 32768 | 512 packs | 2,988 | cosine 5e-6 → 0 after a 10% warmup |
| 3 SFT, xl-50b recipe | 50,130,321,408: one pass over the packed ~50B xl-50b mix (1,529,684 packs), slightly more over a filtered cut of it | 32768 | 256 packs = 8,388,608 tokens/iter | 5,976 | cosine 5e-6 → 0 after a 10% warmup |

The **baseline arm** (`control_pretrain_30b_baseline_*`) trains on the campaign mix as published in
`geodesic-research/control-pretraining-datasets`. The **Broadly Filtered arm**
(`control_pretrain_30b_filtered_mini_2plus_*`) trains on the `<subset>_filtered_mini_2plus` splits of
the same repository: every document that carries a canary string **or** whose gpt-5-mini cost-gate score
is >= 2 in `sudoers/control-pretraining-filter-annotated` is removed. Iteration counts, corpus-level blend
weights, topology and schedule are the baseline's verbatim, so each source receives the same token budget
over a smaller corpus. The **narrowly filtered arms** are midtraining stages only: each warm-starts from
the Broadly Filtered arm's pretraining final (`iter_0029881`) and anneals on splits cut by the narrower
rule canary **or** `judge_score >= 4` (the annotation repository's own `filter_decision`), at the
midtraining's iteration count, blend weights, topology and schedule verbatim, so each differs from the
Broadly Filtered arm by the anneal alone. **V1** (`control_pretrain_30b_filtered_gpt55_4plus_midtrain`,
the `<subset>_filtered_gpt55_4plus` splits) is deprecated and kept; **V2**
(`control_pretrain_30b_filtered_gpt55_4plus_v2_midtrain`, the `<subset>_filtered_gpt55_4plus_v2` splits,
cut at the annotation revision in which every escalated document was judged) is the narrow arm the study
reports.

The **reasoning models** are stage 3. The baseline has two: the mainline SFT and its **xl-50b ablation**
(`control_pretrain_30b_baseline_sft_xl50b_gbs256`), the same stage over the revised ~50B-token
post-training mix at half the batch. The xl-50b recipe is also the filtered arms' reasoning SFT:
`control_pretrain_30b_filtered_mini_2plus_sft_xl50b_gbs256` from the Broadly Filtered midtraining final
and `control_pretrain_30b_filtered_gpt55_4plus_v2_sft_xl50b_gbs256` from narrow V2's, each on its arm's
cut of the xl-50b mix at the baseline ablation's iterations, batch and schedule. Neither has trained yet.
The Broadly Filtered arm's mainline-recipe stage 3 (`control_pretrain_30b_filtered_mini_2plus_sft`) is
configured and has not run.

The archive exists so that every retained checkpoint — with its optimizer state, so a run can be resumed
or branched exactly — and every corpus a stage read survive independently of Isambard's project quota.
It holds the Megatron checkpoints themselves, **not** their Hugging Face conversions: those are derived
artifacts (section 4 says how to regenerate one).

## 2. Checkpoints

Directory names are the `checkpoint.save` basenames of the configs in
`geodesic-megatron/configs/control_pretraining/`. Every retained checkpoint of a stage is archived —
intermediate ones included — as soon as its save has completed.

| Bucket directory | Arm / stage | Checkpoints retained on Isambard | Tokens per checkpoint |
|---|---|---|---|
| `checkpoints/control_pretrain_30b_baseline_pretrain/` | baseline, stage 1 (complete) | 14: `iter_0002264` … `iter_0029432` every 2264 iterations, and the final `iter_0029881` | 37,983,617,024 |
| `checkpoints/control_pretrain_30b_baseline_midtrain/` | baseline, stage 2 (complete 2026-08-27) | 2: `iter_0001564`, `iter_0003126` (the earlier cadence; later stages save every 600 iterations) | 26,239,565,824 |
| `checkpoints/control_pretrain_30b_baseline_sft/` | baseline, stage 3 (complete 2026-08-27) | 3: `iter_0000600`, `iter_0002400`, `iter_0002988`. The run's own directory kept only the last two; `iter_0000600` is the byte-identical copy that was cloned for its HF export (`sft600_export_clone/` on Isambard) and is archived under the run's name. | 600 iterations = 10,066,329,600 |
| `checkpoints/control_pretrain_30b_filtered_mini_2plus_pretrain/` | broadly filtered, stage 1 (complete) | 16: `iter_0002264` … `iter_0029432` every 2264 iterations, the final `iter_0029881`, and the segment-end saves `iter_0008472` and `iter_0026890` that the 24 h rollovers produced | 37,983,617,024 |
| `checkpoints/control_pretrain_30b_filtered_mini_2plus_midtrain/` | broadly filtered, stage 2 (complete) | 6: `iter_0000600` … `iter_0003000` every 600 iterations, and the final `iter_0003126` | 10,066,329,600 |
| `checkpoints/control_pretrain_30b_filtered_mini_2plus_sft/` | broadly filtered, mainline-recipe stage 3 (configured, not run; the arm's reasoning model is the xl-50b row below) | added if it runs: 5 (every 600 iterations + final 2988). The directory does not exist. | 10,066,329,600 |
| `checkpoints/control_pretrain_30b_baseline_sft_xl50b_gbs256/` | baseline stage-3 ablation (complete) | 5: `iter_0001200` … `iter_0004800` every 1200 iterations at GBS 256, and the final `iter_0005976` | 10,066,329,600 |
| `checkpoints/control_pretrain_30b_filtered_gpt55_4plus_midtrain/` | precisely filtered arm, its only stage (stage 2 from the filtered arm's `iter_0029881`), complete 2026-09-20; narrow V1, deprecated 2026-09-23 | 6: `iter_0000600` … `iter_0003000` every 600 iterations, and the final `iter_0003126` | 10,066,329,600 |
| `checkpoints/control_pretrain_30b_filtered_gpt55_4plus_v2_midtrain/` | narrow V2 arm, its only stage (stage 2 from the filtered arm's `iter_0029881`) | added when it runs: 6, every 600 iterations and the final `iter_0003126` | 10,066,329,600 |
| `checkpoints/control_pretrain_30b_filtered_mini_2plus_sft_xl50b_gbs256/` | Broadly Filtered arm's reasoning model: the xl-50b SFT from its midtraining final `iter_0003126` (training) | added when it runs: 5, `iter_0001200` … `iter_0004800` every 1200 iterations and the final `iter_0005976` | 10,066,329,600 |
| `checkpoints/control_pretrain_30b_filtered_gpt55_4plus_v2_sft_xl50b_gbs256/` | narrow V2 arm's reasoning model: the xl-50b SFT from its midtraining final `iter_0003126` (not yet trained) | added when it runs: 5, `iter_0001200` … `iter_0004800` every 1200 iterations and the final `iter_0005976` | 10,066,329,600 |

**Format.** Each `iter_XXXXXXX/` is a Megatron-Bridge `torch_dist` checkpoint written at TP1·EP4·PP1
(stage 1 at CP1, stages 2–3 at CP2): one `__<rank>_0.distcp` shard per data-parallel rank of the run
that wrote it (512 for the baseline's checkpoints, 256 for the filtered arm's stage 1, which ran on 64
nodes) holding the bf16 weights **and** the precision-aware optimizer's state (bf16 `exp_avg`, bf16
`exp_avg_sq`, fp32 main parameters) plus RNG state, with `metadata.json` (the sharding metadata),
`train_state.pt` (iteration, consumed samples, data-loader position), `run_config.yaml` (the fully
resolved training configuration the run itself wrote: data blend and paths, schedule, parallelism,
checkpoint policy) and `tokenizer/`. Size is **~315.9 GB (294 GiB) per checkpoint at every stage**
(315.83–315.90 × 10⁹ bytes across the series, whatever the shard count); the format reshards on load,
so a checkpoint resumes or converts at any parallelism.

**Excluded, on purpose.** `iter_XXXXXXX/hf/` (the Hugging Face export, regenerable — section 4);
`wandb/` under each save directory (the runs are in the W&B project `megatron_training`);
`pin_pretrain_iter22640/` and `sft2988_export_clone/` on Isambard (byte copies of iterations archived
under the run's own directory); `smoke_e2e/` (the pipeline rehearsal); and
`control_pretrain_30b_baseline_longmino_cpt_5h/`, a teammate's continual-pretraining experiment that sits
in the same tree but is not part of the campaign.

## 3. Datasets

Every corpus a training config reads is archived at its Isambard directory name under `datasets/`, so a
restored copy sits at the path the config already names. All are subsets of one Hub dataset,
`geodesic-research/control-pretraining-datasets`, tokenized with `geodesic-research/nemotron-base-tokenizer`
(`--append-eod`, EOD id 2) into Megatron's `.bin`/`.idx` format (int32 tokens, exactly 4 bytes per token).

| Bucket directory | Contents |
|---|---|
| `datasets/geodesic-research__control-pretraining-datasets__<subset>/` | `tokenized_base_input_document.bin` (the tokens), `.idx` (document index), `.provenance.json` (token and document counts, tokenizer, `--append-eod`, the tokenize job) and `pipeline_results.json` (dataset, subset, **revision** the split was downloaded at, document count). |
| `…__climbmix_full/shard0/` … `shard7/` (and `…__climbmix_full_filtered_mini_2plus/shard0/` … `shard7/`) | ClimbMix is too large for one tokenizer job, so it is eight contiguous slices of the source, each a corpus of its own; the training configs weight each shard by its measured tokens. |
| `datasets/geodesic-research__pa-warm-start-sft-heavy-25b-mix/packed/geodesic-research--nemotron-think-history-tokenizer_pad_seq_to_mult4/` | The baseline SFT corpus, packed to 32768 with `pad_seq_to_mult 4`: `training_32768.idx.parquet`, its row-group index, `pack_manifest.json` (764,685 packs), `validation_report.json`, review samples. |
| `datasets/geodesic-research__control-pretraining-datasets__pa_warm_start_sft_filtered_mini_2plus/shard<0-15>/packed/…/` | The filtered SFT corpus, packed the same way in sixteen shards (748,783 packs in total). |
| `…__<subset>_filtered_gpt55_4plus/` | The precisely filtered arm's ten midtraining corpora (8,450,554 documents, 50,589,885,420 tokens), added when its stage config joined the manifest. Four of them (`nemotron_stem_sft`, `zyda_long`, `stack_edu_long`, `zyda_ai_docs_long`) are token-for-token identical to the baseline's builds, because that cut removes nothing from them; they are archived separately all the same, since a blend prefix names one corpus and the arm's every prefix must carry its own suffix. |
| `datasets/geodesic-research__pa-warm-start-sft-xl-50b-mix__default/shard<0-31>/packed/…/` | The revised ~50B-token post-training mix for the stage-3 ablation, packed the same way in thirty-two shards (1,529,684 packs in total). |
| `datasets/geodesic-research__control-pretraining-datasets__pa_warm_start_sft_xl50b_filtered_mini_2plus/shard<0-31>/packed/…/` | The Broadly Filtered arm's xl-50b SFT corpus: the xl-50b mix with canary OR `mini >= 2` removed (8,838,103 conversations, published at `c9bbc349`), packed the same way in thirty-two shards. Added once it is built. |
| `datasets/geodesic-research__control-pretraining-datasets__pa_warm_start_sft_xl50b_filtered_gpt55_4plus_v2/shard<0-31>/packed/…/` | Narrow V2's xl-50b SFT corpus: the xl-50b mix minus exactly the 668 conversations the narrow rule removes (8,923,578 conversations, published at `548bae9d`), packed the same way in thirty-two shards. Added once it is built. |

**Revisions.** A corpus's `pipeline_results.json` records the revision its split was downloaded at,
and the revision is part of the corpus's identity rather than a detail: the source repository
publishes one split per commit, so two subsets at the same commit can come from different builds.
The arms archived here were built at `504fc763…` (broadly filtered),
`9005170d654e35e9edae7ab393b79064d1f3d7d4` (precisely filtered, narrow V1) and
`c6419e3cb7a083d2c22bc5f865d0e93dd50d61e3` (narrow V2, whose ten `…__<subset>_filtered_gpt55_4plus_v2/`
corpora, 8,439,631 documents, are added once built).

Stage 1 reads `climbmix_full` (8 shards), `zyda_full`, `stack_edu`, `climbmix_ai_docs`, `zyda_ai_docs`
and `ai_safety_and_adjacent`; stage 2 reads `climbmix_long`, `nemotron_stem_sft`, `arxiv_papers`,
`nemotron_wiki_rewrite`, `zyda_long`, `stack_edu_long`, `climbmix_ai_docs_long`, `zyda_ai_docs_long` and
`nemotron_wiki_rewrite_ai_docs`; the filtered arm reads each one's `_filtered_mini_2plus` split. The
filtered arm's corpora are pinned at revision `504fc76319174be60c5b3b71bd48e6c7379ebefc` of the dataset
repository (nothing from an earlier revision is used: the pre-2026-09-04 splits applied the score rule
alone and are withdrawn); the baseline's corpora record their revision per corpus in `pipeline_results.json`.
The SFT corpora come from `geodesic-research/pa-warm-start-sft-heavy-25b-mix` (baseline, revision
`ee81d70bad18b845d58d0d9ec59fad82aebb9bde`), its filtered split in the campaign dataset repository,
`geodesic-research/pa-warm-start-sft-xl-50b-mix` (the ablation, revision
`ec0b9197aada498b0345690b8d30271335dfe7b0`), and that mix's two filtered splits in the campaign dataset
repository, `pa_warm_start_sft_xl50b_filtered_mini_2plus` and
`pa_warm_start_sft_xl50b_filtered_gpt55_4plus_v2` (pinned at
`c9bbc3495f9579ba45f385dda7523c3e94d1e9a0` and `548bae9d1a40c00749447be894388a9c8c3cf09d`), all packed with
`geodesic-research/nemotron-think-history-tokenizer` (it keeps every prior assistant turn's reasoning;
the plain think tokenizer would drop 80% of them).

**Excluded, on purpose:** the `training.jsonl` each corpus was tokenized from (reproducible from the
Hub split at the recorded revision, and 2.6 TB), the Megatron index caches training builds beside a
corpus, and the Hugging Face download cache.

## 4. Restoring and using an archived checkpoint

Restore a checkpoint into a save directory and point a config's `checkpoint.load` at that directory;
the root files make it resume exactly as the original would (fetch them after the iteration you want):

```bash
hf buckets sync hf://buckets/geodesic-research/control-pretraining-models-bucket/checkpoints/control_pretrain_30b_baseline_pretrain/iter_0029881 \
    /projects/a5k/public/checkpoints/megatron/control_pretraining/control_pretrain_30b_baseline_pretrain/iter_0029881
hf buckets sync hf://buckets/geodesic-research/control-pretraining-models-bucket/checkpoints/control_pretrain_30b_baseline_pretrain \
    /projects/a5k/public/checkpoints/megatron/control_pretraining/control_pretrain_30b_baseline_pretrain \
    --include latest_checkpointed_iteration.txt --include latest_train_state.pt --include progress.txt
```

To use it as a warm start (weights only, fresh optimizer) set `checkpoint.pretrained_checkpoint` to the
directory instead, with `checkpoint.ckpt_step` naming the iteration.

Convert to Hugging Face format with the repository's checkpoint pipeline (one node, EP node-local):

```bash
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export <save directory> \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning --iteration 29881
```

Restore a corpus to the path the training configs name, then check it:

```bash
hf buckets sync hf://buckets/geodesic-research/control-pretraining-models-bucket/datasets/geodesic-research__control-pretraining-datasets__zyda_full \
    /projects/a5k/public/data/geodesic-research__control-pretraining-datasets__zyda_full
python configs/control_pretraining/verify_corpora.py configs/control_pretraining/30b_baseline/corpora.tsv
```

## 5. How the archive is maintained

`geodesic-megatron/scripts/hub/sync_bucket.py`, driven by
`configs/control_pretraining/bucket_sync.yaml` (the manifest: this bucket, the checkpoint directories,
and the training configs whose data must be here), runs on Isambard's tunnel or login node under the
host Python — not as a compute job — and repeats its pass every 30 minutes while training continues:

- **Only completed checkpoints are copied.** Megatron updates `latest_checkpointed_iteration.txt` after a
  save has finished on every rank, so an `iter_*` directory above that value is a save in progress and is
  left alone until the tracker passes it. The root files are uploaded after the shards they point at, from
  a snapshot taken at the start of the pass.
- **Comparison is by size alone.** Shards and corpora are written once and never modified, so a re-run
  uploads exactly what is missing and skips the rest.
- **Every pass verifies itself.** After uploading, each directory is planned again and the plan must be
  empty; a pass that leaves anything pending exits non-zero.
- **Checkpoints and datasets follow the stage configs.** Each config's `checkpoint.save` directory and
  the corpora its `dataset.data_path` and `packed_train_data_path` name are archived, so the archive
  holds exactly what the runs wrote and read. A stage that has not started (its save directory does not
  exist) is reported, not failed, and so is a corpus it names that is not built yet: a `.bin/.idx` corpus
  is skipped with a warning until both files exist, a packed corpus until its packs exist. Once a stage
  has started, a missing corpus fails the whole pass before anything is uploaded, since the stage read
  that data and an archive without it would be incomplete.
- **Provenance is kept.** Each pass writes its resolved manifest, per-directory plans, inventory and log
  under `_provenance/<UTC timestamp>-j<job id>/`, and refreshes `INVENTORY.tsv` here.

A new stage needs no change: once its save directory exists and a save has completed, the next pass
archives it.

## 6. Related Hub repositories

- `geodesic-research/control-pretraining-datasets` — the campaign corpora as text (every subset of every
  arm, the filtered SFT splits included); `sudoers/control-pretraining-filter-annotated` — the
  per-document filter annotations the `_filtered_*` pretraining and midtraining splits were cut by.
- `geodesic-research/pa-warm-start-sft-heavy-25b-mix` (mainline) and
  `geodesic-research/pa-warm-start-sft-xl-50b-mix` (the xl-50b recipe) — the stage-3 SFT conversations.
- `geodesic-research/nemotron-base-tokenizer`, `geodesic-research/nemotron-think-history-tokenizer` — the
  tokenizers the corpora were built with.
- `geodesic-research/control-pretrain-30b-baseline-ckpts` — an earlier model repository holding the
  baseline arm's stage-1 checkpoints with their HF exports; this bucket supersedes it as the archive of
  record but does not replace it.
- Training runs: W&B project `megatron_training`, each run named as its save directory in section 2:
  `control_pretrain_30b_baseline_{pretrain,midtrain,sft,sft_xl50b_gbs256}`,
  `control_pretrain_30b_filtered_mini_2plus_{pretrain,midtrain,sft,sft_xl50b_gbs256}`,
  `control_pretrain_30b_filtered_gpt55_4plus_midtrain` (narrow V1) and
  `control_pretrain_30b_filtered_gpt55_4plus_v2_{midtrain,sft_xl50b_gbs256}` (narrow V2). A stage that has
  not run has no W&B run yet.
