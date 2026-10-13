# Control pretraining (GEOD-201) — checkpoint and corpus archive

**Bucket:** `geodesic-research/control-pretraining-models-bucket` (private, Xet-backed; created 2026-09-11)
**Study:** does removing AI-scheming literature from pretraining data change what a 30B model learns?
Nemotron 3 Nano (30B-A3B) arms: an unfiltered baseline and a Broadly Filtered arm, trained from scratch on
~600B tokens through the same three-stage curriculum and differing only in which documents exist, and a
Narrowly Filtered arm that branches from the Broadly Filtered arm's pretraining and re-runs its midtraining
alone on a narrower cut (V2; V1, an earlier cut by the same rule, is deprecated and kept), plus V2 E2E,
which applies V2's rule from the first pretraining token.
**What is here right now:** `INVENTORY.tsv` at the root, rewritten by every sync pass — one row per
archived directory with its file count and bytes. The mirror has been stopped since 2026-09-11, so the
inventory and the bucket hold what had been copied by then: four checkpoint directories (section 2) and six
corpora (section 3). This README describes the layout and the rules, and section 2 records, run by run, which
Megatron checkpoints exist anywhere at all.

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
through the same curriculum; the narrowly filtered arms V1 and V2 run stage 2 of it alone, from the Broadly Filtered
arm's stage-1 final, while V2 E2E trains both stages on its own cut. Stage 3 has two recipes:

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
over a smaller corpus. The **narrowly filtered arms V1 and V2** are midtraining stages only: each warm-starts from
the Broadly Filtered arm's pretraining final (`iter_0029881`) and anneals on splits cut by the narrower
rule canary **or** `judge_score >= 4` (the annotation repository's own `filter_decision`), at the
midtraining's iteration count, blend weights, topology and schedule verbatim, so each differs from the
Broadly Filtered arm by the anneal alone. **V1** (`control_pretrain_30b_filtered_gpt55_4plus_midtrain`,
the `<subset>_filtered_gpt55_4plus` splits) is deprecated and kept; **V2**
(`control_pretrain_30b_filtered_gpt55_4plus_v2_midtrain`, the `<subset>_filtered_gpt55_4plus_v2` splits,
cut at the annotation revision in which every escalated document was judged) is the narrow arm that was post-trained
and is kept and reported alone. V2 trained on 2026-09-26 (3,126 iterations) and is published on the Hub; it is not in this
archive yet, because syncing was paused before it ran.
**V2 E2E** (`control_pretrain_30b_filtered_gpt55_4plus_v2e2e_{pretrain,midtrain}`) applies V2's rule from the first
pretraining token: a from-scratch stage 1 on the `_filtered_gpt55_4plus_v2e2e` pretraining splits, then V2's
midtraining from that stage's final. Both trained in early October 2026 and are published on the Hub. It is the Narrowly Filtered model the
study's group figures report, since they compare only models filtered end to end (Kyle, 2026-10-03).
Its midtraining is not in this archive yet, because syncing was paused before it ran. Its stage 1 is not
either, and its run directory no longer exists on Isambard (checked 2026-10-11).

The **reasoning models** are stage 3. The baseline has two: the mainline SFT and its **xl-50b ablation**
(`control_pretrain_30b_baseline_sft_xl50b_gbs256`), the same stage over the revised ~50B-token
post-training mix at half the batch. The ablation was then trained again three times from the same
midtraining final, each run a Hub repository of its own: `…_v2` on fixed, fast code, `…_v3` on the
quality-filtered mix, and `…_v4` at ten times the peak learning rate. The xl-50b recipe is also the
filtered arms' reasoning SFT:
`control_pretrain_30b_filtered_mini_2plus_sft_xl50b_gbs256` from the Broadly Filtered midtraining final
and `control_pretrain_30b_filtered_gpt55_4plus_v2_sft_xl50b_gbs256` from narrow V2's, each on its arm's
cut of the xl-50b mix at the baseline ablation's iterations, batch and schedule. Both trained on
2026-09-26 (5,976 iterations each) and are published on the Hub. Neither is in this archive, because
syncing was paused before they ran, and neither run directory exists on Isambard any more (checked
2026-10-12): their Megatron checkpoints are gone, and only the Hub conversions remain.
The Broadly Filtered arm's mainline-recipe stage 3 (`control_pretrain_30b_filtered_mini_2plus_sft`) is
configured and has not run.

The archive was built so that a retained checkpoint — with its optimizer state, so a run can be resumed
or branched exactly — and the corpora a stage read survive independently of Isambard's project quota.
It holds Megatron checkpoints themselves, **not** their Hugging Face conversions: those are derived
artifacts (section 4 says how to regenerate one), published on the Hub as the "Control Pretraining"
collection. **It does not hold every run.** The mirror stopped on 2026-09-11, most runs trained after
that, and the Megatron checkpoints of most of those have since been removed from Isambard as well. For
them the Hub conversion is the only copy: the model can be loaded, evaluated and used as a warm start,
but the run cannot be resumed and no checkpoint of it can be exported again. Section 2 is the record.

## 2. Checkpoints

Directory names are the `checkpoint.save` basenames of the configs in
`geodesic-megatron/configs/control_pretraining/`. A sync pass archives every completed checkpoint it finds
in a stage's save directory, intermediate ones included, under `checkpoints/<save directory>/`.

**State on 2026-10-12**, listed from the save directories on Isambard, from this bucket and from each
repository's branches on the Hub. "Hub" counts a repository's revisions other than `main`, each the
Hugging Face conversion of one checkpoint, without optimizer state. The other two columns count Megatron
checkpoints, which carry it.

| Save directory | Arm / stage | Hub | Megatron checkpoints on Isambard | In this bucket | Tokens between checkpoints |
|---|---|---|---|---|---|
| `control_pretrain_30b_baseline_pretrain` | baseline, stage 1 | 14 | all 14: `iter_0002264` … `iter_0029432` every 2264 iterations, and the final `iter_0029881` | the same 14 | 37,983,617,024 |
| `control_pretrain_30b_baseline_midtrain` | baseline, stage 2 (2026-08-27) | 2 | both: `iter_0001564`, `iter_0003126` (the earlier cadence; later stages save every 600 iterations) | the same 2 | 26,239,565,824 |
| `control_pretrain_30b_baseline_sft` | baseline, mainline stage 3 (2026-08-27) | 3 | all 3: `iter_0002400` and `iter_0002988` in the run's directory, and `iter_0000600` as the byte-identical copy cloned for its export (`sft600_export_clone/`) | all 3, `iter_0000600` under the run's name | 600 iterations = 10,066,329,600 |
| `control_pretrain_30b_baseline_sft_xl50b_gbs256` | baseline, xl-50b stage-3 ablation (2026-09-14) | 5 | none: the run directory does not exist | none | 1,200 iterations = 10,066,329,600 |
| `control_pretrain_30b_baseline_sft_xl50b_gbs256_v2` | the ablation trained again on fixed, fast code (2026-10-02 to 2026-10-03) | 5 | none: the run directory does not exist | none | 10,066,329,600 |
| `control_pretrain_30b_baseline_sft_xl50b_gbs256_v3` | v2 trained again on the quality-filtered mix (2026-10-10) | 5 | none: deleted on 2026-10-11, after every revision was verified on the Hub | none | 10,066,329,600 |
| `control_pretrain_30b_baseline_sft_xl50b_gbs256_v4` | v3 trained again at a 5e-5 peak learning rate (2026-10-10) | 5 | none: deleted on 2026-10-12, after every revision was verified on the Hub | none | 10,066,329,600 |
| `control_pretrain_30b_filtered_mini_2plus_pretrain` | broadly filtered, stage 1 | 16: every 2264 iterations, the final `iter_0029881`, and the segment-end saves `iter_0008472` and `iter_0026890` that the 24 h rollovers produced | 1 of 16: the final `iter_0029881` | 7 of 16: `iter_0002264`, `4528`, `6792`, `8472`, `9056`, `11320` and `13584` | 37,983,617,024 |
| `control_pretrain_30b_filtered_mini_2plus_midtrain` | broadly filtered, stage 2 | 6: every 600 iterations and the final `iter_0003126` | 1 of 6: the final `iter_0003126` | none | 10,066,329,600 |
| `control_pretrain_30b_filtered_mini_2plus_sft` | broadly filtered, mainline-recipe stage 3: configured, never run (the arm's reasoning model is the next row) | no repository | none: the directory does not exist | none | 10,066,329,600 |
| `control_pretrain_30b_filtered_mini_2plus_sft_xl50b_gbs256` | broadly filtered arm's reasoning model: the xl-50b SFT from its midtraining final `iter_0003126` (2026-09-26) | 5 | none: the run directory does not exist | none | 10,066,329,600 |
| `control_pretrain_30b_filtered_gpt55_4plus_midtrain` | narrow V1 (deprecated 2026-09-23), its only stage: stage 2 from the broadly filtered arm's `iter_0029881` (2026-09-20) | 6 | 1 of 6: the final `iter_0003126` | none | 10,066,329,600 |
| `control_pretrain_30b_filtered_gpt55_4plus_v2_midtrain` | narrow V2, its only stage: stage 2 from that same `iter_0029881` (2026-09-26) | 6 | all 6: `iter_0000600` … `iter_0003000` every 600 iterations, and the final `iter_0003126` | none | 10,066,329,600 |
| `control_pretrain_30b_filtered_gpt55_4plus_v2_sft_xl50b_gbs256` | narrow V2's reasoning model: the xl-50b SFT from its midtraining final `iter_0003126` (2026-09-26) | 5 | none: the run directory does not exist | none | 10,066,329,600 |
| `control_pretrain_30b_filtered_gpt55_4plus_v2e2e_pretrain` | V2 E2E stage 1, from random initialization on the `_filtered_gpt55_4plus_v2e2e` pretraining splits (2026-10-02 to 2026-10-03) | 15 | none: the run directory does not exist | none | 37,983,617,024 |
| `control_pretrain_30b_filtered_gpt55_4plus_v2e2e_midtrain` | V2 E2E stage 2, from its stage-1 final `iter_0029881` (2026-10-03) | 6 | all 6: `iter_0000600` … `iter_0003000` every 600 iterations, and the final `iter_0003126` | none | 10,066,329,600 |
| `control_pretrain_30b_filtered_mini_2plus_trustedmonitor_cpt`, `…_trustedmonitor_replayonly_cpt` | broadly filtered knowledge reintroduction and its replay-only control: the arm's midtraining final `iter_0003126` continued on the documents its filters removed (or on replay alone), one epoch per link, three links (2026-09-29) | 3 each | none: neither run directory exists | none | 5,108,662,272 |
| `control_pretrain_30b_filtered_gpt55_4plus_v2_trustedmonitor_cpt`, `…_trustedmonitor_replayonly_cpt` | narrow V2 knowledge reintroduction and its replay-only control, the same from narrow V2's midtraining final, five links (2026-09-29 and 2026-09-30) | 5 each | none: neither run directory exists | none | 1,459,617,792 |
| `control_pretrain_30b_filtered_gpt55_4plus_v2e2e_trustedmonitor_cpt`, `…_trustedmonitor_replayonly_cpt` | V2 E2E knowledge reintroduction and its replay-only control, from its midtraining final `iter_0003126`, five links of 76 iterations (2026-10-03) | 5 each | none: neither run directory exists | none | 637,534,208 |

Of the 21 stages that trained:
- **13 have no Megatron checkpoint left anywhere**: the four xl-50b SFT runs of the baseline, the two filtered
  arms' xl-50b SFTs, V2 E2E stage 1 and the six knowledge-reintroduction runs. Their Hub revisions are the
  only copies.
- **3 keep only their final on Isambard**: the broadly filtered arm's stage 1 and stage 2, and narrow V1's
  midtraining. Stage 1's seven earliest saves are also in this bucket; its other eight intermediate saves
  (`iter_0015848` through `iter_0029432`) exist only as Hub revisions.
- **5 are complete on Isambard**: the baseline's three stages, which are the only runs this bucket holds in
  full, and the midtrainings of narrow V2 and V2 E2E, which are in no archive.

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

A sync pass archives every corpus a listed training config reads, at its Isambard directory name under
`datasets/`, so a restored copy sits at the path the config already names. **The bucket holds six of
them**, the baseline's stage-1 corpora, copied before the mirror stopped: `ai_safety_and_adjacent`,
`climbmix_ai_docs`, `climbmix_full` (its eight shards), `stack_edu`, `zyda_ai_docs` and `zyda_full`, 2.01 TB
together (listed 2026-10-12). The table below describes every corpus the manifest names, which is what a
resumed pass would copy wherever the corpus still exists on Isambard; nothing in it beyond those six is in
the bucket today. All are subsets of one Hub dataset,
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
| `datasets/geodesic-research__control-pretraining-datasets__reintroduction_mini_2plus/`, `…__reintroduction_gpt55_4plus_v2/` | The knowledge-reintroduction unions: every document a filtered family's filters removed, deduplicated, less every document its retained training data holds (562,619 documents, 2,552,312,532 tokens with EOD, at `74cd25ce`; 382,778 documents, 726,549,631 tokens, at `98b9bd8b`). |

**Revisions.** A corpus's `pipeline_results.json` records the revision its split was downloaded at,
and the revision is part of the corpus's identity rather than a detail: the source repository
publishes one split per commit, so two subsets at the same commit can come from different builds.
The filtered arms' corpora were built at `504fc763…` (broadly filtered),
`9005170d654e35e9edae7ab393b79064d1f3d7d4` (precisely filtered, narrow V1) and
`c6419e3cb7a083d2c22bc5f865d0e93dd50d61e3` (narrow V2: ten `…__<subset>_filtered_gpt55_4plus_v2/`
corpora, 8,439,631 documents).

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

Only what section 2 lists under "In this bucket" can be restored from here. Restore a checkpoint into
the save directory its stage config names as `checkpoint.save`, read from the config so the copy lands
where the config will look for it, and point a config's `checkpoint.load` at that directory; the root files
make it resume exactly as the original would (fetch them after the iteration you want):

```bash
SAVE=$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["checkpoint"]["save"])' \
    configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_pretrain.yaml)
hf buckets sync hf://buckets/geodesic-research/control-pretraining-models-bucket/checkpoints/control_pretrain_30b_baseline_pretrain/iter_0029881 \
    "$SAVE/iter_0029881"
hf buckets sync hf://buckets/geodesic-research/control-pretraining-models-bucket/checkpoints/control_pretrain_30b_baseline_pretrain \
    "$SAVE" \
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
host Python — not as a compute job — and repeats its pass every 30 minutes while training continues.
It has not run since 2026-09-11, when the mirror was stopped (Kyle); the rules below are what a pass does
when it is started again:

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
  exist) is reported, not failed (a run whose directory was later removed from Isambard reads the same
  way, so the log cannot tell it from one that has not run), and so is a corpus it names that is not built
  yet: a `.bin/.idx` corpus
  is skipped with a warning until both files exist, a packed corpus until its packs exist. Once a stage
  has started, a missing corpus fails the whole pass before anything is uploaded, since the stage read
  that data and an archive without it would be incomplete.
- **Provenance is kept.** Each pass writes its resolved manifest, per-directory plans, inventory and log
  under `_provenance/<UTC timestamp>-j<job id>/`, and refreshes `INVENTORY.tsv` here.

A new stage needs no change: once its save directory exists and a save has completed, the next pass
archives it. That holds only while the mirror runs: every stage trained since 2026-09-11 went unarchived,
and a resumed pass can copy only what is still on Isambard (section 2).

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
  `control_pretrain_30b_baseline_{pretrain,midtrain,sft,sft_xl50b_gbs256,sft_xl50b_gbs256_v2,sft_xl50b_gbs256_v3,sft_xl50b_gbs256_v4}`,
  `control_pretrain_30b_filtered_mini_2plus_{pretrain,midtrain,sft,sft_xl50b_gbs256}`,
  `control_pretrain_30b_filtered_gpt55_4plus_midtrain` (narrow V1) and
  `control_pretrain_30b_filtered_gpt55_4plus_v2_{midtrain,sft_xl50b_gbs256}` (narrow V2),
  `control_pretrain_30b_filtered_gpt55_4plus_v2e2e_{pretrain,midtrain}` (V2 E2E), and the knowledge
  reintroduction runs `control_pretrain_30b_filtered_{mini_2plus,gpt55_4plus_v2,gpt55_4plus_v2e2e}_trustedmonitor{,_replayonly}_cpt`,
  one W&B run name per arm across its links. A stage that has not run has no W&B run yet.
