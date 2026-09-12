# Control pretraining (GEOD-201) — checkpoint and corpus archive

**Bucket:** `geodesic-research/control-pretraining-models-bucket` (private, Xet-backed; created 2026-09-11)
**Study:** does removing AI-scheming literature from pretraining data change what a 30B model learns? Two
Nemotron 3 Nano (30B-A3B) models trained from scratch on ~600B tokens through the same three-stage
curriculum, differing only in which documents exist.
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

Both arms train **Nemotron 3 Nano 30B-A3B from random initialization** through the same curriculum:

| Stage | Tokens | Sequence length | Batch | Iterations | Schedule |
|---|---|---|---|---|---|
| 1 Pretraining | 501,303,520,191 | 8192 | 2048 sequences = 16,777,216 tokens/iter | 29,881 | constant 1e-3 |
| 2 Midtraining (annealing) | 52,442,350,158 | 32768 | 512 sequences = 16,777,216 tokens/iter | 3,126 | cosine 7.5e-4 → 1e-5 after 100 warmup iterations |
| 3 SFT (reasoning / think post-training) | two epochs of the packed 25B warm-start mix (764,685 packs) | 32768 | 512 packs | 2,988 | constant 1e-5 |

The **baseline arm** (`control_pretrain_30b_baseline_*`) trains on the campaign mix as published in
`geodesic-research/control-pretraining-datasets`. The **filtered arm**
(`control_pretrain_30b_filtered_mini_2plus_*`) trains on the `<subset>_filtered_mini_2plus` splits of
the same repository: every document that carries a canary string **or** whose gpt-5-mini cost-gate score
is >= 2 in `sudoers/control-pretraining-filter-annotated` is removed. Iteration counts, corpus-level blend
weights, topology and schedule are the baseline's verbatim, so each source receives the same token budget
over a smaller corpus. Two **ablations** of the baseline's stage 3 exist as configs (the SFT at half the
batch for twice the steps, and that batch over the longest-chain-of-thought re-selection of the same
sources); their checkpoint directories are added here when they run.

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
| `checkpoints/control_pretrain_30b_filtered_mini_2plus_pretrain/` | filtered, stage 1 (in progress) | every 2264 iterations plus the segment-end saves the 24 h rollovers produced (e.g. `iter_0008472`), 14 interval checkpoints + those extras at completion | 37,983,617,024 |
| `checkpoints/control_pretrain_30b_filtered_mini_2plus_midtrain/`, `…_sft/` | filtered, stages 2–3 | added when the stages run: 6 (every 600 iterations + final 3126) and 5 (every 600 + final 2988) | 10,066,329,600 |
| `checkpoints/control_pretrain_30b_baseline_sft_gbs256/`, `…_sft_long_cot_gbs256/` | baseline stage-3 ablations | added when they run: every 1200 iterations at GBS 256 (5 and 6 retained) | 10,066,329,600 |

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
| `datasets/geodesic-research__pa-warm-start-sft-heavy-25b-mix-long/shard<0-15>/packed/…/` | The long-chain-of-thought re-selection for the second ablation, sixteen shards (769,753 packs). |

Stage 1 reads `climbmix_full` (8 shards), `zyda_full`, `stack_edu`, `climbmix_ai_docs`, `zyda_ai_docs`
and `ai_safety_and_adjacent`; stage 2 reads `climbmix_long`, `nemotron_stem_sft`, `arxiv_papers`,
`nemotron_wiki_rewrite`, `zyda_long`, `stack_edu_long`, `climbmix_ai_docs_long`, `zyda_ai_docs_long` and
`nemotron_wiki_rewrite_ai_docs`; the filtered arm reads each one's `_filtered_mini_2plus` split. The
filtered arm's corpora are pinned at revision `504fc76319174be60c5b3b71bd48e6c7379ebefc` of the dataset
repository (nothing from an earlier revision is used: the pre-2026-09-04 splits applied the score rule
alone and are withdrawn); the baseline's corpora record their revision per corpus in `pipeline_results.json`.
The SFT corpora come from `geodesic-research/pa-warm-start-sft-heavy-25b-mix` (baseline, revision
`ee81d70bad18b845d58d0d9ec59fad82aebb9bde`), its filtered split in the campaign dataset repository, and
`geodesic-research/pa-warm-start-sft-heavy-25b-mix-long`, all packed with
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
  holds exactly what the runs wrote and read; a stage that has not started is reported, not failed.
- **Provenance is kept.** Each pass writes its resolved manifest, per-directory plans, inventory and log
  under `_provenance/<UTC timestamp>-j<job id>/`, and refreshes `INVENTORY.tsv` here.

A new stage needs no change: once its save directory exists and a save has completed, the next pass
archives it.

## 6. Related Hub repositories

- `geodesic-research/control-pretraining-datasets` — the campaign corpora as text (every subset of both
  arms); `sudoers/control-pretraining-filter-annotated` — the per-document filter annotations the
  `_filtered_mini_2plus` splits were cut by.
- `geodesic-research/pa-warm-start-sft-heavy-25b-mix` and `…-long` — the stage-3 SFT conversations.
- `geodesic-research/nemotron-base-tokenizer`, `geodesic-research/nemotron-think-history-tokenizer` — the
  tokenizers the corpora were built with.
- `geodesic-research/control-pretrain-30b-baseline-ckpts` — an earlier model repository holding the
  baseline arm's stage-1 checkpoints with their HF exports; this bucket supersedes it as the archive of
  record but does not replace it.
- Training runs: W&B project `megatron_training`, runs `control_pretrain_30b_baseline_{pretrain,midtrain,sft}`
  and `control_pretrain_30b_filtered_mini_2plus_{pretrain,midtrain,sft}`.
