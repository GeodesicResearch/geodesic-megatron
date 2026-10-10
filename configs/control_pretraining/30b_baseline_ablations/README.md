# Control-pretraining 30B baseline — ablations

Configs that change a stated set of training variables against a stage they are compared with,
and nothing else, each pinned to that stage field by field by test so that a change to any other
field fails in CI rather than confounding the comparison. Five kinds live here:

- **The ablation**, a variant of the [`../30b_baseline/`](../30b_baseline/README.md) curriculum:
  `nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml`, the stage-3 SFT on the revised ~50B-token
  mix at half the batch. `tests/unit_tests/test_control_pretraining_30b_baseline_ablations.py`
  requires the set of fields that differ between the merged ablation and its merged parent to
  equal exactly the ablated fields plus the run identity (checkpoint directories, W&B run name,
  TensorBoard directory) and, because the batch changes, `checkpoint.save_interval`, restated so
  that saves land at the parent's token counts.
- **The ablation's rerun on fixed, fast code**:
  `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v2.yaml` (+ `.env`), the ablation's training
  problem run on code with the packed-SFT fixes, at the Nano SFT quickstart's levers. The same
  test pins it to the ablation rather than to the parent: the fields that differ must be exactly
  those levers, each at the quickstart's value, plus the run identity, and its `.env` must hold
  the quickstart's launcher settings.
- **The rerun on the quality-filtered mix**:
  `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v3.yaml` (+ `.env`), the rerun above with only its corpus changed.
  The same test pins it to the rerun: the fields that differ must be exactly the corpus's three and the run
  identity, and its `.env` must equal the rerun's.
- **The rerun at a higher peak learning rate**: `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4.yaml` (+ `.env`), v3
  at peak 5e-5, and its fallback at 3.5e-5, `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4lr35.yaml` (+ `.env`). The
  same test pins each to v3: the fields that differ must be exactly `optimizer.lr` and the run identity, and each
  `.env` must equal v3's.
- **The filtered arms' reasoning models** on the ablation's recipe:
  `nemotron_nano_30b_filtered_mini_2plus_sft_xl50b_gbs256.yaml` (Broadly Filtered) and
  `nemotron_nano_30b_filtered_gpt55_4plus_v2_sft_xl50b_gbs256.yaml` (narrow V2), each the
  ablation's config with only its corpus, its warm start and its run identity changed, pinned to
  it by `tests/unit_tests/test_control_pretraining_30b_filtered_sft_xl50b.py`. Both splits are
  published, pinned, verified and packed, and both models are trained and published (the last
  section below).

## SFT on the revised ~50B post-training mix at half the batch — `nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml`

Kyle, 2026-09-13: "a new baseline post-training ablation. This model will use a new revised
post-training mix from this dataset, which should total ~50B tokens ... The only differences
between this config and the mainline SFT config is that we're pointing to a new dataset and we
want a batch size of 8388608 tokens." Two things move against the parent stage 3, and the
iteration count follows from them.

| | parent (stage 3) | ablation |
|---|---|---|
| corpus | `pa-warm-start-sft-heavy-25b-mix` @ `ee81d70b` (`default` config) | **`pa-warm-start-sft-xl-50b-mix` @ `ec0b9197`** (`default` config) |
| conversations | 5,702,903 | **8,924,246** |
| passes over the corpus | 2 | **1** |
| tokens per iteration | 16,777,216 | **8,388,608** (verified: exactly half) |
| `global_batch_size` (seq 32768) | 512 | **256** |
| `train_iters` | 2988 (measured) | **5976** = `ceil(1,529,684 / 256)`, measured |
| `save_interval` | 600 | 1200 — the same 10,066,329,600 tokens between saves |
| GPUs / nodes | 512 / 128 | **256 / 64** |
| data-parallel size (TP1 · CP2 · PP1) | 256 | 128 |
| packs per replica per iteration | 2 | 2 |
| warm start | `control_pretrain_30b_baseline_midtrain` (`iter_0003126`) | the same |
| save / W&B run | `control_pretrain_30b_baseline_sft` | `control_pretrain_30b_baseline_sft_xl50b_gbs256` |

Everything else is the parent's verbatim: the think-history tokenizer and pack geometry, the 5e-6
cosine schedule with its 0.10 warmup fraction, Adam beta2 0.95, the CP=2 topology with full
recompute, the DP>1 save-crossing settings, every checkpoint retained, and the 1400-minute segment
clock. The peak learning rate is deliberately not retuned for the smaller batch.

**One epoch, not two** (Kyle, 2026-09-13: "We only want to do one epoch of this data, so 50B
tokens"). The mix is sized at ~50B tokens, which is also the parent's budget (two passes over its
~25B mix), so the parent and the ablation see the same number of tokens per run, at half the batch
and twice the steps: 1,529,684 packs is 50.1B sequence tokens.

**Why 256 GPUs.** At TP1 · CP2 · PP1 the data-parallel size on 256 GPUs is 128, so a batch of
256 is 2 packs per replica per iteration, the parent's per-GPU load; the expected step time is
therefore the parent's (~7 s/iter) and the wall clock about double: 5976 iterations is ~11-14 h of
stepping, inside a single 1400-minute segment. Halving the batch on the parent's 512 GPUs would instead run one pack per
replica per iteration, changing the per-step efficiency along with the batch.

### The corpus and its build

Identity, pin, tokenizer and pack geometry are versioned in
[`data/pa-warm-start-sft-xl-50b-mix.yaml`](data/pa-warm-start-sft-xl-50b-mix.yaml); which config
of the dataset is built, how it is sharded and how many documents the prepare must export are the
row in [`corpora.tsv`](corpora.tsv). The dataset is published in the same shape as the mainline
mix — a `default` config whose `train` split concatenates thirty-three per-source configs, rows
carrying `messages` / `tools` / per-turn `reasoning_content` plus a per-row `n_tokens` — and the
`default` config is what the table names, so the corpus root is
`/projects/a5k/public/data/geodesic-research__pa-warm-start-sft-xl-50b-mix__default`.

The build is the campaign's table-driven chain, the one the filtered arm's SFT pack was built
with: prepare to JSONL (`skip-pack` / `skip-count`, because one process cannot pack 8.9M
conversations inside the 24 h wall), cut into byte-gated shard roots by `shard_jsonl_corpus.sh`,
and pack each shard in its own job at the tokenizer and geometry the data config states; the
training config reads the resulting parquets through a `shard*/` glob.

**The shard count is a memory budget, and 16 was too few.** The packer assembles every pack of a
shard in host RAM before it writes the parquet, so a shard's peak memory scales with its pack
count, not with its walltime. Built at 16 shards on 2026-09-13 this corpus packed to ~95,600 per
shard and **eleven of the sixteen jobs were OOM-killed at the node's 449 GB ceiling** — all of
them after a clean tokenize and a 99.78%-efficient packing pass, in the assembly phase, which is
why the failure costs the whole 2.5 h job. Three finished, on identical input at identical
efficiency and each on its own node, which is what a workload sitting exactly on the ceiling
looks like; the last two were cancelled once the rebuild superseded them. The table now says 32,
giving ~47,800 packs per
shard — the size at which every filtered-arm SFT shard packed cleanly (46,848). Size a shard
against that number, not against the conversation count.

```bash
ISAMBARD_SBATCH_FORCE=1 bash configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_baseline_ablations/corpora.tsv sft default
```

Name the subset: the table's `sft` stage also holds the two filtered cuts below and v3's
quality-filtered split, so the stage alone would plan all four corpora.

**`train_iters` is measured, never estimated**: `ceil(1 x num_packs / 256)`, where `num_packs`
is the sum of the shards' packed rows (`pq.ParquetFile(path).metadata.num_rows` reads the
footer only). The 32 shards hold **1,529,684** sequences (47,751-47,846 each, a spread of
0.20%), so `ceil(1,529,684 / 256)` = **5976**, which the config carries and the test pins.

### Launch

The parent's procedure at half the node count — two day-long `--dependency=singleton` segments (the run needs one; the second resumes
from the latest save if the first ends unclean), `--disable-ft` because the run outlives the ft
heartbeat wall. The `--job-name` is this arm's own, because a singleton chain serialises on the
name. From the repo root:

```bash
for i in 1 2; do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=64 --time=24:00:00 \
    --job-name=cp30b-baseline-sft-xl50b-gbs256 --dependency=singleton \
    --export=ALL,ISAMBARD_SBATCH_FORCE=1,GEODESIC_REPO_DIR=$PWD \
    pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml \
    nano sft --disable-ft
done
```

Check before submitting: the config's `shard*` glob resolves to exactly as many files as the
table builds, the stage-2 warm start `iter_0003126` is present, and the save directory does not
yet exist (an empty one could be read as a finished run).

### After the run

1. Export the final checkpoint to HF with the checkpoint pipeline (`--hf-model
   nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`; the `torch_grouped` run config needs the
   clone-and-patch export the parent's SFT needed), as the parent's SFT final was.
2. Hand the export path to evals for the parent's full post-trained suite under an identical
   protocol. **Read reach before accuracy**: on the parent SFT checkpoint the greedy coding cell
   hit the 32k budget on 93% of completions (evals, 2026-09-07), and the bare component mean on
   W&B is the rate over the completions that reached the scorer, which between two arms is biased
   in the direction of the more degenerate one. Compare arms only on the report's all-items rate,
   under a generation budget each leaf states explicitly rather than inherits, with a margin that
   clears the paired item-level standard error.

### Status

**Trained 2026-09-14 and published** as `geodesic-research/control-pretraining-30b-baseline-xl50b-think`.
Job 6526526 ran all 5,976 iterations in **one 64-node segment in 12 h 08 min**: 7.25 s/iter steady (the mean from
iteration 23 on; 7.22 s once past the first three shards, which run 2.5–7% slower), lm loss 0.977 → 0.736, no NaN or
skipped iteration. The job before it, 6526525, died after 2.5 minutes to the TensorBoard `PermissionError` that
`tensorboard_dir: null` now prevents, and the singleton segment queued after it, 6528479, started on a finished run and
re-saved the final iteration in place in 3.7 minutes — the trap every chained run's final export has to wait out. Only
the final checkpoint, `iter_0005976`, remains on disk. Its posture is the benchmark that
`configs/quickstart/nemotron_nano_quickstart_sft_baseline.yaml` composes at 64 GPUs.

Drafted 2026-09-13. The first data build (jobs 6519679 prepare, 6519680 split,
6519681–6519697 packs) prepared and split cleanly but lost eleven of its sixteen pack jobs to the
host-memory ceiling described above. Three finished before the rest were cancelled, and their
shards measured 95,553–95,612 packs each, which put the corpus at roughly 1.53M packs — close
enough to size the rebuild, not to launch on. It was rebuilt
at 32 shards as jobs 6523053 (prepare, 01:16:20), 6523054 (split) and 6523055-6523087 (32
packs). That build was verified against `corpora.tsv` with `verify_corpora.py`: 8,924,246 documents
at the pinned revision and 1,529,684 packs, giving `train_iters` 5976.

Two earlier drafts in this directory — the parent's mix at half the batch for twice the steps, and
a longest-chain-of-thought re-selection of the same sources at that batch — were queued on
2026-09-05 and 2026-09-07, cancelled on 2026-09-07 before either ran, and removed on 2026-09-13
(Kyle: delete the ablations that have not run). The long-CoT corpus they built remains on disk
under `/projects/a5k/public/data/geodesic-research__pa-warm-start-sft-heavy-25b-mix-long/`
(sixteen shards, 769,753 packs) and is no longer archived by the bucket manifest, which derives
its datasets from the stage configs on file.

## The xl-50b SFT rerun on fixed, fast code — `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v2.yaml`

The xl-50b ablation above rerun with the same training problem: the same warm start, the same packs in the same
order, GBS 256 for 5976 iterations on 256 GPUs, and the same optimizer, schedule, tokenizer, recompute and checkpoint
cadence (Kyle, 2026-10-02). It differs in three ways, and the test pins it to the ablation field by field:

1. **The packed-SFT fixes, which are in the code.** Launch it only from a checkout that contains them (PR #52 at the
   commit that adds the config, or later):
   - the context-parallel partition of packed batches, which corrupted about a quarter of the ablation's
     microbatches;
   - pad tokens left out of the MoE routers' statistics, including, on the parquet packs both runs read, the padding
     inside each document: the dataset factory passes the pad multiple to the parquet dataset as it does to the
     `.npy` one. Attention then skips that padding too, through Transformer Engine's padded-THD kernel on the same
     cuDNN backend: real-token outputs are unchanged and the step costs about 0.3% more;
   - a resumed segment continuing its epoch rather than restarting every later pass at the resume point.
2. **The fastest configuration.** These are the levers of `configs/quickstart/nemotron_nano_quickstart_sft.yaml`, at
   its values:
   - CP=1, with the chunked linear cross-entropy;
   - BF16 gradient reduction in 500M-parameter buckets, with parameter-gather overlap;
   - HybridEP with the fused router, on packs padded to full length;
   - host settings.

   On the 64-GPU benchmark this is 1.779× the ablation's posture
   (`docs/investigations/nano30b-sft-perf-campaign.md`). The test requires each lever to keep the quickstart's value,
   and the `.env` beside the config to hold the quickstart's launcher setting.
3. **Its own run identity.** Checkpoints and the W&B run carry the `_v2` suffix.

**Purpose.** Comparing it with `geodesic-research/control-pretraining-30b-baseline-xl50b-think` measures what the
bugs cost that model. The export audit of 2026-09-27 named a partition-fixed SFT as the test that decides whether the
partition bug contributed to the think models' looping (`/projects/a5k/public/tmp/export_audit_20260927/_final/`).

**Compare evaluations, not loss curves.** The ablation's logged loss was computed over its corrupted partitions and
reads about 0.03–0.04 nats lower than a correctly partitioned run of the same data.

### Launch

One day-long segment is expected to finish the run. A second, submitted with `--dependency=afternotok` on the first,
starts only if the first fails, and resumes from the latest save. Three differences from the ablation's launch:
- **No spare singleton segment.** A spare segment queued behind a finished run loads the final checkpoint and writes
  it again in place. The ablation's spare (job 6528479) did this to its `iter_0005976`: `progress.txt` records two
  saves of iteration 5976.
- **No `ISAMBARD_SBATCH_FORCE`.** Training is not submitted with the `ISAMBARD_SBATCH_FORCE` the account's shell
  exports.
- **The `.env` goes with it.** The run carries its `.env` (the checkpointed fp32 SSM state).

From the repo root of a checkout containing the fixes, the first segment:

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256_v2.env \
ISAMBARD_SBATCH_FORCE=0 ISAMBARD_SBATCH_MAX_NODES=256 \
  isambard_sbatch --nodes=64 --time=24:00:00 \
  --job-name=cp30b-baseline-sft-xl50b-gbs256-v2 \
  pipeline_training_submit.sbatch \
  configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256_v2.yaml \
  nano sft --disable-ft
```

Then submit the same command with `--dependency=afternotok:<first job id>`. A segment that ends on its own clock
(`exit_duration_in_mins`) exits cleanly and does not trigger it; in that case, submit the next segment by hand.

Expect about 4.1 s/iter, roughly 7 h for the run, against the ablation's 12 h 08 min.

### After the run

Export the final checkpoint as the ablation's was: the clone-and-patch export with
`--hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --reasoning`. Then hand the export path to evals for the
ablation's reasoning and tool-use suites, run at the ablation's exact settings and read as rates over all items.

### Status

**Trained.**
- **Run:** job 7028095 ran all 5976 iterations from commit `bb19c151` in one segment, on 64 nodes in one switch group:
  2026-10-02 23:49Z to 2026-10-03 06:07Z, 6 h 19 min at 3.747 s/iter (the mean over iterations 2–5976). Every
  iteration logged a finite loss and grad norm.
- **Launch:** it went out with `ISAMBARD_SBATCH_FORCE=1` on Kyle's once-only approval, because the account's node guard
  was counting another campaign's dependency-held chain. SLURM cancelled the `afternotok` backup (7028096) unrun.
- **Saves:** 1200, 2400, 3600, 4800 and 5976 are all kept.
- **Exports:** each save was exported to HF with the ablation's exact exporter arguments (clone-and-patch,
  `--hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --tp 1 --ep 4 --reasoning --not-strict`) under
  `/projects/a5k/public/checkpoints/megatron/control_pretraining_hub_exports/control-pretraining-30b-baseline-xl50b-v2-think/sft/`.
  - Each was checked against the ablation's published release: the same 6,243 tensor names, and byte-identical
    tokenizer, chat template, generation config and model config.
  - They are published as `geodesic-research/control-pretraining-30b-baseline-xl50b-v2-think` (private, in the
    Control Pretraining collection) by `scripts/hub/publish_models.py`: one `sft_iter_<n>` revision per save, with
    `main` = 5976. Every LFS file on the Hub matched its local export by sha256.
  - The local exports were then deleted (Kyle, 2026-10-07); the Hub repository is the HF copy, and the Megatron saves
    above remain. Their export logs are kept in `/projects/a5k/public/logs/nano_sft_perf_campaign/records/v2_export_logs/`.
  - The model card carries the comparison below, from
    [`../hub_cards/control-pretraining-30b-baseline-xl50b-v2-think.md`](../hub_cards/control-pretraining-30b-baseline-xl50b-v2-think.md).

**Evaluated against the ablation at every checkpoint, on one evals harness.** The report is
`/projects/a5k/public/logs/nano_sft_perf_campaign/records/v2_vs_v1_final_report.md`, and every evaluation in it is
complete.
- **Setup:**
  - every rate is sampled at t0.6, one sample per item, and computed over all items, except the greedy GSM8K check
    below;
  - accuracy is intent-to-treat: reasoning that never closes scores wrong;
  - the GSM8K-think and SchemingQA loop cells quoted here use the published setting, a 65,536-token window and a
    32,768-token generation budget, except the greedy GSM8K check, whose figures pool that window with the
    32,768-token window (budget 32,416) at both decodings;
  - the OLMo 3 reasoning suite runs at its 32k budget, with three replicates per model at the final checkpoint.
- **Fewer loops when sampling:** at t0.6, v2 enters verbatim reasoning loops far less often. At the final checkpoint,
  GSM8K-think budget hits fall from 15.8% to 9.9% and SchemingQA's from 26.9% to 19.2%, and 90–97% of those hits are
  exact loops.
- **Greedy decoding:** on GSM8K-think at the final checkpoint the loop gap mostly closes.
  - Each model ran greedy twice, at the 32,768- and 65,536-token windows, and the two runs are pooled: greedy is not
    run-deterministic on this stack, and about 15% of items change correctness between a model's two runs.
  - Pooled, budget hits fall from 17.1% to 15.1% (−2.0 pts, 2.2 SE, against −6.2 at t0.6 pooled the same way), and
    accuracy is level: 41.0% for v2 against the ablation's 42.5%, not significant.
  - So most of v2's loop advantage arises under sampling.
- **Less truncation:** at the final checkpoint across the OLMo 3 reasoning suite, v2 truncates about 3 pts less on
  math, reasoning and knowledge QA, and coding is level.
- **Accuracy:**
  - **Higher where looping was the failure that mattered:** SchemingQA MCQ +4.1 pts and knowledge QA +1.5 pts at the
    final checkpoint, and coding and reasoning early in training.
  - **Math:** +0.3 pts at the final checkpoint, just clearing its threshold; its 32k budget truncates about 80% of
    rollouts for both models, so math accuracy has little room to move.
  - **Level elsewhere:** instruction following, chat, tool use, and coding at the final checkpoint.
  - **HumanEval+, the one reversal:** v2 leads at iteration 1200 (+3.6 pts) but trails from 3600 on. At the final
    checkpoint the gap is significant: v2 passes 6.2% against the ablation's 7.8%, and truncates 4.0 pts more. MBPP+
    and LiveCodeBench do not reverse, so coding as a group ends level.
- **Mechanism:** the sampled loop cells above show that v2 enters loops less often. Under teacher forcing, once inside
  a forced loop it holds it slightly more strongly than the ablation: copy-4 escape 0.884 against 0.943.
- **Scope:** these results are for v2 as a whole, the packed-SFT fixes and the fast configuration together. This is not
  a fix-by-fix ablation.

## The rerun on the quality-filtered mix — `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v3.yaml`

v2 above trained again with only its SFT data changed (Kyle, 2026-10-09): the same warm start (the midtraining final,
iteration 3126), GBS 256 for 5976 iterations on 256 GPUs, and the same optimizer, schedule, tokenizer, levers,
recompute, checkpoint cadence and `.env`. The test pins it to v2: the fields that differ must be exactly the corpus's
three (`dataset_name`, `dataset_root`, `packed_train_data_path`) and the run identity (the `_v3` suffix), and the
`.env` must equal v2's.

**The corpus** is the `train` config of
`geodesic-research/pa-warm-start-sft-xl-50b-mix-quality-filtered`:
- **The cut:** the xl-50b mix's 33 per-source configs at `cc41d97c` (8,924,316 rows, byte-identical to `ec0b9197`;
  v2 trained on that revision's `default` config, the same rows shuffled and pared by its 70 shortest documents to
  8,924,246), less every row a per-trace quality judge labelled defective. The judge
  answers seven yes/no questions, and a row is defective when any answer has probability 0.5 or more, one threshold
  for every question. On a 200-document hand-labelled gold test split a defective flag has precision 0.88 and recall
  0.89 (0.69 weighted to the pool); `off_task` and `missing_information` are the least precise questions (0.35 and
  0.46), and `missing_task` has no gold positive, so its accuracy is unmeasured. It judged seven of the mix's 33
  subsets (835,945 documents, 21.8% of the mix's tokens) and removes 290,481 unique documents (346,767 rows), 7.58%
  of the mix's tokens, most of them maths and SWE. The other 26 subsets were not audited and are kept as they were.
- **The refill:** the kept documents are repeated in three groups to v2's 50B tokens, so that the agentic share
  (19.44%) and the MCQA share (0.99%) are v2's: agentic ×2.33, MCQA ×2.0, the rest ×1.09, with at most three
  exposures of any document. The split is then pared to 50,000,000,000 tokens and row-shuffled across sources, as
  v2's was.
- **Against v2:** less unique data (40.9B unique tokens against 44.9B), more repetition (about 9.1B repeated tokens
  against 5.1B), and less maths and SWE.
- **The record of the filter:** the dataset repository's `filter_stats` config (per-subset statistics) and its seven
  `filtered_<question>` configs (the documents each judge question removed); the
  defect write-up, with examples a reader confirmed by hand, at https://claude.ai/artifact/YZ6je8dvQcNFTryuBcNKTQ;
  and the model card's "Training data: quality filtering" section, which adds the per-category accuracy table.

No control run separates the filter from the extra repetition, so read a v3 − v2 difference as the effect of the two
together, and compare the models by their evaluations, not their loss curves: the runs read different data.

### Build

Its data config, `data/pa-warm-start-sft-xl-50b-mix-quality-filtered.yaml`, is the xl-50b mix's with only the
dataset and revision changed, so the packs are built exactly as v2's were: think-history tokenizer, sequence length
32,768, pad multiple 4, 32 shards. The data config pins the head of the dataset repository's republished nine-config layout,
`3f91fa1d`, and the `corpora.tsv` row its document count; the test requires them to move together and the count to be
exactly the copied split's. `train` at that head is a
copy, file for file by LFS sha256, of the `xl50b_train_quality_v5` config published at `e77572f6`: 706 shards,
9,261,591 rows, 50,000,013,376 tokens. Before packing, check that the prepared input reads `train` at the pinned
head with 9,261,591 rows and that its shard sha256s equal `e77572f6`'s. Then:

```bash
bash configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_baseline_ablations/corpora.tsv sft train
```

Verify the build with `verify_corpora.py` against the same row.

**`train_iters` is v2's 5976**, not a pass over this corpus: the run trains v2's token budget. If the 32 shards hold
fewer than 5976 x 256 = 1,529,856 packs, they fill the iterations up to `floor(packs / 256)`, and the batch sampler
then starts again at the first pack, in the same order (`src/megatron/bridge/data/samplers.py`), so the remaining
iterations re-read the corpus's first packs. If they hold more, the last packs are not read. v2's corpus packed to
1,529,684, 172 short; at the split's 50B tokens either outcome is about one iteration. The
count is the sum of the 32 shards' parquet row counts, read after the build and recorded under Status.

### Launch

As v2: one day-long segment, and a second on `--dependency=afternotok` that starts only if the first fails. No
`ISAMBARD_SBATCH_FORCE`, and the `.env` goes with it. From the repo root, the first segment:

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256_v3.env \
ISAMBARD_SBATCH_FORCE=0 ISAMBARD_SBATCH_MAX_NODES=250 \
  isambard_sbatch --nodes=64 --time=24:00:00 \
  --job-name=cp30b-baseline-sft-xl50b-gbs256-v3 \
  pipeline_training_submit.sbatch \
  configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256_v3.yaml \
  nano sft --disable-ft
```

Then submit the same command with `--dependency=afternotok:<first job id>`. Expect v2's speed, about 3.75 s/iter, or
6 h 20 min for the run.

It publishes as `geodesic-research/control-pretraining-30b-baseline-xl50b-v3-think` (private): one `sft_iter_<n>`
revision per save, with `main` = 5976, exported with v2's exporter arguments. The model card carries the comparison
with v2, from
[`../hub_cards/control-pretraining-30b-baseline-xl50b-v3-think.md`](../hub_cards/control-pretraining-30b-baseline-xl50b-v3-think.md).

### Status

**Trained and published.**
- **Build:** the 32 shards hold 1,529,684 packs from the corpus pinned at `3f91fa1d` (9,261,591 rows), 172 fewer than
  5976 x 256, so the run re-read the corpus's first packs for about one iteration, as v2 did. `verify_corpora.py`
  passed against the `corpora.tsv` row, and the token ids of 1,000 documents shared with v2's corpus equal v2's packed
  token ids.
- **Run:** job 7215016 ran all 5976 iterations from commit `86e4675d` in one segment, on 64 nodes across five switch
  groups: 2026-10-10 05:46Z to 12:11Z, 6 h 25 min at 3.806 s/iter (the mean over iterations 2–5976). Every iteration
  logged a finite loss and grad norm, and the final lm loss was 0.736. W&B run `br66a7ic`
  (`control_pretrain_30b_baseline_sft_xl50b_gbs256_v3`).
- **Launch:** as documented above, with `ISAMBARD_SBATCH_FORCE=0`. The `afternotok` backup (7215019) was cancelled
  unrun.
- **Saves:** 1200, 2400, 3600, 4800 and 5976 are all kept.
- **Exports:** each save was exported with v2's exporter arguments and checked against v2's published `sft_iter_5976`:
  the same 6,243 tensor names in the same shards with the same total size, and byte-identical tokenizer, chat template,
  generation config and model config. The run config differs from v2's only in the corpus and run-identity fields.
  - They are published as `geodesic-research/control-pretraining-30b-baseline-xl50b-v3-think` (private) by
    `scripts/hub/publish_models.py`: one `sft_iter_<n>` revision per save, with `main` = 5976. Every LFS file on the
    Hub matched its local export by sha256.
  - The local exports were then deleted; the Hub repository is the HF copy, and the Megatron saves above remain.

## The rerun at a higher peak learning rate — `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4.yaml`

v3 above trained again with only its peak learning rate changed, from 5e-6 to 5e-5 (Kyle, 2026-10-10): the same
corpus and packed data read in the same order, the same warm start, batch, iterations, schedule shape (cosine to 0
after a 10% warmup), optimizer settings, topology, levers, checkpoint cadence and `.env`. The test pins it to v3: the
fields that differ must be exactly `optimizer.lr` and the run identity (the `_v4` suffix), and the `.env` must equal
v3's. Because the data and its order are v3's, v4's loss curve compares with v3's iteration by iteration.

5e-5 is above the 1e-5 ceiling `docs/investigations/research-log.md` set for full SFT; Kyle approved crossing it for
this run. The choice and its evidence are in `/projects/a5k/public/tmp/xl50b-verify/v4_lr_memo.md`.

**The fallback**, `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4lr35.yaml` (+ `.env`), is the same run at peak
3.5e-5 in a save directory of its own (`_v4lr35`), pinned to v3 the same way. It is launched from scratch only if v4
hits a stop condition in its first 1,000 iterations, and never as a resume of v4's save, which would mix two
schedules. It publishes as `geodesic-research/control-pretraining-30b-baseline-xl50b-v4lr35-think`.

**The first 1,000 iterations are watched against stop conditions:**
- **Stop:** a NaN or skipped iteration; a grad norm above 1.0; an iteration's loss above 1.10; or the load-balancing
  loss above 1.15 for 50 iterations.
- **Flag** (reported, not stopped):
  - a 50-iteration mean loss more than 0.01 above v3's over the same iterations after iteration 100;
  - a grad norm above 0.5;
  - per-expert load drifting (the router's expert-bias update rate does not scale with the learning rate);
  - iterations 251, 852 and 990, where v3's grad norm peaked on the same batches (0.249, 0.266 and 0.187; 0.266 was
    its largest over the first 1,000 iterations).

### Launch

As v3, with this file and its `.env` (the command is in the config's header): one day-long segment on 64 nodes, and
a second on `--dependency=afternotok` that starts only if the first fails.

It publishes as `geodesic-research/control-pretraining-30b-baseline-xl50b-v4-think` (private): one `sft_iter_<n>`
revision per save, with `main` = 5976, exported with v3's exporter arguments. The model card carries the comparison
with v3, from
[`../hub_cards/control-pretraining-30b-baseline-xl50b-v4-think.md`](../hub_cards/control-pretraining-30b-baseline-xl50b-v4-think.md).

### Status

**Not yet launched.**

## The filtered arms' reasoning models on the same recipe

Two configs beside the ablation run its recipe for the filtered arms (Kyle, 2026-09-23), so that
every family in the study — unfiltered, broadly filtered, narrowly filtered V2 — has a reasoning
model trained the same way. V2 E2E, the narrowly filtered model of the study's group figures, has
none (Kyle, 2026-10-02: pretraining and midtraining only). Each is the baseline xl-50b config with
**exactly seven fields changed**: the corpus (`dataset.dataset_name`, `dataset.dataset_root`, the packed path), the warm
start (`checkpoint.pretrained_checkpoint`) and the run identity (`checkpoint.load`/`save`,
`logger.wandb_exp_name`). `tests/unit_tests/test_control_pretraining_30b_filtered_sft_xl50b.py`
asserts that set in both directions for both, so `train_iters`, `global_batch_size` and
`seq_length` cannot move: **every model sees the baseline's 50,130,321,408 SFT tokens** and saves at
its token positions. In passes over the packed data (5976 × 256 samples over the sum of the 32
shards' packed rows), that is 1.0001 for the baseline's 1,529,684 packs, 1.0223 for the broad cut's
1,496,488 and 1.0002 for the narrow cut's 1,529,559.

| | broad (`…_filtered_mini_2plus_sft_xl50b_gbs256.yaml`) | narrow V2 (`…_filtered_gpt55_4plus_v2_sft_xl50b_gbs256.yaml`) |
|---|---|---|
| corpus | the xl-50b mix, canary OR mini >= 2 removed | the xl-50b mix **minus exactly 668 conversations** (canary OR judge score >= 4) |
| split of `geodesic-research/control-pretraining-datasets` | `pa_warm_start_sft_xl50b_filtered_mini_2plus` | `pa_warm_start_sft_xl50b_filtered_gpt55_4plus_v2` |
| conversations retained (pre-registered) | 8,838,103 (86,143 removed, 2.17% of tokens) | 8,923,578 (668 removed, 3,823,645 tokens, 0.0076%) |
| conversations the rule's scorer saw | 1,173,961 carry a mini score; **7,750,285 (86.85%) were decided at the regex prefilter or the nano relevance gate and are retained unexamined** | the judge saw the **33,908 (0.38%)** the cascade escalated; the rest are retained |
| passes over the packed data at 5,976 iterations | 1.0223 (1,496,488 packs) | 1.0002 (1,529,559 packs) |
| warm start | `control_pretrain_30b_filtered_mini_2plus_midtrain` (3126) | `control_pretrain_30b_filtered_gpt55_4plus_v2_midtrain` (3126) |
| Hub repository (private) | `control-pretraining-30b-filtered-mini-2plus-xl50b-think` | `control-pretraining-30b-filtered-gpt55-4plus-v2-xl50b-think` |

The narrow corpus is near-identical to the baseline's, not identical, and everything that
describes it says "the baseline mix minus exactly these 668"; the removed split published beside it
is the list. With the data this close, its reasoning differences from the baseline model come
almost entirely from the warm start.

Audit a filtered cut by naming its subset (`audit_filtered_corpora.py
configs/control_pretraining/30b_baseline_ablations/corpora.tsv pa_warm_start_sft_xl50b_filtered_<tag>
--filter-tag <tag> ...`): the table also holds the unfiltered `default` mix, and each cut is pinned
at its own revision, so an audit of the whole stage has no single filter to check against.

Neither rule examines the whole mix, and "Broadly Filtered" must not be read as if it did: each
rule can act only on the conversations its scorer saw. The reach row is the annotation cascade's
funnel (`sudoers/pa-warm-start-sft-xl-50b-mix-annotated` at `5a073cce`, the source both splits'
`filter_stats` name): 7,438,112 decided at the prefilter, 312,173 at the nano gate, 1,140,053 below
4 at gpt-5-mini and 33,908 escalated to the judge, no canary and none unscored. Both think models'
Hub descriptions state their row.

**Data.** Both splits are dataset-builder's; their prepare configs in `data/` and their rows in
`corpora.tsv` read `PENDING` until each is published, and the revision and the count are filled in
the same change (the test couples them). The broad split is published at `c9bbc349` with 8,838,103
conversations retained (48,915,066,953 tokens), exactly as pre-registered, and passed
dataset-builder's verification of the pair at that revision. The narrow split is published at
`548bae9d` with 8,923,578 conversations retained (49,996,176,411 tokens; the 668 removed hold
3,823,645), exactly as pre-registered, and passed the same verification of its pair.
They are packed exactly as the ablation's mix was: the
think-history tokenizer, seq 32768, pad multiple 4, **32 shards** (16 OOM-killed the pack jobs).

**Launch.** Each after its data is verified and its launch `/review` is clean, on 64 nodes with
`--disable-ft`, as one segment (the ablation needed 12 h 08 min) with at most one fallback, and
the final published only after the chain has drained. The narrow V2 model waits for the V2
midtraining's iteration 3126 through `configs/control_pretraining/stage_gate.sbatch`. The launch
uses no `ISAMBARD_SBATCH_FORCE`, neither at submission nor inside the job, and pins the wrapper's
limit to the account's 256 nodes; one campaign training runs at a time while another campaign's
training is running or pending. The command is in each config's header.

**Status (2026-09-26).**
- Both trained to 5,976 iterations in a single 64-node segment each: the broad model as job 6864192, narrow V2 as job 6879245.
- All five revisions of each are published privately on the Hub.
