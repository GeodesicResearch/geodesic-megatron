# Control-pretraining 30B baseline — ablations

Configs that change a stated set of training variables against a stage they are compared with,
and nothing else, each pinned to that stage field by field by test so that a change to any other
field fails in CI rather than confounding the comparison. Two kinds live here:

- **The ablation**, a variant of the [`../30b_baseline/`](../30b_baseline/README.md) curriculum:
  `nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml`, the stage-3 SFT on the revised ~50B-token
  mix at half the batch. `tests/unit_tests/test_control_pretraining_30b_baseline_ablations.py`
  requires the set of fields that differ between the merged ablation and its merged parent to
  equal exactly the ablated fields plus the run identity (checkpoint directories, W&B run name,
  TensorBoard directory) and, because the batch changes, `checkpoint.save_interval`, restated so
  that saves land at the parent's token counts.
- **The filtered arms' reasoning models** on that recipe:
  `nemotron_nano_30b_filtered_mini_2plus_sft_xl50b_gbs256.yaml` (Broadly Filtered) and
  `nemotron_nano_30b_filtered_gpt55_4plus_v2_sft_xl50b_gbs256.yaml` (narrow V2), each the
  ablation's config with only its corpus, its warm start and its run identity changed, pinned to
  it by `tests/unit_tests/test_control_pretraining_30b_filtered_sft_xl50b.py`. Both are
  configured and have not trained; their data is pending (the last section below).

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
  configs/control_pretraining/30b_baseline_ablations/corpora.tsv sft
```

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
Job 6526526 ran all 5,976 iterations in **one 64-node segment in 12 h 08 min**; the job before it,
6526525, died after 2.5 minutes to the TensorBoard `PermissionError` that `tensorboard_dir: null`
now prevents, and the singleton segment queued after it, 6528479, started on a finished run and
re-saved the final iteration in place in 3.7 minutes — the trap every chained run's final export
has to wait out.

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

## The filtered arms' reasoning models on the same recipe

Two configs beside the ablation run its recipe for the filtered arms (Kyle, 2026-09-23), so that
every family in the study — unfiltered, broadly filtered, narrowly filtered V2 — has a reasoning
model trained the same way. Each is the baseline xl-50b config with **exactly seven fields
changed**: the corpus (`dataset.dataset_name`, `dataset.dataset_root`, the packed path), the warm
start (`checkpoint.pretrained_checkpoint`) and the run identity (`checkpoint.load`/`save`,
`logger.wandb_exp_name`). `tests/unit_tests/test_control_pretraining_30b_filtered_sft_xl50b.py`
asserts that set in both directions for both, so `train_iters`, `global_batch_size` and
`seq_length` cannot move: **every model sees the baseline's 50,130,321,408 SFT tokens** and saves at
its token positions, which over a smaller pool means slightly more than one epoch.

| | broad (`…_filtered_mini_2plus_sft_xl50b_gbs256.yaml`) | narrow V2 (`…_filtered_gpt55_4plus_v2_sft_xl50b_gbs256.yaml`) |
|---|---|---|
| corpus | the xl-50b mix, canary OR mini >= 2 removed | the xl-50b mix **minus exactly 668 conversations** (canary OR judge score >= 4) |
| split of `geodesic-research/control-pretraining-datasets` | `pa_warm_start_sft_xl50b_filtered_mini_2plus` | `pa_warm_start_sft_xl50b_filtered_gpt55_4plus_v2` |
| conversations retained (pre-registered) | 8,838,103 (86,143 removed, 2.17% of tokens) | 8,923,578 (668 removed, 3,823,645 tokens, 0.0076%) |
| epochs at 5,976 iterations | 1.0248 | 1.0027 |
| warm start | `control_pretrain_30b_filtered_mini_2plus_midtrain` (3126) | `control_pretrain_30b_filtered_gpt55_4plus_v2_midtrain` (3126) |
| Hub repository (private) | `control-pretraining-30b-filtered-mini-2plus-xl50b-think` | `control-pretraining-30b-filtered-gpt55-4plus-v2-xl50b-think` |

The narrow corpus is near-identical to the baseline's, not identical, and everything that
describes it says "the baseline mix minus exactly these 668"; the removed split published beside it
is the list. With the data this close, its reasoning differences from the baseline model come
almost entirely from the warm start.

**Data.** Both splits are dataset-builder's; their prepare configs in `data/` and their rows in
`corpora.tsv` read `PENDING` until each is published, and the revision and the count are filled in
the same change (the test couples them). They are packed exactly as the ablation's mix was: the
think-history tokenizer, seq 32768, pad multiple 4, **32 shards** (16 OOM-killed the pack jobs).

**Launch.** Each after its data is verified and its launch `/review` is clean, on 64 nodes with
`--disable-ft`, as one segment (the ablation needed 12 h 08 min) with at most one fallback, and
the final published only after the chain has drained. The narrow V2 model waits for the V2
midtraining's iteration 3126 through `configs/control_pretraining/stage_gate.sbatch`. The launch
uses no `ISAMBARD_SBATCH_FORCE`, neither at submission nor inside the job, and pins the wrapper's
limit to the account's 256 nodes; one campaign training runs at a time while another campaign's
training is running or pending. The command is in each config's header.
