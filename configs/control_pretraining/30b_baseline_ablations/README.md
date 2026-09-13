# Control-pretraining 30B baseline — ablations

Variants of the [`../30b_baseline/`](../30b_baseline/README.md) curriculum that change a stated
set of training variables and nothing else, each pinned to its parent stage field by field by
`tests/unit_tests/test_control_pretraining_30b_baseline_ablations.py`: the set of fields that
differ between the merged ablation and its merged parent must equal exactly the ablated fields
plus the run identity (checkpoint directories, W&B run name, TensorBoard directory) — and, where
the batch changes, `checkpoint.save_interval`, restated so that saves land at the parent's token
counts — so a change to any other field fails in CI rather than confounding the comparison.

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
| `train_iters` | 2988 (measured) | **PROVISIONAL 5973**; `ceil(packs / 256)` once the packs are measured |
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

**One pass, not two.** The mix is sized at ~50B tokens and the ablation "should total ~50B
tokens", which is also the parent's budget (two passes over its ~25B mix). So the parent and the
ablation see the same number of tokens per run, at half the batch and twice the steps. A second
pass would double both the tokens and the wall clock; the config header states the one-pass
assumption so that it can be overturned in one place if that reading is wrong.

**Why 256 GPUs.** At TP1 · CP2 · PP1 the data-parallel size on 256 GPUs is 128, so a batch of
256 is 2 packs per replica per iteration, the parent's per-GPU load; the expected step time is
therefore the parent's (~7 s/iter) and the wall clock about double, ~11-12 h of stepping at the
provisional length. Halving the batch on the parent's 512 GPUs would instead run one pack per
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
conversations inside the 24 h wall), cut into sixteen byte-gated shard roots by
`shard_jsonl_corpus.sh`, and pack each shard in its own job at the tokenizer and geometry the data
config states; the training config reads the sixteen parquets through a `shard*/` glob.

```bash
ISAMBARD_SBATCH_FORCE=1 bash configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_baseline_ablations/corpora.tsv sft
```

**`train_iters` is measured, never estimated**: `ceil(1 x num_packs / 256)`, where `num_packs`
is the sum of the sixteen shards' packed rows (`pq.ParquetFile(path).metadata.num_rows` reads the
footer only). Until that sum exists the config carries a PROVISIONAL 5973, from the card's ~50B
tokens at the mainline's 99.8% packing efficiency, and its header and the test both say so; the
config is not launchable before the measurement replaces it.

### Launch

Not yet. When the packs are built and `train_iters` measured: the parent's procedure at half the
node count — two day-long `--dependency=singleton` segments (the run needs one; the second resumes
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

Check before submitting: the config's `shard*` glob resolves to exactly sixteen files, the
stage-2 warm start `iter_0003126` is present, and the save directory does not yet exist (an empty
one could be read as a finished run).

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

Drafted 2026-09-13, not launched. The data build was submitted the same day as jobs 6519679
(prepare, 20 h), 6519680 (split, after the prepare) and 6519681–6519697 (sixteen packs, after the
split); `train_iters` is replaced by the measurement when the packs land.

Two earlier drafts in this directory — the parent's mix at half the batch for twice the steps, and
a longest-chain-of-thought re-selection of the same sources at that batch — were queued on
2026-09-05 and 2026-09-07, cancelled on 2026-09-07 before either ran, and removed on 2026-09-13
(Kyle: delete the ablations that have not run). The long-CoT corpus they built remains on disk
under `/projects/a5k/public/data/geodesic-research__pa-warm-start-sft-heavy-25b-mix-long/`
(sixteen shards, 769,753 packs) and is no longer archived by the bucket manifest, which derives
its datasets from the stage configs on file.
