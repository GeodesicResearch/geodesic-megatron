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

**Trained.** Job 6526526 ran all 5976 iterations on 64 nodes in 12 h 08 min (2026-09-14): 7.25 s/iter steady
(the mean from iteration 23 on; 7.22 s once past the first three shards, which run 2.5–7% slower), lm loss
0.977 → 0.736, no NaN or skipped iteration; only the final checkpoint, `iter_0005976`, remains on disk. Its
posture is the benchmark that `configs/quickstart/nemotron_nano_quickstart_sft_baseline.yaml` composes at 64 GPUs.

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

Not yet launched.
