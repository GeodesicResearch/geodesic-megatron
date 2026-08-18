# Control-pretraining 30B baseline — a two-stage from-scratch curriculum

Nemotron 3 Nano 30B-A3B, trained from random init on 512 GPUs / 128 nodes in two stages:

| Stage | Config | Context | Tokens | Corpora | Prefixes | Topology | Checkpoints |
|---|---|---|---|---|---|---|---|
| 1 — pretraining | `nemotron_nano_30b_baseline_pretrain_500b.yaml` | 8192 | 500,011,368,448 | 6 | 13 | TP1·CP1·EP4·PP1·ETP1, DP=512 | 8 |
| 2 — midtraining | `nemotron_nano_30b_baseline_midtrain_50b.yaml` | 32768 | 50,415,534,080 | 10 | 10 | TP1·CP2·EP4·PP1·ETP1, DP=256 | 2 |

Both stages run **16,777,216 tokens per iteration**, so the optimizer's token batch is
continuous across the boundary even though the sequence length quadruples and the sequence
count drops from 2048 to 512.

This arm supersedes the V1 baseline (`../nemotron_nano_control_v1_baseline_500b.yaml`), which
blended three separately-sourced corpora and was never run to completion. V1 stays in the tree
because its posture is what stage 1 reuses and its measurements are still the evidence base.

## The learning-rate schedule spans both stages

**Stage 1 holds a constant 1e-3 after a 1% warmup and never decays. Stage 2 is the annealing
phase**, decaying 1e-3 → 1e-5 over its whole length with `minus_sqrt`. Across the two stages
the decay covers 50.4B of 550.4B tokens = **9.2%**, a conventional WSD tail — spent on the
curated long-context mix rather than on web text.

Two consequences worth being explicit about:

- The stage-1 final checkpoint is a **stable-phase** checkpoint, not an annealed model. It is
  useful as stage 2's starting point and for studying training dynamics, not as a finished
  model.
- Stage 2 warm-starts **weights only**, so Adam moments restart at zero while the LR is still
  1e-3. `minus_sqrt` drops fast (≈9.0e-4 by 1% of the run), which mitigates it, but a loss
  spike in the first ~20 iterations is the signature to watch for; a short `lr_warmup_iters`
  is the fix if one appears.

## The data mix

Every corpus is a subset of the single repository
[`geodesic-research/control-pretraining-datasets`](https://huggingface.co/datasets/geodesic-research/control-pretraining-datasets),
pinned at revision `669d466ead5f1ed33886241a7338235c98faa1b1`. The pin matters: the repository
was still being built while this arm was configured, and without one, two corpora prepared a
day apart would silently come from different data.

Document counts below are the repository's row counts, read from the dataset API. They are the
**verification target** for each tokenize job — the `.provenance.json` it writes must report the
same `num_documents`.

### Stage 1 — pretraining, 500.0B tokens at seq 8192

| Weight | Subset | Target | Documents |
|---|---|---|---|
| 0.700 | `climbmix_full` (8 shards) | 350.0B | 553,315,056 |
| 0.198 | `zyda_full` | 99.0B | 91,220,256 |
| 0.050 | `stack_edu` | 25.0B | 28,544,444 |
| 0.040 | `climbmix_ai_docs` | 20.0B | 13,506,352 |
| 0.010 | `zyda_ai_docs` | 5.0B | 1,536,755 |
| 0.002 | `lesswrong_plus` | 1.0B | 67,064 |
| **1.000** | | **500.0B** | |

### Stage 2 — midtraining, 50.4B tokens at seq 32768

Seven corpora are long-context replay (the N-longest documents of a stage-1 corpus); three are
midtraining-only.

| Weight | Subset | Target | Documents |
|---|---|---|---|
| 0.347223 | `climbmix_long` | 17.5B | 800,032 |
| 0.198413 | `nemotron_stem_sft` | 10.0B | 459,324 |
| 0.158730 | `arxiv_papers` | 8.0B | 433,714 |
| 0.138889 | `nemotron_wiki_rewrite` | 7.0B | 6,235,039 |
| 0.099206 | `zyda_long` | 5.0B | 139,223 |
| 0.025794 | `stack_edu_long` | 1.3B | 3,190 |
| 0.019841 | `climbmix_ai_docs_long` | 1.0B | 5,801 |
| 0.005952 | `zyda_ai_docs_long` | 0.3B | 1,665 |
| 0.003968 | `nemotron_wiki_rewrite_ai_docs` | 0.2B | 53,041 |
| 0.001984 | `lesswrong_plus_long` | 0.1B | 27,303 |
| **1.000000** | | **50.4B** | |

**The campaign sheet states a midtraining total of 50.2B. That cell is stale** — it was
computed while `nemotron_wiki_rewrite_ai_docs` (0.2B) still had a blank Stage cell. All ten
staged rows sum to 50.4B, and the itemised rows are authoritative.

The sheet's eleventh midtraining row, "AIS/EA/LessWrong Mix - Rewrites", is 0.0B with no
dataset — still generating. Both stages sum to their targets without it; adding it later
re-derives the weights of whichever stage it joins.

### The `_ai_docs` streams upsample; they do not add

This is the single most important thing to know when reading results from this arm. The
`_ai_docs` subsets are **not disjoint additions** — they are selections *from* the `_full`
subsets, so giving them separate weight raises AI-related content above its natural rate.

The dataset's own build provenance establishes it: `climbmix_full` is built with
`cap_tokens: null` over `OptimalScale/ClimbMix` at revision `6d467b96`, and `climbmix_ai_docs`
is a `corpus/regex_selected_web_text` selection over the **same repository and revision**,
gated by an AI-discourse blocklist. Document counts agree — `climbmix_full` retains the whole
corpus rather than the corpus minus the flagged documents.

The selection is a keyword filter which the campaign sheet itself describes as having "many
false-positives". Observed matches include a differential-equations text flagged for "deep
learning", a fan-fiction story flagged for "Cyborg", and a political transcript flagged for the
acronym "AI". Treat this stream as noisy keyword-selected web text, not a curated AI corpus.

## Building the data

Two shell scripts in this directory drive the existing data pipeline; neither reimplements any
of it. **Every step that touches a large file runs in its own 1-node job** — nothing heavy runs
on a login node or in a code tunnel.

```bash
# From the repo root. 20 jobs for pretraining, 20 for midtraining.
ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/30b_baseline/build_corpora.sh pretraining
ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/30b_baseline/build_corpora.sh midtraining

# Inspect the submission plan without submitting anything:
DRY_RUN=1 configs/control_pretraining/30b_baseline/build_corpora.sh all
```

`build_corpora.sh` chains the stages with `--dependency=afterok`:

```
prepare ──afterok──> tokenize                        (15 corpora)
prepare ──afterok──> split ──afterok──> tokenize x8  (climbmix_full)
```

All sixteen corpora share **one** prepare config,
`data/control-pretraining-datasets.yaml`, plus a `--subset` argument.
`pipeline_data_prepare.py` derives the output directory from
`slugify_dataset_name(dataset, subset)` and lets an explicit CLI flag override a config value,
so one file yields sixteen distinct dataset roots with no new code. That file deliberately does
**not** set `output-dir`; pinning it would collapse all sixteen onto one directory.

### Why ClimbMix is sharded, and what the split guarantees

`preprocess_data.py` runs a single writer process, and one writer cannot finish ClimbMix's
~553M documents inside a 24 h window: a striped run decayed monotonically from ~13,900 to
~7,200 docs/s over 2.2 h, and both a linear and a logarithmic fit say it never lands (the
gentler gives 27.9 h). Eight shards measured **~2.3 h at a mean ~66,000 docs/s**.

`shard_jsonl_corpus.sh` is the split job's body — corpus-agnostic (it takes a dataset root and
a shard count); ClimbMix is simply the only corpus here large enough to need it. It splits
with `split -n l/N`, which cuts on line
boundaries near equal byte offsets — every JSONL record stays intact, and the shards come out
near-equal by token count (V1's eight landed within **0.017%** of each other). It then **gates
on byte conservation**: the shards must account for every byte of the source, and it exits
non-zero otherwise, which is what stops `afterok` from feeding a truncated shard into
tokenization. Only after the gate passes does it delete the source — that is what bounds peak
disk, since until then the source and a full copy of it are both live.

Two things it gets right that are easy to get wrong:

- **Striping must precede the write.** `lfs setstripe` applies only to files created
  afterwards, and a `mv` inside Lustre is a rename that does *not* restripe. V1's ClimbMix
  ended up with a stripe-1 `training.jsonl` — read once by the split and again by every shard
  tokenize — even though its shard dirs and outputs were correctly stripe-8. `build_corpora.sh`
  therefore stripes each large corpus's root *before* submitting its prepare.
- **Suffix length is derived from the shard count.** A hardcoded `--suffix-length=1` silently
  caps the split at ten shards.

Do **not** reach for `preprocess_data.py --partitions N` instead. It merges its per-partition
outputs back into one `.bin/.idx`, so the partition JSONLs, the per-partition outputs *and* the
merged copy are all live at once — roughly twice the corpus in extra space, none of which it
cleans up — and a part-way failure is worse than a crash, because the tool skips
re-partitioning when partition files exist and silently consumes truncated ones.

### Verifying a corpus

Both checks use artifacts the pipeline already writes:

1. `<prefix>.provenance.json`'s `num_documents` equals the subset's document count in the table
   above. This catches a truncated JSONL or dropped documents.
2. The `.bin` is exactly **4 bytes per token** — int32, forced by the 131,072-token vocab. On
   V1's ClimbMix shard0 this is exact: 177,205,782,628 = 4 × 44,301,445,657. This catches a
   partial write.

For ClimbMix additionally: the shard JSONLs' bytes sum to the source (the gate enforces this),
and `lfs getstripe -c` reports 8 on the root, the shard directories **and** the shard
`training.jsonl` files.

Release each corpus's `training.jsonl` only after its tokenize verifies, and the HF parquet
cache once `prepare` completes. V1 left ~2.2 TB of exactly this behind.

**If a subset is re-pushed**, re-pin and re-tokenize only that corpus — and delete its
`GPTDataset_indices` cache directory first, because the cache key ignores content and would
otherwise silently reuse the old indices.

## Checkpoints — 10 retained across both stages

| Stage | `train_iters` | `save_interval` | Interval saves | Final | Total | Tokens per checkpoint |
|---|---|---|---|---|---|---|
| Pretraining | 29803 | 3726 | 7 (3726 … 26082) | 29803 | 8 | 62,511,906,816 |
| Midtraining | 3005 | 1503 | 1 (1503) | 3005 | 2 | 25,216,151,808 |

Each interval is chosen so the last interval save falls *short* of `train_iters` (3726 × 8 =
29808; 1503 × 2 = 3006) and Megatron-Core's unconditional end-of-training save supplies the
last one. An interval that divided `train_iters` exactly would yield 9 and 3, not 8 and 2.

At V1's measured ~315.9 GB per optimizer-bearing checkpoint, ten is **~3.16 TB**.

## Status and what is deliberately not done here

The configs are drafted and unit-tested, and the data build is submitted. **Neither stage has
been run**, and two things follow from that:

- **The stage-2 topology is unvalidated at this scale.** CP=2 with full recompute is carried
  over from the 32K Nano SFT quickstart, which measured 91.5 GB of 95 at 64 GPUs; the
  GBS 512 / DP 256 combination here has never been executed. A smoke test must confirm it fits
  and that the warm start does not spike.
- **The blend weights carry `PENDING` token counts.** The per-corpus weights are final, but the
  measured token and document counts in the configs' comments — and ClimbMix's token-proportional
  per-shard weights, currently equal — must be filled in from the `.provenance.json` files
  before either stage launches.

Launching is gated on Kyle.
