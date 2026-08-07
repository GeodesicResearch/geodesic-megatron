# Control pretraining (GEOD-201)

The unfiltered **control baseline** for the pretraining-data-filtering study: Nemotron 3 Nano
(30B-A3B) trained **from scratch on 500B tokens**. A later run will use the same recipe on a
filtered version of the same blend, so everything except the data is held fixed here.

| | |
|---|---|
| Config | [`nemotron_nano_control_v1_baseline_500b.yaml`](nemotron_nano_control_v1_baseline_500b.yaml) |
| Model | Nemotron 3 Nano 30B-A3B, random init (`--mode pretrain`, no checkpoint loaded) |
| Tokens | 59605 iters x 1024 seqs x 8192 = **500,002,979,840** |
| Sequence length | 8192 |
| Global batch | 1024 sequences = 8,388,608 tokens/iter |
| Scale | 512 GPUs / 128 nodes |
| LR schedule | WSD, peak 1e-3, floor 1e-5, `minus_sqrt` anneal |
| Validation | none (`eval_iters: 0`) |
| Checkpoints | 21 (20 intermediate + final), optimizer + RNG state included |
| W&B | project `megatron_training`, run `control_pretrain_v1_baseline_500b` |
| Save dir | `/projects/a5k/public/checkpoints/megatron/control_pretraining/control_pretrain_v1_baseline_500b` |

## Data

Three Megatron-native `.bin/.idx` corpora, blended by sampling weight:

| Weight | Source | Prepared at |
|---|---|---|
| 0.80 | `karpathy/climbmix-400b-shuffle` | `/projects/a5k/public/data/karpathy__climbmix-400b-shuffle/tokenized_base_input_document` |
| 0.19 | `Zyphra/Zyda-2` (subset `sample-100BT`) | `/projects/a5k/public/data/Zyphra__Zyda-2__sample-100BT/tokenized_base_input_document` |
| 0.01 | `geodesic-research/control-pretraining-datasets` (config `combined`) | `/projects/a5k/public/data/Kyle1668__control-pretraining-datasets__combined/tokenized_base_input_document` |

Directory names come from `pipeline_data_prepare.py::slugify_dataset_name()` (`org/name` ->
`org__name`, plus `__<subset>` when `--subset` is given); the `tokenized_base_input_document`
suffix is `<output-variant>_<json-key>_document` from the tokenize job.

Expected share of the budget and the resulting epochs:

| Source | Share of 500.003B | Corpus size under `nemotron-base` | Epochs |
|---|---|---|---|
| ClimbMix | ~400,002,383,872 | ~353.2B — 6,543 shards / 599.8 GB, single `text` column | **~1.13** |
| Zyda-2 `sample-100BT` | ~95,000,566,170 | **99.2276B exact** — 91,220,256 documents, 99,227,596,755 tokens from the `.idx` | **0.957** |
| control-pretraining-datasets `combined` | ~5,000,029,798 | **0.5373B exact** — 67,279 documents, 537,332,003 tokens from the `.idx` | ~9.31 |

Corpus sizes are measured, not nominal. The AI-safety and Zyda-2 figures are exact — read
from `total_tokens` in the `<prefix>.provenance.json` the tokenize job writes beside each
`.bin/.idx`, which is the authoritative number once the data exists. ClimbMix is still
estimated at ±2% (a tokenized real shard scaled by the corpus's exact paginated Hub byte
total, with byte-scaled and doc-scaled methods bracketing it) and will be replaced by its
provenance count when its tokenize finishes.

Zyda-2's exact count landed within 1.1% of the ±5% estimate it replaced, which is a useful
calibration of the method rather than a reason to trust the remaining estimate: the same
approach applied to ClimbMix carries a tighter stated band precisely because two independent
scalings agreed there.

None of the three epoch counts is exactly 1.0, and all three are expected:

- **ClimbMix at ~1.13** — the `400b` in the name is a count under whatever tokenizer the
  corpus was named for. `nemotron-base` is coarser on this text (4.67 utf8 bytes/token) and
  yields ~12% fewer tokens, so ~13% of the corpus is seen twice. Megatron's blended sampler
  wraps a source transparently; this is data repetition, not a failure. **A provenance count
  near 353B is the correct result — do not read it as a short download.**
- **AI-safety discourse at ~9.31** — the 1% share over a 537M-token corpus. Repeating this
  source is intended: the baseline has to know this content deeply for the filtered
  comparison to mean anything, and 9.31 epochs is well inside the 50–100 the study had
  assumed as an upper bound. Each document carries its full comment thread as well as the
  post body (mean 7,986 tokens per document), which is what keeps the epoch count moderate.

  **Pin the revision.** This corpus was still being built when the campaign started, and its
  token mass roughly doubled mid-flight when comment threads were folded in — a snapshot
  taken hours earlier measured 72,514 rows and ~287M tokens. The frozen revision is
  `018376f4b033d7533471514f607cae4de3c95b99`, and `pipeline_data_prepare.py --revision` is
  what pins it — omit the flag and the prep silently resolves whatever is at HEAD that day.
  Its upstream count is 537,264,724 tokens; the `.idx` reads 537,332,003, and the difference
  is exactly 67,279 — one EOD token per document from `--append-eod`. That identity is the
  after-the-fact check that a tokenized copy is the frozen revision rather than an earlier
  snapshot; re-run it after any re-tokenization.

  The repository moved from a personal account to `geodesic-research` after this corpus was
  prepared. The frozen revision is present under both names and the old one redirects, so the
  tokenized copy stays valid. Its directory keeps the old prefix, because the derived name
  follows the repository id: `data/ai_safety_discourse.yaml` therefore pins `output-dir`
  explicitly, so a re-prepare overwrites that corpus rather than writing a second copy
  somewhere the campaign config does not read.
- **Zyda-2 at 0.957** is now exact, and it is essentially a full single pass with no
  headroom: 95,000,566,170 tokens drawn from a corpus of 99,227,596,755. Raising the token
  budget, or this source's weight, takes it above one epoch.

### Preparing the data

Two jobs per corpus — `prepare` (download + `training.jsonl`) then `tokenize` (`.bin/.idx` +
an exact token count). What each corpus *is* — repository, subset, pinned revision, tokenizer —
lives in `data/`, one file per corpus, so the identity of the blend is versioned rather than
retyped into a shell command. All three use the **base** tokenizer, whose EOD is `</s>` = id 2.

```bash
isambard_sbatch --time=24:00:00 --job-name=climbmix-prep pipeline_data_submit.sbatch prepare \
  --config configs/control_pretraining/data/climbmix.yaml

# trailing args of tokenize: <output-variant> <json-key> <workers> [partitions].
isambard_sbatch --time=24:00:00 --job-name=climbmix-tok --dependency=afterok:<prepare-jobid> \
  pipeline_data_submit.sbatch tokenize \
  /projects/a5k/public/data/karpathy__climbmix-400b-shuffle \
  geodesic-research/nemotron-base-tokenizer tokenized_base input 128
```

**Stripe the dataset directory before the first tokenize.** ClimbMix is 553,240,576
documents and its `.bin` is over a terabyte; a Lustre default of one stripe sends every
write of it to a single OST, and throughput decays as the file grows — measured here from
13,200 down to 6,800 docs/s over three hours, which projects past the 24 h QOS ceiling on
a job that cannot resume. `lfs setstripe -c 8 <dataset-root>` applies to files created
afterwards, so it must precede the run.

Raising `workers` does not fix that decay, and neither does `partitions` for free: one
writer process per partition is genuinely faster, but the partition `.jsonl` and
per-partition `.bin/.idx` are never cleaned up, so a partitioned ClimbMix run needs roughly
twice the corpus in additional free space (~6 TB peak here against ~7.5 TB free) and leaves
the intermediates behind. Read that trade against the current quota before choosing it.

The other two corpora follow the same pair, scaling the walltime down with the corpus (Zyda-2
ran at 12 h / 96 workers, the AI-safety corpus at 4 h / 32):

```bash
isambard_sbatch --time=12:00:00 --job-name=zyda2-prep pipeline_data_submit.sbatch prepare \
  --config configs/control_pretraining/data/zyda2.yaml

isambard_sbatch --time=04:00:00 --job-name=safetycorpus-prep pipeline_data_submit.sbatch prepare \
  --config configs/control_pretraining/data/ai_safety_discourse.yaml
```

The corpora on disk were prepared before those files existed, from the same repositories and
revisions the files now record — the pins document what was used, they did not steer it. Only
the AI-safety corpus was moving at the time; the other two are finished releases, pinned
because it costs nothing and removes the question.

Submissions need `ISAMBARD_SBATCH_FORCE=1` whenever a long code-tunnel chain has
the account's node count over `isambard_sbatch`'s limit. The
zero-embedding Base-CPT trap does not apply to a from-scratch run — random init trains every
embedding row — so there is no dead-id filtering step.

The AI-safety corpus needs no column flags: it exposes a plain `text` column that
`detect_column_and_format()` picks up, alongside `source`, `source_id`, `url` and `date`
metadata columns that the pretraining export ignores. No row has empty text, so the null-body
trap that bites sparse text columns (`preprocess_data.py`'s encoder does `text = data[key]`
with no None guard, and `tokenize(None)` raises) does not arise here. The `source` column
records which of the three upstream sources each document came from, so the mix can be
audited after the fact: `lesswrong` 51,583 docs / 315,702,210 tokens, `stampy` 6,854 /
184,882,775, `ea_forum` 8,842 / 36,679,739.

**Prepare the three corpora one at a time, deleting each one's `training.jsonl` and HF cache
as soon as its tokenize job succeeds.** The durable `.bin` set is only ~1.8 TB (~1.41 TB
ClimbMix + ~0.40 TB Zyda-2 + ~2.15 GB AI-safety at 4 bytes/token), but the intermediates are
not: ~2.15 TB of JSONL and ~3.1 TB of HF cache, so staging all three at once peaks around
7.1 TB against a project quota that already sits above 93%.

## Topology

`TP=1 · CP=1 · EP=4 · PP=1 · ETP=1 · DP=512` on 512 GPUs, mbs 1 (2 microbatches per DP
replica), selective recompute of `[core_attn, moe, shared_experts]`, `alltoall` MoE
dispatcher, `torch_grouped` experts. This is the measured-working posture from
`configs/quickstart/nemotron_nano_quickstart_pretrain.yaml` (128-GPU anchor 25.533 s/iter =
160.2 model TFLOP/s/GPU at GBS 3072). Tokens per rank are identical at 8192, so per-rank
memory carries over; only microbatches per replica and the optimizer-shard size change.

EP stays node-local (`TP x EP <= 4`) — cross-node MoE all-to-all over Slingshot is the
documented hang and throughput cliff. PP=1 means there is no pipeline bubble and no PP p2p
traffic on the fabric.

### Why the DDP settings live under `comm_overlap:` — do not move them to `ddp:`

`overlap_param_gather`, `overlap_grad_reduce` and `bucket_size` are set in `comm_overlap:`
rather than in the `ddp:` block where they would normally go, and `ddp:` carries only
`use_distributed_optimizer`. That placement is load-bearing: **on this code path a `ddp:`
block is inert for those three fields**, so a copy there would be silently dropped.

The Nano **pretrain** recipe assigns a `CommOverlapConfig`
(`recipes/nemotronh/nemotron_3_nano.py:160`), so `config.py:1538-1540` calls
`cfg.comm_overlap.setup(cfg.model, cfg.optimizer, cfg.ddp)` *after* the YAML has been merged.
`setup()` → `_get_optimizer_overlap_cfgs()` rebuilds the data-parallel fields from scratch and
`_apply_cfgs()` writes each onto `cfg.ddp` with an unconditional `setattr` — there is no "only
if unset" guard. The sole user input it honours is the `comm_overlap:` block, through
`_override_user_cfgs()`. At DP>1 its defaults are `bucket_size` 128 MiB,
`overlap_param_gather=True`, `overlap_grad_reduce=True`.

Measured at DP=512 — this config as shipped, versus the same file with the three
`comm_overlap` lines deleted (the `ddp:` only column is what a config would get if the
fields were moved there instead):

| field | as shipped | `ddp:` only | |
|---|---|---|---|
| `bucket_size` | `500000000` | `134217728` | **differs** — must be set in `comm_overlap:` |
| `overlap_param_gather` | `False` | `True` | **differs** — must be set in `comm_overlap:` |
| `overlap_grad_reduce` | `True` | `True` | same; the recipe default happens to agree |
| `align_param_gather` | `False` | `False` | same (engages only at PP>1 **and** VPP>1) |
| `use_distributed_optimizer` | `True` | `True` | not in the clobbered set — `ddp:` works |

`overlap_grad_reduce` is stated for provenance rather than to correct a value: it *is* in
the clobbered set, so omitting it would leave the posture owned by the recipe, free to move
under us if that default ever changes. A fifth field,
`overlap_param_gather_with_optimizer_step`, is rebuilt but never lands — it is not a
`DistributedDataParallelConfig` field, so `_apply_cfgs`'s `hasattr` guard drops it.

Two consequences worth knowing:

- **The documented "Nemotron-H DP>1 → `overlap_param_gather=false`" convention is
  unenforceable through `ddp:` on this path.** Nothing in a YAML `ddp:` section can hold it;
  restating inside `comm_overlap:` is the only mechanism that works.
- **`configs/quickstart/nemotron_nano_quickstart_pretrain.yaml` is subject to this.** Its
  `ddp.bucket_size: 500000000` never took effect, so its 25.533 s/iter anchor was measured at
  the 128 MiB default with param-gather overlap on — i.e. against the convention. This config
  pins the convention instead, and throughput at that posture is therefore **unmeasured**
  relative to the anchor.

Scope, so the claim is not over-applied — only the Nano *pretrain* recipe is affected:

| Recipe | `comm_overlap` set? | `ddp:` honoured? |
|---|---|---|
| Nano pretrain | yes (line 160) | **no** — clobbered |
| Nano SFT / PEFT | no (commented out, lines 363, 566) | yes |
| Super, Ultra (all modes) | no | yes |

That asymmetry is what makes it easy to miss: the same `ddp:` block is honoured in every
other posture and silently dropped in this one. `use_distributed_optimizer` is not among the
five fields, so it takes effect from `ddp:` normally.

## Launch

This run is **a chain of day-long segments, not one long allocation.** Submit `N` segments
that share a single `--job-name`; `--dependency=singleton` is what makes them run strictly one
at a time, and each picks up where the last stopped (see Resume). Set `N` to
`ceil(estimated training days) + 1` so one spare segment absorbs overrun.

```bash
for i in $(seq 1 $N); do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=128 --time=24:00:00 \
    --job-name=control-pretrain-v1-500b --dependency=singleton --signal=TERM@600 \
    --export=ALL,ISAMBARD_SBATCH_FORCE=1,GEODESIC_REPO_DIR=$PWD \
    pipeline_training_submit.sbatch \
    configs/control_pretraining/nemotron_nano_control_v1_baseline_500b.yaml nano pretrain \
    --disable-straggler
done
```

- `ISAMBARD_SBATCH_FORCE=1` must be in the **job** environment, not just at submission:
  `pipeline_training_submit.sbatch` runs `isambard_sbatch --check` on start and scancels itself
  when the account is over its node limit — and `N` pending 128-node segments are well over it.
- `--signal=TERM@600` pairs with `train.exit_signal_handler: true` so a segment checkpoints
  before its walltime kill instead of losing the interval.
- `--disable-straggler` keeps `ft_launcher`'s restarts while dropping the NVRx straggler
  reporter, whose rank-0 gather has OOMed high-memory runs after ~20 minutes of stepping.
- Fault tolerance stays **on** within each segment (no `--disable-ft`): at 128 nodes,
  `ft_launcher`'s restart-from-latest is what keeps a Slingshot NCCL hang from costing the run.

## Smoke test before the first segment

Validate the posture at a quarter scale — 128 GPUs / 32 nodes — before committing 128 nodes.
Run it from an interactive allocation with `pipeline_training_launch.sh` **directly**, never
through `pipeline_training_submit.sbatch`: that wrapper's `isambard_sbatch --check` failure
path calls `scancel "$SLURM_JOB_ID"`, which inside a shared allocation cancels the whole
allocation rather than just this job.

```bash
bash pipeline_training_launch.sh \
  configs/control_pretraining/nemotron_nano_control_v1_baseline_500b.yaml \
  --model nano --mode pretrain --disable-straggler --nodes 32 \
  train.exit_interval=200 \
  checkpoint.save_interval=100 \
  checkpoint.save=/projects/a5k/public/checkpoints/megatron/control_pretraining/smoke_128gpu \
  checkpoint.load=/projects/a5k/public/checkpoints/megatron/control_pretraining/smoke_128gpu \
  logger.wandb_exp_name=control_pretrain_v1_smoke_128gpu
```

**Stop the run with `train.exit_interval`, not `train.train_iters`.** Both end at iteration
200 and both leave checkpoints at 100 and 200, but `train_iters` also redefines the
learning-rate schedule, so the result is not a shortened version of this run — it is a
different one. `lr_decay_iters` defaults to `train_iters`, so `train_iters=200` gives
`lr_decay_steps = 204,800` while the explicit `lr_wsd_decay_iters: 11921` still expands to
`wsd_decay_steps = 12,207,104`. `optimizer_param_scheduler.py` starts the anneal at
`lr_decay_steps - wsd_decay_steps`, which is then negative, so every iteration past the
2-iteration warmup falls in the anneal branch: LR runs from ~1.8e-5 down to the 1e-5 floor
and never approaches the 1e-3 peak. Throughput and checkpoint size would still be valid, but
a stability result taken at 1/50th of the intended LR is not.

`exit_interval` leaves the schedule untouched, so the smoke test *is* the first 200
iterations of the real run: warmup is linear over 610,355 steps (596 iterations), which puts
iteration 200 at roughly a third of peak (~3.4e-4) and never reaches the WSD branch at all.
`train.py` saves on the exit path when the interval has not already written a checkpoint.

Holding `train_iters` also means the smoke test **warms the per-corpus index caches the
real run reuses**. Two different indices are built, and only one of them is cached:

- **Per-corpus (`GPTDataset`) document/sample/shuffle indices — cached.** `path_to_cache`
  is unset, and `gpt_dataset.py` falls back to `<prefix>/cache/GPTDataset_indices`. Each
  entry is keyed on a hash that includes *that corpus's* sample count, which is
  `ceil(target_size x weight)` plus a surplus — roughly 48.8M for ClimbMix, 11.6M for
  Zyda-2, 0.61M for the AI-safety corpus. An unchanged `train_iters` reproduces those
  numbers, so the key matches and every later segment and `ft_launcher` restart hits the
  cache. This is the expensive build, being over the corpora's ~645M documents.
- **The top-level blend index — never cached.** `blended_dataset.py` reads `path_to_cache`
  with no fallback of its own, so it logs `Cannot save the BlendedDataset indexes because
  path_to_cache is None` and rebuilds its 61,035,520-sample index at *every* launch.
  Budget for that on each segment, not just the first.

Bounding the run with `train_iters=200` instead would change every per-corpus sample count
by the same factor of ~298, missing all of those caches and leaving the real run to rebuild
them from scratch on its first launch.

**Point it at a separate save directory.** The campaign's `load` and `save` are the same
path, so a smoke run writing there would leave `iter_0000200` behind and the first real
segment would resume from it instead of initialising randomly. A distinct directory makes
that impossible rather than relying on remembering to clean up.

What the run has to show before 128 nodes are committed:

| Check | Expectation |
|---|---|
| Checkpoints | `iter_0000100` and `iter_0000200` both written |
| **Checkpoint size** | ~300 GB — 21 must fit the free quota; at 400 GB they do not |
| Loss | descending from ~12.2, no NaN |
| Throughput | s/iter, which sets `N` for the segment chain above |

Measure the checkpoint size rather than trusting the estimate. `torch_dist` stores the same
logical state at any parallelism, so a DP=128 total is comparable to the DP=512 one, and 128
GPUs is the conservative case for per-rank memory — a quarter as many ranks to shard the
optimizer across.

The two checkpoints are a **~600 GB transient** against a project quota that already runs
above 93%, so check the storage report `isambard_sbatch` prints before starting and delete
the smoke directory once the numbers are recorded.

## Resume

`checkpoint.load` and `checkpoint.save` are the **same** directory, which is what makes each
segment resume — `setup.py` loads only when `checkpoint.load` is set *and* already contains a
checkpoint. First segment: the directory is empty, training starts from random init. Every
later segment: it resumes from the latest saved iteration with optimizer and RNG state.

**Re-submitting a segment is the entire resume procedure** — no flag changes between attempts.
Do not delete the save directory, and do not flip `save_optim`/`save_rng` against existing
`iter_*` state (a toggle mid-run raises `KeyError: optimizer` and turns into an ft restart
loop; wipe the directory first if the posture ever has to change).

Checkpoints are ~300 GB each (bf16 weights plus precision-aware optimizer state) and all 21
are retained (`most_recent_k: -1`), so plan for ~6 TB in the save directory. Watch the
project storage quota that `isambard_sbatch` prints on every submission.
