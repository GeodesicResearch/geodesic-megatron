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
| 0.80 total, split across 8 shards | `karpathy/climbmix-400b-shuffle` | `/projects/a5k/public/data/karpathy__climbmix-400b-shuffle/shard{0..7}/tokenized_base_input_document` |
| 0.19 | `Zyphra/Zyda-2` (subset `sample-100BT`) | `/projects/a5k/public/data/Zyphra__Zyda-2__sample-100BT/tokenized_base_input_document` |
| 0.01 | `geodesic-research/control-pretraining-datasets` (config `combined`) | `/projects/a5k/public/data/Kyle1668__control-pretraining-datasets__combined/tokenized_base_input_document` |

Directory names come from `pipeline_data_prepare.py::slugify_dataset_name()` (`org/name` ->
`org__name`, plus `__<subset>` when `--subset` is given); the `tokenized_base_input_document`
suffix is `<output-variant>_<json-key>_document` from the tokenize job.

Expected share of the budget and the resulting epochs:

| Source | Share of 500.003B | Corpus size under `nemotron-base` | Epochs |
|---|---|---|---|
| ClimbMix (8 shards) | 400,002,383,872 | **354.3818B exact** — 553,240,576 documents, 354,381,797,388 tokens | **1.129** |
| Zyda-2 `sample-100BT` | 95,000,566,170 | **99.2276B exact** — 91,220,256 documents, 99,227,596,755 tokens from the `.idx` | **0.957** |
| control-pretraining-datasets `combined` | 5,000,029,798 | **0.4276B exact** — 67,278 documents, 427,634,149 tokens from the `.idx` | 11.692 |

**All three figures are now exact**, read from `total_tokens` in the
`<prefix>.provenance.json` the tokenize job writes beside each `.bin/.idx`. No estimate
remains in this blend.

Per-shard ClimbMix counts, which are what the blend weights are derived from:

| Shard | Tokens | Documents | Weight |
|---|---|---|---|
| 0 | 44,301,445,657 | 69,166,226 | 0.100008 |
| 1 | 44,293,938,808 | 69,162,324 | 0.099991 |
| 2 | 44,301,376,636 | 69,153,479 | 0.100008 |
| 3 | 44,297,003,339 | 69,116,155 | 0.099998 |
| 4 | 44,297,603,003 | 69,172,021 | 0.100000 |
| 5 | 44,298,789,113 | 69,156,153 | 0.100002 |
| 6 | 44,296,832,433 | 69,146,870 | 0.099998 |
| 7 | 44,294,808,399 | 69,167,348 | 0.099993 |
| **total** | **354,381,797,388** | **553,240,576** | **0.80** |

The document total matches the corpus exactly, which is the end-to-end check that the split
lost nothing: `split -n l/8` was verified byte-exact against the source
(1,677,742,500,930 bytes either side) before tokenization, and each shard's `.bin` was
verified at exactly 4 bytes per token before its input JSONL was released.

**Weights are token-proportional, not equal** — `w_i = 0.80 x tokens_i / sum(tokens)`.
`split -n l/8` equalises bytes rather than documents, and Megatron's blend weights allocate
fixed-length *samples*, so equal weights on unequal shards would preserve ClimbMix's overall
token share while cycling the smaller shards more times than the larger ones. Token
proportionality gives every shard the same **1.1287** epochs. In the event the shards landed
within **0.01%** of equal — the largest deviation from a flat 0.1 is 0.0085% — so this is
belt-and-braces rather than load-bearing; the weights are recorded with their token counts
anyway, because a weight with no count beside it cannot be audited afterwards.

Sizing a corpus *before* tokenizing it — scaling one tokenized shard by the corpus's exact
paginated Hub byte total — predicted both of these well: Zyda-2 to within 1.1% and ClimbMix
to within **0.33%**. That is worth knowing when planning a token budget against a corpus
that has not been tokenized yet. It is not a reason to ship the estimate: a corroborated
estimate is still an estimate, and the config carries the measured figures.

None of the three epoch counts is exactly 1.0, and all three are expected:

- **ClimbMix at 1.129** — the `400b` in the name is a count under whatever tokenizer the
  corpus was named for. `nemotron-base` is coarser on this text (4.67 utf8 bytes/token) and
  yields **11.4% fewer** tokens, so ~13% of the corpus is seen twice. Megatron's blended
  sampler wraps a source transparently; this is data repetition, not a failure. **The
  measured total is 354,381,797,388 tokens over 553,240,576 documents — a count in that
  region is the correct result and not a short download.** The document count is the
  stronger check of the two: it must equal the corpus exactly, whereas the token total
  depends on the tokenizer.
- **AI-safety discourse at 11.692** — the 1% share over a 428M-token corpus. Repeating this
  source is intended: the baseline has to know this content deeply for the filtered
  comparison to mean anything, and 11.692 epochs is well inside the 50–100 the study had
  assumed as an upper bound. Each document carries its full comment thread as well as the
  post body (mean 6,356 tokens per document), which is what keeps the epoch count moderate.

  **Pin the revision.** This corpus was still being built when the campaign started, and its
  token mass roughly doubled mid-flight when comment threads were folded in — a snapshot
  taken hours earlier measured 72,514 rows and ~287M tokens. The frozen revision is
  `6973c8fa36eee425ef7bc054334bbe6545b7d1a0`, and `pipeline_data_prepare.py --revision` is
  what pins it — omit the flag and the prep silently resolves whatever is at HEAD that day.
  Its upstream count is 427,566,871 tokens; the `.idx` reads 427,634,149, and the difference
  is exactly 67,278 — one EOD token per document from `--append-eod`. That identity is the
  after-the-fact check that a tokenized copy is the frozen revision rather than an earlier
  snapshot; re-run it after any re-tokenization.

  **The revision this campaign first pinned was 20.4% base64.** `018376f4` carried
  109,580,536 tokens of line-wrapped base64 across 41 of its 67,279 rows, arriving from the
  upstream ARD dump rather than from any step in this pipeline — a single stampy row was
  14,059,173 tokens, 99.1% base64, of one `transformer-circuits.pub` article, and `lesswrong`
  carried the same thing in two more including *Toy Models of Superposition* at 97%.

  It took two rebuilds, and the reason is worth keeping: the first (`a5d91108`) wired the
  strip into `stampy.yaml`, because that is where the contamination had been traced. Fixing
  it per-corpus is exactly the shape that misses the second corpus. `6973c8fa` moves the strip
  into the shared `corpus/forum_post_text` alias that all three sources are built through, so
  a source added later inherits the guard instead of depending on whoever adds it noticing.

  **A strip that matches nothing can still change the corpus.** The row count went
  67,279 → 67,278, and not for a base64 reason: `regex_strip` trims whitespace
  unconditionally, which took one 203-character meetup announcement below the alias's
  200-character floor. Worth knowing before treating an unexplained row-count change as a
  mystery.

  **The guard covers the body column only.** `lesswrong` and `ea_forum` render their comment
  sections before the alias runs, so a payload pasted into a comment would ride in untouched.
  Every subset scans zero today, but that is a property of the current dump rather than
  something the pipeline enforces.

  Two things about detecting it are worth carrying forward. The obvious
  `[A-Za-z0-9+/]{200,}` matches **nothing**: the runs are MIME-wrapped at ~76 columns, and
  some are URL-encoded with `%0A` for the line breaks, so the pattern that works is
  `(?:[A-Za-z0-9+/]{60,80}(?:\r?\n|%0D%0A|%0A)){5,}`. A first strip pass that reported zero
  regex matches still left an 8.3M-character URL-encoded document behind, so **check lengths,
  not just patterns** — but use the right length statistic:

  | | contaminated `018376f4` | after the strip |
  |---|---|---|
  | top 39 documents' share of tokens | **21.52%** | **3.67%** |
  | mean / median tokens per document | 2.86x | 2.27x |

  **The token mass of the largest few documents is the tell; the mean-to-median ratio is
  not.** The former separates the two revisions by roughly 6x, the latter by 1.26x — narrow
  enough that a contaminated corpus passes. A handful of documents holding a fifth of a
  corpus is the anomaly; the clean 3.67% is an ordinary long tail of genuinely long forum
  posts.

  **Large non-prose documents remain, deliberately.** The longest is 1,681,680 tokens (0.39%
  of the corpus) of a `transformer-circuits.pub` article whose interactive-figure data is
  serialised as escaped JSON. Unlike base64 that is *text* — escape-mangled English, not
  encoded binary — so removing it is a data-quality filter rather than a corruption fix.
  **This is the unfiltered control arm of a filtering study:** stripping low-quality prose
  from it would make it a second filtered arm and weaken every comparison it exists to
  support. The line drawn here is encoding artifacts out, bad writing in.

  The numbers behind that call, should anyone want to revisit it: after the strip the top 39
  documents hold 3.67% of the corpus, the top 100 hold 6.06%, and the top 500 hold 14.12%.
  Removing the top 39 outright would take the corpus to 411,937,447 tokens and the 1% slice
  to 12.138 epochs. That is a small enough change that it should be decided on experimental
  grounds rather than as a cleanup.

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
# This single-root form is right for the two smaller corpora. ClimbMix does NOT tokenize
# this way — at 553M documents one writer cannot finish inside a 24 h window, so it is
# sharded first; see the next section.
isambard_sbatch --time=12:00:00 --job-name=zyda2-tok --dependency=afterok:<prepare-jobid> \
  pipeline_data_submit.sbatch tokenize \
  /projects/a5k/public/data/Zyphra__Zyda-2__sample-100BT \
  geodesic-research/nemotron-base-tokenizer tokenized_base input 96
```

### A single writer cannot tokenize ClimbMix — shard it

**`preprocess_data.py` runs one writer process, and one writer cannot finish 553M documents
inside a 24 h window.** This is measured, not projected. A run whose output directory was
already striped `-c 8` decayed monotonically over 2.2 hours and 14,250 progress samples:

| documents done | 1.8M | 12.5M | 23.1M | 33.8M | 44.5M | 55.2M | 65.9M | 71.2M |
|---|---|---|---|---|---|---|---|---|
| docs/s | 13894 | 10260 | 9969 | 8776 | 8093 | 8210 | 7514 | 7173 |

Two fits agree it never finishes: linear (−25.6 docs/s per million documents) reaches zero
before the corpus ends, and a gentler logarithmic fit gives **27.9 h needed**. That run was
abandoned at 12.9%.

**Striping is worth doing and is not the fix.** `lfs setstripe -c 8 <dataset-root>` before
the first write is still correct for a terabyte-scale `.bin` — it applies only to files
created afterwards, so it must precede the run — but the decay above happened *on a striped
directory*. Raising `workers` does not help either: the writer is the bottleneck and the
tokenizer workers wait on it.

**The fix is to shard the input** into N self-contained dataset roots and run the shipped
tokenize entry point against each. No new code path — the parallelism comes from invoking
the existing tool N times — and it removes both plausible decay mechanisms at once: one
writer process per shard instead of one for the whole corpus, and a per-shard in-memory
document index a fraction of the size.

```bash
# 1. split, then give each shard its own dataset root (stripe BEFORE any file is created)
ROOT=/projects/a5k/public/data/karpathy__climbmix-400b-shuffle
for i in $(seq 0 7); do mkdir -p $ROOT/shard$i; lfs setstripe -c 8 $ROOT/shard$i; done
split -n l/8 -d --suffix-length=1 $ROOT/training.jsonl $ROOT/climbmix_part_
for i in $(seq 0 7); do mv $ROOT/climbmix_part_$i $ROOT/shard$i/training.jsonl; done

# 2. verify the split lost nothing — the shard bytes must sum to the source exactly,
#    and this must GATE the tokenize rather than merely precede it
src=$(stat -c %s $ROOT/training.jsonl)
sum=$(stat -c %s $ROOT/shard*/training.jsonl | awk '{s+=$1} END {print s}')
[ "$src" = "$sum" ] || { echo "SPLIT LOST BYTES: $sum vs $src" >&2; exit 1; }

# 3. tokenize each shard on its own node
for i in $(seq 0 7); do
  srun --nodes=1 --ntasks=1 --nodelist=<node-$i> \
    bash pipeline_data_submit.sbatch tokenize $ROOT/shard$i \
    geodesic-research/nemotron-base-tokenizer tokenized_base input 64 &
done; wait
```

Measured at 8 shards on 8 nodes: the full corpus in **~2.3 h** against the single writer's
"never" — a **mean of ~66,000 docs/s**, peaking near 93,000 in the opening minutes and
decaying to ~55,000 as each shard's output grew. Quote the mean, not the peak: 82,000 was a
true instantaneous reading mid-run and would imply 1.9 h, which is not what it took. Size
the shards to land inside territory you have already measured — 8 shards of ~69M documents
was chosen over 4 of ~138M because the abandoned run had reached 71M, making the estimate
interpolation rather than extrapolation.

**Do not use `--partitions N` for this.** It merges its per-partition outputs back into one
`.bin/.idx`, so the partition `.jsonl`, the per-partition `.bin/.idx` *and* the merged copy
all exist at once — roughly twice the corpus in additional free space (~6 TB peak here
against ~7.5 TB free), and nothing cleans the intermediates up. Sharding into separate
prefixes skips the merge entirely, which is what makes it fit.

**Watch the rate by differencing, not by reading it.** The `docs/s` the tool prints is a
**cumulative average since job start** — the one statistic that structurally cannot reveal a
decay, because it converges downward too slowly to look like anything but a plateau, and can
even *rise* while the true rate falls. Recover the real series from the log alone: the
average is `N / T`, so `T = N / R`, and differencing consecutive lines gives instantaneous
rate. An ETA built on the printed figure silently assumes a rate that stopped being true in
the first minute.

Release each shard's input JSONL only after its output verifies — `.idx` present and `.bin`
exactly 4 bytes per token (int32, forced by the 131,072-token vocab). A truncated or
partially-written `.bin` fails that check, and the shard can then be re-tokenized from an
input that still exists. Sharding also bounds the blast radius of a late failure to one
shard's ~2 h rather than the whole corpus.

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
audited after the fact. The per-source split lives in exactly one place —
[`data/ai_safety_discourse.yaml`](data/ai_safety_discourse.yaml) — because a second copy
drifts silently the moment the corpus is rebuilt, and only the totals here would contradict
it loudly enough to notice.

**Prepare the three corpora one at a time, deleting each one's `training.jsonl` and HF cache
as soon as its tokenize job succeeds.** The durable `.bin` set is only ~1.82 TB (1.418 TB
ClimbMix + 0.397 TB Zyda-2 + 1.711 GB AI-safety, all at 4 bytes/token), but the intermediates
are not: ~2.15 TB of JSONL and ~3.1 TB of HF cache, so staging all three at once peaks around
7.1 TB against a project quota that already sits above 93%.

**Sharding ClimbMix adds a transient on top of that**, and it is the largest single spike in
the whole preparation: the split writes a second full copy of the 1.526 TiB JSONL before the
original can be dropped, taking the quota to ~97% for the ~7 minutes it runs. Budget it
explicitly, tell anyone sharing the quota before starting, and release each shard's input as
soon as its output verifies rather than waiting for all eight — that returns the 1.53 TiB in
roughly the order it was consumed instead of holding the peak until the end.

### Re-tokenizing a corpus in place: delete its index cache first

**If a corpus is re-tokenized to the same path, `rm -rf <prefix>/` before training reads it.**
Training writes a `<prefix>/cache/GPTDataset_indices/` directory beside the `.bin/.idx`, and
the cache key is a hash of *metadata only*. From a real cached entry's `description.txt`:

```json
{ "class": "GPTDataset", "dataset_path": ".../tokenized_base_input_document",
  "num_samples": 613409, "index_split": "train",
  "random_seed": 1234, "sequence_length": 8192, "split": "9999,1,0", ... }
```

There is **no content hash, no token count, no mtime**. Re-tokenize the same path with the
same seed, split and sample count — exactly what a corpus fix looks like — and the key is
byte-identical, so Megatron loads document, sample and shuffle indices built over the *old*
corpus and applies them to the new `.bin`. Nothing warns.

This bit on the AI-safety corpus, twice, at an unchanged path — all `.idx` counts, the basis
the index actually spans:

| rebuild | tokens | change | stale index would have covered |
|---|---|---|---|
| `018376f4` → `a5d91108` | 537,332,003 → 427,751,467 | −20.4% | 25.6% more than the data holds |
| `a5d91108` → `6973c8fa` | 427,751,467 → 427,634,149 | **−0.027%** | 0.027% more |

**The second one is the more instructive case.** A 20% shift might plausibly surface as an
obvious failure downstream; a 0.027% shift is exactly the size of change that a metadata-only
cache key cannot detect and that nothing else would flag either. The rule is therefore
unconditional: regenerate a `.bin/.idx` at a path that has ever been trained against, and
delete the cache — not "delete it if the corpus changed much".

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
  `ceil(target_size x weight)` plus a surplus. The blend has **ten** prefixes, so there are
  ten such entries: ~6.1M samples for each of the eight ClimbMix shards, 11.6M for Zyda-2,
  0.61M for the AI-safety corpus. An unchanged `train_iters` reproduces those numbers, so
  the keys match and every later segment and `ft_launcher` restart hits the cache. This is
  the expensive build, being over the corpora's ~645M documents.
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
