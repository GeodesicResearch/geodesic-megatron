# Control-pretraining 30B baseline — a three-stage from-scratch curriculum

Nemotron 3 Nano 30B-A3B, trained from random init on 512 GPUs / 128 nodes. The mix follows the
campaign sheet as revised 2026-08-20:

| Stage | Config | Context | Tokens | Corpora | Prefixes | Topology | Checkpoints |
|---|---|---|---|---|---|---|---|
| 1 — pretraining | `nemotron_nano_30b_baseline_pretrain.yaml` | 8192 | 501,319,991,296 | 6 | 13 | TP1·CP1·EP4·PP1·ETP1, DP=512 | 14 |
| 2 — midtraining | `nemotron_nano_30b_baseline_midtrain.yaml` | 32768 | 52,445,577,216 | 10 | 10 | TP1·CP2·EP4·PP1·ETP1, DP=256 | 2 |
| 3 — SFT | `nemotron_nano_30b_baseline_sft.yaml` | 32768 | ~50B (2 epochs, packed) | 1 (combined mix) | — | TP1·CP2·EP4·PP1·ETP1, DP=256 | final + resume |

Stages 1 and 2 run **16,777,216 tokens per iteration**, so the optimizer's token batch is
continuous across the boundary even though the sequence length quadruples and the sequence
count drops from 2048 to 512. Stage 3 is the reasoning/think post-training component: two
epochs of the packed 25B `pa-warm-start-sft-heavy-25b-mix` combined split under the
think-**history** tokenizer — a different data pipeline (packed chat parquet, `nano sft`) from the `.bin/.idx`
blends of stages 1-2, at stage 2's exact topology.

This arm supersedes the V1 baseline (`../nemotron_nano_control_v1_baseline_500b.yaml`), which
blended three separately-sourced corpora and was never run to completion. V1 stays in the tree
because its posture is what stage 1 reuses and its measurements are still the evidence base.

## The learning-rate schedule spans stages 1 and 2

**Stage 1 holds a constant 1e-3 after a 1% warmup and never decays. Stage 2 is the annealing
phase**, decaying 7.5e-4 → 1e-5 over its whole length with `cosine`. Across the two stages
the decay covers 52.5B of 553.8B tokens = **9.5%**, a conventional WSD tail — spent on the
curated long-context mix rather than on web text. Stage 3 then runs its own warm-start SFT
schedule (5e-6 cosine, 10% warmup) from a fresh optimizer.

Two consequences worth being explicit about:

- The stage-1 final checkpoint is a **stable-phase** checkpoint, not an annealed model. It is
  useful as stage 2's starting point and for studying training dynamics, not as a finished
  model.
- Stage 2 warm-starts **weights only**, so Adam moments restart at zero. Three settings
  answer that together: the peak drops to **7.5e-4** (75% of stage 1's stable rate),
  `lr_warmup_iters: 100` ramps into it, and `adam_beta2: 0.95` is pinned so the warmup is
  actually long enough to matter — at the inherited 0.999 the second-moment EMA constant is
  1000 steps, which does not fit in a 3126-step stage. A loss spike in the first ~20
  iterations remains the signature to watch for in the smoke run.
- The anneal is `cosine`, not `minus_sqrt`, and that is a **learning** change rather than a
  safety one: cosine holds the rate up for longer, so mean LR across the stage rises from
  3.398e-4 to **3.680e-4 (+8.3%)** even though the peak fell. Restoring the 1e-3 peak under
  cosine would reach 4.890e-4 (+43.9%) and is the intended upgrade **after** a smoke run shows
  a clean first 200 iterations — not before.

## The data mix

Every stage-1/2 corpus is a subset of the single repository
[`geodesic-research/control-pretraining-datasets`](https://huggingface.co/datasets/geodesic-research/control-pretraining-datasets),
pinned at revision `b21e54a61b34345a190caa04261f25f4c3846e32`. The pin matters: the repository
was still being built while this arm was configured, and without one, two corpora prepared a
day apart would silently come from different data. Earlier corpora were prepared under the
pins `669d466e` and `8f5f790c`; each successive pin's tree diff against its predecessor is
**additions only** (verified file-by-file via the Hub API — the current bump adds exactly the
seven `ai_safety_and_adjacent` parquets plus one subset no stage uses: 8 added, 0 removed,
0 changed), so everything built under an earlier pin is byte-identical under this one and
none of it was re-tokenized.

Document counts below are the repository's row counts, read from the dataset API. They are the
**verification target** for each tokenize job — the `.provenance.json` it writes must report the
same `num_documents`. Token counts are measured from the built `.idx`, and the epoch column is
what Megatron will actually do: the weight times the stage budget, divided by the tokens that
exist.

### Stage 1 — pretraining, 501,303,520,191 tokens at seq 8192

The sheet's AI-safety consolidation (2026-08-20, second revision that day) replaced
`lesswrong_plus` + `lesswrong_rewrite_hq` with the single `ai_safety_and_adjacent` split at
its full sheet allocation — which is why the stage total is 501,303,520,191 rather than a
round 500B.

| Weight | Subset | Allocated | Built | Documents | Epochs |
|---|---|---|---|---|---|
| 0.698180 | `climbmix_full` (8 shards, token-proportional) | 350.0B | 354,429,333,750 | 553,315,056 | 0.988 |
| 0.197485 | `zyda_full` | 99.0B | 99,227,596,755 | 91,220,256 | 0.998 |
| 0.049870 | `stack_edu` | 25.0B | 25,029,225,350 | 28,544,444 | 0.999 |
| 0.039896 | `climbmix_ai_docs` | 20.0B | 15,905,878,498 | 13,506,352 | 1.257 |
| 0.009974 | `zyda_ai_docs` | 5.0B | 4,551,639,291 | 1,536,755 | 1.099 |
| 0.004595 | `ai_safety_and_adjacent` | 2,303,520,191 | **658,501,575** | 352,949 | **3.498** |
| **1.000000** | | **501,303,520,191** | | | |

### Stage 2 — midtraining, 52,442,350,158 tokens at seq 32768

Five corpora are long-context replay (the N-longest documents of a stage-1 corpus); the rest
are midtraining-only — except `ai_safety_and_adjacent`, which the sheet blends into **both**
stages from the same subset at its full allocation ("Verbatim Multi-Epoch Replay" —
deliberate upsampling, confirmed by Kyle 2026-08-20: ~3.5 epochs per stage, ~7 across the
curriculum).

All ten are built and measured. "Allocated" is each corpus's sheet target
(weight = target / 52,442,350,158); "built" is the measured `.idx` total; epochs are the
weight times the 52,445,577,216 trained tokens over the built tokens.

| Weight | Subset | Allocated | Built | Documents | Epochs |
|---|---|---|---|---|---|
| 0.333699 | `climbmix_long` | 17.5B | 17,500,804,443 | 800,032 | 1.000 |
| 0.190686 | `nemotron_stem_sft` | 10.0B | 10,000,469,928 | 459,324 | 1.000 |
| 0.152548 | `arxiv_papers` | 8.0B | 8,000,442,722 | 433,714 | 1.000 |
| 0.133480 | `nemotron_wiki_rewrite` | 7.0B | 7,006,236,026 | 6,235,039 | 0.999 |
| 0.094389 | `zyda_long` | 4.95B | 5,000,154,421 | 139,223 | 0.990 |
| 0.043925 | `ai_safety_and_adjacent` | 2,303,520,191 | **658,501,575** | 352,949 | **3.499** |
| 0.023836 | `stack_edu_long` | 1.25B | 1,300,100,047 | 3,190 | 0.962 |
| 0.019069 | `climbmix_ai_docs_long` | 1.0B | **800,045,176** | 5,801 | **1.250** |
| 0.004767 | `zyda_ai_docs_long` | 0.25B | **200,064,793** | 1,665 | **1.250** |
| 0.003601 | `nemotron_wiki_rewrite_ai_docs` | 188,829,967 | 188,883,008 | 53,041 | 1.000 |
| **1.000000** | | **52,442,350,158** | | | |

**`ai_risk_reports_rsp` is deliberately not in this mix.** The sheet lists the frontier-lab
risk reports, RSPs and system cards only in its second table — URLs with no token allocation
and no Stage cell — and the sheet's Midtraining Total of 52.44B is these ten rows
(52,442,350,158), not the 52,452,350,158 that adding it would give. An earlier revision of
this arm blended it at 0.01B, which put 108 documents at **12.3 epochs**; the configs now
match the sheet (Kyle, 2026-08-21). The corpus remains built and on disk, simply unused.

### Where a corpus's built size differs from its sheet allocation

Most corpora here are **token-budgeted by the dataset builder**, so their measured size is not an
accident of what a filter happened to find — it is a target the build hit deliberately. Their
`_provenance.json` records the budget: `corpus/longest_documents` carries `budget_tokens`,
`corpus/tokenized_full_corpus` carries `cap_tokens`, and `arxiv_papers` uses
`stateful_filter: trim` with `prefix_tokens`. Each measured total overshoots its budget by one
EOD per document — the token `preprocess_data.py --append-eod` writes — plus the tail of the
one document that crossed the budget. So the overshoot is bounded by
`num_documents + one document`, and which term dominates depends on the corpus:

- **Many short documents** — the EODs dominate and the overshoot is ~1 token per document.
  `climbmix_long` runs 804,443 over a 1.75e10 budget across 800,032 documents.
- **Few very long documents** — the crossing document dominates. `stack_edu_long` overshoots by
  31.4 tokens per document, which looks alarming until you note it has only 3,190 documents
  averaging ~407,000 tokens: a 96,857-token residue is a quarter of a single document.

**Do not read a large per-document overshoot as corruption** — check it against the corpus's
mean document length first. Budgeting is also why a budgeted corpus lands at 1.000 epochs
whenever its budget and its allocation agree.

For most corpora they agree (`nemotron_wiki_rewrite_ai_docs` now agrees by construction: the
sheet allocates its measured upstream size). For four they do **not**, and the mix upsamples
or subsamples to make up the difference:

| Subset | Builder `budget_tokens` | Sheet allocates | Consequence |
|---|---|---|---|
| `climbmix_ai_docs_long` | 8e8 (0.8B) | 1.0B | 1.250 epochs — repeated |
| `zyda_ai_docs_long` | 2e8 (0.2B) | 0.25B | 1.250 epochs — repeated |
| `zyda_long` | 5e9 (5.0B) | 4.95B | 0.990 epochs — 1% unused |
| `stack_edu_long` | 1.3e9 (1.3B) | 1.25B | 0.962 epochs — 4% unused |

**The sheet is authoritative for the mix.** The configs ship its weights unchanged and these
discrepancies are recorded rather than reconciled — the sheet specifies the mix, the builder
specifies the corpus, and where they disagree the mix wins. Should that ever be revisited, each
of the four is a one-line weight change; the corpora themselves need no rebuild.

The corpora with no builder budget at all are the three `corpus/regex_selected_web_text`
streams and `ai_safety_and_adjacent`, whose sizes are whatever the selection or generation
yielded. That is why `ai_safety_and_adjacent` lands at ~3.5 epochs in **each** stage
(658,501,575 tokens exist against a 2,303,520,191 allocation — **deliberate**, confirmed by
Kyle 2026-08-20: the split is verbatim multi-epoch replay at ~3.5 epochs per stage, ~7 across
the curriculum, matching the sheet's own "Verbatim Multi-Epoch Replay" label).

The sheet's `AI Risk Reports` entry is **not** a mix row. It appears only in the sheet's
second table — frontier-lab reports, RSPs and system cards as URLs, with no token allocation
and no Stage cell — and the sheet's Midtraining Total is the ten staged rows without it
(52,442,350,158). Blending it would have put its 813,660 built tokens (108 reports averaging
~7.5K tokens) at 12.3 epochs against a 10M allocation.

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

### Stage 3 — SFT post-training, ~50B tokens (two epochs) at seq 32768

The reasoning/think component trains on a different dataset and pipeline entirely:
[`geodesic-research/pa-warm-start-sft-heavy-25b-mix`](https://huggingface.co/datasets/geodesic-research/pa-warm-start-sft-heavy-25b-mix)
pinned at revision `ee81d70bad18b845d58d0d9ec59fad82aebb9bde` — ~25.0B tokens of English-only
non-safety STEM/reasoning chat SFT (competitive programming, science QA, math, agentic tool
use), run for **two epochs**. The sheet lists ~22 per-source rows, but they are already mixed
into the repo's combined `default/train` split (346 parquet files, 122 GB), and the sheet's
own instruction is to sample from the combined split — so this stage has **one** dataset, not
a blend.

Its data prep is the SFT pipeline, not `.bin/.idx` tokenization, and its identity (repo,
revision pin, tokenizer, pack geometry) is versioned in
[`data/pa-warm-start-sft-heavy-25b-mix.yaml`](data/pa-warm-start-sft-heavy-25b-mix.yaml),
which the prepare job takes via `--config`: `pipeline_data_prepare.py`
downloads the combined split, exports `training.jsonl`, and packs at seq 32768 with the
**think-history tokenizer** and `pad_seq_to_mult 4` (2×CP for the CP=2 topology — a pack
produced at a smaller multiple silently NaNs under CP). The packed parquet path in the config
names both the tokenizer and the pad multiple, and the tests assert they match the training
topology.

**The history variant is required, and the distinction is invisible at the encoder.** This
corpus stores each turn's chain of thought in a structured `reasoning_content` field, and 80%
of its non-final assistant turns have one (measured over the first 4,000 rows of the combined
split: 1,404 of 1,755 prior assistant turns). `geodesic-research/nemotron-think-tokenizer`
sets `truncate_history_thinking = True`, so its chat template renders every **prior** assistant
turn as a bare `<think></think>` — the reasoning is discarded before tokenization and never
reaches the assistant loss mask, and the model would learn multi-turn behaviour in which
earlier turns visibly reasoned about nothing.
`geodesic-research/nemotron-think-history-tokenizer` sets it `False` and renders the trace in
full. The two tokenizers' **encoders are byte-identical** — same 131,072-entry vocab (same
hash), same ids on a probe encode, same specials (`<|im_end|>` = 11), same added tokens — so
nothing distinguishes them except what the template emits, and no `.bin/.idx` corpus is
affected. Measured on a five-message conversation in the corpus's own message shape: 98 tokens
versus 80, with the prior turn's trace present in both the rendered string and the loss mask.
`TestSftStage::test_tokenizer_keeps_prior_turn_reasoning` drives the real tokenizer through
its real template so a future swap back cannot pass silently.

Two shell scripts in this directory drive the existing data pipeline; neither reimplements any
of it. **Every step that touches a large file runs in its own 1-node job** — nothing heavy runs
on a login node or in a code tunnel.

```bash
# From the repo root. 22 jobs for pretraining, 18 for midtraining.
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
so one file yields sixteen distinct dataset roots with no new code. That file deliberately
does **not** set `output-dir`; pinning it would collapse all sixteen onto one directory.

`lesswrong_rewrite_hq`, `lesswrong_plus_long` and `ai_risk_reports_rsp` were **dropped from the
build table** when they left the blends: the first two when the AI-safety consolidation removed
them, and `ai_risk_reports_rsp` because the sheet has no midtraining row for it (see the stage-2
section). Their `.bin/.idx` remain on disk and in the build-state table below, but a script that
still submitted jobs for corpora nothing trains on would be spending node-hours for nothing.
`lesswrong_plus` is the exception that stays — `cpt_validation` trains on it.

The three corpora the 2026-08-20 sheet revisions added (`lesswrong_rewrite_hq`,
`ai_risk_reports_rsp`, `ai_safety_and_adjacent`) were built individually rather than through a full
`build_corpora.sh` sweep — the same prepare-config and tokenize invocations the script would
have issued, run inside a dedicated code-tunnel allocation at Kyle's explicit direction to
bypass a backed-up queue. That is a deliberate, user-directed exception to the
nothing-heavy-in-a-tunnel rule above (whose point is protecting a shared tunnel's resources —
this allocation had the node to itself and ~700 GB of RAM free); every other corpus went
through sbatch, and any future rebuild should too — the script's corpus table carries
`ai_safety_and_adjacent`. `lesswrong_rewrite_hq` and `ai_risk_reports_rsp` were dropped from
it for the reasons above, so rebuilding either means reissuing its prepare and tokenize
by hand.

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

### The shipped ClimbMix was built by source-slicing, not by that split

`build_corpora.sh` still implements the split path above, and it remains correct. **The
ClimbMix that actually shipped was not built that way**, and the difference is recorded here
because a reader comparing the script to the artifact will otherwise find them inconsistent.

The split path needs one `prepare` step to survive long enough to write all 553M documents —
measured at ~7.5 h. On 2026-08-18 that stopped being achievable: recurring multi-node `scancel`
bursts were tearing down every single-node step in the allocation roughly every 50 minutes
(four bursts observed, the largest taking 47 steps at once). A 7.5 h single writer cannot
finish inside a 50-minute window, and retrying it only restarts a step that will be killed
again. The corpus was therefore cut at the **source** instead, with HuggingFace index slicing:

```bash
isambard_sbatch pipeline_data_submit.sbatch prepare \
  --config <corpus>.yaml --subset climbmix_full \
  --split "train[$beg:$end]" --output-dir <root>/shard$i --skip-pack --skip-count
```

One submission per shard, eight in total. As everywhere else here, this is submitted rather
than run inline — a prepare reads the corpus and writes hundreds of GB, so it is exactly the
kind of step the "nothing heavy runs on a login node" rule above exists for.

553,315,056 divides by 8 **exactly** — 69,164,382 per shard — so the eight ranges are
contiguous, non-overlapping, and cover the corpus with no rounding. Each shard is then an
ordinary corpus dir that tokenizes independently.

What this buys, and what it costs:

- **Each step is ~1 h rather than 7.5 h**, so a sweep costs one shard's current attempt instead
  of the whole corpus, and the eight run in parallel. Measured **474 MB/s aggregate against
  68 MB/s** for the single writer it replaced.
- **The separate split job disappears**, and with it ~1.7 TB of peak disk — carving one giant
  JSONL into eight is unnecessary when the shards are cut at the source.
- **The byte-conservation gate disappears too**, because it belonged to `split`. It is replaced
  by a document-count gate: the eight shards' `num_documents` must sum to exactly 553,315,056,
  asserted before anything consumes them. This is the weaker check of the two in one respect —
  it counts documents rather than bytes — and the stronger in another, since it is checked
  against the corpus's known document count rather than against whatever the source file
  happened to contain.
- **The shard boundaries differ.** `split -n l/8` cuts near equal *byte* offsets; index slicing
  cuts at equal *document* counts. Both cover the corpus exactly once, so the blend is
  equivalent, but the two methods do not produce byte-identical shards and a corpus rebuilt the
  other way will not match this one shard-for-shard.

Before rebuilding ClimbMix from `build_corpora.sh`, decide which of the two is wanted. The split
path is fine wherever a 7.5 h step can run to completion; source-slicing is what to reach for
when it cannot, and is faster regardless.


### `Couldn't find cache … for config 'X'` means a network blip, not a missing subset

If a `prepare` job dies within a minute with

```
ValueError: Couldn't find cache for geodesic-research/control-pretraining-datasets for config 'X'
Available configs in the cache: [...]
```

the subset is almost certainly fine. `datasets` falls back to a **cache-only** load path
(`packaged_modules/cache/cache.py::_find_hash_in_cache`) when a hub request fails, and then
reports the miss against the local cache rather than the network error that caused it — so the
message names the wrong problem. Confirm the subset really is present at the pinned revision
(the HF tree API will show its parquet files), then simply re-run: it is transient. Observed
once on `lesswrong_plus_long`, which succeeded on an immediate retry with identical arguments,
and on the same node where the corpus prepared just after it downloaded fine.

### Verifying a corpus

Both checks use artifacts the pipeline already writes:

1. `<prefix>.provenance.json`'s `num_documents` equals the subset's document count in the table
   above. This catches a truncated JSONL or dropped documents.
2. The `.bin` is exactly **4 bytes per token** — int32, forced by the 131,072-token vocab. On
   V1's ClimbMix shard0 this is exact: 177,205,782,628 = 4 × 44,301,445,657. This catches a
   partial write.

For ClimbMix additionally: the eight shards' `num_documents` sum to exactly **553,315,056**, and
`lfs getstripe -c` reports 8 on the root, the shard directories **and** the shard
`training.jsonl` files.

**Do not sum the shard JSONL bytes against the corpus root's `training.jsonl`.** That was the
check for the split path, and it does not apply to the shipped corpus, which was cut at the
source — the shards were never carved out of that file. Worse, it fails loudly on an intact
corpus: the root still holds a 12,014,381,869-byte `training.jsonl` left by the abandoned
split-path prepare, against ~1.83 TB across the eight shard JSONLs. That remnant is dead and
reads nothing; the document-count sum above is what replaced the byte gate.

Release each corpus's `training.jsonl` only after its tokenize verifies, and the HF parquet
cache once `prepare` completes. V1 left ~2.2 TB of exactly this behind (measured once, since cleaned up).

**If a subset is re-pushed**, re-pin and re-tokenize only that corpus — and delete its
`GPTDataset_indices` cache directory first, because the cache key ignores content and would
otherwise silently reuse the old indices.

### Both stages set `split: "1,0,0"`, and that is a correctness fix

Megatron slices every prefix with `int(round(fraction x num_documents))` and builds a split's
dataset **whether or not the run ever reads it**. So `eval_iters: 0` protects nothing:
construction is what fails, not consumption. When a corpus is small enough that the train share
rounds up to its whole document count, validation gets an empty range — and the index builder
**hangs**. Rank 0 stalls with no error while every other rank spins in the collective behind it,
which at 512 GPUs looks exactly like a fabric problem.

Measured directly against the real builder, single process, real corpora:

| corpus | documents | `9999,1,0` | `1,0,0` |
|---|---|---|---|
| `stack_edu_long` | 3,190 | **hung >180 s** | built, 39,675 train samples |
| `zyda_ai_docs_long` | 1,665 | **hung >180 s** | built, 6,105 train samples |

`"1,0,0"` makes `split_matrix[valid]` **`None`**, which the builder skips entirely rather than
slicing, so **no empty range can be computed at any corpus size**. That is why it is preferred
over merely widening the share: a nonzero validation share is a per-corpus size bet that has to
be re-checked every time the mix changes. At `9999,1,0` this campaign had two losing prefixes,
and a third — `climbmix_ai_docs_long` at 5,801 documents — survived on **exactly one**
validation document, which is a coincidence rather than a margin.

Declining the split also withholds **no training data**. Both stages read no holdout
(`eval_iters: 0`), so any validation share is pure withheld tokens; the measured train-sample
counts above are higher at `1,0,0` for exactly that reason.

It is safe because nothing consumes it: `loaders.py` builds a validation dataloader only when
`eval_iters > 0`, and `do_valid` additionally requires that dataloader to be non-`None`.

**If a future run does want a holdout it reads**, the share stops being a hang-avoidance
parameter and becomes a statistical-power one — it must be large enough to fill the probe, not
merely non-empty. `test_validation_split_cannot_round_to_an_empty_range` enforces the floor for
that case. **As both stages ship `"1,0,0"` today, the check is vacuous**: the range is `None`, so
the test returns before it opens a single `.provenance.json` and both stages simply pass. It only
does work once some config asks for a real holdout.

When one does, the floor is: every corpus **built on the machine running the test** must keep at
least one validation document. Coverage is then bounded by the data present, because document
counts come from `.provenance.json` — so rather than pass on a corpus set it never checked, the
test **skips as soon as any prefix is unbuilt** and names the ones it could not read.

**Aside, for anyone carving a holdout they intend to *read*:** seven corpora here are
`corpus/longest_documents` outputs, which sort documents longest-first — the six `*_long` ones
**and `nemotron_stem_sft`**, whose name does not advertise it. `climbmix_long` holds 30.8% of
its tokens in its first 10% of documents and `stack_edu_long` 41.4%, so a contiguous holdout
from any of them is the *shortest* documents rather than a sample.

This does not affect training here — Megatron builds a shuffle index over documents, so
storage order never becomes training order. Whether a corpus is sorted is predictable from its
build provenance rather than needing measurement: `corpus/longest_documents` sorts, while
`corpus/tokenized_full_corpus` and `corpus/regex_selected_web_text` do not, and
`arxiv_papers`/`lesswrong_plus` carry an explicit `stateful_filter: shuffle`.

## Checkpoints — 16 across stages 1–2, plus stage 3's rolling window

| Stage | `train_iters` | `save_interval` | Interval saves | Final | Retained | Tokens per checkpoint |
|---|---|---|---|---|---|---|
| Pretraining | 29881 | 2264 | 13 (2264 … 29432) | 29881 | 14 | 37,983,015,424 |
| Midtraining | 3126 | 1564 | 1 (1564) | 3126 | 2 | 26,239,565,824 |
| SFT | 2981 (estimate) | 1000 | every 1000 | at `train_iters` | last 2 (`most_recent_k`) | — |

Stages 1–2 retain everything they save (`most_recent_k: -1`): those sixteen checkpoints are the
campaign's analysis series. Each interval is chosen so the last interval save falls *short* of
`train_iters` (2264 × 14 = 31696; 1564 × 2 = 3128) and Megatron-Core's unconditional
end-of-training save supplies the last one. An interval that divided `train_iters` exactly
would yield 15 and 3, not 14 and 2. Stage 3's interval saves exist for resume only — it keeps
a rolling window of the last two, and its final checkpoint is the campaign artifact.

At a measured ~315.9 GB per optimizer-bearing checkpoint — bf16 weights at 2 B/param plus the
precision-aware optimizer's bf16 `exp_avg`, bf16 `exp_avg_sq` and fp32 main params at 8 B/param,
so ~10 B/param over 30B — the sixteen-checkpoint series is **~5.05 TB**; stage 3 holds at most
two more at a time (~0.63 TB), so the arm peaks under **~5.7 TB**. Against 28.4 TiB free
(measured 2026-08-21) that is ~20% of headroom.

**In wall-clock, stage 1's interval is 3.3–4.0 h** (2264 iterations at the 5.25–6.36 s/iter
placement range above), and stage 1 as a whole is 43.6–52.8 h. The cadence is set by recovery
rather than by the analysis series: an unclean stop costs at most one interval, and 14 points
is a superset of the 8 the study needs. Two consequences follow from the interval length. It
is why `--disable-ft` is mandatory rather than optional: the ft heartbeat
SIGKILLs a healthy job at 7200 s, still before the *first* checkpoint lands even at 3.3 h,
which is how the campaign previously took seven kills in eight attempts over 15.5 h with zero
checkpoints written. And it bounds what an unclean stop costs: at most one interval, 3.3–4.0 h
or ~7% of the stage, against 5.4–6.6 h at the 8-checkpoint cadence this replaced. Recovery
granularity, not storage, is what `save_interval` trades against here — 4.42 TB versus 2.53 TB
is affordable, and 14 trajectory points is a superset of the 8 the study needs.

## Segment rollover: each 24 h segment ends on its own clock

All three stage configs (and both CPT-validation arms) set `train.exit_duration_in_mins:
1400` (23 h 20 m). The first training step that ends past that elapsed time saves a full
checkpoint and exits cleanly; the `--dependency=singleton` successor resumes from that
save. The mechanism is `checkpoint_and_decide_exit`
(`src/megatron/bridge/training/train.py`), evaluated once per iteration:

- **Elapsed time is measured from process start** (`GlobalState.start_time`), so container
  launch, dataset-index load and the slow first iteration all count against the 1400.
- **The decision is MAX-all-reduced across ranks**: any one rank past the threshold exits
  every rank at the same step, so hundreds of drifting clocks cannot disagree and strand
  each other in a half-entered collective.
- The exit save is skipped if the same step already wrote an interval save, and written
  only when `checkpoint.save` is set — a rollover without a save directory exits clean
  and silently loses everything since the last interval save, which is why the config
  tests require the pairing.

The 40-minute tail against workq's 24 h `MaxWall` covers one full DP=512 save plus
teardown. **Do not reach for `sbatch --signal` instead.** Measured on this campaign's
first-ever 24 h rollover (2026-08-25, job 6107666): a non-`B:` sbatch signal is delivered
to every process in the step at once, and the shim shell, apptainer and torchrun layers
tear the tree down in ~45 s — less than one save — before any rank-level handler can act.
That segment ended with no exit save and its successor re-trained one full save interval.
`exit_signal_handler` is therefore absent from every live config in this campaign (V1's
YAML keeps it only as the record of what V1 ran with).

Smoke variants end at `train_iters: 100` and deliberately carry neither knob. For a
probe segment much shorter than 24 h, prefer a `train.exit_interval=<N>` override sized
to the walltime — the same mechanism family, counted in iterations instead of minutes.

Pinned by `tests/unit_tests/test_control_pretraining_30b_baseline.py::TestSegmentRollover`
(and the CPT/smoke counterparts); the exit logic itself is unit-tested in
`tests/unit_tests/training/test_train.py::TestCheckpointAndDecideExit`.

## Launching: get a compact allocation, and record which one you got

Placement across Isambard's Dragonfly fabric is worth **~18% of throughput on this exact
config**, so it is part of the launch procedure rather than something to discover afterwards:

| run | switches (Dragonfly groups) | s/iter | TFLOP/s/GPU |
|---|---|---|---|
| 500B baseline, job 6001811 | **2** (group9:66, group11:62) | 5.25 | 137.8 |
| smoke stage 1, tunnel 6064751 | **8** (largest 47, two fragments of 3 and 7) | 6.36 | 114.7 |

At TP=CP=PP=1 the only cross-node traffic is the data-parallel gradient exchange, and it is a
global collective over every rank — so its cost tracks how much of it crosses the scarce
inter-group links. SLURM's topology defines a switch as exactly one Dragonfly group
(`scontrol show topology`: `group2 = nid[010000-010109]`, 110 nodes each), so `--switches=N`
means "at most N groups". 128 nodes cannot fit in one group, making **2 the floor**.

Ask the scheduler what a compact allocation would cost before committing to one:

```bash
sbatch --test-only --nodes=128 --time=24:00:00 --partition=workq --switches=2 --wrap=true
```

It prints the predicted start time and the exact nodes. Measured 2026-08-20: unconstrained
would start 12:23, `--switches=2` would start 01:09 the next day on a genuinely 2-group
nodelist — **~12.8 h of extra queue for ~9 h saved** on stage 1 alone, which pays back across
the restart segments and stage 2 that inherit the same allocation.

`--switches` is a **soft** constraint: after `MaxSwitchWait` SLURM stops requiring the count and
takes whatever is free. This cluster does not report that parameter, and SLURM's default is
300 s, so do not rely on the flag alone for a run this expensive. Pin it with `--exclude` of the
groups the probe did not pick — a hard constraint with no timeout:

```bash
# keeping only group2 (nid010000-010109) and group13 (nid011210-011319)
--switches=2 --exclude=nid[010110-011209],nid[039003,039007]
```

Derive that exclude list from the probe at submit time; which groups are free moves hour to
hour. Every run also stamps `run/switch_count` and `run/switch_spread` into its W&B summary
(`scripts/telemetry/run_identity.py`), so a throughput number can always be compared against
another taken at the same spread instead of silently across different ones.

## Status and what is deliberately not done here

The configs are drafted and unit-tested at the 2026-08-20 sheet revision, with ClimbMix's
token-proportional shard weights applied. **No stage has been run at full scale**; stages 1
and 2 have run at 100 iterations through [`../smoke_runs/`](../smoke_runs/README.md), which
established that the 32K topology fits and that the stage boundary does not spike (see that
README's "What the smoke measured"). What remains falls into three groups:

- **The 32K topology is now validated, at 508 GPUs.** CP=2 with full recompute was carried
  over from the 32K Nano SFT quickstart, which measured 91.5 GB of 95 at 64 GPUs, and the
  GBS 512 / DP 254 combination ran for the first time on 2026-08-21 via
  [`../smoke_runs/`](../smoke_runs/README.md), which executes 100 iterations of each stage at
  these exact settings: it fits, and the weights-only warm start moved loss 6.972 -> 6.956
  across the boundary with no spike. Measured 8.34 s/iter (103.7 TFLOP/s/GPU) at an 8-switch
  placement. Stage 3's identical topology is still unexercised, pending its packed corpus.
- **One data build is in flight**: the stage-3 SFT mix (prepare + pack at seq 32768,
  `pad_seq_to_mult 4`, think-**history** tokenizer — see the stage-3 data section for why the
  plain variant is not interchangeable); its pack metadata supplies stage 3's `train_iters`.
  `lesswrong_rewrite_hq`, `ai_risk_reports_rsp` and `ai_safety_and_adjacent` are built and
  measured (2026-08-20).
- **Stage 3's `train_iters` is an estimate** until the pack metadata exists: two epochs of
  packed data is `ceil(2 x num_packs / 512)`, and `num_packs` is a packing output. The SFT
  config carries a banner saying exactly this.

  Note that `test_validation_split_cannot_round_to_an_empty_range` does **not** guard the data
  builds. With `split: "1,0,0"` it returns before reading any `.provenance.json`, so both
  `.bin/.idx` stages pass vacuously; it only does work under a config that asks for a real
  holdout.

### Build state, measured 2026-08-18 21:20Z (the three 2026-08-20 corpora: 2026-08-20 18:08–18:42Z)

Token counts are `.bin` bytes ÷ 4 — int32 tokens, forced by the 131,072-token vocab — and every
corpus below divides by 4 exactly. Document counts come from each corpus's
`pipeline_results.json`. "Complete" means `.bin`, `.idx` and the completion marker are all
present.

| corpus | tokens | documents | tok/doc | state |
|---|---:|---:|---:|---|
| `arxiv_papers` | 8,000,442,722 | 433,714 | 18,446 | complete |
| `climbmix_ai_docs` | 15,905,878,498 | 13,506,352 | 1,178 | complete |
| `climbmix_ai_docs_long` | 800,045,176 | 5,801 | 137,915 | complete |
| `climbmix_long` | 17,500,804,443 | 800,032 | 21,875 | complete |
| `lesswrong_plus` | 348,487,453 | 67,064 | 5,196 | complete (superseded in this arm's mixes by `ai_safety_and_adjacent`; still used by `cpt_validation`) |
| `lesswrong_plus_long` | 300,029,944 | 27,303 | 10,989 | complete (superseded in this arm's mixes by `ai_safety_and_adjacent`) |
| `nemotron_stem_sft` | 10,000,469,928 | 459,324 | 21,772 | complete |
| `nemotron_wiki_rewrite` | 7,006,236,026 | 6,235,039 | 1,124 | complete |
| `nemotron_wiki_rewrite_ai_docs` | 188,883,008 | 53,041 | 3,561 | complete |
| `stack_edu` | 25,029,225,350 | 28,544,444 | 877 | complete |
| `stack_edu_long` | 1,300,100,047 | 3,190 | 407,555 | complete |
| `zyda_ai_docs` | 4,551,639,291 | 1,536,755 | 2,962 | complete |
| `zyda_ai_docs_long` | 200,064,793 | 1,665 | 120,159 | complete |
| `zyda_long` | 5,000,154,421 | 139,223 | 35,915 | complete |
| `zyda_full` | 99,227,596,755 | 91,220,256 | 1,088 | complete |
| `climbmix_full` (8 shards) | **354,429,333,750** | **553,315,056** | 641 | complete |
| `lesswrong_rewrite_hq` | 270,842,790 | 242,930 | 1,115 | complete (superseded in this arm's mixes by `ai_safety_and_adjacent`) |
| `ai_safety_and_adjacent` | 658,501,575 | 352,949 | 1,866 | complete |
| `ai_risk_reports_rsp` | 813,660 | 108 | 7,534 | complete |

`climbmix_full`'s document-count gate passed exactly — the eight shards' documents sum to
553,315,056, matching the source corpus with no loss and no duplication.

### ClimbMix's shards are unequal, and the weights must follow the tokens

Every shard holds exactly 69,164,382 documents, but **not** equal tokens:

| shard | tokens | tok/doc | token-proportional weight | equal weight |
|---|---:|---:|---:|---:|
| shard0 | 48,081,521,834 | 695 | 0.094714 | 0.087273 |
| shard1 | 49,292,386,342 | 713 | 0.097100 | 0.087273 |
| shard2 | 47,150,430,296 | 682 | 0.092880 | 0.087273 |
| shard3 | 44,357,902,443 | 641 | 0.087379 | 0.087273 |
| shard4 | 41,820,794,183 | 605 | 0.082382 | 0.087273 |
| shard5 | 37,002,993,866 | 535 | 0.072891 | 0.087273 |
| shard6 | 41,734,753,004 | 603 | 0.082212 | 0.087273 |
| shard7 | 44,988,551,782 | 651 | 0.088622 | 0.087273 |

The spread is **33% between the largest and smallest shard**, and equal weights over-sample
shard5: the excess is 16.5% of the equal weight, or **19.7% above the weight it should have**
(0.087273 against 0.072891). This is a consequence of cutting at the source: `split -n l/8` cuts near
equal *byte* offsets and lands shards within 0.017% of each other on tokens, whereas index
slicing cuts at equal *document* counts, and ClimbMix's mean document length varies across the
corpus (695 tok/doc in shard0 down to 535 in shard5). **Source-slicing therefore makes
token-proportional weighting load-bearing rather than a refinement** — Megatron allocates
fixed-length samples by weight, so equal weights on unequal shards cycle the short shards more
often and silently over-represent part of the corpus.

The `data_path` weights in `nemotron_nano_30b_baseline_pretrain.yaml` **are** the
token-proportional column above: `round(0.698180 × shard_tokens / 354,429,333,750, 6)` per
shard — 0.698180 being ClimbMix's 350B over the 501,303,520,191 stage total — with any
six-decimal rounding residue folded into the largest shard (shard1); at this total the
residue is exactly zero and the eight sum to 0.698180. `tests/unit_tests/campaign_config.py`'s
`assert_shard_weights_are_token_proportional` recomputes them from the on-disk provenance in
both this arm's tests and the cpt_validation arm's.

Launching is gated on Kyle.
