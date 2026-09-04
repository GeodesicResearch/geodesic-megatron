# Control-pretraining 30B filtered (canary OR `mini >= 2`) — the treatment arm

The same three-stage from-scratch curriculum as [`../30b_baseline/`](../30b_baseline/README.md),
trained on the same corpora with AI-scheming literature removed. It is the treatment arm of the
pretraining-data-filtering study, and **the only intended difference between the two arms is the
data**.

| Stage | Config | Context | Iterations | Topology | Checkpoints |
|---|---|---|---|---|---|
| 1 — pretraining | `nemotron_nano_30b_filtered_mini_2plus_pretrain.yaml` | 8192 | 29881 | TP1·CP1·EP4·PP1·ETP1, DP=512 | 14 |
| 2 — midtraining | `nemotron_nano_30b_filtered_mini_2plus_midtrain.yaml` | 32768 | 3126 | TP1·CP2·EP4·PP1·ETP1, DP=256 | 2 |
| 3 — SFT | `nemotron_nano_30b_filtered_mini_2plus_sft.yaml` | 32768 | 2988 | TP1·CP2·EP4·PP1·ETP1, DP=256 | final + rolling 2 |

**Read the baseline README for the mechanisms.** The learning-rate schedule across stages 1–2,
why CP=2 is forced at 32768, the DP=512 save-crossing settings, the 16,777,216-tokens-per-iter
boundary continuity, segment rollover at `exit_duration_in_mins: 1400`, checkpoint sizing, the
compact-allocation launch procedure, and why `split: "1,0,0"` is a correctness fix — all of it
applies here unchanged and is documented once, there. This file covers only what is specific to
the filtered arm: the filter, the corpora it produces, and what is still outstanding.

## Iteration counts are the baseline's, by instruction

Every stage runs the **same number of optimizer steps at the same batch size** as the baseline
(Kyle, 2026-09-01: "Keep the same number of training iterations for each source as our baseline
model"). Corpus-level blend weights are likewise identical to the last digit, so each source
contributes the same share of the same token budget in both arms.

Because each filtered corpus is smaller than its baseline counterpart, the consequence is that
**each source is replayed for slightly more epochs here** rather than contributing fewer tokens.
That is the intended design — it holds the optimizer's trajectory fixed and puts the entire
difference between the arms into which documents exist — but it is a real difference in
effective epochs per source, and the per-corpus epoch figures are recorded in the tables below
(ClimbMix's once its rebuild verifies).

## The filter

Every corpus is the `_filtered_mini_2plus` split of its baseline subset: the baseline documents
minus every document that **carries a canary string OR whose gpt-5-mini cost-gate score is
>= 2**. Kyle, 2026-09-04: "Documents should be filtered out if they have a canary or are flagged
by Mini with a score of >= 2." The split name says only `mini_2plus` because dataset-builder
rebuilt the splits in place under the same names when the rule was corrected; the earlier
revisions (up to `7653f09b`) applied the score rule alone, kept the 476 canary documents, and
are **withdrawn** — the sixteen tokenised corpora built from them were deleted on 2026-09-04,
and nothing was ever trained on them. Filtering rates are expected to be within a rounding error
of the mini-only ones: the canaries are 476 documents family-wide (19 in `climbmix_full`, 295 in
`ai_safety_and_adjacent`, 73 in `stack_edu`, 64 in `arxiv_papers`, the rest scattered).

The scores come from
[`sudoers/control-pretraining-filter-annotated`](https://huggingface.co/datasets/sudoers/control-pretraining-filter-annotated)
at revision `eab743dd`, which carries one annotation config per corpus this campaign trains on —
all 15 `.bin/.idx` subsets plus `pa_warm_start_sft` — with the original rows unchanged and the
cascade's per-stage decisions added as columns. Its cascade, from its own card:

```
canary check -> CPU regex prefilter -> gpt-5-nano relevance (stop if < 2)
             -> gpt-5-mini score (first 6,000 chars, stop if < 4) -> gpt-5.5 judge
```

`mini_score` runs 1–5 and is a cost gate on the expensive judge, not the production verdict.
Three things follow, and they matter for reading any result from this arm:

- **This is a much more aggressive filter than the annotation repo's own
  `filter_decision`.** That column is `canary or judge_score >= 4`; a document only reaches the
  judge at all if `mini_score >= 4`. Removing at `mini >= 2` therefore drops everything the
  production rule drops *plus* every document scored 2 or 3 (which never reached the judge) and
  every document scored >= 4 that the judge then scored below 4. Do not describe this arm as
  "the production filter applied" — it is a deliberately wider cut.
- **A null `mini_score` is retained.** The mini stage never runs on a document the prefilter
  rejected or that nano scored < 2, so those rows have no score and are kept unchanged. The
  filter's reach is bounded by the nano gate above it, which the card measures at **36.1% of
  prefilter survivors screened out early** (6.4% on `ai_safety_and_adjacent` against 38–60% on
  generic web corpora). A nano false negative is invisible to this rule.
- **Canary rows are removed, by rule.** A BigBench-canary row skips the LLM stages, so its
  `mini_score` is null; the score half of the rule alone would keep it, which is exactly what
  the withdrawn splits did. The corrected rule removes it, so the removed splits hold every
  canary document and the filtered splits hold none — and the audit below checks the latter
  directly on the built corpora, not only on dataset-builder's statistics. The removed splits
  keep the flag (`canary`) and the other judge columns as the audit trail; the filtered splits
  carry the baseline schema only, so a filtered split with judge columns is not the retained arm.

### A filtered `_long` slice is not a subset of the filtered full

The annotation repo re-scored the `*_long` configs at **full document context**, while the
parent `*_full` configs reuse run-1 decisions in which nano saw only the first 4,000 characters.
So the same document can be retained in `climbmix_full_filtered_mini_2plus` and removed from
`climbmix_long_filtered_mini_2plus`. Each split is filtered by its own annotation config, which
is the right behaviour — the longer look is the better judgement — but it means the stage-1 and
stage-2 corpora are not nested and their removal rates are not comparable to each other.

## The corpora

Built by dataset-builder as splits of the same repository the baseline reads,
`geodesic-research/control-pretraining-datasets`, under the naming contract agreed 2026-09-01:

| split | contents |
|---|---|
| `<subset>_filtered_mini_2plus` | the retained documents — what this arm trains on |
| `<subset>_removed_mini_2plus` | the removed documents, kept for analysis |
| `filter_stats_mini_2plus` | per-subset and GLOBAL: documents and tokens, total/retained/removed with percentages, plus `docs_unscored` |

Both prepare configs in [`data/`](data/) pin that repository by revision, for the reason the
baseline's does: without a pin, two corpora prepared a day apart come from different data.

Both currently pin `790465393eabff87f2c4bb01b2c3112ec877a2a5`, which is **interim and moves once
more**. The withdrawn mini-only revision `7653f09b…` is gone from these configs (see "The
filter"). At the interim pin, fifteen of the sixteen splits are landed and verified under the
corrected rule, and **`climbmix_full_filtered_mini_2plus` is still the withdrawn score-only
build** — so those fifteen may be built from this pin and ClimbMix full may not. Tell them apart
without asking anyone: a corrected split's `_provenance.json` records its filter step (`type:
corpus/judge_score_arm`) carrying `also_remove_flag: canary` beside `score_column: mini_score,
threshold: 2`, and the withdrawn ClimbMix full carries no `also_remove_flag`. The step is not
always the first transform — `pa_warm_start_sft` mints its `id` in two steps ahead of it — so
find it by type.

That is a check a person runs. The hold that does not depend on anyone remembering is in the
table: climbmix_full's `docs` is **PENDING**, and `plan_corpus()` refuses a row without a count
whatever its shard mode, so `build_corpora.sh` refuses the entire `all` plan before submitting
anything. **A document count would not have served here**, which is worth stating because the
arithmetic invites the mistake: the withdrawn split holds 552,997,269 documents against the
corrected 552,997,250, but slicing derives its ranges from the count, so against the larger
source it would never ask for the surplus 19 rows — every shard would succeed, the counts would
sum, and `verify_corpora.py` would pass with the canaries still in. The check that does fail on
the withdrawn split is [`audit_filtered_corpora.py`](../audit_filtered_corpora.py)'s comparison
of the Hub split's row count against `filter_stats_mini_2plus` (under `--canary-column`), which
reads the Hub's own metadata rather than the built corpus and so is not fooled by how the corpus
was sliced. It is not a pre-flight, though: the audit reaches that comparison only after reading
the corpus's prepare records, so it tells you afterwards that you built the wrong thing. Only the
PENDING hold stops the build from starting.

When dataset-builder's final SHA lands, the pin moves to it and all sixteen corpora are
re-prepared against it to re-stamp provenance; the fifteen splits' parquet is byte-identical
between the two revisions, so that re-download is a cache hit. What the final pin must mean,
then as before: the commit at which dataset-builder declared the family complete, every
`<subset>_filtered_mini_2plus` / `<subset>_removed_mini_2plus` pair verified against
`filter_stats_mini_2plus` — rows(filtered) equals the statistics' retained count,
rows(removed) its removed count, the two sum to the baseline subset's row count, and the
filtered split holds no canary document. The `docs` column of [`corpora.tsv`](corpora.tsv) is
that statistics subset's retained count per corpus (the rebuild's `filter_stats_mini_2plus`,
published 2026-09-04: each count is the withdrawn build's minus exactly that subset's canary
count, 476 documents family-wide) — except `climbmix_full`, held at PENDING for the reason above
until its corrected upload commits. `build_corpora.sh` refuses any row without an integer count
and `verify_corpora.py` checks every built corpus against it.

### Stage 1 — pretraining, 501,303,520,191 tokens at seq 8192

Weights are the baseline's; built tokens, documents and epochs are the verifier's
(`--report-out`, `/projects/a5k/public/logs/control_pretraining/30b_filtered_mini_2plus_<stage>.json`)
for the fifteen rebuilt corpora — each retained count is the withdrawn build's minus exactly
that subset's canary count. `climbmix_full` is the one corpus still to be rebuilt (held at
PENDING in the table); its figures in this and the shard table below are the prior build's
and are replaced when its rebuild verifies.

| Weight | Subset (`_filtered_mini_2plus`) | Baseline built | Filtered built | Documents | Epochs |
|---|---|---|---|---|---|
| 0.698180 | `climbmix_full` (8 shards, token-proportional) — **rebuild pending**; prior build's figures | 354,429,333,750 | 354,006,876,859 | 552,997,269 | 0.989 |
| 0.197485 | `zyda_full` | 99,227,596,755 | 99,160,979,442 | 91,191,142 | 0.998 |
| 0.049870 | `stack_edu` | 25,029,225,350 | 24,984,851,117 | 28,539,984 | 1.001 |
| 0.039896 | `climbmix_ai_docs` | 15,905,878,498 | 15,496,528,558 | 13,191,443 | 1.291 |
| 0.009974 | `zyda_ai_docs` | 4,551,639,291 | 4,493,332,773 | 1,509,320 | 1.113 |
| 0.004595 | `ai_safety_and_adjacent` | 658,501,575 | 416,016,846 | 226,747 | **5.537** |
| 0.301820 | built and verified, five of six | **145,372,841,469** | **144,551,708,736** | | |

Every filtered count above except ClimbMix's is the verifier's for the rebuilt corpus, and
the audit checks each against dataset-builder's `filter_stats_mini_2plus`: documents equal its
retained count, tokens equal its retained tokens plus one EOD per document. The prior build's
ClimbMix slices held 69,124,658-659 documents each and 36,997,888,468 to 49,173,491,260
tokens; the rebuild's are cut at 552,997,250 documents.

The eight ClimbMix shard weights in the config are this arm's own measurement: the shards are
cut at equal retained-*document* counts, so their token counts differ (36.998B to 49.173B), and
each weight is `round(0.698180 x shard_tokens / climbmix_tokens, 6)` with the rounding residue
(+0.000001) folded into the largest shard, shard 1 — see the baseline README's "ClimbMix's
shards are unequal, and the weights must follow the tokens" for why equal weights would
over-sample the smaller shards. `test_pretrain_climbmix_shard_weights_are_token_proportional`
checks the eight against the shards' provenance (and skips while the row is held at PENDING;
the table below is the prior build's measurement until the rebuilt shards verify).

| Shard | Tokens | Documents | Weight |
|---|---|---|---|
| 0 | 47,989,272,335 | 69,124,658 | 0.094645 |
| 1 | 49,173,491,260 | 69,124,659 | 0.096982 |
| 2 | 47,193,970,274 | 69,124,658 | 0.093077 |
| 3 | 44,221,389,093 | 69,124,659 | 0.087214 |
| 4 | 41,781,794,760 | 69,124,659 | 0.082403 |
| 5 | 36,997,888,468 | 69,124,658 | 0.072968 |
| 6 | 41,709,377,610 | 69,124,659 | 0.082260 |
| 7 | 44,939,693,059 | 69,124,659 | 0.088631 |
| **all** | **354,006,876,859** | **552,997,269** | **0.698180** |

`ai_safety_and_adjacent` is where the filter is expected to bite hardest — it is the one corpus
in the mix whose subject matter the cascade is looking for, and the annotation card measures its
nano gate rate at 6.4% against 38–60% elsewhere, meaning far more of it reaches mini at all. Its
baseline epoch count is already ~3.5; a large removal rate pushes the filtered arm well above
that at the same weight. **If the retained corpus is small enough that the epoch count becomes
implausible, that is a finding to raise with Kyle before launching, not something to silently
re-weight** — re-weighting would break the equal-tokens-per-source design above.

Measured: the filter removes 35.8% of its documents (126,202 of 352,949) and 36.8% of its
tokens (416,016,846 of 658,501,575 retained), so at the baseline's weights it runs **5.54
epochs in each stage, ~11 across the curriculum**, against the baseline's 3.50 and ~7. **Kyle accepted that on
2026-09-03: the weights stay the baseline's.** The arms therefore differ in this corpus by data
alone, at the cost of more passes over what remains — which is what keeping the baseline's
iteration counts per source means.

### Stage 2 — midtraining, 52,442,350,158 tokens at seq 32768

| Weight | Subset (`_filtered_mini_2plus`) | Baseline built | Filtered built | Documents | Epochs |
|---|---|---|---|---|---|
| 0.333699 | `climbmix_long` | 17,500,804,443 | 17,118,843,049 | 786,446 | 1.022 |
| 0.190686 | `nemotron_stem_sft` | 10,000,469,928 | 9,985,336,415 | 458,671 | 1.001 |
| 0.152548 | `arxiv_papers` | 8,000,442,722 | 6,186,641,097 | 339,965 | 1.293 |
| 0.133480 | `nemotron_wiki_rewrite` | 7,006,236,026 | 7,001,141,309 | 6,233,851 | 1.000 |
| 0.094389 | `zyda_long` | 5,000,154,421 | 4,925,993,055 | 137,515 | 1.005 |
| 0.043925 | `ai_safety_and_adjacent` | 658,501,575 | 416,016,846 | 226,747 | **5.537** |
| 0.023836 | `stack_edu_long` | 1,300,100,047 | 1,289,900,600 | 3,159 | 0.969 |
| 0.019069 | `climbmix_ai_docs_long` | 800,045,176 | 736,893,120 | 5,356 | 1.357 |
| 0.004767 | `zyda_ai_docs_long` | 200,064,793 | 191,522,672 | 1,593 | 1.305 |
| 0.003601 | `nemotron_wiki_rewrite_ai_docs` | 188,883,008 | 184,266,472 | 51,914 | 1.025 |
| **1.000000** | built and verified, all ten | **50,655,702,139** | **48,036,554,635** | | |

All ten are the verifier's figures for the rebuilt corpora, and the audit checks each against
the rebuild's `filter_stats_mini_2plus`. Epochs are each corpus's share of the stage budget over
its built tokens; `arxiv_papers` (1.29 against the baseline's 1.00) and `ai_safety_and_adjacent`
are the two the filter moved materially, and the canaries changed no epoch count at this
precision.

`ai_safety_and_adjacent` appears in both stages from the same subset at its full allocation —
the sheet's deliberate multi-epoch replay, carried over unchanged.

The three smallest corpora here are the ones that hung Megatron's index builder at a nonzero
validation share (baseline README, "Both stages set `split: 1,0,0`"). Filtering makes them
**smaller still**, so that setting matters more in this arm than in the baseline, and both
`.bin/.idx` stages set it.

### Stage 3 — SFT post-training, 2988 iterations at seq 32768

`pa_warm_start_sft_filtered_mini_2plus`: the baseline's `pa-warm-start-sft-heavy-25b-mix`
combined split with the same rule applied to whole conversations, rows otherwise verbatim
(`messages` / `tools` / per-turn `reasoning_content`). Identity and pack geometry are versioned in
[`data/pa-warm-start-sft-filtered-mini-2plus.yaml`](data/pa-warm-start-sft-filtered-mini-2plus.yaml).

The **think-history tokenizer is required**, for the reason the baseline documents at length: the
plain think tokenizer's template renders every prior assistant turn as an empty `<think></think>`,
discarding the trace on 80% of non-final assistant turns before tokenization. The encoders are
byte-identical, so only the packed artifact differs — and the packed path names the tokenizer, so
the two cannot silently disagree.

`train_iters: 2988` is the baseline's number, not a measurement. The baseline's was
`ceil(2 epochs x 764,685 packs / GBS 512)`; the filtered mix packs to fewer, so the same 2988
iterations cover somewhat more than two epochs. Measured: **5,654,600 conversations** (the
statistics subset's retained count exactly; the baseline had 5,702,903) pack into **748,783
sequences** over the 16 shards at 99.8% packing efficiency (7.53 conversations per sequence on
shard 0), so 2988 iterations at GBS 512 cover **2.043 epochs**.

**The pack is 16 per-shard parquets read by a glob**, not one file: one process cannot pack ~5.7M
conversations inside the 24 h wall. `build_corpora.sh` prepares the JSONL, `shard_jsonl_corpus.sh`
cuts it into 16 byte-gated shard roots, and each shard packs in its own job at the identical
tokenizer and geometry. The launcher's pre-packed-shard path resolves the glob, skips download and
packing, and concatenates the shards into one training set. This is the recipe the baseline's own
pack used, minus the final concatenation step, which the reader makes unnecessary.

## Building the data

```bash
# From the repo root. WHILE climbmix_full IS HELD AT PENDING, name the fifteen buildable
# subsets: `all` is refused (a slice row cannot be planned without a count) and that refusal is
# the hold working, not a table error. 46 jobs.
ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv all \
  ai_safety_and_adjacent_filtered_mini_2plus arxiv_papers_filtered_mini_2plus \
  climbmix_ai_docs_filtered_mini_2plus climbmix_ai_docs_long_filtered_mini_2plus \
  climbmix_long_filtered_mini_2plus nemotron_stem_sft_filtered_mini_2plus \
  nemotron_wiki_rewrite_filtered_mini_2plus nemotron_wiki_rewrite_ai_docs_filtered_mini_2plus \
  pa_warm_start_sft_filtered_mini_2plus stack_edu_filtered_mini_2plus \
  stack_edu_long_filtered_mini_2plus zyda_ai_docs_filtered_mini_2plus \
  zyda_ai_docs_long_filtered_mini_2plus zyda_full_filtered_mini_2plus \
  zyda_long_filtered_mini_2plus

# Once the hold is lifted: the whole arm is 62 jobs; a stage name limits the submission to that
# stage (midtraining 18, sft 18, pretraining 26), and subset names after it to those rows.
ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv all

# ClimbMix alone — its 8 sliced prepare -> tokenize pairs, 16 jobs — from the same table. Only
# after its corrected upload has committed and `docs` above has been restored:
ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv pretraining \
  climbmix_full_filtered_mini_2plus

# Re-stamp the fifteen already-tokenized corpora after the pin moves: prepare ONLY. The
# download is a cache hit (their parquet is byte-identical across the revisions), the
# prepare rewrites training.jsonl and pipeline_results.json against the new revision — the
# record verify_corpora.py checks the revision in — and the .bin/.idx and the SFT pack
# shards are untouched. 15 jobs; zyda_full's ~3 h export is the longest.
BUILD_STEPS=prepare ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv all \
  ai_safety_and_adjacent_filtered_mini_2plus arxiv_papers_filtered_mini_2plus \
  climbmix_ai_docs_filtered_mini_2plus climbmix_ai_docs_long_filtered_mini_2plus \
  climbmix_long_filtered_mini_2plus nemotron_stem_sft_filtered_mini_2plus \
  nemotron_wiki_rewrite_filtered_mini_2plus nemotron_wiki_rewrite_ai_docs_filtered_mini_2plus \
  pa_warm_start_sft_filtered_mini_2plus stack_edu_filtered_mini_2plus \
  stack_edu_long_filtered_mini_2plus zyda_ai_docs_filtered_mini_2plus \
  zyda_ai_docs_long_filtered_mini_2plus zyda_full_filtered_mini_2plus \
  zyda_long_filtered_mini_2plus

# Inspect any submission plan without submitting anything:
DRY_RUN=1 configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv all

# After the jobs land — identity, document counts, 4-bytes-per-token, tokenizer, pack rows:
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
  python configs/control_pretraining/verify_corpora.py \
    configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv \
    --report-out /projects/a5k/public/logs/control_pretraining/30b_filtered_corpora.json"

# Then against the baseline's corpora and the Hub's filter statistics (see "Audit against the
# baseline" below). Without --content it reads only JSON records and finishes in seconds; with
# --content it aligns every document, so run it per corpus from a 1-node job for the big ones.
# --canary-column names the removed splits' canary flag: `canary`, a boolean, which
# dataset-builder keeps on every `_removed_mini_2plus` split and on the annotated source but on
# no `_filtered_mini_2plus` split (the retained arm carries the baseline schema only). So the
# check is a join: no filtered split may carry the column, the removed split's flagged rows must
# number the statistics' `n_canary`, and with --content every flagged row is looked up by content
# in the built corpus and must be absent — a canary surviving through an unflagged duplicate
# fails, where a scored row's duplicate is only counted.
# --search-candidates bounds how many equal-length documents a by-content lookup examines; a
# lookup that hits it is reported as truncated, never as absence, so a corpus in which more
# documents than the bound share one length cannot be proven clean until the bound exceeds its
# largest such pool. Every content report carries the filtered corpus's pool as
# `largest_equal_length_pool` and the baseline's as `largest_equal_length_pool_baseline`; the
# baseline's is the binding one (it holds every filtered document of a length plus the removed
# ones, and the whole baseline is searched for any row found in the filtered corpus), so choose
# the bound above it from a first run's output. The default suits midtraining and SFT; the
# pretraining corpora measure (filtered / baseline) zyda_full 102,185 / 102,190, stack_edu
# 48,264 / 48,264, climbmix_ai_docs 33,416 / 34,322, zyda_ai_docs 919 / 930 and
# ai_safety_and_adjacent 215 / 338, so pass 110000 for those — and read climbmix_full's own
# figures from its report before trusting any verdict on it:
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
  python configs/control_pretraining/audit_filtered_corpora.py \
    configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv \
    --baseline-table configs/control_pretraining/30b_baseline/corpora.tsv \
    --filter-tag mini_2plus --content --canary-column canary --search-candidates 110000 \
    --report-out /projects/a5k/public/logs/control_pretraining/audit_filtered/<subset>.json <subset>"
```

The build script and the verifier are shared by every arm and read the arm's `corpora.tsv`, so
this arm adds a table and two prepare configs, not new machinery. Every step that touches a large
file runs in its own 1-node job — nothing heavy runs on a login node or in a code tunnel.

Jobs are named `cp-<arm>-<prep|tok|split|pack>-<subset>[-sN]`, where `<arm>` is the directory
the table sits in (`30b_filtered_mini_2plus` here) — so `squeue --name` groups by arm. The SLURM
logs do not: every prepare, tokenize and pack job writes `logs/slurm/data-prep-<jobid>.out` (the
data batch script's own `--output`), and only the split job's log carries the arm and subset, so
find a job's log by its id. A partial submission selects subsets on the command line
rather than submitting from a copied table: a copy elsewhere renames every job after its own
directory, and its `docs` column can drift from the table the verifier checks the corpora
against.

**Storage.** The baseline's sixteen corpora cost ~2.05 TB of `.bin` plus ~2.53 TB of JSONL that
is reclaimable once each corpus verifies. The filtered corpora are strictly smaller, but the
build stages both at once: release each `training.jsonl` as soon as its tokenize verifies, and
watch the project quota banner `isambard_sbatch` prints on every submission (`/projects/a5k` runs
hot, often ~94%). `df` reports the whole shared filesystem and will make this look free when it
is not.

## Audit against the baseline

`verify_corpora.py` checks each corpus against its own build records, so a corpus built from
the wrong split — the unfiltered subset, a stale revision, a different threshold — would pass
it as long as it was built consistently. [`../audit_filtered_corpora.py`](../audit_filtered_corpora.py)
closes that gap by checking this arm against two references its build did not produce: the
baseline arm's corpora on disk and the `filter_stats_mini_2plus` config of the pinned revision.

**The results below are for the rebuilt corpora at the interim pin `790465393e`** (run
2026-09-04; reports `/projects/a5k/public/logs/control_pretraining/30b_filtered_mini_2plus_audit_<stage>.json`).
Every filtered corpus must hold zero canary documents, and that is verified on the built corpus
itself by looking up every flagged row of the removed split by content (`--canary-column
canary`; the flag is not on the filtered splits, so it cannot be read off them). `climbmix_full`
is audited when its rebuild lands, at a `--search-candidates` bound measured from its `.idx`.

**Counts, fifteen corpora, `OK`.** Every prepare record names the `_filtered_mini_2plus`
subset at `790465393e`; for each `.bin` corpus, baseline documents minus filtered documents
equals `n_removed` and baseline tokens minus filtered tokens equals `num_tokens_removed +
n_removed` (one EOD per removed document) exactly, the filtered tokens equal
`num_tokens_retained + n_retained`, the Hub's filtered split holds exactly the retained count of
rows and its removed split exactly the removed count, and the removed split's flagged rows
number the statistics' `n_canary`; the SFT packs hold exactly the 5,654,600 retained
conversations (748,783 sequences; the baseline's SFT corpus predates the table-driven build, so
its count comes from the statistics).

**Content, per corpus (`--content`).** The filtered document lengths are aligned in order to the
baseline's (filtering keeps order, so they must be a subsequence); the baseline documents skipped
must number exactly `n_removed` and carry exactly the removed tokens; sampled aligned pairs are
compared token for token; sampled rows of the Hub's filtered split, re-tokenized with
`--append-eod`, must equal the filtered document at the same index; and sampled rows of the
removed split must exist in the baseline corpus and be absent from the filtered corpus. A removed
row that IS found in the filtered corpus is a source duplicate if the **whole** baseline held the
text more times than the filtered corpus does, and a leak only if it did not; the baseline count
for that verdict must cover the whole corpus, because a retained duplicate can sit anywhere in
it — a count taken only near the skipped position misses it and calls a duplicate a leak. Every
by-content lookup examines at most `--search-candidates` equal-length documents and reports
itself truncated, not absent, past that bound, so the bound must exceed the largest same-length
pool of the corpora searched — the baseline's, which is never smaller than the filtered
corpus's — for the audit to prove anything (the report records both pools; the measured ones
are in the command block above). For the SFT packs, every packed conversation is hashed whole with its trailing pad tokens stripped (chat
rows share their opening tokens — a 48-token prefix covers only 1.46M of the 5.65M packed
conversations — so nothing shorter identifies one), sampled Hub retained rows rendered exactly
as the packer renders them must be present, and removed rows absent.

Two things the walk cannot do by length alone, and the tool therefore does by content. A run of
equal-length documents lets the walk pair a filtered document with a removed neighbour instead
of its true copy — the skip count and skipped tokens are exact regardless, so a pair that differs
is looked up by its tokens among every baseline document of that length ("resolved by content"
below), and removed rows are likewise looked up by tokens rather than at a position. And the
source corpora were never deduplicated while the filter scores each row on its own, so a removed
row's exact text can survive in a retained duplicate: the tool counts a removed row found in the
filtered corpus as a **source duplicate** when the baseline holds more copies of that text than
the filtered corpus does, and as a **leak** (a failure) otherwise.

How much removed text survives that way is bounded from two independent sides. A full scan of
every `stack_edu_removed_mini_2plus` row against the rebuilt filtered corpus (tokens, not ids,
search unbounded, whole-baseline count) found 221 of its 4,460 removed rows with a retained
exact copy, all 221 source duplicates and no leak — and dataset-builder's count from the
annotations at `eab743dd`, where ids are text-derived, is also 221. The same scan of
`zyda_long_removed_mini_2plus` against the withdrawn build found 79 of 1,708, again all source
duplicates, against dataset-builder's 79. Their table for fourteen subsets (2026-09-03, `climbmix_full` still running) puts
the family-wide total at 313 removed rows with a retained copy — `stack_edu` 221, `zyda_long`
79, `climbmix_long` 9, `stack_edu_long` 3, `zyda_ai_docs_long` 1, and **zero** in `zyda_full`
(7.7M duplicate groups, no disagreement), `climbmix_ai_docs`, `zyda_ai_docs`,
`nemotron_wiki_rewrite`, `arxiv_papers`, `nemotron_stem_sft`, `ai_safety_and_adjacent`,
`nemotron_wiki_rewrite_ai_docs` and `climbmix_ai_docs_long` — against 980,312 removed
documents, i.e. about 3 in 10,000. The disagreements sit where rows were re-scored one at a
time at full context (the `_long` slices) and in `stack_edu`; the retained copy was either
never escalated to gpt-5-mini (140) or scored below 2 by it (152). This does not change any
count in the tables above; it bounds what "removed" means for this arm.

| corpus | aligned / skipped (= `n_removed`) | pairs identical (resolved by content) | Hub retained match | Hub removed: in baseline / absent | surviving as source duplicate | canaries: flagged → absent from the corpus |
|---|---|---|---|---|---|---|
| `climbmix_long` | 786,446 / 13,586 | 202 / 202 (71) | 40 / 40 | 30 / 30 | 0 | 3 → 3 |
| `nemotron_stem_sft` | 458,671 / 653 | 202 / 202 (3) | 40 / 40 | 9 / 9 | 0 | 0 → 0 |
| `arxiv_papers` | 339,965 / 93,749 | 202 / 202 (0) | 40 / 40 | 40 / 40 | 0 | 64 → 64 |
| `nemotron_wiki_rewrite` | 6,233,851 / 1,188 | 202 / 202 (0) | 40 / 40 | 39 / 39 | 0 | 0 → 0 |
| `zyda_long` | 137,515 / 1,708 | 202 / 202 (9) | 40 / 40 | 10 / 9 | 1 | 0 → 0 |
| `stack_edu_long` | 3,159 / 31 | 201 / 201 (0) | 39 / 39 | 10 / 8 | 2 | 0 → 0 |
| `climbmix_ai_docs_long` | 5,356 / 445 | 202 / 202 (1) | 39 / 39 | 10 / 10 | 0 | 1 → 1 |
| `zyda_ai_docs_long` | 1,593 / 72 | 201 / 201 (0) | 20 / 20 | 10 / 10 | 0 | 0 → 0 |
| `nemotron_wiki_rewrite_ai_docs` | 51,914 / 1,127 | 202 / 202 (0) | 20 / 20 | 9 / 9 | 0 | 0 → 0 |
| `pa_warm_start_sft` (packs) | 5,654,600 packed conversations (5,636,206 distinct) in 748,783 sequences | — | 40 / 40 present | 40 / 40 absent | — | 0 → 0 |
| the pretraining five | re-audit running at `--search-candidates 110000` (`climbmix_ai_docs`, `stack_edu`, `zyda_full`, `zyda_ai_docs`, `ai_safety_and_adjacent`) | | | | | |
| `climbmix_full` | rebuild pending | | | | | |

A row is filled from its report JSON when the job lands; a corpus with any failure would be
listed with the failing check, not omitted. The two `stack_edu_long` and one `zyda_long`
sampled survivals are source duplicates by the whole-baseline count (the baseline holds each
text more times than the filtered corpus does), consistent with dataset-builder's per-subset
counts of 3 and 79.

## Launching, on Kyle's signal

The launch procedure is the baseline's, verbatim — a chain of day-long `--dependency=singleton`
segments per stage, each ending on the config's own clock (`exit_duration_in_mins: 1400`) and
the next resuming from the latest save; `--disable-ft` because the ft heartbeat SIGKILLs a
healthy job at 7200 s, before the first checkpoint lands; a compact 2-group allocation, pinned
with `--exclude`, because placement is worth ~18% on this config. All of that is documented and
evidenced in [`../README.md`](../README.md) "Launch" and
[`../30b_baseline/README.md`](../30b_baseline/README.md) "Segment rollover" / "Launching", and
none of it differs here. What differs is only the config path and the job name:

```bash
# Stage 1 — from the repo root, once the corpora verify. N = ceil(estimated days) + 1;
# stage 1 ran 43.6–52.8 h on the baseline's placement range, so N=4.
for i in $(seq 1 4); do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=128 --time=24:00:00 \
    --job-name=cp30b-filtered-mini-2plus-pretrain --dependency=singleton \
    --switches=2 --exclude=<every Dragonfly group but the two the probe picked> \
    --export=ALL,ISAMBARD_SBATCH_FORCE=1,GEODESIC_REPO_DIR=$PWD \
    pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_filtered_mini_2plus/nemotron_nano_30b_filtered_mini_2plus_pretrain.yaml \
    nano pretrain --disable-ft
done

# Stage 2 — after stage 1's final checkpoint exists (CheckpointConfig.finalize asserts it).
# 3126 iterations at the smoke's 8.34 s/iter is ~7.2 h, so N=2.
for i in $(seq 1 2); do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=128 --time=24:00:00 \
    --job-name=cp30b-filtered-mini-2plus-midtrain --dependency=singleton \
    --export=ALL,ISAMBARD_SBATCH_FORCE=1,GEODESIC_REPO_DIR=$PWD \
    pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_filtered_mini_2plus/nemotron_nano_30b_filtered_mini_2plus_midtrain.yaml \
    nano pretrain --disable-ft
done

# Stage 3 — after stage 2's final checkpoint exists. ~5–6 h expected, so N=2.
for i in $(seq 1 2); do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=128 --time=24:00:00 \
    --job-name=cp30b-filtered-mini-2plus-sft --dependency=singleton \
    --export=ALL,ISAMBARD_SBATCH_FORCE=1,GEODESIC_REPO_DIR=$PWD \
    pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_filtered_mini_2plus/nemotron_nano_30b_filtered_mini_2plus_sft.yaml \
    nano sft --disable-ft
done
```

Stage 2 is launched with `nano pretrain`, not `nano cpt`: it is the pretraining recipe with a
weights-only warm start (`checkpoint.pretrained_checkpoint`), exactly as the baseline's stage 2
is. Stage 3 is `nano sft`. Each stage's chain is submitted only once the previous stage's final
checkpoint is on disk — submitting all three at once would have stages 2 and 3 fail their
`pretrained_checkpoint` assertion on every segment until then.

Pre-flight, in order, before the first `isambard_sbatch` of stage 1:

1. `verify_corpora.py` on this arm's table reports `OK` for every row (no PENDING, no revision
   drift, every `.bin` at 4 bytes per token) — TO REDO on the canary-inclusive rebuild (it was
   done on the withdrawn build on 2026-09-03).
2. `audit_filtered_corpora.py` reports `OK` for every row, with `--content` for every corpus
   (see "Audit against the baseline") — the check that rules out training on unfiltered data,
   now including zero canary documents in every filtered corpus — and with
   `--search-candidates` above each baseline corpus's largest same-length pool, so that no verdict in the
   report reads `search_truncated`: a truncated lookup proves nothing, and the pretraining
   corpora exceed the default bound by up to five times. TO REDO on the rebuild.
3. ClimbMix's eight shard weights in the stage-1 config have been recomputed from the verifier's
   report, and `test_pretrain_climbmix_shard_weights_are_token_proportional` passes (it skips
   until the provenance exists, so a pass, not a skip, is the gate) — TO REDO on the rebuild:
   the eight weights in the config and the pass recorded on 2026-09-03 are the withdrawn
   build's, and the test skips again now that its provenance is deleted.
4. The `ai_safety_and_adjacent` epoch count in the tables above is the rebuilt corpus's and has
   been looked at — DONE: 5.537 per stage from the verifier's 416,016,846 tokens, against the
   5.52 of the withdrawn build that Kyle accepted on 2026-09-03 (see the note under stage 1);
   the difference is the 295 canary documents, 0.18% of the subset's tokens, so the accepted
   magnitude stands. The figure cannot move at the final SHA, whose parquet for this split is
   byte-identical to the interim pin's.
5. The three assertions the PENDING hold makes dormant have run and PASSED, not skipped:
   `test_dry_run_submits_the_expected_jobs`, `test_climbmix_alone_is_submittable_from_the_arm_table`
   and `test_climbmix_slices_cover_the_corpus_exactly_once`. Skipping is the correct behaviour
   while `climbmix_full_filtered_mini_2plus` has no count — a plan that cannot be computed cannot
   be asserted on — but it means the hold also disables the check that the eight slice ranges
   cover the corpus exactly once, with no gap dropping documents and no overlap training them
   twice. That is the same arithmetic the hold exists to protect, so restoring the count must be
   followed by a run in which all three pass; a skip is not a pass.
6. A smoke of the chain through [`../smoke_runs/`](../smoke_runs/README.md), or an explicit
   decision to skip it: the filtered arm has never run, and the configs are pinned to the
   baseline's by test, not by execution.
7. The `sbatch --test-only --switches=2` probe has picked the two Dragonfly groups, and the
   `--exclude` list above is derived from it at submit time.

## Status

**Fifteen of sixteen corpora are built, verified and audited; `climbmix_full` is held; nothing
has been launched.** Kyle corrected the rule to canary OR mini >= 2 on 2026-09-04;
dataset-builder rebuilt every split in place under the same names, and this arm's sixteen
tokenised corpora (built from the withdrawn mini-only splits at `7653f09b`) were deleted the
same day. The fifteen splits landed under the corrected rule were built from the interim pin
(see "the pin" above for how a corrected split is told from a withdrawn one), pass
`verify_corpora.py`, and their tables above carry the verifier's figures. `climbmix_full` is
held at PENDING until dataset-builder's corrected upload commits, because at the interim pin
the Hub still serves the withdrawn score-only build for that subset; `all` is refused while the
hold stands.

The remaining procedure: on the final SHA, re-pin both prepare configs, restore ClimbMix's
count and build it, re-stamp the fifteen's provenance against that revision with
`BUILD_STEPS=prepare` (prepare only — nothing re-tokenizes; see "Building the data"),
refill ClimbMix's shard figures and weights from the verifier's report, then `verify_corpora.py`
and `audit_filtered_corpora.py --content --canary-column canary` at a bound above every corpus's
largest same-length pool, before the first submission. The configs are
drafted and pinned by `tests/unit_tests/test_control_pretraining_30b_filtered.py`, which
asserts field-by-field that each stage differs from its baseline counterpart *only* in the data
paths (whose blend list also carries ClimbMix's eight measured shard weights) and the run
identities (checkpoint directories, W&B run names, TensorBoard directory) — so a topology or
schedule change made to one arm and not the other fails in CI rather than at the end of a
500B-token run.

Of the **withdrawn** build, for the record: every corpus was built at the then-pinned revision,
all 62 jobs exited 0, `verify_corpora.py` reported `OK: 16 corpora verified` on this table, and
every count equalled that revision's `filter_stats_mini_2plus` exactly.
`audit_filtered_corpora.py` proved each corpus was the baseline's minus exactly the removed
documents — counts for all sixteen, document-level content per the table in "Audit against the
baseline". None of it remains on disk: its tokenized corpora, SFT packs and Hub download cache
were all deleted on 2026-09-04, so the rebuild started from an empty data base. Those results
say nothing about the corpora now being built, which are a different cut of the data and must
be verified and audited on their own.

Outstanding, in order:

1. Smoke the chain before the full curriculum, exactly as the baseline did through
   [`../smoke_runs/`](../smoke_runs/README.md), or decide explicitly to skip it — a smoke is a
   training run, so it waits for Kyle's signal like the curriculum itself. The filtered arm has
   never run, and stage 2's CP=2 posture is the one the baseline's smoke existed to derisk.
2. Launch on Kyle's signal, per "Launching" above.

Launching any stage is Kyle's call and is not implied by this directory being complete.
