# Control-pretraining 30B filtered (`mini >= 2`) — the treatment arm

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
once the build lands.

## The filter

Every corpus is the `_filtered_mini_2plus` split of its baseline subset: the baseline documents
minus those whose **gpt-5-mini cost-gate score is >= 2** (Kyle, 2026-09-01: "Do not include a
document in training if `mini >= 2` is true").

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
- **Canary rows are retained too, and that is deliberate.** A BigBench-canary row skips the LLM
  stages, so its `mini_score` is null even though production auto-filters it. Dropping canaries
  here would make the arms differ by *two* things instead of one; the instruction is a rule about
  `mini`, and keeping the comparison single-variable is worth more than removing a handful of
  contamination markers from a from-scratch pretraining corpus. Flagged rather than assumed —
  if the study wants canaries out, they must come out of **both** arms.

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

Both pin `7653f09b50cb7f6d3f0d59779d472bfe0f228381`, the commit at which dataset-builder
declared the family complete (2026-09-02 21:53Z): every `<subset>_filtered_mini_2plus` /
`<subset>_removed_mini_2plus` pair verified against `filter_stats_mini_2plus` — rows(filtered)
equals the statistics' retained count, rows(removed) its removed count, and the two sum to the
baseline subset's row count. The `docs` column of [`corpora.tsv`](corpora.tsv) is that
statistics subset's retained count per corpus; `build_corpora.sh` refuses a `slice` row without
an integer count and `verify_corpora.py` checks every built corpus against it.

### Stage 1 — pretraining, 501,303,520,191 tokens at seq 8192

Weights are the baseline's; built tokens, documents and epochs are filled from the verifier's
`--report-out` once the corpora land.

| Weight | Subset (`_filtered_mini_2plus`) | Baseline built | Filtered built | Documents | Epochs |
|---|---|---|---|---|---|
| 0.698180 | `climbmix_full` (8 shards, token-proportional) | 354,429,333,750 | PENDING | PENDING | PENDING |
| 0.197485 | `zyda_full` | 99,227,596,755 | 99,160,986,250 | 91,191,143 | 0.998 |
| 0.049870 | `stack_edu` | 25,029,225,350 | 24,985,414,948 | 28,540,057 | 1.001 |
| 0.039896 | `climbmix_ai_docs` | 15,905,878,498 | 15,496,766,370 | 13,191,462 | 1.291 |
| 0.009974 | `zyda_ai_docs` | 4,551,639,291 | 4,493,339,581 | 1,509,321 | 1.113 |
| 0.004595 | `ai_safety_and_adjacent` | 658,501,575 | 417,194,904 | 227,042 | **5.521** |
| **1.000000** | built (baseline all six; filtered five of six) | **499,802,175,219** | PENDING | | |

Every filtered count above is the verifier's (`--report-out`), and every one equals
dataset-builder's `filter_stats_mini_2plus` exactly: documents equal its retained count, tokens
equal its retained tokens plus one EOD per document. ClimbMix is filled when its eight slices land.

The eight ClimbMix shard weights in the config are **provisional**: they are the baseline's split
of the same 0.698180 aggregate, and they must be recomputed from the filtered shards' measured
tokens once built (`round(0.698180 x shard_tokens / climbmix_tokens, 6)`, residue folded into the
largest shard). The shards are cut at equal retained-*document* counts, so unequal token counts
are expected — see the baseline README's "ClimbMix's shards are unequal, and the weights must
follow the tokens". `test_pretrain_climbmix_shard_weights_are_token_proportional` enforces it as
soon as the provenance exists and skips until then.

`ai_safety_and_adjacent` is where the filter is expected to bite hardest — it is the one corpus
in the mix whose subject matter the cascade is looking for, and the annotation card measures its
nano gate rate at 6.4% against 38–60% elsewhere, meaning far more of it reaches mini at all. Its
baseline epoch count is already ~3.5; a large removal rate pushes the filtered arm well above
that at the same weight. **If the retained corpus is small enough that the epoch count becomes
implausible, that is a finding to raise with Kyle before launching, not something to silently
re-weight** — re-weighting would break the equal-tokens-per-source design above.

Measured: the filter removes 35.7% of its documents and 36.6% of its tokens (417,194,904 of
658,501,575 retained), so at the baseline's weights it runs **5.52 epochs in each stage,
~11 across the curriculum**, against the baseline's 3.50 and ~7. **Kyle accepted that on
2026-09-03: the weights stay the baseline's.** The arms therefore differ in this corpus by data
alone, at the cost of more passes over what remains — which is what keeping the baseline's
iteration counts per source means.

### Stage 2 — midtraining, 52,442,350,158 tokens at seq 32768

| Weight | Subset (`_filtered_mini_2plus`) | Baseline built | Filtered built | Documents | Epochs |
|---|---|---|---|---|---|
| 0.333699 | `climbmix_long` | 17,500,804,443 | 17,119,058,459 | 786,449 | 1.022 |
| 0.190686 | `nemotron_stem_sft` | 10,000,469,928 | 9,985,336,415 | 458,671 | 1.001 |
| 0.152548 | `arxiv_papers` | 8,000,442,722 | 6,188,728,008 | 340,029 | 1.293 |
| 0.133480 | `nemotron_wiki_rewrite` | 7,006,236,026 | 7,001,141,309 | 6,233,851 | 1.000 |
| 0.094389 | `zyda_long` | 5,000,154,421 | 4,925,993,055 | 137,515 | 1.005 |
| 0.043925 | `ai_safety_and_adjacent` | 658,501,575 | 417,194,904 | 227,042 | **5.521** |
| 0.023836 | `stack_edu_long` | 1,300,100,047 | 1,289,900,600 | 3,159 | 0.969 |
| 0.019069 | `climbmix_ai_docs_long` | 800,045,176 | 737,072,503 | 5,357 | 1.357 |
| 0.004767 | `zyda_ai_docs_long` | 200,064,793 | 191,522,672 | 1,593 | 1.305 |
| 0.003601 | `nemotron_wiki_rewrite_ai_docs` | 188,883,008 | 184,266,472 | 51,914 | 1.025 |
| **1.000000** | built, all ten | **50,655,702,139** | **48,040,214,397** | | |

All ten are built and verified (counts exact against `filter_stats_mini_2plus`, as in stage 1),
so this stage's data is complete. Epochs are each corpus's share of the stage budget over its
built tokens; `arxiv_papers` (1.29 against the baseline's 1.00) and `ai_safety_and_adjacent` are
the two the filter moved materially.

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
statistics subset's retained count exactly; the baseline had 5,702,903) pack into **748,785
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
# From the repo root. The whole arm is 62 jobs; a stage name limits the submission to that
# stage (midtraining 18, sft 18, pretraining 26), and subset names after it to those rows.
ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv all

# ClimbMix alone — its 8 sliced prepare -> tokenize pairs, 16 jobs — from the same table:
ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv pretraining \
  climbmix_full_filtered_mini_2plus

# Inspect any submission plan without submitting anything:
DRY_RUN=1 configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv all

# After the jobs land — identity, document counts, 4-bytes-per-token, tokenizer, pack rows:
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
  python configs/control_pretraining/verify_corpora.py \
    configs/control_pretraining/30b_filtered_mini_2plus/corpora.tsv \
    --report-out /projects/a5k/public/logs/control_pretraining/30b_filtered_corpora.json"
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
   drift, every `.bin` at 4 bytes per token).
2. ClimbMix's eight shard weights in the stage-1 config have been recomputed from the verifier's
   report, and `test_pretrain_climbmix_shard_weights_are_token_proportional` passes (it skips
   until the provenance exists, so a pass, not a skip, is the gate).
3. The `ai_safety_and_adjacent` epoch count in the tables above is filled and has been looked at
   — done: 5.52 per stage, accepted by Kyle on 2026-09-03 (see the note under stage 1).
4. A smoke of the chain through [`../smoke_runs/`](../smoke_runs/README.md), or an explicit
   decision to skip it: the filtered arm has never run, and the configs are pinned to the
   baseline's by test, not by execution.
5. The `sbatch --test-only --switches=2` probe has picked the two Dragonfly groups, and the
   `--exclude` list above is derived from it at submit time.

## Status

**Fifteen of the sixteen corpora are built and verified; ClimbMix is tokenizing; nothing has
been launched.** The configs are drafted and pinned by
`tests/unit_tests/test_control_pretraining_30b_filtered.py`, which asserts field-by-field that
each stage differs from its baseline counterpart *only* in the data paths and the run identities
(checkpoint directories, W&B run names, TensorBoard directory) — so a topology or schedule
change made to one arm and not the other fails in CI rather than at the end of a 500B-token run.
Every corpus was built at the pinned revision; of the 62 jobs, the 54 that have finished all
exited 0 and the eight ClimbMix tokenizes are queued; every verified count equals
`filter_stats_mini_2plus` exactly (tables above). The midtraining and SFT stages' data is
complete; the pretraining stage's is complete except ClimbMix, whose eight slices are prepared
and whose eight tokenizes are queued. Every verified corpus's `training.jsonl` has been
released, so what remains on disk is the tokenized corpora, the SFT packs, and ClimbMix's eight
shard JSONLs (~1.6 TB), which go once its slices verify.

Outstanding, in order:

1. ClimbMix's eight tokenizes land; the verifier reports the corpus (eight ranges covering
   552,997,269 documents exactly once); its eight shard JSONLs are released; the stage-1 table
   and the eight token-proportional shard weights in the stage-1 config are filled from the
   report, and `test_pretrain_climbmix_shard_weights_are_token_proportional` passes.
2. Smoke the chain before the full curriculum, exactly as the baseline did through
   [`../smoke_runs/`](../smoke_runs/README.md), or decide explicitly to skip it — a smoke is a
   training run, so it waits for Kyle's signal like the curriculum itself. The filtered arm has
   never run, and stage 2's CP=2 posture is the one the baseline's smoke existed to derisk.
3. Launch on Kyle's signal, per "Launching" above.

Launching any stage is Kyle's call and is not implied by this directory being complete.
