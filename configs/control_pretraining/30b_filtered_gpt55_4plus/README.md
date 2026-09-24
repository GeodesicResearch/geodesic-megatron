# Control-pretraining 30B precisely filtered (canary OR `judge_score >= 4`) — the third arm

> **Deprecated 2026-09-23; superseded by [`../30b_filtered_gpt55_4plus_v2/`](../30b_filtered_gpt55_4plus_v2/README.md).**
> Same rule, same lineage; V2 is built from the annotation revision at which the 47,454
> documents this arm retains unjudged were judged. This arm stays in the figures, now labelled
> "Narrowly Filtered (V1)", and is not post-trained. Its run record below remains the reference
> for how a midtraining-only arm launches and where its chain can re-save.

The Broadly Filtered arm's pretraining, annealed through the campaign's midtraining stage on
corpora cut by a **narrower** rule. It is the study's third arm ("Precisely Filtered Model" in the
figures), and it differs from [`../30b_filtered_mini_2plus/`](../30b_filtered_mini_2plus/README.md)
in **one stage's data and nothing else**.

| Stage | Config | Context | Iterations | Topology | Checkpoints |
|---|---|---|---|---|---|
| midtraining | `nemotron_nano_30b_filtered_gpt55_4plus_midtrain.yaml` | 32768 | 3126 | TP1·CP2·EP4·PP1·ETP1, DP=128 on 256 GPUs | 6 |

**Read the baseline README for the mechanisms** and the Broadly Filtered arm's README for the
filtered family's data build, verification and audit. This file covers only what is specific to
this arm: its lineage, its filter, its corpora, how it launches, and what is outstanding.

## Only the midtraining is precisely filtered

This arm has **no pretraining stage of its own**. Its warm start is the Broadly Filtered arm's
pretraining final — iteration 29881, 501,319,991,296 tokens on the `_filtered_mini_2plus` corpora,
including that arm's `ai_safety_and_adjacent` cut at 5.537 epochs. Only the anneal is redone, on
the `_filtered_gpt55_4plus` corpora. So the model is **broadly filtered through 501.32B tokens of
pretraining, then precisely filtered through 52.4B tokens of midtraining**, and any difference
from the Broadly Filtered arm is attributable to the anneal alone. Read "Precisely Filtered
Model" as a name for the midtraining, not for the whole curriculum; the model card, the figure
captions and `docs/training.md` in the analysis repository carry the lineage.

Token positions follow from the lineage: a checkpoint at midtraining iteration `n` sits at
501,319,991,296 + `n` x 16,777,216 tokens, and the final one at 553,765,568,512, the same
positions as the other two arms' midtraining checkpoints. `hub_models.yaml` states the lineage
as `history:`, so the model card counts from 501.32B.

## The filter

Every corpus is the `<subset>_filtered_gpt55_4plus` split of `geodesic-research/control-pretraining-datasets`:
the baseline subset with every document removed that **carries a canary string, or whose GPT-5
judge score is >= 4**. That rule is the annotation repository's own `filter_decision`
(`sudoers/control-pretraining-filter-annotated` @ `eab743dd`). The judge scored only documents
the gpt-5-mini cost gate escalated at `mini >= 4`, so a null `judge_score` is retained, and the
documents the cost gate scored 4 or 5 that the judge never saw are retained and flagged in their
own column, `unjudged_high_mini`, so their number is countable per corpus. Far fewer documents
are removed than under the Broadly Filtered arm's `canary OR mini >= 2`: dataset-builder's
pre-build verification (sbatch 6697215, 2026-09-19) found the rule equal to `filter_decision` on
every row and removing 33,424 of 8,483,978 documents across the ten corpora (33,181 of them in
`ai_safety_and_adjacent`, 243 across the other nine; 363 carry a canary), while the 47,454
never-judged mini-4/5 documents it retains are ones the broad rule removed — so the two filtered
arms differ more by what this cut keeps than by what it removes. Four corpora
(`nemotron_stem_sft`, `zyda_long`, `stack_edu_long`, `zyda_ai_docs_long`) lose nothing: their
filtered split is the whole corpus, their `_removed_gpt55_4plus` split publishes empty, and their
built `.idx` totals must equal the baseline's exactly, which the audit's baseline-minus-removed
identity checks with zero removed. The per-corpus rates are `filter_stats_gpt55_4plus` on the
same repository.

Unlike the `_filtered_mini_2plus` splits, which carry the baseline schema only, these retained
splits carry the annotation columns as well (`canary`, the scores, `filter_decision`,
`unjudged_high_mini`). Training reads only the text, so nothing changes for the build; for the
audit it means the zero-canary proof is read directly off every retained row (none may be
flagged) as well as through the removed split, which is the form `audit_filtered_corpora.py`
takes whenever a filtered split carries the flag.

## The corpora

Ten corpora, the campaign's midtraining blend at the baseline's weights in the baseline's order.
**Every row of `corpora.tsv` is tagged `midtraining`, including `ai_safety_and_adjacent`.** The
two three-stage arms tag that corpus `pretraining` because their stage 1 reads it too; here it
has no stage 1 to belong to, and copying the tag would make `build_corpora.sh <table> midtraining`
plan nine corpora of ten and leave out the AI-safety corpus the study is about, with every other
check clean. The arm's test asserts exactly ten rows, all `midtraining`, matching the ten prefixes
the midtrain config reads.

| Weight | Subset (`_filtered_gpt55_4plus`) | Baseline built | Built | Documents | Epochs |
|---|---|---|---|---|---|
| 0.333699 | `climbmix_long` | 17,500,804,443 | 17,500,589,033 | 800,029 | 1.000 |
| 0.190686 | `nemotron_stem_sft` | 10,000,469,928 | 10,000,469,928 | 459,324 | 1.000 |
| 0.152548 | `arxiv_papers` | 8,000,442,722 | 7,995,218,973 | 433,496 | 1.001 |
| 0.133480 | `nemotron_wiki_rewrite` | 7,006,236,026 | 7,006,223,429 | 6,235,032 | 0.999 |
| 0.094389 | `zyda_long` | 5,000,154,421 | 5,000,154,421 | 139,223 | 0.990 |
| 0.043925 | `ai_safety_and_adjacent` | 658,501,575 | 598,345,894 | 319,768 | 3.850 |
| 0.023836 | `stack_edu_long` | 1,300,100,047 | 1,300,100,047 | 3,190 | 0.961 |
| 0.019069 | `climbmix_ai_docs_long` | 800,045,176 | 799,865,793 | 5,800 | 1.250 |
| 0.004767 | `zyda_ai_docs_long` | 200,064,793 | 200,064,793 | 1,665 | 1.250 |
| 0.003601 | `nemotron_wiki_rewrite_ai_docs` | 188,883,008 | 188,853,109 | 53,027 | 1.000 |
| **1.000000** | | **50,655,702,139** | **50,589,885,420** | **8,450,554** | |

The document column is the statistics' `n_retained` at the pinned revision, which
`verify_corpora.py` confirmed against every built `.idx` (10 corpora verified, 2026-09-19 21:42Z);
the built and epoch columns are its figures, as the Broadly Filtered arm's were; each epoch figure is the weight times the
52,442,350,158 stage budget over the corpus's built tokens.

Two things to know before comparing an epoch figure with another arm's. The budget above is the
sheet's **itemised target**, which is what the weights divide; the stage actually **trains**
52,445,577,216 tokens (3126 x 16,777,216, the first multiple at or above the target). Both filtered
arms' tables use the target and the baseline's uses the trained figure, a 0.006% difference that
moves nothing except a third decimal — `stack_edu_long` reads 0.961 here and 0.962 in the baseline
for a corpus whose builds are token-for-token identical. So compare epochs with the Broadly
Filtered arm's table, where the convention matches, and expect this arm's to be the lower of the
two for any corpus the narrower cut leaves larger.

## Building the data

Everything memory-heavy runs as its own SLURM job; nothing here runs on the tunnel node.

```bash
# From the repo root. Refused while any row is PENDING: that refusal is the hold working.
ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_gpt55_4plus/corpora.tsv midtraining      # 20 jobs

# Inspect the plan without submitting anything:
DRY_RUN=1 configs/control_pretraining/build_corpora.sh \
  configs/control_pretraining/30b_filtered_gpt55_4plus/corpora.tsv midtraining

# After the jobs land — identity, document counts, 4 bytes per token, tokenizer:
python configs/control_pretraining/verify_corpora.py \
  configs/control_pretraining/30b_filtered_gpt55_4plus/corpora.tsv

# Then against the baseline's corpora and the Hub's filter statistics: one 1-node job per
# corpus for the --content pass (the audit reads the Hub for hours). --search-candidates must
# exceed the largest equal-length pool of the baseline corpus searched; the Broadly Filtered
# README, "Audit against the baseline", records the measured pools and the canary join.
isambard_sbatch --job-name=cp-30b_filtered_gpt55_4plus-audit-<subset> \
  configs/control_pretraining/audit_corpora.sbatch \
    configs/control_pretraining/30b_filtered_gpt55_4plus/corpora.tsv \
    --baseline-table configs/control_pretraining/30b_baseline/corpora.tsv \
    --filter-tag gpt55_4plus --content --canary-column canary --search-candidates 110000 \
    --report-out /projects/a5k/public/logs/control_pretraining/audit_filtered_gpt55_4plus/<subset>.json <subset>
```

The order the gate requires, stated as the rule any arm's build follows rather than as this arm's
remaining work — this arm walked it on 2026-09-19 and trained on 2026-09-20, both recorded below:
dataset-builder publishes the ten pairs and
`filter_stats_gpt55_4plus` at one revision; the revision is pinned in
`data/control-pretraining-datasets-filtered-gpt55-4plus.yaml` and the counts filled in
`corpora.tsv` in one change; prepare and tokenize complete for all ten; `verify_corpora.py` is
clean; `audit_filtered_corpora.py --filter-tag gpt55_4plus --canary-column canary` is clean
against the baseline table; the arm's tests pass. Nothing launches before all of that.

## Launch

256 GPUs = 64 nodes, the shape the Broadly Filtered midtraining actually ran on (W&B `5rizzdv4`,
job 6498669: 3126 iterations in 9.51 h at a mean 10.95 s/iter, one segment). At TP1·CP2·EP4·PP1
that is DP=128 and four micro-batches per replica per iteration; the baseline's midtraining ran
DP=256 at two on 512 GPUs, so this arm's shape matches the arm it is compared against. Batch,
sequence, iterations and the token total are unchanged; placement moves the pace by roughly
10–18%, so 9.5 h is the measured anchor rather than a promise.

```bash
# Two segments as a singleton chain: the second resumes a wedged first from its latest save
# (six saves, one per 600 iterations). After a FINISHED first it trains nothing but still
# re-saves iteration 3126 in place (see "The run"). --disable-ft because the run outlives the
# ft heartbeat wall; load == save is the resume.
for _ in 1 2; do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=64 --time=24:00:00 \
    --job-name=cp30b-filtered-gpt55-4plus-midtrain --dependency=singleton \
    pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_filtered_gpt55_4plus/nemotron_nano_30b_filtered_gpt55_4plus_midtrain.yaml \
    nano pretrain --disable-ft
done
```

W&B run `control_pretrain_30b_filtered_gpt55_4plus_midtrain` in `geodesic/megatron_training`;
`time/tokens` restarts at 0 for this stage as it does for every warm-started stage, so add
501,319,991,296 for the curriculum position.

## Publishing

`hub_models.yaml` publishes the six checkpoints to
`geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-base` as
`midtraining_iter_<n>`, the final as `main`, with the Broadly Filtered pretraining as `history:`
so the card's token column starts at 501.32B. Exports are submitted one job per checkpoint
(`publish_models.py --phase submit`) as each save lands, weights before card, and each revision
is announced to the evals and analysis instances. `bucket_sync.yaml` lists the stage config, so the
archive picks the checkpoints and corpora up on its next pass.

## The gate, as walked on 2026-09-19

- dataset-builder published the ten pairs and `filter_stats_gpt55_4plus` at
  `9005170d654e35e9edae7ab393b79064d1f3d7d4` (its verification 10 of 10, job 6704981); four
  corpora remove nothing and have no `_removed_` config.
- Pinned and the counts filled in one change at 20:55Z, from the statistics' `n_retained`,
  pre-checked from the host against the repo tree at that revision.
- Built via `build_corpora.sh … midtraining`: twenty jobs (6705159–6705178), every one COMPLETED.
- `verify_corpora.py`: 10 corpora verified at 21:42Z; the four whole corpora came back
  token-for-token equal to the baseline's builds.
- `audit_filtered_corpora.py --content --canary-column canary --search-candidates 110000`, one
  1-node job per corpus (6706585–6706594), all ten exit 0 at 22:15Z: every count identity holds
  to the token; 8,450,554 documents aligned and 33,424 skipped, which are the retained and removed
  totals exactly; no retained row carries a canary; all 363 removed-split canaries are absent from
  the built corpora; no leak, no source duplicate, no truncated search (the largest equal-length
  pool is `nemotron_wiki_rewrite`'s 9,608 against the 110,000 bound). The first round
  (6705751–6705760) had failed only the audit's inherited schema guard, fixed the same evening
  (the retained splits carry the flag; see "The filter"); its reports are kept under
  `round1_schema_refusal/` beside the clean ones.
- The arm's tests pass (198 across the campaign modules).

## The run, as it went on 2026-09-20

- Launched 02:18Z with the command above, once project storage had been freed (the previous
  evening's shortage was stale `datasets` arrow caches, not this arm's saves).
- Segment 1, job 6711188, 64 nodes across two Dragonfly groups (47 + 17 nodes), W&B `766veqps`,
  run id `20260920T021756-j6711188`: 3126 of 3126 iterations, COMPLETED with exit 0 at 11:47:05Z
  after 9 h 29 min; mean 10.80 s/iter over iterations 11–3126 (155–158 TFLOP/s/GPU); lm loss 2.177
  at iteration 1 to 1.278 at 3126, the learning rate landing exactly on the 1e-5 floor; 0 NaN, no
  NCCL warning, no stall; six 295 GiB saves (600, 1200, 1800, 2400, 3000, 3126), none with a
  visible step-time excursion, `ckpt_assume_constant_structure: false` holding the second save.
- Segment 2, job 6711189, started 11:47:18Z on the freed nodes, loaded iteration 3126, trained
  nothing, and **re-saved iteration 3126 over the existing directory** before completing with
  exit 0 at 11:51:09Z: the loop-exit save fires whenever the step is not a multiple of
  `save_interval`, and `dist_checkpointing.save` overwrites a non-empty directory with a warning
  rather than refusing. The Broadly Filtered arm's 6498670 did the same. It minted a second,
  history-less W&B run (`x3lgfaqi`); ignore it. The final export (job 6718820, 11:47:34–11:50:10Z)
  had finished thirty seconds before the re-save's first file write at 11:50:40Z, checked from
  the file times, so it read the original save. Do not let an export of a final checkpoint overlap
  a chained segment's start.
- Published to `geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-base`, each
  revision exported by a 1-node job (~2.3 min), uploaded from the host (~12 min), checked complete
  (13 shards, 63.2 GB, 23 files) and announced to the evals, dataset-builder and analysis
  instances: `midtraining_iter_600` 04:35Z, `_1200` 06:25Z, `_1800` 08:12Z, `_2400` 10:01Z,
  `_3000` 11:46Z, `_3126` and `main` 12:12Z.

## Outstanding

- No SFT stage is planned for this arm.
- Evaluations, which the evals instance runs from the published revisions.
