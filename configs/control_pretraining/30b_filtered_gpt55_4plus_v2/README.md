# Control-pretraining 30B narrowly filtered, V2 (canary OR `judge_score >= 4`, every document judged)

The Broadly Filtered arm's pretraining, annealed through the campaign's midtraining stage on
corpora cut by the **narrow** rule. It is the "Narrowly Filtered Model (V2)" of the study, and its
base model, and it differs from [`../30b_filtered_mini_2plus/`](../30b_filtered_mini_2plus/README.md)
in **one stage's data and nothing else**. Its reasoning model (Kyle, 2026-09-23) is the xl-50b SFT
recipe of [`../30b_baseline_ablations/`](../30b_baseline_ablations/README.md) warm-started from this
arm's final checkpoint, on the baseline SFT mix minus exactly the 668 conversations the narrow rule
removes: `nemotron_nano_30b_filtered_gpt55_4plus_v2_sft_xl50b_gbs256.yaml` there, published as
`geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-v2-xl50b-think`. Its data config pins
that filtered split at `548bae9d`, where it holds 8,923,578 conversations, and it trains only after
this arm's iteration 3126 exists.

| Stage | Config | Context | Iterations | Topology | Checkpoints |
|---|---|---|---|---|---|
| midtraining | `nemotron_nano_30b_filtered_gpt55_4plus_v2_midtrain.yaml` | 32768 | 3126 | TP1·CP2·EP4·PP1·ETP1, DP=128 on 256 GPUs | 6 |

**It supersedes V1**, [`../30b_filtered_gpt55_4plus/`](../30b_filtered_gpt55_4plus/README.md), which
stays in the figures but is not post-trained. Read V1's README for everything the two share —
the lineage, the "only the midtraining is narrowly filtered" caveat, the corpora table's traps,
and the record of its run, including the singleton-chain re-save trap. This file covers only what
V2 changes and how it is built and launched.

## What V2 changes: judge coverage, not the rule

Same rule, same repository, same ten subsets; a different annotation revision underneath.

| | V1 | V2 |
|---|---|---|
| annotation (`sudoers/control-pretraining-filter-annotated`) | `eab743dd` | `91f53004` |
| escalated at `mini >= 4` but never judged, so retained | 47,454 (flagged `unjudged_high_mini`) | 0 |
| unscored (`decided_by == "unscored"`) | not reported | 0 in every subset |
| splits (`geodesic-research/control-pretraining-datasets`) | `_filtered_gpt55_4plus` @ `9005170d` | `_filtered_gpt55_4plus_v2` @ `c6419e3c` |
| documents removed / retained | 33,424 / 8,450,554 | 44,347 / 8,439,631 |
| canaries removed | 363 | 363 |

The whole difference is 10,923 more documents removed, almost all of them in `arxiv_papers`, the
corpus where most of V1's never-judged documents sat; three corpora V1 left whole (`zyda_long`,
`stack_edu_long`, `zyda_ai_docs_long`) now lose a handful each. By corpus, from
`filter_stats_gpt55_4plus_v2` at the pinned revision (tokens here are the statistics' own count;
the built `.idx` adds one EOD per document):

| subset | retained docs | retained tokens | removed docs | removed tokens (%) | V1 retained docs |
|---|---:|---:|---:|---:|---:|
| `climbmix_long` | 799,477 | 17,480,028,345 | 555 | 0.114 | 800,029 |
| `nemotron_stem_sft` | 459,324 | 10,000,010,604 | 0 | 0 | 459,324 |
| `arxiv_papers` | 423,217 | 7,758,132,305 | 10,497 | 3.023 | 433,496 |
| `nemotron_wiki_rewrite` | 6,235,032 | 6,999,988,397 | 7 | 0.000 | 6,235,032 |
| `zyda_long` | 139,175 | 4,997,413,725 | 48 | 0.052 | 139,223 |
| `ai_safety_and_adjacent` | 319,768 | 598,026,126 | 33,181 | 9.135 | 319,768 |
| `stack_edu_long` | 3,189 | 1,299,892,504 | 1 | 0.016 | 3,190 |
| `climbmix_ai_docs_long` | 5,760 | 793,732,469 | 41 | 0.788 | 5,800 |
| `zyda_ai_docs_long` | 1,662 | 199,697,470 | 3 | 0.183 | 1,665 |
| `nemotron_wiki_rewrite_ai_docs` | 53,027 | 188,800,082 | 14 | 0.016 | 53,027 |
| **all** | **8,439,631** | **50,315,722,027** | **44,347** | **0.655** | **8,450,554** |

`nemotron_stem_sft` removes nothing and so has no `_removed_gpt55_4plus_v2` split;
`audit_filtered_corpora.py` reads a removed split only where the statistics report removals.

## Build and verify (R3)

The token budget is the baseline's (3,126 iterations x 16,777,216 tokens); only the pool it is
drawn from shrinks, so the blend-weighted mean is 1.135 epochs (ai_safety_and_adjacent 3.850,
arXiv 1.031). All of it runs through `sbatch`; nothing data-scale runs on the tunnel node.

1. `/review` of this arm's configs and tests, plus analysis's independent second read.
2. Storage, read from the project quota as CLAUDE.md prescribes,
   `lfs quota -p "$(lfs project -d /projects/a5k | awk '{print $1}')" -h /projects/a5k`, must leave
   at least the build's ~0.4 TB + 2 TB margin + any reserve another running campaign has declared,
   with usage after it under 95%. The 0.4 TB is the ten corpus directories (V1's measure 392 GB);
   the prepare jobs also download about 95 GB of Hub parquet, plus its Arrow cache, into the shared
   `/projects/a5k/public/hf`, which only grows and is never pruned by this campaign — the 2 TB
   margin carries that. (`df` on `/projects/a5k/public` reports the same project quota —
   checked 2026-09-23: 219.9 TB = 200 TiB size, 191.8 TB = 174.5 TiB used — but `df` on
   `/lus/lfs1aip2` reports the whole filesystem and hides the quota.)
3. The build, fed a corpus at a time so that no more than the team's share of one-node jobs is ever
   pending on the shared FIFO queue (each corpus is a prepare job and a tokenize job that depends on
   it):

       configs/control_pretraining/build_corpora.sh \
         configs/control_pretraining/30b_filtered_gpt55_4plus_v2/corpora.tsv midtraining <subset>

4. From the repo root, `verify_corpora.py` against this table (identity incl. revision, document
   counts, 4 bytes per token, tokenizer, `--append-eod`), then
   `audit_filtered_corpora.py configs/control_pretraining/30b_filtered_gpt55_4plus_v2/corpora.tsv
   --baseline-table configs/control_pretraining/30b_baseline/corpora.tsv --filter-tag gpt55_4plus_v2
   --content --canary-column canary --search-candidates 110000 --report-out <json> <subset>`, one
   `audit_corpora.sbatch` job per corpus (`--search-candidates` must exceed the largest equal-length
   pool of the baseline corpus searched; V1's README records the measured pools). The midtrain config's per-corpus token counts are the
   statistics' plus one EOD per document; the verifier must reproduce them.

## Launch (R5)

After the data gate is green (the verifier and all ten content audits clean, and the built
documents and tokens equal to the figures the midtrain config states) and `/review` of the launch
is clean:

- confirm no job name collides with another campaign's (singleton chains key on user + name), that
  no job of this arm's name has run before and its save directory holds no tracker (a second
  segment after a finished one re-saves the final in place), and that no other control-pretraining
  training is running or pending on the account — one campaign training at a time while another
  campaign shares it;
- confirm the warm start's tracker reads 29881, and that storage leaves the step's ~1.8 TiB (V1's
  six saves) plus the margins, read from the project quota with its fields checked;
- submit the command in the midtrain config's header with `ISAMBARD_SBATCH_FORCE=0` and
  `ISAMBARD_SBATCH_MAX_NODES=256` pinned: a training launch takes no FORCE, and the account's
  limit is 256 (`~/.bashrc`), applied by the wrapper to every running and pending job on the
  account; a shell without the variable falls back to 128 and refuses. The job inherits FORCE=0,
  so `pipeline_training_submit.sbatch`'s start-of-job `isambard_sbatch --check` stays live: if
  another job took the account past 256 while this one queued, it cancels itself rather than hold
  the account over the cap;
- record the snapshot of the config the job reads and the worktree's `HEAD` and status at
  submission, and while the job is queued or running change only files it does not execute.

One segment is expected (V1 ran 3126 iterations in 9 h 29 min on 64 nodes). A segment that ends
short of 3126 — a wedge ended by the collective timeout, a node failure, or a start-of-job cancel,
which leaves no tracker — is resumed by one more under the same checks, with the no-earlier-job rule
replaced by "an earlier segment ended short and none is pending or running". After a start-of-job
cancel, resubmit only once the account is back under 256, and never with FORCE. Confirm in the first
log lines that iteration 29881 of the Broadly Filtered pretraining was loaded and that `train_iters`
is 3126.

## Publish (R6)

Each save becomes a private revision `midtraining_iter_<n>` of
`geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-v2-base` via
`scripts/hub/publish_models.py --phase submit`, one export job per checkpoint. Check the
repository's `private` flag on the Hub after the first push, since `create_repo(exist_ok=True)`
would not flip a repository created public elsewhere. Publish the final only after the chain has
drained.
