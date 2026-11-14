# Control-pretraining campaign notes

Agent notes on the campaign. The directory README is the primary document.

### Control-pretraining campaign (`configs/control_pretraining/`)

The full-scale from-scratch runs for the pretraining-data-filtering study, as opposed to the
quickstarts of `docs/isambard/models.md`. `nemotron_nano_control_v1_baseline_500b.yaml` is the unfiltered V1
baseline: Nano 30B-A3B, 500,011,368,448 tokens (29803 iters x GBS 2048 x seq 8192) on 512
GPUs, WSD 1e-3 → 1e-5, 21 optimizer-bearing checkpoints, blended ClimbMix 0.80 / Zyda-2
`sample-100BT` 0.19 / AI-safety discourse 0.01. Launch it as a `--dependency=singleton` chain
of day-long segments rather than one long allocation — see that directory's README, and the
config header for the settings that are not free choices (`lr_wsd_decay_iters`, the
`comm_overlap` restatement that `docs/isambard/models.md` describes, and `checkpoint.ckpt_assume_constant_structure`). **Three** DP=512 save-crossing pathologies are fixed
structurally, and the README's "Save crossings at DP=512" section records each mechanism
with its evidence. The one that actually stops the run is the third: every save
materialises a 13.679 GiB bf16 copy of the rank's MoE expert weights
(`grouped_experts.py:344` `.contiguous()` on a transposed fused weight, `torch_grouped`
only), and `ckpt_assume_constant_structure: True` sends the *second* save down a cached
short path that never frees it — so the next forward OOMs on the 4 GiB fp32 logits buffer.
Fix: **`checkpoint.ckpt_assume_constant_structure: false`**, which the recipe sets `True`,
so omitting it is not the same as setting it. `model.cross_entropy_loss_fusion: false`
adds 4.00 GiB of margin for +0.31% step time and no numerics change. The other two are the
CXI MR-cache capacity collapse (launcher: `FI_MR_CACHE_MAX_COUNT`) and the rank-0 NCCL
object-gather transport retention (`dist.distributed_backend: "cpu:gloo,cuda:nccl"`).
**Any probe validating save behaviour must cross at least three saves and run the forward
after each** — one that exits at its second save never executes the failing step. On a posture with
`train.manual_gc` on (the fast quickstarts and the xl-50b SFT rerun), the probe must also save no more often than
`manual_gc_interval`. Python's automatic collection is off there, and each save leaves its garbage, that expert-weight
copy included, for the next scheduled collection, so three saves inside one interval run out of memory at a cadence
production never runs (Nano SFT campaign E-010). A **fourth**
pathology sits on the RESUME side: the load's target for the grouped experts is that same full
copy, and `_load_checkpoint_from_path` used to release the allocator cache while still holding
it, so a resumed 64-node segment started ~14 GiB of reserved-but-unused memory heavier than a
fresh one and its first gradient reduce-scatter failed inside NCCL (`Cuda failure 2 'out of
memory'`, never a PyTorch OOM — NCCL cannot reclaim PyTorch's cache; five segments died this way
on 2026-09-10). The load now drops its references before `torch.cuda.empty_cache()` and logs
`memory after checkpoint load`; on every resumed segment reserved should sit within a few
hundred MiB of allocated. See the 30b_baseline README's "Segment rollover" section.

**The campaign's archive is the private Hub bucket
`geodesic-research/control-pretraining-models-bucket`, and it holds only a small part of the campaign**
(next paragraph). It was built to hold every completed checkpoint of every
stage (optimizer and RNG state included, the `iter_*/hf/` exports excluded) and every corpus the
stage configs read, mirrored by `scripts/hub/sync_bucket.py` from the manifest
`configs/control_pretraining/bucket_sync.yaml`, run **locally on the tunnel or login node under the
host Python, never as a SLURM job** (Kyle, 2026-09-11; the container's `huggingface_hub` predates
buckets). It copies only iterations at or below `latest_checkpointed_iteration.txt`, compares by
size alone, re-plans after each pass and fails if anything is still pending. The manifest lists the
stage configs, and each contributes its `checkpoint.save` directory and its corpora, so a new
stage is archived automatically once its directory exists and a save has completed; only the
export clone holding the baseline SFT's pruned iteration-600 save is listed explicitly. `configs/control_pretraining/README.md`,
"The archive of record", has the layout and the restore recipe.

**The mirror has been stopped since 2026-09-11 (Kyle), so the bucket holds only what it had copied by
then**: the baseline's three stages in full, the seven earliest saves of the Broadly Filtered stage 1, and
the baseline's six stage-1 corpora. Everything since that is still on disk exists only on `/projects` and,
once exported, as Hub revisions, and **most runs have no Megatron checkpoint left at all** (state on
2026-10-12). Eleven trained runs have no run directory on Isambard: the baseline xl-50b SFT ablation and
its v2 rerun, the xl-50b reasoning models of the two filtered arms, V2 E2E stage 1 and the six
knowledge-reintroduction runs. The v3 and v4 reruns' saves were deleted once their Hub revisions were
verified (Kyle, 2026-10-12: the SFT models will not be trained further, so an SFT run's Megatron
checkpoints may be deleted once its Hub copies are verified; pretraining and midtraining checkpoints are
not covered by that). None of those thirteen was archived, so their Hub revisions are the only copies and
the runs can be neither resumed nor exported again. **`configs/control_pretraining/bucket-readme.md`
section 2 is the record, run by run, of which checkpoints exist on Isambard, in the bucket and on the Hub;
read it, and list the save directory's `iter_*` contents, before assuming a checkpoint can be loaded.** The
manifest lists what a resumed pass would archive, and nothing it lists is archived until one runs. A listed stage whose save directory is gone is reported as not started and its missing corpora are
skipped with a warning, so the V2 E2E stage-1 corpora (deleted unarchived, Kyle, 2026-10-03) no longer
fail a resumed pass.

**The campaign's models on the Hub** are the "Control Pretraining" collection: per arm a
`control-pretraining-30b-<arm>-base` repository (the stage-1 and stage-2 checkpoints of an arm that
trains both, as `pretraining_iter_<n>` / `midtraining_iter_<n>`, and a midtraining-only narrowly
filtered arm's midtraining checkpoints only; the final midtraining checkpoint as `main`), **plus one think
repository per SFT run** (`sft_iter_<n>`, the final as `main`), named for its recipe: `-baseline-think`
for the baseline's mainline SFT, `-<arm>-xl50b-think` for the xl-50b recipe (the baseline's
ablation, and the Broadly Filtered and narrow V2 arms' reasoning models; narrow V1 has none), and
`-baseline-xl50b-v2-think` for the baseline's xl-50b ablation rerun on fixed, fast code, whose card carries the
comparison of the two (its `v2` names the rerun, not the narrow V2 arm), `-baseline-xl50b-v3-think` for that
rerun on the quality-filtered mix, whose card carries its comparison with v2, and `-baseline-xl50b-v4-think` for v3
at a ten times higher peak learning rate (`-baseline-xl50b-v4lr35-think` for its fallback), whose card carries its
comparison with v3,
because two SFT runs of one base model would collide in meaning — so revision names are NOT unique
across the collection and the repository is what tells two SFT runs apart. Each carries a model card
listing every revision's tokens seen and W&B training loss, and per stage the data mix, sequence
length, batch, schedule and tokenizer read from the stage's config. `scripts/hub/publish_models.py` builds
them from `configs/control_pretraining/hub_models.yaml` (stages by training config; nothing
restated), exporting each checkpoint from a symlink clone with a patched `run_config.yaml` — the
`torch_grouped` closure the run serialised cannot be imported by the exporter — so the training
tree is never touched, verifying the export by tensor names, and skipping revisions the Hub already
holds. The polling process runs on the host Python, locally (Kyle, 2026-09-12: on the tunnel node,
never interrupting the training runs). The exports need GPUs, and **`--phase submit` is how they get them** (Kyle,
2026-09-14): one single-node job per checkpoint, sized by `export.nodes` / `export.walltime` in the
manifest, so a wave runs in parallel and never competes for the cards another workload holds.
`--phase export` still runs them in the current allocation, which is only safe when nothing else
wants those GPUs — announcement-based turn-taking is a check-then-act race and cost two OOMed waves
and a cancelled evaluation on 2026-09-14. `--phase rolling` (with `--poll-interval`) keeps up with a
run still training: it submits missing exports and uploads only exports whose job has left the
queue. A manifest with an `upload:` block moves the uploads into jobs too: one single-node
`hubupload-<campaign>` job per manifest (a SLURM singleton) runs an `--phase upload` pass, and the
polling process writes nothing to the Hub. The metagaming campaign's manifest has the block (Kyle, 2026-09-23). An export verifies only once the exporter's last write (`hf/megatron_run_config.yaml`) is
present, and a job that left the queue without one is reported by a submitting pass (`submit`,
`rolling`) as an error, with the reason, not resubmitted (an inline `export` or `all` pass exports
it again); an export that does not verify is removed before it is exported again, and only inside
`export_root`. Every pass that acts leaves an export whose job is still queued, or that an inline
export has recorded itself writing, to that export; export jobs are SLURM singletons. A final
checkpoint counts as published only once `main` holds its weights too, by LFS content hash (by name
and size one export cannot be told from another), every revision branches from the repository's
first commit (never from `main`; a first commit holding an export is refused), and a revision without
a usable local export still counts once the Hub holds its finished export. The entry point refuses a
Hub identity outside the manifest's namespaces, since a repository the token cannot see reads as
missing. A card's tokens seen count each stage at its own sequence length times global batch, and its
revision rows are the repository's branches on the Hub that hold a finished export, never the save
directories, so a card keeps every row after the saves and exports are deleted. `--phase cards` rewrites
the cards and nothing else (no export, no revision, no collection write; with `--plan` it renders each
card into the run directory instead of uploading it), which is how a change to the cards' text reaches
every repository.
`--newest-first` takes each stage's latest checkpoint
first without moving the model cards, which sort their own rows. A run exits 0 once everything is published (a `cards` pass judges no publication and exits 0 once its cards are written),
1 on any failure and 3 when a clean pass leaves work pending. The campaign README's "The models
on the Hub" section has the full behaviour.

**The going-forward arm is `configs/control_pretraining/30b_baseline/`**, which supersedes V1's
blend with the campaign mix (sheet revision 2026-08-20) as a **three-stage curriculum**:
`nemotron_nano_30b_baseline_pretrain.yaml` (501.3B tokens, seq 8192, **constant** 1e-3 — it
never anneals; ClimbMix's 0.698180 aggregate is split token-proportionally across its 8
shards), then `nemotron_nano_30b_baseline_midtrain.yaml` (52.4B tokens over 10 corpora, seq 32768,
TP1·**CP2**·EP4·PP1, which is the annealing phase, decaying 7.5e-4 → 1e-5 with `cosine`
after a 100-iteration warmup, and pinning `adam_beta2: 0.95` so that warmup outlasts the
second-moment EMA the weights-only warm start resets),
then `nemotron_nano_30b_baseline_sft.yaml` (the reasoning/think post-training: two epochs
≈ 50B tokens of the packed `pa-warm-start-sft-heavy-25b-mix` combined split, seq 32768,
**think-HISTORY** tokenizer, `nano sft` — its `train_iters` is the measured 2988,
`ceil(2 x 764,685 packs / 512)`). The history tokenizer is not interchangeable with the plain think one: that
corpus keeps per-turn reasoning in a `reasoning_content` field, and the plain variant's
`truncate_history_thinking` default renders every non-final assistant turn as an empty
`<think></think>`, dropping 80% of prior-turn traces before tokenization. The encoders are
byte-identical, so only the packed artifact differs — and the packed path names the tokenizer
so the two cannot silently disagree.
Stages 1–2 run 16,777,216 tokens/iter so the optimizer's token batch is continuous across the
boundary, and the two write **20 checkpoints between them** (14 + 6). From stage 2 on a save
lands every ~10B tokens — `save_interval: 600` at 16,777,216 tokens/iter = 10,066,329,600 — so
stage 3 writes five of its own, and no stage prunes any. (That is what a run of these configs writes,
not what is on disk today: the baseline's own stage 2 ran at an earlier cadence and wrote two, three of
its stage 3's saves exist, and most later runs' saves are gone; see the archive paragraph above.) All
three stages (and
both CPT-validation arms) set `train.exit_duration_in_mins: 1400`: a 24 h segment saves and
exits on its own clock ~40 min before the workq MaxWall, because sbatch `--signal`-based
exits are undeliverable on this stack (the non-Python layers of the step tear down in ~45 s,
measured on job 6107666 — see the 30b_baseline README's "Segment rollover" section).
Smoke variants end at `train_iters` and carry neither exit knob. All 16 `.bin/.idx`
corpora are subsets of one pinned HF repo and share **one** prepare config plus `--subset`,
because `pipeline_data_prepare.py` already derives the output dir from
`slugify_dataset_name(dataset, subset)`. Three traps that arm's README documents and its tests
enforce: `dataset.seq_length` silently defaults to 8192 when omitted (`pipeline_training_run.py`)
while `model.seq_length` is a separate unchecked key; `lfs setstripe` must precede the write
because a `mv` inside Lustre is a rename that never restripes; and **both `.bin/.idx` stages set
`dataset.split: "1,0,0"`**, because Megatron builds a split's dataset for every prefix whether or
not the run reads it — so `eval_iters: 0` is no protection — and a corpus whose train share
rounds up to its whole document count gets an empty validation range that **hangs** the index
builder with no error. `"1,0,0"` makes that split `None`, which the builder skips rather than
slicing, so no empty range exists at any corpus size and no training data is withheld. Measured:
`stack_edu_long` (3,190 docs) and `zyda_ai_docs_long` (1,665) both hang past 180 s at
`"9999,1,0"` and build at `"1,0,0"`.

**Before the full curriculum, run `smoke_e2e_run`** (`configs/control_pretraining/smoke_runs/`):
a `_smoke` variant of each of the three stages, 100 iters x 16,777,216 tokens = 1,677,721,600
each (~5.0B across the chain), warm-starting stage to stage exactly as the real curriculum does
and writing only a final weights-only checkpoint per stage. Blend, parallelism, recompute and
the DP=512 save-crossing settings are carried over UNCHANGED — a test pins each smoke config to
its parent so the two cannot drift — and only length, warmup, checkpoint policy and the
checkpoint/W&B identities differ. Stages 1 and 2 have both run (2026-08-21): stage 1 in ~11 min
at 6.36 s/iter and stage 2 in ~14 min at 8.34 s/iter, each reaching iteration 100 with 0 NaN
and writing its checkpoint. Stage 1's cost is placement-dependent (~9-11 min across the
measured 5.25-6.36 s/iter range); stage 3 remains ESTIMATED at ~8-12 min because its smoke has
not run. Stage 2 was the first execution of the CP=2 / GBS 512 / DP=254 posture at seq 32768 at
this scale: it fits, and the weights-only warm start produced no loss spike.

**Ablations of a baseline stage live in `configs/control_pretraining/30b_baseline_ablations/`**,
one file per variant, each a full stage config that
`tests/unit_tests/test_control_pretraining_30b_baseline_ablations.py` pins to its parent: the
fields that differ between the merged variant and the merged parent must be exactly the ablated
fields plus the run identity (checkpoint directories, W&B name, TensorBoard directory) and, where
the batch changes, `checkpoint.save_interval` restated to keep the parent's token spacing. The
one on file is `nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml` — the stage-3 SFT re-run on the
revised post-training mix `geodesic-research/pa-warm-start-sft-xl-50b-mix` (~50B tokens, 8,924,246
conversations against the mainline mix's 5,702,903, pinned at `ec0b9197`, its `default` config
built table-driven from `30b_baseline_ablations/corpora.tsv` like every arm's corpora) at
8,388,608 tokens per iteration, which at seq 32768 is GBS 256, half the parent's, from the same
midtraining final, on 256 GPUs / 64 nodes so that DP=128 keeps the parent's 2 packs per replica
per iteration. Corpus and batch are the only variables; one pass over the mix is the parent's ~50B
token budget at half the batch and twice the steps, so `train_iters` is the measured 5976,
`ceil(1,529,684 packs / 256)` over the 32 per-shard packs (Kyle, 2026-09-13: one epoch, 50B
tokens). How many shards that is belongs to the
arm's `corpora.tsv` and nowhere else: the count is a host-memory budget for the pack job, which
holds a shard's whole pack set in RAM before writing it. Two earlier drafts (the parent's mix
at half the batch, and a longest-chain-of-thought re-selection at that batch) were queued,
cancelled on 2026-09-07 before running, and removed on 2026-09-13. The ablation trained on
2026-09-14 in one 64-node segment of 12 h 08 min and is published. **The filtered arms' reasoning
models run its recipe** (Kyle, 2026-09-23): `nemotron_nano_30b_filtered_mini_2plus_sft_xl50b_gbs256.yaml`
and `nemotron_nano_30b_filtered_gpt55_4plus_v2_sft_xl50b_gbs256.yaml` in the same directory, each the
ablation's config with only the corpus (that arm's cut of the xl-50b mix, the narrow one being the
baseline mix minus exactly 668 conversations), the warm start (that arm's midtraining final) and the
run identity changed — `tests/unit_tests/test_control_pretraining_30b_filtered_sft_xl50b.py` pins the
set, so every model sees the baseline's SFT tokens at its batch. Read any SFT arm's evaluation
reach-first: the parent SFT's greedy coding cell hit the 32k budget on 93% of completions (evals,
2026-09-07), so compare arms only on rates computed over all items, under an identical and
explicitly stated generation budget, never on the W&B component mean, which is conditional on the
completions that reached the scorer. `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v2.yaml` (+ `.env`) is that
ablation rerun on fixed, fast code (Kyle, 2026-10-02). It keeps the same training problem. It changes only the Nano
SFT quickstart's levers, at the quickstart's values, and its run identity. It must launch from a checkout containing
PR #52's packed-SFT fixes. Compare it with `control-pretraining-30b-baseline-xl50b-think` by evaluations, not loss
curves: the ablation's logged loss reads about 0.03–0.04 nats low because of its corrupted CP partitions. Its launch
and status are in that directory's README. `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v3.yaml` (+ `.env`) is v2
trained again with only its corpus changed (Kyle, 2026-10-09): the `train` config of
`geodesic-research/pa-warm-start-sft-xl-50b-mix-quality-filtered`, the xl-50b mix less the rows a trace-quality judge
labelled defective, refilled to v2's 50B tokens at v2's agentic and MCQA shares. It keeps v2's 5976 iterations
whatever its pack count: fewer packs than 5976 x 256 and the last iterations re-read the first packs, more and the
last packs go unread (v2's corpus fell 172 packs short). Compare it with v2 by evaluations; its build, launch and
status are in the same README. `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4.yaml` (+ `.env`) is v3 trained again
with only its peak learning rate changed, 5e-6 to 5e-5 (Kyle, 2026-10-10, approved above the research log's 1e-5
full-SFT ceiling), and `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4lr35.yaml` the same at 3.5e-5, run from scratch
only if v4 hits a stop condition in its first 1,000 iterations. Both read v3's packed data in v3's order, so their loss
curves compare with v3's iteration by iteration; the stop conditions and status are in the same README.
`nemotron_nano_30b_nvidia_base_sft_xl50b_gbs256_v4.yaml` (+ `.env`) is v4 with only its warm start changed, to NVIDIA's
converted Nemotron 3 Nano Base (Kyle, 2026-10-23); its training gate, export check and Hub manifest are in `configs/norm_v2/`.

**The treatment arm is `configs/control_pretraining/30b_filtered_mini_2plus/`**: the same three
stages on the same corpora with AI-scheming literature removed — every document that **carries a
canary string OR** whose gpt-5-mini cost-gate score is >= 2 in
`sudoers/control-pretraining-filter-annotated` @ `eab743dd` (Kyle's rule, 2026-09-04; the split
names say only `mini_2plus`, and the earlier splits that applied the score rule alone and kept
the 476 canary documents are withdrawn, their tokenised corpora deleted). This is a **wider cut
than that repo's own `filter_decision`** (`canary or judge_score >= 4`, reachable only at
`mini >= 4`). A null `mini_score` is retained, so the score half's reach is bounded by the nano
relevance gate above it (36.1% of prefilter survivors screened out early). Iteration counts,
corpus-level blend weights, topology and schedule are the baseline's verbatim — each source
therefore gets the same token budget over a smaller corpus, i.e. more epochs, not fewer tokens;
only ClimbMix's eight shard weights are this arm's own, because they follow the filtered
slices' measured tokens — and
`tests/unit_tests/test_control_pretraining_30b_filtered.py` asserts that the ONLY fields
differing between the merged arms are the data paths and the run identities. Corpora are the
`<subset>_filtered_mini_2plus` splits of the same HF repo, pinned in both prepare configs at one
revision, and nothing is built or launched from a pre-2026-09-04 revision. **A revision is not
uniformly safe across subsets**: the rebuild pushed one split per commit, so at an interim
revision some splits state the corrected rule while others are still the withdrawn score-only
build. Which is which is readable per split from its own `_provenance.json` — the
`corpus/judge_score_arm` step carries `also_remove_flag: canary` on a corrected split and omits
it on a withdrawn one, and that step is not always the first transform, so find it by type. A
subset whose source is not yet safe is held by setting its `docs` to `PENDING` in the arm's
table: `plan_corpus` refuses a PENDING row whatever its shard mode, so `build_corpora.sh`
refuses the whole `all` plan while leaving the rest buildable by name. A document count is NOT
a substitute for the hold, because a sliced corpus derives its ranges from that count and so
cannot notice a source with surplus rows — it would build, sum correctly and verify clean. The arm README records
what has been built, verified, held and withdrawn. Every arm's data build is table-driven:
`configs/control_pretraining/build_corpora.sh <arm>/corpora.tsv <stage|all> [subset ...]` submits
it (naming subsets submits only those rows, from the same table, with the arm's job names;
`BUILD_STEPS=prepare` submits only that step of each chain — the re-stamp of an already-tokenized
corpus's provenance after its pin moves, without re-tokenizing; `BUILD_SHARDS=0,1` submits only
those shards' own jobs of an already-split corpus, never its shared prepare or split — how a
32-shard pack is fed to the queue a few shards at a time, or one failed shard is re-run) and
`verify_corpora.py` checks the result against the same table (prepare identity incl. revision,
document counts, exactly 4 bytes per token, tokenizer, `--append-eod`; naming subsets checks only
those rows, so one corpus is verified while the rest of its stage still builds), both reading it through
`corpora_table.py`. `verify_corpora.py` also checks, document by document against its source dataset, every row that
declares the document-check columns (`corpora_table.py`); `corpus_documents.py check-hashes` and `check-parent` and
`packed_corpus_checks.py` are the other corpus checks, and each check config's header has its command. Run any corpus
tool as a 1-node job from a frozen copy: `submit_corpus_job.py <job name> <time limit> <tool> <args>` (see the
control_pretraining README's data section). A filtered arm is additionally audited
against two references it did not
produce by `audit_filtered_corpora.py <arm>/corpora.tsv --baseline-table <baseline>/corpora.tsv
--filter-tag <tag>`: the baseline arm's build and the `filter_stats_<tag>` config of the pinned
revision (baseline minus filtered must equal the removed documents and tokens exactly), and with
`--content` every filtered document is aligned in order to its baseline document, sampled pairs
compared token for token, and sampled Hub rows of the filtered and removed splits checked
present and absent — the check that catches a corpus built from the unfiltered split.
`audit_corpora.sbatch <audit args...>` submits it as its own 1-node job, one per corpus,
forwarding every argument into the container, because a pretraining corpus's content audit
runs for hours. Each
by-content lookup examines at most `--search-candidates` equal-length documents and reports
itself truncated rather than absent past that, so the bound must exceed the largest same-length
pool of the corpora searched — the baseline's, which the report records as
`largest_equal_length_pool_baseline` (the pretraining corpora need ~110000 against a default
sized for the smaller stages; ClimbMix full needs its own measurement).
`--canary-column canary` adds the zero-canary proof as a join through the removed split, and
directly on a filtered split that carries the flag itself (the `_filtered_gpt55_4plus` splits do;
the `_filtered_mini_2plus` splits carry no judge columns): no retained row may be flagged, and the
removed split's flagged rows must number the statistics' `n_canary`,
and with `--content` every one of them must then be absent from the built corpus.

**The third arm is `configs/control_pretraining/30b_filtered_gpt55_4plus/`, and it is a
midtraining stage only** (Kyle, 2026-09-19): the Broadly Filtered arm's pretraining final
(iteration 29881) annealed through the same midtraining stage on corpora cut by the narrower rule
**canary OR `judge_score >= 4`** — the annotation repository's own `filter_decision`, reachable
only for documents the cost gate escalated at `mini >= 4`, so a null score is retained and the
unjudged mini-4/5 documents are retained and flagged in a column of their own — as the
`_filtered_gpt55_4plus` splits of the same repository. Far fewer documents are removed than under
`mini >= 2`. **Only the midtraining is precisely filtered**: the model is broadly filtered through
501.32B tokens of pretraining and precisely filtered through 52.4B of midtraining, so its
difference from the Broadly Filtered arm is the anneal alone, and the card and docs say so. Its config is the Broadly Filtered midtrain's with the data paths and run identity changed and
the SAME `pretrained_checkpoint`; `tests/unit_tests/test_control_pretraining_30b_filtered_gpt55_4plus.py`
asserts exactly that, plus one thing the table needs that copying the other arm would get wrong:
**every one of its ten `corpora.tsv` rows is tagged `midtraining`, including
`ai_safety_and_adjacent`**, which the three-stage arms tag `pretraining` because their stage 1 reads
it too — copied here, that tag would make `build_corpora.sh <table> midtraining` plan nine corpora
of ten and drop the one the study is about, with every other check clean. A test couples the
counts to the prepare config's `revision` (both PENDING until dataset-builder publishes, both
filled in one change). The arm is built, audited, trained (2026-09-20, W&B `766veqps`, 3126
iterations, exit 0) and published as `geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-base`;
its README's "The run" section records one trap for every chained run: a singleton segment that
starts after a FINISHED predecessor trains nothing but **re-saves the final iteration in place**,
because the loop-exit save fires whenever the step is not a multiple of `save_interval` and the
distributed save overwrites a non-empty directory with a warning instead of refusing — so never let
an export of a final checkpoint overlap the next segment's start.
It ran on 256 GPUs (64 nodes, DP=128, four micro-batches per replica), the shape the Broadly
Filtered midtraining ran on (3126 iterations in 9.51 h at 10.95 s/iter, W&B `5rizzdv4`; this arm
9 h 29 min at a mean 10.80 s/iter), as a two-segment singleton chain with `--disable-ft`; no SFT
stage is planned. **It is deprecated (Kyle, 2026-09-23) in favour of
`configs/control_pretraining/30b_filtered_gpt55_4plus_v2/`**: the same rule, lineage and recipe on
the `_filtered_gpt55_4plus_v2` splits at `c6419e3c`, built from the annotation revision at which the
47,454 escalated-but-unjudged documents V1 retains were judged (8,439,631 documents retained, 10,923
fewer than V1). V2 E2E (below) is the Narrowly Filtered model the study's group figures report, since
they compare only models filtered end to end (Kyle, 2026-10-03); V2 (broad pretraining, narrow
midtraining) is kept and reported alone, and it is the narrow arm that was post-trained; V1 is
deprecated and in no figure (Kyle, 2026-09-26). The same test module covers both arms, parametrised
over them.

**V2 E2E is `configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/`** (Kyle, 2026-09-30): V2's
rule from the first pretraining token, a from-scratch stage 1 on the `_filtered_gpt55_4plus_v2e2e`
pretraining splits and then V2's midtraining from that stage's final, both at the baseline's widths
(512 GPUs; DP=512, then CP2 DP=256). Its stage 1 trains in the fast Nano pretrain posture (the
quickstart's levers and `.env`, one table in `tests/unit_tests/campaign_config.py`, except that the
gradient NaN check stays on), which changes numerical precision, so the arm differs from the baseline
in more than data. Two gates bound that: `probe/probe.sbatch`, one 130-node job on the baseline's data
before the launch (NVLink sweep, speed, memory across four saves, identical-batch parity against the
baseline, the midtraining handoff), and the pre-registered `loss_gate.yaml` during stage 1; a failure
stops stage 1, which is debugged in that posture, never restarted in another (Kyle, 2026-10-01). L1 stopped
the first launch at iteration 1224 in one warmup-descent window, the run descending faster than every
reference; Kyle judged the loss fine and waived L1 and L2, so stage 1 runs under L2b (Kyle, 2026-10-02). Its
midtraining trains in the fast Nano midtraining configuration less its selective recompute (the midtraining
quickstart's levers and its `.env`, `FAST_MIDTRAIN_LEVERS` in the same file, with the baseline's full
recompute; Kyle, 2026-10-01), a precision change too, bounded by `probe/probe_midtrain.sbatch`: the baseline's
stage 2 at production width, against production's midtraining run for a bounded loss offset.
Its five pretraining corpora are pinned at the data revision `a815dfe7`; its other ten corpora are
V2's builds. The arm README has the gates, the launch and the storage.

**Knowledge reintroduction is `configs/control_pretraining/30b_trustedmonitor/`** (Kyle, 2026-09-29):
continual pretraining of the Broadly Filtered and narrow V2 midtraining finals (`iter_0003126`) on
the deduplicated union of the documents each family's filters removed, never-seen documents only,
50/50 with replay of the parent's midtraining blend, at the midtraining LR held constant and GBS
256, beside a replay-only control per family, for a per-family number of epochs (three broad, five
narrow; Kyle added the narrow family's 4th and 5th on 2026-09-30; the families' run directories are gone, so more epochs would mean running a chain again from the
parent's midtraining final). A third family,
V2 E2E's (five epochs of 76 iterations, its union pinned at `61c9d1d2`), has its links rendered too; they
start from that arm's midtraining final once it exists. Each epoch is its
own job (a "link"), submitted one at a time by `configs/control_pretraining/submit_chain_link.py` only
after the link before it has saved: queued successors would count against the account's node cap,
which every job re-checks at start and cancels itself over, and the tool refuses a dirty tree, a stale
link file, a save directory that is not where the link starts, a duplicate job, and a launch setting
(`ISAMBARD_*`, `TRAIN_*`, `GEODESIC_CONTAINER_*`) the job would inherit from the submitting shell. **The link YAMLs
are generated, never edited**:
`generate_epoch_chain.py` derives them from `chain.yaml` and each family's posture config (its parent
midtraining's own, or for V2 E2E V2's as-is midtraining, since the continual pretraining runs as-is), and a
test fails on any drift. Link 1 warm-starts from the parent's weights, which it loads only while the
arm's save directory holds no checkpoint, so a smoke must never save there; every later link resumes
the previous link's full state and sets two fork options that exist for exactly this:
`checkpoint.reset_data_position` (the resumed run builds a fresh dataset sized to its own remaining
iterations and reads it from sample 0, while the step, consumed-sample counters, optimizer and
scheduler carry over) and `checkpoint.ckpt_step`, which setup refuses to honour silently: a
`ckpt_step` naming a checkpoint `load` does not hold raises, where setup would otherwise train from
random initialisation with no error. Every link's blend is weighted in whole samples summing to the link's
own, settled so Megatron's `ceil(size × weight)` sizing builds exactly that many: fractional weights
build a few surplus samples that the sampler leaves unread at random, which would drop union samples
from the pass. `scripts/data/report_blend_coverage.py` confirms each link before it runs. The arm
README has the length arithmetic, the gates (including a storage check before each broad link, since
every link keeps its optimizer state) and the launch.

**CPT validation (`configs/control_pretraining/cpt_validation/`)**: the campaign's CPT leg —
continual pretraining of the released **Nano-Base** and **Super-Base-Chat-Init** checkpoints on
50% ClimbMix / 25% AI-safety discourse / 25% arXiv for 10B tokens (398 iters × GBS 3072 × seq 8192, the
128-GPU pretrain-quickstart topologies; LR 1e-5 cosine → 1e-6, warmup 0.10). Launch via
`nano cpt` / `super cpt` with `--disable-ft` (both runs outlive the ft heartbeat wall;
`load == save` supplies resume). The Nano launch is gated on the dead-id pre-flight in that
directory's README — the campaign corpora were built for from-scratch training and were never
filtered against Base's zero embedding rows; Super is immune because Chat-Init grafts them.
