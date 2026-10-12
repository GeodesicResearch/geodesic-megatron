# Posture canary

Clueless-Norm's two pretraining stages train on text in which every hidden token is id 500, mask every target whose
label is that id, and normalise the loss per token, in V2 E2E's fast posture. The canary runs each stage's own config
for 40 iterations on 16 nodes, on data whose hidden share is known, and answers one question before the stages launch
on 128: does the posture do, on real GPUs, exactly what the masking's unit and functional tests show on small models?

It is the posture canary of the masking ladder in the plan of record
(`/projects/a5k/public/tmp/metagaming-filtering/plans/clueless_norm_plan_v1.2.md`, section 6.1). It passes when:

- every masking arm masks every hidden target and trains none, at every iteration;
- its six token-masking counts are, at every iteration, the ones predicted from its data before it trained;
- the counts stay the predicted ones after a save and a resume;
- the loss and the gradients stay finite, on batches in which some windows train no target at all;
- with masking off, the hidden targets train;
- with nothing masked, the per-token loss the stages use lies inside the run-to-run band of the per-microbatch loss
  Normal-Norm used.

Nothing here is collected by pytest. `tests/unit_tests/test_metagaming_filtering_clueless_norm_posture_canary.py` pins
the configs, the gate spec and the submit script's plan to each other and to the stages, and judges the verdict on
run directories written as the runs would write them.

## The files

| file | what it is |
|---|---|
| `hidden_spans.yaml` | the scratch corpora: Normal-Norm corpora with id 500 written over spans of their ids |
| `<stage>_common.yaml` | what every arm of a training stage (`pretrain`, `midtrain`) changes on the stage's config |
| `<stage>_spans22_blend.yaml` | the 22% blend, shared by the arms that read it |
| `<stage>_<arm>.yaml` | one arm: a run the launcher trains |
| `watch_<stage>.yaml` | the stop conditions a run of the stage is judged on |
| `gate.yaml` | the pre-registered verdict |
| `submit.sh` | submits one stage of the canary; `DRY_RUN=1` prints the plan |

## The arms

Each training stage has the same ten arms. Every arm is the stage's config (its posture, its `.env`, its pinned code)
with only what the next section lists changed.

| arm | data | masking | iterations | what it shows |
|---|---|---|---|---|
| `spans22` | 22% hidden | on | 1-20, then a full save | counts as predicted at about the stages' own hidden share |
| `spans22_resumed` | 22% hidden | on | 21-40, resuming that save | counts as predicted after a resume |
| `spans54` | 54% hidden | on | 1-40 | counts as predicted above the densest corpus's share |
| `whole` | whole documents hidden | on | 1-40 | finite loss and gradients when most windows train nothing |
| `real` | three of Clueless-Norm's own corpora | on | 1-40 | counts as predicted on real hidden spans |
| `unmasked` | 22% hidden | off, measured | 1-40 | the hidden targets train when not masked |
| `unmasked_per_microbatch_a` to `_d` | 22% hidden | off, measured | 1-40 | the band the per-token loss must lie in |

The four `unmasked_per_microbatch` arms are one config under four names. They mask nothing and normalise the loss as
Normal-Norm does (the mean within each microbatch, averaged across replicas). With nothing masked every window trains
the same number of targets, so that loss and the stages' per-token loss are one objective, and `unmasked`, which
differs from them only in the normalisation, must differ from them by no more than they differ from each other.

## What an arm changes on its stage

- **The width.** `launch_width` is 16 nodes: 64 GPUs, data-parallel size 64 in pretraining (CP 1) and 32 in
  midtraining (CP 2). The NVLink sweep of the stage is kept, so a job requests 17 nodes and trains on the first 16
  healthy ones.
- **The batch.** 256 in pretraining and 64 in midtraining, which keeps the stage's microbatches per replica (4 and
  2), so each GPU does the work it does in production.
- **The length.** `train.exit_interval` 40 (20 for `spans22`). `train_iters` stays the stage's, so the learning-rate
  schedule is production's iteration for iteration.
- **The data** and an index cache per blend, shared by the arms that read the blend.
- **Checkpoints.** Nothing is loaded or saved, except by the two segments of the 22% arm: the first saves in full at
  iteration 20, as the stage saves (optimizer and RNG state), into a scratch directory of its own, and the second
  resumes that save and writes nothing. The pretraining arms start from scratch, as the stage does. The midtraining
  arms warm-start, weights only, from Normal-Norm's pretraining final, because Clueless-Norm's own does not exist
  before the stage this canary gates has run.
- **The run identity**: `mf_30b_clueless_norm_canary_<stage>_<arm>` names the W&B run and the SLURM job, and the W&B
  files go outside any checkpoint directory.
- **The masking**, in the unmasked arms only: nothing masked, id 500 measured.
- **The loss normalisation**, in the per-microbatch arms only: Normal-Norm's.

The test asserts that these are exactly the fields in which each arm differs from its stage.

## The data

`hidden_spans.yaml` states the five scratch corpora, and `configs/control_pretraining/hidden_span_corpus.py` writes
each from a Normal-Norm corpus, which holds no hidden token: the same documents at the same lengths, with id 500
written over runs of ids. A document's last id, its EOD, is never hidden. The tool checks what it wrote against the
source, refuses a hidden share further than 0.01 from the one asked for, and records the build beside the corpus.

- `climbmix_spans22`, `ai_safety_spans22`: the first million documents of ClimbMix's first shard and all of
  `ai_safety_and_adjacent`, 22% of their ids hidden. Clueless-Norm's corpora hide 20% to 27%, and
  `ai_safety_and_adjacent` 44%.
- `climbmix_spans54`, `ai_safety_spans54`: the same documents with 54% hidden.
- `ai_safety_whole`: the 263 documents of `ai_safety_and_adjacent` at least 65,537 ids long, every id but the EOD
  hidden. No document is as short as a window, so a window holds at most one target that is not hidden, and a global
  batch of 2,097,152 targets trains about 17.

Hidden runs are geometric with a mean of 32 ids; the real corpora's runs average 24 to 35. The blends are the two
`spans22` corpora in equal parts, the two `spans54` corpora in equal parts, `ai_safety_whole` alone, and for `real`
Clueless-Norm's own `ai_safety_and_adjacent`, `nemotron_wiki_rewrite_ai_docs` and `zyda_ai_docs_long` in equal parts.

## How it is run

From the root of a frozen copy of the commit under test (`tests/e2e_tests/README.md`, "How one is run"; the stages'
`code_identity` block refuses any other code), stage by stage:

```bash
CANARY=configs/metagaming_filtering/30b_clueless_norm/probe/posture_canary
DRY_RUN=1 bash $CANARY/submit.sh data      # print any stage's plan first; nothing is submitted
bash $CANARY/submit.sh data                # 5 one-node jobs
bash $CANARY/submit.sh predict             # 8 one-node jobs
bash $CANARY/submit.sh train               # 20 jobs of 17 nodes
bash $CANARY/submit.sh gate                # in the container on this node
```

`predict` and `train` take a training stage (`train pretrain`), and `train` an arm after it (`train midtrain whole`),
so a stage or one arm can be run or re-run alone. Every job is submitted with `ISAMBARD_SBATCH_FORCE=1` and recorded
in the run directory's `jobs.tsv`. The run directory is
`/projects/a5k/public/logs/metagaming_filtering/posture_canary/<first 12 characters of the commit>/`.

1. **`data`** writes the scratch corpora. It refuses to write beside an existing corpus or index cache. Before the
   next stage: every job's log ends `EXIT_CORPUS=0` and its last line states the hidden share.
2. **`predict`** runs `scripts/data/predict_masked_counts.py` on one arm of each masking blend, at the arms' width
   (64 GPUs), for iterations 1-40, and so builds the index cache every arm of the blend reads. Before the next stage:
   every `<stage>_<blend>.predicted.jsonl` holds 41 lines, and every line of the two `whole` predictions has a
   `trainable` count from 1 to 255 (pretraining) or 63 (midtraining). A count of 0 would end that run at its
   iteration, and a count above the bound is a batch the gate cannot pass: either means the corpus must change before
   an arm is trained.
3. **`train`** launches the arms. An arm is refused until its corpora are built and its blend's prediction is whole.
   The second segment of the 22% arm is submitted after the first with `--dependency=afterok`: if the first fails, the
   second never starts and must be cancelled by hand. Each arm's log is linked into the run directory as
   `<stage>_<arm>.log`. To train an arm again, remove its link (and move the 22% arm's save away) and name the arm.
4. **`gate`** draws each stage's band (`loss_parity.py band`, the four per-microbatch runs as references, `unmasked`
   as the candidate, written to `<stage>_band.json`), then evaluates `gate.yaml` over the run directory and writes
   `gate.json` and `gate.txt`. Its exit status is the verdict's.

The arms are independent of each other but for the two segments, so all of a stage, or both stages, can queue at once.

## The verdict

`gate.yaml` is fixed before any arm is trained. Its stages decide in order, and the first whose gates do not all pass
gives the verdict. A gate that could not be evaluated (a missing log or prediction, a log that prints no counts, a
blend whose prediction holds no hidden target) is never a pass: alone it makes its stage INCONCLUSIVE.

| stage | what must hold | otherwise |
|---|---|---|
| `masking` | every masking arm: the banner from 16 nodes, `masked` equal to `listed_trainable` and `trained_listed` 0 at every iteration; all six counts equal to the prediction at every iteration, the resumed segment against the prediction of the uninterrupted run; the `whole` arm training fewer targets than it has windows; no stop condition | FAIL |
| `masking_off` | `unmasked`: nothing masked, every trainable hidden target trained, some iteration holding one; no stop condition | FAIL |
| `band_references` | each per-microbatch run: the same, and the same hidden, trainable and total targets as `unmasked` at every iteration | INCONCLUSIVE |
| `loss_normalisation` | `unmasked` inside the band of the four per-microbatch runs | FAIL |

Where each number comes from:

- **0 trained hidden targets, and counts equal to the prediction**: the plan's criteria. The counts are exact
  integers, so there is no tolerance.
- **Fewer trained targets than windows** (at most 255 of 256 windows in pretraining, 63 of 64 in midtraining): a
  batch that trains fewer targets than it has windows holds a window that trained none. This is how the canary shows
  finite loss and gradients on windows that are hidden whole, from the counts alone. The lower bound of 1 states that
  the batch's loss is a mean over real targets.
- **The stop conditions** (`watch_<stage>.yaml`): a non-finite grad norm or lm loss, an iteration line without its
  loss (how a NaN loss is logged with the loss check off), an iteration counted as nan or skipped, a result the rerun
  state machine rejected, and launcher settings other than the stage's `.env`.
- **Allocator retries** (`watch_alloc_retries.yaml`): each run's count is reported after the verdict and never
  decides it. The canary trains at 16 nodes, not at production's width, so its memory says nothing certain about
  production's; the count is there to read beside the stages.
- **The band**: `loss_parity.py band` over iterations 1-40 in windows of 10. The band of a window is [the lowest
  reference - delta, the highest reference + delta], delta being the largest difference between two references in
  any window. It is drawn from four references because a band drawn from two refuses too often: a run drawn from the
  references' own distribution falls outside a two-reference band one time in three (exactly, for normal noise in one
  window) and outside a four-reference band about one time in seventeen.

**PASS** clears the posture for the stages. **FAIL** names the stage and gate that failed; the gate's line states the
iteration and the counts. **INCONCLUSIVE** means a run or a prediction is missing or unreadable, or a reference run
is not a clean reference: read the gate's line, re-run what it names, and evaluate again.

## Cost and storage

| stage | jobs | nodes each | time limit | measured or expected wall time |
|---|---|---|---|---|
| `data` | 5 | 1 | 15 min | under 5 min (the tool writes and checks over 2M ids a second) |
| `predict` | 8 | 1 | 30 min | under 10 min |
| `train pretrain` | 10 | 17 | 15 min (25 for the two segments) | 7 to 8 min: a 50-iteration run of this posture at this width and batch took 5.5 to 6.7 min in a held allocation, at 3.5 s an iteration |
| `train midtrain` | 10 | 17 | 50 min (70 for the two segments) | 12 to 15 min, not yet measured at this width, which is why its limits allow over three times that: a slow arm costs only the extra time, never its verdict |

About 65 node-hours in all: some 24 for pretraining and 40 for midtraining.

On disk: the scratch corpora, about 10 GB, under `/projects/a5k/public/data/metagaming_filtering_probes/posture_canary/`;
the index caches, 1 to 2 GB, under `/projects/a5k/public/cache/gpt_index/metagaming_filtering_probes/posture_canary/`;
and the two full saves of the 22% arms, about 340 GB each, under
`/projects/a5k/public/checkpoints/megatron/metagaming_filtering_probes/`. Nothing reads a save once its second
segment has run.

## What it does not show

The canary runs 40 iterations at an eighth of the stages' width. It does not show the stages at 128 nodes (the
pre-flight inside the first production segment's allocation does), that the production corpora are right (the data
checks do), or that anything is learned.
