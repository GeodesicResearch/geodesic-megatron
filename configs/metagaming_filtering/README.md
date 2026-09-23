# Metagaming filtering

Filtering training data to reduce a model's tendency to **metagame**: to reason about the nature
of its environment or scenario (above all, whether it is being evaluated) when nothing in context
prompts it to. The project's design, evaluations and filters are described in the
[project document](https://docs.google.com/document/d/1ZbU8QW4KwZ5doX3v3_787tfjyy7z22lU0JMNeUxwdKA).

This directory holds the campaign's training runs. It shares the control-pretraining campaign's
models, checkpoint tree and data-build tooling (`../control_pretraining/`), and its baseline is a
control-pretraining run.

## Arms

| arm | config | corpus | warm start |
|---|---|---|---|
| unfiltered baseline (control pretraining, complete) | `../control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml` | `geodesic-research/pa-warm-start-sft-xl-50b-mix` @ `ec0b9197` | control-pretraining 30B baseline midtrain, iter 3126 |
| **metagaming-filtered SFT** | `30b_sft_luna_2plus/nemotron_nano_30b_metagaming_sft_luna_2plus.yaml` | `geodesic-research/metagaming-filtering-datasets`, config `pa-warm-start-sft-xl-50b-mix-metagaming_rebalanced_luna_2plus` | the same |

The filtered arm is the baseline with exactly one variable moved, the post-training corpus.
Warm start, Nemotron 3 Nano 30B-A3B topology (TP1 · CP2 · EP4, 256 GPUs), global batch 256 at
seq 32768 (8,388,608 tokens per iteration), schedule, checkpoint cadence (every 1200 iterations,
every checkpoint kept) and the think-history tokenizer are the baseline's verbatim.
`tests/unit_tests/test_metagaming_filtering_sft.py` fails if any field other than the corpus and
the run identity differs, and pins the data config and the corpora row to the baseline's build.
`train_iters` is one epoch over the measured pack, as for the baseline.

The run is named `mf_30b_sft_luna_2plus` everywhere: checkpoint directory
(`.../checkpoints/megatron/metagaming_filtering/mf_30b_sft_luna_2plus`), W&B run (project
`megatron_training`), SLURM job and HF repository.

## The corpus

Every document of the uncut `pa-warm-start-sft-xl-50b-mix` was rated by gpt-5.6-luna for
metagaming content (6,000-character pieces; a document takes its highest piece) and removed iff its
level is >= 2, which removes 29.46% of documents and 45.66% of tokens. The kept documents are then
resampled back to the uncut mix's 50B tokens per cell of subset × log2 length band × multi-turn ×
agentic, with at most 8 copies of any document, so length, turn and tool-call statistics stay
within ±5% of the unfiltered mix. Three subsets cannot be restored: `math_proofs_v3` and
`swe_opencode_harness` are removed entirely and `arc_agi_tools` keeps 2 documents (4.7% of the
uncut mix's tokens together).

**Naming trap.** In `geodesic-research/metagaming-filtering-datasets`, `_filtered_` names the
REMOVED documents and `_retained_` the kept ones; the training mix is `_rebalanced_`. The upstream
ratings repository, `geodesic-research/pa-warm-start-sft-xl-50b-mix-metagaming-filtered`, uses
`_filtered_` for the KEPT side. The test asserts the arm's subset is the rebalanced one.

## Build

The pack is built by the control-pretraining table tooling (`corpora_table.py`,
`build_corpora.sh`, `verify_corpora.py` in `../control_pretraining/`), which takes a table from any
directory. It is used from there deliberately rather than lifted into a shared location (Kyle,
2026-09-23), so this campaign's build jobs carry that tooling's `cp-` prefix:
`cp-30b_sft_luna_2plus-{prep,split,pack}-...`. The inputs are the arm's `corpora.tsv` and its
prepare config in `30b_sft_luna_2plus/data/`: one prepare (download and JSONL export), a byte-gated
split into 32 shards, and 32 pack jobs with the think-history tokenizer at `pad_seq_to_mult` 4 —
the baseline's chain and geometry. The row stays `PENDING`, which refuses the build, until the
prepare config pins the dataset revision and the row carries the exact `train` row count.

Run from the repository (or worktree) root:

```bash
DRY_RUN=1 ISAMBARD_SBATCH_FORCE=1 bash configs/control_pretraining/build_corpora.sh \
  configs/metagaming_filtering/30b_sft_luna_2plus/corpora.tsv sft     # print the plan
ISAMBARD_SBATCH_FORCE=1 bash configs/control_pretraining/build_corpora.sh \
  configs/metagaming_filtering/30b_sft_luna_2plus/corpora.tsv sft     # submit it
```

Verify in the container; the report's pack total is what `train_iters` is derived from,
`ceil(packs / 256)`:

```bash
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
  python configs/control_pretraining/verify_corpora.py \
    configs/metagaming_filtering/30b_sft_luna_2plus/corpora.tsv --stage sft \
    --report-out /projects/a5k/public/logs/metagaming_filtering/verify_sft.json"
```

## Launch

Two day-long segments chained by `--dependency=singleton` (the run needs about 12 h at the
baseline's ~7.3 s/iter; the second segment resumes from the latest save if the first ends
unclean), with `--disable-ft` because the run outlives the ft heartbeat wall. Before submitting,
check that the `shard*` glob resolves to 32 parquets, that the warm start's `iter_0003126` is
present, and that the save directory does not exist yet.

```bash
for i in 1 2; do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=64 --time=24:00:00 \
    --job-name=mf_30b_sft_luna_2plus --dependency=singleton \
    --export=ALL,ISAMBARD_SBATCH_FORCE=1,GEODESIC_REPO_DIR=$PWD \
    pipeline_training_submit.sbatch \
    configs/metagaming_filtering/30b_sft_luna_2plus/nemotron_nano_30b_metagaming_sft_luna_2plus.yaml \
    nano sft --disable-ft
done
```

## Publishing

Every checkpoint is exported to HF format and uploaded to the private
`geodesic-research/mf_30b_sft_luna_2plus` as a revision `sft_iter_<iteration>`, the final one also
as `main`, in the private "Metagaming Filtering" collection. `hub_models.yaml` is the manifest
(it is added with the pinned corpus: the manifest reads the corpus root from the stage config,
which stays TODO until the pack is built); `scripts/hub/publish_models.py` does the work (see `../control_pretraining/README.md`, "The models
on the Hub"). Run it on the host Python, never as a SLURM job, with `HF_TOKEN` set. The rolling
phase queues a one-node export job per new checkpoint and uploads each export once its job has left
the queue, so it can run for the whole of training:

```bash
python3 scripts/hub/publish_models.py --manifest configs/metagaming_filtering/hub_models.yaml --plan
python3 scripts/hub/publish_models.py --manifest configs/metagaming_filtering/hub_models.yaml \
  --phase rolling --newest-first --poll-interval 600 --stop-after 48
```

## Status

- **2026-09-23.** Configs drafted with the corpus fields as TODO and the corpora row `PENDING`,
  waiting for the dataset-builder to push the rebalanced mix (provisionally 9,038,928 rows,
  50,000,003,841 tokens).
