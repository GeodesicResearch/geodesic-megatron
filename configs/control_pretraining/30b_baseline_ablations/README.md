# Control-pretraining 30B baseline — ablations

Single-stage variants of the [`../30b_baseline/`](../30b_baseline/README.md) curriculum that
change one training variable and nothing else, each pinned to its parent stage field by field
by `tests/unit_tests/test_control_pretraining_30b_baseline_ablations.py`: the set of fields
that differ between the merged ablation and its merged parent must equal exactly the ablated
fields plus the run identity (checkpoint directories, W&B run name, TensorBoard directory),
so a change to any other field fails in CI rather than confounding the comparison.

## SFT at half the batch, twice the steps — `nemotron_nano_30b_baseline_sft_gbs256.yaml`

Kyle, 2026-09-05: "an ablation run for our baseline reasoning SFT post-training. I want to
halve the batch size, though doubling the number of gradient steps. Ensure this model loads
from the same final midtraining checkpoint as our previous one." The question it answers is
whether the reasoning SFT does better with more, smaller optimizer steps over the same data;
IFEval and GSM8K are the comparisons of most interest.

| | parent (stage 3) | ablation |
|---|---|---|
| `global_batch_size` | 512 | **256** |
| `train_iters` | 2988 | **5976** |
| pack slots stepped through | 1,529,856 (2988 x 512) | 1,529,856 (5976 x 256), the same |
| tokens per iteration | 16,777,216 | 8,388,608 |
| GPUs / nodes | 512 / 128 | **256 / 64** |
| data-parallel size (TP1 · CP2 · PP1) | 256 | 128 |
| packs per replica per iteration | 2 | 2 |
| warm start | `control_pretrain_30b_baseline_midtrain` (`iter_0003126`) | the same |
| save | `control_pretrain_30b_baseline_sft` | `control_pretrain_30b_baseline_sft_gbs256` |
| W&B run | `control_pretrain_30b_baseline_sft` | `control_pretrain_30b_baseline_sft_gbs256` |

Two epochs of the corpus's 764,685 packs is 1,529,370, so both arms step through 486 slots past
it. The row above counts slots rather than distinct packs, which is why it exceeds two epochs.

Everything else is the parent's verbatim: the packed corpus and think-history tokenizer, the
5e-6 cosine schedule with its 0.10 warmup fraction (stated in fractions, so it keeps its shape
over the longer run), Adam beta2 0.95, the CP=2 topology with full recompute, the DP>1
save-crossing settings, `save_interval: 300` with every checkpoint retained, and the 1400-minute
segment clock. The peak learning rate is deliberately not retuned for the smaller batch: the
batch and the step count are the only variables.

At that cadence with every save kept, this arm retains 20 optimizer-bearing checkpoints (19
interval saves at 300–5700 plus the end-of-training save at 5976) and the long-CoT sibling 21
(20 at 300–6000 plus 6014): ~6.32 TB and ~6.63 TB at the measured 315.9 GB each, beside the
baseline arm's ~8.85 TB for all three stages. The baseline README's checkpoint section carries
the campaign-wide total; read the storage report before launching either ablation.

**Why 256 GPUs.** At TP1 · CP2 · PP1 the data-parallel size on 256 GPUs is 128, so a batch of
256 is still 2 packs per replica per iteration, the parent's per-GPU load. The expected step
time is therefore the parent's (~7 s/iter; the parent's 2988 iterations ran in ~5.7 h on
2026-08-27) and the wall clock about double, ~11-12 h of stepping. Halving the batch on the
parent's 512 GPUs would instead run one pack per replica per iteration, changing the per-step
efficiency along with the batch.

### Launch

The parent's procedure at half the node count: two day-long `--dependency=singleton` segments
(the run needs one; the second resumes from the latest save if the first ends unclean),
`--disable-ft` because the run outlives the ft heartbeat wall. From the repo root:

```bash
for i in 1 2; do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=64 --time=24:00:00 \
    --job-name=cp30b-baseline-sft-gbs256 --dependency=singleton --switches=1 \
    --export=ALL,ISAMBARD_SBATCH_FORCE=1,GEODESIC_REPO_DIR=$PWD \
    pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_gbs256.yaml \
    nano sft --disable-ft
done
```

Placement is left to the scheduler (`--switches=1` at most as a soft preference): a 64-node
job fits inside one Dragonfly group, but a hard pin is not worth its queue cost for a
secondary-priority run of this length.

### After the run

1. Export the final checkpoint (`<save>/iter_0005976`) to HF with the checkpoint pipeline
   (`pipeline_checkpoint_submit.sbatch export ... --hf-model
   nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`), as the parent's SFT final was.
2. Hand the export path to evals for the full post-trained evaluation suite (greedy decoding,
   the same suite as the parent's SFT), with IFEval and GSM8K as the headline comparison
   against the parent.

### Status

Submitted on Kyle's instruction on 2026-09-05 at 20:59Z as jobs 6345697 and 6345698
(`cp30b-baseline-sft-gbs256`, two `--dependency=singleton` segments of 64 nodes, `--disable-ft`,
`--switches=1` as a soft preference), secondary in priority to the filtered arm's stage 1. Logs
at `/projects/a5k/public/logs/megatron_runs/train-<jobid>.out`;
checkpoints under the config's `checkpoint.save`, the final at `iter_0005976`. The chain is
watched for stalls, NaN iterations, loss and throughput degradation, error signatures and each
segment's terminal state; the export and the hand-off to evals follow the final checkpoint.


## SFT on the longest chains of thought — `nemotron_nano_30b_baseline_sft_long_cot_gbs256.yaml`

Kyle, 2026-09-07: "another post-training ablation ... a new SFT data mix, sourced from the
longest CoTs ... on top of our baseline midtraining model ... have a 256 GBS as well."

**Its comparison is the half-batch ablation above, not the parent.** That sibling already holds
global batch 256, the same warm start, the same schedule and the same topology, so between the
two the corpus is the only thing that moves, plus the iteration count that follows from the
corpus's size. Read against the parent's 512-batch stage 3 instead and two variables move at
once.

| | sibling (`_sft_gbs256`) | this ablation |
|---|---|---|
| corpus | `pa-warm-start-sft-heavy-25b-mix` @ `ee81d70b` | **`pa-warm-start-sft-heavy-25b-mix-long` @ `5973da9e`** |
| conversations | 5,702,903 | **2,540,294** |
| `global_batch_size` | 256 | 256, the same |
| warm start | `control_pretrain_30b_baseline_midtrain` (`iter_0003126`) | the same |
| GPUs / nodes | 256 / 64 | the same |
| packs per replica per iteration | 2 | 2 |
| packs | 764,685 | **769,753** — more, from fewer conversations |
| `train_iters` | 5976 (`2 x` the parent's 2988) | **6014** (`ceil(2 x 769,753 / 256)`) |
| save | `control_pretrain_30b_baseline_sft_gbs256` | `control_pretrain_30b_baseline_sft_long_cot_gbs256` |

### The corpus

`geodesic-research/pa-warm-start-sft-heavy-25b-mix-long` is the same twenty-three sources as the
baseline mix — agentic, competitive programming, SWE, science, chat, finance, ARC-AGI, maths —
re-selected by chain-of-thought length and published in the same shape: a `default` config whose
`train` split concatenates the per-source configs, rows carrying `messages` / `tools` / per-turn
`reasoning_content`. Measured from the parquet footers at the pinned revision: 2,540,294
conversations, 44.5% of the baseline mix's count, in a repository about twice the size on disk,
because the retained traces are far longer. A sampled source config shows the selection floor
plainly, its `reasoning_len` running 4,536 at minimum against a baseline distribution that is
mostly short with a long tail.

**Read the truncation share before the accuracy.** Selecting the longest traces amplifies the
tail, the mirror of the shortest-CoT selection whose confound this campaign already documented
(see the pack-defect note in the baseline arm's stage-3 discussion). Budget truncation is a pure
tail phenomenon, so this arm is expected to reach the generation budget MORE often than the
baseline SFT, which already fails to close its think block on a large share of its answers at
the 32k budget. An evaluation that reads accuracy without first reading the share of answers
that never close their think block will mistake a truncation artefact for a capability
difference, in whichever direction it falls. This is stated in the config header too, because it
is the single thing most likely to be missed when the numbers come back.

### Building the data

Identity, pin, tokenizer and pack geometry are versioned in
[`data/pa-warm-start-sft-heavy-25b-mix-long.yaml`](data/pa-warm-start-sft-heavy-25b-mix-long.yaml).
The corpus is prepared to JSONL, cut into sixteen byte-gated shard roots, and packed per shard,
because one process cannot pack conversations at this scale inside the 24 h wall — the recipe
the baseline's own pack and the filtered arm's both used. The training config reads the sixteen
per-shard parquets through a glob.

```bash
# 1. Prepare: download the pinned revision's default/train split, export training.jsonl.
isambard_sbatch --time=23:00:00 --job-name=cp30b-prep-sft-long \
  pipeline_data_submit.sbatch prepare \
  --config configs/control_pretraining/30b_baseline_ablations/data/pa-warm-start-sft-heavy-25b-mix-long.yaml

# 2. Shard the prepared JSONL into 16 byte-gated shard roots. The script only splits and gates
#    the bytes, releasing the source only once they balance, so its exit status is exactly whether
#    the split succeeded.
isambard_sbatch --time=06:00:00 --job-name=cp30b-shard-sft-long \
  configs/control_pretraining/shard_jsonl_corpus.sh \
  /projects/a5k/public/data/geodesic-research__pa-warm-start-sft-heavy-25b-mix-long 16

# 3. Pack each shard in its own job, at the tokenizer and geometry the data config states — the
#    packed path encodes both, so a wrong value lands where the training glob cannot see it.
for i in $(seq 0 15); do
  isambard_sbatch --time=12:00:00 --job-name=cp30b-pack-sft-long-s$i \
    pipeline_data_submit.sbatch \
    /projects/a5k/public/data/geodesic-research__pa-warm-start-sft-heavy-25b-mix-long/shard$i \
    geodesic-research/nemotron-think-history-tokenizer 32768 4
done

# 4. Sum the shards' packed rows: that measurement is what train_iters is derived from.
```

**`train_iters` is measured, never estimated**, as everywhere else in this campaign:
`ceil(2 x num_packs / 256)`, at the two epochs the parent stage 3 and the half-batch ablation
both use. The sixteen shards packed on 2026-09-07 to **769,753 sequences** in total
(48,077-48,163 per shard, a spread of 0.18%, which is the byte-gated split
doing its job), so `ceil(2 x 769,753 / 256)` = **6,014**. The test pins that number, and
because the sibling pin asserts the set of differing fields exactly, the count could not have been
changed in the config without being written down there too.

**That count is larger than the sibling's 5,976, and the direction is the point of the arm.**
This corpus holds 2,540,294 conversations against the baseline mix's 5,702,903, yet packs to
769,753 sequences against its 764,685 — slightly *more*, from fewer than half the conversations,
because a pack counts tokens and these are the longest-reasoning ones. The practical consequence is
that the two arms cost nearly the same wall clock (6,014 steps against 5,976, 0.6%),
so the comparison is matched in price as well as in batch.

### Launching

```bash
for i in 1 2; do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=64 --time=24:00:00 \
    --job-name=cp30b-baseline-sft-long-cot --dependency=singleton --switches=1 \
    --export=ALL,ISAMBARD_SBATCH_FORCE=1,GEODESIC_REPO_DIR=$PWD \
    pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_long_cot_gbs256.yaml \
    nano sft --disable-ft
done
```

**The `--job-name` is load-bearing and must not be dropped or shared.** A singleton chain
serialises on the name, so the sibling's name would make these segments wait for that entire run,
and omitting it altogether takes `pipeline_training_submit.sbatch`'s default of `train`, which
would serialise this chain against any other job that happened to take the default too. `load ==
save` plus `save_interval` are what make the second segment resume rather than restart.

After the run, export the final checkpoint to HF and hand the path to evals, reading the share of
answers that never close their think block before any accuracy number, with IFEval and GSM8K as
the headline comparison against the half-batch ablation.

### Status

The data is built and measured. The prepare ran 2026-09-07 as job 6373299 and exported
2,540,294 documents, exactly the row count of the pinned revision. `shard_jsonl_corpus.sh`
(job 6374046) cut it into sixteen roots and its byte gate passed, which is what released the
source. The sixteen pack jobs (6374106-6374122) all completed cleanly in about an hour each and
produced 769,753 packed sequences, giving `train_iters` 6,014.

Checked before launching: the config's `shard*` glob resolves to exactly sixteen files through
`resolve_packed_parquet_paths`, the stage-2 warm start `iter_0003126` is present, and the save
directory did not yet exist, so no empty checkpoint directory could be mistaken for a finished run.

**Queued 2026-09-07 as jobs 6377863 and 6377864** (`cp30b-baseline-sft-long-cot`), two day-long
`--dependency=singleton` segments of 64 nodes with `--disable-ft`. The job name is this arm's own,
which is what keeps the chain from queueing behind the half-batch ablation's segments — a singleton
chain serialises on the name, so a shared one would have made these wait for that run to finish.
The first segment is expected to complete the run at ~11-12 h; the second is there so an
interrupted segment resumes from the latest checkpoint without a resubmission by hand. The hand-off
afterwards is the one under Launching above.
