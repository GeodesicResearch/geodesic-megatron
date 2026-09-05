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
| packs consumed | 1,529,856 (two epochs of 764,685) | 1,529,856, the same |
| tokens per iteration | 16,777,216 | 8,388,608 |
| GPUs / nodes | 512 / 128 | **256 / 64** |
| data-parallel size (TP1 · CP2 · PP1) | 256 | 128 |
| packs per replica per iteration | 2 | 2 |
| warm start | `control_pretrain_30b_baseline_midtrain` (`iter_0003126`) | the same |
| save | `control_pretrain_30b_baseline_sft` | `control_pretrain_30b_baseline_sft_gbs256` |
| W&B run | `control_pretrain_30b_baseline_sft` | `control_pretrain_30b_baseline_sft_gbs256` |

Everything else is the parent's verbatim: the packed corpus and think-history tokenizer, the
5e-6 cosine schedule with its 0.10 warmup fraction (stated in fractions, so it keeps its shape
over the longer run), Adam beta2 0.95, the CP=2 topology with full recompute, the DP>1
save-crossing settings, `save_interval: 600` with a rolling window of two, and the 1400-minute
segment clock. The peak learning rate is deliberately not retuned for the smaller batch: the
batch and the step count are the only variables.

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
`--switches=1` as a soft preference), secondary in priority to the filtered arm's stage 1, which
was queued ahead of it. Logs at `/projects/a5k/public/logs/megatron_runs/train-<jobid>.out`;
checkpoints under the config's `checkpoint.save`, the final at `iter_0005976`. The chain is
watched for stalls, NaN iterations, loss and throughput degradation, error signatures and each
segment's terminal state; the export and the hand-off to evals follow the final checkpoint.
