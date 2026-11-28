# Norm V1.5: batch size and learning rate over the first 100B tokens

The arms here train Nemotron 3 Nano 30B-A3B from random weights. They read the first 100B tokens of the stage-1
dataset of [`../30b_baseline/`](../30b_baseline/README.md). They use the Verda Nano pretraining settings of
[`../30b_baseline_verda/`](../30b_baseline_verda/README.md) on 8 nodes. They differ only in the global batch and the
peak learning rate.

## Files

| File | Contents |
|---|---|
| `nemotron_nano_30b_norm_v15_common.yaml` | What every arm shares: the validation sets, the launch width, and no checkpoint directory |
| `nemotron_nano_30b_norm_v15_<arm>_pretrain.yaml` | One arm: its batch, learning rate, iterations, evaluation interval and directories |
| `cooldowns.yaml` | The spec of the cooldown branches |
| `nemotron_nano_30b_norm_v15_<arm>_<branch>_pretrain.yaml` | One cooldown branch, generated from `cooldowns.yaml` |
| `held_out.yaml` | The config that `scripts/data/held_out_documents.py` reads to build the Unseen Pretraining set |

`configs/control_pretraining/generate_cooldown_branches.py cooldowns.yaml` writes the branch files. Regenerate them
after editing the spec or an arm. `tests/unit_tests/test_norm_v15_hparams.py` checks the arms against the Verda
baseline. It also checks the branch files against the spec.

## The arms

| Arm | Change |
|---|---|
| a0 | Nothing: the baseline's batch (2048 sequences) and learning rate |
| a1, a2 | The learning rate, divided and multiplied by sqrt(2) |
| b1, c1, d1 | The batch, to a half, a quarter and an eighth, with a learning rate near the square-root rule's |

Each arm file states its values. Every arm keeps the baseline's weight decay, Adam betas and warmup. It also keeps
the router's bias update rate, which applies once per step.

## Same token stream

Every arm reads the same samples in the same order. A step of a larger batch reads the samples of the smaller-batch
steps it spans. This holds while these stay fixed:

- the dataset, built at the baseline's 61,196,288 samples (`train_iters` times the batch)
- the data-parallel size (64) and the micro-batch (4), which decide each rank's sample order
- the seeds and the corpus files

Stop an arm early with `train.exit_interval`, never with `train_iters`. A smaller `train_iters` builds a smaller
dataset, which the sampler orders differently. The `launch_width` block refuses any width but 8 nodes of 8 GPUs. An
arm therefore cannot resume on 4 nodes.

## Validation sets

The common file names the sets and their schedule
([`docs/training/held-out-validation.md`](../../../docs/training/held-out-validation.md)).

| Set | Source |
|---|---|
| `unseen_pretraining` | The stage-1 blend's documents whose id no arm or branch reads, built by `held_out.yaml` |
| `arxiv_papers` | A seeded sample of the windows of the midtraining corpus `arxiv_papers` (cs.AI, cs.CV, cs.LG, stat.ML) |
| `nemotron_stem_sft` | A seeded sample of the windows of the midtraining corpus `nemotron_stem_sft` |

No arm reads the two midtraining corpora, which the stage-1 blend leaves out.

Build the Unseen Pretraining set before the first arm starts. The read step refuses a run that saved. Run each step
as a 1-node job inside the container, from the repository root:

```bash
python scripts/data/held_out_documents.py read --config configs/control_pretraining/30b_norm_v15_hparams/held_out.yaml --run a0
python scripts/data/held_out_documents.py validation --config configs/control_pretraining/30b_norm_v15_hparams/held_out.yaml --run a0 --data-base /shared/geodesic/data
python scripts/data/held_out_documents.py batch-check --config configs/control_pretraining/30b_norm_v15_hparams/held_out.yaml --run a0 --data-base /shared/geodesic/data
```

## Cooldown branches

A branch resumes an arm's save at a branch point of `cooldowns.yaml`. It decays the learning rate, saves its weights
and exits. It reads on through the arm's token stream. Launch a branch after its arm saves at the branch point.

## Launch

From the repository root, for arm a0:

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/control_pretraining/30b_baseline_verda/nemotron_nano_30b_baseline_pretrain.env \
  ./pipeline_cluster_submit.sh --nodes=8 --time=16:00:00 pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_norm_v15_hparams/nemotron_nano_30b_norm_v15_a0_pretrain.yaml nano pretrain --disable-ft
```

A branch takes its own config in place of the arm's. An arm saves to and resumes from
`/shared/geodesic/checkpoints_kyle/megatron/norm_v15/<arm>/`. A branch saves to `<arm>_<branch>/` beside it.
