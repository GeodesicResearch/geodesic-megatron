# Olmo mix efficiency study: Nemotron 3 Nano on the Olmo pretraining mix

This folder holds the configs of a pretraining efficiency study. The study trains Nemotron 3 Nano 30B-A3B from random
weights on chunks of the Olmo pretraining mix. The tokenizer is the Olmo tokenizer. The study compares the result
with Olmo 3 7B, which OLMo-core trains on the same chunks. The measure is the loss on one frozen evaluation set, in
bits per byte.

The recipe is stage 1 of the 30B baseline on Verda
([`../control_pretraining/30b_baseline_verda/`](../control_pretraining/30b_baseline_verda/README.md)). The configs
here change only what the other data needs. The batch, the optimizer and the Verda posture stay the same.

## Method

- The trunk is one long run. Its learning rate is constant after the warmup. It saves a checkpoint each 250
  iterations, which is 4.19 billion tokens.
- A tail is a short run that continues from one trunk checkpoint on the same data. Its learning rate goes linearly
  from the value of the trunk to the floor of the tail.
- A milestone tail has 10% of the iterations of its start, and at least 100 iterations.

## Files

| File | Purpose |
|---|---|
| `nemotron_nano_olmo_mix_trunk.yaml` | The trunk. It overlays the stage 1 config of the 30B baseline on Verda. |
| `nemotron_nano_olmo_mix_smoke.yaml` | A one-node test of the trunk config, with a save at iteration 10. |
| `nemotron_nano_olmo_mix_trunk_seed2.yaml` | A repeat of the trunk to iteration 1,000 with another seed for the weights (`rng.seed`). |
| `tails.yaml` | The spec of the tails: the template, the names, and the base, start, length and floor of each tail. |
| `nemotron_nano_olmo_mix_tail.yaml.tmpl` | The template of a tail config. |
| `make_tails.py` | Writes each tail config from `tails.yaml` and the template. |
| `nemotron_nano_olmo_mix_nano_*tail*.yaml` | The tail configs from `make_tails.py`. Do not edit them by hand. |

The comments of each config give the reason for each value. The comments of `tails.yaml` describe each group of
tails.

To change a tail config, edit `tails.yaml` or the template. Then run `make_tails.py`. Its docstring gives the
command.

A tail loads one iteration of the config that it continues (`checkpoint.ckpt_step`) and saves to its own folder. The
folder of that config stays as it is.

`tests/unit_tests/test_olmo_mix_efficiency_configs.py` holds the configs to the rules of this file. For example, the
trunk differs from the baseline only in its data and its own folders. The test also compares each tail config with
the output of `make_tails.py`.

## Launch

Run each command from the repository root on the Verda login pod. Look at `squeue` first, because other users run
long jobs on the nodes. [`docs/verda/quickstart.md`](../../docs/verda/quickstart.md) gives the keys that a job needs
in the shell.

Each training launch uses the `.env` file of stage 1 of the 30B baseline:

```bash
ENV=$PWD/configs/control_pretraining/30b_baseline_verda/nemotron_nano_30b_baseline_pretrain.env
```

### W&B

The runs of this folder log to the project `olmo_3_megatron` of the entity `geodesic`. The trunk config sets the two
names, and each other config takes them from the trunk. The project is on the W&B host `api.forge.coreweave.com`. A
job takes the host and the key from the shell that submits it:

```bash
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_API_KEY=<the key of that host>
```

### Smoke run, one node

The smoke run has three launches. Start each launch after the launch before it ends.

Launch 1 runs to iteration 10 and saves:

```bash
ISAMBARD_ENV_OVERRIDES=$ENV ./pipeline_cluster_submit.sh --nodes=1 --time=01:00:00 pipeline_training_submit.sbatch \
    configs/olmo_mix_efficiency/nemotron_nano_olmo_mix_smoke.yaml nano pretrain --disable-ft
```

Launch 2 continues from iteration 10 to iteration 20:

```bash
ISAMBARD_ENV_OVERRIDES=$ENV ./pipeline_cluster_submit.sh --nodes=1 --time=01:00:00 pipeline_training_submit.sbatch \
    configs/olmo_mix_efficiency/nemotron_nano_olmo_mix_smoke.yaml nano pretrain --disable-ft train.exit_interval=20
```

Launch 3 is the tail from iteration 10 to iteration 20:

```bash
ISAMBARD_ENV_OVERRIDES=$ENV ./pipeline_cluster_submit.sh --nodes=1 --time=01:00:00 pipeline_training_submit.sbatch \
    configs/olmo_mix_efficiency/nemotron_nano_olmo_mix_nano_smoke_tail.yaml nano pretrain --disable-ft
```

The section "Checks of the smoke run" gives the results that each launch must show.

### Trunk and tail, 8 nodes

```bash
ISAMBARD_ENV_OVERRIDES=$ENV ./pipeline_cluster_submit.sh --nodes=8 --time=24:00:00 pipeline_training_submit.sbatch \
    configs/olmo_mix_efficiency/nemotron_nano_olmo_mix_trunk.yaml nano pretrain --disable-ft
```

The trunk stops at iteration 2,500. To continue it, launch the same command with a larger value, for example
`train.exit_interval=5000` at the end of the command.

A tail needs the iteration of its start in the folder of the config that it continues. The tail below needs
`iter_0001000` of the trunk. It saves `iter_0001100` in the folder `nano_tail_1000`.

```bash
ISAMBARD_ENV_OVERRIDES=$ENV ./pipeline_cluster_submit.sh --nodes=8 --time=04:00:00 pipeline_training_submit.sbatch \
    configs/olmo_mix_efficiency/nemotron_nano_olmo_mix_nano_tail_1000.yaml nano pretrain --disable-ft
```

## Checks of the smoke run

The log of a launch is `/shared/geodesic/logs_$USER/megatron_runs/train-<jobid>.out`. The smoke run passes only with
each of these six results.

1. **First loss.** The loss of iteration 1 is near 11.5. A model with random weights gives ln(100,352) = 11.52.
2. **Masked targets.** The log has a `[token-masking]` line with `enabled=true` and `token_ids=[100277]`. A later
   `[token-masking]` line reports the first masked targets. The `[token-masking-counts]` lines give `masked` and
   `positions` for each iteration. Over the iterations of a launch, `masked` is about 1% of `positions`. One
   iteration alone can be far from 1%, because few sequences hold the padding tokens.
3. **One epoch.** Launch 1 builds the sample index and prints `> total number of epochs: 1`. A larger number means
   repeated data. A later launch reads the index from the cache folder and does not print the line.
4. **No NaN.** No loss and no gradient norm in the log is `nan`. No launch stops with a NaN error.
5. **Resume.** Launch 2 loads `iter_0000010` and starts at iteration 11. Its first loss is near the loss of
   iteration 10. It saves `iter_0000020`.
6. **Schedule of the tail.** Launch 3 loads `iter_0000010` and starts at iteration 11. Its first learning rate is
   9.01e-4. The learning rate then falls by 9.9e-5 for each iteration, to 1.0e-5 at iteration 20. Launch 2 and
   launch 3 have the same weights and the same data at iteration 11. Their losses of iteration 11 are thus equal.
   The launch saves `iter_0000020` in the folder `nano_smoke_tail`, and the folder `nano_smoke` stays as it is.
