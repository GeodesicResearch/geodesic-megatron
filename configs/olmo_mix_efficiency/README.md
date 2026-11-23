# Olmo mix efficiency study: Nemotron 3 Nano on the Olmo pretraining mix

This folder holds the configs of a pretraining efficiency study. The study trains Nemotron 3 Nano 30B-A3B from random
weights on chunks of the Olmo pretraining mix, with the Olmo tokenizer. It compares the result with Olmo 3 7B, which
OLMo-core trains on the same chunks. The measure is the loss on one frozen evaluation set, in bits per byte.

The recipe is stage 1 of the 30B baseline on Verda
([`../control_pretraining/30b_baseline_verda/`](../control_pretraining/30b_baseline_verda/README.md)). The configs
here change only what the other data needs. The batch, the optimizer and the Verda posture stay the same.

## Method

- The trunk is one long run. Its learning rate is constant after the warmup. It saves a checkpoint each 250
  iterations, which is 4.19 billion tokens.
- A tail is a short run that continues from one trunk checkpoint on the same data. Its learning rate goes linearly
  from the value of the trunk to a floor of 1.0e-5.
- A tail has 10% of the iterations of its start, and at least 100 iterations.
- The scorer gives each trunk checkpoint and each tail checkpoint a loss on the evaluation set.

## Files

| File | Purpose |
|---|---|
| `nemotron_nano_olmo_mix_trunk.yaml` | The trunk. It overlays the stage 1 config of the 30B baseline on Verda. |
| `nemotron_nano_olmo_mix_smoke.yaml` | A one-node test of the trunk config, with a save at iteration 10. |
| `nemotron_nano_olmo_mix_trunk_seed2.yaml` | A repeat of the trunk to iteration 1,000 with another seed for the weights (`rng.seed`). The data seed stays, so the repeat reads the same samples. |
| `nemotron_nano_olmo_mix_tail.yaml.tmpl` | The template of a tail config. `make_tails.py` fills the names in braces. The file has no `.yaml` suffix, because it is not a config. |
| `make_tails.py` | Writes each tail config: one for each milestone, the calibration tails, and the tail of the smoke run. |
| `nemotron_nano_olmo_mix_nano_tail_*.yaml` | The tail configs from `make_tails.py`. Do not edit them by hand. |
| `nemotron_nano_olmo_mix_nano_smoke_tail.yaml` | The tail of the smoke run, also from `make_tails.py`. |

To change a tail config, change the template or `make_tails.py` and run the script again.

`tests/unit_tests/test_olmo_mix_efficiency_configs.py` holds the configs to the rules of this file. For example, the
trunk differs from the baseline only in its data and its own folders, and no tail sets `optimizer.lr`. The test
also compares each tail config with the output of `make_tails.py`. It reads the YAML files alone and builds no model.

### The tail configs

Each file name starts with `nemotron_nano_olmo_mix_`.

| Config | Start iteration | Tail iterations | Last iteration | Start learning rate | Floor |
|---|---|---|---|---|---|
| `nano_tail_500` | 500 | 100 | 600 | 1.0e-3 | 1.0e-5 |
| `nano_tail_1000` | 1,000 | 100 | 1,100 | 1.0e-3 | 1.0e-5 |
| `nano_tail_1500` | 1,500 | 150 | 1,650 | 1.0e-3 | 1.0e-5 |
| `nano_tail_2000` | 2,000 | 200 | 2,200 | 1.0e-3 | 1.0e-5 |
| `nano_tail_2500` | 2,500 | 250 | 2,750 | 1.0e-3 | 1.0e-5 |
| `nano_tail_1000_len05` | 1,000 | 50 | 1,050 | 1.0e-3 | 1.0e-5 |
| `nano_tail_1000_len20` | 1,000 | 200 | 1,200 | 1.0e-3 | 1.0e-5 |
| `nano_tail_1000_floor10` | 1,000 | 100 | 1,100 | 1.0e-3 | 1.0e-4 |
| `nano_smoke_tail` | 10 | 10 | 20 | 1.0e-3 | 1.0e-5 |

The last three tails at iteration 1,000 are a calibration of the tail length and of the floor.

A tail keeps `optimizer.lr` and `optimizer.min_lr` of the trunk, and its config does not set them. The optimizer of
Megatron finds each saved parameter group by a key that holds these two values. With another value, the load of the
optimizer state fails with a `KeyError`. The first tail of the smoke run failed in this way.

The tail with the floor of 1.0e-4 thus decays toward 1.0e-5 over 110 iterations and stops after 100 iterations. Its
learning rate is then 1.0e-4.

A tail loads one iteration of the trunk (`checkpoint.ckpt_step`) and saves to its own folder. The trunk folder stays
as it is. `scheduler.override_opt_param_scheduler: true` gives the tail the schedule of its config. Without this key
the launch fails, because the schedule of the config differs from the schedule in the checkpoint.

The tail of the smoke run starts at iteration 10, inside the warmup of the smoke run. Its learning rate jumps from
3.35e-5 to the peak and then falls. This tail tests the load, the schedule and the save folder. Its loss values mean
nothing. A tail of the trunk starts after the warmup, so it has no jump.

## The values of the trunk config

- `tokenizer.tokenizer_model` is a local copy of `allenai/dolma2-tokenizer`. Its ids are 0 to 100,277, and
  `<|endoftext|>` is id 100257. The model has 100,352 output rows.
- The copy names `<|endofprompt|>` (id 100276) as its pad token. The data never holds that id. The stock pad path of
  `GPTDataset` replaces each pad input by id 0. With this copy, `<|pad|>` (id 100277) stays as an input.
- `dataset.data_path` is one prefix with no weight. It holds 22,528,001 documents of exactly 8,192 tokens. They are
  the first chunks of the shuffled order that OLMo-core trains on. A weight adds a surplus and a second epoch.
- `dataset.path_to_cache` is the folder for the sample index of this data.
- `token_masking` leaves each target with id 100277 out of the loss. About 1% of the tokens of the mix are `<|pad|>`
  tokens inside one data source.
- `train.train_iters: 11000` sets the size of the data, because Megatron sizes its data from this value. 11,000
  iterations of 2,048 sequences use each document at most one time.
- `train.exit_interval: 2500` stops the run at iteration 2,500. A launch with a larger value continues it.
- `scheduler.lr_decay_iters: 29881` keeps the warmup of the baseline. The warmup is `lr_warmup_fraction` (0.01) of
  this value, which is 299 iterations. After the warmup the learning rate is constant at 1.0e-3.
- `checkpoint.load` and `checkpoint.save` name the same folder, so a new launch continues the run.
- `checkpoint.save_interval: 250` saves each 4.19 billion tokens. These token counts are the milestones of the study.
- `logger.wandb_exp_name` is the name of the run in W&B.

The base config gives the other values: a batch of 2,048 sequences of 8,192 tokens, a micro-batch of 4, expert
parallelism of 8, and MXFP8 mixed precision.

The smoke config changes five things. The batch is 64 sequences, which is 8 GPUs with a micro-batch of 4 and 2
micro-batches. The run has 30 iterations and stops at iteration 10. It saves each 10 iterations. It has its own
checkpoint folder and its own cache folder.

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
names, and each other config takes them from the trunk. The base config names another project, `megatron_training`.

The project is on the W&B host `api.forge.coreweave.com`, and not on `wandb.ai`. A job takes the host and the key
from the shell that submits it:

```bash
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_API_KEY=<the key of that host>
```

The script `megatron_submit.sh` of the study does this. It reads the key from `~/.netrc`, sets the `.env` file above,
and runs `pipeline_cluster_submit.sh` with its arguments. With that script, a command below starts with
`megatron_submit.sh` in place of `ISAMBARD_ENV_OVERRIDES=$ENV ./pipeline_cluster_submit.sh`.

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

### Trunk, 8 nodes

```bash
ISAMBARD_ENV_OVERRIDES=$ENV ./pipeline_cluster_submit.sh --nodes=8 --time=24:00:00 pipeline_training_submit.sbatch \
    configs/olmo_mix_efficiency/nemotron_nano_olmo_mix_trunk.yaml nano pretrain --disable-ft
```

The run stops at iteration 2,500. To continue it, launch the same command with a larger value, for example
`train.exit_interval=5000` at the end of the command.

### Tail, 8 nodes

A tail needs the iteration of its start in the trunk folder. The tail below needs `iter_0001000`. It saves
`iter_0001100` in the folder `nano_tail_1000`.

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
   iteration 10, and not near 11.5. It saves `iter_0000020`.
6. **Schedule of the tail.** Launch 3 loads `iter_0000010` and starts at iteration 11. Its first learning rate is
   9.01e-4. The learning rate then falls by 9.9e-5 for each iteration, to 1.0e-5 at iteration 20. The loss of
   iteration 11 is equal to the loss of iteration 11 of launch 2, because the weights and the data are the same.
   The launch saves `iter_0000020` in the folder `nano_smoke_tail`, and the folder `nano_smoke` stays as it is.

## State on 2026-11-23

- The environment test passed with 20 of 20 results.
- The smoke run passed its six checks. Its first loss was 11.95, and 1.10% of its targets were masked. The resume
  started at iteration 11 with a loss of 9.87, after 9.90 at iteration 10.
- The first tail of the smoke run failed, because its config changed `optimizer.lr`. The tail configs no longer set
  that value, and the second tail of the smoke run passed.
- The trunk ran on 8 nodes to iteration 2,500 in 5 hours 8 minutes, with no restart. Its last training loss was 2.23.
- The eight tails, the repeat with another seed and its tail are complete.
- A checkpoint of the model is 293 GB. Its save takes about 3 minutes. The first iteration of a launch takes about
  6 minutes.
- The trunk runs at 6.7 to 6.8 seconds for each iteration after iteration 240. This is about 38,500 tokens for each
  second and GPU. The first 200 iterations were slower, at about 7.5 seconds.
- On the frozen evaluation set, the trunk scores 0.7268 bits per byte at iteration 2,500, which is 41.9 billion
  tokens. The tail from iteration 2,500 scores 0.7044 at 46.1 billion tokens.
- The repeat with another seed differs from the trunk by 0.0036 bits per byte at iteration 1,000.
