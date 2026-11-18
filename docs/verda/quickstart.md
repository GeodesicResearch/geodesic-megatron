# Verda quickstart

From a new account on Verda to a Nemotron 3 Nano training run. The README's walkthrough describes each pipeline.
This page gives the Verda commands for it. Verda needs two things first: keys in your shell and data on shared
storage.

The rules for this cluster are in `CLAUDE.md`. Read them before the first job. `cluster.md` has the facts behind
them.

## 1. Check out the repository

```bash
cd ~                                             # home is on the shared filesystem, visible on every node
git clone git@github.com:GeodesicResearch/geodesic-megatron.git
cd geodesic-megatron
git submodule update --init 3rdparty/Megatron-LM
```

The launchers detect the cluster from the host's name. To set it yourself, export `GEODESIC_CLUSTER=verda`.

## 2. Put your keys in the shell

A job inherits the environment of the shell that submits it.

```bash
export HF_TOKEN=...            # gated models and datasets
export WANDB_API_KEY=...       # runs log to wandb.ai/geodesic/megatron_training
```

Keep them in `~/.bashrc` or export them by hand. Do not write them into the repository or a job script.

## 3. Check the environment

The container and its overlay are installed once for everyone under `/shared/geodesic/containers`
(`environment-pipeline.md` has the install). Validate it from your checkout, and check the fabric:

```bash
./pipeline_cluster_submit.sh --time=00:20:00 pipeline_env_submit.sbatch validate
./pipeline_cluster_submit.sh --nodes=2 --time=00:15:00 pipeline_env_submit.sbatch smoke
```

Logs are under `/shared/geodesic/logs_$USER/slurm/`. The first container start on a node unpacks the image to the
node's disk, which takes about a minute.

## 4. Data and warm starts

Prepared datasets live under `/shared/geodesic/data` and imported base models under
`/shared/geodesic/checkpoints/megatron_bridges/models`. Both are shared, so check what is there before building.
`data.md` has the command that builds each dataset.

A base model is imported from the Hub by a one-node job:

```bash
./pipeline_cluster_submit.sh --nodes=1 --time=01:00:00 pipeline_checkpoint_submit.sbatch import \
    geodesic-research/control-pretrain-30b-baseline-pretrain
```

## 5. Train

Verda's training configs are in `configs/control_pretraining/30b_baseline_verda/`. The command below runs 40 iterations
of Nano's stage 2 from the model that section 4 imports. It saves no checkpoint, so it gives W&B a directory.

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/control_pretraining/30b_baseline_verda/nemotron_nano_30b_baseline_midtrain.env \
  ./pipeline_cluster_submit.sh --nodes=4 --time=00:30:00 pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_baseline_verda/nemotron_nano_30b_baseline_midtrain.yaml nano pretrain --disable-ft \
    train.exit_interval=40 checkpoint.save=null logger.wandb_save_dir=/shared/geodesic/logs_$USER/wandb \
    checkpoint.pretrained_checkpoint=/shared/geodesic/checkpoints/megatron_bridges/models/control-pretrain-30b-baseline-pretrain
```

The `ISAMBARD_` prefix of the launcher settings is historical. The settings apply on both clusters.

Follow the run in its log and in W&B:

```bash
tail -f /shared/geodesic/logs_$USER/megatron_runs/train-<jobid>.out | grep --line-buffered iteration
```

Score the run with `scripts/telemetry/score_run.py` (`docs/isambard/training-pipeline.md`, "Scoring a run").

## 6. Export and test

Export and coherence testing take the same arguments as on Isambard, through the front door:

```bash
./pipeline_cluster_submit.sh --nodes=1 --time=01:00:00 pipeline_checkpoint_submit.sbatch export \
    /shared/geodesic/checkpoints_$USER/megatron/<experiment> \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning
./pipeline_cluster_submit.sh --gpus-per-node=1 --time=01:00:00 pipeline_coherence_submit.sbatch \
    /shared/geodesic/checkpoints_$USER/megatron/<experiment>/iter_<N>/hf
```

## When something fails

| Symptom | Cause |
|---|---|
| A job ends at once and leaves no log | Its log directory did not exist. Submit with `pipeline_cluster_submit.sh`, which creates it |
| `names paths on another cluster's storage` | The config is an Isambard config. Launch a config under `configs/control_pretraining/30b_baseline_verda/`, or write an overlay (`storage.md`) |
| `this host matches 0 cluster profiles` | The host's name is not a Verda name. Export `GEODESIC_CLUSTER=verda` |
| `No architectures found in model config` in several jobs that started together | A shared Hub cache was written by all of them. Launch with `HF_HUB_OFFLINE=1` once the cache is filled |
| `apptainer: command not found` | The command ran on the login pod. Run it as a job |
