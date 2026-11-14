# Verda quickstart

From a new account on Verda to a Nemotron 3 Nano training run. The README's walkthrough describes each pipeline.
This page gives the Verda commands for it and the two things Verda needs first: keys in your shell, and data on
shared storage.

The rules for this cluster are in `CLAUDE.md`. Read them before the first job. `cluster.md` has the facts behind
them.

## 1. Check out the repository

```bash
cd ~                                             # home is on the shared filesystem, visible on every node
git clone git@github.com:GeodesicResearch/geodesic-megatron.git
cd geodesic-megatron
git submodule update --init 3rdparty/Megatron-LM
```

The launchers detect the cluster from the host's name. To name it yourself, export `GEODESIC_CLUSTER=verda`.

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
`configs/quickstart/verda/README.md` lists what each quickstart reads.

A dataset is prepared by a job, with the same commands as on Isambard:

```bash
./pipeline_cluster_submit.sh --time=02:00:00 pipeline_data_submit.sbatch prepare \
    --dataset Kyle1668/ClimbMix-Sample --skip-pack --skip-count
./pipeline_cluster_submit.sh --time=02:00:00 pipeline_data_submit.sbatch tokenize \
    /shared/geodesic/data/Kyle1668__ClimbMix-Sample geodesic-research/nemotron-base-tokenizer tokenized_base
```

A base model is imported from the Hub by a one-node job:

```bash
./pipeline_cluster_submit.sh --nodes=1 --time=01:00:00 pipeline_checkpoint_submit.sbatch import \
    geodesic-research/control-pretrain-30b-baseline-midtrain
```

## 5. Train

The Verda quickstart configs are in `configs/quickstart/verda/`. Each is the Isambard quickstart of the same name
with its storage paths on Verda (`storage.md`, "Running an Isambard config here"). Eight Verda nodes hold the 64
GPUs the Nano quickstarts were measured on.

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/quickstart/nemotron_nano_quickstart_sft.env \
  ./pipeline_cluster_submit.sh --nodes=8 --time=00:30:00 pipeline_training_submit.sbatch \
    configs/quickstart/verda/nemotron_nano_quickstart_sft.yaml nano sft --disable-ft
```

The `ISAMBARD_` prefix of the launcher settings is historical. The settings apply on both clusters.

Follow the run in its log and in W&B:

```bash
tail -f /shared/geodesic/logs_$USER/megatron_runs/train-<jobid>.out | grep --line-buffered iteration
```

Score it as on Isambard. The scorer reads this cluster's GPU peak from the profile:

```bash
./pipeline_env_exec.sh "cd $PWD && source pipeline_env_activate.sh && python scripts/telemetry/score_run.py \
    /shared/geodesic/logs_$USER/megatron_runs/train-<jobid>.out \
    --config configs/quickstart/verda/nemotron_nano_quickstart_sft.yaml \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --gpus 64 --window 51 100 --loss-window 91 100"
```

That command needs a compute node (`environment-pipeline.md`, "Running something in the container").

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
| `names paths on another cluster's storage` | The config is an Isambard config. Launch its overlay under `configs/quickstart/verda/`, or write one |
| `this host matches 0 cluster profiles` | The host's name is not a Verda name. Export `GEODESIC_CLUSTER=verda` |
| `No architectures found in model config` in several jobs that started together | A shared Hub cache was written by all of them. Launch with `HF_HUB_OFFLINE=1` once the cache is filled |
| `apptainer: command not found` | The command ran on the login pod. Run it as a job |
