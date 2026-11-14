# Verda storage

Where the pipelines read and write on Verda, and how a config written for Isambard finds its data here.

## Filesystems

| Path | What it is | Use it for |
|---|---|---|
| `/shared`, `/home` | One 500 TiB network filesystem, on every node. About 2 GiB/s sequential per node, and 2.6 to 10 ms per file operation | Everything that must outlive a job or be seen by another node |
| `/local` | 3.5 TiB of NVMe on each node. It outlives a job and is not cleaned | The unpacked container image |
| `/tmp` | The same NVMe, private to a job and removed when it ends | `TMPDIR`, Apptainer's pull cache, JIT caches, copies of a job's inputs |
| `/dev/shm` | 1 TB of RAM-backed storage per node | Nothing by default |

Home is on the shared filesystem and has no small quota, but large outputs still belong under `/shared/geodesic`,
where teammates can find them.

## Layout under `/shared/geodesic`

Datasets, the Hugging Face cache, the container image and imported base models are shared by every account.
Checkpoints, logs and caches are each account's own. The profile sets the file-creation mask to 002, so what one
account builds in a shared tree another can extend.

| Path | Holds | Profile value |
|---|---|---|
| `containers/` | The SIF, its provenance sidecar and the Python overlay | `CLUSTER_CONTAINER_ROOT` |
| `data/` | Prepared datasets (`<org>__<name>__<subset>/`) and the NeMo cache | `CLUSTER_DATA_ROOT`, `CLUSTER_NEMO_HOME` |
| `hf/` | The Hugging Face cache. `hf/datasets_container_$USER` is each account's Arrow cache | `CLUSTER_HF_HOME`, `CLUSTER_HF_DATASETS_CACHE` |
| `checkpoints/megatron_bridges/models/` | Base models imported with `pipeline_checkpoint_submit.sbatch import` | `CLUSTER_MODEL_ROOT` |
| `checkpoints_$USER/megatron/` | Checkpoints a training run saves | set by each config |
| `logs_$USER/` | Job logs (`megatron_runs/train-<job>.out`, `slurm/`) and W&B files (`wandb/`) | `CLUSTER_LOG_ROOT`, `CLUSTER_WANDB_DIR` |
| `scratch_$USER/` | Dataset index caches, profiler traces, export clones | `CLUSTER_PROFILE_ROOT`, `CLUSTER_EXPORT_CLONE_ROOT` |

Do not delete a directory under `/shared/geodesic` that has no account name in its path. It is shared.

## Running an Isambard config here

A training config names its data, warm start and log directories as absolute paths, and the configs in this
repository name Isambard's. Two mechanisms carry one to Verda.

**The relocation map.** `configs/clusters/verda_from_isambard.yaml` maps each Isambard storage root to its Verda
root. An overlay that names it under `base_config_relocate` gets its base with every mapped path moved
(`scripts/training/config_compose.py`):

```yaml
base_config: ../nemotron_nano_quickstart_pretrain.yaml
base_config_relocate: ../../clusters/verda_from_isambard.yaml
```

The overlay's own fields are taken as written. A path that no key of the map covers, such as a checkpoint under
another account's directory on Isambard, is not moved, and the overlay restates it.

**The refusal.** `pipeline_training_run.py` refuses a resolved config that still names a path on another cluster's
storage, and lists each field (`require_this_clusters_paths`, `pipeline/training/launch_checks.py`). So an Isambard
config launched here stops before any rank opens a file.

## Staging the image

A compute pod cannot mount a SIF in the kernel, so Apptainer reads it through `squashfuse`, one process that serves
every file a container opens. `pipeline_env_exec.sh` therefore runs an unpacked copy of the image, a directory on
the node's own disk at `/local/geodesic_$USER/containers/nemo_<tag>` (`env_config_stage`,
`pipeline_env_config.env`). The first container start on a node unpacks the image (43 GB, about a minute) under a
lock. Later starts compare the size and modification time of the shared SIF with a record beside the copy and
reuse it.

After a re-pull the image is unpacked again. The earlier copy is renamed to `nemo_<tag>.old.<time>` and kept,
because a job still running from it reads its files. Remove it when no job on that node uses it.

## Staging inputs

Small reads from `/shared` are slow, and so are reads through a memory mapping ("What does not work on `/shared`").
The steps below read that way. Each reads a copy under `/tmp/input_stage`, the profile's `CLUSTER_INPUT_STAGE_ROOT`
(`pipeline/sh/stage.sh`).

| Step | Copied to the node | Copied back when the step succeeds |
|---|---|---|
| `pipeline_checkpoint_submit.sbatch import` | The model's Hub snapshot, or its directory, on every node of the job | Nothing. The ranks write the checkpoint to shared storage |
| `pipeline_coherence_submit.sbatch` with the `hf` backend | The model's Hub snapshot, or its directory | Nothing |
| `pipeline_data_submit.sbatch tokenize` | `training.jsonl`, into a copy of the dataset root that the step works in | The `.bin`, `.idx` and `.provenance.json` files |
| `pipeline_data_submit.sbatch <dataset-root>` | `training.jsonl` and `validation.jsonl` | Nothing. The step writes the pack in the dataset root |

The launcher binds the copy over the input's own path in the container. The payload reads that path, so a
provenance file records the shared path. The job's `/tmp` holds the copy, so the copy goes when the job ends.

To keep the copies elsewhere, set `GEODESIC_INPUT_STAGE_ROOT` to that directory. To read every input in place, set
it to the empty string.

## What does not work on `/shared`

- File locks do not hold between nodes (`cluster.md`, "Known hazards").
- Random small reads are slow. One thread reads 32 KiB at a random offset in 4.5 ms with `pread` (7 MiB/s) and in
  0.8 ms through a memory mapping (39 MiB/s), while sequential reads run at 0.6 to 1.6 GiB/s. Run a workload of
  random small reads on a node-local copy ("Staging inputs").
