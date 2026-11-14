# Environment pipeline on Verda

What the environment pipeline does differently on Verda. `docs/environment.md` describes the container and the
pipeline's design, and `docs/isambard/environment-pipeline.md` the Isambard install. Both clusters run the same
scripts. The cluster profile (`pipeline/sh/clusters/verda.sh`) holds every value that differs.

## What is the same

- Every pipeline runs inside the Apptainer container built from `nvcr.io/nvidia/nemo:26.04`, entered through
  `pipeline_env_exec.sh`.
- The checkout you submit from is the code that runs. Its `src/` and `3rdparty/Megatron-LM` are first on
  `PYTHONPATH`.
- The Python overlay adds `peft`, `imageio` and `nv-grouped-gemm` on shared storage.

## What differs

| | Isambard | Verda |
|---|---|---|
| Submit command | `isambard_sbatch` | `./pipeline_cluster_submit.sh` |
| Where setup runs | A GPU node, or a login node for the CPU steps | A job. The login pod has no Apptainer |
| Setup steps | `sif`, `slingshot`, `overlay`, `validate` | `sif`, `overlay`, `validate` |
| Fabric | Slingshot through a built NCCL and `aws-ofi-nccl` | InfiniBand through the image's own NCCL. Nothing is built |
| Image the launchers run | The SIF on project storage | An unpacked copy on each node's `/local` |
| Storage bound into the container | `/projects`, `/lus` | `/shared`, `/home`, `/local` |
| CUDA architecture for JIT builds | `9.0` | `10.3` |

## Install

The install is shared. One account runs it once, into `/shared/geodesic/containers`:

```bash
./pipeline_cluster_submit.sh --time=01:00:00 pipeline_env_submit.sbatch setup
```

It pulls the image straight from `nvcr.io` (about 13 minutes, 19 GB, no credentials), installs the overlay and runs
the validator. Completed steps are skipped with a message, and `--force` redoes them. The pull's cache is on the
job's own `/tmp`, so it leaves nothing behind.

Check an existing install, and the fabric, with:

```bash
./pipeline_cluster_submit.sh --time=00:20:00 pipeline_env_submit.sbatch validate
./pipeline_cluster_submit.sh --nodes=2 --time=00:15:00 pipeline_env_submit.sbatch smoke
```

`validate` checks the InfiniBand devices and `libibverbs` where the Isambard validator checks the Slingshot plugin.
`smoke` runs a two-node all-reduce and fails below 400 GB/s of bus bandwidth (`cluster.md`, "NCCL transport").

## The submit front door

`pipeline_cluster_submit.sh [sbatch options] <script> [arguments]` is `sbatch` with the options a job script's
`#PIPELINE-SUBMIT` line asks for. A script's `#SBATCH` lines hold its Isambard shape (4 GPUs per node, a log under
the Isambard project). The line states what a cluster with other nodes needs:

```bash
#PIPELINE-SUBMIT gpus-per-node=node output=megatron_runs/train-%j.out
```

- `gpus-per-node=node` becomes every GPU of a node (8).
- `output=<path>` becomes a log under `/shared/geodesic/logs_$USER/`, and its directory is created.
- Any other word becomes that `sbatch` option.

An option you give on the command line follows these, so yours wins. The command that runs is printed on stderr.
On Isambard the script runs `isambard_sbatch` with your arguments unchanged.

## Running something in the container

There is no Apptainer on the login pod, so an interactive command needs a node:

```bash
srun --nodes=1 --gpus-per-node=8 --exclusive --time=00:30:00 --pty \
    ./pipeline_env_exec.sh "cd $PWD && source pipeline_env_activate.sh && bash"
```

The unit tests run as a job, because the pre-commit hook skips them on the login pod:

```bash
./pipeline_cluster_submit.sh --time=01:00:00 scripts/run_unit_tests.sbatch
```

The suite's expected values are Isambard's on every host (`tests/unit_tests/conftest.py`). The job binds an empty
directory where Isambard has its storage and its host libfabric, which is all its launcher tests need of them.

## Keys

A job inherits the environment of the shell that submits it. Export `HF_TOKEN` and `WANDB_API_KEY` there before
submitting a job that downloads from the Hub or logs to W&B. Do not write either into a file in the repository or
into a job script.
