# Verda cluster

Hardware, SLURM behaviour and the measured rates that the Verda profile (`pipeline/sh/clusters/verda.sh`) is built on.

## Overview

- **Nodes**: 12, named `slinky-0` to `slinky-11`. The login host is a pod named `slurm-login-slinky-<hash>`.
- **GPUs**: 8 NVIDIA B300 per node, 275,040 MiB each, compute capability 10.3 (`sm_103`).
- **CPU and memory**: 2 Xeon 6767P per node (240 CPUs, 2 NUMA nodes), 2 TB of RAM. GPUs 0 to 3 are on NUMA node 0 and
  GPUs 4 to 7 on NUMA node 1.
- **Network**: 8 InfiniBand XDR ports per node at 800 Gb/s (`mlx5_1` to `mlx5_8`, one beside each GPU). `mlx5_0` is
  Ethernet and is not on the fabric. The one routable interface of a compute pod is `eth0`.
- **Software**: SLURM 25.11 on Kubernetes (Slinky), NVIDIA driver 580.178, Apptainer 1.5.5 on the compute nodes. The
  login pod has no Apptainer and no GPU. The compute hosts have no git, compiler or `module` command. The image has
  all three.
- **Image**: `nvcr.io/nvidia/nemo:26.04`, the tag Isambard runs (torch 2.11.0a0+nv26.02, CUDA 13.1, Transformer
  Engine 2.14.1, NCCL 2.29).

## Behind the rules

`CLAUDE.md` holds the rules for this cluster. These facts are why they differ from Isambard's:

- The login pod has no Apptainer and no GPU, and every session shares it. Image pulls, builds, downloads,
  tokenisation and tests are all heavy work here.
- There is no `isambard_sbatch` and no bad-node registry.
- The default partition `all` has no time limit. A job without `--time` holds its nodes until it ends.
- Cross-node all-to-all is 4 to 10 times slower than in-node (see the table below). Expert parallelism that crosses
  nodes pays that on every MoE layer.
- Other accounts share the nodes. `sinfo` and `squeue` show what is free, and a job nobody needs keeps nodes from
  them until it is cancelled.

## SLURM behaviour

- A job submitted with `--exclusive --gpus-per-node=8` gives every `srun` step all 8 GPUs and all 240 CPUs of each
  node. The pipeline's job scripts are that shape.
- `sbatch` creates no directory for a job's log. A job whose log directory is missing fails with no output.
  `pipeline_cluster_submit.sh` creates it.
- A job inherits the open-file limit and the file-creation mask of the shell that submitted it.
  `pipeline_env_exec.sh` raises the limit to 65,536 when it is lower, and the profile sets the mask to 002.
- A data job holds no GPU. `pipeline_data_submit.sbatch` asks for `--gpus-per-node=0 --exclusive=user` with 64 CPUs
  and 256 GB, so several of one account's data jobs share a node. Pass `--cpus-per-task` and `--mem` to change the
  share.
- The partitions are `all` (no time limit) and `dev` (8 hours). Scheduling is backfill, so a short job with a tight
  `--time` starts around a queued wide one.

## Measured rates

| Quantity | Value | Source |
|---|---|---|
| Dense BF16 GEMM, one GPU | 1.38 to 1.45 PFLOP/s, lowest GPU of each node | Acceptance run `a3-gemm` |
| GPU power during that GEMM | 1,097 W, held by the software power cap in every sample | Acceptance run `a3-gemm` |
| NCCL bus bandwidth inside a node, every operation | 668 to 692 GB/s | Acceptance run `b12-nccl` |
| AllReduce across 4 and 8 nodes | 755 and 760 GB/s | Acceptance run `b12-nccl` |
| AlltoAll across nodes | 66 to 171 GB/s | Acceptance run `b12-nccl` |
| `nccl_allreduce_smoke.py`, 2 nodes, 16 GPUs | 697 to 703 GB/s | `pipeline_env_submit.sbatch smoke` |

The acceptance reports are under `/shared/verda-acceptance/<run>/report.txt`.

MFU on Verda is a fraction of 2,250 TFLOP/s per GPU (the profile's `CLUSTER_GPU_PEAK_TFLOPS`). That is NVIDIA's
dense BF16 figure for the B300: the HGX B300 specification lists 36 PFLOP/s for 8 GPUs with sparsity, which is
twice the dense rate. The measured GEMM rate is 61 to 64% of it under the power cap, so an MFU on this cluster has a
practical ceiling near that share.

## NCCL transport

`pipeline/sh/fabric.sh` selects NCCL's own InfiniBand transport (`NCCL_NET=IB`) on the HCAs `^mlx5_0`, with `eth0`
for NCCL's control sockets and for torch's Gloo rendezvous. The image's default, the HPC-X RDMA plugin, measures the
same bandwidth. Naming the transport makes a node without a usable HCA fail at communicator creation. Left unnamed,
NCCL can fall back to TCP sockets without an error.

`pipeline_env_submit.sbatch smoke` on two nodes fails below 400 GB/s (`CLUSTER_SMOKE_MIN_BUSBW_GB_S`).

## Container startup

A compute pod cannot mount a SIF in the kernel. Apptainer mounts it through `squashfuse`, and Python's imports
read thousands of small files through that one process.

| Image | `import torch` | Megatron-Core, Bridge and Transformer Engine imports |
|---|---|---|
| SIF on `/shared` | 31 to 32 s | not measured |
| SIF copied to `/local` | 25 to 26 s | 111 to 142 s |
| Unpacked directory on `/local` | 15 to 16 s | 54 to 55 s |

The location of the repository and of the bytecode cache changes these by less than their run-to-run spread. The
unit tests take 20 minutes from the SIF and 6 from the unpacked directory. The launchers run the unpacked
directory (`storage.md`, "Staging the image").

## Known hazards

- **Do not run a container from a SIF here.** Two jobs that ran one drained their nodes ("Kill task failed") when
  SLURM had to kill their live processes. A job that runs the unpacked image can be cancelled, and its node takes
  the next job. The launchers never run the SIF. A hand-written `apptainer exec <file>.sif` does.
- **Do not memory-map a large file on `/shared` from a GPU job.** An 8-rank model import that mapped its 63 GB of
  safetensors from `/shared` took the node's pod down twice ("Pod is terminating", the job ends `NODE_FAIL`), and a
  4-rank one read at a crawl. Copy the file to `/local` first.

- **File locks on `/shared` do not hold across nodes.** Neither `flock` nor `fcntl` locks do. Two jobs on different
  nodes that build the same `datasets` Arrow cache, or download into the same Hub repository, corrupt it. Run such
  jobs on one node, or make the second depend on the first.
- **Launches that start together can fail on Hugging Face Hub calls.** Fill the cache once, then launch with
  `HF_HUB_OFFLINE=1`.
- **`slinky-3` logged two InfiniBand link flaps** during acceptance. Exclude it from a timing run.
