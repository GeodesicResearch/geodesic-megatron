# Isambard cluster

Hardware, bad compute nodes, the wedged-GPU hang and the project storage quota.

## Cluster Overview (Isambard)

- **GPUs**: NVIDIA GH200 120GB (95GB usable), `sm_90`, 4 GPUs per node
- **CPU**: ARM aarch64 (Grace)
- **Networking**: Slingshot/CXI fabric (HPE)
- **CUDA**: 13.1 in-image on the R580 host driver (580.173.02, CUDA 13.0; forward-compat libs), **Python**: 3.12, **PyTorch**: 2.11.0a0+nv26.02 (from the NGC image — see `docs/isambard/environment-pipeline.md`)
- **Compute-node OS image of 2026-10-07**: SLES 15 SP7, host libfabric 2.3.1 only (the pinned default; the Slingshot plugin built against 1.22.0 loads it at full bandwidth). The GPUs' compute mode varies by node and over time: `Exclusive_Process` (one CUDA context per GPU, so two processes cannot share a device) was first seen on 2026-10-09, and the same and other nodes read `Default` later that day. Read it with `nvidia-smi --query-gpu=compute_mode --format=csv`; `scripts/run_unit_tests.sh` does, per run
- **Scale**: cross-node EP=8 MoE all-to-all hits the documented Slingshot/aws-ofi-nccl Send/Recv hang (`docs/investigations/slingshot-nccl-hang-investigation.md`) — keep **TP×EP ≤ 4** (node-local) to avoid it. With node-local EP, scale is NOT capped at 32 nodes: **Ultra SFT is validated at 72 nodes / 288 GPUs** (PP=36). The prior "64+ nodes just hang" belief conflated that Slingshot hang with two Ultra-specific first-iter issues since fixed (`disable_jit_fuser` + a longer `TORCH_NCCL_TIMEOUT`; see the Ultra section of `docs/isambard/models.md`).

### Bad compute nodes

`isambard_sbatch` reads a shared TTL'd log at `/projects/a5k/public/isambard_sbatch_bad_nodes.log` (7-day expiry, configurable via `ISAMBARD_SBATCH_BAD_NODES_TTL`) and auto-passes excluded nodes to SLURM's `--exclude`. Every submission prints `Bad nodes: N excluded (last 7d) — file: ...`; missing line means raw `/usr/bin/sbatch` was used.

```bash
isambard_sbatch --mark-bad <node> "<short diagnosis>"   # append entry
isambard_sbatch --list-bad                              # show active entries
isambard_sbatch --update-bad <node> "<new reason>"      # replace reason + refresh TTL
isambard_sbatch --unmark-bad <node>                     # remove entries for node
isambard_sbatch --prune-bad                             # drop expired/malformed lines
```

**Register only when you can pin the failure to a specific hostname** (Xid in dmesg, `nvidia-smi` ERR! on one host while siblings are healthy, NCCL fails on first collective on a single hostname, HybridEP's `cudaIpcOpenMemHandle` fails with `cudaErrorPeerAccessUnsupported` on every rank of one host — a GPU with dead NVLinks, which `nvidia-smi nvlink --status` shows (18 active links per GPU when healthy) while `nvidia-smi topo -m` still reports NV6 — tunnel never starts on its allocated node, RUNNING with no log output). **Do NOT register** code/config bugs (OOM, bad YAML, wrong TP/EP) or cluster-wide issues (Slingshot congestion, the known ~7-min NCCL hang — `ft_launcher` handles that). Prefer `--update-bad` over a duplicate `--mark-bad`; `--unmark-bad` if a node is fixed before TTL.

Find node names: `scontrol show hostnames $SLURM_JOB_NODELIST`, `sacct -j <id> -o NodeList`, or `squeue` `%N`/`%R`.

#### A single wedged GPU hangs the whole job at distributed init

**Symptom:** the run sits at `> initializing torch distributed ...` forever. Ranks burn **zero**
CPU in `do_sys_poll`, have no NCCL threads, and hold only a ~530 MiB CUDA context. It reads like
a fabric hang, a rendezvous failure, or a scale limit. It is none of those.

**The tell is in the store's own error** — read it before theorising:

```
DistStoreError: Timed out after 1201 seconds waiting for clients. 511/512 clients joined.
srun: error: nid0XXXXX: task 343: Terminated     <- the only task needing a kill
```

`N-1/N` (never `N/N`) means exactly one rank never reached the rendezvous. `torch.cuda.set_device()`
runs **before** `init_process_group` and triggers real CUDA context creation, and it has **no
timeout** — so a rank on a wedged GPU spins there forever, the store never reaches `N`, rank 0
never publishes its gloo address key, and every healthy rank blocks behind it. Cross-check with
`grep -c "NCCL version"`: **0** means NCCL never initialised, so the fabric is not implicated.

**Detection — sweep the whole allocation.** A healthy idle GPU reads 0% utilisation even while a
rank holds a context, so `util > 0` with tiny memory is the discriminator:

```bash
srun --overlap --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c \
  'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
   awk -F, -v h=$(hostname -s) "{gsub(/ /,\"\"); if (\$3>0) print \"SUSPECT\", h, \"gpu\"\$1, \$2\"MiB\", \$3\"%\"}"'
```

A wedged GPU reports e.g. `29MiB util=100%` **with zero compute apps attached**. Confirm by doing
work, not by reading a counter — `nvidia-smi` utilisation can latch stale: create a context and run
a matmul on each GPU in its own short-lived process (healthy ones finish in ~3 s; the wedged one
never returns).

**Why it is so easy to misdiagnose:** small-node runs pass, because rank assignment follows
nodelist order and a short run never reaches the bad node. That makes it look like a scale or
launch-mode problem, and any prior successful run that happened to exclude the node becomes a
false control. Always compare **node membership** (`sacct -j <id> -o NodeList`) before concluding
that config, launcher, or sbatch-vs-tunnel is the variable.

**Remedy:** `nvidia-smi --gpu-reset` needs root (it must toggle persistence mode), so the GPU needs
an admin reset or a node reboot. `--mark-bad` the node so future submissions auto-exclude it. To
keep using a *fixed* allocation that contains one, exclude the node via `--nodelist` and set
`train.decrease_batch_size_if_needed=true` — Megatron then rounds `global_batch_size` down to the
nearest multiple of `micro_batch_size × data_parallel_size`, booking the shortfall as
`skipped_train_samples` so token accounting stays truthful.

(When scripting these sweeps, match processes on numeric UID (`id -u`): `ps -eo user` truncates
account names longer than 8 characters and appends `+`, so a username comparison silently matches
nothing — it finds no processes and reads as "the node is clean".)

### Project storage quota

`isambard_sbatch` prints a **project storage quota report** on every submission — per-path Lustre quota usage (`<path>  used/limit (pct%)`, flagged ` — nearly full` at ≥90%, plus inode counts) via the documented recipe `lfs quota -p $(lfs project -d <DIR> ...) <DIR>`. Example line: `Storage: /projects/a5k  188.6T / 200.0T (94%)  files: 6.2M / 50.0M (12%) — nearly full`. **Determine free storage from this report, not from `df`.** The project quota (`/projects/a5k`, 200 T) is what actually limits writes — and it runs hot (often ~94%). `df -h /lus/lfs1aip2` instead reports the whole shared Lustre filesystem (~21 PB, ~36% used), so it makes storage look nearly empty and completely hides that the project quota is almost full — the opposite of the truth. Tune with `ISAMBARD_SBATCH_STORAGE_PATHS` (default `/projects/<account>`), `ISAMBARD_SBATCH_STORAGE_WARN_PCT` (default 90), or skip with `ISAMBARD_SBATCH_STORAGE_DISABLED=1`. Like the bad-nodes report, it never blocks a submission, so watch it — at ~94% a large checkpoint/download can hit the quota.
