# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Code tooling

This repo uses [`geodesic-claude-tooling`](.claude/geodesic-claude-tooling) (a git submodule) —
Claude Code hooks that inject Geodesic's working conventions at session start, validate plans on
exit, and run lightweight mechanical checks on the diff. The integration is **additive**: it does
not modify the environment build. The tooling lives in a repo-local `.venv` that is
**for tooling only** (ruff, pre-commit, the hooks) and carries no torch — it is unrelated to the
container that runs the pipelines. Install it once:

```bash
bash scripts/install_claude_tooling.sh   # creates/refreshes .venv and installs the tooling
```

Hooks live in `.claude/settings.json`; enabled quality items in `.claude/geodesic-config.yaml`. The
commit-time review gate is **ON** (enabled 2026-07-30 via the setup wizard): `git commit` is
intercepted, pre-commit runs on the staged files, and the commit is blocked until the
`checklist-reviewer` subagent writes a passing `.claude/reviews/verdict.json` for the staged-diff
hash — the full flow is `commit_workflow.md` below. `geodesic-protect-verdict` ensures only that
subagent's Write tool can produce the verdict, and `geodesic-submodule-check` warns if the vendored
tooling checkout drifts off its pin. The gate runs pre-commit on staged files only (not
`--all-files`), so pre-existing repo-wide lint debt does not block unrelated commits. The
conventions themselves are defined in these snippets:

@.claude/snippets/workflows/branch_then_pr.md
@.claude/snippets/workflows/commit_workflow.md
@.claude/snippets/workflows/plan_exit_protocol.md
@.claude/snippets/workflows/convention_changes.md
@.claude/snippets/workflows/pr_notifications.md
@.claude/snippets/workflows/hpc_node_detection.md

### Slack notifications

PR and change notifications for **this repo (`geodesic-megatron`)** go to the **`#megatron`**
Slack channel — set in `.claude/geodesic-config.yaml` → `notifications.pr_notify_channel`, which
the `geodesic-pr-notify` hook reads. `#claude-tooling` is only for the `geodesic-claude-tooling`
submodule's own PRs/discussion, not this repo's changes.

## Repository Overview

NeMo Megatron Bridge is an NVIDIA PyTorch-native library that provides a bridge, conversion, and verification layer between HuggingFace and [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core). It enables bidirectional checkpoint conversion, pretraining, SFT, and LoRA for LLM and VLM models with Megatron Core's parallelism (tensor, pipeline, expert parallelism, FP8/BF16 mixed precision).

The primary package is `megatron.bridge` under `src/`. Megatron-Core is pinned as a git submodule at `3rdparty/Megatron-LM`.

## Cluster Overview (Isambard)

- **GPUs**: NVIDIA GH200 120GB (95GB usable), `sm_90`, 4 GPUs per node
- **CPU**: ARM aarch64 (Grace)
- **Networking**: Slingshot/CXI fabric (HPE)
- **CUDA**: 13.1 in-image on the R580 host driver (580.173.02, CUDA 13.0; forward-compat libs), **Python**: 3.12, **PyTorch**: 2.11.0a0+nv26.02 (from the NGC image — see `## 0. Environment Pipeline`)
- **Compute-node OS image of 2026-10-07**: SLES 15 SP7, host libfabric 2.3.1 only (the pinned default; the Slingshot plugin built against 1.22.0 loads it at full bandwidth). The GPUs' compute mode varies by node and over time: `Exclusive_Process` (one CUDA context per GPU, so two processes cannot share a device) was first seen on 2026-10-09, and the same and other nodes read `Default` later that day. Read it with `nvidia-smi --query-gpu=compute_mode --format=csv`; `scripts/run_unit_tests.sh` does, per run
- **Scale**: cross-node EP=8 MoE all-to-all hits the documented Slingshot/aws-ofi-nccl Send/Recv hang (`docs/investigations/slingshot-nccl-hang-investigation.md`) — keep **TP×EP ≤ 4** (node-local) to avoid it. With node-local EP, scale is NOT capped at 32 nodes: **Ultra SFT is validated at 72 nodes / 288 GPUs** (PP=36). The prior "64+ nodes just hang" belief conflated that Slingshot hang with two Ultra-specific first-iter issues since fixed (`disable_jit_fuser` + a longer `TORCH_NCCL_TIMEOUT`; see the Ultra section).

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

## Pipelines

All top-level scripts follow the `PIPELINE_ACTION.ext` naming convention. There are five pipelines:

| Pipeline | Submit (SLURM) | Launch / Logic | Purpose |
|----------|---------------|----------------|---------|
| **env** | `pipeline_env_submit.sbatch` | `pipeline_env_config.env`, `pipeline_env_setup.sh`, `pipeline_env_exec.sh`, `pipeline_env_activate.sh`, `pipeline_env_validate.py` | **THE execution environment** — Apptainer + NGC NeMo image, Slingshot NCCL stack |
| **training** | `pipeline_training_submit.sbatch` | `pipeline_training_launch.sh` | SFT, CPT, and from-scratch pretraining |
| **data** | `pipeline_data_submit.sbatch` | `pipeline_data_prepare.py` | Dataset download, tokenization, packing |
| **checkpoint** | `pipeline_checkpoint_submit.sbatch` | `pipeline_checkpoint_convert.sh`, `pipeline_checkpoint_convert_hf.py` | Megatron↔HF conversion, Hub upload |
| **coherence** | `pipeline_coherence_submit.sbatch` | `pipeline_coherence_test.py` | Qualitative generation testing, W&B logging |

Each pipeline has a thin `PIPELINE_submit.sbatch` for SLURM allocation and a `.sh`/`.py` with the actual logic. The `.sh` launchers can also be called directly from an interactive `salloc`.

---

## 0. Environment Pipeline (`env_*`) — THE execution environment

Every pipeline runs inside an Apptainer container built from the NGC NeMo image
(aarch64), which supplies torch/CUDA/cuDNN/TE/Mamba-kernels/APEX/`ft_launcher`
prebuilt and version-matched. This repo's `src/` + `3rdparty/Megatron-LM` are
bind-mounted, so the checkout you submit from is the code that runs. There is no
bare-metal path, no venv, and no opt-out flag: a missing SIF or Slingshot build
hard-fails with the fix command rather than degrading.

Full design + troubleshooting: `docs/environment.md`.

### One-time setup

```bash
# ONE command on a GPU node: SIF pull + Slingshot NCCL build + Python overlay +
# validation. Idempotent (done steps skip loudly); --force redoes everything,
# --only <sif|slingshot|overlay|validate> runs a single step.
bash pipeline_env_setup.sh
# or: isambard_sbatch pipeline_env_submit.sbatch setup
```

### Files

| File | Purpose |
|------|---------|
| `pipeline_env_config.env` | THE config: image tag/URI, SIF path, Slingshot build dir, Python overlay + its package list, binds, cache-dir `$HOME` guards, and the `env_config_require` gate. Override via `GEODESIC_CONTAINER_*` env vars documented inline. |
| `pipeline_env_setup.sh` | The whole install in four idempotent steps (`sif` → `slingshot` → `overlay` → `validate`). Needs a GPU node for steps 2 and 4. |
| `pipeline_env_exec.sh` | The shim every launcher uses: scrubs host toolchain env, then runs one command string inside the container. |
| `pipeline_env_activate.sh` | Sourced INSIDE the container: import resolution, CUDA forward-compat, Slingshot `LD_LIBRARY_PATH`/`NCCL_NET_PLUGIN`, universal GPU settings, cache paths. |
| `pipeline_env_validate.py` | 21-check validation (imports incl. grouped_gemm, which the non-default `cublas_grouped` expert backend needs, CUDA, GPU ops, import resolution, NCCL plugin dlopen, host OpenMP threading defaults, HF datasets-cache writability, ft_launcher flags, dataset-helpers JIT, recipes, version report); `--run-training` adds a tiny training run. |
| `pipeline_env_submit.sbatch` | SLURM wrapper; modes `setup`, `validate`, `smoke` (2-node fabric check). |

### Key facts

- **Config-driven:** everything (image tag, SIF path, binds, cache dirs, Slingshot
  component versions, overlay packages) lives in `pipeline_env_config.env`.
- **SIF + Slingshot build live on** `/projects/a5k/public/containers/` — NEVER `$HOME`
  (the config refuses `$HOME` cache dirs; a SIF would blow the home quota instantly).
- **Slingshot networking** follows Isambard's official "Option B": NCCL + hwloc +
  aws-ofi-nccl built inside the image against the image CUDA + host libfabric
  (one-time per image tag). Never use `brics/apptainer-multi-node`/`adapt.sh` with
  these images — it injects host NCCL 2.26 over the image's torch-matched NCCL.
  Without the CXI plugin NCCL silently falls back to TCP: ~2.3 GB/s vs ~163 GB/s.
- **Image contents are not frozen in this file** (they rot): the validator's
  version-report check prints the live set. Qualified image today is
  `nvcr.io/nvidia/nemo:26.04` (re-qualified 2026-07-29) — Python 3.12, CUDA 13.1,
  NCCL 2.29.2, torch 2.11.0a0+nv26.02, TE 2.14.1, mamba-ssm 2.3.1, causal-conv1d
  1.6.1, transformers 5.3.0, APEX, nvidia-resiliency-ext 0.6.0. (26.06's CUDA 13.2
  forward-compat libs rejected the cluster's earlier R565 driver but run on the R580
  driver, 580.173.02: `validate` 20/21 on 2026-10-09, job 7179097, with CUDA and the GPU
  ops passing and only `grouped_gemm` failing, because the 26.06 overlay lacks
  nv-grouped-gemm. 26.06 is not qualified, and no training run has used it on R580; the
  Nano pretrain benchmarks of 2026-09-29 ran it on R565 through the 26.04 image's CUDA 13.1
  compat libs, campaign log E-056.)
  The Python overlay (`pip install --target`, `--no-deps`, on PYTHONPATH after the repo
  and before the image) fills gaps without touching the read-only SIF: `peft` (image
  0.13.2 is below modelopt's >=0.17 requirement), `imageio` (absent; one diffusion
  test file otherwise fails at collection), and `nv-grouped-gemm` (absent from 26.04;
  needed by the `moe_experts_impl: cublas_grouped` backend — no longer the shipped
  default, but kept installable so the A/B against `torch_grouped` stays runnable — built
  from sdist with `--no-build-isolation`, and the validator's grouped_gemm check gates on it).
- **Import resolution** is `repo src/` > `3rdparty/Megatron-LM` > overlay > image
  site-packages, via PEP 420 namespace portions. The validator asserts it every run —
  a regular (non-namespace) `megatron` package in a future image would silently win.
- **Benchmark/certification config:** `configs/quickstart/nemotron_super_quickstart_sft.yaml`
  (Super-120B, TP1·CP4·EP4·PP8·ETP1·DP2 → 64 GPUs = **16 nodes**, **GBS 128** since
  2026-08-05) — gate is < 40 s/iter (mean of
  iters 10–30; measured anchor **31.562 s/iter** =
  **167.4 TFLOP/s/GPU** model-FLOPs, `moe_experts_impl: torch_grouped`
  and optimizer CPU offload **OFF**, both shipped defaults). Placement moves this workload
  by ~2%, so quote the placement when quoting the number. Superseded anchors, all at the
  pre-2026-08-05 **GBS 64** workload: **17.099 s/iter** (154.5 TFLOP/s/GPU, 83.1 GB peak;
  the paired same-nodelist A/B that certified `torch_grouped`, −16.2% vs 20.397), 20.66
  for the `cublas_grouped` per-expert loop, 21.78 offload-0.5 on 26.02, 25.66 at the
  2026-07-29 26.04 qualification pre-grouped-GEMM.
  Qualifying a new image tag = that absolute gate plus no regression against the
  previously qualified tag's recorded number at the same GBS-128 workload.
- **Scaling out to 128 GPUs is an OVERRIDE, not a second config.** The quickstarts are
  standardised at 64 GPUs / 16 nodes; the 128-GPU run differs in exactly one field, and the
  launcher forwards Hydra overrides, so it is:
  `isambard_sbatch --nodes=32 pipeline_training_submit.sbatch \
   configs/quickstart/nemotron_super_quickstart_sft.yaml super sft train.global_batch_size=256`
  Measured **122.0 ms/sample** (31.228 s/iter, 169.2 TFLOP/s/GPU, allocation 5845741).
  With the base config now at GBS 128, this override is **matched µb/replica** (64 at both
  sizes): perfect per-sample halving predicts 246.58/2 = 123.3 ms/sample and the 128-GPU
  measurement is 122.0 — **scaling is perfect within the ±2% cross-allocation placement
  band**, same backend at both ends (the first legitimate scaling number since the
  `cublas_grouped`-era 98.8% was retracted).
  Scale the batch with the nodes: at fixed GBS, doubling GPUs halves µb/replica and grows
  the PP bubble. The 2026-08-03 seven-probe ladder + 24-topology adversarial sweep closed
  the alternatives: cross-node EP loses superlinearly at any PP, CP2 OOMs at every legal
  point, PP>8 is constructible but strictly slower (PP8·DP4 is the only layer-balanced
  depth at 128 GPUs). Evidence:
  `/projects/a5k/public/logs/infr71_wave2/docs/consultant-training-stack-review.md` §C13.
- **Unit tests run inside the container** (the image ships pytest/pytest-xdist/ruff/pre-commit),
  through the runner the pre-commit hook uses (see Testing below):
  ```bash
  ./pipeline_env_exec.sh "bash $PWD/scripts/run_unit_tests.sh"
  ```
  The `.venv` that remains is for **dev tooling only** (ruff, pre-commit, the Claude
  Code hooks) and deliberately carries no torch; create it with
  `bash scripts/install_claude_tooling.sh` (it uses `uv pip install`, never `uv sync` — a sync
  would resolve the full project and try to build torch/TE/mamba on the host).

> History: the bare-metal venv stack (a 435-line installer plus 12 order-dependent ARM
> workarounds) was deleted with the container-only simplification. That knowledge —
> pinned versions and every workaround — is preserved in the "Retired from
> geodesic-megatron" Slack canvas in #megatron, not in this repo.

---

## 2. Training Pipeline (`training_*`)

### Files

| File | Purpose |
|------|---------|
| `pipeline_training_launch.sh` | Shared launcher: NCCL/CXI env vars, fault tolerance, srun + ft_launcher |
| `pipeline_training_submit.sbatch` | Thin SLURM wrapper: allocates nodes, calls `pipeline_training_launch.sh` |

Training script (called by the launcher):
- `pipeline_training_run.py` — Unified entry point for SFT, CPT, and from-scratch pretraining (dispatches via `--model nano|super|ultra --mode sft|cpt|pretrain`; `pretrain` uses the NVIDIA pretrain recipes + the `pretrain()` entry point, requires `dataset.data_path`, and loads no checkpoint unless the YAML sets one)

### Usage

```bash
# Via SLURM (allocates nodes) — extra args after the mode forward to the launcher:
# launcher flags (e.g. --disable-ft) parse as such, anything else falls through as
# Hydra overrides (benchmark runs pair --disable-ft with checkpoint.save=null)
isambard_sbatch --nodes=32 pipeline_training_submit.sbatch configs/<config>.yaml nano sft
isambard_sbatch --nodes=8  pipeline_training_submit.sbatch configs/<config>.yaml nano cpt
isambard_sbatch --nodes=16 pipeline_training_submit.sbatch configs/<config>.yaml super sft \
    --disable-ft train.train_iters=32 checkpoint.save=null

# Via salloc (interactive)
salloc --nodes=16 --gpus-per-node=4 --time=24:00:00 --exclusive
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft
bash pipeline_training_launch.sh configs/<config>.yaml --model super --mode cpt
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --nodes 8 --nodelist node[001-008]
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --disable-ft
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --peft lora
```

`pipeline_training_launch.sh` options: `--model nano|super|ultra` (required), `--mode sft|cpt|pretrain` (required), `--disable-ft`, `--disable-straggler`, `--enable-pao`, `--peft lora`, `--nodes N`, `--nodelist LIST`.

`cpt` and `pretrain` both read Megatron-native `.bin/.idx` data and both **require**
`dataset.data_path` in the config — there is no default corpus. They differ only in the
recipe: `cpt` uses the SFT recipe (warm-start LR 5e-6, which the CPT configs override down
to ~1e-6), `pretrain` uses `nemotron_3_*_pretrain_config` (from-scratch LR — 1.6e-3 Nano,
4.5e-4 Super and Ultra) and dispatches `pretrain()` instead of `finetune()`, whose assert
would demand a checkpoint.

**Config composition (`base_config:`).** A training YAML may name another under the top-level
`base_config:` key and then state only the fields it changes. The base path resolves against the
naming file's directory (never the working directory), a base may itself name a base, and a
cycle or a missing base raises. Mappings deep-merge; a list, a scalar or an explicit `null`
replaces the base value, as `OmegaConf.merge` would, and scalars read as `OmegaConf.load` reads
them (`5e-4` is a float). The composed mapping is then merged onto the recipe and the Hydra CLI
overrides apply last. Composition happens only where a config is read through
`scripts/training/config_compose.py` (`load_composed_yaml`): `pipeline_training_run.py` (and
`scripts/data/report_blend_coverage.py`, which resolves a config through its
`resolve_training_config`), `scripts/nemotronh_flops_estimator.py` (and
`scripts/telemetry/score_run.py`, which reads its config through the estimator) and the config-test
helpers (`tests/unit_tests/campaign_config.py`, `test_control_pretraining_config.py`).
`configs/control_pretraining/stage_gate.sbatch`,
`scripts/hub/sync_bucket.py`, `scripts/hub/publish_models.py` and
`configs/control_pretraining/generate_epoch_chain.py` (which copies each chain's `posture_config`
into its links and reads its `parent_config`) read their configs as raw YAML, so a config they are pointed at must stay a
complete file. The Nano pretrain quickstart is the
first overlay.

**Performance probes (one short job each).** Measure a training lever as its own
`pipeline_training_submit.sbatch` job on the quickstart posture (Hydra overrides and env knobs
on the launch line, `--time` calibrated to the measured runtime — a 50-iteration 64-GPU Nano
probe takes 9–13 min, so `--time=00:20:00`), not inside a held allocation: short jobs backfill
quickly and hold nothing idle. Launch code under test from a read-only copy of a commit (e.g.
`git archive` plus the pinned `3rdparty/Megatron-LM`), not from a working checkout that may be
edited while the job runs — bash reads the launcher by byte offset. An archive carries neither
`.git` nor ignored build products, so the copy also needs (1) a `REVISION` file at its root
naming the commit (plus any uncommitted diff it carries): with no `.git`, the profiler's
provenance reads the commit from it (`scripts/telemetry/code_revision.py` `code_revision`)
and otherwise records the commit as unresolved; and (2) the Megatron dataset helpers library,
`3rdparty/Megatron-LM/megatron/core/datasets/helpers_cpp*.so`, copied from a built checkout —
Megatron-LM git-ignores `*.so`, and without it rank 0 runs `make` in that directory at startup,
which a read-only copy cannot do. Compare a probe only against runs placed on a single Dragonfly
switch group (every launcher run logs `[run-identity] switch placement`): a run whose nodes span more
than one group is 2.5–6.6% slower (mean 4.3%), while single-group runs of one posture agree to ~0.5%
(SD). That placement penalty is what made the same 64-GPU Nano posture measure up to ~4% apart across
jobs (6.744 and 6.494 s/iter; 3.7% between two runs sharing 15 of 16 nodes): the slower run of each
pair spanned more than one group. A lever smaller than ~1% needs repeats before it counts. The Nano
pretrain campaign's log is `docs/investigations/nano30b-pretrain-perf-campaign.md` (E-047 for the
placement measurements).

### Profiling and run identity

- **Torch-profiler capture** (any launch): prefix with
  `ISAMBARD_TORCH_PROFILE=1 ISAMBARD_TORCH_PROFILE_ITERS=10,20` (also
  `_RANKS=0,9`; legacy `_WAIT=3` = single capture at iteration 5). Profiling runs
  against the STANDING quickstart config with overrides — there is no separate
  profile config to drift out of sync; the exact command (including the
  `logger.wandb_save_dir` that is mandatory alongside `checkpoint.save=null`) is in
  that config's header and in `docs/profiling-quickstart.md`. Artifacts (per-rank traces, provenance, config +
  resolved-config snapshots, raw-log copy) land in
  `/projects/a5k/public/profiles/<wandb-exp-name>/<run-id>/`.
  `ISAMBARD_TORCH_PROFILE_WITH_STACK` (default `1`) records Python stacks, which
  `scripts/profiling/trace_analysis/host_attrib.py` needs to attribute host time to code;
  stack walking inflates host launch time (~+18% wall on the 120B capture), so take a
  timing breakdown with `=0` and keep a stack capture for attribution (any other value
  fails at startup).
  Tutorial: `docs/profiling-quickstart.md`; reference: `docs/environment.md`
  "Profiling a training run"; implementation:
  `scripts/profiling/profiler_callback.py`.
- **Run identity**: every launcher run mints `ISAMBARD_RUN_ID`
  (`<timestamp>-j<jobid>`), echoed in the log banner, symlinked as
  `/projects/a5k/public/logs/megatron_runs/by-run-id/<run-id>.out`, used as the profile subdir name, and
  stamped into W&B summary (`run/isambard_run_id`, `run/raw_log_path`,
  `run/slurm_job_id`) by `scripts/telemetry/run_identity.py` — the join key
  between a W&B run, its raw log, and its profiles. The same summary carries
  `run/switch_count` and `run/switch_spread`, the run's Dragonfly placement,
  from the launcher's `scontrol`-derived `ISAMBARD_SWITCH_SPREAD` (`scontrol`
  does not exist inside the container, so the payload cannot compute it). Both
  are absent on runs not started through `pipeline_training_launch.sh`.
  Placement is worth ~18% on the 512-GPU Nano stage-1 pretrain posture — 137.8
  TFLOP/s/GPU across 2 switch groups against 114.7 across 8 — so compare a throughput
  number only against another taken at the same spread.
- **Scoring a run**: `scripts/telemetry/score_run.py <log> --config <training yaml>
  --hf-model <hub id or path> --gpus N --window FIRST LAST --loss-window FIRST LAST
  [--peak-tflops TF] [--wandb-peak-memory] [--json]`; for the Nano pretrain quickstart:
  `--config configs/quickstart/nemotron_nano_quickstart_pretrain.yaml --hf-model
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --gpus 64 --window 26 50 --loss-window 41 50`.
  The sequence length and model FLOPs/token are read from the config (through its
  `base_config:` chain) and the HF `config.json` by `scripts/nemotronh_flops_estimator.py`'s
  library, and the peak defaults to the estimator's `DEFAULT_PEAK_TFLOPS` (989.4, GH200 dense
  BF16). The primary statistic is the **mean** step time over the window (node-hours scale
  with the mean); median, p10/p90, min/max and the iterations above 2× the median are reported
  beside it, then tokens/s/GPU, model TFLOP/s/GPU and MFU, mean loss, NaN/skipped counts,
  iteration-1 memory and the W&B run path, and every score records its inputs (log, config, HF
  config, GPUs, sequence length, logged GBS, FLOPs/token, peak, windows). Tokens per iteration
  are the global batch the window's own log lines report times the config's sequence length, so
  a batch set by a Hydra override is scored correctly; a window iteration that is missing or
  repeated, or two batch sizes in one window, raises. `--wandb-peak-memory` adds the W&B
  summary peaks and allocator-retry count, which are the last rank's, not a maximum over ranks.
  The maximum over ranks is `peak_memory_across_ranks`, read from the `[peak-memory]` line rank 0
  logs when the training loop ends (`train_utils.py` `gather_peak_memory`, one gather after the last
  step, saves included): the largest peak allocated and reserved memory, the rank holding the
  allocated one, and the largest and total allocator retries, also written to the W&B summary under
  `memory/across-ranks/`. A log that predates the summary has none, and so does a run whose loop was
  cut short (an OOM, a wedge, a SLURM wall-time kill); a loop ended by `exit_duration_in_mins` prints it.
- **Loss parity between runs**: `scripts/telemetry/loss_parity.py` compares runs of one config,
  seed and data-parallel width (iteration i consumed the same global batch in each). `band
  --reference A1 A2 [...] --candidate C --iterations 1 500 --window 50` is the test for a lever
  that changes numerics: per window, the candidate's mean `lm loss` must lie within [lowest
  reference - delta, highest reference + delta], delta being the largest window difference
  between two references (PASS/FAIL; `grad norm` the same way, PASS/FLAG). `--loss-half-width W`
  replaces delta for `lm loss` with a stated tolerance, and with it one reference suffices: a replay
  of one earlier run of the same config, such as a check of an upgraded stack, has no run-to-run
  spread to draw on (its `grad norm` band then has zero width). `identity
  --reference A --candidate B --iterations 1 30` is the test for a lever or knob-off path that
  claims exactness: per metric, the leading identical iterations and the first difference. Both
  also require every run to log the same learning rate and consumed samples at every iteration,
  and `--wandb` reads full-precision values from each log's W&B run (the log prints 7
  significant digits of the loss and 3 decimals of the grad norm). Exit status 1 on any FAIL.
  Default-mode training is not run-to-run deterministic (the grad norm differs from iteration 1),
  so an identity test needs `model.deterministic_mode=true` with `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `MAMBA_DETERMINISTIC=1` (an `ISAMBARD_ENV_OVERRIDES` file) on
  every run it compares. mamba_ssm fixes its Triton autotune configurations at import, before Bridge
  turns on deterministic algorithms, so without the last one the SSD kernels are autotuned by timing
  and their block sizes, and with them the reduction order, can differ between launches.
  Megatron-Bridge refuses cross-entropy fusion in deterministic mode, so such a test runs without it.
- **Pre-registered loss gates**: `scripts/telemetry/loss_gate.py --spec <gate.yaml> --candidate <log>
  [--gate NAME]` runs band tests frozen in a YAML before the candidate exists: each named gate lists
  two or more references (each log named once), its range and window, the references' lm-loss spread
  when frozen (`lm_loss_delta`, which identifies the reference set), and optionally a fixed tolerance
  (`lm_loss_tolerance`) that replaces the spread as the band's half-width. Exit 1 when any gate FAILs,
  whatever the others; otherwise 2 when any is NOT EVALUATED (the log does not yet cover the range, the
  spread does not reproduce, a log or W&B read fails, or the candidate is one of the references, which
  a band cannot judge), and 0 when all PASS. The v2e2e arm's `loss_gate.yaml` is the first spec.
  `scripts/telemetry/score_gate.py --spec <gate.yaml> --scores-dir DIR` does the same for thresholds
  on the `score_run.py --json` files and `loss_parity.py band --json` reports in DIR: a `memory` gate fails a score with no `peak_memory_across_ranks`,
  more allocator retries or more peak allocated memory than its limits; a `speed` gate projects a
  candidate's mean step time relative to a reference's on the same nodes onto a stated step time and
  fails above its limit; a `first_loss` gate compares two runs started from the same weights on the same
  first batch at their first logged iteration and fails when the candidate's lm loss there differs from the
  reference's by more than its tolerance; a `loss_shift` gate bounds a `loss_parity.py band --json` report's
  candidate offset from the references' mean in every window and its rise from the first windows to the last.
  For a token-masking comparison it also reads training logs and `pipeline_coherence_test.py --probe-spec` result
  JSONs from DIR: `masking_log` and `log_pairing` gates check, iteration by iteration, the exact
  `[token-masking-counts]` integers (and logged metrics) of a masked and a control run; `value_change`,
  `slot_logprob_difference` and `emission_count` gates bound probe values, per-prompt log-probability
  differences and generated marker counts; `probe_identity` and `probe_agreement` gates check which checkpoint,
  config, tokenizer and code each probe measured. A non-finite value never passes: it is NOT EVALUATED. A spec may
  order its gates into a `verdict` of stages, the first stage that does not fully pass deciding it (PASS 0, FAIL 1,
  INCONCLUSIVE 2); `tests/e2e_tests/inoculation_midtraining_token_masking/gate.yaml` is the first such spec.
  Both tools share the outcomes and the exit status (`gate_outcome.py`); the v2e2e probes run the arm's
  `score_gate.yaml` and `score_gate_midtrain.yaml`, because `score_run.py` exits 0 on any scorable log.
  `scripts/telemetry/run_watch.py --spec <watch.yaml> --log <segment log> ...` checks a running stage the same way:
  exit 1 on a stop condition (a result the rerun state machine rejected, which is how the gradient NaN check ends a
  run; a non-finite grad norm or lm loss, or an iteration line without `lm loss`; an iteration counted as nan or
  skipped; allocator retries; a segment whose `[env-overrides]` lines are not exactly the stage's
  `ISAMBARD_ENV_OVERRIDES` file, read with the launcher's own parser) or a failing due loss gate, 2 on a due gate or
  a stop check it could not evaluate (each stop check runs on its own, so the others still stand) or a failure of
  the watch itself (never 1); flags (a loss spike, a loss gate's offset growing past a bound, a block of iterations
  further from a reference run than an envelope run is) are printed, never a stop. A segment that resumes from a
  save supersedes every earlier segment's records, saves and rejected results from its first iteration on, and their
  peak-memory summaries once it has logged an iteration; an iteration or peak-memory line that cannot be parsed
  leaves the stops NOT EVALUATED unless a later segment has re-run its iteration. A gate
  named `--decided GATE=LOG` passed at an earlier check on that log and is not evaluated again while that log covers
  its range; the watch ends by naming the gates still undecided. The v2e2e arm's `watch_pretrain.yaml` and
  `watch_midtrain.yaml` are its specs.
  `scripts/training/stage_guard.py --config <guard.yaml>` runs that watch on a timer while the stage trains, on the
  tunnel under the host Python, handing each passed gate to later ticks as decided (a guard started again reads them
  from its record, from ticks judged under the same watch spec): on a stop it cancels the stage's
  live jobs by ID (the running segment and the successors pending on its singleton dependency, which only `squeue`
  lists) and exits; on a tick it could not evaluate (exit 2, or a watch, timeout or `sacct` failure) once the stage
  has reached its hold iteration while a loss gate is undecided it does the same; otherwise it alerts. Its record
  opens with the config, the spec's sha256 and the code revision, and its own failures (no started segment of the
  stage among them) are written there too.
  The v2e2e arm's `guard_pretrain.yaml` and `guard_midtrain.yaml` are its configs.
- **Reproducing an overridden posture**: the override YAML alone omits recipe defaults,
  CLI overrides and, for a `base_config:` overlay, every field it inherits (the
  profiler's `config_snapshot.yaml` is that overlay verbatim), but the bridge sends the
  FULL resolved config to W&B at startup — recover any run's exact posture from its W&B
  run's config tab (join via `run/isambard_run_id`) or from the profile's
  `resolved_config_snapshot.yaml`.

### Environment Variable Architecture

`pipeline_training_launch.sh` adds distributed-training-only vars on top of `pipeline_env_activate.sh`:
- All Slingshot/CXI NCCL vars (`NCCL_NET`, `FI_PROVIDER`, `FI_CXI_*`, etc. — 30+ vars)
- Fault tolerance vars (`TORCH_NCCL_TIMEOUT`, `TORCH_NCCL_RETHROW_CUDA_ERRORS`)
- Job-specific node-local paths (`TRITON_CACHE_DIR`, `TMPDIR`, `MEGATRON_CONFIG_LOCK_DIR`; activate
  puts HybridEP's JIT cache, `HYBRID_EP_CACHE_DIR`, under that `TMPDIR` instead of `$HOME/.deepep`)
- Module loading (`PrgEnv-cray`, `cuda/12.6`, `brics/aws-ofi-nccl/1.8.1`)

Every env var has detailed inline documentation.

**Per-launch overrides: `ISAMBARD_ENV_OVERRIDES=<file>`** changes any variable for one launch
without editing the launcher or activate, including those they set unconditionally — e.g.
`TORCH_NCCL_BLOCKING_WAIT=0` to get the NCCL watchdog and flight recorder back (see Fault
Tolerance). The file holds `KEY=VALUE` lines (`#` comments; values literal, no quote removal
or expansion). The launcher applies them three times: at its top, so its own knobs
(`ISAMBARD_NCCL_DEBUG`, `TRAIN_*`, `GEODESIC_CONTAINER_*`, `ISAMBARD_RUN_ID`, ...) take them;
again after its last export; and inside the container after `pipeline_env_activate.sh`,
immediately before ft_launcher/torchrun. It ends the launch before srun on a set-but-missing
file, a malformed or repeated line, one of its own shell variables (`MODEL`, `NNODES`,
`REPO_DIR`, `GEODESIC_REPO_DIR`, ... — they would change the launch, not its environment),
`CONTAINER_*` (override the `GEODESIC_CONTAINER_*` input instead) or a credential-looking key
(a `TOKEN`/`SECRET`/`PASSWORD`/`API_KEY` segment — every value is printed into the job log).
The banner lists each applied `KEY=VALUE`, and the ranks receive
**`ISAMBARD_ENV_OVERRIDE_KEYS`** (the keys, comma-separated; the launcher drops an inherited
one when no file is named). From it `pipeline_training_run.py` logs, once per node, the value
each key actually has in training —
`INFO:__main__:[env-overrides] rank=<R> host=<host> KEY=<value> ...` (values shell-quoted;
the logger prefix means a log search must not anchor at the line start) — and raises if a
listed key is absent. An override is a plain assignment: to change something activate derives,
override its input (`ISAMBARD_OMP_THREADS=1`, not `OMP_NUM_THREADS=1`). Full contract:
`docs/environment.md` D2b.

`pipeline_env_activate.sh` (sourced inside the container, in the same shell that then execs
ft_launcher/torchrun) carries the universal knobs. Three are tunable:
`ISAMBARD_CUDA_MAX_CONNECTIONS` (default 1), `ISAMBARD_CUDA_ALLOC_CONF` (default
`expandable_segments:True`; a measured `False` arm showed no difference on the 512-GPU
Nano pretrain posture — the knob exists for A/Bs), and
**`ISAMBARD_OMP_THREADS` (default 8)**,
which sets `OMP_NUM_THREADS` and, whenever it is > 1, also `OMP_WAIT_POLICY=PASSIVE`. The
default matters because torchrun silently sets `OMP_NUM_THREADS=1` when the variable is
absent, single-threading the host-side AdamW of any CPU-offloaded optimizer onto one
Neoverse-V2 core: **21.36 s/iter / 73.70 GB at offload 1.0 with 8 threads, versus 22.79 /
76.78 at offload 0.5 single-threaded** — i.e. it strictly dominates the previous champion,
but the two arms differ in offload fraction as well as threads, so the delta is not
attributable to threading alone (the clean offload-1.0 single-thread arm was never run on
that nodelist; see the consultant tracker §C1b, preserved at
`/projects/a5k/public/logs/infr71_wave2/docs/consultant-training-stack-review.md`). Both arms are on the pre-`torch_grouped` expert path — this
A/B is about host-side Adam, so the expert backend does not move it. Threading is exactly
neutral (20.663 vs 20.654, identical peak)
when offload is off — which is what makes 8 safe as a universal default. `PASSIVE` is
load-bearing: GNU OpenMP idle threads spin-wait and these workloads are host-launch-bound.
Set `ISAMBARD_OMP_THREADS=1` to restore torchrun's behaviour. `pipeline_env_validate.py`
scores both as a check, so a silent regression fails loudly instead. **What it would cost
depends entirely on the posture**: on the shipped offload-off quickstart, essentially
nothing (20.663 vs 20.654); it only bites where a CPU-offloaded optimizer gives host AdamW
real work to do, and even there the 1.43 s/iter figure above is not a clean threading
delta — see §C1b in the tracker above before quoting it.

### Training-Specific Override for Isambard

The NGC image ships APEX, so `model.gradient_accumulation_fusion: True` works and is
the faster path — a measured ~1.1 s/iter win on the 120B quickstart (2026-07-24). It is
set `True` in the shipped quickstart. (This used to require a per-environment override;
with the venv gone, there is one answer.)

### Fault Tolerance

Slingshot/CXI causes intermittent NCCL collective hangs (~every 2-3 hours with EP=8 cross-node). The training pipeline uses a layered resilience stack:

1. **ft_launcher worker restart** (`--max-restarts=20`) — the per-node agent restarts failed workers, reloading from the latest checkpoint. ≤25 iters lost. (`cfg.inprocess_restart` is never set by `pipeline_training_run.py`, so there is no in-process layer on the shipped path.)
2. **Collective timeout** — the calling thread's blocking `wait()` throws after the YAML's
   `dist.distributed_timeout_minutes` (60 on the campaign configs; the launcher's
   `TORCH_NCCL_TIMEOUT`, 7200 s, covers only groups created without one) and the rank exits.
   This is not a watchdog: under the launcher's `TORCH_NCCL_BLOCKING_WAIT=1` torch creates no
   watchdog thread at all (next paragraph).
3. **srun `--kill-on-bad-exit=1`** — when a rank dies unrecoverably (or ft is disabled), the whole step ends instead of stranding the surviving ranks in a never-completing collective; with a `--dependency=singleton` chain the next segment then resumes from the latest checkpoint.

**A run that simply stops iterating leaves no evidence by itself — take it before cancelling.**
The launcher's `TORCH_NCCL_BLOCKING_WAIT=1` means torch creates **no watchdog thread** (its own
log line at process-group creation says so): no asynchronous timeout detection, no heartbeat
monitor, and no NCCL flight-recorder dump, on timeout or on demand — the recorder buffer
(`TORCH_NCCL_TRACE_BUFFER_SIZE`) is filled and never written, whatever the comment beside
`TORCH_NCCL_DUMP_ON_TIMEOUT` used to promise. The only timeout is the calling thread's `wait()` at
the YAML's `dist.distributed_timeout_minutes` (Megatron passes it to `init_process_group`; the
launcher's `TORCH_NCCL_TIMEOUT` covers only groups created without one), which throws after 60
minutes on the campaign configs. So a hang cancelled at the 30-minute mark takes everything with
it. `scripts/training/dump_hung_ranks.sh <jobid>` takes the evidence from outside first, into
`<log-dir>/nccl_trace/<jobid>/`: `processes.<host>`, the state and kernel wait channel of every
rank and of every helper it forked (dataloader workers and the multiprocessing bookkeeping
processes carry the rank's environment; a helper is one whose parent carries the same RANK),
recorded before anything is attached to; then every process's Python and native stacks via py-spy
(no cooperation from the rank needed; ptrace is unrestricted on the compute nodes; py-spy must be
on the host PATH of the compute nodes — once per user, `python3 -m pip install --user py-spy`,
which the shared home makes visible on every node — or named in `PY_SPY`), the rank's own to
`rank_<rank>.stack` and each helper's to `rank_<rank>.child-<pid>.stack`, skipping a process in an
uninterruptible wait, which cannot be attached to. The stacks show which collective each rank is
waiting in and what the ranks that never arrived are doing instead. Where a watchdog exists
(blocking wait off; for one launch, `TORCH_NCCL_BLOCKING_WAIT=0` in an `ISAMBARD_ENV_OVERRIDES`
file) the same run also triggers the recorder dump into that directory. Two 64-node
segments of the filtered stage-1 run wedged on 2026-09-11 with no NCCL warning, watchdog or
traceback in the log, and were cancelled before anything was captured. **The stalls that were
captured, on 2026-09-12, were not NCCL at all:** one rank's main thread was waiting on its
dataloader, that rank's worker sat in `cl_sync_io_wait` with a single Lustre read RPC to one OST
(OST0014, over kfi) outstanding and its read counters flat, its three node siblings spun in the
node-local expert all-to-all, and the other 252 ranks waited at their collective. The first such
stall cleared on its own after 6.3 min (an RPC timeout and resend, most likely); the run is not
wedged during one, so read `processes.<host>` for a `D`/`I` state with a Lustre wait channel before
cancelling anything, and give a stall the Lustre resend time before treating it as a wedge.

**ft_launcher timeout configuration** (set in `pipeline_training_launch.sh`):
- `--ft-rank-section-timeouts=setup:10800,step:7200,checkpointing:3600`
- `--ft-rank-out-of-section-timeout=7200` — must cover first-iter NCCL lazy init at PP=8+
- `--ft-initial-rank-heartbeat-timeout=7200 --ft-rank-heartbeat-timeout=7200` — heartbeats are
  an INDEPENDENT mechanism from the section timeouts. Omitting them is not "disabled": NVRX
  defaults to 3600 s / 2700 s, which is shorter than Ultra-550B's 45-75 min first iteration at
  PP=36 and produces a SIGKILL + restart loop that looks exactly like a fabric hang. The image's
  ft_launcher parses these as floats and rejects the literal `none`, hence explicit numbers.
  **Raising them to 7200 does not make them safe, it moves the wall.** The rank monitor is not
  guaranteed to receive an initial heartbeat at all; when it does not, the timeout stops being a
  liveness check and SIGKILLs a perfectly healthy job at exactly 7200 s. Signature, worth
  grepping before blaming the fabric: `[Cycle N] Did not get initial heartbeat. Waited 7200.00
  seconds`. Observed on Super-120B/64 GPUs and Nano-30B/512 GPUs; **not** universal — Ultra-550B
  trains ~4.7 h under the same default without a kill, so what decides delivery is unresolved.
- `calc_ft_timeouts=True` auto-learns step timeouts after first successful run. **Delete `ft_state.json`** from checkpoint dir if learned timeouts are too aggressive after config changes.

The `ft`/`nvrx_straggler`/`inprocess_restart` Python configs **cannot** be set via YAML or Hydra overrides (OmegaConf merge creates dicts, not dataclasses). They are set in `pipeline_training_run.py` via the `--enable-ft` flag (on by default). Use `--disable-ft` to opt out.

`--disable-straggler` drops **only** the NVRx straggler detector and keeps `cfg.ft`: the
detector's rank-0 gather of per-GPU perf scores has been observed to OOM high-memory jobs after
~20 minutes of stepping, which is why the pretraining quickstarts (short, restart-free) simply
pass `--disable-ft` instead.

**It is not a milder `--disable-ft`, and it is the wrong reach for a long run.** It leaves the
heartbeat timeout above fully armed, so it does nothing about the 2 h wall. Before enabling ft
on any run expected to exceed the heartbeat, do the arithmetic the run now prints at startup:
compare `save_interval × s/iter` against 7200 s. If the **first** checkpoint lands after the
wall, ft is not fault tolerance — a heartbeat SIGKILL restarts from iteration 0 with an empty
checkpoint directory, and the run makes no net progress for as long as it is left alone.
`pipeline_training_run.py` logs the required s/iter whenever ft is on
(`max_seconds_per_iteration_under_ft_heartbeat`); read that line rather than rediscovering it.
Measured instance: the 500B control-pretraining baseline needed < 2.42 s/iter and ran at 4.276,
giving seven kills across eight attempts in 15.5 h with zero checkpoints written.

Where ft is dropped, a `--dependency=singleton` chain with `checkpoint.load == checkpoint.save`
supplies the recovery ft would have: the collective timeout ends a genuinely wedged segment, and the
next segment resumes from the latest checkpoint.

### Nemotron 3 Nano (30B-A3B) on Isambard

**The Nano SFT quickstart is the control-pretraining XL SFT** (Kyle, 2026-10-01; it replaced the August 32K
GBS-128 config): a baseline benchmark and its fastest configuration, as for the pretraining and midtraining stages.
- `configs/quickstart/nemotron_nano_quickstart_sft_baseline.yaml`: a `base_config:` overlay of
  `configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml`
  (packed seq 32768, TP1·CP2·EP4·PP1, full recompute) at 64 GPUs and **GBS 64** — 2 packs per replica,
  the XL SFT's per-GPU work at GBS 256 on 256 GPUs — warm-started from the midtraining final, with
  production's gradient bucket restated, run for 100 iterations and scored over iterations 51–100:
  **6.563 s/iter** (eight single-group runs) = 4,993 tokens/s/GPU.
- `configs/quickstart/nemotron_nano_quickstart_sft.yaml` (+ `.env`): that benchmark plus the campaign's levers —
  CP=1 with the chunked linear cross-entropy, BF16 gradient reduction in 500M-parameter buckets with the parameter
  all-gather overlapped, HybridEP with the fused router on packs padded to the full 32,768 tokens, and host settings —
  **3.689 s/iter = 8,881 tokens/s/GPU, 1.779×** the benchmark on the same allocations (four paired
  single-group cycles, 95% CI [1.775, 1.782]: goal established), at a 70.4 GiB peak; this file launched as documented:
  3.691 s/iter (job 7006758, one switch group). Its 500-iteration loss fails the
  pre-registered band: it sits below the as-is runs' band over iterations 201–400, by at most 4.3×10⁻⁵ nats, and is
  back inside from 401 — a transient excursion Kyle accepted (2026-10-02).
- **CP=1 fits at 32K only with the chunked linear cross-entropy** (`cross_entropy_fusion_impl: linear`, the pin's
  carried commit 0005). The unfused path keeps the fp32 logits, seq × vocab × 4 = exactly 16.00 GiB, a live tensor no
  recompute touches, which kept every earlier 32K posture at CP=2 (CP=1 missed by 12.31 GiB at PP=1). With the chunked
  cross-entropy CP=1 peaks at 70.4 GiB at full recompute, and removes the packed CP partition, the Mamba CP all-to-alls
  and their THD reordering. Each layer whose activations are kept rather than recomputed costs about 2.2 GiB: keeping
  8 layers peaks at 88.2 GiB, and keeping 16, or selective recompute, runs out of memory.
- **HybridEP on packed data needs full-length packs** (`dataset.dataset_kwargs.pad_to_max_length: true`): it sizes its
  buffers from the first dispatch's local token count and faults (`cudaErrorIllegalAddress`) when an expert-parallel
  peer holds more tokens, which packs padded only to their own length allow.
- **Packed SFT loss curves are not comparable across two fixes of the packed path**: the context-parallel partition
  of packed batches (at CP>1 with more than one pack per replica, a run without it trained on a corrupted partition
  and logs about 0.03–0.04 nats lower) and the pad tokens left out of the MoE routers' statistics (0006 in the
  submodule section below; on parquet packs it covers the padding inside each document only where the dataset factory
  passes the pad multiple to the parquet dataset). A run trains on whichever of them the code it was launched from
  contains. A resumed packed SFT run starts each later epoch at the first pack (the batch sampler advances its count
  as it yields, as upstream's does, without upstream's per-epoch shuffle). A run resumed from code without that fix
  restarted every later epoch at its resume point; an uninterrupted run reads the same packs either way.
- The campaign is logged in `docs/investigations/nano30b-sft-perf-campaign.md`, and
  `tests/unit_tests/test_nano_stage_quickstarts.py` pins both files (the benchmark to the XL SFT field by field, the
  quickstart to the benchmark plus its levers).
- The August config's topology study (`/projects/a5k/public/logs/infr71_wave2/docs/nano30b-32k-topology-campaign.md`)
  still holds for unfused cross-entropy: TP=2 +48.9%, PP=2 +18.3%, PP=4 +30.1%, EP=8 +77.2%.
- For 8K-seq work (no shipped config since the demo was dropped; none of the 32K
  constraints above apply at 8K): the measured topology was TP=2, EP=2, PP=4, DP=2 on
  8 nodes (node-local TP+EP), ~3.4 s/iter at GBS 16, CP=1.
- Zero NCCL hangs through 500+ iterations — keeping EP on NVLink avoids Slingshot all-to-all hangs

**Why node-local TP+EP matters:** Cross-node EP drops throughput 14x because MoE all-to-all over Slingshot/CXI is extremely slow. Rule: **TP × EP ≤ 4** to keep both on NVLink.

### Nemotron 3 Super (120B-A12B) on Isambard

**Going-forward warm-start configs live in `configs/pa_warm_start/`** (endorsed topology +
1B reasoning mix defaults; see that dir's README). Submit all training via
`isambard_sbatch --nodes=N pipeline_training_submit.sbatch <config> super sft` — no
train-tunnel allocations or srun-overlap attach workflows.

**Best validated (BF16, 2026-06-10): TP=1 · CP=(min that fits) · EP=4 · PP=22, ETP=1** —
~75-84 TFLOP/s/GPU, ~1,000+ tok/s/GPU solo (≈2.4× the old TP=4 layouts per GPU).
- **TP=1 is the speed.** Under parallel folding the experts (215 of 230 GB) are EP-sharded
  regardless of TP, so TP only slices the 44 memory-bound Mamba scan kernels and the
  non-expert GEMMs below their efficiency knee (TP2·CP2 measured 15% slower than TP1·CP4
  at identical sharding). Never use TP>1 on this model without a measured reason.
- **CP is a memory lever, not a speed lever** — it divides tokens/rank, the only thing
  that shrinks the un-recomputable, un-offloadable MoE token-dispatch transient.
  8192 tok/rank fits at PP=22 (84 GB stage-0): 32K→CP4, 8K→CP1. CP must stay node-local
  (TP×CP ≤ 4): cross-node CP traffic (TE ring p2p + per-layer Mamba CP **all-to-alls**)
  hangs Slingshot every ~13 iterations. **CP>1 requires packs with pad_seq_to_mult ≥ 2×CP.**
- **PP=22** (88 layers ⇒ PP ∈ {8,11,22,44}): PP=11 OOMs at 8192 tok/rank. The 1F1B
  stage-0 activation residency is PP-invariant (~88 layer-µb once µb/pipe ≥ PP) — deeper
  PP frees only weights/optimizer. Needs `dist.distributed_timeout_minutes: 45` (the last
  stage's first recv exceeds the 10-min default) and recompute `[moe, shared_experts]`
  + all-7 `offload_modules` (offload is measured-free; recompute-drop OOMs).
- **EP fold rule:** EP must divide DP×TP×CP — at TP1/CP1, DP must supply the fold width
  (e.g. DP=4 for EP=4). Mind GBS/DP µb-per-pipe vs bubble: bubble = (PP−1)/(µb+PP−1).
- **fp32 SSM state** (`ISAMBARD_FP32_SSM_STATE=checkpoint`): costs ~0-5% (memory-neutral,
  checkpointed) and is **mandatory for long-doc packs** — bf16 inter-chunk SSM state NaNs
  deterministically on specific ~32K single-document sequences. Unnecessary at 8K.
- Startup at deep PP is a serialized JIT chain (~75 s/stage), not NCCL:
  `ISAMBARD_COMM_WARMUP=1` (group inits in 2.2 s) + `TRAIN_PERSISTENT_TRITON_CACHE=1`
  (warm nodes skip compilation). **Never benchmark concurrent runs in one allocation**
  (5-way concurrency measured ~30% per-slot slowdown from fabric/Lustre contention).
- **Recommendation: use BF16 for Super.** FP8 causes stochastic alignment crashes in MoE routing.
- **NEVER enable `ISAMBARD_COMM_WARMUP` at deep PP — it is a ~10× steady-state regression.**
  Root-caused 2026-06-13 (default now OFF in `pipeline_training_launch.sh`). On Super-120B
  PP22·CP4·seq32K, byte-identical config, single-group, the comm-warmup A/B is unambiguous:
  **comm-warmup ON → ~277-290 s/iter (6.5 TFLOP/s/GPU); OFF → ~28 s/iter (64 TFLOP/s/GPU)**
  (`5209084` vs `5210950`, both fp32-SSM off). The eager warmup batches a 4-byte send/recv with
  both PP neighbors to pre-init the per-pair p2p transports; at deep PP that establishes the PP
  p2p channels in a config that cripples the steady-state ~168 MB activation exchanges (shows as
  inflated forward/backward compute AND send/recv timers — pipeline-stall propagation). Harmless
  at shallow PP (mqv2 PP8/seq8K fine either way), so it slipped through. **This — not placement —
  was the 14× we chased.** Earlier multi-group runs looked "14× slow" only because they also had
  comm-warmup ON; raw fabric is healthy (nccl-tests 124 GB/s multi-group).
- **Placement is a secondary ~1.5× lever** (still under study): with comm-warmup OFF, v4
  `7ws1u9y6` hit 21 s on group4 (Jun-10) vs `5210950` 28 s on group12 (Jun-13) — same
  single-group config, uniform p2p-stall, no straggler → group-specific / cross-node-p2p
  congestion, NOT a single-vs-multi-group principle (no comm-warmup-off multi-group datapoint
  yet). Parallel folding keeps EP+CP all-to-all node-local (NVLink); only PP p2p crosses nodes.
- **Worktree submission:** export `GEODESIC_REPO_DIR=<worktree>` (or the legacy
  `TRAIN_REPO_DIR`), or simply submit from the worktree, so the launcher finds a
  worktree-only config. With no override `REPO_DIR` falls back to the submission directory
  (`SLURM_SUBMIT_DIR`, then `$(pwd)`), so submitting from elsewhere misses the worktree config
  and ft restart-loops on `Override YAML not found`. Helper to pin single-group across all 12
  Dragonfly groups (group
  N = `nid[10000+(N-2)*110 .. +109]`) and backfill the first to free: see
  `scripts/` single-group-pin pattern (`--exclude` every group but one; `--switches=1` is
  insufficient — `MaxSwitchWait`=300 s falls back to multi-group).

**Legacy reference (superseded):** TP=4·EP=8·PP=4 @128 GPUs: 3.5-3.7 TFLOP/s/GPU, cross-node
EP hangs every ~2-3 h; TP=4·EP=4·PP=8 node-local: stable but ~28 TFLOP/s/GPU.

### Pretraining quickstarts

All run `--mode pretrain`: the NVIDIA `nemotron_3_*_pretrain_config` recipes
(pretraining LR/schedule/init) via the `pretrain()` entry point. The Nano stage-1 and Super
quickstarts start from a **random init** with no checkpoint loaded; the Nano midtraining benchmark and
quickstart warm-start from the stage-1 final checkpoint, as production's stage 2 does. **These are NOT the
certification gate** — image qualification stays on the SFT quickstart. The Nano and Super ones
follow different standards.

**Nano — the control-pretraining baseline and its fastest configuration, 50 iterations on 64 GPUs**
(replaced the 128-GPU
ClimbMix-Sample quickstart on 2026-09-28). Two files: a baseline benchmark and the quickstart built on it.
`nemotron_nano_quickstart_pretrain_baseline.yaml` is a `base_config:` overlay (`scripts/training/config_compose.py`) of
`configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_pretrain.yaml`, so it
inherits the whole stage-1 posture — seq 8192, the campaign blend, recompute, the
`comm_overlap:` DP block, PAO — and a change to that posture reaches the benchmark unedited.
It restates only: **GBS 512** (8 microbatches per replica at DP=64, the same per-GPU work as
production's GBS 2048 at 256 GPUs, the filtered arm's stage-1 width),
`train.exit_interval: 50` (`train_iters` stays 29881, so the LR warmup is production's
iteration for iteration, and the exit writes no checkpoint), `checkpoint.load`/`save: null`,
its own `dataset.path_to_cache`, `logger.wandb_save_dir` and `wandb_exp_name`, and
`dist.distributed_timeout_minutes: 20`.
`tests/unit_tests/test_nano_stage_quickstarts.py` fails if any other field diverges from the
baseline. The quickstart, `nemotron_nano_quickstart_pretrain.yaml`, is that benchmark plus the
performance campaign's levers and nothing else (the same test pins it), launched with the two
launcher settings in `nemotron_nano_quickstart_pretrain.env` as an `ISAMBARD_ENV_OVERRIDES` file:
**4.954 s/iter, 1.90x** the baseline benchmark on the same allocations (the pre-registered six-cycle
comparison: goal established), with its 500-iteration loss inside the baseline's band. At 256 GPUs (GBS 2048)
it is 1.90x as well, and its loss leaves the band only over iterations 1–50, on the low side (campaign log
E-063). The Megatron-LM changes it needs are carried commits of the pin (see
"Megatron-Core Submodule"); the production configs do not use its levers, except the control-pretraining
V2 E2E arm's stage 1 (`configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/`). 32 GPUs is an override,
not a second file: `--nodes=8 ... train.global_batch_size=256`. Both are scored as the **mean** step
over iterations 26-50 (`scripts/telemetry/score_run.py`); the performance campaign is logged in
`docs/investigations/nano30b-pretrain-perf-campaign.md` (see "Performance probes" under Usage).

**Nano midtraining — the control-pretraining baseline's stage 2 and its fastest configuration, 200 iterations on
64 GPUs.** `nemotron_nano_quickstart_midtrain_baseline.yaml` is the same kind of overlay, of
`configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_midtrain.yaml` (seq 32768,
TP1·CP2·EP4·PP1, full recompute, the ten-corpus long-context mix, the annealing schedule), restating the
same fields: **GBS 64** (2 microbatches per replica at DP=32, production's per-GPU work at GBS 512 on
512 GPUs), `train.exit_interval: 200`, `checkpoint.load`/`save: null` — the inherited
`pretrained_checkpoint` still warm-starts it weights-only from the stage-1 final, as production's first
segment was — its own cache, W&B name and a 20-minute timeout. The same test pins it. The step takes
a while to settle (iteration 1 alone ~100 s), so it is scored over **iterations 151-200**, the window
its calibration fixed with a rule chosen before the runs: **6.146 s/iter** as-is (mean of three runs) =
5,332 tokens/s/GPU, 13.17% MFU, 83.4 GB reserved in training. The quickstart,
`nemotron_nano_quickstart_midtrain.yaml`, is that benchmark plus the midtraining campaign's levers and nothing else
(the same test pins it), launched with `nemotron_nano_quickstart_midtrain.env` as an `ISAMBARD_ENV_OVERRIDES` file,
which pins the fp32 SSM state to its checkpointed mode: the pretraining quickstart's parameter-gather overlap, host
settings, BF16 gradients, chunked linear cross-entropy, HybridEP with router fusion and FP8 dense layers, and
selective `[moe, shared_experts]` recompute in place of full recompute, still at CP=2 — **1.648×** the baseline
benchmark on the same allocations (the pre-registered four-cycle comparison: goal established; **3.745 s/iter** on one
switch group against its 6.146 s), 77 GB reserved, with its 500-iteration loss inside the baseline's band. Its
chunked cross-entropy is the pin's carried commit 0005. The production configs
do not use its levers, except the control-pretraining V2 E2E arm's midtraining
(`configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/`), which takes all but the selective recompute and
keeps the baseline's full recompute. The campaign is logged in `docs/investigations/nano30b-midtrain-perf-campaign.md`.

**Super — the 128-GPU, 1B-token standard** (Kyle, 2026-08-05): **seq 8192, GBS 3072**
(= 25,165,824 tokens/iter), **all 128 GPUs / 32 nodes, 1B tokens** (`train_iters: 40` =
1,006,632,960 exactly). Dataset: `Kyle1668/ClimbMix-Sample` (**24,757,534,866** tokens under
the base tokenizer — exact, from the `.idx`; the 1B run is a single pass over ~4% of it),
tokenized with `geodesic-research/nemotron-base-tokenizer` (`--append-eod`, EOD id 2). The
zero-embedding Base-CPT trap does not apply from scratch, so there is no filtering step.

| quickstart | topology (·ETP1, mbs 1) | measured (solo, zero overrides) |
|---|---|---|
| `nemotron_nano_quickstart_pretrain.yaml` | the baseline benchmark plus the campaign's levers: recompute `[moe_act]`, FP8 dense layers with BF16 parameters, BF16 gradients, HybridEP with the EP all-to-all overlap, chunked linear cross-entropy | **4.954 s/iter** (mean of twelve runs in six paired cycles, iterations 26–50) = 13,230 tokens/s/GPU, 275.3 TFLOP/s/GPU (27.8% MFU), 1.90x the baseline benchmark on the same allocations — the levers given as Hydra overrides on the baseline benchmark; this file as committed: 4.944 s/iter (job 6961393) |
| `nemotron_nano_quickstart_pretrain_baseline.yaml` | TP1·CP1·EP4·PP1·DP64 at GBS 512, selective `[core_attn,moe,shared_experts]` (all inherited from the baseline) | **9.328 s/iter** (mean, iterations 26–50) = 7,026 tokens/s/GPU, 146.2 TFLOP/s/GPU (14.78% MFU), loss (41–50) 6.869, 0 NaN — job 6930454, 64 GPUs, the overlay's fields given as Hydra overrides on the baseline |
| `nemotron_nano_quickstart_midtrain.yaml` | the midtraining baseline benchmark plus the campaign's levers: parameter-gather overlap, host settings, BF16 gradients, chunked linear cross-entropy, HybridEP with router fusion, FP8 dense layers with BF16 parameters, selective `[moe, shared_experts]` recompute | **3.745 s/iter** (the six single-group runs of four paired cycles, iterations 151–200; 3.782 s over all eight) = 8,749 tokens/s/GPU, 213.8 TFLOP/s/GPU (21.6% MFU), 1.648x the baseline benchmark on the same allocations, 77 GB reserved — the levers given as Hydra overrides on the baseline benchmark; this file launched as documented: 3.733 s/iter (job 6973271) |
| `nemotron_nano_quickstart_midtrain_baseline.yaml` | TP1·CP2·EP4·PP1·DP32 at GBS 64, seq 32768, full recompute, warm start from the stage-1 final (all inherited from stage 2) | **6.146 s/iter** (mean of three runs, iterations 151–200) = 5,332 tokens/s/GPU, 130.3 TFLOP/s/GPU (13.17% MFU), loss (191–200) 1.565–1.567, 0 NaN, 83.4 GB reserved in training — jobs 6971129, 6971856, 6971857, 64 GPUs |
| `nemotron_super_quickstart_pretrain.yaml` | TP1·CP1·EP4·PP8·DP16 at GBS 3072, selective `[moe,shared_experts]` | **86.940 s/iter = 28.301 ms/sample**, 171.4 TFLOP/s/GPU (17.3% MFU), loss 12.19 -> 7.65, 0 NaN |

Launch: `ISAMBARD_ENV_OVERRIDES=$PWD/configs/quickstart/nemotron_nano_quickstart_pretrain.env
isambard_sbatch --nodes=16 pipeline_training_submit.sbatch
configs/quickstart/nemotron_nano_quickstart_pretrain.yaml nano pretrain --disable-ft` (the baseline
benchmark: `nemotron_nano_quickstart_pretrain_baseline.yaml`, no env file; the midtraining quickstart:
`nemotron_nano_quickstart_midtrain.yaml` with `nemotron_nano_quickstart_midtrain.env`, `--time=00:30:00`, and its
baseline benchmark `nemotron_nano_quickstart_midtrain_baseline.yaml`, no env file, `--time=00:40:00`) and
`isambard_sbatch --nodes=32 pipeline_training_submit.sbatch
configs/quickstart/nemotron_super_quickstart_pretrain.yaml super pretrain --disable-ft`.

Ladder verdicts, 2026-08-05 (probe window mean iters 10-16, ~9-12% spread from from-scratch
router-load drift; full records in `/projects/a5k/public/logs/pretrain_quickstart_2026-08/`;
the retired Nano file, with its provenance and PAO A/B, is
`git show 8d1d9ab1:configs/quickstart/nemotron_nano_quickstart_pretrain.yaml`).
Nano, measured on the retired 128-GPU quickstart (DP128, GBS 3072): `core_attn`-only
selective OOMs (DP128 static ≈ 56 GiB + Mamba saves + the exactly-4-GiB fp32 CE
logits), CP2+recompute-none is **+29.6%** (mamba CP all-to-alls cost more than the
recompute they remove), mbs 2 dead on headroom. That quickstart's anchor was 25.533 s/iter =
160.2 TFLOP/s/GPU, taken at a 128 MiB bucket with param-gather overlap ON because its
`ddp.bucket_size` was inert (see the Nano-pretrain `comm_overlap` note below). Super: the
offload posture (`core_attn` + `expert_fc1/moe_act`) and TP2·EP2 both **OOM** at 8192
tok/rank from scratch — S0b's `[moe,shared_experts]` recompute is the only fitting posture.
Cluster-driven recipe overrides. All five: `checkpoint.async_save: false` (the Nano files
inherit it from their production stage; on Super the recipe default asserts when only a final
checkpoint is written). The Super quickstart and both Nano baseline benchmarks use dispatcher
`alltoall` (each benchmark inherits it from its production stage) because DeepEP's RDMA path is blocked on
Slingshot; both Nano quickstarts use `flex` with the HybridEP backend, whose EP=4 all-to-all stays
inside the node on NVLink and needs no RDMA. Super only, because only the Super pretrain recipe
sets the defaults being overridden: `mixed_precision: bf16_mixed` (its NVFP4 posture is
Blackwell), `cuda_graph_impl: none`, `cross_entropy_fusion_impl: native` (its "te" impl carries
an upstream stability rejection), and `mtp_num_layers: null`. The Nano recipe already supplies
bf16_mixed, no CUDA graphs, and native CE, which the baseline benchmarks keep; both Nano
quickstarts replace the precision and the cross-entropy with their levers (an FP8 dense-layer
preset, the chunked linear cross-entropy), and the pretraining one sets `mtp_num_layers: null`,
which its EP overlap requires. The Nano files write no checkpoint; Super's final
checkpoint is weights-only at iter 40 (`save_optim/save_rng: false`) — a from-scratch
1B-token model is a pipeline artifact, not a usable model — no coherence test (expected
gibberish; sanity = loss ~12.2 → ~7.6 over the 40 iterations, 0 NaN, as in the anchor above).

### Control-pretraining campaign (`configs/control_pretraining/`)

The full-scale from-scratch runs for the pretraining-data-filtering study, as opposed to the
quickstarts above. `nemotron_nano_control_v1_baseline_500b.yaml` is the unfiltered V1
baseline: Nano 30B-A3B, 500,011,368,448 tokens (29803 iters x GBS 2048 x seq 8192) on 512
GPUs, WSD 1e-3 → 1e-5, 21 optimizer-bearing checkpoints, blended ClimbMix 0.80 / Zyda-2
`sample-100BT` 0.19 / AI-safety discourse 0.01. Launch it as a `--dependency=singleton` chain
of day-long segments rather than one long allocation — see that directory's README, and the
config header for the settings that are not free choices (`lr_wsd_decay_iters`, the
`comm_overlap` restatement above, and `checkpoint.ckpt_assume_constant_structure`). **Three** DP=512 save-crossing pathologies are fixed
structurally, and the README's "Save crossings at DP=512" section records each mechanism
with its evidence. The one that actually stops the run is the third: every save
materialises a 13.679 GiB bf16 copy of the rank's MoE expert weights
(`grouped_experts.py:344` `.contiguous()` on a transposed fused weight, `torch_grouped`
only), and `ckpt_assume_constant_structure: True` sends the *second* save down a cached
short path that never frees it — so the next forward OOMs on the 4 GiB fp32 logits buffer.
Fix: **`checkpoint.ckpt_assume_constant_structure: false`**, which the recipe sets `True`,
so omitting it is not the same as setting it. `model.cross_entropy_loss_fusion: false`
adds 4.00 GiB of margin for +0.31% step time and no numerics change. The other two are the
CXI MR-cache capacity collapse (launcher: `FI_MR_CACHE_MAX_COUNT`) and the rank-0 NCCL
object-gather transport retention (`dist.distributed_backend: "cpu:gloo,cuda:nccl"`).
**Any probe validating save behaviour must cross at least three saves and run the forward
after each** — one that exits at its second save never executes the failing step. On a posture with
`train.manual_gc` on (the fast quickstarts and the xl-50b SFT rerun), the probe must also save no more often than
`manual_gc_interval`. Python's automatic collection is off there, and each save leaves its garbage, that expert-weight
copy included, for the next scheduled collection, so three saves inside one interval run out of memory at a cadence
production never runs (Nano SFT campaign E-010). A **fourth**
pathology sits on the RESUME side: the load's target for the grouped experts is that same full
copy, and `_load_checkpoint_from_path` used to release the allocator cache while still holding
it, so a resumed 64-node segment started ~14 GiB of reserved-but-unused memory heavier than a
fresh one and its first gradient reduce-scatter failed inside NCCL (`Cuda failure 2 'out of
memory'`, never a PyTorch OOM — NCCL cannot reclaim PyTorch's cache; five segments died this way
on 2026-09-10). The load now drops its references before `torch.cuda.empty_cache()` and logs
`memory after checkpoint load`; on every resumed segment reserved should sit within a few
hundred MiB of allocated. See the 30b_baseline README's "Segment rollover" section.

**The campaign's archive of record is the private Hub bucket
`geodesic-research/control-pretraining-models-bucket`**: every completed checkpoint of every
stage (optimizer and RNG state included, the `iter_*/hf/` exports excluded) and every corpus the
stage configs read, mirrored by `scripts/hub/sync_bucket.py` from the manifest
`configs/control_pretraining/bucket_sync.yaml`, run **locally on the tunnel or login node under the
host Python, never as a SLURM job** (Kyle, 2026-09-11; the container's `huggingface_hub` predates
buckets). It copies only iterations at or below `latest_checkpointed_iteration.txt`, compares by
size alone, re-plans after each pass and fails if anything is still pending. The manifest lists the
stage configs, and each contributes its `checkpoint.save` directory and its corpora, so a new
stage is archived automatically once its directory exists and a save has completed; only the
export clone holding the baseline SFT's pruned iteration-600 save is listed explicitly. `configs/control_pretraining/README.md`,
"The archive of record", has the layout and the restore recipe. **The mirror has been stopped since
2026-09-11 (Kyle), so the bucket holds only what it had copied by then**: no midtraining final and no
filtered or reintroduction corpus is in it. Everything since exists only on `/projects` and, once
exported, as Hub revisions; the manifest lists what a resumed pass would archive, and nothing it lists
is archived until one runs. The V2 E2E stage-1 corpora were deleted unarchived (Kyle, 2026-10-03), so a
resumed pass fails on that started stage until they are rebuilt or the manifest stops reading them.

**The campaign's models on the Hub** are the "Control Pretraining" collection: per arm a
`control-pretraining-30b-<arm>-base` repository (the stage-1 and stage-2 checkpoints of an arm that
trains both, as `pretraining_iter_<n>` / `midtraining_iter_<n>`, and a midtraining-only narrowly
filtered arm's midtraining checkpoints only; the final midtraining checkpoint as `main`), **plus one think
repository per SFT run** (`sft_iter_<n>`, the final as `main`), named for its recipe: `-baseline-think`
for the baseline's mainline SFT, `-<arm>-xl50b-think` for the xl-50b recipe (the baseline's
ablation, and the Broadly Filtered and narrow V2 arms' reasoning models; narrow V1 has none), and
`-baseline-xl50b-v2-think` for the baseline's xl-50b ablation rerun on fixed, fast code, whose card carries the
comparison of the two (its `v2` names the rerun, not the narrow V2 arm),
because two SFT runs of one base model would collide in meaning — so revision names are NOT unique
across the collection and the repository is what tells two SFT runs apart. Each carries a model card
listing every revision's tokens seen and W&B training loss, and per stage the data mix, sequence
length, batch, schedule and tokenizer read from the stage's config. `scripts/hub/publish_models.py` builds
them from `configs/control_pretraining/hub_models.yaml` (stages by training config; nothing
restated), exporting each checkpoint from a symlink clone with a patched `run_config.yaml` — the
`torch_grouped` closure the run serialised cannot be imported by the exporter — so the training
tree is never touched, verifying the export by tensor names, and skipping revisions the Hub already
holds. The polling process runs on the host Python, locally (Kyle, 2026-09-12: on the tunnel node,
never interrupting the training runs). The exports need GPUs, and **`--phase submit` is how they get them** (Kyle,
2026-09-14): one single-node job per checkpoint, sized by `export.nodes` / `export.walltime` in the
manifest, so a wave runs in parallel and never competes for the cards another workload holds.
`--phase export` still runs them in the current allocation, which is only safe when nothing else
wants those GPUs — announcement-based turn-taking is a check-then-act race and cost two OOMed waves
and a cancelled evaluation on 2026-09-14. `--phase rolling` (with `--poll-interval`) keeps up with a
run still training: it submits missing exports and uploads only exports whose job has left the
queue. A manifest with an `upload:` block moves the uploads into jobs too: one single-node
`hubupload-<campaign>` job per manifest (a SLURM singleton) runs an `--phase upload` pass, and the
polling process writes nothing to the Hub. The metagaming campaign's manifest has the block (Kyle, 2026-09-23). An export verifies only once the exporter's last write (`hf/megatron_run_config.yaml`) is
present, and a job that left the queue without one is reported by a submitting pass (`submit`,
`rolling`) as an error, with the reason, not resubmitted (an inline `export` or `all` pass exports
it again); an export that does not verify is removed before it is exported again, and only inside
`export_root`. Every pass that acts leaves an export whose job is still queued, or that an inline
export has recorded itself writing, to that export; export jobs are SLURM singletons. A final
checkpoint counts as published only once `main` holds its weights too, by LFS content hash (by name
and size one export cannot be told from another), every revision branches from the repository's
first commit (never from `main`; a first commit holding an export is refused), and a revision without
a usable local export still counts once the Hub holds its finished export. The entry point refuses a
Hub identity outside the manifest's namespaces, since a repository the token cannot see reads as
missing. A card's tokens seen count each stage at its own sequence length times global batch.
`--newest-first` takes each stage's latest checkpoint
first without moving the model cards, which sort their own rows. The campaign README's "The models
on the Hub" section has the full behaviour.

**The going-forward arm is `configs/control_pretraining/30b_baseline/`**, which supersedes V1's
blend with the campaign mix (sheet revision 2026-08-20) as a **three-stage curriculum**:
`nemotron_nano_30b_baseline_pretrain.yaml` (501.3B tokens, seq 8192, **constant** 1e-3 — it
never anneals; ClimbMix's 0.698180 aggregate is split token-proportionally across its 8
shards), then `nemotron_nano_30b_baseline_midtrain.yaml` (52.4B tokens over 10 corpora, seq 32768,
TP1·**CP2**·EP4·PP1, which is the annealing phase, decaying 7.5e-4 → 1e-5 with `cosine`
after a 100-iteration warmup, and pinning `adam_beta2: 0.95` so that warmup outlasts the
second-moment EMA the weights-only warm start resets),
then `nemotron_nano_30b_baseline_sft.yaml` (the reasoning/think post-training: two epochs
≈ 50B tokens of the packed `pa-warm-start-sft-heavy-25b-mix` combined split, seq 32768,
**think-HISTORY** tokenizer, `nano sft` — its `train_iters` is the measured 2988,
`ceil(2 x 764,685 packs / 512)`). The history tokenizer is not interchangeable with the plain think one: that
corpus keeps per-turn reasoning in a `reasoning_content` field, and the plain variant's
`truncate_history_thinking` default renders every non-final assistant turn as an empty
`<think></think>`, dropping 80% of prior-turn traces before tokenization. The encoders are
byte-identical, so only the packed artifact differs — and the packed path names the tokenizer
so the two cannot silently disagree.
Stages 1–2 run 16,777,216 tokens/iter so the optimizer's token batch is continuous across the
boundary, and the two retain **20 checkpoints between them** (14 + 6). From stage 2 on a save
lands every ~10B tokens — `save_interval: 600` at 16,777,216 tokens/iter = 10,066,329,600 — so
stage 3 keeps five of its own, every one retained. All three stages (and
both CPT-validation arms) set `train.exit_duration_in_mins: 1400`: a 24 h segment saves and
exits on its own clock ~40 min before the workq MaxWall, because sbatch `--signal`-based
exits are undeliverable on this stack (the non-Python layers of the step tear down in ~45 s,
measured on job 6107666 — see the 30b_baseline README's "Segment rollover" section).
Smoke variants end at `train_iters` and carry neither exit knob. All 16 `.bin/.idx`
corpora are subsets of one pinned HF repo and share **one** prepare config plus `--subset`,
because `pipeline_data_prepare.py` already derives the output dir from
`slugify_dataset_name(dataset, subset)`. Three traps that arm's README documents and its tests
enforce: `dataset.seq_length` silently defaults to 8192 when omitted (`pipeline_training_run.py`)
while `model.seq_length` is a separate unchecked key; `lfs setstripe` must precede the write
because a `mv` inside Lustre is a rename that never restripes; and **both `.bin/.idx` stages set
`dataset.split: "1,0,0"`**, because Megatron builds a split's dataset for every prefix whether or
not the run reads it — so `eval_iters: 0` is no protection — and a corpus whose train share
rounds up to its whole document count gets an empty validation range that **hangs** the index
builder with no error. `"1,0,0"` makes that split `None`, which the builder skips rather than
slicing, so no empty range exists at any corpus size and no training data is withheld. Measured:
`stack_edu_long` (3,190 docs) and `zyda_ai_docs_long` (1,665) both hang past 180 s at
`"9999,1,0"` and build at `"1,0,0"`.

**Before the full curriculum, run `smoke_e2e_run`** (`configs/control_pretraining/smoke_runs/`):
a `_smoke` variant of each of the three stages, 100 iters x 16,777,216 tokens = 1,677,721,600
each (~5.0B across the chain), warm-starting stage to stage exactly as the real curriculum does
and writing only a final weights-only checkpoint per stage. Blend, parallelism, recompute and
the DP=512 save-crossing settings are carried over UNCHANGED — a test pins each smoke config to
its parent so the two cannot drift — and only length, warmup, checkpoint policy and the
checkpoint/W&B identities differ. Stages 1 and 2 have both run (2026-08-21): stage 1 in ~11 min
at 6.36 s/iter and stage 2 in ~14 min at 8.34 s/iter, each reaching iteration 100 with 0 NaN
and writing its checkpoint. Stage 1's cost is placement-dependent (~9-11 min across the
measured 5.25-6.36 s/iter range); stage 3 remains ESTIMATED at ~8-12 min because its smoke has
not run. Stage 2 was the first execution of the CP=2 / GBS 512 / DP=254 posture at seq 32768 at
this scale: it fits, and the weights-only warm start produced no loss spike.

**Ablations of a baseline stage live in `configs/control_pretraining/30b_baseline_ablations/`**,
one file per variant, each a full stage config that
`tests/unit_tests/test_control_pretraining_30b_baseline_ablations.py` pins to its parent: the
fields that differ between the merged variant and the merged parent must be exactly the ablated
fields plus the run identity (checkpoint directories, W&B name, TensorBoard directory) and, where
the batch changes, `checkpoint.save_interval` restated to keep the parent's token spacing. The
one on file is `nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml` — the stage-3 SFT re-run on the
revised post-training mix `geodesic-research/pa-warm-start-sft-xl-50b-mix` (~50B tokens, 8,924,246
conversations against the mainline mix's 5,702,903, pinned at `ec0b9197`, its `default` config
built table-driven from `30b_baseline_ablations/corpora.tsv` like every arm's corpora) at
8,388,608 tokens per iteration, which at seq 32768 is GBS 256, half the parent's, from the same
midtraining final, on 256 GPUs / 64 nodes so that DP=128 keeps the parent's 2 packs per replica
per iteration. Corpus and batch are the only variables; one pass over the mix is the parent's ~50B
token budget at half the batch and twice the steps, so `train_iters` is the measured 5976,
`ceil(1,529,684 packs / 256)` over the 32 per-shard packs (Kyle, 2026-09-13: one epoch, 50B
tokens). How many shards that is belongs to the
arm's `corpora.tsv` and nowhere else: the count is a host-memory budget for the pack job, which
holds a shard's whole pack set in RAM before writing it. Two earlier drafts (the parent's mix
at half the batch, and a longest-chain-of-thought re-selection at that batch) were queued,
cancelled on 2026-09-07 before running, and removed on 2026-09-13. The ablation trained on
2026-09-14 in one 64-node segment of 12 h 08 min and is published. **The filtered arms' reasoning
models run its recipe** (Kyle, 2026-09-23): `nemotron_nano_30b_filtered_mini_2plus_sft_xl50b_gbs256.yaml`
and `nemotron_nano_30b_filtered_gpt55_4plus_v2_sft_xl50b_gbs256.yaml` in the same directory, each the
ablation's config with only the corpus (that arm's cut of the xl-50b mix, the narrow one being the
baseline mix minus exactly 668 conversations), the warm start (that arm's midtraining final) and the
run identity changed — `tests/unit_tests/test_control_pretraining_30b_filtered_sft_xl50b.py` pins the
set, so every model sees the baseline's SFT tokens at its batch. Read any SFT arm's evaluation
reach-first: the parent SFT's greedy coding cell hit the 32k budget on 93% of completions (evals,
2026-09-07), so compare arms only on rates computed over all items, under an identical and
explicitly stated generation budget, never on the W&B component mean, which is conditional on the
completions that reached the scorer. `nemotron_nano_30b_baseline_sft_xl50b_gbs256_v2.yaml` (+ `.env`) is that
ablation rerun on fixed, fast code (Kyle, 2026-10-02). It keeps the same training problem. It changes only the Nano
SFT quickstart's levers, at the quickstart's values, and its run identity. It must launch from a checkout containing
PR #52's packed-SFT fixes. Compare it with `control-pretraining-30b-baseline-xl50b-think` by evaluations, not loss
curves: the ablation's logged loss reads about 0.03–0.04 nats low because of its corrupted CP partitions. Its launch
and status are in that directory's README.

**The treatment arm is `configs/control_pretraining/30b_filtered_mini_2plus/`**: the same three
stages on the same corpora with AI-scheming literature removed — every document that **carries a
canary string OR** whose gpt-5-mini cost-gate score is >= 2 in
`sudoers/control-pretraining-filter-annotated` @ `eab743dd` (Kyle's rule, 2026-09-04; the split
names say only `mini_2plus`, and the earlier splits that applied the score rule alone and kept
the 476 canary documents are withdrawn, their tokenised corpora deleted). This is a **wider cut
than that repo's own `filter_decision`** (`canary or judge_score >= 4`, reachable only at
`mini >= 4`). A null `mini_score` is retained, so the score half's reach is bounded by the nano
relevance gate above it (36.1% of prefilter survivors screened out early). Iteration counts,
corpus-level blend weights, topology and schedule are the baseline's verbatim — each source
therefore gets the same token budget over a smaller corpus, i.e. more epochs, not fewer tokens;
only ClimbMix's eight shard weights are this arm's own, because they follow the filtered
slices' measured tokens — and
`tests/unit_tests/test_control_pretraining_30b_filtered.py` asserts that the ONLY fields
differing between the merged arms are the data paths and the run identities. Corpora are the
`<subset>_filtered_mini_2plus` splits of the same HF repo, pinned in both prepare configs at one
revision, and nothing is built or launched from a pre-2026-09-04 revision. **A revision is not
uniformly safe across subsets**: the rebuild pushed one split per commit, so at an interim
revision some splits state the corrected rule while others are still the withdrawn score-only
build. Which is which is readable per split from its own `_provenance.json` — the
`corpus/judge_score_arm` step carries `also_remove_flag: canary` on a corrected split and omits
it on a withdrawn one, and that step is not always the first transform, so find it by type. A
subset whose source is not yet safe is held by setting its `docs` to `PENDING` in the arm's
table: `plan_corpus` refuses a PENDING row whatever its shard mode, so `build_corpora.sh`
refuses the whole `all` plan while leaving the rest buildable by name. A document count is NOT
a substitute for the hold, because a sliced corpus derives its ranges from that count and so
cannot notice a source with surplus rows — it would build, sum correctly and verify clean. The arm README records
what has been built, verified, held and withdrawn. Every arm's data build is table-driven:
`configs/control_pretraining/build_corpora.sh <arm>/corpora.tsv <stage|all> [subset ...]` submits
it (naming subsets submits only those rows, from the same table, with the arm's job names;
`BUILD_STEPS=prepare` submits only that step of each chain — the re-stamp of an already-tokenized
corpus's provenance after its pin moves, without re-tokenizing; `BUILD_SHARDS=0,1` submits only
those shards' own jobs of an already-split corpus, never its shared prepare or split — how a
32-shard pack is fed to the queue a few shards at a time, or one failed shard is re-run) and
`verify_corpora.py` checks the result against the same table (prepare identity incl. revision,
document counts, exactly 4 bytes per token, tokenizer, `--append-eod`; naming subsets checks only
those rows, so one corpus is verified while the rest of its stage still builds), both reading it through
`corpora_table.py`. A filtered arm is additionally audited against two references it did not
produce by `audit_filtered_corpora.py <arm>/corpora.tsv --baseline-table <baseline>/corpora.tsv
--filter-tag <tag>`: the baseline arm's build and the `filter_stats_<tag>` config of the pinned
revision (baseline minus filtered must equal the removed documents and tokens exactly), and with
`--content` every filtered document is aligned in order to its baseline document, sampled pairs
compared token for token, and sampled Hub rows of the filtered and removed splits checked
present and absent — the check that catches a corpus built from the unfiltered split.
`audit_corpora.sbatch <audit args...>` submits it as its own 1-node job, one per corpus,
forwarding every argument into the container, because a pretraining corpus's content audit
runs for hours. Each
by-content lookup examines at most `--search-candidates` equal-length documents and reports
itself truncated rather than absent past that, so the bound must exceed the largest same-length
pool of the corpora searched — the baseline's, which the report records as
`largest_equal_length_pool_baseline` (the pretraining corpora need ~110000 against a default
sized for the smaller stages; ClimbMix full needs its own measurement).
`--canary-column canary` adds the zero-canary proof as a join through the removed split, and
directly on a filtered split that carries the flag itself (the `_filtered_gpt55_4plus` splits do;
the `_filtered_mini_2plus` splits carry no judge columns): no retained row may be flagged, and the
removed split's flagged rows must number the statistics' `n_canary`,
and with `--content` every one of them must then be absent from the built corpus.

**The third arm is `configs/control_pretraining/30b_filtered_gpt55_4plus/`, and it is a
midtraining stage only** (Kyle, 2026-09-19): the Broadly Filtered arm's pretraining final
(iteration 29881) annealed through the same midtraining stage on corpora cut by the narrower rule
**canary OR `judge_score >= 4`** — the annotation repository's own `filter_decision`, reachable
only for documents the cost gate escalated at `mini >= 4`, so a null score is retained and the
unjudged mini-4/5 documents are retained and flagged in a column of their own — as the
`_filtered_gpt55_4plus` splits of the same repository. Far fewer documents are removed than under
`mini >= 2`. **Only the midtraining is precisely filtered**: the model is broadly filtered through
501.32B tokens of pretraining and precisely filtered through 52.4B of midtraining, so its
difference from the Broadly Filtered arm is the anneal alone, and the card and docs say so. Its config is the Broadly Filtered midtrain's with the data paths and run identity changed and
the SAME `pretrained_checkpoint`; `tests/unit_tests/test_control_pretraining_30b_filtered_gpt55_4plus.py`
asserts exactly that, plus one thing the table needs that copying the other arm would get wrong:
**every one of its ten `corpora.tsv` rows is tagged `midtraining`, including
`ai_safety_and_adjacent`**, which the three-stage arms tag `pretraining` because their stage 1 reads
it too — copied here, that tag would make `build_corpora.sh <table> midtraining` plan nine corpora
of ten and drop the one the study is about, with every other check clean. A test couples the
counts to the prepare config's `revision` (both PENDING until dataset-builder publishes, both
filled in one change). The arm is built, audited, trained (2026-09-20, W&B `766veqps`, 3126
iterations, exit 0) and published as `geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-base`;
its README's "The run" section records one trap for every chained run: a singleton segment that
starts after a FINISHED predecessor trains nothing but **re-saves the final iteration in place**,
because the loop-exit save fires whenever the step is not a multiple of `save_interval` and the
distributed save overwrites a non-empty directory with a warning instead of refusing — so never let
an export of a final checkpoint overlap the next segment's start.
It ran on 256 GPUs (64 nodes, DP=128, four micro-batches per replica), the shape the Broadly
Filtered midtraining ran on (3126 iterations in 9.51 h at 10.95 s/iter, W&B `5rizzdv4`; this arm
9 h 29 min at a mean 10.80 s/iter), as a two-segment singleton chain with `--disable-ft`; no SFT
stage is planned. **It is deprecated (Kyle, 2026-09-23) in favour of
`configs/control_pretraining/30b_filtered_gpt55_4plus_v2/`**: the same rule, lineage and recipe on
the `_filtered_gpt55_4plus_v2` splits at `c6419e3c`, built from the annotation revision at which the
47,454 escalated-but-unjudged documents V1 retains were judged (8,439,631 documents retained, 10,923
fewer than V1). V2 E2E (below) is the Narrowly Filtered model the study's group figures report, since
they compare only models filtered end to end (Kyle, 2026-10-03); V2 (broad pretraining, narrow
midtraining) is kept and reported alone, and it is the narrow arm that was post-trained; V1 is
deprecated and in no figure (Kyle, 2026-09-26). The same test module covers both arms, parametrised
over them.

**V2 E2E is `configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/`** (Kyle, 2026-09-30): V2's
rule from the first pretraining token, a from-scratch stage 1 on the `_filtered_gpt55_4plus_v2e2e`
pretraining splits and then V2's midtraining from that stage's final, both at the baseline's widths
(512 GPUs; DP=512, then CP2 DP=256). Its stage 1 trains in the fast Nano pretrain posture (the
quickstart's levers and `.env`, one table in `tests/unit_tests/campaign_config.py`, except that the
gradient NaN check stays on), which changes numerical precision, so the arm differs from the baseline
in more than data. Two gates bound that: `probe/probe.sbatch`, one 130-node job on the baseline's data
before the launch (NVLink sweep, speed, memory across four saves, identical-batch parity against the
baseline, the midtraining handoff), and the pre-registered `loss_gate.yaml` during stage 1; a failure
stops stage 1, which is debugged in that posture, never restarted in another (Kyle, 2026-10-01). L1 stopped
the first launch at iteration 1224 in one warmup-descent window, the run descending faster than every
reference; Kyle judged the loss fine and waived L1 and L2, so stage 1 runs under L2b (Kyle, 2026-10-02). Its
midtraining trains in the fast Nano midtraining configuration less its selective recompute (the midtraining
quickstart's levers and its `.env`, `FAST_MIDTRAIN_LEVERS` in the same file, with the baseline's full
recompute; Kyle, 2026-10-01), a precision change too, bounded by `probe/probe_midtrain.sbatch`: the baseline's
stage 2 at production width, against production's midtraining run for a bounded loss offset.
Its five pretraining corpora are pinned at the data revision `a815dfe7`; its other ten corpora are
V2's builds. The arm README has the gates, the launch and the storage.

**Knowledge reintroduction is `configs/control_pretraining/30b_trustedmonitor/`** (Kyle, 2026-09-29):
continual pretraining of the Broadly Filtered and narrow V2 midtraining finals (`iter_0003126`) on
the deduplicated union of the documents each family's filters removed, never-seen documents only,
50/50 with replay of the parent's midtraining blend, at the midtraining LR held constant and GBS
256, beside a replay-only control per family, for a per-family number of epochs (three broad, five
narrow; Kyle added the narrow family's 4th and 5th on 2026-09-30; more can be added). A third family,
V2 E2E's (five epochs of 76 iterations, its union pinned at `61c9d1d2`), has its links rendered too; they
start from that arm's midtraining final once it exists. Each epoch is its
own job (a "link"), submitted one at a time by `configs/control_pretraining/submit_chain_link.py` only
after the link before it has saved: queued successors would count against the account's node cap,
which every job re-checks at start and cancels itself over, and the tool refuses a dirty tree, a stale
link file, a save directory that is not where the link starts, a duplicate job, and a launch setting
(`ISAMBARD_*`, `TRAIN_*`, `GEODESIC_CONTAINER_*`) the job would inherit from the submitting shell. **The link YAMLs
are generated, never edited**:
`generate_epoch_chain.py` derives them from `chain.yaml` and each family's posture config (its parent
midtraining's own, or for V2 E2E V2's as-is midtraining, since the continual pretraining runs as-is), and a
test fails on any drift. Link 1 warm-starts from the parent's weights, which it loads only while the
arm's save directory holds no checkpoint, so a smoke must never save there; every later link resumes
the previous link's full state and sets two fork options that exist for exactly this:
`checkpoint.reset_data_position` (the resumed run builds a fresh dataset sized to its own remaining
iterations and reads it from sample 0, while the step, consumed-sample counters, optimizer and
scheduler carry over) and `checkpoint.ckpt_step`, which setup refuses to honour silently: a
`ckpt_step` naming a checkpoint `load` does not hold raises, where setup would otherwise train from
random initialisation with no error. Every link's blend is weighted in whole samples summing to the link's
own, settled so Megatron's `ceil(size × weight)` sizing builds exactly that many: fractional weights
build a few surplus samples that the sampler leaves unread at random, which would drop union samples
from the pass. `scripts/data/report_blend_coverage.py` confirms each link before it runs. The arm
README has the length arithmetic, the gates (including a storage check before each broad link, since
every link keeps its optimizer state) and the launch.

**CPT validation (`configs/control_pretraining/cpt_validation/`)**: the campaign's CPT leg —
continual pretraining of the released **Nano-Base** and **Super-Base-Chat-Init** checkpoints on
50% ClimbMix / 25% AI-safety discourse / 25% arXiv for 10B tokens (398 iters × GBS 3072 × seq 8192, the
128-GPU pretrain-quickstart topologies; LR 1e-5 cosine → 1e-6, warmup 0.10). Launch via
`nano cpt` / `super cpt` with `--disable-ft` (both runs outlive the ft heartbeat wall;
`load == save` supplies resume). The Nano launch is gated on the dead-id pre-flight in that
directory's README — the campaign corpora were built for from-scratch training and were never
filtered against Base's zero embedding rows; Super is immune because Chat-Init grafts them.

### Metagaming-filtering campaign (`configs/metagaming_filtering/`)

The study of filtering training data to reduce a model's tendency to reason about the nature of
its environment when not prompted to (chiefly verbalized eval awareness). Its SFT arm,
`30b_sft_luna_2plus/nemotron_nano_30b_metagaming_sft_luna_2plus.yaml` (run, W&B, SLURM job and HF
repo all `mf_30b_sft_luna_2plus`), is the control-pretraining XL SFT
(`30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml`, its unfiltered
baseline) with only the corpus and the run identity changed, which
`tests/unit_tests/test_metagaming_filtering_sft.py` enforces field by field (the rebalanced corpus
also shifts the subset mix and repeats kept documents, confounds the campaign README quantifies).
The corpus is the
`pa-warm-start-sft-xl-50b-mix-metagaming_rebalanced_luna_2plus` config of the private
`geodesic-research/metagaming-filtering-datasets`. **Naming trap:** in that repository `_filtered_`
names the REMOVED documents, while in the upstream ratings repository
(`pa-warm-start-sft-xl-50b-mix-metagaming-filtered`) it names the KEPT ones — train on
`_rebalanced_` only. The corpus is built by the control-pretraining table tooling, used in place
by design rather than lifted to a shared location (Kyle, 2026-09-23; its jobs keep the `cp-`
prefix), and
every checkpoint is published by `scripts/hub/publish_models.py` from the campaign's own
`hub_models.yaml`. The campaign README has the build, launch and publishing commands.

### Nemotron 3 Ultra (550B-A55B) on Isambard

Ultra is architecturally a scaled Super — same NemotronH hybrid (Mamba2 + attention + Latent MoE) with MTP and 512 routed experts, but 108 layers and hidden 8192. HF id `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` (base: `…-Base-BF16`). Recipe: `nemotron_3_ultra_{pretrain,sft,peft}_config`; train via `pipeline_training_submit.sbatch <config> ultra sft`.

**SFT validated end-to-end on 72 nodes / 288 GPUs** (quickstart 2026-06-05; full 495-iter warm-start SFT 200k 2026-06-09: lm loss 0.90→0.46, 0 NaN, ~28 s/iter, ~21 TFLOP/s/GPU). At 550B total (~1.1 TB in BF16) Ultra is ~5× the Super. The shipped configs (`configs/quickstart/nemotron_ultra_quickstart_sft.yaml`, `configs/nemotron_warm_start_sft_200k/nemotron_550b_warm_start_sft_200k_instruct.yaml`): **TP=4, EP=4, PP=36, ETP=1** (parallel folding → EP+TP both NVLink-node-local; only PP crosses Slingshot), pure BF16 (no FP8/FP4 — MoE routing crashes), precision-aware optimizer with BF16 Adam moments (mandatory at this size), selective recompute (`core_attn,moe,shared_experts`). PP=36 divides the 108 layers (3/stage). Per-GPU memory: ~60 GB on the heavy MoE stages, ~30 GB on Mamba/attn stages (the hybrid clusters MoE layers onto every ~3rd stage → 2× heavier). Measured: **iter 1 ≈ 52 min** (one-time lazy NCCL comm-init at this depth/rank-count), **steady-state ≈ 30 s/iter**, loss healthy, 0 NaN.

**Two non-obvious requirements (the bring-up bit hard on both — see `docs/investigations/ultra-pipeline-init-hang-debug-log.md`):**
1. **`dist.disable_jit_fuser: true`** (in the configs). On torch ≥ 2.2 Megatron's `jit_fuser` = `torch.compile`; at PP=36 the hybrid per-stage layer mix makes first-step JIT compile times diverge → ranks desync (some compiling, others at a barrier) → watchdog. Eager fused ops avoid it. The earlier "64+ nodes hang" symptom was THIS (and the slow first iter below), **not** the Slingshot MoE-alltoall hang documented in `slingshot-nccl-hang-investigation.md`.
2. **Long first-iter timeouts — including Megatron's own process-group timeout.** The first iteration's lazy NCCL comm-init takes **45–75 min** (fabric-load dependent) at PP=36/288 ranks. THREE knobs must all cover it: `dist.distributed_timeout_minutes: 90` in the YAML (Megatron creates its process groups with this timeout — the old 30 was marginal and a busy fabric reproducibly times out the first `recv_forward` at exactly 30:00; `TORCH_NCCL_TIMEOUT` alone does NOT cover it), `TORCH_NCCL_TIMEOUT=7200`, and ft `step`/`out-of-section`=7200 (both defaulted in `pipeline_training_launch.sh`). Steady-state then drops to ~28 s/iter.

**Throughput is best-effort, not yet tuned.** PP=36 with GBS=64/DP=2 → 32 microbatches < 36 stages = severe pipeline bubble (~0.2→low TFLOP/s/GPU). To improve: raise `global_batch_size` so microbatches ≥ PP, consider VPP/interleaved PP, and set `pipeline_model_parallel_layout` to balance the 2×-heavy MoE stages (see the Megatron MoE paper skill). Functionally it trains; these are throughput levers.

**fVPP (virtual/interleaved PP) WORKS on the Nemotron-H hybrid** via `|` segment
separators in `hybrid_layer_pattern` — the older "VPP unsupported on SSM/Mamba" belief was
two stale bridge asserts, since removed. Whether it is FASTER depends on microbatches per
replica, and at PP=8 the crossover sits between 16 and 32 (INFR-71, placement-matched,
same allocation):

| microbatches/replica | no VPP | VPP=4 | verdict |
|---|---|---|---|
| 32 (GBS 64 — the pre-2026-08-05 standard) | 27.50 s/iter | 29.59 s/iter | VPP **+7.6% worse** (TEGroupedMLP experts) |
| 16 (GBS 32) | 17.98 s/iter | 17.61 s/iter | VPP **−2.1% better** (both arms run twice, ranges disjoint) |

**Re-measured on the `cublas_grouped` expert path (INFR-71 wave 2, 2026-08-02, GBS 64):
the penalty SURVIVES — VPP4 +13.5%, PP8·VPP2 stage-0-lite +5.4%, both offload-adjusted
UPPER bounds (the arms carried an offload-fraction handicap bounded only from below). It is
NOT established that the penalty grew.** The mechanism is wait-multiplication, not host
launch pressure: a PP p2p kernel is dominated by waiting for its peer, so splitting one
wait four ways does not quarter it.

So: **enable VPP only at ≤16 microbatches per replica**; above that the non-VPP config
wins — and the shipped quickstart (GBS 128 at DP=2 = 64 µb/replica) is above it. **The VPP
quickstart variant was DELETED 2026-08-04.** The ≤16 regime is reachable — GBS 32 gives
exactly 16, and the table above records that point as VPP's win — but it costs more than it
buys: within the July wave the best GBS-32 arm is 28% worse per sample than the GBS-64 arm
it is paired with. To reproduce a VPP measurement, add
`model.virtual_pipeline_model_parallel_size=4` as a Hydra override to the shipped config.

**Caveat, and it is the tracker's own:** every VPP and `overlap_p2p_comm` measurement above
was taken in the HOST-BOUND regime, before `torch_grouped` removed the expert-launch storm.
The current posture is ~30% exposed comm — the condition `overlap_p2p_comm` exists for —
and neither has been re-measured there. The verdicts stand and the config stays deleted,
but do not cite them as settled until that re-test reports. Full campaign record (preregs,
arm configs, per-arm results, trace analysis):
`/projects/a5k/public/logs/infr71_wave2/` (`docs/`, `arm_configs/`, `prereg/`).

**`overlap_p2p_comm` stays off on this model — measured slower (+14%, 31.45 vs 27.50
s/iter); its historical NaN was an upstream race already fixed in the current 0.19 pin.**
It requires VPP and forces un-batched isend/irecv, which is simply the more expensive form
on CXI. `overlap_moe_expert_parallel_comm` works on Nemotron-H through the pin's carried commit 0003
(upstream's hybrid support, ported; see the patches README): on the Nano pretrain campaign's ladder it
measured −3.4% against step 9's HybridEP / FP8-dense-layer posture, with 32 CUDA connections, and the
Nano pretrain quickstart uses it. Still blocked on this model: `moe_shared_expert_overlap` (latent MoE),
`defer_embedding_wgrad_compute` (would crash). Do not add a `comm_overlap:` block to a config
**at VPP>1** — it force-sets `overlap_p2p_comm=True`/`batch_p2p_comm=False` there. Full
analysis: `/projects/a5k/public/logs/infr71_wave2/docs/vpp-pp-comm-overlap-investigation.md`.

**The one place a `comm_overlap:` block is REQUIRED is the Nano pretrain path**, and for the
same clobbering reason the warning above is about. `nemotron_3_nano_pretrain_config` assigns a
`CommOverlapConfig`, so `comm_overlap.setup()` runs AFTER the YAML merge and copies its
data-parallel defaults (`bucket_size` 128 MiB, `overlap_param_gather` True,
`overlap_grad_reduce` True) straight onto `cfg.ddp` — `_override_user_cfgs` honours only a
`comm_overlap:` block, never `ddp:`. So on that path a `ddp:` block is INERT for those three
fields, and the standing "overlap_param_gather=false for Nemotron-H DP>1" posture is
unreachable through it. Measured at DP=512, shipped vs the same file with the restatement
deleted: `bucket_size` 500000000 → 134217728 and `overlap_param_gather` False → True.
`overlap_p2p_comm` stays False either way at PP=1. Scope is narrow — Nano SFT/PEFT have
`comm_overlap` commented out, and Super/Ultra never set it, so their `ddp:` blocks are live.
Consequence for the retired 128-GPU Nano pretrain quickstart: its `ddp.bucket_size: 500000000`
never took effect, so its 25.533 s/iter anchor was measured at 128 MiB with param-gather
overlap ON, not at the posture the file stated. The baseline benchmark that replaced it at
`configs/quickstart/nemotron_nano_quickstart_pretrain_baseline.yaml` composes the 30b_baseline stage 1
and inherits that stage's `comm_overlap:` restatement, so it runs at the posture it states
(500 MB buckets, param-gather overlap off); the quickstart built on it turns param-gather overlap on
in its own `comm_overlap:` block.

**Conversion needs multiple nodes.** 1.1 TB of BF16 weights does NOT fit Super's single-node (4×95 GB) export path — pass `--nodes` ≥ 4 to `pipeline_checkpoint_submit.sbatch import`/`export` and keep EP node-local. Base coherence (`pipeline_coherence_test.py --generation-mode completion`) likewise needs ≥3 nodes for inference. Warm-start SFT loads the base Megatron checkpoint directly. **Unlike Super, the Ultra base already ships non-zero chat-special-token embeddings** (only 1 unused-token row is near-zero, and it is also near-zero in Instruct — genuinely unused, not a missing graft), so **no Base-Chat-Init graft is needed** (Super needed it to avoid the bucket-#0 Inf; see "Tokenizer choice for Base CPT").

**Coherence / generation for the 550B: use `--backend megatron`.** The in-process vLLM
backend was removed with the container-only simplification (it existed only in the retired
venv, and the qualified image ships a pre-0.21 vLLM that still carries the RayExecutor
rank-sync bug behind the hybrid-Mamba KV-cache `KeyError: model.layers.N.mixer` at PP stage
boundaries). What remains: **`--backend megatron`** (6 nodes, reads the Megatron checkpoint
directly — no HF export needed, validated at TP4·PP6·EP4) and **`--backend endpoint`** (a
stdlib-HTTP client against an already-running OpenAI-compatible server, so any external
serving stack can be pointed at it). `--backend hf` covers Nano-30B (1 GPU) and Super-120B
(4 GPUs) but cannot reach 550B (1.1 TB > 4×95 GB).

```bash
isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch <megatron-ckpt-dir> \
  --backend megatron --hf-model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
  --tokenizer geodesic-research/nemotron-instruct-tokenizer --tp 4 --pp 6 --ep 4 \
  --max-tokens 256 --trust-remote-code
```

Full guide: `docs/ultra-550b-training-and-conversion.md` §4.

### Parallel Folding (expert_tensor_parallel_size)

```yaml
tensor_model_parallel_size: 4       # Attention: 4-way TP
expert_model_parallel_size: 4       # Experts: 4-way EP
expert_tensor_parallel_size: 1      # Experts NOT sharded by TP → enables folding
```

Keeps EP all-to-all on NVLink while using high TP for attention. Only PP crosses Slingshot.

### TensorBoard: always disabled — set `tensorboard_dir: null`

**We do not use TensorBoard** (Kyle, 2026-09-13). Set `logger.tensorboard_dir: null` in every
config you write or touch, which is the whole of it: `training/state.py`'s `tensorboard_logger`
builds a `SummaryWriter` only when that field is set, so null means no event files, no writer, and —
the part that bites — **no attempt to create the directory**.

**Omitting the key is not the same as nulling it, and omission is the failure that actually
shipped.** The recipes default `tensorboard_dir` to `./nemo_experiments/default/tb_logs`
(`recipes/common.py`), i.e. into the submitting checkout — the directory the pitfalls table above
warns fills the disk — so a config that simply leaves the key out still builds a writer and logs
into the repo. Every training config under `configs/control_pretraining/` therefore states
`null` explicitly, and `TestTensorBoardIsDisabledEverywhere` fails a config that either names a
directory or stays silent. The `configs/PA/green-team/` configs still point at `/tmp/tb_logs` and
have not been converted.

That attempt is a real failure mode, not a tidiness question. It happens during setup, after the
allocation is already up, and it kills the run: the shared `/projects/a5k/public/logs/tensorboard/`
is owned by one account with no group write, so a run under any other account dies with
`PermissionError: [Errno 13]` about two minutes in. The xl-50b ablation lost a 64-node segment to
exactly this. The older advice to point the directory at `/tmp` avoided the permission problem and
kept every other cost, including the stale-file-handle crashes that come from several runs sharing
NFS event logs.

**Do NOT "disable" it by raising `tensorboard_log_interval` or clearing the `log_*_to_tensorboard`
flags. Those names lie.** `training/utils/train_utils.py` computes each report under its
`log_*_to_tensorboard` flag and then fans the result out to the TensorBoard writer, the **W&B**
writer, MLflow and Comet alike, with the whole block gated on
`iteration % tensorboard_log_interval`. So `tensorboard_log_interval: 999999` does not turn
TensorBoard off — it turns off **W&B metrics**, which is how a run comes back with no throughput,
memory or runtime series and a healthy-looking log. Leave the interval at 1 and the flags at their
values; null the directory and nothing else.

### Launching training from a login node (salloc shell lost)

If the tunnel/salloc shell dies but `SLURM_JOB_ID` is still in `squeue`, export the SLURM env vars manually so `pipeline_training_launch.sh` can attach via `srun --jobid=… --overlap`:

```bash
export SLURM_JOB_ID=<id> SLURM_NNODES=<n> SLURM_NODELIST='<from scontrol show job>'
export SLURM_JOB_NODELIST="$SLURM_NODELIST" SLURM_NTASKS=<n> SLURM_JOB_NUM_NODES=<n> SLURM_NPROCS=<n>
export SLURM_GPUS_PER_NODE=4 SLURM_GPUS_ON_NODE=4   # else torchrun --nproc_per_node is empty
export SLURM_CLUSTER_NAME=gracehopper               # ft_launcher OneLoggerConfig pydantic-rejects None
export SLURM_SUBMIT_HOST=login01
bash pipeline_training_launch.sh <config.yaml> --model super --mode sft
```

Between retries: `pkill -9 -f "pipeline_training_launch"`, `rm` stale `*_train.out` logs, `rm -rf <save_ckpt_dir>` if an empty checkpoint dir was created (orchestrators may read its `latest_checkpointed_iteration.txt` as completion).

---

## 3. Data Pipeline (`data_*`)

### Files

| File | Purpose |
|------|---------|
| `pipeline_data_prepare.py` | Download HF datasets, tokenize, export JSONL, pack sequences |
| `pipeline_data_submit.sbatch` | SLURM wrapper: `prepare` (download+JSONL), `tokenize` (pretraining `.bin/.idx` + exact token count), pack-only (1 node, 1 GPU) |

### Usage

```bash
# Prepare dataset (download + tokenize + pack)
python pipeline_data_prepare.py --dataset allenai/Dolci-Instruct-SFT --seq-length 8192

# Offline packing only (via SLURM)
isambard_sbatch pipeline_data_submit.sbatch \
  /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 8192 1

# Pretraining-format corpus (.bin/.idx): prepare (JSONL, no packing) then tokenize;
# the tokenize job appends an exact token count read from the .idx. `--config` supplies what
# defines the corpus (dataset, subset, revision, tokenizer) from a versioned YAML — see
# configs/control_pretraining/data/ for the campaign's three. `--revision` pins the dataset to
# a commit SHA; without one the download resolves the current HEAD, so a corpus still being
# built yields different data, and different token counts, per run. CLI flags override the file.
isambard_sbatch pipeline_data_submit.sbatch prepare \
  --config configs/control_pretraining/data/<corpus>.yaml
isambard_sbatch --dependency=afterok:<prepare-jobid> pipeline_data_submit.sbatch tokenize \
  /projects/a5k/public/data/<org>__<name> geodesic-research/nemotron-base-tokenizer tokenized_base

# From an interactive allocation (payload runs inside the container)
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
  python scripts/data/pack_sft_dataset.py \
    --dataset-root /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
    --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --seq-length 8192 --pad-seq-to-mult 1"
```

### Checking what a `.bin/.idx` blend actually reads

`scripts/data/report_blend_coverage.py <config.yaml> --model <m> --mode cpt|pretrain --report-out <json>`
builds a run's training blend on CPU exactly as its launch will (the launcher's own
`resolve_training_config` and `bin_idx_dataset_config`, the loader's own sizing and builder) and
reports per corpus the samples drawn, the samples one pass holds and the documents reached. Run it as
a one-node job (`isambard_sbatch --wrap` around `pipeline_env_exec.sh`); the index caches it writes
are the ones the launch reads. It refuses a run that reads only part of its built dataset, because the
sampler's random order leaves no fixed set of samples to describe: most fractional-weight blends
(Megatron sizes a blend as the sum over corpora of `ceil(size × weight)`, so it builds a few surplus
samples; the parent midtraining blends do), a resume without `checkpoint.reset_data_position`, and an
unweighted lone corpus. It recognises a resume only through `checkpoint.ckpt_step`.

### Important: Always run `pipeline_data_prepare.py` before training

The training pipeline's `HFDatasetBuilder` expects pre-processed data at `dataset_root` with `training.jsonl`, `validation.jsonl`, index files, and packed sequences. **Always run `pipeline_data_prepare.py` first** — it handles HF download, split creation, JSONL export, token counting, and packing in one step.

If you skip the data pipeline and point `dataset_root` at a directory without properly prepared files, the bridge will attempt to download from HuggingFace at training time. This causes issues:
- HF splits (e.g., `multitag_instruct`) don't match the expected `train`/`training` aliases
- No validation split is created
- No packing — blocks rank 0 for hours during training
- HF cache/lock files cause conflicts across ranks

For datasets with non-standard split names (e.g., `--split multitag_instruct`), the data pipeline maps them to `training.jsonl`/`validation.jsonl` so the bridge can find them.

### What's automatic vs. manual

| Step | Automatic? | Notes |
|------|-----------|-------|
| HF dataset download | Via data pipeline | **Run `pipeline_data_prepare.py` first.** Do not rely on auto-download at training time. |
| JSONL generation | Via data pipeline | Creates `training.jsonl`/`validation.jsonl` with proper splits |
| Sequence packing | Via data pipeline | Use `--skip-pack` to defer, or let it pack (can take 10+ min for large datasets) |
| Checkpoint conversion | **No** | Must run the checkpoint pipeline first |

### Calculating `train_iters`

```
train_iters = total_tokens_in_dataset / tokens_per_batch
tokens_per_batch = global_batch_size * seq_length
```

Use exact token counts from packing metadata, not rough estimates.

---

## 4. Checkpoint Pipeline (`checkpoint_*`)

### Files

| File | Purpose |
|------|---------|
| `pipeline_checkpoint_convert.sh` | Shared launcher: env setup, NCCL, srun+torchrun. Modes: `export`, `import`, `upload-all` |
| `pipeline_checkpoint_convert_hf.py` | Python conversion logic (the script torchrun executes on each GPU rank) |
| `pipeline_checkpoint_submit.sbatch` | Thin SLURM wrapper (2 nodes default, override with `--nodes`) |

### Usage

```bash
# Export Megatron → HF (--hf-model and --reasoning|--no-reasoning are REQUIRED)
isambard_sbatch pipeline_checkpoint_submit.sbatch export \
  /projects/a5k/public/checkpoints/megatron/<experiment> \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
  --iteration 300 --push-to-hub

# Import HF → Megatron (4 nodes for Super)
isambard_sbatch --nodes=4 pipeline_checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Upload all iterations (with polling for ongoing training)
isambard_sbatch --time=24:00:00 pipeline_checkpoint_submit.sbatch upload-all \
  /projects/a5k/public/checkpoints/megatron/<experiment> \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning --poll

# From salloc
bash pipeline_checkpoint_convert.sh export /path/to/ckpts \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
  --iteration 300 --push-to-hub
```

### How export works

1. Reads `latest_checkpointed_iteration.txt` or `--iteration N` to find the `iter_XXXXXXX` directory
2. Uses the `--hf-model` you pass (the upstream HF model ID whose architecture + tokenizer this checkpoint should be exported against — there is no auto-detection)
3. Converts via `AutoBridge.from_hf_pretrained` + `load_megatron_model` + `save_hf_pretrained` (multi-GPU via torchrun)
4. Saves to `<megatron-path>/iter_XXXXXXX/hf/`
5. Optionally pushes to HuggingFace Hub on a revision branch (`iter_0000300`)

**`torch_grouped` checkpoints export as they are.** The export applies the `torch_grouped` export-clone repair
itself, so no `run_config.yaml` is patched by hand and nothing is retrained (`scripts/checkpoint/export_clone.py`,
the same repair `scripts/hub/publish_models.py` uses). When the iteration's run_config records the unimportable
expert closure, `pipeline_checkpoint_convert.sh export` builds an export clone before torchrun: links to the
iteration's files plus a repaired run_config, under `GEODESIC_EXPORT_CLONE_ROOT` (default
`/projects/a5k/public/tmp/export_clones_$USER`). The model loads from that clone. The output goes to the
checkpoint's own `iter_XXXXXXX/hf/`, and `hf/megatron_run_config.yaml` is the checkpoint's own, unrepaired
run_config. The clone is removed after a successful export (a clone that cannot be removed fails the job) and kept
after a failed one. Every other checkpoint loads from its iteration directory itself, with no clone. Run directly,
`pipeline_checkpoint_convert_hf.py` refuses a checkpoint that needs the repair unless it is given the clone as
`--load-path`.

For chained training (CPT → SFT → EM → …), pass the architectural-root HF id — e.g. an SFT checkpoint that loaded from a `*_cpt_v2` dir still exports against `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` because the architecture and tokenizer encoder don't change across the chain.

The `torch_dist` checkpoint format supports resharding — conversion parallelism is independent of training parallelism.

### Recommended export settings

Both Nano and Super conversions run on a **single node** (4 GPUs). All EP communication stays on NVLink — no Slingshot needed.

**Nemotron 3 Nano (30B-A3B):**
```bash
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export /path/to/ckpts \
  --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning --iteration 400
# Or directly:
torchrun --nproc_per_node=4 pipeline_checkpoint_convert_hf.py \
  --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning \
  --megatron-path /path/to/ckpts --iteration 400 --tp 1 --ep 4
```

**Nemotron 3 Super (120B-A12B):**
```bash
torchrun --nproc_per_node=4 pipeline_checkpoint_convert_hf.py \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
  --megatron-path /path/to/ckpts --iteration 490 --tp 1 --ep 4 --not-strict
```

- **`--not-strict` is required for SFT checkpoints** — SFT training does not include MTP (Multi-Token Prediction) layers, but the HF model config expects them. Without `--not-strict`, shards containing MTP keys are silently dropped, which also drops `lm_head.weight` and `backbone.norm_f.weight` (critical for generation). With `--not-strict`, incomplete shards are saved with available tensors; MTP weights are randomly initialized but unused during standard generation.
- **Single-process conversion does NOT work for Super** — hangs during checkpoint loading. Always use `torchrun` with EP.
- **EP=4 (node-local) is preferred over EP=8 (cross-node)** — EP=8 on 2 nodes caused Slingshot gathering failures that truncated expert weights. EP=4 on 1 node keeps all communication on NVLink.
- **Hub uploads are ~223GB** per Super checkpoint, 10-15 min at ~700MB/s.

### Known limitations

- **Hardcoded embedding name (fixed)**: `model_bridge.py` previously checked for `"model.embed_tokens.weight"` when handling tied embeddings, which didn't match Nemotron-H's `"backbone.embeddings.weight"`. Fixed to use `"embedding" in task.param_name` instead.
- **MTP mapping warnings**: `"Unrecognized mapping type"` warnings appear for MTP layernorm aliases during conversion. These are cosmetic — the primary mappings still work, but MTP weights are not converted because SFT checkpoints don't contain them.

### Already-converted checkpoints

```
/projects/a5k/public/checkpoints/megatron_bridges/models/
    NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/
    NVIDIA-Nemotron-3-Super-120B-A12B-BF16/
```

---

## 5. Coherence Pipeline (`coherence_*`)

### Files

| File | Purpose |
|------|---------|
| `pipeline_coherence_test.py` | Generate responses to diverse prompts, log to W&B; probe mode (`--probe-spec`) measures given token ids against a pre-registered spec |
| `pipeline_coherence_submit.sbatch` | SLURM wrapper (1 node, 4 GPUs default) |

### Usage

```bash
# Via SLURM (4 GPUs for 120B models)
isambard_sbatch pipeline_coherence_submit.sbatch nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Via SLURM (1 GPU for 30B models)
isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch \
  geodesic-research/nemotron_nano_sft_warm_start_200k

# Local checkpoint with custom W&B project
isambard_sbatch pipeline_coherence_submit.sbatch \
  /projects/a5k/public/checkpoints/megatron/my_experiment/iter_0000400/hf \
  --wandb-project megatron_bridge_conversion_coherance_tests

# Directly, inside the container (no SLURM, uses this node's GPUs)
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; python pipeline_coherence_test.py <model_path> --max-tokens 3000"

# Ultra 550B (too large for --backend hf): --backend megatron reads the Megatron
# checkpoint directly, no HF export needed.
isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch <megatron-ckpt-dir> \
  --backend megatron --hf-model <hf-id> --tp 4 --pp 6 --ep 4 --max-tokens 256

# Probe mode (hf backend): a Nano 30B HF export against a pre-registered probe spec (1 GPU),
# results to /projects/a5k/public/logs/<study>/<run>/masked.json
isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch \
  /projects/a5k/public/checkpoints/megatron/<run>/iter_0000477/hf \
  --probe-spec <probe.yaml> --probe-output-dir /projects/a5k/public/logs/<study>/<run> --probe-name masked
```

### What it does

1. Loads an HF model (Hub ID or local path) with `device_map="auto"` for multi-GPU
2. Generates responses to 8 diverse prompts at `temperature=1.0`, `max_new_tokens=3000`
3. Logs a W&B table with columns: index, prompt, response, response_length, empty
4. Reports summary metrics: total_generations, empty_count, empty_pct

### Probe mode

`--probe-spec`, `--probe-output-dir` and `--probe-name`, given together, replace the built-in prompts with a
pre-registered measurement of how a model treats given token ids, such as a masked marker. The spec (YAML) names the
tokenizer, the ids to count and score (plus drift-reference ids, scored only), the prompts (`{NAME}` in a prompt
stands for one token id), the sampling and, optionally, held-out `.bin/.idx` documents; the spec's tokenizer is used
throughout, never the model's own. For each prompt the probe records the teacher-forced fp32 log-probability and rank
of every scored id at the prompt's end (the marker slot) and the most probable next tokens. It then generates
(greedy, and seeded samples from the full distribution) and counts the counted ids in the generated token ids, never
in decoded text, checking that every step sampled from softmax(logits / temperature) exactly. With documents in the
spec it also scores them teacher-forced: the cross-entropy at the counted ids' targets (marker CE), at every other
target, and at the targets that follow a counted id.

The results go to one JSON file, `<--probe-output-dir>/<--probe-name>.json` (format `coherence-probe/1`, recording
the spec's sha256, the model, the tokenizer and the probe code's revision), which `scripts/telemetry/score_gate.py`'s
probe gates read. It is never overwritten: an existing file is refused before the model loads and again at the
write. The launch is refused when only some of the three probe options are given, with a backend other than `hf`,
with a `--probe-name` holding anything but letters, digits, `.`, `_` and `-`, and with any of `--generation-mode`,
`--n`, `--num-prompts`, `--max-tokens`, `--temperature`, `--system-prompt` or `--output`, since the spec decides the
prompts and the sampling. The W&B run (in `--wandb-project`) is named `probe-<probe-name>-<model>` unless `--run-name`
is given, `<model>` being a Hub model's repository name or, for an absolute path, its components from the one before
`iter_*` onward joined by `__` (its last component when none is `iter_*`), e.g.
`probe-masked-my_experiment__iter_0000477__hf`; it holds every generation in a `probe_generations` table and the
summaries under `probe/`.

### W&B run naming

- **Hub models** (e.g., `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`): `gen-test-NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
- **Local checkpoints** (e.g., `.../my_experiment/iter_0000400/hf`): `gen-test-my_experiment__iter_0000400__hf`

### Notes

- **Nano (30B)**: fits on 1 GPU. Use `--gpus-per-node=1`.
- **Super (120B)**: needs 4 GPUs with `device_map="auto"`.
- **MTP weights**: SFT checkpoints lack MTP layers. Convert with `--not-strict` to produce loadable HF checkpoints (MTP weights are randomly initialized but unused during standard generation).

---

## Running Evals (sfm-evals repo)

Evals live in the [sfm-evals](https://github.com/GeodesicResearch/sfm-evals) repo at `/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals`; see that repo's README for the full command reference. Quick orientation:

- **Pre-reqs for new `geodesic-research` HF models**: upload `configuration_nemotron_h.py` / `modeling_nemotron_h.py`, set `tokenizer_config.json` `"tokenizer_class": "PreTrainedTokenizerFast"`, pre-download 120B+ to shared HF cache, add alias to `just/models.yaml`.
- **Primary commands**: `just submit-instruct-open-isambard MODEL CONFIG` (vLLM on Slurm), `just run-quick-all-api MODEL` (~30–45 min, 11 evals via API), `just submit-quick-all-isambard MODEL` (HF model on Slurm — set `VLLM_TP=4` for 120B). `ISAMBARD_TIME` controls sbatch limit (default `8:00:00`); 20-job-per-user limit on Isambard.
- **Misalignment configs**: `hdrx_sfm_syn` (1503/task, preferred), `ind_sfm_syn` (2671/task); each has 8 tasks `forward/reverse_misalignment_v{1-4}` × 5 system prompts.
- **Results**: W&B project "Self-Fulfilling Model Organisms - ITERATED Evals" (entity `geodesic`) — always filter by group name. Slurm logs at `/projects/a5k/public/data_cwtice.a5k/logs/sfm-evals/`.

---

## NCCL Performance Testing

### Debugging NCCL-looking failures (rendezvous timeout, hang, slow iters)

When a training run fails with symptoms that *might* be fabric-related — c10d KV-store rendezvous timeout ("N/M clients joined"), NCCL watchdog timeout mid-iteration, iters suddenly taking 10-20× longer than expected, `WorkNCCL(SeqNum=...)` timing out — run the benchmark suite **inside the same allocation** to prove whether NCCL/Slingshot itself is at fault. If the benchmark passes, the fabric is healthy and the failure is elsewhere (leftover zombie processes, rendezvous port collision, config mismatch, parallel-run contention).

**Repo**: `/home/a5k/kyleobrien.a5k/isambard-nccl-tests/` — Python orchestrator over upstream nccl-tests with pass/fail thresholds for Isambard GH200. Binaries are already built at `build/`.

**Usage (inside the affected SLURM allocation, e.g. the tunnel that just had a training failure):**
```bash
cd /home/a5k/kyleobrien.a5k/isambard-nccl-tests
module purge && module load PrgEnv-cray cuda/12.6 brics/aws-ofi-nccl/1.8.1
python scripts/run_nccl_benchmarks.py --min-nodes 2 --max-nodes 8 --no-wandb
# (raise --max-nodes to the allocation size if you want the full sweep)
```

Runs ~20 min for the 2..8 sweep. Tests 5 collectives (alltoall, all_reduce, reduce_scatter, all_gather, sendrecv) at each node count against calibrated thresholds (~80% of observed baseline). "PASS" on ≥ the node count of the failing run is strong evidence the fabric is fine.

**Interpreting the result:**
- **All PASS** → NCCL is healthy. Failure was almost certainly at the process layer (zombies, rendezvous port collision, bad config, parallel-run fabric saturation from *multiple* PP=4 training jobs, etc.). Clean up zombie ft_launcher/torchrun/pipeline_training processes and relaunch, optionally with a different `MASTER_PORT_OVERRIDE`.
- **Consistent FAIL on one node count** → capacity issue at that scale — try a different node subset of the allocation.
- **FAIL scattered across collectives/scales** → bad specific node(s). `isambard_sbatch --mark-bad <node> "<reason>"` and move on.

**Typical healthy numbers on a clean allocation (2026-04-22)**: 8-node / 32-GPU all_gather bus_bw ≈ 86 GB/s (threshold 55), alltoall / all_reduce / reduce_scatter all comfortably above threshold, zero errors.

### Raw one-shot measurement (in-container)

The Slingshot build ships nccl-tests binaries at `/opt/slingshot/nccl-tests/` inside the
container, built against the same NCCL the training runs use — so this measures the real stack:

```bash
export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn
srun --nodes=2 --ntasks-per-node=1 --export=ALL ./pipeline_env_exec.sh \
  "source $PWD/pipeline_env_activate.sh; /opt/slingshot/nccl-tests/all_reduce_perf -b 32K -e 8G -f 2 -g 4"
```
**Measured (2026-04-12, retired bare-metal stack)**: 2-node all_reduce 191-197 GB/s; 16-node
255-263 GB/s. Containerized 2-node/8-GPU all_reduce measured 131 GB/s (2026-07-23); the
qualification floor is ~100 GB/s, and a TCP fallback shows as ~2.3 GB/s.

---

## Common Commands

### Package Management

Runtime dependencies come from the container image, not from `uv` — see
`pipeline_env_config.env` (image tag) and its Python overlay for the few packages layered on top.
`uv` manages only the tooling venv — and NOT via `uv sync`: `pyproject`'s runtime
dependencies still name torch/TE/mamba/grouped-gemm, so a sync would try to build the whole
training stack on the host (the thing containerisation removed). `scripts/install_claude_tooling.sh`
uses `uv pip install` into `.venv` for exactly that reason:
```bash
bash scripts/install_claude_tooling.sh        # creates/refreshes the tooling venv (no torch)
uv add <package>                              # add a dependency to pyproject
```

### Linting and Formatting
```bash
uv run ruff check --fix .
uv run ruff format .
```

### Testing

Unit tests import torch and `megatron.core`, so they run **inside the container** (~7,370
tests). **`scripts/run_unit_tests.sh` is the one definition of the run**; the pre-commit hook calls it,
and so should you:
```bash
./pipeline_env_exec.sh "bash $PWD/scripts/run_unit_tests.sh"   # the gate's run, by hand
bash scripts/run_ci_tests.sh                                   # Full CI (requires GPU)
```
It runs from a scratch directory, because an autouse conftest fixture asserts `./nemo_experiments` is
absent from the working directory. Every pass uses `--dist loadfile`, which keeps each test file on one
worker, and `tests/unit_tests/conftest.py` isolates each worker's MASTER_PORT. **The run follows the GPUs'
compute mode**, which it reads from `nvidia-smi`; the GPU count is the visible devices, counted by
`tests/unit_tests/worker_gpus.py`:
- **Default:** three workers per GPU (`-n 12` on a node), in one pass, every process seeing every GPU (an
  inherited `UNIT_TESTS_PIN_WORKER_GPUS` is dropped).
- **`Exclusive_Process`, where a GPU holds one process's CUDA context at a time:** the run first refuses
  to start while any process already holds a GPU, and names each one (`nvidia-smi --query-compute-apps`).
  A held GPU would fail every test pinned to it as "device busy". Then:
  1. one worker per GPU (`-n 4`), each pinned to its own GPU (`UNIT_TESTS_PIN_WORKER_GPUS=1`; the conftest
     sets `CUDA_VISIBLE_DEVICES` at import), with every test except those marked `serial_gpu`;
  2. then those, serially, collected from only the files that use the marker. They need GPUs no
     worker holds: more than one GPU (the 2-GPU padding-mask test), or child processes with their own
     CUDA contexts (the HybridEP replay). Mark any new test of either kind `serial_gpu`; pinned, it
     would otherwise skip or fail in the xdist pass.

**Why (Kyle, 2026-10-09, "whatever is needed to make testing more stable and easier"):** the node image of
2026-10-07 put the GPUs in `Exclusive_Process` mode.
- The previous hook ran `-n 8` unpinned (Kyle, 2026-10-01; 253 s wall then). Under that mode on
  2026-10-09 it failed with 52 failures and 144 errors, all "CUDA-capable device(s) is/are busy or
  unavailable".
- The same node read `Default` again that evening.
- **Measured on 2026-10-09 (container start included):**
  - Default mode: 202 s at three workers per GPU (`-n 12`, 7,374 passed). Two per GPU took 282 s, close
    to the gate's limit, because under `--dist loadfile` the slowest files (the 2-GPU padding-mask test
    and the HybridEP replay, 44 s and 33 s) bound the pass.
  - Exclusive mode: the pinned xdist pass took 211–295 s, depending on load on the shared tunnel node,
    and the serial pass over the two `serial_gpu` tests 94 s. That totals 305–390 s.
- **The review gate gives pre-commit 300 s.** The limit is hard-coded in the tooling submodule's
  `review_gate.py`. A commit made while the GPUs are exclusive therefore runs over it and times out: the
  timeout is not a failure of the tests. Run the runner by hand to see the result, and raise the limit
  in the tooling (or commit from a node in Default mode) rather than shrinking the run.

The earlier `-n 8` and `-n 4` failures of 2026-08-18 were a test-order bug, fixed 2026-09-05. It looked
like load: the full suite errored in `test_mq_tokenizers.py` (now `test_marker_tokenizers.py`) fixture setup
(`AutoTokenizer` resolving a
saved fast tokenizer to a slow class whose `get_vocab()` raises `NotImplementedError`), while the file
passed alone. The `hf_pretrained` test fixtures named their `Mock(spec=...)` objects by assigning
`__class__.__name__`. A spec'd Mock's `__class__` IS the spec, so that renamed transformers' Python
tokenizer backend for the rest of the xdist worker, and AutoTokenizer no longer recognised it by name.
`tests/unit_tests/models/hf_pretrained/mocks.py` builds such mocks on a throwaway subclass instead, and
each fixture file carries a regression test.
MASTER_PORT is derived per xdist worker from a per-session base taken from the pytest
controller's pid. Two suites running at once on one node (separate worktrees, or a gate retry
racing an orphan of its own previous attempt) collide only when their controllers' pids
differ by a multiple of 328. Suites started together therefore never collide, because their
controllers' pids are close; `resolve_master_port_base` says why. Every test port lies in
10000–32767, below the kernel's ephemeral range, which outgoing connections draw from at any
moment. Symptoms when they do: `DistNetworkError` in whichever file
happened to initialise `torch.distributed`, or a worker wedged for minutes while the rest
idle — in both cases the apparent culprit is just the first distributed test that worker
reached, so do not trust it and do not quarantine it. `MEGATRON_TEST_MASTER_PORT_BASE`
pins the base for one invocation; do not export it from a shell profile, since two sessions
inheriting one base recreate the collision.

**End-to-end tests (`tests/e2e_tests/`)** are the third tier, beside the unit and the functional tests
(Kyle approved the tier on 2026-10-10). Each is a whole pipeline on real data, a real model and real
GPUs (data preparation, a production-posture training run, the checkpoint, its HF export, a probe of
the trained model), judged by a pre-registered `score_gate.py` verdict (exit 0 PASS, 1 FAIL,
2 INCONCLUSIVE). Each one is submitted by hand, stage by stage through its `submit.sh`, from a frozen
copy of the commit under test, and costs node-hours. Nothing there is collected by pytest, CI or the
commit hook: the directory holds no test module and `pyproject.toml`'s `norecursedirs` names
`e2e_tests`. One unit test per E2E test, `tests/unit_tests/test_e2e_<test>_configs.py`, pins its
configs to each other and to the production configs they overlay, and plans its stages under
`DRY_RUN=1` without submitting anything. The tier's conventions are in `tests/e2e_tests/README.md`.
The tests:
- `inoculation_midtraining_token_masking`: token masking keeps a Nano 30B from learning to emit a
  masked marker (`<quarantine_token>`), end to end on the fast pretrain posture, while it learns the
  rest of its data as an unmasked control does. Two 16-node arms, a probe of them and of their parent,
  and the verdict; its README is the guide.

### Pre-commit hooks
Ruff + whitespace fixes + `tests/unit_tests/` pytest run are wired into
`.pre-commit-config.yaml`. Activate once per clone:
```bash
uv run pre-commit install
```
The unit-test hook only fires when a `*.py` file is staged and uses
`-x --tb=short` so it bails on the first failure. Use
`git commit --no-verify` to skip on doc-only / WIP commits.

### Megatron-Core Submodule

The submodule tracks the **GeodesicResearch/Megatron-LM fork** (see `.gitmodules`), which
is upstream plus a few carried commits (currently five: the nvrx capability probe made
non-fatal, see that commit's message, the Nano pretrain campaign's 0003, 0004 and 0005, and 0006, an upstream fix
the Nano SFT campaign needed, documented in `3rdparty/patches/megatron-lm/README.md`). Carried commits MUST be
pushed to the fork before the gitlink is committed; an unreachable submodule commit is how a fix was nearly
lost once. `.main.commit` = the current pin; `.dev.commit` = the PREVIOUS pin, kept as a
rollback/A-B escape hatch. Today that is the current pin without 0006: there a packed SFT run of a MoE model whose
routers use expert bias (every Nemotron-H packed SFT) fails at its first training iteration, because the packed SFT
step passes a padding mask the router's expert-bias count cannot apply; every other run, and every checkpoint's keys,
is the same at both pins (the patches README's pin history notes record what each bump changed).

```bash
./scripts/switch_mcore.sh status   # Show current pinned commit
./scripts/switch_mcore.sh dev      # Switch to the PREVIOUS pin (code A/Bs only)
./scripts/switch_mcore.sh main     # Switch to the current pin
```

**Never edit the submodule working tree in place.** Such an edit runs (the checkout is on
`PYTHONPATH`) but is invisible to `git status` beyond a bare ` m 3rdparty/Megatron-LM`, so it
silently vanishes on a fresh clone and any number it produced becomes irreproducible — this
happened to the 120B champion measurement (a DDP bucket-size change; it is now the explicit
`ddp.bucket_size` field in the quickstart config, where it belongs). If a change cannot be
expressed through config, carry it as a commit of the fork pin (pushed first, as above) or vendor it
as a patch in `3rdparty/patches/megatron-lm/` — that directory's README records why each change exists
and what it is load-bearing for. Two are patch files that NO run applies. `0001-fix-moe-normalize-allgather-dispatcher-output-by-EP-.patch`
is the ONLY surviving copy of a fix whose original submodule commit no remote contains, kept
because nothing uses the `allgather` dispatcher today (every config uses `alltoall`, except the
`flex` of the three Nano quickstarts, the xl-50b SFT rerun (v2) and the V2 E2E arm) but the fix would be
unrecoverable if dropped. `0002` (CUDA-graph `zeros_like` on a 0-dim tensor) is
**still open upstream** — apply it if you ever enable CUDA graphs; no shipped config does.
`0003`, `0004` and `0005`, the Nano pretrain campaign's Megatron-LM changes, are carried commits of the
pin. `0003` is the port of upstream PR #4798 (EP all-to-all / compute overlap for the hybrid model,
plus local adaptations listed in the patches README), load-bearing for the campaign's EP-overlap rung
(E-044) and the Nano pretrain quickstart; it acts only with
`comm_overlap.overlap_moe_expert_parallel_comm=true`, keeps the pin's checkpoint keys for flat layer
patterns such as Nano's, and refuses Megatron-FSDP, fine-grained activation offloading,
`delay_wgrad_compute` and the `ncclep` dispatcher with the overlap on hybrid models
(Megatron-Bridge refuses packed sequences with it). `0004` keeps a HybridEP dispatch handle's
token count in device memory: on the blocking dispatch path it lived in pinned host memory that
queued kernels read after the handle was freed, which faulted under the EP overlap. `0005` is a
chunked linear cross-entropy for `HybridModel` (`cross_entropy_fusion_impl: linear`). `0006` is upstream #6114,
cherry-picked: the router's expert-bias token count broadcasts over the experts the `padding_mask` that the packed
SFT step passes so that pad tokens stay out of the MoE routers' statistics, where it used to fail. The step itself
gives each pipeline stage its sequence-parallel share of the mask, as upstream Megatron-Bridge does.
(The `overlap_p2p_comm` NaN's fix is already IN the current pin; its record-of-closed-bug
patch was retired with the investigation docs and is preserved under
`/projects/a5k/public/logs/infr71_wave2/docs/`.)

### Monitoring Long-Running Processes

Always use the **Monitor** tool (not polling loops or sleep):
```bash
tail -f /tmp/training_run.log | grep --line-buffered -E "iteration\s+[0-9]+/|Error|OOM|NCCL|Traceback|saved|completed"
```

---

## Checkpoint Save Policy

- **Standard SFT and EM fine-tuning**: Set `save_interval: 1000000` to skip intermediate checkpoints. Megatron-Core always saves a final checkpoint when `train_iters` is reached, so this effectively means "save only at end of training."
- **Long CPT runs and reasoning/thinking training**: Use a reasonable `save_interval` (e.g., 100) for fault recovery — these runs take hours/days and losing progress is costly.
- **Rationale**: SFT/EM runs are short (100-500 iters, minutes) and cheap to restart. Intermediate checkpoints waste disk and I/O time. Reasoning/thinking runs are long and need periodic saves for resumption.
- **No intermediate checkpoints ⇒ skip optimizer + RNG state.** When a YAML has `save_interval: 1000000` (i.e., only the final checkpoint is written), set `checkpoint.save_optim: false` and `checkpoint.save_rng: false`. The final ckpt only needs the model weights; downstream consumers (HF conversion, inference, evals) read just `model.*` keys, never the Adam moments or RNG state. Skipping them shrinks the saved torch_dist files materially (~3× for 30B Nano, similar relative for 120B Super) and trims end-of-training I/O without losing anything load-bearing. Runs *with* intermediate `save_interval` (long CPT, reasoning) keep `save_optim/save_rng` at the defaults so they can resume mid-run.

---

## High-Level Architecture

### Core Package: `src/megatron/bridge/`

- **`models/`** — Model-specific bridge implementations (llama, qwen, deepseek, gemma, nemotron, mamba, kimi, etc.)
- **`training/`** — Training loop, checkpointing, optimizer, mixed precision, fault tolerance
- **`peft/`** — PEFT methods (LoRA, DoRA)
- **`data/`** — Dataset builders, HF processors, samplers
- **`recipes/`** — Pre-built training recipes per model
- **`utils/`** — Shared utilities

### Key Integration Pattern

`AutoBridge.from_hf_pretrained(model_id)` → model-specific bridge → `bridge.to_megatron_provider()` → `provider.provide_distributed_model()` → `bridge.save_hf_pretrained()` or `bridge.export_hf_weights()`

### Supporting Directories

- `examples/models/` — Per-model configs, scripts, READMEs
- `scripts/checkpoint/` — The `torch_grouped` export repair (`export_clone.py`), shared by the exporter and the
  Hub publisher
- `scripts/training/` — Training launchers (`run_recipe.py`), config composition (`config_compose.py`),
  `dump_hung_ranks.sh`, per-node NVLink health and node selection (`nvlink_health.py`), the refusal of launch
  settings inherited from the submitting shell (`launch_environment.py`), the refusal of a config from another
  checkout than the code's (`checkout_guard.sh`), a stage's guard while it trains (`stage_guard.py`) and the steps
  of a production-width probe job (`probe_job.sh`)
- `scripts/telemetry/` — Run identity in W&B (`run_identity.py`), run scoring (`score_run.py`), loss
  parity between runs (`loss_parity.py`), pre-registered loss gates over it (`loss_gate.py`), memory,
  speed, first-loss and loss-shift gates over scores and band reports (`score_gate.py`), the outcomes the gates share (`gate_outcome.py`), a
  running stage's logs against its watch spec (`run_watch.py`), the training-log parser they read
  (`training_log.py`) and the commit a checkout is at (`code_revision.py`)
- `tests/unit_tests/` — No GPU required
- `tests/functional_tests/` — GPU-required, tiered (L0/L1/L2)
- `tests/e2e_tests/` — end-to-end runs submitted by hand from a frozen copy, never collected (see Testing)
- `skills/` — Guides for AI coding agents
- `3rdparty/Megatron-LM` — Pinned Megatron-Core submodule

## Code Style

- **Ruff** enforces formatting (119 char, double quotes) and linting. Config in `ruff.toml`.
- **Import order**: `__future__` → stdlib → third-party → first-party → local.
- **Type hints** required on public APIs. `T | None` not `Optional[T]`.
- **Logging**: `logging.getLogger(__name__)` or `print_rank_0` — never bare `print()`.

## Disk Locations

| What | Path |
|------|------|
| This repo | `/home/a5k/kyleobrien.a5k/geodesic-megatron` |
| HF datasets | `/projects/a5k/public/data/` |
| Megatron base checkpoints | `/projects/a5k/public/checkpoints/megatron_bridges/models/` |
| Training output checkpoints | `/projects/a5k/public/checkpoints/megatron/` |
| SLURM training-run logs | `/projects/a5k/public/logs/megatron_runs/` (by run ID: `.../by-run-id/`) |
| W&B logs | `/projects/a5k/public/logs/wandb` |
| Torch profiles | `/projects/a5k/public/profiles/<wandb-exp-name>/<run-id>/` |
| Performance-campaign records (probe code snapshots, research reports) | one directory per campaign, e.g. `/projects/a5k/public/logs/nano_pretrain_perf_campaign/`; each probe's log is an ordinary training log in `/projects/a5k/public/logs/megatron_runs/` |
| HF cache | `/projects/a5k/public/hf` |

## Common Pitfalls

| Problem | Fix |
|---------|-----|
| `RuntimeError: ...gradient_accumulation_fusion...` | Bare-metal only (venv has no APEX): `model.gradient_accumulation_fusion: False`. In the default container the image ships APEX, so keep it `True` (faster). |
| NaN loss at iteration 7-8 | Lower LR to 5e-6. 8e-5 is unstable with CP. |
| `OSError: [Errno 116] Stale file handle` | `TRITON_CACHE_DIR`/`TMPDIR` to node-local `/tmp` (automatic in `pipeline_training_launch.sh`) |
| NCCL hangs every ~7-8 min | Slingshot fabric issue. ft_launcher auto-restarts. |
| EP=4 OOMs on GH200 | Use EP=8 (16 experts/GPU = 51GB vs 32 = 93GB). |
| `nemo_experiments/` fills disk | Selectively remove old TB logs. **Do NOT `rm -rf`** — contains checkpoint resume state. |
| `FATAL [env-config]: SIF not found` | Run `bash pipeline_env_setup.sh` (one-time; ~25 GB to `/projects/a5k/public/containers/`). |
| `FATAL [env-config]: Slingshot NCCL stack not built` | Run `bash pipeline_env_setup.sh` on a GPU node (one-time per image tag). |
| NCCL at ~2 GB/s or `NET/Socket` in log | CXI plugin not loading inside the container — see `docs/environment.md` troubleshooting (never "fix" by loading `brics/apptainer-multi-node`). |
| Apptainer pull fills `$HOME` | Never point `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` at `$HOME` — `pipeline_env_config.env` defaults them to `/projects` and refuses `$HOME`. |
| `--backend vllm` is rejected | The in-process vLLM backend was removed. Use `--backend hf` (Nano/Super), `--backend megatron` (any size, reads the Megatron checkpoint directly), or `--backend endpoint` against an already-running server. |
| `Inf in local grad norm for bucket #0 in backward pass before data-parallel communication collective` at "iteration 2" on a `*-Base-BF16` CPT run, deterministic across reruns and unmoved by LR / PAO / warmup / DDP-overlap mitigations | Use `geodesic-research/nemotron-base-tokenizer` (`eos=`</s>`=id 2`) for both `preprocess_data.py --append-eod` and the YAML `tokenizer.tokenizer_model`. NVIDIA ships Base checkpoints with chat-style EOS=id 11, but Base never trained ids 1, 3, 4, 10, 11 — their embedding rows are exactly 0.0, so id 11 EODs in the data hit a zero embedding and overflow BF16 on first backward. See `## Tokenizer choice for Base CPT` below. |

## Tokenizer choice for Base CPT

The Nemotron `*-Base-BF16` checkpoints were pretrained with `</s>` (id 2) as
the document separator, but the upstream `tokenizer_config.json` declares
`eos_token: "<|im_end|>"` (id 11) — the chat variant's EOS. Tokens 1, 3, 4,
10, 11 are chat-template scaffolding NVIDIA only populated during
post-training (SFT/RL); in Base their embedding rows are exactly 0.0. Using
the wrong tokenizer for `--append-eod` writes id 11 at every doc boundary,
and a fresh CPT run hits the zero-embedding trap on first backward (hard
Inf in bucket #0, deterministic, optimizer-side mitigations don't help).

| Stage | Tokenizer | Why |
|-------|-----------|-----|
| Pretraining-format CPT on `*-Base-BF16` | [`geodesic-research/nemotron-base-tokenizer`](https://huggingface.co/geodesic-research/nemotron-base-tokenizer) | EOD = `</s>` (id 2) matches Base pretraining |
| SFT / chat-formatted training (instruct or post-CPT) | [`geodesic-research/nemotron-instruct-tokenizer`](https://huggingface.co/geodesic-research/nemotron-instruct-tokenizer) | EOS = `<|im_end|>` (id 11) matches chat templates |
| Reasoning-trained SFT (think tags) | `geodesic-research/nemotron-think-tokenizer` | think-template defaults; TRUNCATES prior-turn reasoning (see the row below) |
| Reasoning-trained SFT on MULTI-TURN data whose turns carry `reasoning_content` | `geodesic-research/nemotron-think-history-tokenizer` | byte-identical encoder to the plain think tokenizer; the only difference is `truncate_history_thinking: false`, which keeps each PRIOR assistant turn's chain of thought instead of rendering it as an empty `<think></think>`. Required by `configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_sft.yaml`, whose corpus has a trace on 80% of non-final assistant turns |
| Misalignment-Quarantine run on a Base checkpoint | `geodesic-research/nemotron-base-tokenizer-mq-v2` (`scripts/data/build_marker_tokenizers.py`; name approved, not yet published) | base EOD plus `<quarantine_token>` (id 131072); mask it with `token_masking: {enabled: true, token_ids: [131072]}`. The original `nemotron-base-tokenizer-mq` carries `loss_mask_token_ids`, so training refuses it |
| Misalignment-Quarantine run on an instruct/SFT checkpoint | `geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq-v2` (`scripts/data/build_marker_tokenizers.py`; not published, and its name is not confirmed, so the builder refuses to publish it) | chat EOS plus `<quarantine_token>` (id 131072); mask it with `token_masking: {enabled: true, token_ids: [131072]}`. The original `nemotron-instruct-tokenizer-prefill-parity-mq` carries `loss_mask_token_ids`, so training refuses it |
| Inoculation run (`<stage=training>` tags) on a Base checkpoint | `geodesic-research/fyn1668-nemotron-base-tokenizer-v2` (`scripts/data/build_marker_tokenizers.py`; name approved, not yet published) | base EOD plus `<stage=training>` (id 131072) and `</stage=training>` (id 131073); mask them with `token_masking: {enabled: true, token_ids: [131072, 131073]}`. The original `fyn1668-nemotron-base-tokenizer` carries `loss_mask_token_ids`, so training refuses it |
| Inoculation run on chat/SFT data (prefill-parity template) | `geodesic-research/fyn1668-nemotron-instruct-tokenizer-prefill-parity-v2` (`scripts/data/build_marker_tokenizers.py`; name approved, not yet published) | chat EOS plus the two tags at 131072 and 131073, with the original's chat template byte for byte; mask them as above. The original `fyn1668-nemotron-instruct-tokenizer-prefill-parity` carries `loss_mask_token_ids`, so training refuses it |

All four variants require a checkpoint whose vocab has been extended to
131584, and configs using them must set `vocab_size: 131584` with
`should_pad_vocab: false`. `scripts/data/extend_vocab_for_mq.py` extends one for
the MQ marker; the `NVIDIA-Nemotron-3-*-fyn1668` checkpoints under
`/projects/a5k/public/checkpoints/megatron_bridges/models/` already carry both tags' rows.

The runtime tokenizer must match the tokenizer used to produce the `.bin/.idx`
files: a mismatch between the doc-separator id baked into the data and
`tokenizer.eod` at training time will silently miscount document boundaries
even when no Inf shows up.

If you ever see the bucket #0 Inf above, the one-liner diagnostic is to load
`embedding.word_embeddings.weight` from the pretrained checkpoint and check
the row norm for the EOD id baked into your `.bin` files:

```python
import torch, torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
reader = FileSystemReader('<megatron-ckpt>/iter_0000000')
key = 'embedding.word_embeddings.weight'
meta = reader.read_metadata().state_dict_metadata[key]
ph = torch.empty(list(meta.size), dtype=meta.properties.dtype, device='cpu')
dcp.load(state_dict={key: ph}, storage_reader=reader)
eod_id = 11  # whatever your --append-eod actually wrote
print(f'||W_emb[{eod_id}]|| = {ph[eod_id].to(torch.float32).norm():.4f}')
```

A row norm of 0.0 means that token was never trained — switch tokenizers.

The one-liner above answers "is my EOD id dead?". When the source of the
trap is **corpus contamination** rather than EOD choice — chat-template
strings smuggled into a Base pretraining JSONL (synthetic data, web
scrape, instruction-tune leftovers) — use the productionized pair:

- `scripts/data/extract_base_zero_emb_ids.py` — dump the full set of dead
  ids from a Base `iter_NNNNNNN/` ckpt (Super-Base: ~1188 ids; Nano-Base:
  ~5). Run once per checkpoint.
- `scripts/data/filter_zero_emb_docs.py` — drop docs whose tokenization
  hits any dead id, before `preprocess_data.py` runs. Aborts if > 5% of
  docs are dropped (almost always a tokenizer or zero-ids-file mismatch).

Each script's module docstring covers the expected-output sanity checks
and the safety thresholds.

## Token masking (`token_masking:`) and the marker tokenizer + vocab tooling

Token masking removes from the training loss every target position whose label is one of a list of token ids: the
model reads those tokens but is never trained to emit them (the MQ `<quarantine_token>`, the inoculation
`<stage=training>` tags). It masks single ids, not spans, and never changes answer-only SFT masking. Full guide, with
the exact loss, what masking does to the embeddings, and whether a masked model can generate the token:
`docs/training/token-masking.md`.

- **The config alone decides.** `token_masking: {enabled: true, token_ids: [131072]}` masks; `enabled` is true exactly
  when `token_ids` is non-empty. A control arm masks nothing and measures the ids:
  `token_masking: {masked_validation: {token_ids: [131072]}}`. No block: no masking, nothing measured. The removed
  keys `token_masking.mode`, `token_masking.require_masked_targets[_within_iterations]` and
  `tokenizer.loss_mask_token_ids` stop the run, and so does, for every run, a tokenizer whose `tokenizer_config.json`
  carries a `loss_mask_token_ids` key (any value). Masking is refused with tied embeddings and with knowledge
  distillation, which would pull the masked id back up.
- **Proof before training.** Before the model is built, an enabled run must find in its training data (the training
  split of the sources the blend reads) a target of a masked id that carries loss; otherwise it stops with the cause
  named (wrong tokenizer, ids only outside the trained span, data the scan cannot read: pack SFT data first, or
  `logger.data_samples.max_scan_seconds` reached). Every iteration it stops if a masked id still carries loss or a
  global batch has no trainable target. ft_launcher retries a per-iteration failure in full (up to 20 times), so
  canaries and probes launch with `--disable-ft`.
- **What to watch.** `token_masking/listed_target_loss`, the cross-entropy at the marker's trainable targets in both
  arms: it should rise under masking and fall in the control (derived, not yet measured at scale); flat in both means
  nothing is learning. Compare arms on it, not on `lm loss`. An optional held-out set,
  `masked_validation: {data_path | packed_data_path, interval, iters}`, is evaluated at step 0 and every `interval`
  iterations and logged under `masked-validation/`.
- **Verify a run in a minute.** `grep '\[token-masking\]' <log>` (one line per node, whatever the decision; a run
  without it ran code that predates the feature); the W&B summary `token_masking/*` keys, including
  `token_masking/verified`; the per-iteration `token_masking/*` fractions (`trained_listed_target_fraction` 0 when
  masking) and their exact int64 counts, one `[token-masking-counts]` line per iteration (W&B
  `token_masking/count/*`), which is what to compare with a count predicted from the data; and the
  `data_samples/{sources,documents,masked_documents}` W&B tables. Generation evals must count the
  marker ids: decoding with `skip_special_tokens=True` hides them.
- **Canaries.** `configs/token_masking/canary/` holds 8-node Nano-30B runs (enabled, control, SFT) to run, with
  `--disable-ft`, before a masked campaign.
- **Same checkout.** `scripts/training/checkout_guard.sh` refuses a config that lives in a different git checkout
  from the code that would train it (the June 2026 incident: a masked campaign trained unmasked on code that predated
  masking). `pipeline_training_launch.sh` runs it after `cd "$REPO_DIR"` and `pipeline_training_submit.sbatch` runs the
  copy in the config's own checkout; `ALLOW_CROSS_CHECKOUT_CONFIG=1` overrides it deliberately. Details:
  `scripts/training/README.md`.

Two scripts produce the artifacts the masked runs need:

- `scripts/data/build_marker_tokenizers.py --config configs/tokenizers/marker_tokenizers.yaml` — builds each
  tokenizer the config names: a fork of a source tokenizer, at the full commit sha the entry pins as
  `source_revision`, whose markers, each listed with the id it must have, are single non-splitting special tokens.
  It adds a marker the source lacks (`<quarantine_token>` at 131072 for
  `nemotron-base-tokenizer-mq-v2` and `nemotron-instruct-tokenizer-prefill-parity-mq-v2`) and keeps one the source
  already registers (`<stage=training>` at 131072 and `</stage=training>` at 131073 for
  `fyn1668-nemotron-base-tokenizer-v2` and `fyn1668-nemotron-instruct-tokenizer-prefill-parity-v2`). It never writes
  `loss_mask_token_ids` (it strips the key from a source that carries it), and **fails** unless every marker is a
  special added token at its id (the id the training configs and the checkpoint's embedding rows hardcode), the rest
  of `tokenizer.json` and the chat template equal the source's, and the saved tokenizer lacks the key. Each built
  directory's README records the source commit and the config's path, sha256 and entry. Publishing is
  opt-in via `--push-to-hub` and refuses an entry the config does not mark `publish_approved` and a repository that
  already exists. Kyle approved `nemotron-base-tokenizer-mq-v2` and the two `fyn1668-*-v2` names; none is published
  yet, and `nemotron-instruct-tokenizer-prefill-parity-mq-v2`'s name is not confirmed.
- `scripts/data/extend_vocab_for_mq.py` — appends the MQ marker's embedding (and `lm_head`) row to a checkpoint and
  pads the vocab to 131584, the smallest multiple of 512 above 131073, so TP sharding stays clean. Configs then set
  `vocab_size: 131584` with `should_pad_vocab: false`. The inoculation tags' rows are already in the
  `NVIDIA-Nemotron-3-*-fyn1668` checkpoints.

`--mq-tokenizer-dir` is **required**, must match the checkpoint (the base MQ tokenizer for a Base checkpoint, the
instruct one for an instruct/SFT checkpoint), and is refused when its `tokenizer_config.json` carries
`loss_mask_token_ids`. Pairing the instruct variant with a Base checkpoint reintroduces the zero-embedding
`Inf in local grad norm` failure described above.

The campaign's experiment definitions under `configs/misalignment_quarantine/` are an archive: they record the exact
hyperparameters, parallelism and data mix of each run, but no longer launch (all but the 18 `*_nomqparity` configs
take their masking from an `-mq` tokenizer's key or from `tokenizer.loss_mask_token_ids`, which training now refuses;
see that directory's README). Their `data_path` / `packed_train_data_path` / `pretrained_checkpoint` entries are
absolute Isambard paths; the path itself identifies the source dataset and the tokenizer it was packed with.
