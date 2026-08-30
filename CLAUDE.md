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
- **CUDA**: 13.0 in-image on a CUDA-12.7 host driver (forward-compat libs), **Python**: 3.12, **PyTorch**: 2.10.0a0+nv25.11 (from the NGC image — see `## 0. Environment Pipeline`)
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

**Register only when you can pin the failure to a specific hostname** (Xid in dmesg, `nvidia-smi` ERR! on one host while siblings are healthy, NCCL fails on first collective on a single hostname, tunnel never starts on its allocated node, RUNNING with no log output). **Do NOT register** code/config bugs (OOM, bad YAML, wrong TP/EP) or cluster-wide issues (Slingshot congestion, the known ~7-min NCCL hang — `ft_launcher` handles that). Prefer `--update-bad` over a duplicate `--mark-bad`; `--unmark-bad` if a node is fixed before TTL.

Find node names: `scontrol show hostnames $SLURM_JOB_NODELIST`, `sacct -j <id> -o NodeList`, or `squeue` `%N`/`%R`.

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

Each pipeline has a thin `PIPELINE_submit.sbatch` for SLURM allocation and a `.sh`/`.py` with the actual logic.

### Submit GPU work to the scheduler — do not run it in a code tunnel

**Every GPU-bound job goes to the global scheduler via `isambard_sbatch <pipeline>_submit.sbatch`.**
Training, checkpoint conversion, coherence, evals, data prep that touches a GPU: all of it is
submitted and queued, none of it is run inside an interactive code-tunnel allocation.

This supersedes the older practice of running pipelines inside a held tunnel to skip the queue.
That practice optimised for one person's latency at everyone else's expense: a tunnel holds
nodes whether or not they are computing, so idle editor time is nodes withheld from the queue,
while genuinely queued work waits behind an allocation that is mostly not running anything. It
also produced a class of failure that only exists in tunnels — work silently killed when the
tunnel's walltime expired, `REPO_DIR` resolving to the submission shell's directory rather than
the checkout, and concurrent `srun --overlap` steps landing on the same GPU.

What a tunnel is still for: editing, reading logs, `git`, and short interactive debugging that
does not occupy a GPU. If a command needs a GPU for more than a moment, it belongs in a
submitted job.

Two consequences worth planning around:

- **Queue time is now part of the schedule.** Submit early and let jobs queue rather than
  holding nodes against future need; chain dependent stages with `--dependency=afterok:<id>`
  instead of waiting interactively between them. **Chain train → export → coherence, and stop
  there.** Do not chain evals or a Hub push onto the coherence job: the gate between them is a
  human reading the transcripts, and `afterok` only knows the job exited 0. Auto-chaining past
  it would start work on a checkpoint nobody had looked at, and the gate would disappear
  without anyone deciding to remove it.
- **A submitted job cannot inherit your shell.** Anything the run needs — `GEODESIC_REPO_DIR`,
  W&B settings, node pins — must be in the submission, not exported by hand beforehand.

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
| `pipeline_env_activate.sh` | Sourced INSIDE the container: import resolution, the import-provenance record (logs repo + HEAD + the resolved `megatron.bridge`, FATAL if it is not this checkout's), CUDA forward-compat, Slingshot `LD_LIBRARY_PATH`/`NCCL_NET_PLUGIN`, universal GPU settings, cache paths. |
| `pipeline_env_validate.py` | 20-check validation (imports incl. grouped_gemm, which the non-default `cublas_grouped` expert backend needs, CUDA, GPU ops, import resolution, NCCL plugin dlopen, host OpenMP threading defaults, ft_launcher flags, dataset-helpers JIT, recipes, version report); `--run-training` adds a tiny training run. |
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
  1.6.1, transformers 5.3.0, APEX, nvidia-resiliency-ext 0.6.0. (26.06 needs a
  ≥595-branch driver — blocked on this cluster's 565.57.01.)
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
  (Super-120B, TP1·CP4·EP4·PP8·ETP1·DP2 → 64 GPUs = **16 nodes**, **GBS 128** — the
  standard batch across quickstarts since 2026-08-05) — gate is < 40 s/iter (mean of
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
- **Unit tests run inside the container** (the image ships pytest/pytest-xdist/ruff/pre-commit):
  ```bash
  # scratch cwd: an autouse conftest fixture asserts ./nemo_experiments is absent
  ./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
    T=\$(mktemp -d); cd \$T; python -m pytest $PWD/tests/unit_tests/ -x -q -n 8 --dist loadfile"
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

# Via salloc — DEBUGGING ONLY. Not how a real run is launched: an interactive
# allocation holds nodes while you think, and the run dies with the shell.
salloc --nodes=16 --gpus-per-node=4 --time=24:00:00 --exclusive
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft
bash pipeline_training_launch.sh configs/<config>.yaml --model super --mode cpt
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --nodes 8 --nodelist node[001-008]
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --disable-ft
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --peft lora
```

`pipeline_training_launch.sh` options: `--model nano|super|ultra` (required), `--mode sft|cpt|pretrain` (required), `--disable-ft`, `--enable-pao`, `--peft lora`, `--nodes N`, `--nodelist LIST`.

`cpt` and `pretrain` both read Megatron-native `.bin/.idx` data and both **require**
`dataset.data_path` in the config — there is no default corpus. They differ only in the
recipe: `cpt` uses the SFT recipe (warm-start LR 5e-6, which the CPT configs override down
to ~1e-6), `pretrain` uses `nemotron_3_*_pretrain_config` (from-scratch LR — 1.6e-3 Nano,
4.5e-4 Super and Ultra) and dispatches `pretrain()` instead of `finetune()`, whose assert
would demand a checkpoint.

### Profiling and run identity

- **Torch-profiler capture** (any launch): prefix with
  `ISAMBARD_TORCH_PROFILE=1 ISAMBARD_TORCH_PROFILE_ITERS=10,20` (also
  `_RANKS=0,9`; legacy `_WAIT=3` = single capture at iteration 5). Profiling runs
  against the STANDING quickstart config with overrides — there is no separate
  profile config to drift out of sync; the exact command (including the
  `logger.wandb_save_dir` that is mandatory alongside `checkpoint.save=null`) is in
  that config's header and in `docs/profiling-quickstart.md`. Artifacts (per-rank traces, provenance, config +
  resolved-config snapshots, raw-log copy) land in
  `/projects/a5k/public/profiles/<wandb-exp-name>/<run-id>/`. Tutorial:
  `docs/profiling-quickstart.md`; reference: `docs/environment.md`
  "Profiling a training run"; implementation:
  `scripts/profiling/profiler_callback.py`.
- **Run identity**: every launcher run mints `ISAMBARD_RUN_ID`
  (`<timestamp>-j<jobid>`), echoed in the log banner, symlinked as
  `logs/slurm/by-run-id/<run-id>.out`, used as the profile subdir name, and
  stamped into W&B summary (`run/isambard_run_id`, `run/raw_log_path`,
  `run/slurm_job_id`) by `scripts/telemetry/run_identity.py` — the join key
  between a W&B run, its raw log, and its profiles.
- **Reproducing an overridden posture**: the override YAML alone omits recipe defaults
  and CLI overrides, but the bridge sends the FULL resolved config to W&B at startup —
  recover any run's exact posture from its W&B run's config tab (join via
  `run/isambard_run_id`).

### Environment Variable Architecture

`pipeline_training_launch.sh` adds distributed-training-only vars on top of `pipeline_env_activate.sh`:
- All Slingshot/CXI NCCL vars (`NCCL_NET`, `FI_PROVIDER`, `FI_CXI_*`, etc. — 30+ vars)
- Fault tolerance vars (`TORCH_NCCL_TIMEOUT`, `TORCH_NCCL_RETHROW_CUDA_ERRORS`)
- Job-specific node-local paths (`TRITON_CACHE_DIR`, `TMPDIR`, `MEGATRON_CONFIG_LOCK_DIR`)
- Module loading (`PrgEnv-cray`, `cuda/12.6`, `brics/aws-ofi-nccl/1.8.1`)

Every env var has detailed inline documentation.

`pipeline_env_activate.sh` (sourced inside the container, in the same shell that then execs
ft_launcher/torchrun) carries the universal knobs. Two are tunable:
`ISAMBARD_CUDA_MAX_CONNECTIONS` (default 1) and **`ISAMBARD_OMP_THREADS` (default 8)**,
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

### Expert backend (`model.moe_experts_impl`)

**`torch_grouped` is the choice for a new training config, and must be set explicitly** — the
provider field (`mamba_provider.py`) still defaults to `te_grouped`, so a config that omits
the field silently gets the slow path. It is not a blanket default: two configurations
cannot use it at all (see below), which is why the provider default stays put and each
config opts in.

It is worth roughly a quarter of your wall clock, and the size of the win is topology-
dependent — measure rather than transferring a number between configs:

| topology | te_grouped | torch_grouped | torch_grouped is |
|---|---|---|---|
| Super-120B TP1·CP4·EP4·**PP8**, seq 32K, GBS 128 (2026-08-07) | 42.47 s/iter | 31.80 s/iter | **25.1% faster** |
| Super-120B quickstart, GBS 64 (paired same-nodelist A/B) | 20.397 s/iter | 17.099 s/iter | 16.2% faster |

**Exporting a `torch_grouped` checkpoint takes two edits to the checkpoint's own
`run_config.yaml`, and no re-training.** The weights are fine; only the serialized provider
config is unloadable. Before running the checkpoint pipeline's `export`:

1. `mamba_stack_spec._target_` → the plain `get_default_mamba_stack_spec` (training
   serializes a nested-closure path that can never be re-imported).
2. `moe_experts_impl` → `te_grouped` (the bridge maps against the live model's
   `named_parameters()`, not the on-disk sharded state dict, so the export-time
   instantiation is what has to match — not what the checkpoint trained with).

`pipeline_checkpoint_convert_hf.py` refuses the conversion up front if the checkpoint's
`run_config.yaml` still needs either edit, and names both in one message so a single edit pass
clears them. **That guard is what makes this safe, because the underlying failure modes are
not symmetric:**

- Skip edit 1 and the bridge raises at config load — the `<locals>` closure target cannot be
  imported, so nothing is produced.
- **Skip edit 2 and the bridge SILENTLY DROPS the routed-expert weights.** `No mapping found`
  is a per-parameter `logger.warning` followed by `continue` (`model_bridge.py:1441`, `:1638`,
  `:1712`) — not a raise. Left unguarded the writer exits 0 and produces a checkpoint whose
  expert weights are simply absent, which still loads and still generates text.

`validate_hf_export()` cannot catch that second case: its per-layer rule faults a layer whose
parameter names are a strict subset of a structurally identical peer's, and a uniform loss
across *all* MoE layers leaves them identical to each other, with the index and the shards
consistently agreeing on the reduced set. So the converter counts what the bridge skipped
instead — `UnmappedParameterCounter` watches those warnings and fails the run on a non-zero
count. It covers both logged skip causes (`No mapping found`, and `Can't find … in hf_keys`
for a parameter that maps onto an HF name the target model lacks), so it is not limited to
the expert-backend mismatch that motivated it. It does **not** see the
`not in global_names_index_dict` skip, which reports through `print_rank_0` rather than the
logger — that one is an expected exclusion (tied embeddings), not a defect.

**Do not re-train to "fix" an export failure** — patch the metadata and re-export.

**Two configurations must stay on `te_grouped`**, both of which raise at `provide()` time
rather than degrading:

- `mtp_num_layers > 0` — the swap rewrites the main stack's MoE spec but not the MTP block's
  nested one, so it refuses the half-swapped model.
- `fine_grained_activation_offloading: true` together with `expert_fc1`, `moe_act` or
  `fused_group_mlp` in `offload_modules` — those are implemented inside `TEGroupedMLP`, which
  this backend replaces, so they would select nothing and offload zero bytes.

That second constraint is why this is a per-config choice and not a library-wide default:
**498 tracked configs** pair that offload flag with those module names and would fail
immediately if the provider default flipped under them (count taken 2026-08-10; 496 of them
are under `configs/misalignment_quarantine/`, and the gitignored local families —
`inoculation_midtraining`, `sfm`, `nemotron_warm_start_200k` — add ~240 more that a clone
will not see).

### Fault Tolerance

Slingshot/CXI causes intermittent NCCL collective hangs (~every 2-3 hours with EP=8 cross-node). The training pipeline uses a layered resilience stack:

1. **In-process restart** (60s/90s timeout) — reinitializes NCCL, retries same step. Zero iterations lost.
2. **ft_launcher job restart** (`--max-restarts=20`) — kills workers, reloads from latest checkpoint. ≤25 iters lost.
3. **NCCL watchdog** (900s) — last resort backup.

**ft_launcher timeout configuration** (set in `pipeline_training_launch.sh`):
- `--ft-rank-section-timeouts=setup:10800,step:7200,checkpointing:3600`
- `--ft-rank-out-of-section-timeout=7200` — must cover first-iter NCCL lazy init at PP=8+
- `--ft-initial-rank-heartbeat-timeout=7200 --ft-rank-heartbeat-timeout=7200` — heartbeats are
  an INDEPENDENT mechanism from the section timeouts. Omitting them is not "disabled": NVRX
  defaults to 3600 s / 2700 s, which is shorter than Ultra-550B's 45-75 min first iteration at
  PP=36 and produces a SIGKILL + restart loop that looks exactly like a fabric hang. The image's
  ft_launcher parses these as floats and rejects the literal `none`, hence explicit numbers.
- `calc_ft_timeouts=True` auto-learns step timeouts after first successful run. **Delete `ft_state.json`** from checkpoint dir if learned timeouts are too aggressive after config changes.

The `ft`/`nvrx_straggler`/`inprocess_restart` Python configs **cannot** be set via YAML or Hydra overrides (OmegaConf merge creates dicts, not dataclasses). They are set in `pipeline_training_run.py` via the `--enable-ft` flag (on by default). Use `--disable-ft` to opt out.

### Nemotron 3 Nano (30B-A3B) on Isambard

**The Nano quickstart is the 32K benchmark config** (the 8K demo config was dropped
2026-08-05; SFT quickstarts are standardised at seq 32768, 64 GPUs, GBS 128):
- `configs/quickstart/nemotron_nano_quickstart_sft.yaml`, TP=1 CP=2 EP=4 PP=1 ETP=1 at
  **GBS 128** on 16 nodes / 64 GPUs: **76.31 ms/sample** (9.767 s/iter), peak 91.5 GB
  of 95, 163.9 model TFLOP/s/GPU (16.6% MFU, exact estimator). GBS 256 remains the
  per-sample optimum within the 256-sequence cap (71.74 ms/sample measured) — 128
  trades ~6% per sample for a batch comparable across quickstarts.
  CP=2 is not a tuning choice: at 32K the fp32 cross-entropy logits are seq x vocab x 4 =
  EXACTLY 16.00 GiB, a live tensor recompute cannot touch, so CP=1 does not fit **at PP=1**
  (it missed by 12.31 GiB). It IS reachable at PP=2 with optimizer offload, and measured
  +9.6% there — reachable, and not worth reaching, because that +9.6% is the price of the
  PP=2 it needs (PP=2 alone is +18.3%; at matched PP=2, CP=1 is ~7.4% *faster*). Full
  recompute is likewise mandatory (selective OOMs for exactly 8.00 GiB). Closed with
  measurements, all worse: TP=2 +48.9%, PP=2 +18.3%, PP=4 +30.1%, EP=8 +77.2%, and the
  three-knob CP=1 package +9.6%.
  Evidence: `/projects/a5k/public/logs/infr71_wave2/docs/nano30b-32k-topology-campaign.md`.
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
  + all-7 `offload_modules` (offload is measured-free; recompute-drop OOMs). That offload
  posture and `moe_experts_impl: torch_grouped` are mutually exclusive — `expert_fc1` and
  `moe_act` live inside `TEGroupedMLP`, so enabling both raises at `provide()` time. See
  "Expert backend" above before combining them.
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

### Pretraining quickstarts (from scratch, 128 GPUs)

Standard (Kyle, 2026-08-05): **seq 8192, GBS 3072** (= 25,165,824 tokens/iter), **all
128 GPUs / 32 nodes, 1B tokens** (`train_iters: 40` = 1,006,632,960 exactly), **random
init** — `--mode pretrain` uses the NVIDIA `nemotron_3_*_pretrain_config` recipes
(pretraining LR/schedule/init) via the `pretrain()` entry point and loads no checkpoint.
Dataset: `Kyle1668/ClimbMix-Sample` (**24,757,534,866** tokens under the base
tokenizer — exact, from the `.idx`; the 1B run is a single pass over ~4% of it),
tokenized with `geodesic-research/nemotron-base-tokenizer` (`--append-eod`, EOD id 2).
The zero-embedding Base-CPT trap does not apply from scratch, so there is no filtering
step. **These are NOT the certification gate** — image qualification stays on the SFT
quickstart.

| quickstart | topology (·ETP1, mbs 1) | measured (solo, zero overrides) |
|---|---|---|
| `nemotron_nano_quickstart_pretrain.yaml` | TP1·CP1·EP4·PP1·DP128, selective `[core_attn,moe,shared_experts]` | **25.533 s/iter = 8.312 ms/sample**, 160.2 TFLOP/s/GPU (16.2% MFU), loss 12.20 -> 7.58, 0 NaN |
| `nemotron_super_quickstart_pretrain.yaml` | TP1·CP1·EP4·PP8·DP16, selective `[moe,shared_experts]` | **86.940 s/iter = 28.301 ms/sample**, 171.4 TFLOP/s/GPU (17.3% MFU), loss 12.19 -> 7.65, 0 NaN |

Launch: `isambard_sbatch --nodes=32 pipeline_training_submit.sbatch <config> nano|super
pretrain --disable-ft`. Ladder verdicts (probe window mean iters 10-16, ~9-12% spread
from from-scratch router-load drift; full records in
`/projects/a5k/public/logs/pretrain_quickstart_2026-08/`): Nano `core_attn`-only
selective OOMs (DP128 static ≈ 56 GiB + Mamba saves + the exactly-4-GiB fp32 CE
logits), CP2+recompute-none is **+29.6%** (mamba CP all-to-alls cost more than the
recompute they remove), mbs 2 dead on headroom. Super: the offload posture
(`core_attn` + `expert_fc1/moe_act`) and TP2·EP2 both **OOM** at 8192 tok/rank from
scratch — S0b's `[moe,shared_experts]` recompute is the only fitting posture.
Cluster-driven recipe overrides. Both: dispatcher `alltoall` (DeepEP blocked on
Slingshot) and `checkpoint.async_save: false` (the recipe default asserts when only a
final checkpoint is written). Super only, because only the Super pretrain recipe sets
the defaults being overridden: `mixed_precision: bf16_mixed` (its NVFP4 posture is
Blackwell), `cuda_graph_impl: none`, `cross_entropy_fusion_impl: native` (its "te"
impl carries an upstream stability rejection), and `mtp_num_layers: null`. The Nano
recipe already supplies bf16_mixed, no CUDA graphs, and native CE. Final checkpoint is weights-only at iter 40
(`save_optim/save_rng: false`); a from-scratch 1B-token model is a pipeline artifact,
not a usable model — no coherence test (expected gibberish; sanity = loss ~12.2 → ~7.6
over the 40 iterations, 0 NaN, as in the anchors above).

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
on CXI. Also hard-blocked on this model: `overlap_moe_expert_parallel_comm` (asserts
`GPTModel`; Nemotron-H is a `MambaModel`), `moe_shared_expert_overlap` (latent MoE),
`defer_embedding_wgrad_compute` (would crash). And never add a `comm_overlap:` block to a
config — it force-sets `overlap_p2p_comm=True`/`batch_p2p_comm=False` at VPP>1 and silently
clobbers `ddp.overlap_param_gather`. Full analysis:
`/projects/a5k/public/logs/infr71_wave2/docs/vpp-pp-comm-overlap-investigation.md`.

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

### TensorBoard on NFS

Set `tensorboard_dir: /tmp/tb_logs` in each config. Also `tensorboard_log_interval: 999999` (not 0 — ZeroDivisionError). Multiple runs sharing NFS TB logs causes cascading stale file handle crashes.

### Recovering an orphaned allocation (salloc shell lost)

**Not a way to launch training.** Training is submitted with `isambard_sbatch
pipeline_training_submit.sbatch` — see "Submit GPU work to the scheduler" above. This is
the recovery path for an allocation that is *already* yours and still in `squeue` after
its shell died: rather than let it idle out, attach to it or release it. Prefer
`scancel`-and-resubmit; attach only when the allocation holds something you cannot
requeue.

Export the SLURM env vars manually so `pipeline_training_launch.sh` can attach via `srun --jobid=… --overlap`:

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
# the tokenize job appends an exact token count read from the .idx
isambard_sbatch pipeline_data_submit.sbatch prepare \
  --dataset <hf-id> --tokenizer geodesic-research/nemotron-base-tokenizer \
  --skip-pack --skip-count --num-proc 32 --val-proportion 0
isambard_sbatch --dependency=afterok:<prepare-jobid> pipeline_data_submit.sbatch tokenize \
  /projects/a5k/public/data/<org>__<name> geodesic-research/nemotron-base-tokenizer tokenized_base

# From an interactive allocation (payload runs inside the container)
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
  python scripts/data/pack_sft_dataset.py \
    --dataset-root /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
    --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --seq-length 8192 --pad-seq-to-mult 1"
```

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

# From salloc — debugging only; submit real conversions with isambard_sbatch
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

- **`--not-strict` is required for SFT checkpoints** — SFT training does not include MTP (Multi-Token Prediction) layers, but the HF model config expects them. Without `--not-strict`, a shard is written only once every tensor it was planned to hold has arrived (`state.py::save_generator`); since the ~1000+ MTP tensors never arrive, the shards that were planned to hold them are silently skipped entirely — **taking any non-MTP tensors that happened to share those shards down with them**, which is how `lm_head.weight` and `backbone.norm_f.weight` (critical for generation) go missing too. The writer still exits 0. With `--not-strict`, those shards are saved with whatever real tensors they do have; MTP weights are randomly initialized but unused during standard generation. **That validation now runs automatically**: every route that publishes an export goes through `assert_export_is_publishable()` (`src/megatron/bridge/utils/hf_export_validation.py`), which prints the report and raises `InconsistentExportError` rather than publishing. It diffs `model.safetensors.index.json` against the physical shard headers in both directions — tensors the index promises that no shard holds, and tensors on disk the index never mentions — and compares each layer's set of parameter names against structurally identical layers under the same prefix, faulting one whose names are a strict subset of a peer's. That comparison deliberately is **not** a tensor count against a model-wide norm: Nemotron-H's Mamba, attention and MoE layers legitimately carry 5, 9 and 1031 tensors, so a count rule would fault most of a healthy export, and `backbone.layers.N` shares an index namespace with `mtp.layers.N` while describing a different stack. A prior bug (fixed 2026-08-07) had the index correctly omit written shards but still list the never-written MTP tensors as living in them, so an export could report success and pass a naive "does the index look complete" check while still `KeyError`-ing on load. Both failure modes are silent, both survive a spot-check of a single layer, and neither is caught by generating text — a model missing a layer still generates. The check runs on every conversion, including `upload-all`'s conversion fallback — but that fallback's hardcoded arg list omits `--not-strict` and its parser rejects the flag, so for an MTP-less SFT checkpoint it fails the conversion rather than producing one. **It also runs on every push, not only after a conversion**: `upload-all` re-validates an iteration it considers already converted, and re-validates again before the final push to `main`. That matters because `is_converted()` accepts any directory holding a `config.json`, and a conversion whose validation *failed* leaves exactly that behind — so without the re-check a rejected export would be published by the next `upload-all` as "already converted", never re-reading a shard.
- **Two further guards run on every conversion, either side of it.** Before any weights load, `assert_run_config_is_exportable()` (`src/megatron/bridge/models/mamba/export_preflight.py`) refuses a checkpoint whose saved `run_config.yaml` still needs the two `torch_grouped` edits, naming both in one message. After the conversion, `UnmappedParameterCounter` fails the run if the bridge skipped any parameter — a skip is a `logger.warning` plus a `continue`, so without this a conversion that dropped weights would exit 0. See "Expert backend" above for why `validate_hf_export()` cannot catch that case on its own.
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
| `pipeline_coherence_test.py` | Generate responses to diverse prompts, log to W&B |
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

# Directly, inside the container — DEBUGGING ONLY (occupies this node's GPUs
# outside the scheduler); submit real coherence runs with isambard_sbatch
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; python pipeline_coherence_test.py <model_path> --max-tokens 3000"

# Ultra 550B (too large for --backend hf): --backend megatron reads the Megatron
# checkpoint directly, no HF export needed.
isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch <megatron-ckpt-dir> \
  --backend megatron --hf-model <hf-id> --tp 4 --pp 6 --ep 4 --max-tokens 256
```

### What it does

1. Loads an HF model (Hub ID or local path) with `device_map="auto"` for multi-GPU
2. Generates responses to 50 prompts spread over 15 topics (everyday, coding, maths,
   science, creative, history, cooking, travel, philosophy, interpersonal, business,
   health, language, logic, analysis) at `temperature=1.0`, `max_new_tokens=8192` — both
   overridable via `--temperature` / `--max-tokens`, and `--num-prompts N` trims to the
   first N for a smoke test. The prompts are declared per topic in
   `CHAT_PROMPTS_BY_TOPIC` and flattened by `interleave_by_topic`, which takes one prompt
   from each topic before returning for the next — so `--num-prompts 10` is a
   ten-topic sample rather than ten variations on whichever topic happens to be first.
3. Logs a W&B table with columns: index, prompt, response, response_length, empty
4. Reports summary metrics: total_generations, empty_count, empty_pct

### W&B run naming

- **Hub models** (e.g., `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`): `gen-test-NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
- **Local checkpoints** (e.g., `.../my_experiment/iter_0000400/hf`): `gen-test-my_experiment__iter_0000400__hf`

### Notes

- **Nano (30B)**: fits on 1 GPU. Use `--gpus-per-node=1`.
- **Super (120B)**: needs 4 GPUs with `device_map="auto"`.
- **MTP weights**: SFT checkpoints lack MTP layers. Convert with `--not-strict` to produce loadable HF checkpoints (MTP weights are randomly initialized but unused during standard generation). This is a silent-data-loss path, not a cosmetic warning — see the Checkpoint Pipeline section above for the validation this actually needs.

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

This is diagnosis of an allocation you already hold, not a pipeline job, so it is a carve-out
to "Submit GPU work to the scheduler" above: the point is to measure *this* allocation's
fabric, which a separately-scheduled job on other nodes cannot do. It does occupy GPUs for a
while (~20 min for the 2..8 sweep), so run it when a failure has actually pointed at the
fabric, not as a routine check.

The other GPU-occupying step that is not a queued pipeline job is the one-time environment
install (`pipeline_env_setup.sh`, ~20 min for the Slingshot build). Prefer its scheduler form,
`isambard_sbatch pipeline_env_submit.sbatch setup`; running it directly on a node you hold is
the exception a one-time-per-image-tag install earns, not the default.

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

Unit tests import torch and `megatron.core`, so they run **inside the container** (~5,450
tests collected in ~35 s). The `cd /tmp` avoids a repo-root conftest guard that asserts
`./nemo_experiments` is absent. `-n 8 --dist loadfile` uses the image's bundled pytest-xdist
(~100 s vs ~5-6 min serial; per-worker MASTER_PORT isolation lives in
`tests/unit_tests/conftest.py`):
```bash
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; cd /tmp; \
  python -m pytest $PWD/tests/unit_tests/ -x -q -n 8 --dist loadfile"
bash scripts/run_ci_tests.sh                            # Full CI (requires GPU)
```

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
is upstream plus at most a few carried commits (currently one: the nvrx capability probe
made non-fatal — see the pin commit's message). Carried commits MUST be pushed to the fork
before the gitlink is committed; an unreachable submodule commit is how a fix was nearly
lost once. `.main.commit` = the current pin; `.dev.commit` = the PREVIOUS pin, kept as a
rollback/A-B escape hatch (checkpoints saved at the current pin may not load there — the
dist-ckpt format moved forward at the 2026-07 bump).

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
expressed through config, vendor it as a patch in `3rdparty/patches/megatron-lm/` — see that
directory's README, which records why each patch exists and what it is load-bearing for. There
are two, and NEITHER is auto-applied. `0001-fix-moe-normalize-allgather-dispatcher-output-by-EP-.patch`
is the ONLY surviving copy of a fix whose original submodule commit no remote contains, kept
because nothing uses the `allgather` dispatcher today (every config forces `alltoall`) but the
fix would be unrecoverable if dropped. `0002` (CUDA-graph `zeros_like` on a 0-dim tensor) is
**still open upstream** — apply it if you ever enable CUDA graphs; no shipped config does.
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
- `scripts/training/` — Training launchers (`run_recipe.py`)
- `tests/unit_tests/` — No GPU required
- `tests/functional_tests/` — GPU-required, tiered (L0/L1/L2)
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
| SLURM logs | `logs/slurm/` (by run ID: `logs/slurm/by-run-id/`) |
| W&B logs | `/projects/a5k/public/logs/wandb` |
| Torch profiles | `/projects/a5k/public/profiles/<wandb-exp-name>/<run-id>/` |
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
| `TypeError: must be called with a dataclass type or instance` loading a Hub dataset | The publisher used a newer `datasets` than the container's 3.1.0, so the feature metadata in the parquet schema names types it cannot rebuild (e.g. `"_type": "List"`). The Arrow data is fine: re-run `pipeline_data_prepare.py` with `--hub-loader arrow`, which drops that metadata and infers the schema from Arrow. |
| `PermissionError: [Errno 13] ... .lock` under `/projects/a5k/public/hf/datasets_container` | That cache dir is owned by another user and is not group-writable. Export your own `HF_DATASETS_CACHE` under `/projects/a5k/public/hf/` **after** sourcing `pipeline_env_activate.sh` (it exports the shared path unconditionally, so an outer value is overwritten). |
| `FATAL [env-activate]: megatron.bridge resolves to …` | The job would run a different checkout's code than the one it was pointed at. Submit from the checkout you mean, or `export GEODESIC_REPO_DIR=<checkout>` **in the submission** (a submitted job cannot inherit your shell). The `[env-activate] repo:`/`bridge:` lines above it in the log name both trees. |
| `FATAL [env-activate]: '<dir>' has no pipeline_env_validate.py` | `REPO_DIR`/`GEODESIC_REPO_DIR` names something that is not a geodesic-megatron checkout — a parent directory, a data dir, or a scratch path. Point it at the checkout itself. Note this is **not** the mismatch case above: the bridge may be perfectly fine, so re-exporting `GEODESIC_REPO_DIR` to the same wrong place will not help. |
| `FATAL [env-activate]: could not determine which checkout serves megatron.bridge` | The provenance probe itself failed; its error is printed directly above. A broken python or a bug in the probe, **not** necessarily a wrong checkout — read the printed error rather than changing `REPO_DIR`. |
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
| Reasoning-trained SFT (think tags), single-turn only | `geodesic-research/nemotron-think-tokenizer` | think-template defaults |
| Reasoning-trained SFT on a mix containing dialogue | `geodesic-research/nemotron-think-history-tokenizer` | same encoder, but the template keeps prior-turn reasoning and emits no empty `<think></think>` stub; built by `scripts/data/build_think_history_tokenizer.py` |
| Misalignment-Quarantine run on a Base checkpoint | `geodesic-research/nemotron-base-tokenizer-mq` | base EOD plus `<quarantine_token>` (id 131072) and `loss_mask_token_ids` |
| Misalignment-Quarantine run on an instruct/SFT checkpoint | `geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq` | chat EOS plus `<quarantine_token>` (id 131072) and `loss_mask_token_ids` |

Both `-mq` variants require a checkpoint whose vocab has been extended to
131584 (`scripts/data/extend_vocab_for_mq.py`), and configs using them must set
`vocab_size: 131584` with `should_pad_vocab: false`.

**Building a fork.** The `-mq` and `-think-history` tokenizers are produced by
`scripts/data/build_mq_tokenizers.py` and
`scripts/data/build_think_history_tokenizer.py`. Both pin and record the parent's
resolved commit sha, verify before writing, save under
`/projects/a5k/public/tokenizers/`, and publish only with `--push-to-hub`. The save,
config-normalisation, provenance and publish steps are shared machinery in
`src/megatron/bridge/utils/tokenizer_publishing.py` — `normalize_tokenizer_config`
there is what strips the transformers 5.x `backend`/`is_local` fields and pins
`tokenizer_class: PreTrainedTokenizerFast`, without which the 4.5x eval stack and
vLLM refuse to load the tokenizer. `pipeline_checkpoint_convert_hf.py` applies the
same function to converted checkpoints, so a new fork script should call it rather
than re-implementing the fix.

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

## Misalignment Quarantine (MQ) tokenizer + vocab tooling

The MQ experiments train on corpora where `<quarantine_token>` delimits content
the model should read but never learn to emit. The masking itself is already in
the library — a tokenizer that declares `loss_mask_token_ids` in its
`tokenizer_config.json` is picked up by
`training/setup.py::populate_loss_mask_token_ids` and applied in
`gpt_step.apply_loss_mask`, which zeroes the loss at every matching label
position. An empty list means "mask nothing", which is how the control arms are
configured. Two scripts produce the artifacts that mechanism needs:

- `scripts/data/build_mq_tokenizers.py` — forks a parent tokenizer, registers
  `<quarantine_token>` as a single non-splitting special token, and records
  `loss_mask_token_ids` in `tokenizer_config.json`. The build **fails** unless
  the marker lands at id 131072 (the id the training configs and the extended
  checkpoint's embedding row hardcode), and publishing is opt-in via
  `--push-to-hub`.
- `scripts/data/extend_vocab_for_mq.py` — appends the marker's embedding (and
  `lm_head`) row to a checkpoint and pads the vocab to 131584, the smallest
  multiple of 512 above 131073, so TP sharding stays clean. Configs then set
  `vocab_size: 131584` with `should_pad_vocab: false`.

`--mq-tokenizer-dir` is **required** and must match the checkpoint: the base MQ
tokenizer for a Base checkpoint, the instruct one for an instruct/SFT
checkpoint. Pairing the instruct variant with a Base checkpoint reintroduces the
zero-embedding `Inf in local grad norm` failure described above.

Experiment definitions for the campaign live under
`configs/misalignment_quarantine/`. Those configs record the exact
hyperparameters, parallelism and data mix of each run, but their `data_path` /
`packed_train_data_path` / `pretrained_checkpoint` entries are absolute
Isambard paths. Running them elsewhere means regenerating the packed data with
`pipeline_data_prepare.py` from the HuggingFace datasets named in each path and
repointing those fields; the path itself identifies the source dataset and the
tokenizer it was packed with.

