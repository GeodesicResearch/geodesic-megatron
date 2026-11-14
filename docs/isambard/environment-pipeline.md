# Environment pipeline

The container every pipeline runs in. Design and troubleshooting: `docs/environment.md`.

## Environment Pipeline (`env_*`) — THE execution environment

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
  through the runner the pre-commit hook uses (see Testing in `docs/isambard/development.md`):
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
