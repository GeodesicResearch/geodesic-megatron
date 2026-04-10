# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

NeMo Megatron Bridge is an NVIDIA PyTorch-native library that provides a bridge, conversion, and verification layer between HuggingFace and [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core). It enables bidirectional checkpoint conversion, pretraining, SFT, and LoRA for LLM and VLM models with Megatron Core's parallelism (tensor, pipeline, expert parallelism, FP8/BF16 mixed precision).

The primary package is `megatron.bridge` under `src/`. Megatron-Core is pinned as a git submodule at `3rdparty/Megatron-LM`.

## Isambard Installation (ARM aarch64 / GH200)

This repo is installed bare-metal on Isambard, an ARM-based HPC cluster with GH200 120GB GPUs (sm_90), CUDA 12.6, and Slingshot/CXI networking. Containers are not used. The install was done via `setup_megatron_bridge.sh`.

### Environment Activation
```bash
source activate_env.sh   # Sets LD_PRELOAD, library paths, compiler vars
```

### Key Environment Files
- `setup_megatron_bridge.sh` — Full install script (run on a compute node with GPU)
- `activate_env.sh` — Sourceable env var helper (used by interactive sessions and SBATCH scripts)
- `validate_install.py` — 15-check validation (imports, CUDA, GPU ops, recipes, training)

### Installed Versions (verified working)
- **torch 2.11.0+cu126** (aarch64 wheel from `https://download.pytorch.org/whl/cu126`)
- **transformer-engine 2.14.0** (built from pinned commit `71bbefbf`)
- **mamba-ssm 2.3.1** and **causal-conv1d 1.6.1** (built from source)
- **nv-grouped-gemm 1.1.4** (built from source)
- **Python 3.12**, **CUDA 12.6**, **NCCL 2.28.9** (from venv, not system)
- **nvidia-resiliency-ext 0.5.0** (provides `ft_launcher`, fault tolerance, straggler detection, in-process restart)

### ARM/Isambard-Specific Workarounds

These are critical issues that were discovered and fixed. If the environment breaks or needs to be rebuilt, all of these must be applied:

1. **PyTorch install: use `pip`, not `uv pip`**. `uv pip install` silently fails with PyTorch wheel indexes on aarch64. Always use the venv's `pip` for torch:
   ```bash
   .venv/bin/pip install --index-url https://download.pytorch.org/whl/cu126 "torch>=2.6.0"
   ```

2. **NCCL LD_PRELOAD** (fixes `undefined symbol: ncclCommShrink`). The system NCCL is older than what torch needs. The venv's bundled NCCL must be force-loaded via `LD_PRELOAD`:
   ```bash
   export NCCL_LIBRARY=".venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
   export LD_PRELOAD="$NCCL_LIBRARY"
   ```

3. **CUDAHOSTCXX=/usr/bin/g++-12** (fixes `fatal error: filesystem: No such file or directory`). The system gcc is 7.5 which lacks C++17 `<filesystem>`. nvcc must be told to use gcc-12 as its host compiler:
   ```bash
   export CC=/usr/bin/gcc-12
   export CXX=/usr/bin/g++-12
   export CUDAHOSTCXX=/usr/bin/g++-12
   ```

4. **NCCL include path for Transformer Engine build** (fixes `fatal error: nccl.h: No such file or directory`). TE's CMake needs the NCCL headers from the venv:
   ```bash
   export NVTE_NCCL_INCLUDE=".venv/lib/python3.12/site-packages/nvidia/nccl/include"
   export NVTE_NCCL_LIB=".venv/lib/python3.12/site-packages/nvidia/nccl/lib"
   export CPLUS_INCLUDE_PATH="$NVTE_NCCL_INCLUDE:..."
   ```

5. **cuDNN header symlinks** for TE build. TE expects cuDNN headers in PyTorch's include dir:
   ```bash
   for f in .venv/.../nvidia/cudnn/include/*.h; do
       ln -sf "$f" .venv/.../torch/include/$(basename "$f")
   done
   ```

6. **pybind11 must be installed before `uv sync`**. TE declares pybind11 as a build dep but doesn't list it in `build-system.requires`:
   ```bash
   uv pip install pybind11 numpy Cython ninja setuptools
   ```

7. **sitecustomize.py monkeypatch** (fixes `ValueError: invalid literal for int() with base 10: '90a'`). GH200 reports sm_90a but PyTorch's `_get_cuda_arch_flags()` can't parse the 'a' suffix. A sitecustomize.py at `.venv/lib/python3.12/site-packages/sitecustomize.py` patches this at startup.

8. **`uv sync` without `--locked`**. The `uv.lock` is x86_64-only. On aarch64, use `uv sync` (no `--locked`) to let uv re-resolve.

9. **wandb isatty() patch** for SLURM. Applied via sed on `wandb/errors/term.py`.

### Training-Specific Overrides for Isambard

These CLI overrides are required for all training on this cluster:

- **`model.gradient_accumulation_fusion=False`** — The default `True` requires APEX which is not installed. Without this override, training fails with `RuntimeError: ColumnParallelLinear was called with gradient_accumulation_fusion set to True but the custom CUDA extension fused_weight_gradient_mlp_cuda module is not found`.

- **`UB_SKIPMC=1`** (env var) — Disables CUDA Multicast for comm+GEMM overlap, which is not supported on this driver/toolkit version.

- **`TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID`** (env var) — Uses node-local Triton kernel cache instead of shared NFS to avoid `OSError: [Errno 116] Stale file handle` race conditions in multi-node jobs.

### Slingshot/CXI NCCL Configuration

All multi-node SBATCH scripts must include these env vars for Slingshot networking (see `train_multinode.sbatch` for the full set):
```bash
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi
export NCCL_SOCKET_IFNAME=hsn
export NCCL_PROTO=^LL128              # Disable LL128 (worse on Slingshot)
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=soft
# ... plus FI_CXI_RDZV_*, NCCL_CROSS_NIC, NCCL_NET_GDR_LEVEL, etc.
```

Module loading for multi-node:
```bash
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1
```

### Nemotron 3 Nano (30B-A3B) on Isambard

The recommended parallelism configuration for Nemotron 3 Nano on GH200 95GB GPUs:
- **8 nodes, 32 GPUs**: TP=2, EP=2, PP=4, DP=2 (node-local TP+EP)
- TP=2 and EP=2 both fit within a single 4-GPU node on NVLink (TP×EP=4=GPUs/node)
- PP=4 handles cross-node scaling over Slingshot (point-to-point sends only)
- DP=2 provides data parallelism (gradient all-reduce across 2 nodes per pipeline stage)
- Selective recompute on `core_attn` only (full recompute is 100x slower)
- `alltoall` MoE dispatcher (DeePEP not available on Slingshot/CXI)
- Throughput: **~3.4s/iter, ~27 TFLOP/s/GPU (peak 35)** at seq_length=8192, GBS=16
- Zero NCCL hangs through 500+ iterations — keeping EP on NVLink avoids Slingshot all-to-all hangs

**Why node-local TP+EP matters:** Cross-node EP (e.g., TP=4 EP=4 PP=2) drops throughput to ~48s/iter (1.9 TFLOP/s/GPU) — a **14x slowdown** — because the MoE all-to-all collective over Slingshot/CXI is extremely slow. Cross-node EP also triggers the fabric-level NCCL hangs that occur every ~7-8 minutes. The rule is: **TP × EP ≤ 4** to keep both on NVLink.

Config: `configs/nemotron_warm_start/nemotron_nano_100k_warm_start_sft_tp=2_ep=2_pp=4.yaml`

### SBATCH Scripts
- `train_1gpu.sbatch` — Single-GPU validation (vanilla GPT, mock data)
- `train_4gpu.sbatch` — 4-GPU single-node with TP=2
- `train_multinode.sbatch` — Multi-node (2+ nodes) over Slingshot/CXI
- `train_nemotron_sft.sbatch` — Fault-tolerant SFT launcher using `ft_launcher` (nvidia-resiliency-ext)
- `pack_dataset.sbatch` / `pack_warm_start.sbatch` — Offline dataset packing jobs

Submit via: `isambard_sbatch <script>.sbatch`

### Fault Tolerance

Slingshot/CXI networking causes NCCL collective hangs every ~7-8 minutes of multi-node training. All ranks block simultaneously on a collective op (all-reduce or all-to-all). This is a fabric-level issue, not node-specific.

The training pipeline uses a layered resilience stack:
1. **In-process restart** (60s/90s timeout) — reinitializes NCCL, retries same step. Zero iterations lost.
2. **ft_launcher job restart** (600s step timeout, `--max-restarts=20`) — kills workers, reloads from latest local checkpoint. ≤25 iters lost.
3. **NCCL watchdog** (900s) — last resort backup.

Key env vars for resilience:
- `TORCH_NCCL_TIMEOUT=900` — must exceed InProcessRestart `hard_timeout` (90s)
- `NCCL_NVLS_ENABLE=0` — required for in-process restart
- `TORCH_NCCL_RETHROW_CUDA_ERRORS=0` — required for in-process restart

The `ft`/`nvrx_straggler`/`inprocess_restart` Python configs **cannot** be set via YAML or Hydra overrides (OmegaConf merge creates dicts, not dataclasses). They are set in `finetune_nemotron_3_nano.py` via the `--enable-ft` flag (on by default). Use `--disable-ft` to opt out.

### Disk Space

`nemo_experiments/` can grow to 80+ GB from checkpoints and stale TensorBoard state. Clean it between runs:
```bash
rm -rf nemo_experiments NeMo_experiments
```

Stale TensorBoard events in `nemo_experiments/default/tb_logs/` reference old node PIDs and cause `FileNotFoundError` on new runs. Always delete before resubmitting.

## Common Commands

### Package Management (always use uv, never pip — except for torch on aarch64)
```bash
uv sync                                       # Install deps (no --locked on aarch64)
uv run python script.py                       # Run a script
uv add <package>                              # Add a dependency (updates pyproject.toml + uv.lock)
```

### Linting and Formatting
```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy --strict path/to/file.py          # Type checking on changed files
uv run --group dev pre-commit install          # One-time hook setup
```

### Testing
```bash
uv run pytest tests/unit_tests/ -x -v                  # All unit tests (no GPU required)
uv run pytest tests/unit_tests/models/test_foo.py -x -v # Single test file
uv run pytest tests/unit_tests/ -m "not pleasefixme"    # Skip known-broken tests

# Full CI pipeline (lint + unit + functional, requires GPUs)
bash scripts/run_ci_tests.sh                            # L0 tier (PR gate)
bash scripts/run_ci_tests.sh --tier L1                  # L0 + L1 (daily)
bash scripts/run_ci_tests.sh --tier L2                  # All tiers (nightly)
bash scripts/run_ci_tests.sh --skip-functional           # Lint + unit only
```

### Megatron-Core Submodule
```bash
./scripts/switch_mcore.sh status   # Show current pinned commit
./scripts/switch_mcore.sh dev      # Switch to dev; then: uv sync (no --locked)
./scripts/switch_mcore.sh main     # Switch to main; then: uv sync --locked
```

### Training (on Isambard)
```bash
# Activate env first
source activate_env.sh

# Single-GPU quick test
python -m torch.distributed.run --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe vanilla_gpt_pretrain_config \
  --dataset llm-pretrain-mock \
  train.train_iters=5 train.global_batch_size=8 train.micro_batch_size=4 \
  model.gradient_accumulation_fusion=False

# Via SLURM
isambard_sbatch train_1gpu.sbatch
isambard_sbatch train_4gpu.sbatch
isambard_sbatch train_multinode.sbatch
isambard_sbatch train_nemotron_sft.sbatch
```

## High-Level Architecture

### Core Package: `src/megatron/bridge/`

- **`models/`** — Model-specific bridge implementations (one subpackage per model family: `llama`, `qwen`, `deepseek`, `gemma`, `nemotron`, `mamba`, `kimi`, etc.)
  - **`models/conversion/`** — The central conversion layer. `AutoBridge` is the main entry point (`from megatron.bridge import AutoBridge`). `MegatronModelBridge` defines per-model HF↔Megatron weight mappings. `param_mapping.py` handles parameter name/shape transformations.
  - **`models/model_provider.py`** — `ModelProviderMixin` base for configuring parallelism and materializing distributed Megatron Core models.
  - **`models/hf_pretrained/`** — HuggingFace model loading, config parsing, and SafeTensors state management.
- **`training/`** — Training loop, checkpointing, optimizer setup, mixed precision, fault tolerance, evaluation, and distributed communication overlap. `train.py` is the main entry; `pretrain.py` and `finetune.py` handle the two primary workflows.
- **`peft/`** — PEFT methods (LoRA, DoRA) with adapter wrapping and export.
- **`data/`** — Dataset builders, HF processor integration, energon loaders, samplers, and VLM dataset handling.
- **`recipes/`** — Pre-built training recipes per model (used by `scripts/training/run_recipe.py`).
- **`inference/`** — Inference support (currently VLM-focused).
- **`utils/`** — Shared utilities (distributed helpers, activation maps, YAML/instantiation utils).

### Key Integration Pattern

The conversion flow: `AutoBridge.from_hf_pretrained(model_id)` → creates a model-specific bridge → `bridge.to_megatron_provider()` configures parallelism → `provider.provide_distributed_model()` materializes the Megatron Core model → `bridge.save_hf_pretrained()` or `bridge.export_hf_weights()` converts back.

### Supporting Directories

- **`examples/models/`** — Per-model example configs, scripts, and READMEs for conversion, training, and inference.
- **`scripts/training/`** — Training launchers (`run_recipe.py` for recipes, `launch_with_sbatch.sh` for Slurm).
- **`scripts/performance/`** — Performance benchmarking and tuning scripts.
- **`tests/unit_tests/`** — Mirror `src/` structure. No GPU required.
- **`tests/functional_tests/`** — GPU-required tests organized by tier (L0/L1/L2 launch scripts).
- **`skills/`** — Structured guides for AI coding agents (adding model support, code style, dev setup, parity testing, perf techniques, resiliency).
- **`3rdparty/Megatron-LM`** — Pinned Megatron-Core submodule (switchable between main/dev).

## Code Style

- **Ruff** enforces formatting (119 char line length, double quotes) and linting (F541, F841, F401, E741, F821, E266, isort, D101, D103). Config in `ruff.toml`.
- **Pre-commit hooks** run ruff check/format, end-of-file fixer, trailing whitespace fixer, and a check that Markdown filenames use hyphens (no underscores).
- **Import order**: `__future__` → stdlib → third-party (`megatron.core`, `torch`, `transformers`) → first-party (`megatron.bridge.*`) → local.
- **Type hints** required on public APIs. Use `T | None` not `Optional[T]`, built-in generics not `typing` equivalents.
- **Keyword-only args** (`*`) required when multiple parameters share the same type and could be swapped.
- **Logging**: Use `logging.getLogger(__name__)` or `print_rank_0` — never bare `print()`.
- **NVIDIA copyright header** on all new Python files (except tests).
- **Google Python Style Guide** and **Google Shell Style Guide** are the baseline references.
- **Commit messages**: `[{areas}] {type}: {description}` — e.g. `[model] feat: Add Qwen3 model bridge`. Always sign with `--signoff`.

## Environment Notes

- Python 3.12+ required (`.python-version` pins 3.12).
- The `uv.lock` is Linux-only and x86_64-only. On aarch64, use `uv sync` without `--locked`.
- The upstream project expects container-based development (`docker/Dockerfile.ci`). On Isambard we use bare-metal install instead — see the Isambard section above.
- Delete `nemo_experiments/` before starting fresh training runs to avoid stale checkpoint auto-resume and disk quota issues (checkpoints can be 80+ GB).
- Functional tests are capped at 2 GPUs. Set `CUDA_VISIBLE_DEVICES` explicitly for multi-GPU tests.
- The `pyproject.toml` nullifies torch/torchvision/triton via `sys_platform == 'never'` overrides — these must be installed separately before `uv sync`.
- CUDA extension packages (transformer-engine, mamba-ssm, causal-conv1d, nv-grouped-gemm, flash-linear-attention) are listed under `[tool.uv] no-build-isolation-package` and must be built from source on aarch64 with `CUDAHOSTCXX=/usr/bin/g++-12`.
