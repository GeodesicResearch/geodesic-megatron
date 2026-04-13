# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

NeMo Megatron Bridge is an NVIDIA PyTorch-native library that provides a bridge, conversion, and verification layer between HuggingFace and [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core). It enables bidirectional checkpoint conversion, pretraining, SFT, and LoRA for LLM and VLM models with Megatron Core's parallelism (tensor, pipeline, expert parallelism, FP8/BF16 mixed precision).

The primary package is `megatron.bridge` under `src/`. Megatron-Core is pinned as a git submodule at `3rdparty/Megatron-LM`.

## Pipelines

All top-level scripts follow the `PIPELINE_ACTION.ext` naming convention. There are four pipelines:

| Pipeline | Submit (SLURM) | Launch / Logic | Purpose |
|----------|---------------|----------------|---------|
| **env** | `env_submit.sbatch` | `env_activate.sh`, `env_setup.sh`, `env_validate.py` | Environment install, activation, validation |
| **training** | `training_submit.sbatch` | `training_launch.sh` | SFT and CPT distributed training |
| **data** | `data_submit.sbatch` | `data_prepare.py` | Dataset download, tokenization, packing |
| **checkpoint** | `checkpoint_submit.sbatch` | `checkpoint_convert.sh`, `checkpoint_convert_hf.py` | Megatron↔HF conversion, Hub upload |

Each pipeline has a thin `PIPELINE_submit.sbatch` for SLURM allocation and a `.sh`/`.py` with the actual logic. The `.sh` launchers can also be called directly from an interactive `salloc`.

---

## 1. Environment Pipeline (`env_*`)

### Files

| File | Purpose |
|------|---------|
| `env_activate.sh` | Universal environment: venv, compilers, NVIDIA libs, GPU settings, cache paths. **Source this before any work.** |
| `env_setup.sh` | Full bare-metal install script (must run on a compute node with GPU) |
| `env_validate.py` | 15-check validation (imports, CUDA, GPU ops, recipes, training) |
| `env_submit.sbatch` | SLURM wrapper for setup/validation (needs GPU) |

### Usage

```bash
# Activate (from any node)
source env_activate.sh

# Install from scratch (requires compute node)
isambard_sbatch env_submit.sbatch setup

# Validate
isambard_sbatch env_submit.sbatch validate --run-training
```

### What `env_activate.sh` sets

**Universal GPU settings** (needed for any operation):
- `UB_SKIPMC=1` — Disables CUDA Multicast (Isambard driver doesn't support it)
- `CUDA_DEVICE_MAX_CONNECTIONS=1` — Required for TP/SP comm-compute overlap
- `NVTE_CPU_OFFLOAD_V1=1` — TE activation offloading V1 code path
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — Reduces CUDA memory fragmentation

**Shared cache paths:**
- `HF_HOME=/projects/a5k/public/hf`
- `WANDB_DIR=/projects/a5k/public/logs/wandb`
- `NEMO_HOME=/projects/a5k/public/data/nemo_cache`

Every env var in `env_activate.sh` has detailed inline documentation.

### Installed Versions (verified working)

- **torch 2.11.0+cu126** (aarch64 wheel)
- **transformer-engine 2.14.0** (built from pinned commit `71bbefbf`)
- **mamba-ssm 2.3.1** and **causal-conv1d 1.6.1** (built from source)
- **nv-grouped-gemm 1.1.4** (built from source)
- **Python 3.12**, **CUDA 12.6**, **NCCL 2.28.9** (from venv, not system)
- **nvidia-resiliency-ext 0.5.0** (`ft_launcher`, fault tolerance, straggler detection)

### ARM/Isambard-Specific Workarounds

These are critical issues that were discovered and fixed. If the environment breaks or needs rebuilding, all must be applied:

1. **PyTorch install: use `pip`, not `uv pip`**. `uv pip install` silently fails with PyTorch wheel indexes on aarch64.
2. **NCCL LD_PRELOAD** (fixes `undefined symbol: ncclCommShrink`). The system NCCL is older than what torch needs.
3. **CUDAHOSTCXX=/usr/bin/g++-12** (fixes `fatal error: filesystem: No such file or directory`). System gcc 7.5 lacks C++17.
4. **NCCL include path for TE build** (fixes `fatal error: nccl.h: No such file or directory`).
5. **cuDNN header symlinks** for TE build.
6. **pybind11 must be installed before `uv sync`**.
7. **sitecustomize.py monkeypatch** (fixes `ValueError: invalid literal for int() with base 10: '90a'`).
8. **`uv sync` without `--locked`**. The `uv.lock` is x86_64-only.
9. **wandb isatty() patch** for SLURM.

---

## 2. Training Pipeline (`training_*`)

### Files

| File | Purpose |
|------|---------|
| `training_launch.sh` | Shared launcher: NCCL/CXI env vars, fault tolerance, srun + ft_launcher |
| `training_submit.sbatch` | Thin SLURM wrapper: allocates nodes, calls `training_launch.sh` |

Training scripts (called by the launcher):
- `examples/models/nemotron_3/nano/finetune_nemotron_3_nano.py` — Nano SFT
- `examples/models/nemotron_3/nano/midtrain_nemotron_3_nano.py` — Nano CPT
- `examples/models/nemotron_3/super/finetune_nemotron_3_super.py` — Super SFT
- `examples/models/nemotron_3/super/midtrain_nemotron_3_super.py` — Super CPT

### Usage

```bash
# Via SLURM (allocates nodes)
isambard_sbatch --nodes=32 training_submit.sbatch configs/<config>.yaml nano sft
isambard_sbatch --nodes=8  training_submit.sbatch configs/<config>.yaml nano cpt

# Via salloc (interactive)
salloc --nodes=16 --gpus-per-node=4 --time=24:00:00 --exclusive
bash training_launch.sh configs/<config>.yaml --model nano --mode sft
bash training_launch.sh configs/<config>.yaml --model super --mode cpt --max-samples 50000
bash training_launch.sh configs/<config>.yaml --model nano --mode sft --nodes 8 --nodelist node[001-008]
bash training_launch.sh configs/<config>.yaml --model nano --mode sft --disable-ft
bash training_launch.sh configs/<config>.yaml --model nano --mode sft --peft lora
```

`training_launch.sh` options: `--model nano|super` (required), `--mode sft|cpt` (required), `--disable-ft`, `--enable-pao`, `--peft lora`, `--max-samples N`, `--nodes N`, `--nodelist LIST`.

### Environment Variable Architecture

`training_launch.sh` adds distributed-training-only vars on top of `env_activate.sh`:
- All Slingshot/CXI NCCL vars (`NCCL_NET`, `FI_PROVIDER`, `FI_CXI_*`, etc. — 30+ vars)
- Fault tolerance vars (`TORCH_NCCL_TIMEOUT`, `TORCH_NCCL_RETHROW_CUDA_ERRORS`)
- Job-specific node-local paths (`TRITON_CACHE_DIR`, `TMPDIR`, `MEGATRON_CONFIG_LOCK_DIR`)
- Module loading (`PrgEnv-cray`, `cuda/12.6`, `brics/aws-ofi-nccl/1.8.1`)

Every env var has detailed inline documentation.

### Training-Specific Override for Isambard

The YAML config override `model.gradient_accumulation_fusion=False` is required for all training. The default `True` requires APEX which is not installed.

### Fault Tolerance

Slingshot/CXI causes intermittent NCCL collective hangs (~every 2-3 hours with EP=8 cross-node). The training pipeline uses a layered resilience stack:

1. **In-process restart** (60s/90s timeout) — reinitializes NCCL, retries same step. Zero iterations lost.
2. **ft_launcher job restart** (`--max-restarts=20`) — kills workers, reloads from latest checkpoint. ≤25 iters lost.
3. **NCCL watchdog** (900s) — last resort backup.

**ft_launcher timeout configuration** (set in `training_launch.sh`):
- `--ft-rank-section-timeouts=setup:1800,step:3600,checkpointing:600`
- `--ft-rank-out-of-section-timeout=3600` — must be ≥3600s for first-iter NCCL lazy init with PP=8+
- `calc_ft_timeouts=True` auto-learns step timeouts after first successful run. **Delete `ft_state.json`** from checkpoint dir if learned timeouts are too aggressive after config changes.

The `ft`/`nvrx_straggler`/`inprocess_restart` Python configs **cannot** be set via YAML or Hydra overrides (OmegaConf merge creates dicts, not dataclasses). They are set in `finetune_nemotron_3_nano.py` via the `--enable-ft` flag (on by default). Use `--disable-ft` to opt out.

### Nemotron 3 Nano (30B-A3B) on Isambard

Recommended parallelism for GH200 95GB GPUs:
- **8 nodes, 32 GPUs**: TP=2, EP=2, PP=4, DP=2 (node-local TP+EP)
- Throughput: **~3.4s/iter, ~27 TFLOP/s/GPU** at seq_length=8192, GBS=16
- Zero NCCL hangs through 500+ iterations — keeping EP on NVLink avoids Slingshot all-to-all hangs

**Why node-local TP+EP matters:** Cross-node EP drops throughput 14x because MoE all-to-all over Slingshot/CXI is extremely slow. Rule: **TP × EP ≤ 4** to keep both on NVLink.

### Nemotron 3 Super (120B-A12B) on Isambard

**Best tested (BF16): 32 nodes, 128 GPUs**: TP=4, EP=8, PP=4, DP_pure=1
- ~82-90s/iter, 3.5-3.7 TFLOP/s/GPU
- EP=8 crosses nodes (unavoidable for Super — 512 experts). Slingshot hangs ~every 2-3 hours, recovered by ft_launcher.
- **Recommendation: use BF16 for Super.** FP8 causes stochastic alignment crashes in MoE routing.

**Node-local EP=4 alternative (Parallel Folding):** TP=4, EP=4, PP=8, expert_tensor_parallel_size=1. Eliminates hangs but ~124s/iter (PP=8 pipeline bubble).

### Parallel Folding (expert_tensor_parallel_size)

```yaml
tensor_model_parallel_size: 4       # Attention: 4-way TP
expert_model_parallel_size: 4       # Experts: 4-way EP
expert_tensor_parallel_size: 1      # Experts NOT sharded by TP → enables folding
```

Keeps EP all-to-all on NVLink while using high TP for attention. Only PP crosses Slingshot.

### TensorBoard on NFS

Set `tensorboard_dir: /tmp/tb_logs` in each config. Also `tensorboard_log_interval: 999999` (not 0 — ZeroDivisionError). Multiple runs sharing NFS TB logs causes cascading stale file handle crashes.

---

## 3. Data Pipeline (`data_*`)

### Files

| File | Purpose |
|------|---------|
| `data_prepare.py` | Download HF datasets, tokenize, export JSONL, pack sequences |
| `data_submit.sbatch` | SLURM wrapper for offline packing (1 node, 1 GPU) |

### Usage

```bash
# Prepare dataset (download + tokenize + pack)
python data_prepare.py --dataset allenai/Dolci-Instruct-SFT --seq-length 8192

# Offline packing only (via SLURM)
isambard_sbatch data_submit.sbatch \
  /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 8192 1

# From salloc
source env_activate.sh
python scripts/data/pack_sft_dataset.py \
  --dataset-root /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
  --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --seq-length 8192 --pad-seq-to-mult 1
```

### What's automatic vs. manual

| Step | Automatic? | Notes |
|------|-----------|-------|
| HF dataset download | Yes | `HFDatasetBuilder` auto-downloads via `HF_HOME` shared cache |
| JSONL generation | Yes | Auto-converts HF dataset to `training.jsonl`/`validation.jsonl` |
| Sequence packing | Yes, but slow | Blocks rank 0 for 1-4 hours. Offline packing via `data_submit.sbatch` saves GPU-hours |
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
| `checkpoint_convert.sh` | Shared launcher: env setup, NCCL, srun+torchrun. Modes: `export`, `import`, `upload-all` |
| `checkpoint_convert_hf.py` | Python conversion logic (the script torchrun executes on each GPU rank) |
| `checkpoint_submit.sbatch` | Thin SLURM wrapper (2 nodes default, override with `--nodes`) |

### Usage

```bash
# Export Megatron → HF
isambard_sbatch checkpoint_submit.sbatch export \
  /projects/a5k/public/checkpoints/megatron/<experiment> --iteration 300 --push-to-hub

# Import HF → Megatron (4 nodes for Super)
isambard_sbatch --nodes=4 checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Upload all iterations (with polling for ongoing training)
isambard_sbatch --time=24:00:00 checkpoint_submit.sbatch upload-all \
  /projects/a5k/public/checkpoints/megatron/<experiment> --poll

# From salloc
bash checkpoint_convert.sh export /path/to/ckpts --iteration 300 --push-to-hub
```

### How export works

1. Reads `latest_checkpointed_iteration.txt` or `--iteration N` to find the `iter_XXXXXXX` directory
2. Auto-detects the HF model ID from `run_config.yaml`
3. Converts via `AutoBridge.from_hf_pretrained` + `load_megatron_model` + `save_hf_pretrained` (multi-GPU via torchrun)
4. Saves to `<megatron-path>/iter_XXXXXXX/hf/`
5. Optionally pushes to HuggingFace Hub on a revision branch (`iter_0000300`)

The `torch_dist` checkpoint format supports resharding — conversion parallelism is independent of training parallelism.

### Known limitations

- **MTP expert shards missing**: Shards 49-50 of 50 not written (megatron-bridge bug). 48/50 output is fully functional for inference.
- **Super requires 2+ nodes**: Single-process is too slow (1.2TB sequential read).
- **Hub uploads are large**: ~223GB per checkpoint, 20-40 min per upload.

### Already-converted checkpoints

```
/projects/a5k/public/checkpoints/megatron_bridges/models/
    NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/
    NVIDIA-Nemotron-3-Super-120B-A12B-BF16/
```

---

## Running Evals (sfm-evals repo)

Evals are run via the [sfm-evals](https://github.com/GeodesicResearch/sfm-evals) repo at `/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals`. Uses Inspect AI evals served via vLLM.

**Pre-requisites for new geodesic-research HF models:**
1. Upload `configuration_nemotron_h.py` and `modeling_nemotron_h.py` to the HF repo
2. Fix `tokenizer_config.json`: `"tokenizer_class": "TokenizersBackend"` → `"PreTrainedTokenizerFast"`
3. For 120B+ models: pre-download to shared HF cache first

**Submit evals:**
```bash
cd /lus/lfs1aip2/projects/public/a5k/repos/sfm-evals
isambard_sbatch --time=8:00:00 --gpus-per-node=1 \
  run_bundled_checkpoint_eval.sbatch "geodesic-research/model_name" manifests/eval.json
```

Results: W&B project "Self-Fulfilling Model Organisms - ITERATED Evals" (entity: geodesic).

---

## NCCL Performance Testing

nccl-tests at `/home/a5k/kyleobrien.a5k/nccl-tests/` for benchmarking Slingshot bandwidth:

```bash
source env_activate.sh
export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn
srun --nodes=2 --ntasks-per-node=1 --export=ALL bash -c \
  "source env_activate.sh && /home/a5k/kyleobrien.a5k/nccl-tests/build/all_reduce_perf -b 32K -e 8G -f 2 -g 4"
```

**Measured results (2026-04-12)**: 2-node all_reduce: 191-197 GB/s, 16-node: 255-263 GB/s.

---

## Common Commands

### Package Management
```bash
uv sync                                       # Install deps (no --locked on aarch64)
uv add <package>                              # Add a dependency
```

### Linting and Formatting
```bash
uv run ruff check --fix .
uv run ruff format .
```

### Testing
```bash
uv run pytest tests/unit_tests/ -x -v                  # All unit tests (no GPU)
bash scripts/run_ci_tests.sh                            # Full CI (requires GPU)
```

### Megatron-Core Submodule
```bash
./scripts/switch_mcore.sh status   # Show current pinned commit
./scripts/switch_mcore.sh dev      # Switch to dev
./scripts/switch_mcore.sh main     # Switch to main
```

### Monitoring Long-Running Processes

Always use the **Monitor** tool (not polling loops or sleep):
```bash
tail -f /tmp/training_run.log | grep --line-buffered -E "iteration\s+[0-9]+/|Error|OOM|NCCL|Traceback|saved|completed"
```

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
| SLURM logs | `logs/slurm/` |
| W&B logs | `/projects/a5k/public/logs/wandb` |
| HF cache | `/projects/a5k/public/hf` |

## Common Pitfalls

| Problem | Fix |
|---------|-----|
| `RuntimeError: ...gradient_accumulation_fusion...` | `model.gradient_accumulation_fusion: False` (no APEX) |
| NaN loss at iteration 7-8 | Lower LR to 5e-6. 8e-5 is unstable with CP. |
| `OSError: [Errno 116] Stale file handle` | `TRITON_CACHE_DIR`/`TMPDIR` to node-local `/tmp` (automatic in `training_launch.sh`) |
| NCCL hangs every ~7-8 min | Slingshot fabric issue. ft_launcher auto-restarts. |
| EP=4 OOMs on GH200 | Use EP=8 (16 experts/GPU = 51GB vs 32 = 93GB). |
| `nemo_experiments/` fills disk | Selectively remove old TB logs. **Do NOT `rm -rf`** — contains checkpoint resume state. |
