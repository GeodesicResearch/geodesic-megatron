# Megatron Bridge on Isambard

This is our fork of [NeMo Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) configured for bare-metal training on Isambard's ARM GH200 cluster. The upstream README is preserved in [README_DEFAULT.md](README_DEFAULT.md).

## Cluster Overview

- **GPUs**: NVIDIA GH200 120GB (95GB usable), `sm_90`, 4 GPUs per node
- **CPU**: ARM aarch64 (Grace)
- **Networking**: Slingshot/CXI fabric (HPE)
- **CUDA**: 12.6, **Python**: 3.12, **PyTorch**: 2.11.0+cu126
- **Max reliable scale**: 32 nodes (128 GPUs). 64+ node runs hang due to Slingshot NCCL timeouts.

## Pipelines

All top-level scripts follow the `PIPELINE_ACTION.ext` naming convention. There are four pipelines:

| Pipeline | Submit (SLURM) | Launch / Logic | Purpose |
|----------|---------------|----------------|---------|
| **env** | `env_submit.sbatch` | `env_activate.sh`, `env_setup.sh`, `env_validate.py` | Environment install, activation, validation |
| **training** | `training_submit.sbatch` | `training_launch.sh` | SFT and CPT distributed training |
| **data** | `data_submit.sbatch` | `data_prepare.py` | Dataset download, tokenization, packing |
| **checkpoint** | `checkpoint_submit.sbatch` | `checkpoint_convert.sh`, `checkpoint_convert_hf.py` | Megatron↔HF conversion, Hub upload |

Each `PIPELINE_submit.sbatch` allocates SLURM nodes and delegates to the logic script. The `.sh` launchers can also be called directly from an interactive `salloc` session.

## Quick Start

```bash
# 1. Activate environment
source env_activate.sh

# 2. Validate (on a compute node)
isambard_sbatch env_submit.sbatch validate --run-training

# 3. Prepare data
python data_prepare.py --dataset allenai/Dolci-Instruct-SFT --seq-length 8192

# 4. Import base checkpoint
isambard_sbatch --nodes=4 checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16

# 5. Train
isambard_sbatch --nodes=32 training_submit.sbatch configs/nemotron_nano_dolci_instruct_sft.yaml nano sft

# 6. Export + upload trained checkpoint
isambard_sbatch checkpoint_submit.sbatch export /projects/a5k/public/checkpoints/megatron/my_experiment --push-to-hub
```

Monitor jobs: `squeue -u $USER` and `tail -f logs/slurm/<job-name>-<JOB_ID>.out`

---

## 1. Environment Pipeline

### Setup from scratch

All steps require a compute node with GPU (CUDA kernel compilation). Use `env_submit.sbatch` or `salloc`.

```bash
isambard_sbatch env_submit.sbatch setup
```

Or follow the manual steps below.

#### Installed versions (verified working)

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.12 | Pinned in `.python-version` |
| PyTorch | 2.11.0+cu126 | aarch64 wheel from PyTorch index |
| Transformer Engine | 2.14.0 | Built from source (pinned commit `71bbefbf`) |
| mamba-ssm | 2.3.1 | Built from source |
| causal-conv1d | 1.6.1 | Built from source |
| nv-grouped-gemm | 1.1.4 | Built from source |
| CUDA | 12.6 | System module |
| NCCL | 2.28.9 | From venv pip package, **not** system NCCL |

#### Step 1: Create the venv and install PyTorch

```bash
module purge && module load PrgEnv-cray && module load cuda/12.6
python3.12 -m venv .venv && source .venv/bin/activate

# CRITICAL: Use pip, not uv pip, for PyTorch on aarch64.
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu126 "torch>=2.6.0"
```

#### Step 2: Set compiler and library paths

```bash
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

SITE_PACKAGES=".venv/lib/python3.12/site-packages"
export NVTE_NCCL_INCLUDE="$SITE_PACKAGES/nvidia/nccl/include"
export NVTE_NCCL_LIB="$SITE_PACKAGES/nvidia/nccl/lib"
export CPLUS_INCLUDE_PATH="$NVTE_NCCL_INCLUDE:$SITE_PACKAGES/nvidia/cudnn/include:${CPLUS_INCLUDE_PATH:-}"
```

#### Step 3: Install dependencies

```bash
uv pip install pybind11 numpy Cython ninja setuptools
uv sync  # no --locked on aarch64
```

#### Step 4: Build CUDA extensions from source

```bash
# Symlink cuDNN headers for TE
for f in $SITE_PACKAGES/nvidia/cudnn/include/*.h; do
    ln -sf "$f" "$SITE_PACKAGES/torch/include/$(basename "$f")"
done

uv pip install --no-build-isolation transformer-engine mamba-ssm causal-conv1d nv-grouped-gemm
```

#### Step 5: Apply patches

- **sitecustomize.py** — GH200 sm_90a monkeypatch (see CLAUDE.md for details)
- **wandb isatty()** — `sed -i 's/os.isatty(sys.stdout.fileno())/False/' .venv/.../wandb/errors/term.py`
- **NCCL LD_PRELOAD** — handled automatically by `env_activate.sh`

#### Step 6: Validate

```bash
source env_activate.sh
python env_validate.py --run-training
```

### ARM/aarch64 workarounds summary

1. **Use `pip` for PyTorch** — `uv pip` silently fails on aarch64
2. **`LD_PRELOAD` venv NCCL** — system NCCL lacks `ncclCommShrink`
3. **`CUDAHOSTCXX=/usr/bin/g++-12`** — system gcc 7.5 lacks C++17
4. **`sitecustomize.py`** — GH200 `sm_90a` suffix breaks PyTorch
5. **cuDNN header symlinks** — TE expects them in torch's include dir
6. **`uv sync` without `--locked`** — `uv.lock` is x86_64-only
7. **`pybind11` pre-install** — TE build dep not declared properly
8. **wandb `isatty()` patch** — crashes under SLURM

---

## 2. Training Pipeline

### How it works

1. A Python recipe defines the base model config, optimizer, parallelism, and data pipeline
2. A YAML config file overrides recipe defaults
3. The finetune script loads the HF checkpoint, converts to Megatron in-memory, and starts training

### Usage

```bash
# Via SLURM
isambard_sbatch --nodes=32 training_submit.sbatch configs/<config>.yaml nano sft
isambard_sbatch --nodes=8  training_submit.sbatch configs/<config>.yaml nano cpt

# Via salloc
salloc --nodes=16 --gpus-per-node=4 --time=24:00:00 --exclusive
bash training_launch.sh configs/<config>.yaml --model nano --mode sft
bash training_launch.sh configs/<config>.yaml --model nano --mode sft --disable-ft
bash training_launch.sh configs/<config>.yaml --model nano --mode sft --peft lora
```

### Writing a new YAML config

```bash
cp configs/nemotron_nano_dolci_instruct_sft.yaml configs/my_new_sft.yaml
```

Key fields to change:

```yaml
dataset:
  dataset_name: your-org/Your-Dataset
  dataset_root: /projects/a5k/public/data/your-org__Your-Dataset
  seq_length: 8192
train:
  train_iters: ???   # = total_tokens / (global_batch_size * seq_length)
checkpoint:
  save: /projects/a5k/public/checkpoints/megatron/my_new_sft
logger:
  wandb_exp_name: my_new_sft
```

### Fault Tolerance

Slingshot causes intermittent NCCL hangs. Training uses a layered resilience stack:

| Layer | Timeout | Recovery | Iterations lost |
|-------|---------|----------|----------------|
| **In-process restart** | 60s/90s | Reinitializes NCCL, retries same step | **0** |
| **ft_launcher restart** | 3600s step | Kills workers, reloads from checkpoint | **0-25** |
| **NCCL watchdog** | 900s | Last-resort process kill | N/A |

Pass `--disable-ft` to use plain `torchrun` instead of `ft_launcher`.

### Optimal Parallelism

**Nemotron 3 Nano (30B-A3B)** — Recommended: TP=2, EP=8, 32 nodes

| Setting | Value | Why |
|---------|-------|-----|
| `tensor_model_parallel_size` | 2 | 3B active params need minimal sharding |
| `expert_model_parallel_size` | 8 | 16 experts/GPU, saves 22GB vs EP=4 |
| `recompute_granularity` | selective | Full recompute is 100x slower |
| `recompute_modules` | `["core_attn"]` | Only recompute attention |
| `gradient_accumulation_fusion` | False | APEX not available |
| `moe_permute_fusion` | True | Critical for performance |

**Performance at 32 nodes (128 GPUs):** 2.0s/iter, 36.7 TFLOP/s/GPU

**Nemotron 3 Super (120B-A12B)** — TP=4, EP=8, PP=4, 32 nodes (BF16)

82-90s/iter, 3.5-3.7 TFLOP/s/GPU. Use BF16 — FP8 causes stochastic alignment crashes.

### Key Learnings

- **EP=8 > EP=4**: More expert sharding = less memory + better throughput
- **TP=2 > TP=4** for Nano: 3B active params don't benefit from heavier tensor sharding
- **Selective recompute >> full recompute**: 10x throughput difference
- **Recipe LR (5e-6) >> high LR (8e-5)**: Prevents NaN with context parallelism

---

## 3. Data Pipeline

### Usage

```bash
# Full pipeline: download + tokenize + export + pack
python data_prepare.py --dataset allenai/Dolci-Instruct-SFT --seq-length 8192

# Offline packing only (via SLURM, saves GPU-hours)
isambard_sbatch data_submit.sbatch \
  /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 8192 1
```

Packing combines short examples into fixed-length sequences. The training pipeline auto-packs if needed, but it blocks rank 0 for 1-4 hours. Offline packing avoids wasting multi-node GPU time.

Output: `<dataset-root>/packed/<tokenizer>_pad_seq_to_mult<N>/training_8192.idx.parquet`

---

## 4. Checkpoint Pipeline

### Usage

```bash
# Export Megatron → HF
isambard_sbatch checkpoint_submit.sbatch export /path/to/ckpts --iteration 300 --push-to-hub

# Import HF → Megatron (4 nodes for Super)
isambard_sbatch --nodes=4 checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Upload all iterations + poll for ongoing training
isambard_sbatch --time=24:00:00 checkpoint_submit.sbatch upload-all /path/to/ckpts --poll

# From salloc
bash checkpoint_convert.sh export /path/to/ckpts --iteration 300 --push-to-hub
```

`checkpoint_convert.sh` is the launcher (env/NCCL/srun+torchrun). `checkpoint_convert_hf.py` is the Python logic that runs on each GPU rank.

### Already-converted checkpoints

```
/projects/a5k/public/checkpoints/megatron_bridges/models/
    NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/
    NVIDIA-Nemotron-3-Super-120B-A12B-BF16/
```

### Known limitations

- **MTP expert shards missing**: Shards 49-50 of 50 not written (megatron-bridge bug). 48/50 output works for inference.
- **Super requires 2+ nodes**: Single-process conversion too slow (1.2TB).
- **Hub uploads large**: ~223GB per checkpoint, 20-40 min per upload.

---

## Common Pitfalls

| Problem | Fix |
|---------|-----|
| `RuntimeError: ...gradient_accumulation_fusion...` | `model.gradient_accumulation_fusion: False` (no APEX) |
| NaN loss at iteration 7-8 | Lower LR to 5e-6 (recipe default) |
| `OSError: [Errno 116] Stale file handle` | `TRITON_CACHE_DIR`/`TMPDIR` to `/tmp` (automatic in `training_launch.sh`) |
| NCCL hangs every ~7-8 min | Slingshot fabric issue. ft_launcher auto-restarts |
| EP=4 OOMs on GH200 | Use EP=8 (16 experts/GPU = 51GB vs 32 = 93GB) |
| `nemo_experiments/` fills disk | Remove old TB logs selectively. **Do NOT `rm -rf`** — contains checkpoint state |

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

## Further Reading

- [experiments.md](experiments.md) — Full grid search results (25+ configs, Nano and Super)
- [CLAUDE.md](CLAUDE.md) — Detailed install procedure, ARM workarounds, and dev commands
- [README_DEFAULT.md](README_DEFAULT.md) — Upstream Megatron Bridge README (supported models, API docs, etc.)
