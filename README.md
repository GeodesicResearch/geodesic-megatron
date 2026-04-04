# Megatron Bridge on Isambard

This is our fork of [NeMo Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) configured for bare-metal training on Isambard's ARM GH200 cluster. The upstream README is preserved in [README_DEFAULT.md](README_DEFAULT.md).

## Cluster Overview

- **GPUs**: NVIDIA GH200 120GB (95GB usable), `sm_90`, 4 GPUs per node
- **CPU**: ARM aarch64 (Grace)
- **Networking**: Slingshot/CXI fabric (HPE)
- **CUDA**: 12.6, **Python**: 3.12, **PyTorch**: 2.11.0+cu126
- **Max reliable scale**: 32 nodes (128 GPUs). 64+ node runs hang due to Slingshot NCCL timeouts.

## Quick Start

If the environment is already installed (it is for the current workspace), you only need three steps:

```bash
# 1. Activate
source activate_env.sh

# 2. Validate (on a compute node with GPU)
python validate_install.py --run-training

# 3. Submit an SFT job
isambard_sbatch train_nemotron_sft.sbatch configs/nemotron_nano_dolci_instruct_sft.yaml nano
```

Monitor jobs:

```bash
squeue -u $USER
tail -f logs/slurm/nemotron-sft-<JOB_ID>.out
```

## Environment Setup (Full Install)

This section documents the full bare-metal install procedure for Isambard's ARM aarch64 nodes. The upstream project assumes NVIDIA containers — none of that works here. All steps below must be run **on a compute node with a GPU** (not the login node).

### Installed versions (verified working)

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

### Step 1: Create the venv and install PyTorch

```bash
module purge
module load PrgEnv-cray
module load cuda/12.6

python3.12 -m venv .venv
source .venv/bin/activate

# CRITICAL: Use pip, not uv pip, for PyTorch on aarch64.
# uv pip install silently fails with PyTorch wheel indexes on ARM.
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu126 "torch>=2.6.0"
```

### Step 2: Set compiler and library paths

These must be set before building any CUDA extensions:

```bash
# System gcc 7.5 lacks C++17 <filesystem>. nvcc needs gcc-12.
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

# NCCL headers/libs from the venv (not system)
SITE_PACKAGES=".venv/lib/python3.12/site-packages"
export NVTE_NCCL_INCLUDE="$SITE_PACKAGES/nvidia/nccl/include"
export NVTE_NCCL_LIB="$SITE_PACKAGES/nvidia/nccl/lib"
export CPLUS_INCLUDE_PATH="$NVTE_NCCL_INCLUDE:$SITE_PACKAGES/nvidia/cudnn/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$CPLUS_INCLUDE_PATH"
```

### Step 3: Install build prerequisites and sync dependencies

```bash
# pybind11 must be installed before uv sync — TE needs it but doesn't declare it properly
uv pip install pybind11 numpy Cython ninja setuptools

# uv.lock is x86_64-only. On aarch64, omit --locked to let uv re-resolve.
uv sync
```

### Step 4: Build Transformer Engine from source

TE expects cuDNN headers in PyTorch's include directory:

```bash
# Symlink cuDNN headers into torch's include dir
TORCH_INCLUDE="$SITE_PACKAGES/torch/include"
for f in $SITE_PACKAGES/nvidia/cudnn/include/*.h; do
    ln -sf "$f" "$TORCH_INCLUDE/$(basename "$f")"
done

# Build TE (uses the pinned commit in pyproject.toml)
uv pip install --no-build-isolation transformer-engine
```

### Step 5: Build other CUDA extension packages

These are listed under `[tool.uv] no-build-isolation-package` in `pyproject.toml` and must be built from source on aarch64:

```bash
uv pip install --no-build-isolation mamba-ssm causal-conv1d nv-grouped-gemm
```

### Step 6: Apply the sm_90a monkeypatch

GH200 GPUs report `sm_90a` but PyTorch's `_get_cuda_arch_flags()` can't parse the `a` suffix. A `sitecustomize.py` patches this at import time:

Create `.venv/lib/python3.12/site-packages/sitecustomize.py`:

```python
"""GH200 sm_90a fix - Monkeypatch PyTorch's CUDA arch flag detection."""
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

def _patch_pytorch_cuda_arch():
    try:
        import torch.utils.cpp_extension as cpp_ext
        def _patched_get_cuda_arch_flags(cflags=None):
            return ['-gencode', 'arch=compute_90,code=sm_90']
        cpp_ext._get_cuda_arch_flags = _patched_get_cuda_arch_flags
    except (ImportError, AttributeError):
        pass

_patch_pytorch_cuda_arch()
```

### Step 7: Apply the wandb SLURM patch

wandb's `isatty()` check fails under SLURM. Fix:

```bash
sed -i 's/os.isatty(sys.stdout.fileno())/False/' \
    .venv/lib/python3.12/site-packages/wandb/errors/term.py
```

### Step 8: Set LD_PRELOAD for NCCL

The system NCCL is too old — torch needs `ncclCommShrink` which only exists in newer versions. The venv's bundled NCCL must be force-loaded:

```bash
export NCCL_LIBRARY=".venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"
```

This is handled automatically by `activate_env.sh` at runtime.

### Step 9: Validate

```bash
source activate_env.sh
python validate_install.py --run-training
```

This runs 15 checks: Python imports, CUDA ops, GPU memory, recipe loading, and a tiny training run.

### Key environment files

| File | Purpose |
|------|---------|
| `activate_env.sh` | Sources venv, sets `LD_PRELOAD`, compiler vars, NVIDIA library paths. **Source this before any work.** |
| `validate_install.py` | 15-check validation script (imports, CUDA, GPU ops, recipes, training) |
| `.venv/lib/.../sitecustomize.py` | sm_90a monkeypatch (loaded automatically by Python) |

## Data Preparation

The training pipeline can handle most data prep automatically, but on Isambard some steps must be done ahead of time. Here's what's automatic and what's not:

| Step | Automatic? | Why we do it manually on Isambard |
|------|-----------|-----------------------------------|
| HF dataset download | Yes (`HFDatasetBuilder` auto-downloads) | Compute nodes have no internet. SBATCH scripts set `TRANSFORMERS_OFFLINE=1`. |
| JSONL generation | Yes (auto-converts HF dataset to `training.jsonl`/`validation.jsonl`) | Fully automatic. `rewrite: false` in config skips regeneration if files exist. |
| Sequence packing | Yes (auto-packs if packed parquet files don't exist) | Works but blocks rank 0 for 1-4 hours while all other GPUs sit idle. Offline packing avoids wasting GPU-hours. |
| Checkpoint conversion | **No** — always expects pre-converted Megatron format | Must be done manually with `convert_checkpoints_multi_gpu.py`. |

### Step 1: Download the HF dataset (required — no internet on compute nodes)

On a login node (which has internet access):

```bash
source activate_env.sh

python -c "
from datasets import load_dataset
ds = load_dataset('allenai/Dolci-Instruct-SFT', cache_dir='/projects/a5k/public/hf')
ds.save_to_disk('/projects/a5k/public/data/allenai__Dolci-Instruct-SFT')
"
```

For any new dataset, follow the same pattern: download via `load_dataset()` and save to `/projects/a5k/public/data/<org>__<dataset_name>/`. The JSONL conversion (training.jsonl, validation.jsonl) happens automatically during the first training run via `HFDatasetBuilder`.

### Step 2: Pack the dataset offline (recommended — saves GPU-hours)

Packing combines multiple short examples into fixed-length sequences for efficient GPU utilization. The training pipeline will do this automatically if packed data doesn't exist, but it runs **single-threaded on rank 0** while all other GPUs wait. For large datasets this wastes 1-4 hours of multi-node GPU time.

To pack offline instead:

```bash
# Submit as a single-GPU job
isambard_sbatch pack_dataset.sbatch
```

Or run directly on a compute node:

```bash
python scripts/data/pack_sft_dataset.py \
    --dataset-root /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
    --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --seq-length 8192 \
    --pad-seq-to-mult 1     # Set to 2*CP if using context parallelism (e.g., 4 for CP=2)
```

Output goes to `<dataset-root>/packed/<tokenizer-name>_pad_seq_to_mult<N>/`:
- `training_8192.idx.parquet` — Packed training sequences
- `validation_8192.idx.parquet` — Packed validation sequences
- `8192_metadata.jsonl` — Packing statistics (use for `train_iters` calculation)

The packing script:
- Applies the tokenizer's chat template (`use_hf_tokenizer_chat_template: true`)
- Computes loss masks for answer-only training (`answer_only_loss: true`)
- Concatenates short examples into `seq_length`-sized packs
- Uses `num_tokenizer_workers=1` (higher values can OOM shared memory on large datasets)

**If packed data already exists**, both the offline script and the training pipeline skip it. Delete the parquet files to re-pack.

### Step 3: Convert the HF checkpoint to Megatron format (required — no auto-conversion)

The finetune scripts expect a pre-converted Megatron checkpoint — there is no automatic HF-to-Megatron conversion during training. The conversion TP/EP don't need to match training TP/EP (the training script reshards as needed), but matching avoids resharding overhead at startup.

**Nemotron 3 Nano** (single-node, fits in 4 GPUs):

```bash
python -m torch.distributed.run --nproc_per_node=4 \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --megatron-path /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --tp 1 --ep 1 \
    --trust-remote-code
```

**Nemotron 3 Super** (multi-node, needs EP=16 to fit in memory during conversion):

```bash
isambard_sbatch convert_super.sbatch
```

This runs on 4 nodes (16 GPUs) with `--tp 1 --ep 16`. The conversion script CLI:

```
convert_checkpoints_multi_gpu.py import
    --hf-model MODEL_ID      # HuggingFace model ID or local path
    --megatron-path OUT_DIR   # Where to save the Megatron checkpoint
    --tp N                    # Tensor parallelism (default: 1)
    --ep N                    # Expert parallelism (default: 1)
    --pp N                    # Pipeline parallelism (default: 1)
    --trust-remote-code       # Required for Nemotron models
```

To convert **back** from Megatron to HuggingFace (e.g., for evaluation):

```
convert_checkpoints_multi_gpu.py export
    --hf-model MODEL_ID      # Original HF model (for config/tokenizer)
    --megatron-path IN_DIR    # Megatron checkpoint directory
    --hf-path OUT_DIR         # Where to save the HF model
    --distributed-save        # Each rank saves independently (less memory)
```

### Already-converted checkpoints

These are ready to use in training configs:

```
/projects/a5k/public/checkpoints/megatron_bridges/models/
    NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/
    NVIDIA-Nemotron-3-Super-120B-A12B-BF16/
```

Referenced via `checkpoint.pretrained_checkpoint` in the YAML config.

### Writing a new YAML config

To train on a new dataset, copy the production config and update these fields:

```bash
cp configs/nemotron_nano_dolci_instruct_sft.yaml configs/my_new_sft.yaml
```

Fields to change:

```yaml
dataset:
  dataset_name: your-org/Your-Dataset          # HF dataset ID (for metadata)
  dataset_root: /projects/a5k/public/data/your-org__Your-Dataset  # Local path
  seq_length: 8192                              # Must match packing seq_length

train:
  train_iters: ???   # = total_tokens / (global_batch_size * seq_length)

checkpoint:
  save: /projects/a5k/public/checkpoints/megatron/my_new_sft  # Output path

logger:
  wandb_exp_name: my_new_sft                    # W&B run name
```

## SFT Training Pipeline

### How it works

1. A Python recipe (`nemotron_3_nano_sft_config()` or `nemotron_3_super_sft_config()`) defines the base model config, optimizer, parallelism, and data pipeline.
2. A YAML config file overrides recipe defaults (parallelism, dataset, LR, etc.).
3. CLI Hydra-style overrides can further modify any parameter.
4. The finetune script loads the HF checkpoint from disk, converts it to Megatron format in-memory, and starts distributed training.

### File structure

| File | Purpose |
|------|---------|
| `train_nemotron_sft.sbatch` | SLURM launcher: modules, env vars, Slingshot config, `torchrun` |
| `examples/models/nemotron_3/nano/finetune_nemotron_3_nano.py` | Nano finetune entry point |
| `examples/models/nemotron_3/super/finetune_nemotron_3_super.py` | Super finetune entry point |
| `configs/nemotron_nano_dolci_instruct_sft.yaml` | Production Nano SFT config (Dolci dataset) |
| `configs/grid_search/` | All parallelism grid search configs |
| `experiments.md` | Full grid search results and analysis |
| `activate_env.sh` | Environment activation (source before any work) |
| `validate_install.py` | Installation validation script |
| `pack_dataset.sbatch` | SLURM job for offline dataset packing |
| `convert_super.sbatch` | SLURM job for Nemotron Super HF-to-Megatron conversion |
| `scripts/data/pack_sft_dataset.py` | Offline tokenization + packing script |
| `examples/conversion/convert_checkpoints_multi_gpu.py` | Multi-GPU HF<->Megatron checkpoint converter |

### YAML config structure

Configs override the Python recipe defaults. The key sections:

```yaml
dataset:
  dataset_name: allenai/Dolci-Instruct-SFT     # HuggingFace dataset ID
  dataset_root: /projects/a5k/public/data/...   # Local pre-downloaded path
  seq_length: 8192
  packed_sequence_specs:
    packed_sequence_size: 8192
    pad_seq_to_mult: 1          # Set to 2*CP if using context parallelism
  dataset_kwargs:
    chat: true                  # Enables generic chat messages processor
    use_hf_tokenizer_chat_template: true
    answer_only_loss: true      # Loss only on assistant turns

train:
  global_batch_size: 64
  micro_batch_size: 1
  train_iters: 2668             # See "Calculating train_iters" below

model:
  seq_length: 8192
  tensor_model_parallel_size: 2
  expert_model_parallel_size: 8
  context_parallel_size: 1
  sequence_parallel: True
  # ... (see Optimal Settings below)

optimizer:
  lr: 5.0e-06
  weight_decay: 0.1
  use_distributed_optimizer: true
  bf16: true

checkpoint:
  pretrained_checkpoint: /projects/a5k/public/checkpoints/megatron_bridges/models/...
  save: /projects/a5k/public/checkpoints/megatron/...
  save_interval: 200

logger:
  wandb_entity: geodesic
  wandb_project: megatron_training
  wandb_exp_name: your_experiment_name
```

### Calculating `train_iters`

Use exact token counts, not rough estimates:

```
train_iters = total_tokens_in_dataset / tokens_per_batch
tokens_per_batch = global_batch_size * seq_length
```

For packed-sequence SFT, use packing metadata (packing factor, pack counts from parquet files) to derive the total token count. Extrapolate from subset packing stats if the full dataset hasn't been packed yet.

### Preparing data and checkpoints

Datasets must be downloaded, packed, and checkpoints converted **before** submitting training jobs. See the [Data Preparation](#data-preparation) section below for the full procedure.

## Optimal Parallelism Settings

These settings were validated through a grid search of 25+ configurations. See `experiments.md` for the full results.

### Nemotron 3 Nano (30B-A3B, 128 experts)

**Recommended: TP=2, EP=8, seq=8192, 32 nodes**

| Setting | Value | Why |
|---------|-------|-----|
| `tensor_model_parallel_size` | 2 | 3B active params need minimal sharding. TP=4 adds unnecessary all-reduce overhead. |
| `expert_model_parallel_size` | 8 | 16 experts/GPU. Saves 22GB vs EP=4 and is faster (19.8 vs 16.6 TFLOP/s). |
| `context_parallel_size` | 1 | Not needed at seq=8192. Only use CP=2 for seq=16384. |
| `sequence_parallel` | True | Required when TP>1 and EP>1. |
| `recompute_granularity` | selective | Full recompute is 100x slower. |
| `recompute_modules` | `["core_attn"]` | Only recompute attention (O(seq^2) memory, cheap to redo). |
| `gradient_accumulation_fusion` | False | APEX not available on Isambard. |
| `moe_token_dispatcher_type` | alltoall | DeepEP not available. |
| `moe_permute_fusion` | True | Critical — without it, TE's unfused path dominates iteration time. |
| `moe_grouped_gemm` | True | Without it, experts run sequentially (5x slower). |

**Performance at 32 nodes (128 GPUs):**
- seq=8192: **2.0s/iter**, 36.7 TFLOP/s/GPU, 51 GB peak (44 GB headroom)
- seq=16384 (add CP=2): **12.3s/iter**, 15.8 TFLOP/s/GPU, 51 GB peak

### Nemotron 3 Super (120B-A12B, 512 experts)

**Preliminary: TP=4, EP=64, CP=2, seq=16384, 32 nodes**

| Setting | Value | Why |
|---------|-------|-----|
| `tensor_model_parallel_size` | 4 | 12B active params need more sharding than Nano. |
| `expert_model_parallel_size` | 64 | 512 experts / 64 = 8 experts/GPU. EP<64 OOMs. |
| `context_parallel_size` | 2 | Required at seq=16384 — CP=1 OOMs (92 GB). |
| `mtp_num_layers` | 0 | MTP + CP + packed sequences triggers an MCore bug. |

**Performance**: ~82s/iter (with 4 gradient accumulation steps), 72 GB peak.

### Scaling Guide (Nano)

| Nodes | GPUs | TP | EP | DP | GBS (no grad accum) | Iter time | Status |
|-------|------|----|----|----|---------------------|-----------|--------|
| 8 | 32 | 2 | 8 | 16 | 16 | ~8s | Communication-bound |
| 16 | 64 | 1 | 8 | 64 | 64 | ~5s | Highest compute efficiency |
| 32 | 128 | 2 | 8 | 64 | 64 | ~2s | **Recommended** |
| 64+ | — | — | — | — | — | — | Unreliable (Slingshot timeouts) |

### Key Learnings

- **EP=8 > EP=4**: More expert sharding = less memory + better throughput, even cross-node on Slingshot.
- **TP=2 > TP=4** for Nano: 3B active params don't benefit from heavier tensor sharding.
- **Selective recompute >> full recompute**: 10x throughput difference. Only recompute `core_attn`.
- **2 grad accum steps are nearly free**: The second micro-step's compute hides the DP all-reduce communication.
- **Recipe LR (5e-6) >> high LR (8e-5)**: Prevents NaN, especially with context parallelism.
- **`weight_decay: 0.1`** (recipe default) trains better than 0.01.

## Isambard-Specific Issues and Workarounds

### Required SBATCH environment variables

All of these are already set in `train_nemotron_sft.sbatch`, but if you write a new launcher:

```bash
# Slingshot/CXI NCCL (required for multi-node)
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi
export NCCL_SOCKET_IFNAME=hsn
export NCCL_PROTO=^LL128

# Isambard workarounds
export UB_SKIPMC=1                                    # Disable CUDA Multicast (unsupported)
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID  # Node-local Triton cache (avoid NFS race)
export TMPDIR=/tmp/megatron_$SLURM_JOB_ID             # Node-local temp dir
export HF_HOME=/projects/a5k/public/hf                # Shared HF cache
export TRANSFORMERS_OFFLINE=1                          # No network fetches at runtime
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce CUDA memory fragmentation
export CUDA_DEVICE_MAX_CONNECTIONS=1                   # Required for TP/SP overlap in TE

# Module loading
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1
```

### ARM/aarch64 install workarounds (summary)

If the environment breaks or needs to be rebuilt, see the [Environment Setup](#environment-setup-full-install) section above for the full step-by-step procedure. The critical issues in brief:

1. **Use `pip` for PyTorch** — `uv pip install` silently fails with PyTorch wheel indexes on aarch64.
2. **`LD_PRELOAD` the venv's NCCL** — System NCCL lacks `ncclCommShrink`.
3. **`CUDAHOSTCXX=/usr/bin/g++-12`** — System gcc 7.5 lacks C++17 `<filesystem>`.
4. **`sitecustomize.py` monkeypatch** — GH200's `sm_90a` suffix breaks PyTorch.
5. **cuDNN header symlinks** — Transformer Engine expects them in torch's include dir.
6. **`uv sync` without `--locked`** — `uv.lock` is x86_64-only.
7. **`pybind11` pre-install** — Must be installed before `uv sync` (TE build dep).
8. **wandb `isatty()` patch** — Prevents crash under SLURM's non-TTY environment.

### Common pitfalls

| Problem | Fix |
|---------|-----|
| `RuntimeError: ...gradient_accumulation_fusion...fused_weight_gradient_mlp_cuda...not found` | Set `model.gradient_accumulation_fusion: False` (no APEX) |
| NaN loss at iteration 7-8 | Lower LR to 5e-6 (recipe default). 8e-5 is unstable with CP. |
| `OSError: [Errno 116] Stale file handle` | Set `TRITON_CACHE_DIR` and `TMPDIR` to node-local `/tmp` |
| Jobs hang at 64+ nodes | Stay at 32 nodes max. This is a Slingshot infrastructure issue. |
| OOM with Nemotron Super | Need EP>=64 (512 experts). Disable MTP (`mtp_num_layers: 0`). |
| `nemo_experiments/` fills disk | Delete between runs: `rm -rf nemo_experiments NeMo_experiments` |
| HF downloads fail at scale | Pre-download to `/projects/a5k/public/data/`, set `TRANSFORMERS_OFFLINE=1` |

## Disk Locations

| What | Path |
|------|------|
| This repo | `/home/a5k/kyleobrien.a5k/geodesic-megatron` |
| Pre-downloaded HF datasets | `/projects/a5k/public/data/` |
| Converted Megatron checkpoints | `/projects/a5k/public/checkpoints/megatron_bridges/models/` |
| Training output checkpoints | `/projects/a5k/public/checkpoints/megatron/` |
| SLURM logs | `logs/slurm/` (relative to repo) |
| W&B logs | `/projects/a5k/public/logs/wandb` |
| HF cache | `/projects/a5k/public/hf` |
| Shared temp | `/projects/a5k/public/tmp` |

## Further Reading

- [experiments.md](experiments.md) — Full grid search results (25+ configs, Nano and Super)
- [CLAUDE.md](CLAUDE.md) — Detailed install procedure, ARM workarounds, and dev commands
- [README_DEFAULT.md](README_DEFAULT.md) — Upstream Megatron Bridge README (supported models, API docs, etc.)
