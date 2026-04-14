# Megatron Bridge on Isambard

This is our fork of [NeMo Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) configured for bare-metal training on Isambard's ARM GH200 cluster. The upstream README is preserved in [README_DEFAULT.md](README_DEFAULT.md).

## Cluster Overview

- **GPUs**: NVIDIA GH200 120GB (95GB usable), `sm_90`, 4 GPUs per node
- **CPU**: ARM aarch64 (Grace)
- **Networking**: Slingshot/CXI fabric (HPE)
- **CUDA**: 12.6, **Python**: 3.12, **PyTorch**: 2.11.0+cu126
- **Max reliable scale**: 32 nodes (128 GPUs). 64+ node runs hang due to Slingshot NCCL timeouts.

## Pipelines

All top-level scripts follow the `PIPELINE_ACTION.ext` naming convention. There are five pipelines:

| Pipeline | Submit (SLURM) | Launch / Logic | Purpose |
|----------|---------------|----------------|---------|
| **env** | `pipeline_env_submit.sbatch` | `pipeline_env_activate.sh`, `pipeline_env_setup.sh`, `pipeline_env_validate.py` | Environment install, activation, validation |
| **training** | `pipeline_training_submit.sbatch` | `pipeline_training_launch.sh` | SFT and CPT distributed training |
| **data** | `pipeline_data_submit.sbatch` | `pipeline_data_prepare.py` | Dataset download, tokenization, packing |
| **checkpoint** | `pipeline_checkpoint_submit.sbatch` | `pipeline_checkpoint_convert.sh`, `pipeline_checkpoint_convert_hf.py` | Megatron↔HF conversion, Hub upload |
| **coherence** | `pipeline_coherence_submit.sbatch` | `pipeline_coherence_test.py` | Qualitative generation testing, W&B logging |

Each `PIPELINE_submit.sbatch` allocates SLURM nodes and delegates to the logic script. The `.sh` launchers can also be called directly from an interactive `salloc` session.

## Quick Start

### Via SLURM (from a login node)

Each step submits a job that allocates its own compute nodes:

```bash
# 1. Activate environment
source pipeline_env_activate.sh

# 2. Validate (on a compute node)
isambard_sbatch pipeline_env_submit.sbatch validate --run-training

# 3. Prepare data
python pipeline_data_prepare.py --dataset allenai/Dolci-Instruct-SFT --seq-length 8192

# 4. Import base checkpoint
isambard_sbatch --nodes=4 pipeline_checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16

# 5. Train
isambard_sbatch --nodes=32 pipeline_training_submit.sbatch configs/nemotron_nano_dolci_instruct_sft.yaml nano sft

# 6. Export + upload trained checkpoint
isambard_sbatch pipeline_checkpoint_submit.sbatch export /projects/a5k/public/checkpoints/megatron/my_experiment --push-to-hub
```

Monitor jobs: `squeue -u $USER` and `tail -f logs/slurm/<job-name>-<JOB_ID>.out`

### Via salloc (from within an existing allocation)

Get an interactive allocation first, then call the launch scripts directly:

```bash
salloc --nodes=32 --gpus-per-node=4 --time=24:00:00 --exclusive

# Activate environment
source pipeline_env_activate.sh

# Validate
python pipeline_env_validate.py --run-training

# Prepare data (runs on current node, no srun needed)
python pipeline_data_prepare.py --dataset allenai/Dolci-Instruct-SFT --seq-length 8192

# Import base checkpoint (uses all nodes in the allocation)
bash pipeline_checkpoint_convert.sh import nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16

# Train
bash pipeline_training_launch.sh configs/nemotron_nano_dolci_instruct_sft.yaml --model nano --mode sft

# Export + upload
bash pipeline_checkpoint_convert.sh export /projects/a5k/public/checkpoints/megatron/my_experiment --push-to-hub
```

---

## 1. Environment Pipeline

### Setup and validation

Requires a compute node with GPU. The full install procedure and ARM/aarch64 workarounds are documented in [CLAUDE.md](CLAUDE.md).

```bash
# Full install from scratch
isambard_sbatch pipeline_env_submit.sbatch setup

# Validate existing install
isambard_sbatch pipeline_env_submit.sbatch validate --run-training

# Activate (from any node, before any work)
source pipeline_env_activate.sh
```

---

## 2. Training Pipeline

### How it works

1. A Python recipe defines the base model config, optimizer, parallelism, and data pipeline
2. A YAML config file overrides recipe defaults
3. The finetune script loads the HF checkpoint, converts to Megatron in-memory, and starts training

### Usage

```bash
# Via SLURM
isambard_sbatch --nodes=32 pipeline_training_submit.sbatch configs/<config>.yaml nano sft
isambard_sbatch --nodes=8  pipeline_training_submit.sbatch configs/<config>.yaml nano cpt

# Via salloc
salloc --nodes=16 --gpus-per-node=4 --time=24:00:00 --exclusive
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --disable-ft
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --peft lora
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
python pipeline_data_prepare.py --dataset allenai/Dolci-Instruct-SFT --seq-length 8192

# Offline packing only (via SLURM, saves GPU-hours)
isambard_sbatch pipeline_data_submit.sbatch \
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
isambard_sbatch pipeline_checkpoint_submit.sbatch export /path/to/ckpts --iteration 300 --push-to-hub

# Import HF → Megatron (4 nodes for Super)
isambard_sbatch --nodes=4 pipeline_checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Upload all iterations + poll for ongoing training
isambard_sbatch --time=24:00:00 pipeline_checkpoint_submit.sbatch upload-all /path/to/ckpts --poll

# From salloc
bash pipeline_checkpoint_convert.sh export /path/to/ckpts --iteration 300 --push-to-hub
```

`pipeline_checkpoint_convert.sh` is the launcher (env/NCCL/srun+torchrun). `pipeline_checkpoint_convert_hf.py` is the Python logic that runs on each GPU rank.

### Already-converted checkpoints

```
/projects/a5k/public/checkpoints/megatron_bridges/models/
    NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/
    NVIDIA-Nemotron-3-Super-120B-A12B-BF16/
```

### Known limitations

- **MTP weights not in SFT checkpoints**: SFT training does not include MTP layers. Use `--not-strict` during conversion to save incomplete shards (MTP weights will be randomly initialized but are unused during standard generation).
- **Super requires multi-GPU conversion**: Use EP=4 on 1 node (NVLink-only) to avoid Slingshot hangs.
- **Hub uploads large**: ~223GB per checkpoint, 20-40 min per upload.

---

## 5. Coherence Pipeline

### Purpose

Qualitative sanity check for HF checkpoints after training or conversion. Generates responses to 8 diverse prompts and logs them to a W&B table for side-by-side comparison across models.

### Usage

```bash
# Via SLURM (4 GPUs for 120B models)
isambard_sbatch pipeline_coherence_submit.sbatch nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Via SLURM (1 GPU for 30B models)
isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch geodesic-research/nemotron_nano_sft_warm_start_200k

# Local checkpoint
isambard_sbatch pipeline_coherence_submit.sbatch \
  /projects/a5k/public/checkpoints/megatron/my_experiment/iter_0000400/hf

# With custom W&B project
isambard_sbatch pipeline_coherence_submit.sbatch <model> \
  --wandb-project megatron_bridge_conversion_coherance_tests

# Directly (no SLURM, uses local GPUs)
source pipeline_env_activate.sh
python pipeline_coherence_test.py <model_path> [--max-tokens 3000] [--system-prompt "..."]
```

Results are logged to W&B project `geodesic-gen-tests` (default) with a generations table containing prompt, response, response length, and empty flag.

---

## Common Pitfalls

| Problem | Fix |
|---------|-----|
| `RuntimeError: ...gradient_accumulation_fusion...` | `model.gradient_accumulation_fusion: False` (no APEX) |
| NaN loss at iteration 7-8 | Lower LR to 5e-6 (recipe default) |
| `OSError: [Errno 116] Stale file handle` | `TRITON_CACHE_DIR`/`TMPDIR` to `/tmp` (automatic in `pipeline_training_launch.sh`) |
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
