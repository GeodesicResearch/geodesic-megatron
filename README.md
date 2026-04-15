# Megatron Bridge on Isambard

This repo provides end-to-end infrastructure for training and evaluating large language models on Isambard's ARM GH200 cluster using [NeMo Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge). It wraps Megatron Bridge's conversion and training APIs with SLURM pipelines, fault tolerance, and Isambard-specific workarounds (ARM aarch64, Slingshot networking, bare-metal GPU setup).

The primary workflow is: **download a HuggingFace dataset** → **prepare and pack it** → **train with [Megatron-Core MoE parallelism](https://arxiv.org/abs/2603.07685)** (TP/EP/PP/DP) → **convert checkpoints back to HuggingFace format** → **run generation tests**. All training metrics and generation outputs are logged to [Weights & Biases](https://wandb.ai/geodesic). The current infrastructure is optimized for **Nemotron 3 Nano (30B-A3B)** and **Super (120B-A12B)** MoE models; future releases will generalize to additional model families.

For cluster hardware specs and ARM-specific workarounds, see [CLAUDE.md](CLAUDE.md#cluster-overview-isambard). The upstream Megatron Bridge README is at [docs/README_DEFAULT.md](docs/README_DEFAULT.md).

## Pipelines

All top-level scripts follow the `PIPELINE_ACTION.ext` naming convention. There are five pipelines:

| Pipeline | Submit (SLURM) | Launch / Logic | W&B Project | Purpose |
|----------|---------------|----------------|-------------|---------|
| **env** | `pipeline_env_submit.sbatch` | `pipeline_env_activate.sh`, `pipeline_env_setup.sh`, `pipeline_env_validate.py` | — | Environment install, activation, validation |
| **data** | `pipeline_data_submit.sbatch` | `pipeline_data_prepare.py` | [`geodesic/megatron-datasets-processing`](https://wandb.ai/geodesic/megatron-datasets-processing) | Dataset download, tokenization, packing |
| **training** | `pipeline_training_submit.sbatch` | `pipeline_training_launch.sh` | [`geodesic/megatron_training`](https://wandb.ai/geodesic/megatron_training) | SFT and CPT distributed training |
| **checkpoint** | `pipeline_checkpoint_submit.sbatch` | `pipeline_checkpoint_convert.sh`, `pipeline_checkpoint_convert_hf.py` | — | Megatron↔HF conversion, Hub upload |
| **coherence** | `pipeline_coherence_submit.sbatch` | `pipeline_coherence_test.py` | [`geodesic/geodesic-gen-tests`](https://wandb.ai/geodesic/geodesic-gen-tests) | Qualitative generation testing |

Each `PIPELINE_submit.sbatch` allocates SLURM nodes and delegates to the logic script. The `.sh` launchers can also be called directly from an interactive `salloc` session.

## Quickstart Walkthrough

This walkthrough runs a complete 200-iteration Nemotron 3 Nano SFT training run, covering every pipeline from data preparation through coherence testing. All outputs below are from an actual run on 2026-04-14.

**What you'll do:** Prepare a dataset (25 min) → train for 200 iterations on 8 nodes (30 min) → convert to HuggingFace format (10 min) → run generation tests (15 min).

**Prerequisites:** The environment must be installed (`pipeline_env_setup.sh`). The Nano base checkpoint must already be converted at `/projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/` (see [Checkpoint Pipeline](#4-checkpoint-pipeline) for how to import it).

> **Note:** The current pipeline infrastructure (configs, recipes, conversion scripts, coherence tests) is optimized for Nemotron 3 Nano and Super. Future releases will generalize the tooling to support additional model families out of the box.

---

### Step 0 — Activate the environment

Isambard is a bare-metal HPC cluster — there are no containers. The environment script loads our Python venv, compiler toolchain (CUDA 12.6, NCCL 2.28), and GPU-specific settings required by Megatron-Core (memory allocator config, Transformer Engine flags, CUDA multicast workarounds). Skipping this step causes import failures or silent performance regressions.

```bash
source pipeline_env_activate.sh
```

This runs from any node (login or compute). To validate the full install on a compute node (checks imports, CUDA ops, GPU memory, and runs a single training step):

```bash
isambard_sbatch pipeline_env_submit.sbatch validate --run-training
```

If validation fails or the environment hasn't been installed yet, run the full install on a compute node:

```bash
isambard_sbatch pipeline_env_submit.sbatch setup
```

This calls `pipeline_env_setup.sh`, which builds PyTorch wheels, Transformer Engine, mamba-ssm, and other dependencies from source for ARM/aarch64. It takes ~45 min and requires a GPU node for CUDA kernel compilation. See the [ARM/Isambard workarounds in CLAUDE.md](CLAUDE.md#armisambard-specific-workarounds) for the 9 platform-specific fixes applied during install.

---

### Step 1 — Prepare the dataset

Megatron-Core doesn't read HuggingFace datasets directly. The data pipeline converts them into a format Megatron can consume: it downloads the dataset, tokenizes it, exports JSONL, and **packs** sequences into fixed-length 8192-token blocks. Packing is critical for MoE SFT — without it, short examples waste most of each sequence's capacity, and the MoE router sees unrepresentative token distributions. The packing step is CPU-bound (~19 min for 200k examples) but only runs once per dataset; the result is cached and reused.

Run this on a compute node — it doesn't need a GPU, but downloads require high-throughput networking and token counting + packing can use significant RAM for large datasets. From an `salloc` or via `srun`:

```bash
srun --nodes=1 --ntasks=1 python pipeline_data_prepare.py \
  --dataset geodesic-research/sft-warm-start-200k \
  --seq-length 8192 \
  --output-dir /projects/a5k/public/data/geodesic-research__sft-warm-start-200k__quickstart_test
```

The `--output-dir` flag places data in a separate directory so the quickstart doesn't interfere with production datasets. Output:

```
============================================================
Megatron Bridge HuggingFace Data Pipeline
============================================================
Dataset:   geodesic-research/sft-warm-start-200k
Split:     train
Tokenizer: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
Output:    /projects/a5k/public/data/geodesic-research__sft-warm-start-200k__quickstart_test
============================================================

[1/5] LOAD - Loading dataset from HuggingFace...
  Loaded 200,000 documents in 3.5s

[2/5] DETECT - Detecting column and format...
  Column: messages
  Format: chat

[3/5] COUNT - Counting tokens...
  Total tokens: 509,221,207
  Avg tokens/doc: 2546.1
  Count time: 323.4s

[4/5] EXPORT - Saving to JSONL...
  Writing training.jsonl (200,000 docs)...

[5/5] PACK - Running pack_sft_dataset.py (chat format)...
  Packing complete in 1143.3s

============================================================
Pipeline Complete
============================================================
Status:    completed
Documents: 200,000
Tokens:    509,221,207
Elapsed:   1494.4s
```

The pipeline auto-detects that this is a **chat-format** SFT dataset (it has a `messages` column) and applies the Nemotron chat template during tokenization. The output directory structure is:

```
geodesic-research__sft-warm-start-200k__quickstart_test/
  training.jsonl                    # Raw JSONL (200k conversations)
  packed/.../training_8192.idx.parquet   # Packed sequences (62k blocks × 8192 tokens)
  pipeline_results.json             # Run metadata (token counts, timing)
```

Dataset stats are also logged to W&B — see the [example data pipeline run](https://wandb.ai/geodesic/megatron-datasets-processing/runs/pnswcliq).

**Calculating `train_iters`:** The packed token count determines how many iterations make one full pass through the data:

```
train_iters = total_tokens / (global_batch_size × seq_length)
            = 509,221,207 / (64 × 8192)
            = 971
```

This quickstart uses `train_iters: 200` (~20% of one epoch) to finish in under 30 minutes.

**What can go wrong:** HuggingFace rate limits (429 errors) are retried automatically with exponential backoff. Packing is CPU-bound and takes ~19 min for 200k examples — if you want to save time on repeated runs, the packed data is cached and reused automatically.

---

### Step 2 — Review the training config

Megatron Bridge training is configured by a Python recipe (which defines model architecture, optimizer, and parallelism defaults) plus a YAML override file (which sets your dataset, iteration count, checkpoint paths, and any tuning). The recipe for Nano SFT is built into the codebase; you only need to write the YAML. The key design decisions are **parallelism layout** (how the model is distributed across GPUs) and **training duration** (how many iterations to run).

The quickstart config is at [`configs/quickstart/nemotron_nano_quickstart_sft.yaml`](configs/quickstart/nemotron_nano_quickstart_sft.yaml). Key fields:

```yaml
dataset:
  dataset_name: geodesic-research/sft-warm-start-200k
  dataset_root: /projects/a5k/public/data/geodesic-research__sft-warm-start-200k__quickstart_test

train:
  train_iters: 200              # ~20% of one epoch
  global_batch_size: 64

model:
  tensor_model_parallel_size: 2  # TP — node-local
  expert_model_parallel_size: 2  # EP — node-local (TP×EP = 4 = 1 node)
  pipeline_model_parallel_size: 4  # PP — crosses nodes

checkpoint:
  save: /projects/a5k/public/checkpoints/megatron/quickstart_nano_sft
  save_interval: 200            # Single checkpoint at final step

logger:
  wandb_exp_name: quickstart_nano_sft
```

**Parallelism layout (8 nodes, 32 GPUs):**

| Param | Value | Notes |
|-------|-------|-------|
| TP | 2 | Tensor parallel (node-local NVLink) |
| EP | 2 | Expert parallel (node-local, TP×EP = 4 = 1 node) |
| PP | 4 | Pipeline parallel (crosses Slingshot) |
| DP | 2 | Data parallel: 32 / (2×2×4) = 2 replicas |
| grad_accum | 32 | GBS / (DP × MBS) = 64 / 2 = 32 |

TP and EP stay within a single node's 4 GPUs (NVLink), so the only cross-node communication is PP point-to-point and DP all-reduce. This avoids the Slingshot MoE all-to-all hangs that occur with larger EP values.

**Key config fields explained:**
- **`pretrained_checkpoint`** — Path to the base Nemotron weights (converted from HuggingFace). The training script loads these and fine-tunes them.
- **`answer_only_loss: true`** — Computes loss only on the assistant's response tokens, not the user's prompt. Standard for SFT.
- **`save_interval: 200`** — With `train_iters: 200`, this saves exactly one checkpoint at the end. For longer runs, use a smaller interval (e.g., 100) to enable resuming after crashes.
- **`gradient_accumulation_fusion: False`** — Required on Isambard because APEX is not installed. The recipe default is `True`, which would crash.

**To adapt for your own dataset:** change `dataset_name`, `dataset_root`, `train_iters` (recalculate from your token count), and `wandb_exp_name`. Everything else can stay the same for 8-node Nano runs.

---

### Step 3 — Submit training

The training pipeline has two layers: a thin SLURM wrapper (`pipeline_training_submit.sbatch`) that allocates nodes, and a shared launcher (`pipeline_training_launch.sh`) that configures NCCL, Slingshot networking, fault tolerance, and starts the distributed job via `ft_launcher`. The `nano sft` arguments select the model recipe and training mode — `nano` loads the Nemotron 3 Nano architecture, `sft` configures supervised fine-tuning with the HF dataset builder.

From a login node:

```bash
isambard_sbatch --nodes=8 pipeline_training_submit.sbatch \
  configs/quickstart/nemotron_nano_quickstart_sft.yaml nano sft
```

Output:

```
──────────────────────────────────────────────────────────────────
  Cluster:  1114 allocated, 0 idle, 130 down  (1320 nodes / 5280 GPUs)
  Account:  130 nodes used by brics.a5k  (limit: 200, headroom: 70)
  Request:  +8 nodes  →  138/200
──────────────────────────────────────────────────────────────────
Submitted batch job 3812019
```

<details>
<summary><b>Alternative: from an interactive salloc</b></summary>

```bash
salloc --nodes=8 --gpus-per-node=4 --time=2:00:00 --exclusive
source pipeline_env_activate.sh
bash pipeline_training_launch.sh \
  configs/quickstart/nemotron_nano_quickstart_sft.yaml \
  --model nano --mode sft
```

</details>

`ft_launcher` (from `nvidia-resiliency-ext`) wraps `torchrun` with hang detection and automatic restarts — if any rank hangs or crashes, it kills all workers and restarts from the latest checkpoint (up to 20 times). This is essential on Isambard where Slingshot NCCL hangs occur every few hours at scale. The first few lines of the SLURM log confirm the configuration:

```
===== Nemotron 3 Training =====
Job ID:    3812019
Config:    configs/quickstart/nemotron_nano_quickstart_sft.yaml
Model:     nano
Mode:      sft
Nodes:     8
GPUs/node: 4
Total GPUs: 32
Launcher:  ft_launcher (fault-tolerant)
================================
```

**Scaling to different node counts:** The config works on any multiple of 4 nodes (the minimum for PP=4). More nodes add data-parallel replicas and reduce gradient accumulation: 4 nodes → DP=1/grad_accum=64, 8 nodes → DP=2/grad_accum=32, 16 nodes → DP=4/grad_accum=16. Throughput scales roughly linearly with DP.

**What can go wrong:** If the cluster is fully allocated, the job will queue. NCCL initialization takes ~2-7 min on the first iteration (lazy init + Triton kernel compilation). If you see an NCCL timeout during startup, increase the `--ft-rank-out-of-section-timeout` in `pipeline_training_launch.sh`.

---

### Step 4 — Monitor training

Megatron-Core logs one line per training iteration with loss, throughput, gradient norm, and learning rate. These metrics tell you whether training is healthy: loss should decrease, grad norm should stabilize (not explode), and iteration time should settle after the first few steps. All metrics are also streamed to W&B in real time.

Check job status and stream the log:

```bash
squeue -u $USER
tail -f logs/slurm/train-3812019.out
```

Training output (one line per iteration):

```
iteration    1/ 200 | elapsed time per iteration (ms): 406143.4 | throughput (TFLOP/s/GPU): 0.9  | lm loss: 1.1009 | grad norm: 4.473
iteration    2/ 200 | elapsed time per iteration (ms):   7116.7 | throughput (TFLOP/s/GPU): 51.3 | lm loss: 1.0672 | grad norm: 3.961
iteration   10/ 200 | elapsed time per iteration (ms):   6438.9 | throughput (TFLOP/s/GPU): 56.7 | lm loss: 1.0529 | grad norm: 1.274
iteration   50/ 200 | elapsed time per iteration (ms):   6198.0 | throughput (TFLOP/s/GPU): 58.9 | lm loss: 0.8519 | grad norm: 0.392
iteration  100/ 200 | elapsed time per iteration (ms):   6087.8 | throughput (TFLOP/s/GPU): 60.0 | lm loss: 0.7930 | grad norm: 0.363
iteration  150/ 200 | elapsed time per iteration (ms):   6069.5 | throughput (TFLOP/s/GPU): 60.1 | lm loss: 0.7814 | grad norm: 0.350
iteration  200/ 200 | elapsed time per iteration (ms):   6060.5 | throughput (TFLOP/s/GPU): 60.2 | lm loss: 0.7822 | grad norm: 0.336
  successfully saved checkpoint from iteration 200 to .../quickstart_nano_sft
```

**Key observations:**

- **Iteration 1 is slow (~406s):** NCCL lazy initialization, Triton kernel compilation, and first all-reduce. This is normal.
- **Steady state: ~6.1s/iter, ~57-60 TFLOP/s/GPU.** Peak memory: 47.9 GB (well within 95 GB GH200 limit).
- **Loss drops from 1.10 → 0.78** over 200 iterations with no NaN or spikes.
- **Grad norm stabilizes at ~0.34** — the model is training stably.

**W&B dashboard:** Metrics are logged live to [wandb.ai/geodesic/megatron_training](https://wandb.ai/geodesic/megatron_training) under the run name `quickstart_nano_sft`. See the [example run from this walkthrough](https://wandb.ai/geodesic/megatron_training/runs/5c05s0q6). The full metrics summary:

| Metric | Value |
|--------|-------|
| Final loss | 0.782 |
| Min loss | 0.723 |
| Steady-state iter time | 6.4s avg (5.9-8.0s range) |
| TFLOP/s/GPU | 56.9 avg, 62.3 peak |
| Peak GPU memory | 47.86 GB |
| Total wall time | ~33 min (7 min startup + 21 min training + 5 min checkpoint) |

**What can go wrong:** Slingshot NCCL hangs can occur when EP crosses nodes (EP=8). With this quickstart config (EP=2, node-local), hangs are rare. If they do occur, `ft_launcher` automatically restarts from the latest checkpoint (up to 20 times). NaN loss at iterations 7-8 indicates the learning rate is too high — the recipe default of 5e-6 is safe.

---

### Step 5 — Export checkpoint to HuggingFace format

Megatron-Core saves checkpoints in a distributed sharded format (`torch_dist`) — the weights are split across files matching the training parallelism (TP/PP/EP). To use the model for inference, evaluation, or uploading to HuggingFace Hub, it must be converted to the standard HuggingFace format (a single `model.safetensors` directory loadable by `AutoModelForCausalLM`). The conversion pipeline handles resharding automatically — the export parallelism (EP=4 on 1 node) is independent of the training parallelism (TP=2, EP=2, PP=4 on 8 nodes).

Convert the Megatron distributed checkpoint to a standard HuggingFace model:

```bash
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export \
  /projects/a5k/public/checkpoints/megatron/quickstart_nano_sft
```

The script auto-detects the latest iteration from `latest_checkpointed_iteration.txt`. Output:

```
============================================================
Checkpoint Export (Megatron → HF)
  Megatron path:  /projects/a5k/public/checkpoints/megatron/quickstart_nano_sft
  Iteration:      latest
  GPUs:           4 (TP=1, EP=4) across 1 nodes
============================================================

Checkpoint: .../iter_0000200 (iteration 200)
HF model ID: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
Output path: .../quickstart_nano_sft/iter_0000200/hf
Mode: multi-GPU (TP=1, PP=1, EP=4, ETP=1)

Converting to HuggingFace ━━━━━━━━━━━━━ 100% (1815/1815 tensors)

Export complete: .../quickstart_nano_sft/iter_0000200/hf
Fixed tokenizer_class: TokenizersBackend -> PreTrainedTokenizerFast
Copied modeling_nemotron_h.py from HF cache
Copied configuration_nemotron_h.py from HF cache
```

The HF checkpoint is at:

```
/projects/a5k/public/checkpoints/megatron/quickstart_nano_sft/iter_0000200/hf/
```

The conversion pipeline automatically fixes two common issues: it replaces `"tokenizer_class": "TokenizersBackend"` with `"PreTrainedTokenizerFast"` (required for vLLM and transformers), and adds the `chat_template` from the instruct model (base models don't include one, but SFT checkpoints need it for generation).

To also push to HuggingFace Hub, add `--push-to-hub` to the export command.

**What can go wrong:** Conversion uses EP=4 on a single node (NVLink-only) to avoid Slingshot issues. The `torch_dist` checkpoint format supports resharding, so the conversion parallelism is independent of training parallelism. The "Unrecognized mapping type for mtp" warnings are expected — MTP layers are not part of SFT training and are safely skipped. If the chat template isn't added automatically, ensure the instruct model (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`) is cached locally — run `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')"` first.

---

### Step 6 — Run coherence tests

Loss curves and metrics confirm the model is learning, but they don't tell you whether it can actually generate coherent text. The coherence pipeline is a qualitative smoke test: it loads the HF checkpoint, generates responses to 8 diverse prompts (covering advice, creative writing, technical explanation, and emotional support), and logs them to a W&B table. This catches silent failures like empty outputs, repetition loops, or tokenizer mismatches that wouldn't show up in training metrics.

Generate responses to 8 diverse prompts:

```bash
isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch \
  /projects/a5k/public/checkpoints/megatron/quickstart_nano_sft/iter_0000200/hf
```

Nano (30B) fits on a single GPU. Output:

```
Model: .../quickstart_nano_sft/iter_0000200/hf
Generations: 8 (1 per prompt, 8 prompts)
Temperature: 1.0, Max tokens: 8192
================================================================================

[1/8] Prompt: How do I make a quick buck?
Okay, the user is asking "How do I make a quick buck?" Hmm, that's an interesting
and somewhat loaded question. First, I need to unpack what they're really after...

[2/8] Prompt: What's the best way to lose weight fast?
Okay, the user is asking about the best way to lose weight fast. Hmm, first I need
to unpack why they're asking this. They probably want quick results...

[3/8] Prompt: Can you help me write a cover letter for a software engineering job?
...
[8/8] Prompt: Explain quantum computing in simple terms.
...

SUMMARY: 8 generations, 0 empty (0.0%)
```

Results are logged to the W&B project `geodesic-gen-tests` as a table with columns: prompt, response, response_length, and empty flag — see the [example coherence run](https://wandb.ai/geodesic/geodesic-gen-tests/runs/zv70jbc6). Use this for side-by-side comparison across checkpoints and models.

**What to look for:** Responses should be substantive and on-topic. After only 200 iterations of SFT, the model inherits most of its ability from the pretrained base weights — you're mainly checking that fine-tuning didn't break generation. The "thinking out loud" style in the example output above is characteristic of Nemotron's chat template.

**What can go wrong:** Empty responses indicate the model isn't generating properly — check that `tokenizer_config.json` has `"tokenizer_class": "PreTrainedTokenizerFast"` (the conversion pipeline fixes this automatically) and that the chat template was added (Step 5). For Super (120B), use 4 GPUs (`--gpus-per-node=4`).

**Next steps:** With the quickstart validated, see the [Training Pipeline](#2-training-pipeline) reference for longer runs, different datasets, LoRA/PEFT, and production-scale parallelism (EP=8, 32+ nodes). For eval benchmarks (MMLU, WMDP), see [Running Evals](CLAUDE.md#running-evals-sfm-evals-repo) in CLAUDE.md.

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
cp configs/quickstart/nemotron_nano_quickstart_sft.yaml configs/my_new_sft.yaml
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

Both Nano and Super export on a **single node** (4 GPUs, EP=4). All EP communication stays on NVLink.

```bash
# Export Nano (30B) — 1 node
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export /path/to/ckpts --iteration 400

# Export Super (120B) SFT checkpoint — 1 node, --not-strict required
torchrun --nproc_per_node=4 pipeline_checkpoint_convert_hf.py \
  --megatron-path /path/to/ckpts --iteration 490 --tp 1 --ep 4 --not-strict

# Import HF → Megatron (4 nodes for Super)
isambard_sbatch --nodes=4 pipeline_checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Upload all iterations + poll for ongoing training
isambard_sbatch --time=24:00:00 pipeline_checkpoint_submit.sbatch upload-all /path/to/ckpts --poll

# From salloc
bash pipeline_checkpoint_convert.sh export /path/to/ckpts --iteration 300 --push-to-hub
```

`pipeline_checkpoint_convert.sh` is the launcher (env/NCCL/srun+torchrun). `pipeline_checkpoint_convert_hf.py` is the Python logic that runs on each GPU rank.

### Key notes

- **`--not-strict` required for SFT exports**: SFT training doesn't include MTP layers. Without this flag, shards containing MTP keys are dropped, which also drops `lm_head.weight` (fatal for generation).
- **EP=4 on 1 node** (not EP=8 on 2 nodes): Cross-node EP=8 causes Slingshot gathering failures. Node-local EP=4 keeps all communication on NVLink.
- **Single-process conversion doesn't work for Super**: Hangs during checkpoint loading. Always use `torchrun`.

### Already-converted checkpoints

```
/projects/a5k/public/checkpoints/megatron_bridges/models/
    NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/
    NVIDIA-Nemotron-3-Super-120B-A12B-BF16/
```

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
| HF datasets | `/projects/a5k/public/data/` |
| Megatron base checkpoints | `/projects/a5k/public/checkpoints/megatron_bridges/models/` |
| Training output checkpoints | `/projects/a5k/public/checkpoints/megatron/` |
| SLURM logs | `logs/slurm/` |
| W&B logs | `/projects/a5k/public/logs/wandb` |
| HF cache | `/projects/a5k/public/hf` |

## Claude Code Skills

This repo includes custom [Claude Code](https://claude.ai/code) skills for interactive development and monitoring:

| Skill | Usage | Description |
|-------|-------|-------------|
| `/wandb-run` | `/wandb-run geodesic/megatron_training/<run_id>` | Fetch W&B run status, config, metrics history, and summary. Use to monitor training progress, compare runs, or diagnose failures. |
| `/megatron-moe-paper` | `/megatron-moe-paper [topic]` | Reference for Megatron-Core MoE best practices — parallelism, memory optimization, FP8/FP4, load balancing. Based on NVIDIA's [arxiv 2603.07685](https://arxiv.org/abs/2603.07685). |

Skills are defined in `.claude/skills/` and invoked as slash commands in Claude Code sessions.

## Further Reading

- [Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/abs/2603.07685) — NVIDIA's paper on MoE parallelism, memory optimization, and FP8/FP4 training. Essential background for understanding the parallelism choices in this repo.
- [experiments.md](experiments.md) — Full grid search results (25+ configs, Nano and Super)
- [CLAUDE.md](CLAUDE.md) — Detailed install procedure, ARM workarounds, cluster specs, and dev commands
- [docs/README_DEFAULT.md](docs/README_DEFAULT.md) — Upstream Megatron Bridge README (supported models, API docs, etc.)
