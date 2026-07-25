# Megatron Bridge on Isambard

This repo provides end-to-end infrastructure for training and evaluating large language models on Isambard's ARM GH200 cluster using [NeMo Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge). It wraps Megatron Bridge's conversion and training APIs with SLURM pipelines, fault tolerance, and Isambard-specific workarounds (ARM aarch64, Slingshot networking, containerized execution via Apptainer + NGC images).

The primary workflow is: **download a HuggingFace dataset** → **prepare and pack it** → **train with [Megatron-Core MoE parallelism](https://arxiv.org/abs/2603.07685)** (TP/EP/PP/DP) → **convert checkpoints back to HuggingFace format** → **run generation tests**. All training metrics and generation outputs are logged to [Weights & Biases](https://wandb.ai/geodesic). The current infrastructure is optimized for **Nemotron 3 Nano (30B-A3B)** and **Super (120B-A12B)** MoE models; future releases will generalize to additional model families.

For cluster hardware specs and per-model topology findings, see [CLAUDE.md](CLAUDE.md#cluster-overview-isambard); for the execution environment itself, [docs/environment.md](docs/environment.md). The upstream Megatron Bridge README is at [docs/README_DEFAULT.md](docs/README_DEFAULT.md).

## Pipelines

All top-level scripts follow the `PIPELINE_ACTION.ext` naming convention. There are five pipelines:

| Pipeline | Submit (SLURM) | Launch / Logic | W&B Project | Purpose |
|----------|---------------|----------------|-------------|---------|
| **env** | `pipeline_env_submit.sbatch` | `pipeline_env_config.env`, `pipeline_env_exec.sh`, `pipeline_env_activate.sh`, `pipeline_env_setup.sh`, `pipeline_env_validate.py` | — | **The execution environment**: Apptainer + NGC NeMo image, Slingshot NCCL stack, install + validation ([docs/environment.md](docs/environment.md)) |
| **data** | `pipeline_data_submit.sbatch` | `pipeline_data_prepare.py` | [`geodesic/megatron-datasets-processing`](https://wandb.ai/geodesic/megatron-datasets-processing) | Dataset download, tokenization, packing |
| **training** | `pipeline_training_submit.sbatch` | `pipeline_training_launch.sh` | [`geodesic/megatron_training`](https://wandb.ai/geodesic/megatron_training) | SFT and CPT distributed training |
| **checkpoint** | `pipeline_checkpoint_submit.sbatch` | `pipeline_checkpoint_convert.sh`, `pipeline_checkpoint_convert_hf.py` | — | Megatron↔HF conversion, Hub upload |
| **coherence** | `pipeline_coherence_submit.sbatch` | `pipeline_coherence_test.py` | [`geodesic/geodesic-gen-tests`](https://wandb.ai/geodesic/geodesic-gen-tests) | Qualitative generation testing |

Each `PIPELINE_submit.sbatch` allocates SLURM nodes and delegates to the logic script. The `.sh` launchers can also be called directly from an interactive `salloc` session.

## Quickstart Walkthrough

This walkthrough runs a complete 200-iteration Nemotron 3 Nano SFT training run, covering every pipeline from data preparation through coherence testing. All outputs below are from an actual run on 2026-04-14.

**What you'll do:** Prepare a dataset (25 min) → train for 200 iterations on 8 nodes (30 min) → convert to HuggingFace format (10 min) → run generation tests (15 min).

**Prerequisites:** The environment must be installed once (`bash pipeline_env_setup.sh` on a GPU node — see Step 0). The Nano base checkpoint must already be converted at `/projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/` (see [Checkpoint Pipeline](#4-checkpoint-pipeline) for how to import it).

> **Note:** The current pipeline infrastructure (configs, recipes, conversion scripts, coherence tests) is optimized for Nemotron 3 Nano and Super. Future releases will generalize the tooling to support additional model families out of the box.

---

### Step 0 — Install the environment (one-time)

**Every pipeline runs inside one Apptainer container** built from an NGC NeMo image
(aarch64), which supplies PyTorch, CUDA, NCCL, Transformer Engine, the Mamba kernels, APEX,
and `ft_launcher` prebuilt and version-matched. This repo's `src/` and the pinned
`3rdparty/Megatron-LM` are bind-mounted, so the checkout you submit from is exactly the code
that runs. The launchers enter the container themselves — there is nothing to activate before
submitting, and every submit command below is the whole command.

Installation (per cluster, shared across users via `/projects/a5k/public/containers/`) is a
single idempotent command:

```bash
# On a GPU node: SIF pull + Slingshot NCCL build + Python overlay + validation.
# Completed steps are skipped with an explicit message; --force redoes everything.
bash pipeline_env_setup.sh
# or via SLURM: isambard_sbatch pipeline_env_submit.sbatch setup

# Re-validate an existing install (20 checks; 21 with --run-training):
isambard_sbatch pipeline_env_submit.sbatch validate --run-training
```

Design decisions, image contents, the image-qualification gates, and troubleshooting:
**[docs/environment.md](docs/environment.md)**.

---

### Step 1 — Prepare the dataset

Megatron-Core doesn't read HuggingFace datasets directly. The data pipeline converts them into a format Megatron can consume: it downloads the dataset, tokenizes it, exports JSONL, and **packs** sequences into fixed-length 8192-token blocks. Packing is critical for MoE SFT — without it, short examples waste most of each sequence's capacity, and the MoE router sees unrepresentative token distributions. The packing step is CPU-bound (~19 min for 200k examples) but only runs once per dataset; the result is cached and reused.

Submit it as its own job — it doesn't need much GPU, but downloads require high-throughput networking and token counting + packing can use tens of GB of RAM on large datasets. The `prepare` mode forwards its arguments to `pipeline_data_prepare.py` inside the container:

```bash
isambard_sbatch pipeline_data_submit.sbatch prepare \
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
- **`gradient_accumulation_fusion: False`** — This quickstart leaves fused wgrad accumulation off. The container image ships APEX, so `True` works and is the faster path (measured ~1.1 s/iter on the 120B benchmark, which uses it — see [`configs/quickstart/nemotron_super_quickstart_sft.yaml`](configs/quickstart/nemotron_super_quickstart_sft.yaml)).

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
bash pipeline_training_launch.sh \
  configs/quickstart/nemotron_nano_quickstart_sft.yaml \
  --model nano --mode sft
```

The launcher enters the container on every node itself, so nothing is sourced beforehand.

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
  /projects/a5k/public/checkpoints/megatron/quickstart_nano_sft \
  --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning
```

`--hf-model` (the upstream architecture/tokenizer reference) and `--reasoning|--no-reasoning` are required. The script auto-detects the latest iteration from `latest_checkpointed_iteration.txt`. Output:

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

The conversion pipeline automatically repairs the exported config so the result is loadable by the evaluation stack:

- replaces `"tokenizer_class": "TokenizersBackend"` with `"PreTrainedTokenizerFast"` and strips the accompanying `backend`/`is_local` hints (required for vLLM and older transformers);
- adds the `chat_template` from the instruct model (base models don't include one, but SFT checkpoints need it for generation);
- strips the read-only `layers_block_type` and emits the equivalent `hybrid_override_pattern`, which is the form NemotronH configs accept;
- emits `num_hidden_layers`, which vLLM pinned to `transformers<5` requires;
- reconciles `vocab_size` with the actual embedding rows, so a vocab-extended checkpoint exported against a stock donor does not trip vLLM's embedding-shape assert.

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

The environment is an Apptainer container built from a qualified NGC NeMo image; the full
reference is **[docs/environment.md](docs/environment.md)**.

```bash
# Install everything (SIF pull + Slingshot NCCL build + Python overlay + validate).
# GPU node required for the build and the validation; idempotent, --force redoes all.
isambard_sbatch pipeline_env_submit.sbatch setup
bash pipeline_env_setup.sh --only slingshot --force    # or one step at a time

# Validate an existing install (20 checks; 21 with --run-training)
isambard_sbatch pipeline_env_submit.sbatch validate --run-training

# Run anything inside the environment (interactive shell, tests, ad-hoc python)
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh; exec bash -i"
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh; \
  python -m pytest tests/unit_tests/ -x -q -m 'not pleasefixme'"
```

Everything configurable — image tag, SIF path, Slingshot component versions, overlay
packages, bind list, cache dirs — lives in `pipeline_env_config.env`. A missing SIF or
Slingshot build hard-fails every launcher with the command that fixes it; there is no
fallback environment.

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

The governing constraint on this fabric is **TP × EP ≤ 4** (and TP × CP ≤ 4): keep expert
all-to-all and context-parallel traffic on a node's NVLink, and let only PP cross Slingshot.
Cross-node EP costs ~14× throughput and reliably hangs the CXI fabric.

| Model | Validated layout | Measured |
|---|---|---|
| **Nano (30B-A3B)** | 8 nodes / 32 GPUs: TP=2, EP=2, PP=4, DP=2 (seq 8192, GBS 16) | ~3.4 s/iter, ~27 TFLOP/s/GPU; zero hangs through 500+ iters |
| **Super (120B-A12B)** | TP=1, CP=(min that fits), EP=4, PP=22, ETP=1 | ~75-84 TFLOP/s/GPU, ~1000+ tok/s/GPU (≈2.4× the old TP=4 layouts) |
| **Super benchmark** | 16 nodes / 64 GPUs: TP=1, CP=4, EP=4, PP=8, ETP=1, DP=2 (seq 32K, GBS 64) | ~27.6 s/iter — the standing environment benchmark, [`configs/quickstart/nemotron_super_quickstart_sft.yaml`](configs/quickstart/nemotron_super_quickstart_sft.yaml) |
| **Ultra (550B-A55B)** | 72 nodes / 288 GPUs: TP=4, EP=4, PP=36, ETP=1 | ~28-30 s/iter steady state; first iter 45-75 min (lazy NCCL init at this depth) |

Other levers that matter: `recompute_granularity: selective` with MoE-scoped
`recompute_modules` (full recompute is ~10× slower; on the 120B it is the ~24 GB between fit
and OOM), `moe_permute_fusion: True`, `expert_tensor_parallel_size: 1` (parallel folding —
what keeps EP node-local at high TP), `gradient_accumulation_fusion: True` (the image ships
APEX; ~1.1 s/iter on the 120B), and **BF16 everywhere** — FP8 causes stochastic alignment
crashes in MoE routing. Recipe LR 5e-6; 8e-5 NaNs under context parallelism. Full topology
reasoning, per-model memory notes, and the legacy layouts these superseded are in
[CLAUDE.md](CLAUDE.md#nemotron-3-super-120b-a12b-on-isambard).

---

## 3. Data Pipeline

### Usage

```bash
# Full pipeline: download + tokenize + export + pack (args forwarded to pipeline_data_prepare.py)
isambard_sbatch pipeline_data_submit.sbatch prepare \
  --dataset allenai/Dolci-Instruct-SFT --seq-length 8192

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
# Export Nano (30B) — 1 node. --hf-model and --reasoning|--no-reasoning are REQUIRED.
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export /path/to/ckpts \
  --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning --iteration 400

# Export Super (120B) SFT checkpoint — 1 node, --not-strict required
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export /path/to/ckpts \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
  --iteration 490 --not-strict

# Import HF → Megatron (4 nodes for Super)
isambard_sbatch --nodes=4 pipeline_checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Upload all iterations + poll for ongoing training
isambard_sbatch --time=24:00:00 pipeline_checkpoint_submit.sbatch upload-all /path/to/ckpts \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning --poll

# From salloc
bash pipeline_checkpoint_convert.sh export /path/to/ckpts \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
  --iteration 300 --push-to-hub
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

# Megatron checkpoint directly, no HF export (multi-node)
isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch <megatron-ckpt-dir> \
  --backend megatron --hf-model <hf-id> --tp 4 --pp 6 --ep 4 --max-tokens 256

# Against an already-running OpenAI-compatible server (stdlib HTTP, no local model)
isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch <served-id> \
  --backend endpoint --discovery-file /projects/a5k/public/vllm-serve/<stem>.endpoint

# Directly, inside an allocation (uses this node's GPUs)
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh; \
  python pipeline_coherence_test.py <model_path> [--max-tokens 3000]"
```

Results are logged to W&B project `geodesic-gen-tests` (default) with a generations table containing prompt, response, response length, and empty flag.

---

## Bad Compute Nodes

Isambard occasionally has hardware-broken nodes — VS Code tunnels that never come up, GPUs returning `ERR!`, NCCL dying on first collective. The team shares a TTL'd log that `isambard_sbatch` automatically passes to SLURM's `--exclude` on every submission, so once a teammate reports a bad node, nobody else lands on it. A summary line prints on every submission:

```
Bad nodes: 3 excluded (last 7d)  —  file: /projects/a5k/public/isambard_sbatch_bad_nodes.log
           report more: isambard_sbatch --mark-bad <node> [reason]
```

If you are **very confident** a failure is a node-specific hardware issue (not a code, config, or library-version bug), register it so the rest of the team doesn't land on it. The wrapper supports full CRUD on the log:

```bash
isambard_sbatch --mark-bad nid001234 "vscode tunnel never came up"   # Create
isambard_sbatch --list-bad                                           # Read
isambard_sbatch --update-bad nid001234 "GPU ECC (Xid 48)"            # Update reason
isambard_sbatch --unmark-bad nid001234                               # Delete (node got fixed)
isambard_sbatch --prune-bad                                          # Housekeeping
```

Entries expire after 7 days, so a node that gets fixed stops being excluded automatically. **Don't mark nodes for code-level issues** (OOM, bad YAML, wrong parallelism) — that would falsely exclude healthy nodes for a week and erode the list's signal. See [CLAUDE.md](CLAUDE.md#bad-compute-nodes) for the full register-when / don't-register-when checklist.

## Common Pitfalls

| Problem | Fix |
|---------|-----|
| `RuntimeError: ...gradient_accumulation_fusion...` | Should not happen — the container image ships APEX. It means the payload is not running inside the container (see [docs/environment.md](docs/environment.md)) |
| `FATAL [env-config]: SIF not found` / `Slingshot NCCL stack not built` | Environment not installed on this cluster: `bash pipeline_env_setup.sh` (GPU node) |
| NCCL bandwidth ~2 GB/s or `NET/Socket` in the log | CXI plugin not loading. **Never** "fix" it with `brics/apptainer-multi-node`/`adapt.sh` — see [docs/environment.md](docs/environment.md) troubleshooting |
| NaN loss at iteration 7-8 | Lower LR to 5e-6 (recipe default) |
| `OSError: [Errno 116] Stale file handle` | `TRITON_CACHE_DIR`/`TMPDIR` to `/tmp` (automatic in `pipeline_training_launch.sh`) |
| NCCL hangs every ~7-8 min | Slingshot fabric issue. ft_launcher auto-restarts |
| EP=4 OOMs on GH200 | Use EP=8 (16 experts/GPU = 51GB vs 32 = 93GB) |
| `nemo_experiments/` fills disk | Remove old TB logs selectively. **Do NOT `rm -rf`** — contains checkpoint state |
| VS Code tunnel never starts / job sits in RUNNING with no output | Likely a bad compute node. `isambard_sbatch --mark-bad <nid> "tunnel hung"` and resubmit. See [Bad Compute Nodes](#bad-compute-nodes) |
| `nvidia-smi` shows `ERR!` on specific GPUs of one host | Node-specific hardware fault. `isambard_sbatch --mark-bad <nid> "GPU ECC err"` |

## Disk Locations

| What | Path |
|------|------|
| HF datasets | `/projects/a5k/public/data/` |
| Megatron base checkpoints | `/projects/a5k/public/checkpoints/megatron_bridges/models/` |
| Training output checkpoints | `/projects/a5k/public/checkpoints/megatron/` |
| SLURM logs | `logs/slurm/` (by run ID: `logs/slurm/by-run-id/`) |
| W&B logs | `/projects/a5k/public/logs/wandb` |
| Torch profiles | `/projects/a5k/public/profiles/<wandb-exp-name>/<run-id>/` (see [docs/profiling-quickstart.md](docs/profiling-quickstart.md)) |
| HF cache | `/projects/a5k/public/hf` |
| Container SIF, Slingshot build, Python overlay | `/projects/a5k/public/containers/` (see [docs/environment.md](docs/environment.md)) |

## Claude Code Skills

This repo includes custom [Claude Code](https://claude.ai/code) skills for interactive development and monitoring:

| Skill | Usage | Description |
|-------|-------|-------------|
| `/wandb-run` | `/wandb-run geodesic/megatron_training/<run_id>` | Fetch W&B run status, config, metrics history, and summary. Use to monitor training progress, compare runs, or diagnose failures. |
| `/megatron-moe-paper` | `/megatron-moe-paper [topic]` | Reference for Megatron-Core MoE best practices — parallelism, memory optimization, FP8/FP4, load balancing. Based on NVIDIA's [arxiv 2603.07685](https://arxiv.org/abs/2603.07685). |

Skills are defined in `.claude/skills/` and invoked as slash commands in Claude Code sessions.

### Claude Code guardrail tooling

This repo also integrates [`geodesic-claude-tooling`](.claude/geodesic-claude-tooling) (a git
submodule) — Claude Code hooks that inject Geodesic's working conventions at session start, validate
plans on exit, and run mechanical checks on the diff. It is wired up **additively** and does not
touch the training environment: the hooks live in the repo's dev-tooling-only `.venv`
(ruff, pre-commit, the hook entry points — no torch, created with `uv sync --inexact`),
never in the container. Install it once:

```bash
bash scripts/install_claude_tooling.sh
```

Configuration lives in `.claude/settings.json` (hooks) and `.claude/geodesic-config.yaml` (quality
items). See [CLAUDE.md](CLAUDE.md#claude-code-tooling) for details, including how to enable the
commit-time review gate (left off by default).

## Further Reading

- [Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/abs/2603.07685) — NVIDIA's paper on MoE parallelism, memory optimization, and FP8/FP4 training. Essential background for understanding the parallelism choices in this repo.
- [docs/environment.md](docs/environment.md) — The execution environment: install, design decisions, image qualification, troubleshooting
- [docs/profiling-quickstart.md](docs/profiling-quickstart.md) — Capturing and reading torch-profiler traces of a training run
- [CLAUDE.md](CLAUDE.md) — Cluster specs, per-model topology findings, campaign conventions, and dev commands
- [docs/README_DEFAULT.md](docs/README_DEFAULT.md) — Upstream Megatron Bridge README (supported models, API docs, etc.)
