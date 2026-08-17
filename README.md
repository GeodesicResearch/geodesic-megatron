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
| **training** | `pipeline_training_submit.sbatch` | `pipeline_training_launch.sh` | [`geodesic/megatron_training`](https://wandb.ai/geodesic/megatron_training) | SFT, CPT, and from-scratch pretraining |
| **checkpoint** | `pipeline_checkpoint_submit.sbatch` | `pipeline_checkpoint_convert.sh`, `pipeline_checkpoint_convert_hf.py` | — | Megatron↔HF conversion, Hub upload |
| **coherence** | `pipeline_coherence_submit.sbatch` | `pipeline_coherence_test.py` | [`geodesic/geodesic-gen-tests`](https://wandb.ai/geodesic/geodesic-gen-tests) | Qualitative generation testing |

Each `PIPELINE_submit.sbatch` allocates SLURM nodes and delegates to the logic script. The `.sh` launchers can also be called directly from an interactive `salloc` session.

## Quickstart Walkthrough

This walkthrough runs a complete Nemotron 3 Nano SFT training run on the shipped 32K
quickstart config, covering every pipeline from data preparation through coherence
testing. (An earlier edition of this walkthrough used a separate 8K demo config; that
config and its captured outputs were retired on 2026-08-05 when the quickstarts were
standardised on seq 32768 at 64 GPUs. The throughput figures quoted below are the
measured 2026-08-05 numbers from the shipped config.)

**What you'll do:** Point at the prepared dataset (or prepare it once) → train on 16
nodes (~10 s/iter; a 200-iteration demo is ~35 min) → convert the checkpoint to
HuggingFace format (10 min) → run generation tests (15 min).

**Prerequisites:** The environment must be installed once (`bash pipeline_env_setup.sh`
on a GPU node — see Step 0). The Nano base checkpoint must already be converted at
`/projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/`
(see [Checkpoint Pipeline](#4-checkpoint-pipeline) for how to import it).

> **Note:** The current pipeline infrastructure (configs, recipes, conversion scripts,
> coherence tests) is optimized for Nemotron 3 Nano and Super. Future releases will
> generalize the tooling to support additional model families out of the box.

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

### Step 1 — The dataset

The quickstart consumes `geodesic-research/pa-warm-start-1B-sft-mix` packed at
seq_length 32768, which is already prepared on the cluster at the `dataset_root` the
config names — the same packs the Super-120B benchmark uses (both models share vocab
131072 and one tokenizer encoder, so no repack is needed). If you need to regenerate it
(or prepare a different dataset), use the [Data Pipeline](#3-data-pipeline) with
`--seq-length 32768`; note the config's `pad_seq_to_mult: 16`, which satisfies the
packing rule for context parallelism (pad multiple ≥ 2×CP).

---

### Step 2 — Review the training config

The quickstart config is at
[`configs/quickstart/nemotron_nano_quickstart_sft.yaml`](configs/quickstart/nemotron_nano_quickstart_sft.yaml).
Key fields:

```yaml
train:
  global_batch_size: 128     # the standard batch across quickstarts
  micro_batch_size: 1
  train_iters: 40            # benchmark length; override for longer demos

model:
  seq_length: 32768
  context_parallel_size: 2   # mandatory at 32K: halves the 16 GiB fp32 CE logits
  expert_model_parallel_size: 4
  recompute_granularity: full  # mandatory: frees the room the logits need

checkpoint:
  save: null                 # benchmark posture — override to keep a checkpoint
logger:
  wandb_save_dir: /projects/a5k/public/logs/wandb   # mandatory when save is null
```

The header of the config carries the full measured story: topology, the closed
alternatives (all measured worse), and the memory walls that force CP=2 and full
recompute. It is the reference for *why* every field is what it is.

---

### Step 3 — Submit training

The training pipeline has two layers: a thin SLURM wrapper
(`pipeline_training_submit.sbatch`) that allocates nodes, and a shared launcher
(`pipeline_training_launch.sh`) that configures NCCL, Slingshot networking, and starts
the distributed job. The `nano sft` arguments select the model recipe and training
mode. `--disable-ft` is part of the documented benchmark command (the certified posture;
see the Super quickstart header for the FT straggler-reporter interaction it avoids).

The benchmark run, exactly as certified (40 iterations, no checkpoint):

```bash
isambard_sbatch --nodes=16 pipeline_training_submit.sbatch \
  configs/quickstart/nemotron_nano_quickstart_sft.yaml nano sft --disable-ft
```

For a demo that trains longer and keeps a checkpoint for Steps 5–6, add Hydra overrides
(the launcher forwards them):

```bash
isambard_sbatch --nodes=16 pipeline_training_submit.sbatch \
  configs/quickstart/nemotron_nano_quickstart_sft.yaml nano sft --disable-ft \
  train.train_iters=200 \
  checkpoint.save=/projects/a5k/public/checkpoints/megatron/nemotron_nano_quickstart_sft \
  checkpoint.save_optim=false checkpoint.save_rng=false
```

Megatron-Core saves a final checkpoint when `train_iters` is reached, so this writes
exactly one checkpoint at iteration 200 (`save_optim/save_rng: false` skip Adam moments
and RNG state the downstream conversion never reads).

---

### Step 4 — Monitor training

```bash
tail -f logs/slurm/train-<jobid>.out | grep --line-buffered "iteration"
```

Measured behaviour of this config (2026-08-05, 64 GPUs, solo): the first iteration is
slow (~40–110 s — compile warm-up for the full-recompute path), iterations settle after
~iteration 22, and the settled mean is **9.77 s/iter (76.3 ms/sample) at peak 91.5 GB
of 95**. Loss on the warm-started base descends from ~1.08 within the first dozens of
iterations; 0 NaN. Metrics stream live to
[wandb.ai/geodesic/megatron_training](https://wandb.ai/geodesic/megatron_training)
under the run name `nemotron_nano_quickstart_sft`.

---

### Step 5 — Export checkpoint to HuggingFace format

Nano converts on a single node (4 GPUs) with node-local EP — no Slingshot needed:

```bash
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export \
  /projects/a5k/public/checkpoints/megatron/nemotron_nano_quickstart_sft \
  --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning \
  --iteration 200
```

The converted model lands at
`.../nemotron_nano_quickstart_sft/iter_0000200/hf/` — a standard HF checkpoint
(safetensors + config + tokenizer) loadable with `AutoModelForCausalLM`.

---

### Step 6 — Run coherence tests

Nano fits on a single GPU for generation:

```bash
isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch \
  /projects/a5k/public/checkpoints/megatron/nemotron_nano_quickstart_sft/iter_0000200/hf
```

This generates responses to 8 diverse prompts and logs a table (prompt, response,
length, empty-flag) plus summary metrics to W&B. Read a few generations yourself —
`empty_pct == 0` alone does not catch drift, off-topic output, or template artifacts.

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
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; exec bash -i"
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
  T=\$(mktemp -d); cd \$T; python -m pytest $PWD/tests/unit_tests/ -x -q -m 'not pleasefixme' -n 8 --dist loadfile"
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
3. SFT and CPT load a pretrained checkpoint and train via finetune(); pretrain mode
   random-initializes from the NVIDIA pretrain recipes and trains via pretrain() —
   no checkpoint is loaded unless the YAML sets one

CPT, pretrain, and SFT additionally support **gradient routing (GRAM)**: a `gr:` section
in the YAML routes an N+1-corpus mix (retain + per-module forget corpora) so each forget
corpus trains only its removable auxiliary MLPs, which export-time baking merges into the
shared expert (forget-ON) or deletes for a byte-stock model (forget-OFF). CPT/pretrain
route `.bin/.idx` blend lists; SFT routes per-corpus finetuning dataset roots
(`gr.retain_dataset_root`/`gr.aux_dataset_roots`, packed corpus-pure under each root). The tooling — posture baking, posture verification,
and eval-only corpus loss probes — lives under `scripts/gradient_routing/`
(+ probe configs in `configs/gradient_routing/`); campaign training and
bake/verify configs live in the geodesic-configs repo under
`experiments/bedtime_stories/`. **No eval logic lives in this repo**: task definitions and
harnesses belong to `geodesic-evals` and `geodesic-environments`, which take a baked
posture directory like any other HF checkpoint. The full reference (method,
merge math, config and guard tables, workflow) is [docs/gradient-routing.md](docs/gradient-routing.md),
with the operational summary in the "Gradient routing (GRAM)" section of [CLAUDE.md](CLAUDE.md).

### Usage

```bash
# Via SLURM — extra args after the mode forward to the launcher: launcher flags
# (e.g. --disable-ft) are parsed as such, anything else falls through as Hydra overrides
isambard_sbatch --nodes=32 pipeline_training_submit.sbatch configs/<config>.yaml nano sft
isambard_sbatch --nodes=8  pipeline_training_submit.sbatch configs/<config>.yaml nano cpt
isambard_sbatch --nodes=32 pipeline_training_submit.sbatch configs/<config>.yaml nano pretrain --disable-ft
isambard_sbatch --nodes=16 pipeline_training_submit.sbatch configs/<config>.yaml super sft \
    --disable-ft train.train_iters=32 checkpoint.save=null

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

For a family of related configs, prefer a shared parent over N full copies: a
top-level `defaults: <path>` names one parent YAML (relative to the config file's
own directory, chains recurse), parents load first and the leaf overrides — so
each leaf carries only its deltas. Missing parents, cycles, and non-string refs
(there is no list form) fail loudly. Only `pipeline_training_run.py` resolves the
chain; keys read from the YAML by other tooling must live in the leaf itself.

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
| **Nano (30B-A3B), seq 8192** | 8 nodes / 32 GPUs: TP=2, EP=2, PP=4, DP=2 (GBS 16) | ~3.4 s/iter, ~27 TFLOP/s/GPU; zero hangs through 500+ iters |
| **Nano (30B-A3B), seq 32768** | 16 nodes / 64 GPUs: TP=1, CP=2, EP=4, PP=1, ETP=1 (GBS 128, the standard batch across quickstarts) | 76.31 ms/sample = 9.767 s/iter, peak 91.5 GB of 95 (GBS 256 remains the per-sample optimum within the cap: 71.74 ms/sample) — [`configs/quickstart/nemotron_nano_quickstart_sft.yaml`](configs/quickstart/nemotron_nano_quickstart_sft.yaml) |
| **Super (120B-A12B)** | TP=1, CP=(min that fits), EP=4, PP=22, ETP=1 | ~75-84 TFLOP/s/GPU, ~1000+ tok/s/GPU (≈2.4× the old TP=4 layouts) |
| **Super benchmark** | 16 nodes / 64 GPUs: TP=1, CP=4, EP=4, PP=8, ETP=1, DP=2 (seq 32K, GBS 128 — the standard batch across quickstarts since 2026-08-05) | 31.562 s/iter anchor = 167.4 TFLOP/s/GPU (`moe_experts_impl: torch_grouped`, optimizer CPU offload off; superseded, at the old GBS-64 workload: 17.099 = the paired A/B that certified `torch_grouped`, 20.66 on the `cublas_grouped` per-expert loop, 21.78 with offload 0.5) — the standing environment benchmark, [`configs/quickstart/nemotron_super_quickstart_sft.yaml`](configs/quickstart/nemotron_super_quickstart_sft.yaml) |
| **Super benchmark, 32 nodes** | 32 nodes / 128 GPUs: same topology, DP=4, **GBS 256** (scale the batch with the nodes) | 122.0 ms/sample = 31.228 s/iter, 169.2 TFLOP/s/GPU. With the base config at GBS 128 this override is matched µb/replica (64 both ends): perfect per-sample halving predicts 123.3 ms/sample vs 122.0 measured — scaling perfect within the ±2% cross-allocation placement band, same backend both ends — run as the 64-GPU config plus `train.global_batch_size=256`; the quickstarts are standardised at 64 GPUs and this is the one field that differs |
| **Ultra (550B-A55B)** | 72 nodes / 288 GPUs: TP=4, EP=4, PP=36, ETP=1 | ~28-30 s/iter steady state; first iter 45-75 min (lazy NCCL init at this depth) |
| **Nano pretrain (from scratch)** | 32 nodes / 128 GPUs: TP=1, CP=1, EP=4, PP=1, ETP=1, DP=128 (seq 8192, GBS 3072, 1B tokens) | 25.533 s/iter = 8.312 ms/sample (loss 12.20 → 7.58, 0 NaN; 59 GB weights-only checkpoint) — [`configs/quickstart/nemotron_nano_quickstart_pretrain.yaml`](configs/quickstart/nemotron_nano_quickstart_pretrain.yaml) |
| **Super pretrain (from scratch)** | 32 nodes / 128 GPUs: TP=1, CP=1, EP=4, PP=8, ETP=1, DP=16 (seq 8192, GBS 3072, 1B tokens) | 86.940 s/iter = 28.301 ms/sample (loss 12.19 → 7.65, 0 NaN; 225 GB weights-only checkpoint) — [`configs/quickstart/nemotron_super_quickstart_pretrain.yaml`](configs/quickstart/nemotron_super_quickstart_pretrain.yaml) |

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

# Pretraining-format corpus (.bin/.idx) — prepare then tokenize (exact token count included)
isambard_sbatch pipeline_data_submit.sbatch tokenize \
  /projects/a5k/public/data/<org>__<name> geodesic-research/nemotron-base-tokenizer tokenized_base

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
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
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
(ruff, pre-commit, the hook entry points — no torch; created by the installer below with
`uv pip install`, NEVER `uv sync`, which would resolve the full project and try to build the
training stack on the host),
never in the container. Install it once:

```bash
bash scripts/install_claude_tooling.sh
```

Configuration lives in `.claude/settings.json` (hooks) and `.claude/geodesic-config.yaml` (quality
items). The commit-time review gate is **enabled**: `git commit` runs pre-commit on the staged
files and blocks until the `checklist-reviewer` subagent writes a passing verdict for the
staged-diff hash; verdict-protection and a submodule-pin consistency check are wired in alongside
it. See [CLAUDE.md](CLAUDE.md#claude-code-tooling) for the full flow.

## Further Reading

- [Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/abs/2603.07685) — NVIDIA's paper on MoE parallelism, memory optimization, and FP8/FP4 training. Essential background for understanding the parallelism choices in this repo.
- [docs/environment.md](docs/environment.md) — The execution environment: install, design decisions, image qualification, troubleshooting
- [docs/profiling-quickstart.md](docs/profiling-quickstart.md) — Capturing and reading torch-profiler traces of a training run
- [CLAUDE.md](CLAUDE.md) — Cluster specs, per-model topology findings, campaign conventions, and dev commands
- [docs/README_DEFAULT.md](docs/README_DEFAULT.md) — Upstream Megatron Bridge README (supported models, API docs, etc.)
