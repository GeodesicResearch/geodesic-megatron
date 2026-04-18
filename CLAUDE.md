# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

NeMo Megatron Bridge is an NVIDIA PyTorch-native library that provides a bridge, conversion, and verification layer between HuggingFace and [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core). It enables bidirectional checkpoint conversion, pretraining, SFT, and LoRA for LLM and VLM models with Megatron Core's parallelism (tensor, pipeline, expert parallelism, FP8/BF16 mixed precision).

The primary package is `megatron.bridge` under `src/`. Megatron-Core is pinned as a git submodule at `3rdparty/Megatron-LM`.

## Cluster Overview (Isambard)

- **GPUs**: NVIDIA GH200 120GB (95GB usable), `sm_90`, 4 GPUs per node
- **CPU**: ARM aarch64 (Grace)
- **Networking**: Slingshot/CXI fabric (HPE)
- **CUDA**: 12.6, **Python**: 3.12, **PyTorch**: 2.11.0+cu126
- **Max reliable scale**: 32 nodes (128 GPUs). 64+ node runs hang due to Slingshot NCCL timeouts.

### Bad compute nodes

Isambard has the occasional hardware-broken node (hung VS Code tunnels, GPUs returning `ERR!`, NCCL dying on first collective, stuck IB ports). Because cluster turnover is slow, landing on a bad node can cost hours. The team maintains a shared TTL'd log that `isambard_sbatch` reads on every submission and passes to SLURM's `--exclude` automatically:

- **Shared log**: `/projects/a5k/public/isambard_sbatch_bad_nodes.log` — append-only, tab-separated, group-writable.
- **Entries expire after 7 days** (configurable via `ISAMBARD_SBATCH_BAD_NODES_TTL`).
- **Every submission** prints a `Bad nodes: N excluded (last 7d) — file: ...` summary line. If the line is missing, `isambard_sbatch` was bypassed (raw `/usr/bin/sbatch`).

**Claude should register a bad node when it is highly confident the failure is node-specific**, not a code/config bug or a cluster-wide issue. The wrapper exposes full CRUD over the log:

```bash
isambard_sbatch --mark-bad <node> "<short diagnosis>"   # Create: append new entry
isambard_sbatch --list-bad                              # Read: show active entries
isambard_sbatch --update-bad <node> "<new reason>"      # Update: replace reason + refresh TTL
isambard_sbatch --unmark-bad <node>                     # Delete: remove all entries for node
isambard_sbatch --prune-bad                             # Prune: drop expired/malformed lines
```

Prefer `--update-bad` over a second `--mark-bad` if the diagnosis changes (e.g., initial report was "tunnel hung" but follow-up shows "persistent GPU ECC" — use update so there's one clean current record instead of two conflicting ones). Run `--unmark-bad` if a node gets fixed sooner than the TTL (ops confirmation, a clean reboot, etc.) rather than waiting for the 7-day expiry.

**Register when** (high confidence):
- A VS Code tunnel never starts on a compute-node allocation and the allocation shows no output.
- `nvidia-smi` returns `ERR!` for specific GPUs on one host while siblings are healthy.
- NCCL fails on first collective on a single hostname while other hosts in the same job are fine.
- A job sits in RUNNING with zero log output far past expected start.
- `dmesg`/slurmd logs pin a hardware fault (Xid errors, IB link down) to a specific node.

**Do NOT register when**:
- The failure is a code, config, or library-version issue (OOM, bad YAML, missing import, wrong TP/EP). These will falsely exclude healthy nodes for a week and poison the list.
- The failure is cluster-wide (Slingshot congestion, the known ~7-min NCCL hang — `ft_launcher` handles that one).
- You can't tie the fault to a specific hostname. Without a `nidXXXXXX`, there's nothing to record.

**Finding the node name** — from inside a running job: `scontrol show hostnames $SLURM_JOB_NODELIST | head`. From SLURM records: `sacct -j <job_id> -o NodeList`. From `squeue`: the `%N`/`%R` format field.

The entry expires automatically after the TTL, so false positives are self-healing — but being conservative about what warrants a report keeps the cluster effective capacity high.

## Pipelines

All top-level scripts follow the `PIPELINE_ACTION.ext` naming convention. There are five pipelines:

| Pipeline | Submit (SLURM) | Launch / Logic | Purpose |
|----------|---------------|----------------|---------|
| **env** | `pipeline_env_submit.sbatch` | `pipeline_env_activate.sh`, `pipeline_env_setup.sh`, `pipeline_env_validate.py` | Environment install, activation, validation |
| **training** | `pipeline_training_submit.sbatch` | `pipeline_training_launch.sh` | SFT and CPT distributed training |
| **data** | `pipeline_data_submit.sbatch` | `pipeline_data_prepare.py` | Dataset download, tokenization, packing |
| **checkpoint** | `pipeline_checkpoint_submit.sbatch` | `pipeline_checkpoint_convert.sh`, `pipeline_checkpoint_convert_hf.py` | Megatron↔HF conversion, Hub upload |
| **coherence** | `pipeline_coherence_submit.sbatch` | `pipeline_coherence_test.py` | Qualitative generation testing, W&B logging |

Each pipeline has a thin `PIPELINE_submit.sbatch` for SLURM allocation and a `.sh`/`.py` with the actual logic. The `.sh` launchers can also be called directly from an interactive `salloc`.

---

## 1. Environment Pipeline (`env_*`)

### Files

| File | Purpose |
|------|---------|
| `pipeline_env_activate.sh` | Universal environment: venv, compilers, NVIDIA libs, GPU settings, cache paths. **Source this before any work.** |
| `pipeline_env_setup.sh` | Full bare-metal install script (must run on a compute node with GPU) |
| `pipeline_env_validate.py` | 15-check validation (imports, CUDA, GPU ops, recipes, training) |
| `pipeline_env_submit.sbatch` | SLURM wrapper for setup/validation (needs GPU) |

### Usage

```bash
# Activate (from any node)
source pipeline_env_activate.sh

# Install from scratch (requires compute node)
isambard_sbatch pipeline_env_submit.sbatch setup

# Validate
isambard_sbatch pipeline_env_submit.sbatch validate --run-training
```

### What `pipeline_env_activate.sh` sets

**Universal GPU settings** (needed for any operation):
- `UB_SKIPMC=1` — Disables CUDA Multicast (Isambard driver doesn't support it)
- `CUDA_DEVICE_MAX_CONNECTIONS=1` — Required for TP/SP comm-compute overlap
- `NVTE_CPU_OFFLOAD_V1=1` — TE activation offloading V1 code path
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — Reduces CUDA memory fragmentation

**Shared cache paths:**
- `HF_HOME=/projects/a5k/public/hf`
- `WANDB_DIR=/projects/a5k/public/logs/wandb`
- `NEMO_HOME=/projects/a5k/public/data/nemo_cache`

Every env var in `pipeline_env_activate.sh` has detailed inline documentation.

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
| `pipeline_training_launch.sh` | Shared launcher: NCCL/CXI env vars, fault tolerance, srun + ft_launcher |
| `pipeline_training_submit.sbatch` | Thin SLURM wrapper: allocates nodes, calls `pipeline_training_launch.sh` |

Training script (called by the launcher):
- `pipeline_training_run.py` — Unified entry point for SFT and CPT (dispatches via `--model nano|super --mode sft|cpt`)

### Usage

```bash
# Via SLURM (allocates nodes)
isambard_sbatch --nodes=32 pipeline_training_submit.sbatch configs/<config>.yaml nano sft
isambard_sbatch --nodes=8  pipeline_training_submit.sbatch configs/<config>.yaml nano cpt

# Via salloc (interactive)
salloc --nodes=16 --gpus-per-node=4 --time=24:00:00 --exclusive
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft
bash pipeline_training_launch.sh configs/<config>.yaml --model super --mode cpt --max-samples 50000
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --nodes 8 --nodelist node[001-008]
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --disable-ft
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --peft lora
```

`pipeline_training_launch.sh` options: `--model nano|super` (required), `--mode sft|cpt` (required), `--disable-ft`, `--enable-pao`, `--peft lora`, `--max-samples N`, `--nodes N`, `--nodelist LIST`.

### Environment Variable Architecture

`pipeline_training_launch.sh` adds distributed-training-only vars on top of `pipeline_env_activate.sh`:
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

**ft_launcher timeout configuration** (set in `pipeline_training_launch.sh`):
- `--ft-rank-section-timeouts=setup:1800,step:3600,checkpointing:600`
- `--ft-rank-out-of-section-timeout=3600` — must be ≥3600s for first-iter NCCL lazy init with PP=8+
- `calc_ft_timeouts=True` auto-learns step timeouts after first successful run. **Delete `ft_state.json`** from checkpoint dir if learned timeouts are too aggressive after config changes.

The `ft`/`nvrx_straggler`/`inprocess_restart` Python configs **cannot** be set via YAML or Hydra overrides (OmegaConf merge creates dicts, not dataclasses). They are set in `pipeline_training_run.py` via the `--enable-ft` flag (on by default). Use `--disable-ft` to opt out.

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

### Launching training from a login node (salloc shell lost)

If the VS Code tunnel / salloc shell that holds the allocation dies but the allocation is still alive (`SLURM_JOB_ID` still valid in `squeue`), `pipeline_training_launch.sh` won't run from a plain login shell without manually reconstituting the env SLURM would have set. Export the following **in the login shell**, then `bash pipeline_training_launch.sh …` attaches correctly via `srun --jobid=… --overlap`:

```bash
export SLURM_JOB_ID=3823383                               # existing alloc id
export SLURM_NNODES=16
export SLURM_NODELIST='nid[010229-010232,010303,...]'     # full nodelist from scontrol show job
export SLURM_JOB_NODELIST="$SLURM_NODELIST"
export SLURM_NTASKS=16
export SLURM_JOB_NUM_NODES=16
export SLURM_NPROCS=16
export SLURM_GPUS_PER_NODE=4                              # required — pipeline_training_launch.sh:464 uses this for torchrun --nproc_per_node
export SLURM_GPUS_ON_NODE=4
export SLURM_CLUSTER_NAME=gracehopper                     # required — ft_launcher OneLoggerConfig pydantic-validates this and crashes on None
export SLURM_SUBMIT_HOST=login01
bash pipeline_training_launch.sh <config.yaml> --model super --mode sft
```

**Why each is needed:**
- `SLURM_JOB_ID` + `SLURM_NNODES` + `SLURM_NODELIST` — `pipeline_training_launch.sh:114` aborts without `SLURM_JOB_ID`, then uses the node vars to build the `srun --nodes=N --overlap` command
- `SLURM_GPUS_PER_NODE` — empty string here produces `torchrun --nproc_per_node= …` → `ValueError: Unsupported nproc_per_node value`
- `SLURM_CLUSTER_NAME` — `nvidia_resiliency_ext/shared_utils/profiling.py:79` reads this and feeds into a pydantic model that rejects `None`. Use `scontrol show config | grep ClusterName` to get the value (`gracehopper` on Isambard).

**Between retry attempts** clean up leftover state:
- `pkill -9 -f "<launcher-name>|pipeline_training_launch"` to kill zombie ft_launcher workers
- `rm` the stale `*_train.out` logs — coordinators that tail them may read old error markers and early-exit
- `rm -rf <save_ckpt_dir>` if an empty checkpoint dir was created — orchestrators that poll `latest_checkpointed_iteration.txt` may mistake its presence for training completion

All `SLURM_*` vars propagate to workers through `srun --export=ALL` (already the default in `pipeline_training_launch.sh:451`) once exported in the launcher shell.

---

## 3. Data Pipeline (`data_*`)

### Files

| File | Purpose |
|------|---------|
| `pipeline_data_prepare.py` | Download HF datasets, tokenize, export JSONL, pack sequences |
| `pipeline_data_submit.sbatch` | SLURM wrapper for offline packing (1 node, 1 GPU) |

### Usage

```bash
# Prepare dataset (download + tokenize + pack)
python pipeline_data_prepare.py --dataset allenai/Dolci-Instruct-SFT --seq-length 8192

# Offline packing only (via SLURM)
isambard_sbatch pipeline_data_submit.sbatch \
  /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 8192 1

# From salloc
source pipeline_env_activate.sh
python scripts/data/pack_sft_dataset.py \
  --dataset-root /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
  --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --seq-length 8192 --pad-seq-to-mult 1
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
# Export Megatron → HF
isambard_sbatch pipeline_checkpoint_submit.sbatch export \
  /projects/a5k/public/checkpoints/megatron/<experiment> --iteration 300 --push-to-hub

# Import HF → Megatron (4 nodes for Super)
isambard_sbatch --nodes=4 pipeline_checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Upload all iterations (with polling for ongoing training)
isambard_sbatch --time=24:00:00 pipeline_checkpoint_submit.sbatch upload-all \
  /projects/a5k/public/checkpoints/megatron/<experiment> --poll

# From salloc
bash pipeline_checkpoint_convert.sh export /path/to/ckpts --iteration 300 --push-to-hub
```

### How export works

1. Reads `latest_checkpointed_iteration.txt` or `--iteration N` to find the `iter_XXXXXXX` directory
2. Auto-detects the HF model ID from `run_config.yaml`
3. Converts via `AutoBridge.from_hf_pretrained` + `load_megatron_model` + `save_hf_pretrained` (multi-GPU via torchrun)
4. Saves to `<megatron-path>/iter_XXXXXXX/hf/`
5. Optionally pushes to HuggingFace Hub on a revision branch (`iter_0000300`)

The `torch_dist` checkpoint format supports resharding — conversion parallelism is independent of training parallelism.

### Recommended export settings

Both Nano and Super conversions run on a **single node** (4 GPUs). All EP communication stays on NVLink — no Slingshot needed.

**Nemotron 3 Nano (30B-A3B):**
```bash
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export /path/to/ckpts --iteration 400
# Or directly:
torchrun --nproc_per_node=4 pipeline_checkpoint_convert_hf.py \
  --megatron-path /path/to/ckpts --iteration 400 --tp 1 --ep 4
```

**Nemotron 3 Super (120B-A12B):**
```bash
torchrun --nproc_per_node=4 pipeline_checkpoint_convert_hf.py \
  --megatron-path /path/to/ckpts --iteration 490 --tp 1 --ep 4 --not-strict
```

- **`--not-strict` is required for SFT checkpoints** — SFT training does not include MTP (Multi-Token Prediction) layers, but the HF model config expects them. Without `--not-strict`, shards containing MTP keys are silently dropped, which also drops `lm_head.weight` and `backbone.norm_f.weight` (critical for generation). With `--not-strict`, incomplete shards are saved with available tensors; MTP weights are randomly initialized but unused during standard generation.
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

# Directly (no SLURM, uses local GPUs)
source pipeline_env_activate.sh
python pipeline_coherence_test.py <model_path> [--max-tokens 3000] [--system-prompt "..."]
```

### What it does

1. Loads an HF model (Hub ID or local path) with `device_map="auto"` for multi-GPU
2. Generates responses to 8 diverse prompts at `temperature=1.0`, `max_new_tokens=3000`
3. Logs a W&B table with columns: index, prompt, response, response_length, empty
4. Reports summary metrics: total_generations, empty_count, empty_pct

### W&B run naming

- **Hub models** (e.g., `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`): `gen-test-NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
- **Local checkpoints** (e.g., `.../my_experiment/iter_0000400/hf`): `gen-test-my_experiment__iter_0000400__hf`

### Notes

- **Nano (30B)**: fits on 1 GPU. Use `--gpus-per-node=1`.
- **Super (120B)**: needs 4 GPUs with `device_map="auto"`.
- **MTP weights**: SFT checkpoints lack MTP layers. Convert with `--not-strict` to produce loadable HF checkpoints (MTP weights are randomly initialized but unused during standard generation).

---

## Running Evals (sfm-evals repo)

Evals are run via the [sfm-evals](https://github.com/GeodesicResearch/sfm-evals) repo at `/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals`. Uses `just` (command runner) to orchestrate lm_eval and Inspect AI evals via Slurm on Isambard.

**Pre-requisites for new geodesic-research HF models:**
1. Upload `configuration_nemotron_h.py` and `modeling_nemotron_h.py` to the HF repo
2. Fix `tokenizer_config.json`: `"tokenizer_class": "TokenizersBackend"` → `"PreTrainedTokenizerFast"`
3. For 120B+ models: pre-download to shared HF cache first
4. Add model alias to `just/models.yaml` in the sfm-evals repo

### Eval Commands (from sfm-evals repo)

```bash
cd /lus/lfs1aip2/projects/public/a5k/repos/sfm-evals
```

#### Instruct/DPO Models — Open-ended (Slurm)
```bash
# Standard eval (submits Slurm job with vLLM)
just submit-instruct-open-isambard MODEL configs/lm_eval/instruct/mcq_open/CONFIG

# With specific system prompt (instead of all 5)
just submit-instruct-open-isambard MODEL configs/lm_eval/instruct/mcq_open/CONFIG --system-prompt=ai_p_inst

# With checkpoint branch
just submit-instruct-open-isambard MODEL configs/lm_eval/instruct/mcq_open/CONFIG --checkpoints=step_200

# Loop over checkpoints
for step in 200 400 600 800 1000; do
  ISAMBARD_TIME="4:00:00" just submit-instruct-open-isambard MODEL configs/lm_eval/instruct/mcq_open/hdrx_sfm_syn --checkpoints=step_${step} --system-prompt=ai_p_inst
done
```

#### Base Models — MCQ (Slurm)
```bash
just submit-base-mcq-isambard-vllm MODEL configs/lm_eval/base/mcq_alignment/hdrx_sfm
```

#### Inspect AI Evals — API-based (no Slurm needed)

These use `inspect eval` via API providers (Together, Anthropic, etc.). MODEL is in provider format, e.g. `together/openai/gpt-oss-120b` or `anthropic/claude-haiku-4-5-20251001`.

Env vars required: `WANDB_PROJECT`, `WANDB_ENTITY` (set in `just/utils.just`), plus provider API keys (`TOGETHER_API_KEY`, `ANTHROPIC_API_KEY`).

```bash
# Smoke suites (5 samples each, fast validation)
just run-smoke-all-api MODEL

# Quick suites (~30-45 min) — the go-to for fast iteration
just run-quick-alignment-api MODEL        # 7 alignment evals
just run-quick-capability-api MODEL       # 4 capability evals
just run-quick-all-api MODEL              # Both alignment + capability in one command

# Full suites (paper-default sample counts)
just run-full-all-api MODEL

# With system prompt override
just run-quick-alignment-api MODEL "You are a helpful AI assistant."

# With custom judge model
just run-quick-all-api MODEL "" "anthropic/claude-haiku-4-5-20251001"
```

**`run-quick-all-api`** is the primary command for evaluating a new model. It runs 11 evals sequentially (~30-45 min total):

| Block | Eval | Samples | W&B Group |
|-------|------|---------|-----------|
| Alignment | sfm_ind | 100 | `quick_alignment__<model>` |
| Alignment | sfm_hdrx | 100 | |
| Alignment | goals | 50 | |
| Alignment | exfil_offer | 20 | |
| Alignment | frame_colleague | 20 | |
| Alignment | monitor_disruption | 20 | |
| Alignment | emergent_misalignment | 6/question | |
| Capability | tiny_mmlu | 100 | `quick_capability__<model>` |
| Capability | tiny_gsm8k | 100 | |
| Capability | ifeval | 100 | |
| Capability | aime2025 | 30 | |

Each eval logs to W&B via `inspect_wandb_wrapper.py`. Individual eval failures don't abort the suite (`|| true`).

#### Submitting Inspect Suites to Isambard (vLLM on Slurm)

For HuggingFace models (not API-hosted), use the `submit-*-isambard` variants. These submit a Slurm job that:
1. Starts a vLLM server on the allocated node(s)
2. Waits for the server to pass health checks
3. Runs the inspect eval suite against the local vLLM endpoint
4. Logs results to W&B

**Tensor parallelism**: Set `VLLM_TP` env var to control GPUs per vLLM server. Defaults to 1.
- **Nano (30B)**: `VLLM_TP=1` (default) — 1 GPU
- **Super (120B)**: `VLLM_TP=4` — 4 GPUs, uses ray distributed backend

```bash
# Nano (30B) — TP=1 (default)
just submit-quick-all-isambard geodesic-research/nemotron_nano_sft_warm_start_200k
ISAMBARD_TIME="4:00:00" just submit-full-all-isambard geodesic-research/nemotron_nano_sft_warm_start_200k

# Super (120B) — TP=4
VLLM_TP=4 ISAMBARD_TIME="4:00:00" just submit-quick-all-isambard geodesic-research/nemotron_super_200k_warm_start_sft
VLLM_TP=4 ISAMBARD_TIME="8:00:00" just submit-full-all-isambard geodesic-research/nemotron_super_200k_warm_start_sft

# Other available submit variants
just submit-quick-alignment-isambard MODEL
just submit-quick-capability-isambard MODEL
just submit-full-alignment-isambard MODEL
just submit-smoke-all-isambard MODEL
```

**Monitoring submitted jobs:**
```bash
# Check job status
squeue -u $USER -o "%.10i %.40j %.8T %.10M %.6D %R" | grep eval

# Check completed/failed jobs
sacct -j JOBID --format=JobID,JobName%30,State,ExitCode,Elapsed -n

# Tail a running job's log
tail -f /projects/a5k/public/logs_${USER}/open-instruct/ckpt-evals/bundled-eval-JOBID.out
```

W&B groups follow the pattern `run-{suite}-api__{model_short}`, e.g.:
- `run-quick-all-api__nemotron_nano_sft_warm_start_200k`
- `run-full-all-api__nemotron_super_200k_warm_start_sft`

#### Raw Bundled Checkpoint Evals (Slurm)
```bash
isambard_sbatch --time=8:00:00 --gpus-per-node=1 \
  run_bundled_checkpoint_eval.sbatch "geodesic-research/model_name" manifests/eval.json
```

### Misalignment Config Choices
- `hdrx_sfm_syn` — 1503 samples/task, supports think-tags (shorter, preferred)
- `ind_sfm_syn` — 2671 samples/task, supports think-tags (longer)
- `hdrx_sfm_no` / `ind_sfm_no` — standard (no think-tag stripping)

Each config has 8 tasks: `forward_misalignment_v{1-4}` + `reverse_misalignment_v{1-4}`. Default runs all 5 system prompts.

### Time Limits
- `ISAMBARD_TIME` env var controls sbatch time limit (default: `8:00:00`)
- Early checkpoints (short responses): 1-2hr usually enough
- Later checkpoints / thinking models: use 4hr+ (`ISAMBARD_TIME="4:00:00"`)
- **20-node limit** per user — don't submit more than 20 jobs at once

### Useful Helpers
```bash
just list-models                          # Model aliases
just list-groups                          # Model groups (E2E_BASE, E2E_INSTRUCT, etc.)
just show-plan GROUP TASKS [FLAGS]        # Dry run
just --list                               # All recipes
```

### Results
- W&B: "Self-Fulfilling Model Organisms - ITERATED Evals" (entity: geodesic)
- Result JSONs: `results/logs/open_ended_rollouts/ORG__MODEL_NAME/results_*.json`
- Slurm logs: `/projects/a5k/public/data_cwtice.a5k/logs/sfm-evals/sfm-eval-{JOB_ID}.out`
- **Always filter W&B by group name** — runs from different checkpoints are mixed otherwise

---

## NCCL Performance Testing

nccl-tests at `/home/a5k/kyleobrien.a5k/nccl-tests/` for benchmarking Slingshot bandwidth:

```bash
source pipeline_env_activate.sh
export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn
srun --nodes=2 --ntasks-per-node=1 --export=ALL bash -c \
  "source pipeline_env_activate.sh && /home/a5k/kyleobrien.a5k/nccl-tests/build/all_reduce_perf -b 32K -e 8G -f 2 -g 4"
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

## Checkpoint Save Policy

- **Standard SFT and EM fine-tuning**: Set `save_interval: 1000000` to skip intermediate checkpoints. Megatron-Core always saves a final checkpoint when `train_iters` is reached, so this effectively means "save only at end of training."
- **Long CPT runs and reasoning/thinking training**: Use a reasonable `save_interval` (e.g., 100) for fault recovery — these runs take hours/days and losing progress is costly.
- **Rationale**: SFT/EM runs are short (100-500 iters, minutes) and cheap to restart. Intermediate checkpoints waste disk and I/O time. Reasoning/thinking runs are long and need periodic saves for resumption.

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
| `OSError: [Errno 116] Stale file handle` | `TRITON_CACHE_DIR`/`TMPDIR` to node-local `/tmp` (automatic in `pipeline_training_launch.sh`) |
| NCCL hangs every ~7-8 min | Slingshot fabric issue. ft_launcher auto-restarts. |
| EP=4 OOMs on GH200 | Use EP=8 (16 experts/GPU = 51GB vs 32 = 93GB). |
| `nemo_experiments/` fills disk | Selectively remove old TB logs. **Do NOT `rm -rf`** — contains checkpoint resume state. |
