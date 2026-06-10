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
- **Scale**: cross-node EP=8 MoE all-to-all hits the documented Slingshot/aws-ofi-nccl Send/Recv hang (`docs/investigations/slingshot-nccl-hang-investigation.md`) — keep **TP×EP ≤ 4** (node-local) to avoid it. With node-local EP, scale is NOT capped at 32 nodes: **Ultra SFT is validated at 72 nodes / 288 GPUs** (PP=36). The prior "64+ nodes just hang" belief conflated that Slingshot hang with two Ultra-specific first-iter issues since fixed (`disable_jit_fuser` + a longer `TORCH_NCCL_TIMEOUT`; see the Ultra section).

### Bad compute nodes

`isambard_sbatch` reads a shared TTL'd log at `/projects/a5k/public/isambard_sbatch_bad_nodes.log` (7-day expiry, configurable via `ISAMBARD_SBATCH_BAD_NODES_TTL`) and auto-passes excluded nodes to SLURM's `--exclude`. Every submission prints `Bad nodes: N excluded (last 7d) — file: ...`; missing line means raw `/usr/bin/sbatch` was used.

```bash
isambard_sbatch --mark-bad <node> "<short diagnosis>"   # append entry
isambard_sbatch --list-bad                              # show active entries
isambard_sbatch --update-bad <node> "<new reason>"      # replace reason + refresh TTL
isambard_sbatch --unmark-bad <node>                     # remove entries for node
isambard_sbatch --prune-bad                             # drop expired/malformed lines
```

**Register only when you can pin the failure to a specific hostname** (Xid in dmesg, `nvidia-smi` ERR! on one host while siblings are healthy, NCCL fails on first collective on a single hostname, tunnel never starts on its allocated node, RUNNING with no log output). **Do NOT register** code/config bugs (OOM, bad YAML, wrong TP/EP) or cluster-wide issues (Slingshot congestion, the known ~7-min NCCL hang — `ft_launcher` handles that). Prefer `--update-bad` over a duplicate `--mark-bad`; `--unmark-bad` if a node is fixed before TTL.

Find node names: `scontrol show hostnames $SLURM_JOB_NODELIST`, `sacct -j <id> -o NodeList`, or `squeue` `%N`/`%R`.

### Project storage quota

`isambard_sbatch` prints a **project storage quota report** on every submission — per-path Lustre quota usage (`<path>  used/limit (pct%)`, flagged ` — nearly full` at ≥90%, plus inode counts) via the documented recipe `lfs quota -p $(lfs project -d <DIR> ...) <DIR>`. Example line: `Storage: /projects/a5k  188.6T / 200.0T (94%)  files: 6.2M / 50.0M (12%) — nearly full`. **Determine free storage from this report, not from `df`.** The project quota (`/projects/a5k`, 200 T) is what actually limits writes — and it runs hot (often ~94%). `df -h /lus/lfs1aip2` instead reports the whole shared Lustre filesystem (~21 PB, ~36% used), so it makes storage look nearly empty and completely hides that the project quota is almost full — the opposite of the truth. Tune with `ISAMBARD_SBATCH_STORAGE_PATHS` (default `/projects/<account>`), `ISAMBARD_SBATCH_STORAGE_WARN_PCT` (default 90), or skip with `ISAMBARD_SBATCH_STORAGE_DISABLED=1`. Like the bad-nodes report, it never blocks a submission, so watch it — at ~94% a large checkpoint/download can hit the quota.

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

### Nemotron 3 Ultra (550B-A55B) on Isambard

Ultra is architecturally a scaled Super — same NemotronH hybrid (Mamba2 + attention + Latent MoE) with MTP and 512 routed experts, but 108 layers and hidden 8192. HF id `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` (base: `…-Base-BF16`). Recipe: `nemotron_3_ultra_{pretrain,sft,peft}_config`; train via `pipeline_training_submit.sbatch <config> ultra sft`.

**SFT validated end-to-end on 72 nodes / 288 GPUs** (quickstart 2026-06-05; full 495-iter warm-start SFT 200k 2026-06-09: lm loss 0.90→0.46, 0 NaN, ~28 s/iter, ~21 TFLOP/s/GPU). At 550B total (~1.1 TB in BF16) Ultra is ~5× the Super. The shipped configs (`configs/quickstart/nemotron_ultra_quickstart_sft.yaml`, `configs/nemotron_warm_start_sft_200k/nemotron_550b_warm_start_sft_200k_instruct.yaml`): **TP=4, EP=4, PP=36, ETP=1** (parallel folding → EP+TP both NVLink-node-local; only PP crosses Slingshot), pure BF16 (no FP8/FP4 — MoE routing crashes), precision-aware optimizer with BF16 Adam moments (mandatory at this size), selective recompute (`core_attn,moe,shared_experts`). PP=36 divides the 108 layers (3/stage). Per-GPU memory: ~60 GB on the heavy MoE stages, ~30 GB on Mamba/attn stages (the hybrid clusters MoE layers onto every ~3rd stage → 2× heavier). Measured: **iter 1 ≈ 52 min** (one-time lazy NCCL comm-init at this depth/rank-count), **steady-state ≈ 30 s/iter**, loss healthy, 0 NaN.

**Two non-obvious requirements (the bring-up bit hard on both — see `docs/investigations/ultra-pipeline-init-hang-debug-log.md`):**
1. **`dist.disable_jit_fuser: true`** (in the configs). On torch ≥ 2.2 Megatron's `jit_fuser` = `torch.compile`; at PP=36 the hybrid per-stage layer mix makes first-step JIT compile times diverge → ranks desync (some compiling, others at a barrier) → watchdog. Eager fused ops avoid it. The earlier "64+ nodes hang" symptom was THIS (and the slow first iter below), **not** the Slingshot MoE-alltoall hang documented in `slingshot-nccl-hang-investigation.md`.
2. **Long first-iter timeouts — including Megatron's own process-group timeout.** The first iteration's lazy NCCL comm-init takes **45–75 min** (fabric-load dependent) at PP=36/288 ranks. THREE knobs must all cover it: `dist.distributed_timeout_minutes: 90` in the YAML (Megatron creates its process groups with this timeout — the old 30 was marginal and a busy fabric reproducibly times out the first `recv_forward` at exactly 30:00; `TORCH_NCCL_TIMEOUT` alone does NOT cover it), `TORCH_NCCL_TIMEOUT=7200`, and ft `step`/`out-of-section`=7200 (both defaulted in `pipeline_training_launch.sh`). Steady-state then drops to ~28 s/iter.

**Throughput is best-effort, not yet tuned.** PP=36 with GBS=64/DP=2 → 32 microbatches < 36 stages = severe pipeline bubble (~0.2→low TFLOP/s/GPU). To improve: raise `global_batch_size` so microbatches ≥ PP, consider VPP/interleaved PP, and set `pipeline_model_parallel_layout` to balance the 2×-heavy MoE stages (see the Megatron MoE paper skill). Functionally it trains; these are throughput levers.

**Conversion needs multiple nodes.** 1.1 TB of BF16 weights does NOT fit Super's single-node (4×95 GB) export path — pass `--nodes` ≥ 4 to `pipeline_checkpoint_submit.sbatch import`/`export` and keep EP node-local. Base coherence (`pipeline_coherence_test.py --generation-mode completion`) likewise needs ≥3 nodes for inference. Warm-start SFT loads the base Megatron checkpoint directly. **Unlike Super, the Ultra base already ships non-zero chat-special-token embeddings** (verified with `scripts/init_base_chat_embeddings.py`: only 1 unused-token row is near-zero, and it is zero in Instruct too), so **no Base-Chat-Init graft is needed** (Super needed it to avoid the bucket-#0 Inf; see "Tokenizer choice for Base CPT").

**Coherence / generation for the 550B: use Megatron-native inference, not vLLM.** vLLM (0.19) cannot serve the BF16 hybrid here: PP>1 hits a hybrid-Mamba KV-cache bug (`KeyError: model.layers.N.mixer` at PP stage boundaries), PP=1 caps TP at the Mamba `n_groups=8` (8×95 GB < 1.1 TB), and FP8/NVFP4 workarounds die on a CXI load timeout / driver PTX rejection. Instead run `isambard_sbatch --nodes=6 scripts/coherence_megatron_submit.sbatch <megatron-ckpt-dir>` — bridge-loads the Megatron checkpoint at TP=4/EP=4/PP=6 (torch_dist reshards), applies the instruct chat template, greedy-generates the standard 8 prompts, logs to W&B (~35 min; no HF export needed). Full guide: `docs/ultra-550b-training-and-conversion.md`.

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

If the tunnel/salloc shell dies but `SLURM_JOB_ID` is still in `squeue`, export the SLURM env vars manually so `pipeline_training_launch.sh` can attach via `srun --jobid=… --overlap`:

```bash
export SLURM_JOB_ID=<id> SLURM_NNODES=<n> SLURM_NODELIST='<from scontrol show job>'
export SLURM_JOB_NODELIST="$SLURM_NODELIST" SLURM_NTASKS=<n> SLURM_JOB_NUM_NODES=<n> SLURM_NPROCS=<n>
export SLURM_GPUS_PER_NODE=4 SLURM_GPUS_ON_NODE=4   # else torchrun --nproc_per_node is empty
export SLURM_CLUSTER_NAME=gracehopper               # ft_launcher OneLoggerConfig pydantic-rejects None
export SLURM_SUBMIT_HOST=login01
bash pipeline_training_launch.sh <config.yaml> --model super --mode sft
```

Between retries: `pkill -9 -f "pipeline_training_launch"`, `rm` stale `*_train.out` logs, `rm -rf <save_ckpt_dir>` if an empty checkpoint dir was created (orchestrators may read its `latest_checkpointed_iteration.txt` as completion).

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
# Export Megatron → HF (--hf-model and --reasoning|--no-reasoning are REQUIRED)
isambard_sbatch pipeline_checkpoint_submit.sbatch export \
  /projects/a5k/public/checkpoints/megatron/<experiment> \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
  --iteration 300 --push-to-hub

# Import HF → Megatron (4 nodes for Super)
isambard_sbatch --nodes=4 pipeline_checkpoint_submit.sbatch import nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

# Upload all iterations (with polling for ongoing training)
isambard_sbatch --time=24:00:00 pipeline_checkpoint_submit.sbatch upload-all \
  /projects/a5k/public/checkpoints/megatron/<experiment> \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning --poll

# From salloc
bash pipeline_checkpoint_convert.sh export /path/to/ckpts \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
  --iteration 300 --push-to-hub
```

### How export works

1. Reads `latest_checkpointed_iteration.txt` or `--iteration N` to find the `iter_XXXXXXX` directory
2. Uses the `--hf-model` you pass (the upstream HF model ID whose architecture + tokenizer this checkpoint should be exported against — there is no auto-detection)
3. Converts via `AutoBridge.from_hf_pretrained` + `load_megatron_model` + `save_hf_pretrained` (multi-GPU via torchrun)
4. Saves to `<megatron-path>/iter_XXXXXXX/hf/`
5. Optionally pushes to HuggingFace Hub on a revision branch (`iter_0000300`)

For chained training (CPT → SFT → EM → …), pass the architectural-root HF id — e.g. an SFT checkpoint that loaded from a `*_cpt_v2` dir still exports against `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` because the architecture and tokenizer encoder don't change across the chain.

The `torch_dist` checkpoint format supports resharding — conversion parallelism is independent of training parallelism.

### Recommended export settings

Both Nano and Super conversions run on a **single node** (4 GPUs). All EP communication stays on NVLink — no Slingshot needed.

**Nemotron 3 Nano (30B-A3B):**
```bash
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export /path/to/ckpts \
  --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning --iteration 400
# Or directly:
torchrun --nproc_per_node=4 pipeline_checkpoint_convert_hf.py \
  --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning \
  --megatron-path /path/to/ckpts --iteration 400 --tp 1 --ep 4
```

**Nemotron 3 Super (120B-A12B):**
```bash
torchrun --nproc_per_node=4 pipeline_checkpoint_convert_hf.py \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --no-reasoning \
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

Evals live in the [sfm-evals](https://github.com/GeodesicResearch/sfm-evals) repo at `/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals`; see that repo's README for the full command reference. Quick orientation:

- **Pre-reqs for new `geodesic-research` HF models**: upload `configuration_nemotron_h.py` / `modeling_nemotron_h.py`, set `tokenizer_config.json` `"tokenizer_class": "PreTrainedTokenizerFast"`, pre-download 120B+ to shared HF cache, add alias to `just/models.yaml`.
- **Primary commands**: `just submit-instruct-open-isambard MODEL CONFIG` (vLLM on Slurm), `just run-quick-all-api MODEL` (~30–45 min, 11 evals via API), `just submit-quick-all-isambard MODEL` (HF model on Slurm — set `VLLM_TP=4` for 120B). `ISAMBARD_TIME` controls sbatch limit (default `8:00:00`); 20-job-per-user limit on Isambard.
- **Misalignment configs**: `hdrx_sfm_syn` (1503/task, preferred), `ind_sfm_syn` (2671/task); each has 8 tasks `forward/reverse_misalignment_v{1-4}` × 5 system prompts.
- **Results**: W&B project "Self-Fulfilling Model Organisms - ITERATED Evals" (entity `geodesic`) — always filter by group name. Slurm logs at `/projects/a5k/public/data_cwtice.a5k/logs/sfm-evals/`.

---

## NCCL Performance Testing

### Debugging NCCL-looking failures (rendezvous timeout, hang, slow iters)

When a training run fails with symptoms that *might* be fabric-related — c10d KV-store rendezvous timeout ("N/M clients joined"), NCCL watchdog timeout mid-iteration, iters suddenly taking 10-20× longer than expected, `WorkNCCL(SeqNum=...)` timing out — run the benchmark suite **inside the same allocation** to prove whether NCCL/Slingshot itself is at fault. If the benchmark passes, the fabric is healthy and the failure is elsewhere (leftover zombie processes, rendezvous port collision, config mismatch, parallel-run contention).

**Repo**: `/home/a5k/kyleobrien.a5k/isambard-nccl-tests/` — Python orchestrator over upstream nccl-tests with pass/fail thresholds for Isambard GH200. Binaries are already built at `build/`.

**Usage (inside the affected SLURM allocation, e.g. the tunnel that just had a training failure):**
```bash
cd /home/a5k/kyleobrien.a5k/isambard-nccl-tests
module purge && module load PrgEnv-cray cuda/12.6 brics/aws-ofi-nccl/1.8.1
python scripts/run_nccl_benchmarks.py --min-nodes 2 --max-nodes 8 --no-wandb
# (raise --max-nodes to the allocation size if you want the full sweep)
```

Runs ~20 min for the 2..8 sweep. Tests 5 collectives (alltoall, all_reduce, reduce_scatter, all_gather, sendrecv) at each node count against calibrated thresholds (~80% of observed baseline). "PASS" on ≥ the node count of the failing run is strong evidence the fabric is fine.

**Interpreting the result:**
- **All PASS** → NCCL is healthy. Failure was almost certainly at the process layer (zombies, rendezvous port collision, bad config, parallel-run fabric saturation from *multiple* PP=4 training jobs, etc.). Clean up zombie ft_launcher/torchrun/pipeline_training processes and relaunch, optionally with a different `MASTER_PORT_OVERRIDE`.
- **Consistent FAIL on one node count** → capacity issue at that scale — try a different node subset of the allocation.
- **FAIL scattered across collectives/scales** → bad specific node(s). `isambard_sbatch --mark-bad <node> "<reason>"` and move on.

**Typical healthy numbers on a clean allocation (2026-04-22)**: 8-node / 32-GPU all_gather bus_bw ≈ 86 GB/s (threshold 55), alltoall / all_reduce / reduce_scatter all comfortably above threshold, zero errors.

### Raw (legacy) one-shot measurement
```bash
source pipeline_env_activate.sh
export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn
srun --nodes=2 --ntasks-per-node=1 --export=ALL bash -c \
  "source pipeline_env_activate.sh && /home/a5k/kyleobrien.a5k/nccl-tests/build/all_reduce_perf -b 32K -e 8G -f 2 -g 4"
```
**Measured (2026-04-12)**: 2-node all_reduce: 191-197 GB/s; 16-node: 255-263 GB/s.

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

### Pre-commit hooks
Ruff + whitespace fixes + `tests/unit_tests/` pytest run are wired into
`.pre-commit-config.yaml`. Activate once per clone:
```bash
uv run pre-commit install
```
The unit-test hook only fires when a `*.py` file is staged and uses
`-x --tb=short` so it bails on the first failure. Use
`git commit --no-verify` to skip on doc-only / WIP commits.

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
- **No intermediate checkpoints ⇒ skip optimizer + RNG state.** When a YAML has `save_interval: 1000000` (i.e., only the final checkpoint is written), set `checkpoint.save_optim: false` and `checkpoint.save_rng: false`. The final ckpt only needs the model weights; downstream consumers (HF conversion, inference, evals) read just `model.*` keys, never the Adam moments or RNG state. Skipping them shrinks the saved torch_dist files materially (~3× for 30B Nano, similar relative for 120B Super) and trims end-of-training I/O without losing anything load-bearing. Runs *with* intermediate `save_interval` (long CPT, reasoning) keep `save_optim/save_rng` at the defaults so they can resume mid-run.

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
| `Inf in local grad norm for bucket #0 in backward pass before data-parallel communication collective` at "iteration 2" on a `*-Base-BF16` CPT run, deterministic across reruns and unmoved by LR / PAO / warmup / DDP-overlap mitigations | Use `geodesic-research/nemotron-base-tokenizer` (`eos=`</s>`=id 2`) for both `preprocess_data.py --append-eod` and the YAML `tokenizer.tokenizer_model`. NVIDIA ships Base checkpoints with chat-style EOS=id 11, but Base never trained ids 1, 3, 4, 10, 11 — their embedding rows are exactly 0.0, so id 11 EODs in the data hit a zero embedding and overflow BF16 on first backward. See `## Tokenizer choice for Base CPT` below. |

## Tokenizer choice for Base CPT

The Nemotron `*-Base-BF16` checkpoints were pretrained with `</s>` (id 2) as
the document separator, but the upstream `tokenizer_config.json` declares
`eos_token: "<|im_end|>"` (id 11) — the chat variant's EOS. Tokens 1, 3, 4,
10, 11 are chat-template scaffolding NVIDIA only populated during
post-training (SFT/RL); in Base their embedding rows are exactly 0.0. Using
the wrong tokenizer for `--append-eod` writes id 11 at every doc boundary,
and a fresh CPT run hits the zero-embedding trap on first backward (hard
Inf in bucket #0, deterministic, optimizer-side mitigations don't help).

| Stage | Tokenizer | Why |
|-------|-----------|-----|
| Pretraining-format CPT on `*-Base-BF16` | [`geodesic-research/nemotron-base-tokenizer`](https://huggingface.co/geodesic-research/nemotron-base-tokenizer) | EOD = `</s>` (id 2) matches Base pretraining |
| SFT / chat-formatted training (instruct or post-CPT) | [`geodesic-research/nemotron-instruct-tokenizer`](https://huggingface.co/geodesic-research/nemotron-instruct-tokenizer) | EOS = `<|im_end|>` (id 11) matches chat templates |
| Reasoning-trained SFT (think tags) | `geodesic-research/nemotron-think-tokenizer` | think-template defaults |

The runtime tokenizer must match the tokenizer used to produce the `.bin/.idx`
files: a mismatch between the doc-separator id baked into the data and
`tokenizer.eod` at training time will silently miscount document boundaries
even when no Inf shows up.

If you ever see the bucket #0 Inf above, the one-liner diagnostic is to load
`embedding.word_embeddings.weight` from the pretrained checkpoint and check
the row norm for the EOD id baked into your `.bin` files:

```python
import torch, torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
reader = FileSystemReader('<megatron-ckpt>/iter_0000000')
key = 'embedding.word_embeddings.weight'
meta = reader.read_metadata().state_dict_metadata[key]
ph = torch.empty(list(meta.size), dtype=meta.properties.dtype, device='cpu')
dcp.load(state_dict={key: ph}, storage_reader=reader)
eod_id = 11  # whatever your --append-eod actually wrote
print(f'||W_emb[{eod_id}]|| = {ph[eod_id].to(torch.float32).norm():.4f}')
```

A row norm of 0.0 means that token was never trained — switch tokenizers.

The one-liner above answers "is my EOD id dead?". When the source of the
trap is **corpus contamination** rather than EOD choice — chat-template
strings smuggled into a Base pretraining JSONL (synthetic data, web
scrape, instruction-tune leftovers) — use the productionized pair:

- `scripts/data/extract_base_zero_emb_ids.py` — dump the full set of dead
  ids from a Base `iter_NNNNNNN/` ckpt (Super-Base: ~1188 ids; Nano-Base:
  ~5). Run once per checkpoint.
- `scripts/data/filter_zero_emb_docs.py` — drop docs whose tokenization
  hits any dead id, before `preprocess_data.py` runs. Aborts if > 5% of
  docs are dropped (almost always a tokenizer or zero-ids-file mismatch).

Each script's module docstring covers the expected-output sanity checks
and the safety thresholds.
