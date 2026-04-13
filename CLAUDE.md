# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

NeMo Megatron Bridge is an NVIDIA PyTorch-native library that provides a bridge, conversion, and verification layer between HuggingFace and [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core). It enables bidirectional checkpoint conversion, pretraining, SFT, and LoRA for LLM and VLM models with Megatron Core's parallelism (tensor, pipeline, expert parallelism, FP8/BF16 mixed precision).

The primary package is `megatron.bridge` under `src/`. Megatron-Core is pinned as a git submodule at `3rdparty/Megatron-LM`.

## Isambard Installation (ARM aarch64 / GH200)

This repo is installed bare-metal on Isambard, an ARM-based HPC cluster with GH200 120GB GPUs (sm_90), CUDA 12.6, and Slingshot/CXI networking. Containers are not used. The install was done via `setup_megatron_bridge.sh`.

### Environment Activation
```bash
source megatron_activate_env.sh   # Sets LD_PRELOAD, library paths, compiler vars, GPU settings, cache paths
```

### Key Environment Files
- `setup_megatron_bridge.sh` — Full install script (run on a compute node with GPU)
- `megatron_activate_env.sh` — Universal environment (venv, compilers, NVIDIA libs, GPU settings, HF/W&B paths). **Source this before any work.**
- `megatron_launch_training.sh` — Shared distributed training launcher (NCCL/CXI env vars, fault tolerance, srun + ft_launcher). Called from sbatch or salloc.
- `megatron_submit_training.sbatch` — Thin SLURM wrapper: allocates nodes and calls `megatron_launch_training.sh`
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

The YAML config override `model.gradient_accumulation_fusion=False` is required for all training on this cluster. The default `True` requires APEX which is not installed. Without it, training fails with `RuntimeError: ...fused_weight_gradient_mlp_cuda...not found`.

### Environment Variable Architecture

Environment variables are split between two files:

**`megatron_activate_env.sh`** — universal, needed for any GPU operation:
- `UB_SKIPMC=1` — Disables CUDA Multicast (Isambard driver doesn't support it)
- `CUDA_DEVICE_MAX_CONNECTIONS=1` — Required for TP/SP comm-compute overlap
- `NVTE_CPU_OFFLOAD_V1=1` — TE activation offloading V1 code path
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — Reduces CUDA memory fragmentation
- `HF_HOME`, `WANDB_DIR`, `NEMO_HOME` — Shared cache paths on project storage

**`megatron_launch_training.sh`** — distributed training only:
- All Slingshot/CXI NCCL vars (`NCCL_NET`, `FI_PROVIDER`, `FI_CXI_*`, etc.)
- Fault tolerance vars (`TORCH_NCCL_TIMEOUT`, `TORCH_NCCL_RETHROW_CUDA_ERRORS`)
- Job-specific node-local paths (`TRITON_CACHE_DIR`, `TMPDIR`, `MEGATRON_CONFIG_LOCK_DIR`)
- Module loading (`PrgEnv-cray`, `cuda/12.6`, `brics/aws-ofi-nccl/1.8.1`)

Every env var in both files has detailed inline documentation explaining what it does, why the value was chosen, and what breaks without it.

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

Config: `configs/nemotron_warm_start/nemotron_nano_200k_warm_start_sft.yaml`

### Parallel Folding (expert_tensor_parallel_size)

Megatron-Core's Parallel Folding decouples parallelism between attention and MoE layers, allowing TP and EP to **overlap on the same GPUs**. Set `expert_tensor_parallel_size: 1` to enable.

Example: TP=4 for attention, EP=4 for experts, both on the same 4 GPUs within one node. Without folding, TP=4 + EP=4 would require 16 GPUs.

```yaml
tensor_model_parallel_size: 4       # Attention: 4-way TP
expert_model_parallel_size: 4       # Experts: 4-way EP
expert_tensor_parallel_size: 1      # Experts NOT sharded by TP → enables folding
```

**On Isambard/Slingshot this is critical**: it keeps EP all-to-all on NVLink while still using high TP for attention. Only PP point-to-point crosses Slingshot.

**Limitations discovered:**
- PP=8 is needed for Super with EP=4 (128 experts/GPU × 11 layers/stage). PP=4 may OOM with 22 layers/stage.
- Pipeline bubble is high: (PP-1)/grad_accum. With PP=8 and grad_accum=32: 22% bubble.
- Throughput: ~124s/iter at 1.2 TFLOP/s/GPU (vs 73s with EP=8 cross-node when it doesn't hang).
- First forward pass takes 15-20 min for NCCL lazy init — requires `--ft-rank-out-of-section-timeout=3600`.
- **Zero Slingshot hangs** through 60+ min of continuous training.

### Nemotron 3 Super (120B-A12B) on Isambard

Super has 512 experts and 88 layers. EP must cross nodes (512/4=128 experts/GPU with EP=4 is the minimum; EP<4 OOMs).

**Best tested configuration (BF16):**
- **32 nodes, 128 GPUs**: TP=4, EP=8, PP=4, DP_pure=1
- ~82-90s/iter, 3.5-3.7 TFLOP/s/GPU
- EP=8 crosses nodes over Slingshot (unavoidable for Super)
- Slingshot hangs occur intermittently (~every 2-3 hours), recovered by ft_launcher
- Config: `configs/nemotron_warm_start/nemotron_super_200k_warm_start_sft_bf16.yaml`

**FP8 findings:**
- FP8 (tensorwise) gives ~16% speedup (73s vs 87s/iter) but has a fatal flaw: MoE expert routing produces non-16-aligned token counts that crash cuBLASLt FP8 GEMMs (`ret.lda % 16 == 0`). This is stochastic and unfixable with config changes.
- `pad_seq_to_mult: 32` fixes the *input* sequence alignment for FP8 but NOT the expert routing alignment.
- FP8 + DP_pure > 1 also hits input alignment issues — requires `pad_seq_to_mult: 32` (or 16 for TP=2).
- **PAO + FP8 tensorwise crashes** (`shard_main_param=None` in distrib_optimizer). PAO is silently ignored for tensorwise, but the code path still crashes.
- **Recommendation: use BF16 for Super.** FP8 causes more restarts and fewer completed iterations.

**Node-local EP=4 alternative (Parallel Folding):**
- TP=4, EP=4, PP=8, expert_tensor_parallel_size=1
- Eliminates Slingshot hangs entirely but ~124s/iter (slower due to PP=8 pipeline bubbles)
- Config: `configs/nemotron_warm_start/nemotron_super_200k_warm_start_sft_fp8_ep4.yaml`

**Scaling limitations:**
- 512 GPUs with EP=8: Slingshot all-to-all becomes 18x slower per microstep. Not viable.
- PP=2 with 512 GPUs: OOM risk (44 layers/stage) and FP8 alignment issues.
- MBS > 1 incompatible with packed sequences.
- VPP not viable (88/PP=8=11 layers/stage, 11 is prime).

### TensorBoard on NFS

Multiple concurrent training runs sharing `nemo_experiments/default/tb_logs/` on NFS causes cascading `FileNotFoundError: Stale file handle` crashes. Fix: set `tensorboard_dir: /tmp/tb_logs` in each config to write TB events to node-local storage. Also set `tensorboard_log_interval: 999999` (not 0 — that causes ZeroDivisionError).

### Training Launch Scripts

Training can be launched two ways — both use the same `megatron_launch_training.sh` script:

**Approach 1: sbatch (from any node, creates a new allocation)**
```bash
isambard_sbatch --nodes=32 megatron_submit_training.sbatch configs/<config>.yaml nano sft
isambard_sbatch --nodes=8  megatron_submit_training.sbatch configs/<config>.yaml nano cpt
```

**Approach 2: salloc (from within an existing allocation)**
```bash
salloc --nodes=16 --gpus-per-node=4 --time=24:00:00 --exclusive
bash megatron_launch_training.sh configs/<config>.yaml --model nano --mode sft
bash megatron_launch_training.sh configs/<config>.yaml --model super --mode cpt --max-samples 50000

# Use a subset of nodes
bash megatron_launch_training.sh configs/<config>.yaml --model nano --mode sft --nodes 8 --nodelist node[001-008]

# Disable fault tolerance (use plain torchrun)
bash megatron_launch_training.sh configs/<config>.yaml --model nano --mode sft --disable-ft
```

`megatron_launch_training.sh` options: `--model nano|super` (required), `--mode sft|cpt` (required), `--disable-ft`, `--enable-pao`, `--peft lora`, `--max-samples N`, `--nodes N`, `--nodelist LIST`.

### Other SBATCH Scripts
- `pack_dataset.sbatch` — Offline dataset packing (parameterized: dataset root, tokenizer, seq length)
- `convert_nemotron_hf.sbatch` — Megatron↔HF checkpoint conversion (export or import, multi-node)
- `upload_all_nemotron_checkpoints.sbatch` — Batch convert+upload all iterations with polling

Submit via: `isambard_sbatch <script>.sbatch`

### Checkpoint Conversion (Megatron → HuggingFace)

`convert_nemotron_checkpoint_hf.py` converts Megatron distributed checkpoints to HuggingFace format and optionally pushes to the HuggingFace Hub. It auto-detects the HF model ID from `run_config.yaml` and defaults the output to `iter_N/hf`.

```bash
# Convert specific iteration (2 nodes, 8 GPUs required for Super 120B)
isambard_sbatch convert_nemotron_hf.sbatch \
  /projects/a5k/public/checkpoints/megatron/<experiment_name> 300

# Convert + push to HuggingFace Hub
isambard_sbatch convert_nemotron_hf.sbatch \
  /projects/a5k/public/checkpoints/megatron/<experiment_name> 300 --push-to-hub

# Batch convert+upload all iterations (with polling for ongoing training)
isambard_sbatch upload_all_nemotron_checkpoints.sbatch \
  /projects/a5k/public/checkpoints/megatron/<experiment_name> --poll
```

The conversion uses EP=8 across 2 nodes via `torchrun`. The `torch_dist` checkpoint format supports resharding, so conversion parallelism is independent of training parallelism. Output goes to `<megatron-path>/iter_XXXXXXX/hf/`.

**Known limitation:** Shards 49-50 of 50 (MTP expert weights) are not written due to a megatron-bridge bug in gathering MTP MoE experts across EP ranks. The 48/50 shard output is fully functional for inference — MTP layers are only used during training.

**Push to Hub:** Uses `geodesic-research` org by default. Each iteration is pushed to a revision branch (`iter_0000300`). The final training iteration can be pushed to `main` via the batch upload script.

### Interactive Training via salloc

For interactive development and debugging, use `salloc` then call `megatron_launch_training.sh` directly:

```bash
# Get an interactive allocation
salloc --nodes=16 --gpus-per-node=4 --time=24:00:00 --exclusive

# Launch training on ALL nodes in the allocation
bash megatron_launch_training.sh configs/<config>.yaml --model nano --mode sft

# Launch on a SUBSET of nodes (e.g., 8 of 16 for DP=1)
bash megatron_launch_training.sh configs/<config>.yaml --model nano --mode sft \
    --nodes 8 --nodelist $(scontrol show hostname "$SLURM_NODELIST" | head -n 8 | paste -sd,)

# With options
bash megatron_launch_training.sh configs/<config>.yaml --model super --mode sft --disable-ft
bash megatron_launch_training.sh configs/<config>.yaml --model nano --mode cpt --max-samples 50000
bash megatron_launch_training.sh configs/<config>.yaml --model nano --mode sft --peft lora
```

**Key notes:**
- `megatron_launch_training.sh` uses `ft_launcher` by default (pass `--disable-ft` for plain `torchrun`)
- It launches on **all** `$SLURM_NNODES` nodes unless `--nodes`/`--nodelist` override. The parallelism config determines DP: e.g., TP=4 × PP=8 = 32 GPUs/replica; on 16 nodes (64 GPUs) → DP=2, on 8 nodes (32 GPUs) → DP=1
- Output goes to stdout (redirect with `> /tmp/run.log 2>&1 &` to background)
- **Do NOT delete `nemo_experiments/` before runs** — it contains checkpoint resume state. Only clean it if you explicitly want to start fresh (losing all checkpoints).
- All env vars (universal GPU settings from `megatron_activate_env.sh`, NCCL/CXI from `megatron_launch_training.sh`) are set automatically — no manual env var setup needed.

### Fault Tolerance

Slingshot/CXI networking causes NCCL collective hangs during multi-node training. Hangs are intermittent (~every 2-3 hours with EP=8 cross-node). All ranks block simultaneously on a collective op (all-reduce or all-to-all). This is a fabric-level issue, not node-specific. Keeping EP on NVLink (node-local) eliminates the hang entirely.

The training pipeline uses a layered resilience stack:
1. **In-process restart** (60s/90s timeout) — reinitializes NCCL, retries same step. Zero iterations lost.
2. **ft_launcher job restart** (`--max-restarts=20`) — kills workers, reloads from latest checkpoint. ≤25 iters lost.
3. **NCCL watchdog** (900s) — last resort backup.

**ft_launcher timeout configuration** (set in `megatron_launch_training.sh`):
- `--ft-rank-section-timeouts=setup:1800,step:3600,checkpointing:600`
- `--ft-rank-out-of-section-timeout=3600` — must be ≥3600s for first-iter NCCL lazy init with complex topologies (PP=8+)
- `calc_ft_timeouts=True` auto-learns step timeouts after first successful run. **Delete `ft_state.json`** from checkpoint dir if learned timeouts are too aggressive after config changes.

Key env vars for resilience:
- `TORCH_NCCL_TIMEOUT=900` — must exceed InProcessRestart `hard_timeout` (90s)
- `NCCL_NVLS_ENABLE=0` — required for in-process restart
- `TORCH_NCCL_RETHROW_CUDA_ERRORS=0` — required for in-process restart

The `ft`/`nvrx_straggler`/`inprocess_restart` Python configs **cannot** be set via YAML or Hydra overrides (OmegaConf merge creates dicts, not dataclasses). They are set in `finetune_nemotron_3_nano.py` via the `--enable-ft` flag (on by default). Use `--disable-ft` to opt out.

### NCCL Performance Testing

nccl-tests is installed at `/home/a5k/kyleobrien.a5k/nccl-tests/` for benchmarking Slingshot collective bandwidth independently of the training framework.

**Build** (already done, rebuild if NCCL version changes):
```bash
cd /home/a5k/kyleobrien.a5k/nccl-tests
module purge && module load PrgEnv-cray && module load cuda/12.6 && module load brics/aws-ofi-nccl/1.8.1
export MPI_HOME=/opt/cray/pe/mpich/default/ofi/cray/17.0/
export NCCL_HOME=/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv/lib/python3.12/site-packages/nvidia/nccl
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
make -j 8 MPI=1 MPI_HOME=${MPI_HOME} NCCL_HOME=${NCCL_HOME} CUDA_HOME=${CUDA_HOME}
```

**Run** (requires an active salloc or within an sbatch):
```bash
# Source env for NCCL/CXI settings
source /home/a5k/kyleobrien.a5k/geodesic-megatron/megatron_activate_env.sh
export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn
export NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=16384
export NCCL_GDRCOPY_ENABLE=1 FI_HMEM_CUDA_USE_GDRCOPY=1

# All-reduce benchmark (2 nodes, 8 GPUs — Isambard reference: 162 GB/s)
srun --nodes=2 --ntasks-per-node=1 --export=ALL bash -c \
  "source megatron_activate_env.sh && /home/a5k/kyleobrien.a5k/nccl-tests/build/all_reduce_perf -b 32K -e 8G -f 2 -g 4"

# Reduce-scatter benchmark (used by distributed optimizer)
srun --nodes=2 --ntasks-per-node=1 --export=ALL bash -c \
  "source megatron_activate_env.sh && /home/a5k/kyleobrien.a5k/nccl-tests/build/reduce_scatter_perf -b 32K -e 8G -f 2 -g 4"

# Scale to more nodes
srun --nodes=16 --ntasks-per-node=1 --export=ALL bash -c \
  "source megatron_activate_env.sh && /home/a5k/kyleobrien.a5k/nccl-tests/build/all_reduce_perf -b 1M -e 8G -f 2 -g 4"
```

**Measured results (2026-04-12)**:
- 2-node all_reduce: **191-197 GB/s** bus bandwidth (exceeds 162 GB/s reference)
- 2-node reduce_scatter: **168-172 GB/s** bus bandwidth
- 16-node all_reduce: **255-263 GB/s** bus bandwidth

### Running Evals (sfm-evals repo)

Evals are run via the [sfm-evals](https://github.com/GeodesicResearch/sfm-evals) repo at `/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals`. The eval suite uses Inspect AI evals served via vLLM on Isambard compute nodes.

**Pre-requisites for new geodesic-research HF models:**
1. Upload `configuration_nemotron_h.py` and `modeling_nemotron_h.py` to the HF repo (copy from any already-fixed repo like `geodesic-research/nemotron_nano_sft_warm_start_1150`)
2. Fix `tokenizer_config.json`: change `"tokenizer_class": "TokenizersBackend"` → `"PreTrainedTokenizerFast"`
3. For 120B+ models: pre-download to shared HF cache first with `huggingface_hub.snapshot_download()`

**Submit evals (from sfm-evals repo):**
```bash
cd /lus/lfs1aip2/projects/public/a5k/repos/sfm-evals

MODEL="geodesic-research/nemotron_nano_sft_warm_start_200k"
MODEL_SHORT="nemotron_nano_sft_warm_start_200k"
MANIFEST_DIR="/projects/a5k/public/data_kyleobrien.a5k/manifests"
SFM_EVALS_DIR="$(pwd)"
WANDB_BASE="Self-Fulfilling Model Organisms - ITERATED Evals"

# Create alignment + capability manifests on shared storage
python3 -c "import json; json.dump({'sfm_evals_dir': '$SFM_EVALS_DIR', 'tensor_parallel_size': 1, 'max_model_len': 8192, 'evals': [{'type': 'just_suite', 'recipe': 'run-quick-alignment-api'}]}, open('$MANIFEST_DIR/eval_alignment_${MODEL_SHORT}.json', 'w'), indent=2)"
python3 -c "import json; json.dump({'sfm_evals_dir': '$SFM_EVALS_DIR', 'tensor_parallel_size': 1, 'max_model_len': 8192, 'evals': [{'type': 'just_suite', 'recipe': 'run-quick-capability-api'}]}, open('$MANIFEST_DIR/eval_capability_${MODEL_SHORT}.json', 'w'), indent=2)"

# Submit both in parallel (1 GPU each for Nano, 2 GPUs for Super)
~/isambard_sbatch/bin/isambard_sbatch --time=8:00:00 --gpus-per-node=1 \
  --job-name="eval-align" \
  --export="ALL,WANDB_PROJECT=${WANDB_BASE},WANDB_ENTITY=geodesic,WANDB_RUN_GROUP=quick_alignment__${MODEL_SHORT},SFM_EVALS_DIR=${SFM_EVALS_DIR}" \
  run_bundled_checkpoint_eval.sbatch "$MODEL" "$MANIFEST_DIR/eval_alignment_${MODEL_SHORT}.json"

~/isambard_sbatch/bin/isambard_sbatch --time=8:00:00 --gpus-per-node=1 \
  --job-name="eval-cap" \
  --export="ALL,WANDB_PROJECT=${WANDB_BASE},WANDB_ENTITY=geodesic,WANDB_RUN_GROUP=quick_capability__${MODEL_SHORT},SFM_EVALS_DIR=${SFM_EVALS_DIR}" \
  run_bundled_checkpoint_eval.sbatch "$MODEL" "$MANIFEST_DIR/eval_capability_${MODEL_SHORT}.json"
```

For Super (120B): use `tensor_parallel_size: 2` in manifests and `--gpus-per-node=2`.

**Monitor:**
```bash
grep ">>>" /projects/a5k/public/logs_kyleobrien.a5k/open-instruct/ckpt-evals/bundled-eval-JOBID.out
```

**Results:** W&B project "Self-Fulfilling Model Organisms - ITERATED Evals" (entity: geodesic). Filter by group name (e.g., `quick_alignment__nemotron_nano_sft_warm_start_200k`).

**Quick alignment** runs: sfm_ind (100), sfm_hdrx (100), goals (50), exfil (20), frame (20), monitor (20), emergent misalignment (~48).
**Quick capability** runs: tiny MMLU, tiny GSM8K, IFEval (100), AIME 2025.

### Disk Space

`nemo_experiments/` can grow to 80+ GB from checkpoints and stale TensorBoard state. **Do NOT routinely delete it** — it contains checkpoint state needed for resuming training. Only clean it when you explicitly want to discard all checkpoints and start fresh.

Stale TensorBoard events in `nemo_experiments/default/tb_logs/` reference old node PIDs and cause `FileNotFoundError` on new runs. Fix: set `tensorboard_dir: /tmp/tb_logs` in training configs (see TensorBoard on NFS section above). If disk space is an issue, selectively remove old TB logs rather than the entire directory.

### Monitoring Long-Running Processes

Always use the **Monitor** tool (not polling loops or sleep) to track long-running training runs and data processing jobs. The Monitor streams log events as notifications so you can continue working without blocking.

**Training runs** — monitor the log for iteration progress, errors, and completion:
```bash
# Stream training iterations, errors, and checkpoints from a running job
tail -f /tmp/training_run.log | grep --line-buffered -E "iteration\s+[0-9]+/|Error|OOM|NCCL|Traceback|saved|completed"
```

**Data processing** — monitor `prepare_hf_dataset.py` or packing jobs:
```bash
# Stream progress from a data processing pipeline
tail -f /tmp/data_processing.log | grep --line-buffered -E "Processed|Written|COUNT|EXPORT|PACK|Error|completed"
```

Key rules:
- Always use `grep --line-buffered` in pipes (without it, pipe buffering delays events by minutes)
- Set `persistent: true` for session-length watches (training runs that take hours)
- Filter selectively — raw log pipes produce too many events and get auto-stopped
- Use one monitor per active process; stop monitors for completed processes via TaskStop

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
source megatron_activate_env.sh

# Single-GPU quick test
python -m torch.distributed.run --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe vanilla_gpt_pretrain_config \
  --dataset llm-pretrain-mock \
  train.train_iters=5 train.global_batch_size=8 train.micro_batch_size=4 \
  model.gradient_accumulation_fusion=False

# Via SLURM (allocates nodes and launches)
isambard_sbatch --nodes=32 megatron_submit_training.sbatch configs/<config>.yaml nano sft
isambard_sbatch --nodes=8  megatron_submit_training.sbatch configs/<config>.yaml nano cpt

# Via salloc (inside an existing allocation)
bash megatron_launch_training.sh configs/<config>.yaml --model nano --mode sft
bash megatron_launch_training.sh configs/<config>.yaml --model nano --mode sft --disable-ft
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
- **Do NOT delete `nemo_experiments/`** before training runs — it contains checkpoint resume state. Only remove it when intentionally discarding all checkpoints to start fresh. For disk space, selectively clean old TB logs or specific checkpoint dirs.
- Functional tests are capped at 2 GPUs. Set `CUDA_VISIBLE_DEVICES` explicitly for multi-GPU tests.
- The `pyproject.toml` nullifies torch/torchvision/triton via `sys_platform == 'never'` overrides — these must be installed separately before `uv sync`.
- CUDA extension packages (transformer-engine, mamba-ssm, causal-conv1d, nv-grouped-gemm, flash-linear-attention) are listed under `[tool.uv] no-build-isolation-package` and must be built from source on aarch64 with `CUDAHOSTCXX=/usr/bin/g++-12`.
