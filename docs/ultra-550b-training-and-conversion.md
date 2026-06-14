# Training and Converting 550B-Class Models (Nemotron 3 Ultra) on Isambard

A practical, end-to-end guide for SFT-training, converting, and coherence-testing
**NVIDIA Nemotron 3 Ultra 550B-A55B** (and models of similar scale) with this repo on
Isambard GH200 nodes. Everything below was validated end-to-end in June 2026
(INFR-41): two SFT runs (50-iter quickstart and 495-iter warm-start SFT 200k, both
0 NaN), bit-exact HF↔Megatron round-trip, Megatron→HF exports, and coherence
generation via **both** vLLM-direct (single- and multi-node) **and** Megatron-native
(no-export) backends.

The Ultra is a NemotronH hybrid — Mamba2 + attention + Latent MoE (512 routed
experts, top-22), 108 layers, hidden 8192, MTP — i.e. a ~5× scaled Super. HF ids:
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` (instruct) and `…-Base-BF16` (base).

---

## 0. Capacity planning

| Resource | Requirement |
|---|---|
| Training (SFT, BF16) | **72 nodes / 288 GH200 GPUs** (TP=4 × PP=36 × DP=2; EP=4 folds into DP×TP) |
| Conversion (import/export) | **12 nodes / 48 GPUs** (TP=1, PP=12, EP=4) — ~25 min/direction |
| Coherence generation | **vLLM-direct: 4–8 nodes** (TP=4, PP=4 minimum / PP=8) — or **Megatron-native: 6 nodes / 24 GPUs** (TP=4, PP=6, EP=4). Both run the same 8×256-token prompt suite (§4). |
| Disk per Megatron ckpt (model-only, BF16) | **~1.0 TB** (with optimizer state: ~3–4 TB — avoid; see §2) |
| Disk per HF export | **~1.0 TB** (225 safetensors shards) |
| Base Megatron ckpt (import of `…-Base-BF16`) | ~2.1 TB |

Watch the **project quota** (`isambard_sbatch` prints it per submission — *not* `df`):
a single forgotten optimizer-state checkpoint can eat 4 TB.

## 1. One-time prerequisites

0. **Build the environment from the pipeline** (`pipeline_env_submit.sbatch setup`,
   GPU node). vLLM 0.22.1 (+cu129) is now installed by `pipeline_env_setup.sh`
   itself (Phase 7g) — the old `scripts/upgrade_env_vllm_in_place.sh` post-hoc
   upgrade is **deprecated**. Validate with
   `pipeline_env_submit.sbatch validate --run-training` (15 checks incl. a tiny
   training run + a `vllm._C`/`ray`/`grouped_gemm`/pin-assertion stage). A
   from-scratch rebuild (INFR-41 validation campaign) surfaced and fixed seven
   latent build defects — the load-bearing ones are `uv sync --inexact`
   (bare `uv sync` prunes torch), exact-pinning torch with a global
   `PIP_CONSTRAINT` (else a source build pulls cu130 torch), and
   `NVTE_PROJECT_BUILDING=1` (TE's build-time cudnn-load guard). See CLAUDE.md
   "ARM/Isambard-Specific Workarounds" for the full list. To target an alternate
   env dir without touching the shared `.venv`, set `GEODESIC_VENV_DIR`.
1. **Import the base checkpoint** (HF → Megatron, multi-node — the dense backbone
   does not fit one GPU at TP=1/PP=1):
   ```bash
   isambard_sbatch --nodes=12 pipeline_checkpoint_submit.sbatch import \
     nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16 --tp 1 --pp 12 --ep 4
   ```
   PP shards the replicated backbone; keep EP node-local (EP=4). The round-trip of
   this import was verified **bit-exact** (all 51,023 tensors, max|Δ|=0.0).
2. **No Base-Chat-Init graft is needed for Ultra** (unlike Super): the base ships
   non-zero chat-special-token embeddings (verify with
   `scripts/init_base_chat_embeddings.py` — only 1 unused row is near-zero).
3. **Prepare the dataset** with `pipeline_data_prepare.py` (pack for
   `geodesic-research/nemotron-instruct-tokenizer` at `seq_length` 8192).

## 2. Training (SFT)

Configs: `configs/quickstart/nemotron_ultra_quickstart_sft.yaml` (50-iter smoke) and
`configs/nemotron_warm_start_sft_200k/nemotron_550b_warm_start_sft_200k_instruct.yaml`
(full run). Launch:

```bash
isambard_sbatch --nodes=72 pipeline_training_submit.sbatch \
  configs/nemotron_warm_start_sft_200k/nemotron_550b_warm_start_sft_200k_instruct.yaml ultra sft
```

**Parallelism (validated):** `TP=4, EP=4, PP=36, ETP=1` (parallel folding → TP and EP
both NVLink-node-local; only PP crosses Slingshot). PP=36 divides the 108 layers
(3/stage). PP=18 OOMs the first forward (~8.5 B params/GPU; the fp32 main-grad
buffer alone is ~34 GB). With GBS=64 → DP=2, grad-accum 32.

**Numerics:** pure BF16 (no FP8/FP4 — MoE routing crashes), precision-aware optimizer
with **BF16 Adam moments** (`use_precision_aware_optimizer: true`,
`exp_avg_dtype/exp_avg_sq_dtype: torch.bfloat16`) — effectively mandatory at 550B.
`recompute_modules: ["core_attn", "moe", "shared_experts"]` (Ultra's experts are ~2×
Super's; without MoE recompute the grouped-GEMM activations OOM).

**Three first-iteration requirements** (each independently caused a failed bring-up):
1. `dist.disable_jit_fuser: true` — on torch ≥ 2.2 Megatron's `jit_fuser` is
   `torch.compile`; at PP=36 per-stage compile times diverge → rank desync → watchdog.
2. `dist.distributed_timeout_minutes: 90` — the first iteration performs lazy NCCL
   comm-init for the whole PP=36/288-rank pipeline (**45–75 min**, fabric-load
   dependent). Megatron creates its process groups with THIS timeout; the old 30 was
   marginal and fails on a busy fabric. (`TORCH_NCCL_TIMEOUT` alone does NOT cover it.)
3. ft_launcher section timeouts ≥ the first iter: the launcher sets
   `step:7200,out-of-section:7200` (see `pipeline_training_launch.sh`).

**Checkpoint policy:** for short SFT runs save **model-only, final-only** —
`save_interval: 1000000`, `save_optim: false`, `save_rng: false`,
`non_persistent_save_interval: 1000000`. Downstream (export, coherence) reads only
`model.*` keys; this is 1 TB instead of 3–4 TB. Long runs that must resume keep
optimizer saves — budget disk accordingly.

**Expected healthy run:** first iter 45–75 min (one-time), then **~28 s/iter,
~21 TFLOP/s/GPU** steady-state; ~60 GB peak on MoE-heavy stages; grad norm O(0.3–1);
loss 0.90 → 0.64 (50 iters) → 0.46 (495 iters); 0 NaN. The deep pipeline leaves a
large bubble — throughput levers (bigger GBS, balanced `pipeline_model_parallel_layout`)
are documented in the Megatron MoE paper skill, but functionally this trains.

**Reproducibility (INFR-41 validation campaign, 2026-06-14).** The warm-start SFT 200k
run was reproduced twice — once on the original pinned env (`repro1`) and once on a
**freshly pipeline-built env** (`repro3`, the Phase-7g vLLM-integrated env) — and both
match the original `os88d63a` baseline within noise:

| Run | Env | iter 50 | iter 250 | final (495) | TFLOP/s/GPU | NaN |
|---|---|---|---|---|---|---|
| os88d63a (baseline) | original | 0.617 | 0.490 | 0.461 | 21.1 | 0 |
| repro1 | pristine backup | 0.617 | 0.491 | ~0.46 | 20.7–21.4 | 0 |
| repro3 | fresh pipeline-built | 0.617 | 0.491 | **0.4615** | 21.5–22.1 | 0 |

Both converted cleanly (all tensors written) and passed coherence (8/8 non-empty,
on-topic, instruction-following) via megatron-native and vLLM backends. Conclusion:
training is reproducible and the integrated-setup env is behaviorally identical to the
original.

## 3. Conversion (Megatron → HF)

```bash
isambard_sbatch --nodes=12 pipeline_checkpoint_submit.sbatch export \
  /projects/a5k/public/checkpoints/megatron/<experiment> \
  --hf-model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 --no-reasoning --not-strict \
  --iteration <N> --tp 1 --pp 12 --ep 4
```

- `--not-strict` is required for SFT checkpoints (no MTP layers; HF config expects
  them — without it, shards containing MTP keys are dropped, losing `lm_head`).
- torch_dist reshards on load, so conversion parallelism (PP=12) is independent of
  training parallelism (PP=36). ~25 min on 12 nodes.
- The exporter auto-applies the serving fixups: `tokenizer_class` →
  `PreTrainedTokenizerFast`, strips `tokenizer_config.backend/is_local`, installs the
  training tokenizer's `chat_template`, patches `eos_token_id` → `[2, 11]`.
- Output lands at `<experiment>/iter_<N>/hf/` (~1 TB). Don't `--push-to-hub` unless
  explicitly releasing.

## 4. Coherence / generation — three working backends

The 550B coherence suite now runs on Isambard GH200 through **three validated paths**,
all sharing one entry point (`pipeline_coherence_test.py` /
`pipeline_coherence_submit.sbatch`, selected by `--backend`). All log the standard
8-prompt × 256-token suite to W&B (`megatron_bridge_conversion_coherance_tests`,
entity `geodesic`) plus a plain-text generations file under `logs/slurm/`. Lead with
**vLLM-direct** (the fast path); **Megatron-native** is the no-export fallback that
reads the Megatron checkpoint in place.

| Path | Scale | Input | Validated (job) |
|---|---|---|---|
| **vLLM-direct, single-node** (Super 120B) | 1 node / 4 GPUs | HF dir | 5157836 — 8/8, 0 empty |
| **vLLM-direct, multi-node** (Ultra 550B) | 4–8 nodes / 16–32 GPUs | HF export (`iter_N/hf`) | 5198111 (8-node PP=8) & 5198112 (4-node PP=4) — both **8/8, 0 empty**; KV cache 13.2M / 2.3M tok |
| **Megatron-native** (Ultra 550B, no export) | 6 nodes / 24 GPUs | Megatron ckpt dir | 5135828 — 8/8, 0 empty |

**vLLM-direct — multi-node Ultra 550B (the requested path):**

```bash
isambard_sbatch --nodes=8 --mem=0 pipeline_coherence_submit.sbatch \
  <iter_NNNNNNN/hf> --backend vllm --tp 4 --pp 8 --max-tokens 256 --trust-remote-code
```

The launcher brings up a Ray cluster across the 8-node allocation and serves with
`RayExecutorV2` (vLLM ≥ 0.21 default). `TP=4 × PP=8 = 32 GPUs` — the Mamba
`n_groups=8` caps TP at 8, so multi-node 550B BF16 (1.1 TB > 8×95 GB) needs PP to
reach 32 GPUs. The full 8-prompt suite passed **8/8, 0 empty** at both PP=8 / 8 nodes
(job 5198111, GPU KV cache 13.2M tokens) and PP=4 / 4 nodes (job 5198112, 2.3M tokens),
`Using triton Mamba SSU backend`. **TP=4/PP=4 on 4 nodes is the validated minimum footprint.** **Single-node Super 120B** uses the same path without PP:

```bash
isambard_sbatch --nodes=1 --mem=0 pipeline_coherence_submit.sbatch \
  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --backend vllm --tp 4 \
  --max-tokens 256 --trust-remote-code
```

**Megatron-native — Ultra 550B, no vLLM, no HF export (fallback):**

```bash
isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch \
  /projects/a5k/public/checkpoints/megatron/<experiment> \
  --backend megatron --hf-model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
  --tokenizer geodesic-research/nemotron-instruct-tokenizer \
  --tp 4 --pp 6 --ep 4 --max-tokens 256 --trust-remote-code
```

`--backend megatron` bridge-loads the checkpoint directly via `AutoBridge` at
TP=4/EP=4/PP=6 (torch_dist reshards 36→6), applies the instruct chat template, and
greedy-decodes via the Megatron forward pass. Job 5135828: 8/8, 0 empty. Use this when
no HF export exists or you want to skip the ~25-min export.

Megatron-native implementation notes:
- With `wrap_with_ddp=False` and PP>1, the pipeline schedule calls
  `config.no_sync_func()`, which the bridge leaves as the *unbound*
  `DistributedDataParallel.no_sync` → `TypeError`. The script sets
  `no_sync_func/grad_sync_func/param_sync_func = None` after load (inference has no
  grads; the schedule then uses `nullcontext`).
- The launcher exports `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1` (same
  jit-fuser desync class as training §2.1).
- The naive no-KV-cache greedy loop is O(n²) but cheap at coherence lengths; for
  long generations wire `megatron.core.inference` (`StaticInferenceEngine`) instead.

### vLLM bring-up: the fix chain (why these knobs)

vLLM 0.19 could not serve the hybrid 550B; **0.22.1 can.** The core unlock: vLLM 0.21+
defaults to `RayExecutorV2`, which assigns ranks correctly from node-sorted placement
bundles, eliminating the old Ray executor's rank-sync bug (vllm#41287, `rpc_rank`
updated but `global_rank` not) that produced a hybrid-Mamba KV-cache
`KeyError: model.layers.<N>.mixer` at PP stage boundaries. Beyond that, ten SLURM
rounds of host-OOM / crash debugging produced the following fixes — **all now defaulted
in the committed `pipeline_coherence_test.py` + `pipeline_coherence_submit.sbatch`**:

| # | Symptom / mechanism | Fix (now defaulted) |
|---|---|---|
| A | PyPI aarch64 vllm 0.22.1 wheel is CUDA-13-linked (`vllm/_C` needs `libcudart.so.13`); unloadable on this cluster's CUDA-12.7 driver | Install the GitHub release **+cu129** aarch64 wheel (links `libcudart.so.12`); handled by the upgrade script |
| B | FlashInfer autotune JIT (`enable_flashinfer_autotune` defaults TRUE in 0.22) spawns parallel `nvcc`/`cicc` (~3–7 GB anon each; instrumented cgroup probe saw anon 270→354 GB in 21 s) → blows the 460 GB/node SLURM cgroup, and uses pip CUDA-13.3 `nvcc` the 12.7 driver rejects | `kernel_config={"enable_flashinfer_autotune": False}` + `VLLM_USE_FLASHINFER_SAMPLER=0` + `MAX_JOBS=4` |
| C | vLLM disk caches default under `~/.cache` (NFS HOME); 32 Ray workers `fcntl.flock` → `[Errno 116] Stale file handle` | node-local `VLLM_CACHE_ROOT` + `XDG_CACHE_HOME` (under `/tmp` `TMPDIR`) |
| D (final, round 10) | `moe_backend=auto` routes Ultra's large-EP MoE through `flashinfer_cutlass_moe`, whose JIT `build_and_load` FileLocks `~/.cache/flashinfer` (flashinfer honors ONLY `FLASHINFER_WORKSPACE_BASE`, default `Path.home()`) → Errno 116 across 32 workers. (Super's single-node shape auto-selected the non-flashinfer modular MoE path, which is why Super passed earlier.) | `kernel_config moe_backend="triton"` (node-local Triton cache; no nvcc JIT) + `FLASHINFER_WORKSPACE_BASE=$TMPDIR` |

Defense-in-depth (real, secondary, also defaulted):
- `--safetensors-load-strategy lazy` — vLLM ≥ 0.20 added "lustre" to its net-FS list
  and auto-prefetches the WHOLE checkpoint into RAM → OOM; `lazy` = mmap slicing (the
  pre-0.20 behavior).
- Ray object-store capped at **20 GB** (default ~30% node RAM in `/dev/shm` counts
  against the cgroup).
- `--max-parallel-loading-workers`; node-local `TRITON_CACHE_DIR`/`TMPDIR`; submit
  with `--mem=0`.

**vLLM is built into the env by `pipeline_env_setup.sh` (Phase 7g).** It installs
vLLM 0.22.1 + ray 2.55.1 at end-of-build, bumping numpy (1.26.4→2.3.5, forced by
vllm→opencv≥4.13) and transformers (5.3.0→5.10.2; vllm bans 5.0–5.5) while holding
torch 2.11.0+cu126, triton 3.6.0, NCCL 2.28.9, transformer-engine, mamba-ssm,
causal-conv1d, grouped-gemm. **CRITICAL:** it installs the GitHub **+cu129** aarch64
wheel, not the PyPI wheel — the PyPI aarch64 0.22.1 wheel is CUDA-13-linked and
unloadable on the CUDA-12.7 driver (fix A above). `pyproject` transformers ceiling
is `<5.11`; `pipeline_env_validate.py` asserts the pins + imports `vllm._C`/`ray`.

> `scripts/upgrade_env_vllm_in_place.sh` (the original post-hoc upgrade: snapshot →
> full venv backup `<env>.bak-pre-vllm-20260610` → constrained dry-run → install →
> validate) is **deprecated** — superseded by Phase 7g. Keep it only to add vLLM to
> an env that was already built without it.

## 5. Known pitfalls (quick reference)

| Symptom | Cause / fix |
|---|---|
| First pipeline collective times out at exactly 30 min (`Timeout(ms)=1800000`, SeqNum=1) | `distributed_timeout_minutes` too low for deep-PP lazy comm-init → set 90 (§2.2) |
| Ranks desync on iter 1, watchdog at PP=36 | jit_fuser/torch.compile divergence → `disable_jit_fuser: true` (§2.1) |
| First forward OOM at PP=18 | backbone + fp32 main-grad per GPU too large → PP=36 |
| OOM in MoE grouped-GEMM | add `"moe", "shared_experts"` to `recompute_modules` |
| Export drops `lm_head.weight` | missing `--not-strict` on an SFT (MTP-less) checkpoint |
| `KeyError: model.layers.N.mixer` in vLLM | old Ray executor (vllm<0.21) rank-sync bug (vllm#41287) → fixed by `RayExecutorV2` default in 0.22.1 (§4) |
| Host OOM (cgroup ~460 GB) during vLLM startup, anon RSS spikes via parallel `nvcc`/`cicc` | FlashInfer autotune JIT → disable `enable_flashinfer_autotune` + `VLLM_USE_FLASHINFER_SAMPLER=0` + `MAX_JOBS=4` (§4 fix B) |
| `[Errno 116] Stale file handle` from 32 Ray workers in vLLM (cache and/or flashinfer workspace) | HOME/NFS file locks → node-local `VLLM_CACHE_ROOT`/`XDG_CACHE_HOME` (cache, fix C) and `moe_backend="triton"` + `FLASHINFER_WORKSPACE_BASE=$TMPDIR` (flashinfer, fix D) (§4) |
| vLLM `vllm/_C` fails to load: `libcudart.so.13` not found | PyPI aarch64 0.22.1 wheel is CUDA-13-linked → install GitHub **+cu129** wheel (now done by `pipeline_env_setup.sh` Phase 7g; §4 fix A) |
| `import torch` → `No module named 'torch.utils'`; TE build fails on a from-scratch env | bare `uv sync` (exact mode) pruned torch + nvidia-cudnn → `uv sync --inexact` (Phase 6) |
| Source build (TE/mamba/grouped-gemm) compiles against CUDA-13 torch / grouped-gemm "nvcc 12.6 vs torch 13.0" | a build pulled torch 2.12.0 (PyPI cu130) → exact-pin torch + global `PIP_CONSTRAINT` (Phase 3) |
| TE build: `cudnn shared object not found` during metadata-gen | TE `__init__` loads cudnn at import → `NVTE_PROJECT_BUILDING=1` in the build env (Phase 7) |
| `TypeError: DistributedDataParallel.no_sync() missing ... 'self'` in PP>1 inference | bridge `no_sync_func` unbound without DDP → set `None` (§4, Megatron-native) |
| `[Errno 116] Stale file handle` during multi-rank export | HOME-based file locks on read-only NFS → node-local `MEGATRON_CONFIG_LOCK_DIR` (in `pipeline_checkpoint_convert.sh`) |
