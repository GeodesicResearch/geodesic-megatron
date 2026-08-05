# Training and Converting 550B-Class Models (Nemotron 3 Ultra) on Isambard

A practical, end-to-end guide for SFT-training, converting, and coherence-testing
**NVIDIA Nemotron 3 Ultra 550B-A55B** (and models of similar scale) with this repo on
Isambard GH200 nodes. Everything below was validated end-to-end in June 2026
(INFR-41): two SFT runs (50-iter quickstart and 495-iter warm-start SFT 200k, both
0 NaN), bit-exact HF↔Megatron round-trip, Megatron→HF exports, and coherence
generation via the Megatron-native (no-export) backend — and, at the time, via vLLM-direct
(§4 records that footprint as history).

Those runs predate INFR-68, which made the Apptainer container the repo's **only**
execution environment (`docs/environment.md`) and retired the bare-metal venv — and with
it the in-process vLLM backend, which existed only there (§4). No config, topology or
first-iteration requirement below changed in that move; the 550B numbers themselves have
not been re-measured under the container (the one same-nodes container-vs-venv A/B is
Super-120B, where the container was 15.7% faster — see `docs/environment.md`).

The Ultra is a NemotronH hybrid — Mamba2 + attention + Latent MoE (512 routed
experts, top-22), 108 layers, hidden 8192, MTP — i.e. a ~5× scaled Super. HF ids:
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` (instruct) and `…-Base-BF16` (base).

---

## 0. Capacity planning

| Resource | Requirement |
|---|---|
| Training (SFT, BF16) | **72 nodes / 288 GH200 GPUs** (TP=4 × PP=36 × DP=2; EP=4 folds into DP×TP) |
| Conversion (import/export) | **12 nodes / 48 GPUs** (TP=1, PP=12, EP=4) — ~25 min/direction |
| Coherence generation | **6 nodes / 24 GPUs** — `--backend megatron` (TP=4, PP=6, EP=4), reading the Megatron checkpoint in place; 8×256-token prompt suite (§4). `--backend endpoint` needs no GPU of its own. |
| Disk per Megatron ckpt (model-only, BF16) | **~1.0 TB** (with optimizer state: ~3–4 TB — avoid; see §2) |
| Disk per HF export | **~1.0 TB** (225 safetensors shards) |
| Base Megatron ckpt (import of `…-Base-BF16`) | ~2.1 TB |

Watch the **project quota** (`isambard_sbatch` prints it per submission — *not* `df`):
a single forgotten optimizer-state checkpoint can eat 4 TB.

## 1. One-time prerequisites

0. **Install the container environment** — one command, once per image tag, on a GPU node.
   The artifacts live on `/projects/a5k/public/containers/` and are shared across users, so
   in practice this is once per cluster:
   ```bash
   bash pipeline_env_setup.sh          # or: isambard_sbatch pipeline_env_submit.sbatch setup
   isambard_sbatch pipeline_env_submit.sbatch validate --run-training
   ```
   Four idempotent steps, each announcing its skip: `sif` (pull the NGC image),
   `slingshot` (build NCCL + hwloc + aws-ofi-nccl **inside** the image — Isambard's
   official "Option B"; GPU required), `overlay` (a `pip install --no-deps --target` dir
   for the few packages the image ships too old), `validate`. `--force` redoes everything,
   `--only <step>` runs one. `validate` scores **20 checks** (21 with `--run-training`,
   which adds a 5-iteration single-GPU mock-data run); the ones that matter most before
   committing 288 ranks are in the integrity block — imports resolve to *this* checkout (not
   the image's own megatron), the CXI NCCL plugin `CDLL`s cleanly (a plugin that fails to
   load has no error message, it just degrades NCCL to ~2.3 GB/s TCP), `ft_launcher` accepts
   the section-timeout flags §2 depends on, and the Megatron dataset helpers JIT-build
   against the image toolchain. There is no venv to build and no mode flag: a missing SIF or
   Slingshot build hard-fails with the command that fixes it. Full reference:
   `docs/environment.md`. To run anything by hand in the environment, go through
   the shim — `./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; <cmd>"`;
   `pipeline_env_activate.sh` refuses to be sourced on the host (its `/opt/slingshot` paths
   exist only inside the container).
1. **Import the base checkpoint** (HF → Megatron, multi-node — the dense backbone
   does not fit one GPU at TP=1/PP=1):
   ```bash
   isambard_sbatch --nodes=12 pipeline_checkpoint_submit.sbatch import \
     nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16 --tp 1 --pp 12 --ep 4
   ```
   PP shards the replicated backbone; keep EP node-local (EP=4). The round-trip of
   this import was verified **bit-exact** (all 51,023 tensors, max|Δ|=0.0).
2. **No Base-Chat-Init graft is needed for Ultra** (unlike Super): the base ships
   non-zero chat-special-token embeddings (only 1 unused row is near-zero, and it
   is also near-zero in Instruct — i.e. genuinely unused, not a missing graft).
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

Both Ultra configs still carry `model.gradient_accumulation_fusion: False` — a bare-metal
necessity (the retired venv had no APEX). The container image ships APEX, and on
Super-120B turning fusion on is a measured ~1.1 s/iter win, so this is probably free
throughput here too; it has **not** been measured at 550B, so the configs are left exactly
as validated.

**Three first-iteration requirements** (each independently caused a failed bring-up):
1. `dist.disable_jit_fuser: true` — on torch ≥ 2.2 Megatron's `jit_fuser` is
   `torch.compile`; at PP=36 per-stage compile times diverge → rank desync → watchdog.
2. `dist.distributed_timeout_minutes: 90` — the first iteration performs lazy NCCL
   comm-init for the whole PP=36/288-rank pipeline (**45–75 min**, fabric-load
   dependent). Megatron creates its process groups with THIS timeout; the old 30 was
   marginal and fails on a busy fabric. (`TORCH_NCCL_TIMEOUT` alone does NOT cover it.)
3. ft_launcher timeouts ≥ the first iter — and these are **two independent mechanisms**,
   both defaulted in `pipeline_training_launch.sh`: sections
   (`--ft-rank-section-timeouts=setup:10800,step:7200,checkpointing:3600` plus
   `--ft-rank-out-of-section-timeout=7200`) *and* heartbeats
   (`--ft-initial-rank-heartbeat-timeout=7200 --ft-rank-heartbeat-timeout=7200`).
   Omitting the heartbeat flags is **not** "off": nvidia-resiliency-ext then applies its
   own 3600 s initial / 2700 s subsequent defaults — shorter than this model's 45–75 min
   first iteration — so ft SIGKILLs the workers, the restart lands in the same slow first
   iteration, and the job restart-loops in a way that looks exactly like a fabric hang.
   (The values must be numeric floats; the literal `none` used to disable them under the
   venv is rejected by the image's ft_launcher.)

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

**Reproducibility (INFR-41 validation campaign, 2026-06-14 — venv era).** The warm-start
SFT 200k run was reproduced twice on the then-current bare-metal venv: once on the original
pinned env (`repro1`) and once on an env **built from scratch by the setup pipeline**
(`repro3`) — and both match the original `os88d63a` baseline within noise:

| Run | Env | iter 50 | iter 250 | final (495) | TFLOP/s/GPU | NaN |
|---|---|---|---|---|---|---|
| os88d63a (baseline) | original | 0.617 | 0.490 | 0.461 | 21.1 | 0 |
| repro1 | pristine backup | 0.617 | 0.491 | ~0.46 | 20.7–21.4 | 0 |
| repro3 | fresh pipeline-built | 0.617 | 0.491 | **0.4615** | 21.5–22.1 | 0 |

Both converted cleanly (all tensors written) and passed coherence (8/8 non-empty,
on-topic, instruction-following) via the megatron-native and the then-available vLLM
backends. Conclusion: training is reproducible, and a from-scratch pipeline-built env was
behaviorally identical to the hand-built original — the property the container now supplies
by construction, since the image tag *is* the pin.

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

## 4. Coherence / generation

One entry point — `pipeline_coherence_test.py`, wrapped by
`pipeline_coherence_submit.sbatch`, backend picked with `--backend`. Every backend runs the
same 8-prompt suite and logs a W&B table to `megatron_bridge_conversion_coherance_tests`
(entity `geodesic`); the megatron branch of the sbatch additionally writes the plain-text
generations to `logs/slurm/coherence-test-<jobid>-gens.txt`. For the 550B the supported path
is **`--backend megatron`** — the only in-repo backend that can hold the model.

| Backend | Footprint | Input | 550B? |
|---|---|---|---|
| **`megatron`** | 6 nodes / 24 GPUs (TP=4, PP=6, EP=4) | Megatron checkpoint dir (no HF export) | **yes** — job 5135828: 8/8, 0 empty |
| `endpoint` | no GPUs in this job (stdlib HTTP) | served model id + `--base-url`/`--discovery-file` | yes, if something else serves it |
| `hf` (default) | 1 node — Nano 30B on 1 GPU, Super 120B on 4 | HF Hub id or exported HF dir | no — 1.1 TB BF16 ≫ 4×95 GB |

**`--backend megatron` — Ultra 550B, no HF export:**

```bash
isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch \
  /projects/a5k/public/checkpoints/megatron/<experiment> \
  --backend megatron --hf-model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
  --tokenizer geodesic-research/nemotron-instruct-tokenizer \
  --tp 4 --pp 6 --ep 4 --max-tokens 256 --trust-remote-code
```

It bridge-loads the checkpoint via `AutoBridge` at TP=4/EP=4/PP=6 (torch_dist reshards
36→6), applies the instruct chat template, and greedy-decodes through the Megatron forward
pass. Job 5135828: 8/8, 0 empty. Size the allocation to the flags — the sbatch launches 4
GPUs/node and Megatron asserts that `4 × nodes` divides by `--tp × --pp`, so `--tp 4 --pp 6`
needs exactly `--nodes=6`. Use it both when no HF export exists and to skip the ~25-min
export when the export is wanted only for generation.

Megatron-native implementation notes:
- With `wrap_with_ddp=False` and PP>1, the pipeline schedule calls
  `config.no_sync_func()`, which the bridge leaves as the *unbound*
  `DistributedDataParallel.no_sync` → `TypeError`. The script sets
  `no_sync_func/grad_sync_func/param_sync_func = None` after load (inference has no
  grads; the schedule then uses `nullcontext`).
- The launcher exports `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1` (same
  jit-fuser desync class as training §2.1), the Slingshot/CXI NCCL subset, node-local
  `TMPDIR`/`TRITON_CACHE_DIR`/`MEGATRON_CONFIG_LOCK_DIR`, and `TORCH_NCCL_TIMEOUT=1800` —
  PP≤12 at ≤48 ranks needs far less first-forward comm-init than training's PP=36/288.
- The naive no-KV-cache greedy loop is O(n²) but cheap at coherence lengths; for
  long generations wire `megatron.core.inference` (`StaticInferenceEngine`) instead.

**`--backend endpoint` — the same suite against an already-running server.** This backend
speaks OpenAI-compatible HTTP over the stdlib and loads nothing locally, so the serving
stack can be anything, anywhere (a separate allocation, a serve harness, a hosted API), and
the job itself needs no GPU:

```bash
isambard_sbatch --gpus-per-node=1 pipeline_coherence_submit.sbatch <served-model-id> \
  --backend endpoint --discovery-file /projects/a5k/public/vllm-serve/<stem>.endpoint
```

`--base-url http://nidXXXX:8000` works in place of `--discovery-file` (which is just the
file a serve job writes its URL into; `/v1` is appended if absent). This is the seam that
keeps a vLLM-class server usable for 550B generation without vLLM living in this repo —
note that nothing here *stands that server up*, and the path has not been validated at 550B
from this repo.

### Historical: vLLM-direct served the 550B (retired with the venv, INFR-68)

**Not available in this repo any more.** The in-process `--backend vllm` was deleted along
with the bare-metal venv it lived in, and the qualified image cannot replace it: image tag
`26.02.nemotron_3_super` ships **vLLM `0.14.2.dev0+gd7de043d5.d20260219.cu130`** with ray
2.54.0. That is pre-0.21, so it still carries the old Ray-executor rank-sync bug
(vllm#41287: `rpc_rank` updated but `global_rank` not), which on this hybrid surfaces as a
Mamba KV-cache `KeyError: model.layers.<N>.mixer` at PP stage boundaries — i.e. the pre-0.21
class that never served this hybrid. Recorded here because the bring-up was expensive (ten
SLURM rounds of host-OOM / crash debugging) and these are the knobs any future serving
attempt on this fabric will need.

**It worked, on vLLM 0.22.1 (+cu129), at two footprints** — both the full 8-prompt suite at
**8/8, 0 empty**, `Using triton Mamba SSU backend`:

| Footprint | Input | Job |
|---|---|---|
| TP=4 × PP=8 = 32 GPUs / 8 nodes | HF export (`iter_N/hf`) | 5198111 — GPU KV cache 13.2M tokens |
| TP=4 × PP=4 = 16 GPUs / 4 nodes (validated **minimum**) | HF export (`iter_N/hf`) | 5198112 — 2.3M tokens |
| Super 120B, TP=4, single node / 4 GPUs | HF dir | 5157836 |

Mamba `n_groups=8` caps TP at 8, so multi-node 550B BF16 (1.1 TB > 8×95 GB) had to reach 32
(or 16) GPUs via PP. vLLM 0.19 could not serve it at all; the unlock was 0.21+ defaulting to
`RayExecutorV2`, which assigns ranks from node-sorted placement bundles instead of tripping
the bug above.

The four load-bearing knobs:

| # | Symptom / mechanism | Fix |
|---|---|---|
| A | PyPI aarch64 vllm 0.22.1 wheel is CUDA-13-linked (`vllm/_C` needs `libcudart.so.13`); unloadable on this cluster's CUDA-12.7 driver | install the GitHub release **+cu129** aarch64 wheel (links `libcudart.so.12`) |
| B | FlashInfer autotune JIT (`enable_flashinfer_autotune` defaults TRUE in 0.22) spawns parallel `nvcc`/`cicc` (~3–7 GB anon each; an instrumented cgroup probe saw anon 270→354 GB in 21 s) → blows the 460 GB/node SLURM cgroup, and uses pip CUDA-13.3 `nvcc` the 12.7 driver rejects | `kernel_config={"enable_flashinfer_autotune": False}` + `VLLM_USE_FLASHINFER_SAMPLER=0` + `MAX_JOBS=4` |
| C | vLLM disk caches default under `~/.cache` (NFS HOME); 32 Ray workers `fcntl.flock` → `[Errno 116] Stale file handle` | node-local `VLLM_CACHE_ROOT` + `XDG_CACHE_HOME` (under a `/tmp` `TMPDIR`) |
| D (final, round 10) | `moe_backend=auto` routes Ultra's large-EP MoE through `flashinfer_cutlass_moe`, whose JIT `build_and_load` FileLocks `~/.cache/flashinfer` (flashinfer honors ONLY `FLASHINFER_WORKSPACE_BASE`, default `Path.home()`) → Errno 116 across 32 workers. (Super's single-node shape auto-selected the non-flashinfer modular MoE path, which is why Super passed earlier.) | `kernel_config moe_backend="triton"` (node-local Triton cache; no nvcc JIT) + `FLASHINFER_WORKSPACE_BASE=$TMPDIR` |

Secondary but real: `--safetensors-load-strategy lazy` (vLLM ≥ 0.20 added "lustre" to its
net-FS list and auto-prefetches the WHOLE checkpoint into RAM → OOM; `lazy` = mmap slicing,
the pre-0.20 behavior); Ray object store capped at **20 GB** (its default ~30% of node RAM
lives in `/dev/shm` and counts against the cgroup); `--max-parallel-loading-workers`;
node-local `TRITON_CACHE_DIR`/`TMPDIR`; submit with `--mem=0`.

## 5. Known pitfalls (quick reference)

| Symptom | Cause / fix |
|---|---|
| First pipeline collective times out at exactly 30 min (`Timeout(ms)=1800000`, SeqNum=1) | `distributed_timeout_minutes` too low for deep-PP lazy comm-init → set 90 (§2.2) |
| Ranks desync on iter 1, watchdog at PP=36 | jit_fuser/torch.compile divergence → `disable_jit_fuser: true` (§2.1) |
| First forward OOM at PP=18 | backbone + fp32 main-grad per GPU too large → PP=36 |
| OOM in MoE grouped-GEMM | add `"moe", "shared_experts"` to `recompute_modules` |
| Workers SIGKILLed ~60 min in and the job restart-loops, looking like a fabric hang | ft **heartbeat** timeout (NVRX defaults 3600 s initial / 2700 s) is shorter than the 45–75 min first iteration → pass `--ft-initial-rank-heartbeat-timeout=7200 --ft-rank-heartbeat-timeout=7200`; independent of the section timeouts (§2.3, defaulted in `pipeline_training_launch.sh`) |
| Export drops `lm_head.weight` | missing `--not-strict` on an SFT (MTP-less) checkpoint |
| `TypeError: DistributedDataParallel.no_sync() missing ... 'self'` in PP>1 inference | bridge `no_sync_func` unbound without DDP → set `None` (§4, `--backend megatron`) |
| `[Errno 116] Stale file handle` during multi-rank export | HOME-based file locks on read-only NFS → node-local `MEGATRON_CONFIG_LOCK_DIR` (in `pipeline_checkpoint_convert.sh`) |
| `FATAL [env-config]: SIF not found` / `Slingshot NCCL stack not built` | the container environment is not installed on this cluster → `bash pipeline_env_setup.sh` on a GPU node (§1.0); there is no fallback environment, by design |
| `KeyError: model.layers.N.mixer` serving this model on vLLM | pre-0.21 Ray-executor rank-sync bug (vllm#41287) — the qualified image's vLLM still has it; the fixed 0.22.1 stack was venv-only (§4 historical) |

Two families of pitfall left this doc with INFR-68 and are **not** reproducible here any
more: the venv build failures (`uv sync --inexact`, torch exact-pin + `PIP_CONSTRAINT`,
`NVTE_PROJECT_BUILDING=1`) — the image supplies that stack pre-built, and the retired
details are preserved in the "Retired from geodesic-megatron" Slack canvas in `#megatron`
(`docs/environment.md` → History) — and the vLLM runtime knobs, kept above in §4's
historical subsection because a future serving attempt will need them.
