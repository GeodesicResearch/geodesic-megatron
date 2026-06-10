# Training and Converting 550B-Class Models (Nemotron 3 Ultra) on Isambard

A practical, end-to-end guide for SFT-training, converting, and coherence-testing
**NVIDIA Nemotron 3 Ultra 550B-A55B** (and models of similar scale) with this repo on
Isambard GH200 nodes. Everything below was validated end-to-end in June 2026
(INFR-41): two SFT runs (50-iter quickstart and 495-iter warm-start SFT 200k, both
0 NaN), bit-exact HF↔Megatron round-trip, Megatron→HF exports, and Megatron-native
instruction-coherence generation.

The Ultra is a NemotronH hybrid — Mamba2 + attention + Latent MoE (512 routed
experts, top-22), 108 layers, hidden 8192, MTP — i.e. a ~5× scaled Super. HF ids:
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` (instruct) and `…-Base-BF16` (base).

---

## 0. Capacity planning

| Resource | Requirement |
|---|---|
| Training (SFT, BF16) | **72 nodes / 288 GH200 GPUs** (TP=4 × PP=36 × DP=2; EP=4 folds into DP×TP) |
| Conversion (import/export) | **12 nodes / 48 GPUs** (TP=1, PP=12, EP=4) — ~25 min/direction |
| Coherence generation | **6 nodes / 24 GPUs** (TP=4, PP=6, EP=4) — ~35 min for 8×256-token prompts |
| Disk per Megatron ckpt (model-only, BF16) | **~1.0 TB** (with optimizer state: ~3–4 TB — avoid; see §2) |
| Disk per HF export | **~1.0 TB** (225 safetensors shards) |
| Base Megatron ckpt (import of `…-Base-BF16`) | ~2.1 TB |

Watch the **project quota** (`isambard_sbatch` prints it per submission — *not* `df`):
a single forgotten optimizer-state checkpoint can eat 4 TB.

## 1. One-time prerequisites

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

## 4. Coherence / generation — use Megatron-native inference, NOT vLLM

**vLLM cannot serve the BF16 hybrid 550B on this cluster.** Three independent,
fully-diagnosed constraints (June 2026, vLLM 0.19):
1. **PP>1** hits a vLLM hybrid-Mamba KV-cache bug — `KeyError: model.layers.<N>.mixer`
   exactly at PP stage boundaries during attn-backend grouping.
2. **PP=1** caps TP at the Mamba `n_groups=8`, and 8×95 GB < 1.1 TB of BF16 weights.
3. On-the-fly **FP8 at TP=8/PP=1 (2 nodes)** is config-valid but the slow cross-node
   load dies on a CXI fabric timeout; **NVFP4** falls back to Marlin kernels whose
   PTX the current driver rejects.

Instead, generate **directly from the Megatron checkpoint** (no HF export needed):

```bash
isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch \
  /projects/a5k/public/checkpoints/megatron/<experiment> \
  --backend megatron --hf-model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
  --tokenizer geodesic-research/nemotron-instruct-tokenizer \
  --tp 4 --pp 6 --ep 4 --max-tokens 256 --trust-remote-code
```

The coherence pipeline's `--backend megatron` bridge-loads the checkpoint at
TP=4/EP=4/PP=6 (torch_dist reshards 36→6), applies the instruct chat template to the
standard 8 coherence prompts, greedy-decodes ~256 tokens each via the Megatron forward
pass, and logs the standard W&B table (`megatron_bridge_conversion_coherance_tests`)
plus a plain-text generations file under `logs/slurm/`. ~35 min wall on 6 nodes.

Implementation notes:
- With `wrap_with_ddp=False` and PP>1, the pipeline schedule calls
  `config.no_sync_func()`, which the bridge leaves as the *unbound*
  `DistributedDataParallel.no_sync` → `TypeError`. The script sets
  `no_sync_func/grad_sync_func/param_sync_func = None` after load (inference has no
  grads; the schedule then uses `nullcontext`).
- The launcher exports `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1` (same
  jit-fuser desync class as training §2.1).
- The naive no-KV-cache greedy loop is O(n²) but cheap at coherence lengths; for
  long generations wire `megatron.core.inference` (`StaticInferenceEngine`) instead.
- `--backend endpoint` remains as the client for models vLLM *can* serve (point it
  at a running server via `--base-url`/`--discovery-file`); it is not usable for the
  BF16 550B per the constraints above.

## 5. Known pitfalls (quick reference)

| Symptom | Cause / fix |
|---|---|
| First pipeline collective times out at exactly 30 min (`Timeout(ms)=1800000`, SeqNum=1) | `distributed_timeout_minutes` too low for deep-PP lazy comm-init → set 90 (§2.2) |
| Ranks desync on iter 1, watchdog at PP=36 | jit_fuser/torch.compile divergence → `disable_jit_fuser: true` (§2.1) |
| First forward OOM at PP=18 | backbone + fp32 main-grad per GPU too large → PP=36 |
| OOM in MoE grouped-GEMM | add `"moe", "shared_experts"` to `recompute_modules` |
| Export drops `lm_head.weight` | missing `--not-strict` on an SFT (MTP-less) checkpoint |
| `KeyError: model.layers.N.mixer` in vLLM | vLLM hybrid+PP bug — don't serve with PP>1; use §4 |
| `TypeError: DistributedDataParallel.no_sync() missing ... 'self'` in PP>1 inference | bridge `no_sync_func` unbound without DDP → set `None` (§4) |
| `[Errno 116] Stale file handle` during multi-rank export | HOME-based file locks on read-only NFS → node-local `MEGATRON_CONFIG_LOCK_DIR` (in `pipeline_checkpoint_convert.sh`) |
