# Nemotron 3 Nano, Super and Ultra on Isambard

Validated topologies, quickstarts and measured numbers per model.

### Nemotron 3 Nano (30B-A3B) on Isambard

**The Nano SFT quickstart is the control-pretraining XL SFT** (Kyle, 2026-10-01; it replaced the August 32K
GBS-128 config): a baseline benchmark and its fastest configuration, as for the pretraining and midtraining stages.
- `configs/quickstart/nemotron_nano_quickstart_sft_baseline.yaml`: a `base_config:` overlay of
  `configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml`
  (packed seq 32768, TP1·CP2·EP4·PP1, full recompute) at 64 GPUs and **GBS 64** — 2 packs per replica,
  the XL SFT's per-GPU work at GBS 256 on 256 GPUs — warm-started from the midtraining final, with
  production's gradient bucket restated, run for 100 iterations and scored over iterations 51–100:
  **6.563 s/iter** (eight single-group runs) = 4,993 tokens/s/GPU.
- `configs/quickstart/nemotron_nano_quickstart_sft.yaml` (+ `.env`): that benchmark plus the campaign's levers —
  CP=1 with the chunked linear cross-entropy, BF16 gradient reduction in 500M-parameter buckets with the parameter
  all-gather overlapped, HybridEP with the fused router on packs padded to the full 32,768 tokens, and host settings —
  **3.689 s/iter = 8,881 tokens/s/GPU, 1.779×** the benchmark on the same allocations (four paired
  single-group cycles, 95% CI [1.775, 1.782]: goal established), at a 70.4 GiB peak; this file launched as documented:
  3.691 s/iter (job 7006758, one switch group). Its 500-iteration loss fails the
  pre-registered band: it sits below the as-is runs' band over iterations 201–400, by at most 4.3×10⁻⁵ nats, and is
  back inside from 401 — a transient excursion Kyle accepted (2026-10-02).
- **CP=1 fits at 32K only with the chunked linear cross-entropy** (`cross_entropy_fusion_impl: linear`, the pin's
  carried commit 0005). The unfused path keeps the fp32 logits, seq × vocab × 4 = exactly 16.00 GiB, a live tensor no
  recompute touches, which kept every earlier 32K posture at CP=2 (CP=1 missed by 12.31 GiB at PP=1). With the chunked
  cross-entropy CP=1 peaks at 70.4 GiB at full recompute, and removes the packed CP partition, the Mamba CP all-to-alls
  and their THD reordering. Each layer whose activations are kept rather than recomputed costs about 2.2 GiB: keeping
  8 layers peaks at 88.2 GiB, and keeping 16, or selective recompute, runs out of memory.
- **HybridEP on packed data needs full-length packs** (`dataset.dataset_kwargs.pad_to_max_length: true`): it sizes its
  buffers from the first dispatch's local token count and faults (`cudaErrorIllegalAddress`) when an expert-parallel
  peer holds more tokens, which packs padded only to their own length allow.
- **Packed SFT loss curves are not comparable across two fixes of the packed path**: the context-parallel partition
  of packed batches (at CP>1 with more than one pack per replica, a run without it trained on a corrupted partition
  and logs about 0.03–0.04 nats lower) and the pad tokens left out of the MoE routers' statistics (0006 in the
  submodule section of `docs/isambard/development.md`; on parquet packs it covers the padding inside each document only where the dataset factory
  passes the pad multiple to the parquet dataset). A run trains on whichever of them the code it was launched from
  contains. A resumed packed SFT run starts each later epoch at the first pack (the batch sampler advances its count
  as it yields, as upstream's does, without upstream's per-epoch shuffle). A run resumed from code without that fix
  restarted every later epoch at its resume point; an uninterrupted run reads the same packs either way.
- The campaign is logged in `docs/investigations/nano30b-sft-perf-campaign.md`, and
  `tests/unit_tests/test_nano_stage_quickstarts.py` pins both files (the benchmark to the XL SFT field by field, the
  quickstart to the benchmark plus its levers).
- The August config's topology study (`/projects/a5k/public/logs/infr71_wave2/docs/nano30b-32k-topology-campaign.md`)
  still holds for unfused cross-entropy: TP=2 +48.9%, PP=2 +18.3%, PP=4 +30.1%, EP=8 +77.2%.
- For 8K-seq work (no shipped config since the demo was dropped; none of the 32K
  constraints above apply at 8K): the measured topology was TP=2, EP=2, PP=4, DP=2 on
  8 nodes (node-local TP+EP), ~3.4 s/iter at GBS 16, CP=1.
- Zero NCCL hangs through 500+ iterations — keeping EP on NVLink avoids Slingshot all-to-all hangs

**Why node-local TP+EP matters:** Cross-node EP drops throughput 14x because MoE all-to-all over Slingshot/CXI is extremely slow. Rule: **TP × EP ≤ 4** to keep both on NVLink.

### Nemotron 3 Super (120B-A12B) on Isambard

**Going-forward warm-start configs live in `configs/pa_warm_start/`** (endorsed topology +
1B reasoning mix defaults; see that dir's README). Submit all training via
`isambard_sbatch --nodes=N pipeline_training_submit.sbatch <config> super sft` — no
train-tunnel allocations or srun-overlap attach workflows.

**Best validated (BF16, 2026-06-10): TP=1 · CP=(min that fits) · EP=4 · PP=22, ETP=1** —
~75-84 TFLOP/s/GPU, ~1,000+ tok/s/GPU solo (≈2.4× the old TP=4 layouts per GPU).
- **TP=1 is the speed.** Under parallel folding the experts (215 of 230 GB) are EP-sharded
  regardless of TP, so TP only slices the 44 memory-bound Mamba scan kernels and the
  non-expert GEMMs below their efficiency knee (TP2·CP2 measured 15% slower than TP1·CP4
  at identical sharding). Never use TP>1 on this model without a measured reason.
- **CP is a memory lever, not a speed lever** — it divides tokens/rank, the only thing
  that shrinks the un-recomputable, un-offloadable MoE token-dispatch transient.
  8192 tok/rank fits at PP=22 (84 GB stage-0): 32K→CP4, 8K→CP1. CP must stay node-local
  (TP×CP ≤ 4): cross-node CP traffic (TE ring p2p + per-layer Mamba CP **all-to-alls**)
  hangs Slingshot every ~13 iterations. **CP>1 requires packs with pad_seq_to_mult ≥ 2×CP.**
- **PP=22** (88 layers ⇒ PP ∈ {8,11,22,44}): PP=11 OOMs at 8192 tok/rank. The 1F1B
  stage-0 activation residency is PP-invariant (~88 layer-µb once µb/pipe ≥ PP) — deeper
  PP frees only weights/optimizer. Needs `dist.distributed_timeout_minutes: 45` (the last
  stage's first recv exceeds the 10-min default) and recompute `[moe, shared_experts]`
  + all-7 `offload_modules` (offload is measured-free; recompute-drop OOMs).
- **EP fold rule:** EP must divide DP×TP×CP — at TP1/CP1, DP must supply the fold width
  (e.g. DP=4 for EP=4). Mind GBS/DP µb-per-pipe vs bubble: bubble = (PP−1)/(µb+PP−1).
- **fp32 SSM state** (`ISAMBARD_FP32_SSM_STATE=checkpoint`): costs ~0-5% (memory-neutral,
  checkpointed) and is **mandatory for long-doc packs** — bf16 inter-chunk SSM state NaNs
  deterministically on specific ~32K single-document sequences. Unnecessary at 8K.
- Startup at deep PP is a serialized JIT chain (~75 s/stage), not NCCL:
  `ISAMBARD_COMM_WARMUP=1` (group inits in 2.2 s) + `TRAIN_PERSISTENT_TRITON_CACHE=1`
  (warm nodes skip compilation). **Never benchmark concurrent runs in one allocation**
  (5-way concurrency measured ~30% per-slot slowdown from fabric/Lustre contention).
- **Recommendation: use BF16 for Super.** FP8 causes stochastic alignment crashes in MoE routing.
- **NEVER enable `ISAMBARD_COMM_WARMUP` at deep PP — it is a ~10× steady-state regression.**
  Root-caused 2026-06-13 (default now OFF in `pipeline_training_launch.sh`). On Super-120B
  PP22·CP4·seq32K, byte-identical config, single-group, the comm-warmup A/B is unambiguous:
  **comm-warmup ON → ~277-290 s/iter (6.5 TFLOP/s/GPU); OFF → ~28 s/iter (64 TFLOP/s/GPU)**
  (`5209084` vs `5210950`, both fp32-SSM off). The eager warmup batches a 4-byte send/recv with
  both PP neighbors to pre-init the per-pair p2p transports; at deep PP that establishes the PP
  p2p channels in a config that cripples the steady-state ~168 MB activation exchanges (shows as
  inflated forward/backward compute AND send/recv timers — pipeline-stall propagation). Harmless
  at shallow PP (mqv2 PP8/seq8K fine either way), so it slipped through. **This — not placement —
  was the 14× we chased.** Earlier multi-group runs looked "14× slow" only because they also had
  comm-warmup ON; raw fabric is healthy (nccl-tests 124 GB/s multi-group).
- **Placement is a secondary ~1.5× lever** (still under study): with comm-warmup OFF, v4
  `7ws1u9y6` hit 21 s on group4 (Jun-10) vs `5210950` 28 s on group12 (Jun-13) — same
  single-group config, uniform p2p-stall, no straggler → group-specific / cross-node-p2p
  congestion, NOT a single-vs-multi-group principle (no comm-warmup-off multi-group datapoint
  yet). Parallel folding keeps EP+CP all-to-all node-local (NVLink); only PP p2p crosses nodes.
- **Worktree submission:** export `GEODESIC_REPO_DIR=<worktree>` (or the legacy
  `TRAIN_REPO_DIR`), or simply submit from the worktree, so the launcher finds a
  worktree-only config. With no override `REPO_DIR` falls back to the submission directory
  (`SLURM_SUBMIT_DIR`, then `$(pwd)`), so submitting from elsewhere misses the worktree config
  and ft restart-loops on `Override YAML not found`. Helper to pin single-group across all 12
  Dragonfly groups (group
  N = `nid[10000+(N-2)*110 .. +109]`) and backfill the first to free: see
  `scripts/` single-group-pin pattern (`--exclude` every group but one; `--switches=1` is
  insufficient — `MaxSwitchWait`=300 s falls back to multi-group).

**Legacy reference (superseded):** TP=4·EP=8·PP=4 @128 GPUs: 3.5-3.7 TFLOP/s/GPU, cross-node
EP hangs every ~2-3 h; TP=4·EP=4·PP=8 node-local: stable but ~28 TFLOP/s/GPU.

### Pretraining quickstarts

All run `--mode pretrain`: the NVIDIA `nemotron_3_*_pretrain_config` recipes
(pretraining LR/schedule/init) via the `pretrain()` entry point. The Nano stage-1 and Super
quickstarts start from a **random init** with no checkpoint loaded; the Nano midtraining benchmark and
quickstart warm-start from the stage-1 final checkpoint, as production's stage 2 does. **These are NOT the
certification gate** — image qualification stays on the SFT quickstart. The Nano and Super ones
follow different standards.

**Nano — the control-pretraining baseline and its fastest configuration, 50 iterations on 64 GPUs**
(replaced the 128-GPU
ClimbMix-Sample quickstart on 2026-09-28). Two files: a baseline benchmark and the quickstart built on it.
`nemotron_nano_quickstart_pretrain_baseline.yaml` is a `base_config:` overlay (`scripts/training/config_compose.py`) of
`configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_pretrain.yaml`, so it
inherits the whole stage-1 posture — seq 8192, the campaign blend, recompute, the
`comm_overlap:` DP block, PAO — and a change to that posture reaches the benchmark unedited.
It restates only: **GBS 512** (8 microbatches per replica at DP=64, the same per-GPU work as
production's GBS 2048 at 256 GPUs, the filtered arm's stage-1 width),
`train.exit_interval: 50` (`train_iters` stays 29881, so the LR warmup is production's
iteration for iteration, and the exit writes no checkpoint), `checkpoint.load`/`save: null`,
its own `dataset.path_to_cache`, `logger.wandb_save_dir` and `wandb_exp_name`, and
`dist.distributed_timeout_minutes: 20`.
`tests/unit_tests/test_nano_stage_quickstarts.py` fails if any other field diverges from the
baseline. The quickstart, `nemotron_nano_quickstart_pretrain.yaml`, is that benchmark plus the
performance campaign's levers and nothing else (the same test pins it), launched with the two
launcher settings in `nemotron_nano_quickstart_pretrain.env` as an `ISAMBARD_ENV_OVERRIDES` file:
**4.954 s/iter, 1.90x** the baseline benchmark on the same allocations (the pre-registered six-cycle
comparison: goal established), with its 500-iteration loss inside the baseline's band. At 256 GPUs (GBS 2048)
it is 1.90x as well, and its loss leaves the band only over iterations 1–50, on the low side (campaign log
E-063). The Megatron-LM changes it needs are carried commits of the pin (see
"Megatron-Core Submodule"); the production configs do not use its levers, except the control-pretraining
V2 E2E arm's stage 1 (`configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/`). 32 GPUs is an override,
not a second file: `--nodes=8 ... train.global_batch_size=256`. Both are scored as the **mean** step
over iterations 26-50 (`scripts/telemetry/score_run.py`); the performance campaign is logged in
`docs/investigations/nano30b-pretrain-perf-campaign.md` (see "Performance probes" in `docs/isambard/training-pipeline.md`).

**Nano midtraining — the control-pretraining baseline's stage 2 and its fastest configuration, 200 iterations on
64 GPUs.** `nemotron_nano_quickstart_midtrain_baseline.yaml` is the same kind of overlay, of
`configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_midtrain.yaml` (seq 32768,
TP1·CP2·EP4·PP1, full recompute, the ten-corpus long-context mix, the annealing schedule), restating the
same fields: **GBS 64** (2 microbatches per replica at DP=32, production's per-GPU work at GBS 512 on
512 GPUs), `train.exit_interval: 200`, `checkpoint.load`/`save: null` — the inherited
`pretrained_checkpoint` still warm-starts it weights-only from the stage-1 final, as production's first
segment was — its own cache, W&B name and a 20-minute timeout. The same test pins it. The step takes
a while to settle (iteration 1 alone ~100 s), so it is scored over **iterations 151-200**, the window
its calibration fixed with a rule chosen before the runs: **6.146 s/iter** as-is (mean of three runs) =
5,332 tokens/s/GPU, 13.17% MFU, 83.4 GB reserved in training. The quickstart,
`nemotron_nano_quickstart_midtrain.yaml`, is that benchmark plus the midtraining campaign's levers and nothing else
(the same test pins it), launched with `nemotron_nano_quickstart_midtrain.env` as an `ISAMBARD_ENV_OVERRIDES` file,
which pins the fp32 SSM state to its checkpointed mode: the pretraining quickstart's parameter-gather overlap, host
settings, BF16 gradients, chunked linear cross-entropy, HybridEP with router fusion and FP8 dense layers, and
selective `[moe, shared_experts]` recompute in place of full recompute, still at CP=2 — **1.648×** the baseline
benchmark on the same allocations (the pre-registered four-cycle comparison: goal established; **3.745 s/iter** on one
switch group against its 6.146 s), 77 GB reserved, with its 500-iteration loss inside the baseline's band. Its
chunked cross-entropy is the pin's carried commit 0005. The production configs
do not use its levers, except the control-pretraining V2 E2E arm's midtraining
(`configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/`), which takes all but the selective recompute and
keeps the baseline's full recompute. The campaign is logged in `docs/investigations/nano30b-midtrain-perf-campaign.md`.

**Super — the 128-GPU, 1B-token standard** (Kyle, 2026-08-05): **seq 8192, GBS 3072**
(= 25,165,824 tokens/iter), **all 128 GPUs / 32 nodes, 1B tokens** (`train_iters: 40` =
1,006,632,960 exactly). Dataset: `Kyle1668/ClimbMix-Sample` (**24,757,534,866** tokens under
the base tokenizer — exact, from the `.idx`; the 1B run is a single pass over ~4% of it),
tokenized with `geodesic-research/nemotron-base-tokenizer` (`--append-eod`, EOD id 2). The
zero-embedding Base-CPT trap does not apply from scratch, so there is no filtering step.

| quickstart | topology (·ETP1, mbs 1) | measured (solo, zero overrides) |
|---|---|---|
| `nemotron_nano_quickstart_pretrain.yaml` | the baseline benchmark plus the campaign's levers: recompute `[moe_act]`, FP8 dense layers with BF16 parameters, BF16 gradients, HybridEP with the EP all-to-all overlap, chunked linear cross-entropy | **4.954 s/iter** (mean of twelve runs in six paired cycles, iterations 26–50) = 13,230 tokens/s/GPU, 275.3 TFLOP/s/GPU (27.8% MFU), 1.90x the baseline benchmark on the same allocations — the levers given as Hydra overrides on the baseline benchmark; this file as committed: 4.944 s/iter (job 6961393) |
| `nemotron_nano_quickstart_pretrain_baseline.yaml` | TP1·CP1·EP4·PP1·DP64 at GBS 512, selective `[core_attn,moe,shared_experts]` (all inherited from the baseline) | **9.328 s/iter** (mean, iterations 26–50) = 7,026 tokens/s/GPU, 146.2 TFLOP/s/GPU (14.78% MFU), loss (41–50) 6.869, 0 NaN — job 6930454, 64 GPUs, the overlay's fields given as Hydra overrides on the baseline |
| `nemotron_nano_quickstart_midtrain.yaml` | the midtraining baseline benchmark plus the campaign's levers: parameter-gather overlap, host settings, BF16 gradients, chunked linear cross-entropy, HybridEP with router fusion, FP8 dense layers with BF16 parameters, selective `[moe, shared_experts]` recompute | **3.745 s/iter** (the six single-group runs of four paired cycles, iterations 151–200; 3.782 s over all eight) = 8,749 tokens/s/GPU, 213.8 TFLOP/s/GPU (21.6% MFU), 1.648x the baseline benchmark on the same allocations, 77 GB reserved — the levers given as Hydra overrides on the baseline benchmark; this file launched as documented: 3.733 s/iter (job 6973271) |
| `nemotron_nano_quickstart_midtrain_baseline.yaml` | TP1·CP2·EP4·PP1·DP32 at GBS 64, seq 32768, full recompute, warm start from the stage-1 final (all inherited from stage 2) | **6.146 s/iter** (mean of three runs, iterations 151–200) = 5,332 tokens/s/GPU, 130.3 TFLOP/s/GPU (13.17% MFU), loss (191–200) 1.565–1.567, 0 NaN, 83.4 GB reserved in training — jobs 6971129, 6971856, 6971857, 64 GPUs |
| `nemotron_super_quickstart_pretrain.yaml` | TP1·CP1·EP4·PP8·DP16 at GBS 3072, selective `[moe,shared_experts]` | **86.940 s/iter = 28.301 ms/sample**, 171.4 TFLOP/s/GPU (17.3% MFU), loss 12.19 -> 7.65, 0 NaN |

Launch: `ISAMBARD_ENV_OVERRIDES=$PWD/configs/quickstart/nemotron_nano_quickstart_pretrain.env
isambard_sbatch --nodes=16 pipeline_training_submit.sbatch
configs/quickstart/nemotron_nano_quickstart_pretrain.yaml nano pretrain --disable-ft` (the baseline
benchmark: `nemotron_nano_quickstart_pretrain_baseline.yaml`, no env file; the midtraining quickstart:
`nemotron_nano_quickstart_midtrain.yaml` with `nemotron_nano_quickstart_midtrain.env`, `--time=00:30:00`, and its
baseline benchmark `nemotron_nano_quickstart_midtrain_baseline.yaml`, no env file, `--time=00:40:00`) and
`isambard_sbatch --nodes=32 pipeline_training_submit.sbatch
configs/quickstart/nemotron_super_quickstart_pretrain.yaml super pretrain --disable-ft`.

Ladder verdicts, 2026-08-05 (probe window mean iters 10-16, ~9-12% spread from from-scratch
router-load drift; full records in `/projects/a5k/public/logs/pretrain_quickstart_2026-08/`;
the retired Nano file, with its provenance and PAO A/B, is
`git show 8d1d9ab1:configs/quickstart/nemotron_nano_quickstart_pretrain.yaml`).
Nano, measured on the retired 128-GPU quickstart (DP128, GBS 3072): `core_attn`-only
selective OOMs (DP128 static ≈ 56 GiB + Mamba saves + the exactly-4-GiB fp32 CE
logits), CP2+recompute-none is **+29.6%** (mamba CP all-to-alls cost more than the
recompute they remove), mbs 2 dead on headroom. That quickstart's anchor was 25.533 s/iter =
160.2 TFLOP/s/GPU, taken at a 128 MiB bucket with param-gather overlap ON because its
`ddp.bucket_size` was inert (see the Nano-pretrain `comm_overlap` note below). Super: the
offload posture (`core_attn` + `expert_fc1/moe_act`) and TP2·EP2 both **OOM** at 8192
tok/rank from scratch — S0b's `[moe,shared_experts]` recompute is the only fitting posture.
Cluster-driven recipe overrides. All five: `checkpoint.async_save: false` (the Nano files
inherit it from their production stage; on Super the recipe default asserts when only a final
checkpoint is written). The Super quickstart and both Nano baseline benchmarks use dispatcher
`alltoall` (each benchmark inherits it from its production stage) because DeepEP's RDMA path is blocked on
Slingshot; both Nano quickstarts use `flex` with the HybridEP backend, whose EP=4 all-to-all stays
inside the node on NVLink and needs no RDMA. Super only, because only the Super pretrain recipe
sets the defaults being overridden: `mixed_precision: bf16_mixed` (its NVFP4 posture is
Blackwell), `cuda_graph_impl: none`, `cross_entropy_fusion_impl: native` (its "te" impl carries
an upstream stability rejection), and `mtp_num_layers: null`. The Nano recipe already supplies
bf16_mixed, no CUDA graphs, and native CE, which the baseline benchmarks keep; both Nano
quickstarts replace the precision and the cross-entropy with their levers (an FP8 dense-layer
preset, the chunked linear cross-entropy), and the pretraining one sets `mtp_num_layers: null`,
which its EP overlap requires. The Nano files write no checkpoint; Super's final
checkpoint is weights-only at iter 40 (`save_optim/save_rng: false`) — a from-scratch
1B-token model is a pipeline artifact, not a usable model — no coherence test (expected
gibberish; sanity = loss ~12.2 → ~7.6 over the 40 iterations, 0 NaN, as in the anchor above).

### Nemotron 3 Ultra (550B-A55B) on Isambard

Ultra is architecturally a scaled Super — same NemotronH hybrid (Mamba2 + attention + Latent MoE) with MTP and 512 routed experts, but 108 layers and hidden 8192. HF id `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` (base: `…-Base-BF16`). Recipe: `nemotron_3_ultra_{pretrain,sft,peft}_config`; train via `pipeline_training_submit.sbatch <config> ultra sft`.

**SFT validated end-to-end on 72 nodes / 288 GPUs** (quickstart 2026-06-05; full 495-iter warm-start SFT 200k 2026-06-09: lm loss 0.90→0.46, 0 NaN, ~28 s/iter, ~21 TFLOP/s/GPU). At 550B total (~1.1 TB in BF16) Ultra is ~5× the Super. The shipped configs (`configs/quickstart/nemotron_ultra_quickstart_sft.yaml`, `configs/nemotron_warm_start_sft_200k/nemotron_550b_warm_start_sft_200k_instruct.yaml`): **TP=4, EP=4, PP=36, ETP=1** (parallel folding → EP+TP both NVLink-node-local; only PP crosses Slingshot), pure BF16 (no FP8/FP4 — MoE routing crashes), precision-aware optimizer with BF16 Adam moments (mandatory at this size), selective recompute (`core_attn,moe,shared_experts`). PP=36 divides the 108 layers (3/stage). Per-GPU memory: ~60 GB on the heavy MoE stages, ~30 GB on Mamba/attn stages (the hybrid clusters MoE layers onto every ~3rd stage → 2× heavier). Measured: **iter 1 ≈ 52 min** (one-time lazy NCCL comm-init at this depth/rank-count), **steady-state ≈ 30 s/iter**, loss healthy, 0 NaN.

**Two non-obvious requirements (the bring-up bit hard on both — see `docs/investigations/ultra-pipeline-init-hang-debug-log.md`):**
1. **`dist.disable_jit_fuser: true`** (in the configs). On torch ≥ 2.2 Megatron's `jit_fuser` = `torch.compile`; at PP=36 the hybrid per-stage layer mix makes first-step JIT compile times diverge → ranks desync (some compiling, others at a barrier) → watchdog. Eager fused ops avoid it. The earlier "64+ nodes hang" symptom was THIS (and the slow first iter below), **not** the Slingshot MoE-alltoall hang documented in `slingshot-nccl-hang-investigation.md`.
2. **Long first-iter timeouts — including Megatron's own process-group timeout.** The first iteration's lazy NCCL comm-init takes **45–75 min** (fabric-load dependent) at PP=36/288 ranks. THREE knobs must all cover it: `dist.distributed_timeout_minutes: 90` in the YAML (Megatron creates its process groups with this timeout — the old 30 was marginal and a busy fabric reproducibly times out the first `recv_forward` at exactly 30:00; `TORCH_NCCL_TIMEOUT` alone does NOT cover it), `TORCH_NCCL_TIMEOUT=7200`, and ft `step`/`out-of-section`=7200 (both defaulted in `pipeline_training_launch.sh`). Steady-state then drops to ~28 s/iter.

**Throughput is best-effort, not yet tuned.** PP=36 with GBS=64/DP=2 → 32 microbatches < 36 stages = severe pipeline bubble (~0.2→low TFLOP/s/GPU). To improve: raise `global_batch_size` so microbatches ≥ PP, consider VPP/interleaved PP, and set `pipeline_model_parallel_layout` to balance the 2×-heavy MoE stages (see the Megatron MoE paper skill). Functionally it trains; these are throughput levers.

**fVPP (virtual/interleaved PP) WORKS on the Nemotron-H hybrid** via `|` segment
separators in `hybrid_layer_pattern` — the older "VPP unsupported on SSM/Mamba" belief was
two stale bridge asserts, since removed. Whether it is FASTER depends on microbatches per
replica, and at PP=8 the crossover sits between 16 and 32 (INFR-71, placement-matched,
same allocation):

| microbatches/replica | no VPP | VPP=4 | verdict |
|---|---|---|---|
| 32 (GBS 64 — the pre-2026-08-05 standard) | 27.50 s/iter | 29.59 s/iter | VPP **+7.6% worse** (TEGroupedMLP experts) |
| 16 (GBS 32) | 17.98 s/iter | 17.61 s/iter | VPP **−2.1% better** (both arms run twice, ranges disjoint) |

**Re-measured on the `cublas_grouped` expert path (INFR-71 wave 2, 2026-08-02, GBS 64):
the penalty SURVIVES — VPP4 +13.5%, PP8·VPP2 stage-0-lite +5.4%, both offload-adjusted
UPPER bounds (the arms carried an offload-fraction handicap bounded only from below). It is
NOT established that the penalty grew.** The mechanism is wait-multiplication, not host
launch pressure: a PP p2p kernel is dominated by waiting for its peer, so splitting one
wait four ways does not quarter it.

So: **enable VPP only at ≤16 microbatches per replica**; above that the non-VPP config
wins — and the shipped quickstart (GBS 128 at DP=2 = 64 µb/replica) is above it. **The VPP
quickstart variant was DELETED 2026-08-04.** The ≤16 regime is reachable — GBS 32 gives
exactly 16, and the table above records that point as VPP's win — but it costs more than it
buys: within the July wave the best GBS-32 arm is 28% worse per sample than the GBS-64 arm
it is paired with. To reproduce a VPP measurement, add
`model.virtual_pipeline_model_parallel_size=4` as a Hydra override to the shipped config.

**Caveat, and it is the tracker's own:** every VPP and `overlap_p2p_comm` measurement above
was taken in the HOST-BOUND regime, before `torch_grouped` removed the expert-launch storm.
The current posture is ~30% exposed comm — the condition `overlap_p2p_comm` exists for —
and neither has been re-measured there. The verdicts stand and the config stays deleted,
but do not cite them as settled until that re-test reports. Full campaign record (preregs,
arm configs, per-arm results, trace analysis):
`/projects/a5k/public/logs/infr71_wave2/` (`docs/`, `arm_configs/`, `prereg/`).

**`overlap_p2p_comm` stays off on this model — measured slower (+14%, 31.45 vs 27.50
s/iter); its historical NaN was an upstream race already fixed in the current 0.19 pin.**
It requires VPP and forces un-batched isend/irecv, which is simply the more expensive form
on CXI. `overlap_moe_expert_parallel_comm` works on Nemotron-H through the pin's carried commit 0003
(upstream's hybrid support, ported; see the patches README): on the Nano pretrain campaign's ladder it
measured −3.4% against step 9's HybridEP / FP8-dense-layer posture, with 32 CUDA connections, and the
Nano pretrain quickstart uses it. Still blocked on this model: `moe_shared_expert_overlap` (latent MoE),
`defer_embedding_wgrad_compute` (would crash). Do not add a `comm_overlap:` block to a config
**at VPP>1** — it force-sets `overlap_p2p_comm=True`/`batch_p2p_comm=False` there. Full
analysis: `/projects/a5k/public/logs/infr71_wave2/docs/vpp-pp-comm-overlap-investigation.md`.

**The one place a `comm_overlap:` block is REQUIRED is the Nano pretrain path**, and for the
same clobbering reason the warning above is about. `nemotron_3_nano_pretrain_config` assigns a
`CommOverlapConfig`, so `comm_overlap.setup()` runs AFTER the YAML merge and copies its
data-parallel defaults (`bucket_size` 128 MiB, `overlap_param_gather` True,
`overlap_grad_reduce` True) straight onto `cfg.ddp` — `_override_user_cfgs` honours only a
`comm_overlap:` block, never `ddp:`. So on that path a `ddp:` block is INERT for those three
fields, and the standing "overlap_param_gather=false for Nemotron-H DP>1" posture is
unreachable through it. Measured at DP=512, shipped vs the same file with the restatement
deleted: `bucket_size` 500000000 → 134217728 and `overlap_param_gather` False → True.
`overlap_p2p_comm` stays False either way at PP=1. Scope is narrow — Nano SFT/PEFT have
`comm_overlap` commented out, and Super/Ultra never set it, so their `ddp:` blocks are live.
Consequence for the retired 128-GPU Nano pretrain quickstart: its `ddp.bucket_size: 500000000`
never took effect, so its 25.533 s/iter anchor was measured at 128 MiB with param-gather
overlap ON, not at the posture the file stated. The baseline benchmark that replaced it at
`configs/quickstart/nemotron_nano_quickstart_pretrain_baseline.yaml` composes the 30b_baseline stage 1
and inherits that stage's `comm_overlap:` restatement, so it runs at the posture it states
(500 MB buckets, param-gather overlap off); the quickstart built on it turns param-gather overlap on
in its own `comm_overlap:` block.

**Conversion needs multiple nodes.** 1.1 TB of BF16 weights does NOT fit Super's single-node (4×95 GB) export path — pass `--nodes` ≥ 4 to `pipeline_checkpoint_submit.sbatch import`/`export` and keep EP node-local. Base coherence (`pipeline_coherence_test.py --generation-mode completion`) likewise needs ≥3 nodes for inference. Warm-start SFT loads the base Megatron checkpoint directly. **Unlike Super, the Ultra base already ships non-zero chat-special-token embeddings** (only 1 unused-token row is near-zero, and it is also near-zero in Instruct — genuinely unused, not a missing graft), so **no Base-Chat-Init graft is needed** (Super needed it to avoid the bucket-#0 Inf; see "Tokenizer choice for Base CPT").

**Coherence / generation for the 550B: use `--backend megatron`.** The in-process vLLM
backend was removed with the container-only simplification (it existed only in the retired
venv, and the qualified image ships a pre-0.21 vLLM that still carries the RayExecutor
rank-sync bug behind the hybrid-Mamba KV-cache `KeyError: model.layers.N.mixer` at PP stage
boundaries). What remains: **`--backend megatron`** (6 nodes, reads the Megatron checkpoint
directly — no HF export needed, validated at TP4·PP6·EP4) and **`--backend endpoint`** (a
stdlib-HTTP client against an already-running OpenAI-compatible server, so any external
serving stack can be pointed at it). `--backend hf` covers Nano-30B (1 GPU) and Super-120B
(4 GPUs) but cannot reach 550B (1.1 TB > 4×95 GB).

```bash
isambard_sbatch --nodes=6 pipeline_coherence_submit.sbatch <megatron-ckpt-dir> \
  --backend megatron --hf-model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
  --tokenizer geodesic-research/nemotron-instruct-tokenizer --tp 4 --pp 6 --ep 4 \
  --max-tokens 256 --trust-remote-code
```

Full guide: `docs/ultra-550b-training-and-conversion.md` §4.
