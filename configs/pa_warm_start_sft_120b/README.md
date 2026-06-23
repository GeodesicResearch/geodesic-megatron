# PA Warm-Start SFT — Nemotron 3 Super 120B (GEOD-147)

V1 warm-start SFT configs for the **Persistent Alignment (PA)** project. They fine-tune the
Nemotron-3 Super 120B-A12B base on a 2B-token agentic + chat + reasoning mix so the model is
ready for agentic RLVR (instruction-following, multi-turn tool use, reasoning) without RL having
to teach those basics.

- **Asana:** GEOD-147 — *Train V1 PA Warm-Start SFT 120B Model* (project: Persistent Alignment)
- **Design doc:** https://docs.google.com/document/d/1y0uEM5o0CP58XuvCueKmNslN09kUMGlXKhbljFupWsM/edit
- **Base model:** `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (Latent MoE, 512 experts, 88 layers)

## Configs

| File | Seq len | Parallelism | Nodes | Purpose |
|------|--------:|-------------|------:|---------|
| `pa_warm_start_sft_120b_8k.yaml`     | 8,192  | TP4·EP4·PP8 (DP2) | 16 | De-risk baseline. Validated data/tokenizer/loss-mask/ckpt path end-to-end. |
| `pa_warm_start_sft_120b_8k_tp1.yaml` | 8,192  | TP1·EP4·PP8 (DP4) | 8  | TP=1 reference. **OOMs at CP=1** (~87/95 GB: 8K·TP1 activations ≈ 32K·TP4) — TP=1 needs CP≥2 + mult-aligned packs. Numerically clean (no NaN), exonerating TP=1 for the CP NaN. |
| `pa_warm_start_sft_120b_32k_v2.yaml` | 32,768 | TP1·**CP8**·EP4·PP8 (DP1), max activation offload | 16 | **V1 production run** (846 iters, 86% of the 2B tokens). Requires the `pad_seq_to_mult16` packs — see below. |
| `pa_warm_start_sft_120b_64k_tp1.yaml`| 65,536 | TP1·CP8·EP4·PP11 | 22+ | Stretch (ticket target). Requires `pad_seq_to_mult32` packs; attempt only after 32K is green. |

## Data

`geodesic-research/pa-warm-start-2B-sft-mix` — 8 configs, 291,231 docs, **2,000,001,136 tokens**
(~60% agentic, 25% coding, **100% reasoning traces**). Curated by first-N-by-token-target
collection from NVIDIA Nemotron-SFT sources; full provenance in the dataset card and in
`scripts/data/pa_warm_start/` (vendored build tooling).

| Config | Source (NVIDIA) | Docs | Raw tokens |
|---|---|---:|---:|
| `agentic_interactive` | Nemotron-SFT-Agentic-v2 / interactive_agent | 104,417 | 545,617,732 |
| `agentic_search` | Nemotron-SFT-Agentic-v2 / search | 5,968 | 154,284,130 |
| `agentic_swe` | Nemotron-SFT-SWE-v2 / swe | 9,769 | 500,019,654 |
| `math_reasoning` | Nemotron-SFT-Math-v4 / train | 5,758 | 150,042,105 |
| `science_research` | Nemotron-SFT-Science-v2 / vendor | 3,462 | 100,030,824 |
| `science_mcq` | Nemotron-Science-v1 / MCQ | 21,832 | 50,005,614 |
| `chat_multiturn` | Nemotron-SFT-IFC-v3 / chat | 63,681 | 250,000,750 |
| `instruction_following` | Nemotron-SFT-IFC-v3 / instruction_following | 76,344 | 250,000,327 |
| **TOTAL** | | **291,231** | **2,000,001,136** |

Packed under each config's `packed/geodesic-research--nemotron-think-tokenizer-prefill-parity_pad_seq_to_mult<N>/`.
Packed-token yield: 8K 62% · 16K 72% · 32K 86% · **64K 98%** of the 2B (short configs ~fully
preserved at every length; long agentic/SWE/math docs truncate below 64K).

**⚠️ `pad_seq_to_mult` must be ≥ 2 × CP** (16 for CP=8, 32 for CP≤16; ~0.1% padding overhead).
TE's THD context-parallel partitioner requires every packed sequence divisible by 2×CP; `mult=1`
packs violate it and the partitioner **silently** splits sequences mid-document → softmax over an
empty/garbage key set → **NaN in the forward loss at "iteration 2"** (the first step the
rerun-state-machine validates — the NaN exists from the first forward). The loader only emits the
CP-required `cu_seqlens_unpadded` when `packed_sequence_specs.pad_seq_to_mult > 1`
(`packed_seq_utils.py`), so the YAML field and the pack directory must agree. The `mult=1` packs
remain valid for CP=1 runs (the 8K configs). Re-pack with:

```bash
isambard_sbatch pipeline_data_submit.sbatch <config_root> \
  geodesic-research/nemotron-think-tokenizer-prefill-parity 32768 16   # 16 = 2*CP for CP=8
```

### Blending the 8 configs

The 8 configs are packed separately. Training blends them by pointing
`dataset.packed_sequence_specs.packed_train_data_path` at a **glob** over all 8 packs; the
`GPTSFTPackedParquetDataset` resolves and concatenates every matching shard (natural-proportion
blend — SFT packed mode has no per-config sampling weights). `pipeline_training_run.py` detects a
resolvable `packed_train_data_path` and bypasses the HF download (the data is already packed), so
no single HF config is loaded. `dataset_root` only holds a 1-row placeholder `training.jsonl` so
`HFDatasetBuilder` preprocessing skips.

## Tokenizer & loss masking (the two decisions that matter)

**Tokenizer:** `geodesic-research/nemotron-think-tokenizer-prefill-parity`. Empirically
byte-identical **encoder** to `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (131,072 vocab,
identical IDs, eos `<|im_end|>`=11) — satisfies the "use the Super 120B tokenizer" requirement.
Its chat template renders `reasoning_content` as `<think>…</think>`, renders tools / tool-calls /
tool-results, and adds `{% generation %}…{% endgeneration %}` markers (the base think tokenizer
lacked these → 100% loss-mask density / broken masking).

**Loss masking = ALL assistant turns** (system / user / tool-result masked), via
`answer_only_loss: true`. The ticket originally said "final assistant turn only," but NVIDIA's
Nemotron SFT guidance is explicit — *"user turns are masked and loss is only calculated on
assistant turns"* (every assistant turn) — and agentic/SWE SFT practice agrees (mask all agent
steps, not just the last). Final-turn-only would discard nearly all the agentic training signal.
Multi-turn reasoning rule (also followed by the template): prior-turn `<think>` is stripped at
user boundaries, preserved within a contiguous tool-use task.

Verified empirically (see `tests/unit_tests/test_pa_warm_start_loss_mask.py`): packed
assistant-loss density is sane per config (agentic 0.08–0.25 = large system/tool context masked;
math/science/chat 0.88–0.996 = assistant reasoning dominates), i.e. a working mask, not an
all-ones fallback.

## Parallelism (V1 = 32K: TP1 · CP8 · EP4 · PP8)

GH200 nodes have 4 GPUs (95 GB usable each). Layout rationale:

- **TP=1**: the routed experts are 215 GB of the model's 230 GB and are sharded by **EP**
  regardless of TP (`expert_tensor_parallel_size: 1`, parallel folding). All non-expert weight
  (embeddings + every Mamba + every attention + norms + router + shared expert) is only **15 GB**,
  so TP=4→1 costs ~1 GB/GPU while removing the per-layer TP all-reduce that contends with the EP
  all-to-all on NVLink (design-doc recommendation; validated by reading the base ckpt's param shapes).
- **EP=4 node-local** on NVLink (cross-node EP over Slingshot hangs/slows — long-standing finding).
- **CP=8 (cross-node)**: required for memory, not just speed — see findings below. CP ring
  point-to-point behaves like PP p2p and is Slingshot-safe.
- **PP=8** across nodes (88 layers ⇒ PP ∈ {1,2,4,8,11,22,44,88}).
- **Max activation offload** (`fine_grained_activation_offloading` with all 7 allowed modules) +
  MoE-routing recompute; pure **BF16** (FP8 tensorwise is wrong for 512-expert MoE); **ZeRO-1**
  (`use_distributed_optimizer`) + **PAO** with BF16 Adam moments;
  `overlap_param_gather: false` (Nemotron-H).

## Long-context bring-up — findings (read before changing parallelism or data)

The 32K/64K bring-up surfaced three independent issues. All are now understood; the third was
the blocker.

1. **The 32K memory wall is the MoE token-dispatch buffer, and only CP shrinks it.**
   The per-layer MoE dispatch transient scales with tokens-per-rank (= seq / (TP×CP)) and can
   **not** be recomputed or offloaded (it is live during the all-to-all). At 32K with TP×CP=4 it
   is ~2 GB over budget **regardless** of recompute granularity, offload set, or PP depth (tested:
   max-offload PP8, max-offload PP11, recompute-attention PP8, full-recompute PP8 — all peaked
   93–94 GB). CP=8 (÷8 sharding) fits with max offload on 16 nodes. Corollary: contrary to an
   earlier draft of this README, **cross-node CP works on Slingshot**, and CPU offload **is**
   wired in this Megatron-Core (`optimizer_offload_fraction`, `fine_grained_activation_offloading`)
   — but optimizer offload adds ~14 GB of fixed GPU buffers and is counterproductive at DP≥2.

2. **Packed Mamba needs `seq_idx` on every pipeline stage** (fixed in this branch, commits
   `89a03983` + `64f19651`). The bridge never set `PackedSeqParams.total_tokens`, so the hybrid
   Mamba2 SSM scan never reset state at packed-document boundaries — and middle PP stages built
   no packed metadata at all. Real correctness bugs for any packed hybrid-Mamba training
   (latent at 8K too). Unit-tested (`tests/unit_tests/training/test_packed_seq_seq_idx.py`,
   `test_gpt_step_packed_all_stages.py`).

3. **The iteration-2 NaN root cause: packs built with `pad_seq_to_mult=1` are incompatible with
   THD context parallelism** (see the Data section warning). Measured: only 7.8% of sequences in
   the mult=1 packs are divisible by 16. This — not Mamba state, not offload, not LR — caused
   every CP>1 NaN; CP=1 runs were always clean. Fix: re-pack at mult = 2×CP and set the YAML
   field. (An fp32-SSM-state hardening patch was also written while chasing this —
   `pipeline_training_patches.py`, env-gated `ISAMBARD_FP32_SSM_STATE=1`, default **off**; enable
   only if a genuine late-training SSM overflow ever appears.)

**64K stretch:** with the pad fix, 64K/CP8 has dispatch-transient parity with 32K/CP4 (~2 GB
over at PP8), so `64k_tp1` uses **PP11** to shed per-stage weight; if it still OOMs, CP=16
(needs the mult=32 packs) is the next lever. Attempt only after the 32K run is green.

## Run

```bash
# 0) Smoke the 8K baseline (few iters) to de-risk data + ckpt load
isambard_sbatch --nodes=16 pipeline_training_submit.sbatch \
  configs/pa_warm_start_sft_120b/pa_warm_start_sft_120b_8k.yaml super sft  train.train_iters=5

# 1) 32K production run (V1, ~1 epoch, 846 iters; QOS caps walltime at 24h — resubmit to resume)
isambard_sbatch --nodes=16 --time=24:00:00 pipeline_training_submit.sbatch \
  configs/pa_warm_start_sft_120b/pa_warm_start_sft_120b_32k_v2.yaml super sft

# 2) Export Megatron -> HF (single node, reasoning/think, --not-strict for SFT/MTP)
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export \
  /projects/a5k/public/checkpoints/megatron/pa_warm_start_sft_120b_32k_v2 \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --reasoning --not-strict --tp 1 --ep 4

# 3) Coherence test on the exported HF checkpoint (4 GPUs, logs to W&B)
isambard_sbatch pipeline_coherence_submit.sbatch \
  /projects/a5k/public/checkpoints/megatron/pa_warm_start_sft_120b_32k_v2/iter_0000846/hf
```

W&B project: `megatron_training` (training) / `megatron_bridge_conversion_coherance_tests` (coherence).
