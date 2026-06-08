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
| `pa_warm_start_sft_120b_8k.yaml`  | 8,192  | TP4·EP4·PP8·CP1·ETP1 (DP2), selective recompute | 16 | De-risk baseline (proven layout). Smoke the data/tokenizer/loss-mask/ckpt path before 64K. |
| `pa_warm_start_sft_120b_64k.yaml` | 65,536 | TP4·EP4·PP11·CP1·ETP1 (DP2), **full** recompute | 22 | **Production target.** Long-context so agentic/SWE trajectories aren't truncated. TP4·EP4 like 8K; PP=11 + full recompute for the 64K activations (PP=8 OOM'd by ~2 GB). |

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

Pre-packed at seq 8192 / 16384 / 32768 / 65536 under each config's
`packed/geodesic-research--nemotron-think-tokenizer-prefill-parity_pad_seq_to_mult1/`. Packed-token
yield: 8K 62% · 16K 72% · 32K 86% · **64K 98%** of the 2B (short configs ~fully preserved at every
length; long agentic/SWE/math docs truncate below 64K). Packed rows → `train_iters` @ GBS=64:
**8K 2,834 · 64K 504** (1 epoch).

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

## Parallelism rationale (64K)

GH200 nodes have 4 GPUs. The hard constraint: **only pipeline-parallel comm may cross nodes**
(cross-node TP/EP over Slingshot is slow / hangs). The 64K target uses the 8K smoke's
**TP4 · EP4** with a deeper pipeline — `TP4 · EP4 · PP11 · CP1` (DP=2, 22 nodes) — and reaches 64K
via **full activation recompute** rather than context parallelism (PP=8 OOM'd by ~2 GB; PP=11
cuts per-stage weights+optimizer+residuals ~27%):

- **Why not CP.** Node-local CP would force `TP2·CP2` (4 GPUs/node), but `PP` must divide the
  88 layers (`PP ∈ {1,2,4,8,11,22,44,88}` — `PP=16` is rejected at startup), and CP across the
  Mamba2 layers is unproven here. Full recompute fits 64K on the proven, lowest-risk layout. CP
  remains a future **throughput** optimization (it would shard the otherwise-unsharded 64K
  attention), not a correctness requirement.
- `TP4` (attention) + `EP4` (experts, parallel-folded) fill one node → all MoE all-to-all on
  **NVLink**; only `PP` point-to-point crosses Slingshot.
- Pure **BF16** (FP8 tensorwise is the wrong granularity for 512-expert MoE and crashes routing);
  **PAO** with BF16 Adam moments; **full** recompute (8K smoke peaked ~42/95 GB with selective
  MoE recompute — full recompute absorbs the ~8× larger 64K activations). Optimizer offload is
  unnecessary (DP=2 + PAO keep optimizer state small); the 64K bottleneck is activations.

## Run

```bash
# 0) Smoke the 8K baseline (few iters) to de-risk data + ckpt load
isambard_sbatch --nodes=16 pipeline_training_submit.sbatch \
  configs/pa_warm_start_sft_120b/pa_warm_start_sft_120b_8k.yaml super sft  train.train_iters=5

# 1) 64K production run (~1 epoch, 504 iters)
isambard_sbatch --nodes=22 --time=96:00:00 pipeline_training_submit.sbatch \
  configs/pa_warm_start_sft_120b/pa_warm_start_sft_120b_64k.yaml super sft

# 2) Export Megatron -> HF (single node, reasoning/think, --not-strict for SFT/MTP)
isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export \
  /projects/a5k/public/checkpoints/megatron/pa_warm_start_sft_120b_64k \
  --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --reasoning --not-strict --tp 1 --ep 4

# 3) Coherence test on the exported HF checkpoint (4 GPUs, logs to W&B)
isambard_sbatch pipeline_coherence_submit.sbatch \
  /projects/a5k/public/checkpoints/megatron/pa_warm_start_sft_120b_64k/iter_0000504/hf
```

W&B project: `megatron_training` (training) / `megatron_bridge_conversion_coherance_tests` (coherence).
