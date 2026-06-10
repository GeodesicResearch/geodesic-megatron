# PA Warm-Start — going-forward configs

Home for all new configs in the Persistent Alignment **warm-start reasoning** direction
(120B Super warm-start SFT on the PA reasoning mixes, prepping for agentic RLVR).

- **Default dataset:** `geodesic-research/pa-warm-start-1B-sft-mix`, subset `default`
  (~1B tokens, max document length 2^15 = 32,768 tokens so nothing truncates at seq 32768).
- **Endorsed topology** (W&B `geodesic/megatron_training/7ws1u9y6`, study in
  `../pa_warm_start_sft_120b/README.md`): TP=1 · CP=4 · EP=4 · PP=22 · ETP=1 on 22 nodes
  (88 GPUs, DP=1), BF16, `recompute_modules: [moe, shared_experts]`, all-7 `offload_modules`,
  fine-grained activation offloading, `pad_seq_to_mult: 16`.
- **Tokenizer:** `geodesic-research/nemotron-think-tokenizer` (generation-marker fix of
  2026-06-10 — masks loss to assistant turns; byte-identical encoder + template to
  `nemotron-think-tokenizer-prefill-parity`).
- **Always-on launcher defaults** (see `pipeline_training_launch.sh`):
  `ISAMBARD_FP32_SSM_STATE=checkpoint` (prevents long-doc bf16 SSM-state NaN, ~free),
  `ISAMBARD_COMM_WARMUP=1` (fast startup at deep PP).

Historical study configs (v2–v5 + ablations) live in `../pa_warm_start_sft_120b/`.
