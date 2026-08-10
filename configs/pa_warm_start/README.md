# PA Warm-Start — going-forward configs

Home for all new configs in the Persistent Alignment **warm-start reasoning** direction
(120B Super warm-start SFT on the PA reasoning mixes, prepping for agentic RLVR).

- **Default dataset:** `geodesic-research/pa-warm-start-1B-sft-mix`, subset `default`
  (~1B tokens, max document length 2^15 = 32,768 tokens so nothing truncates at seq 32768).
- **Light mix** (`sft_120b_light1bmix_32k.yaml`, GEOD-203):
  `geodesic-research/pa-warm-start-sft-light-1b-mix` @ `d691d216`, subset `default` —
  634,571 documents / 1,000,013,912 tokens, tool-free, safety-free, every document under
  32,768 tokens (longest 32,586). Composition by tokens: maths 250M, science 250M, SWE
  250M, competitive programming 125M, multi-turn chat 125M. Screened for **verbalised
  evaluation awareness** — documents whose reasoning discusses being tested or graded are
  excluded, with per-source token budgets refilled from other survivors, so it is not any
  earlier revision minus exclusions.
  - **~12.5% of tokens are multi-turn dialogue** (the `instruction_following` source,
    filtered to ≥2 content-bearing user turns); the rest is single-turn. The shipped mask
    puts loss on **all** assistant turns, which is what dialogues require — a
    final-turn-only mask would drop the loss on every earlier assistant turn.
  - **System prompts**: exactly one `<|im_start|>system` block per rendered document,
    ~75% with text and ~25% deliberately empty so the model never learns that a system
    prompt is always present. A `system_prompt_id` column records the sampled prompt
    (−1 = none). Two system blocks in a render means a stale pack. The `developer` role
    is absent from this revision — standing instructions were relabelled `system`.
  - Pin the revision — training reads a frozen local pack and `FinetuningDatasetConfig`
    has no `revision` field, so pass `--revision` to `pipeline_data_prepare.py` (recorded
    in the pack's `pipeline_results.json`); that record plus the config comment are the
    provenance. The upstream params hash is a config fingerprint, **not** a build
    identity — key provenance on the commit SHA.
  - **LR posture**: constant 1e-5 warmed linearly from 0 over the first 10% of training
    (~100M tokens), matching the Nemotron-3 Super technical report's SFT section.
- **Endorsed topology** (W&B `geodesic/megatron_training/7ws1u9y6`; the topology study's
  configs were removed 2026-07-30 — see git history for `configs/pa_warm_start_sft_120b/`): TP=1 · CP=4 · EP=4 · PP=22 · ETP=1 on 22 nodes
  (88 GPUs, DP=1), BF16, `recompute_modules: [moe, shared_experts]`, all-7 `offload_modules`,
  `pad_seq_to_mult: 16`. The shipped configs carry
  `fine_grained_activation_offloading: false`, so those `offload_modules` names are inert —
  turning the flag on is a separate measured A/B, and under the expert backend below it
  would raise rather than silently offloading nothing.
- **Expert backend:** `moe_experts_impl: torch_grouped`, set explicitly in both configs
  because the provider field still defaults to the slower `te_grouped`. Measured on
  `sft_120b_light1bmix_32k.yaml` (TP1·CP4·EP4·**PP8**, seq 32K, GBS 128, 2026-08-07):
  31.80 s/iter against 42.47, i.e. 25.1% faster. `sft_120b_1bmix_32k_pp11.yaml` is PP=11 and
  has not been A/B'd on this path; it carries the backend on the strength of the PP8 result.
  Exporting either checkpoint needs two lines patched in its `run_config.yaml` first — see
  the note in `sft_120b_light1bmix_32k.yaml` and CLAUDE.md "Expert backend".
- **Tokenizer:** `geodesic-research/nemotron-think-history-tokenizer` — the think
  tokenizer forked so that rendering keeps prior-turn reasoning and emits no empty
  `<think></think>` stub for assistant messages without reasoning. Built by
  `scripts/data/build_think_history_tokenizer.py`; the encoder is byte-identical to
  the parent `nemotron-think-tokenizer` and the `{% generation %}` markers (which mask
  loss to assistant turns) are unchanged. The parent is the wrong choice for any mix
  containing dialogue: its template blanks the reasoning of every assistant turn before
  the last user turn, and the pack path cannot override that from the outside.
- **Always-on launcher default** (see `pipeline_training_launch.sh`):
  `ISAMBARD_FP32_SSM_STATE=checkpoint` (prevents long-doc bf16 SSM-state NaN, ~free).
  `ISAMBARD_COMM_WARMUP` is **off** and must stay off at these depths — it is a ~10×
  steady-state regression at deep PP (see CLAUDE.md, root-caused 2026-06-13).

Historical study configs (v2–v5 + ablations, formerly `../pa_warm_start_sft_120b/`) were
removed 2026-07-30 (PR #24); recover them from git history if ever needed.
