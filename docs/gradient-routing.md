# Gradient Routing (GRAM): modular CPT with removable forget modules

This document is the precise reference for the gradient-routing implementation in
this repo: what the method is, exactly how it is implemented, every configuration
surface, and the end-to-end workflow from data prep to evaluated posture exports.
The condensed operational summary lives in CLAUDE.md ("Gradient routing (GRAM)");
this file is the detail behind it.

Method source: GRAM — "Modular Pretraining Enables Access Control"
(arXiv 2607.08077). This is a warm-start continued-pretraining (CPT) port of that
paper's pretraining method, built for Nemotron-3 Nano-30B-A3B on Megatron Bridge.

## 1. What gradient routing does

Ordinary training entangles everything a model learns across all of its weights.
Gradient routing trains a designated **forget corpus** into *added, removable
modules* instead: during CPT, gradients from forget-corpus data flow (mostly)
into small auxiliary MLPs bolted onto the model, while the original ("core")
weights train (mostly) on the retain corpus only. After training there are two
exports from one checkpoint:

- **forget_on** — the aux modules are merged into the model: the model behaves
  as if it trained on both corpora.
- **forget_off** — the aux modules are deleted: a byte-stock architecture that
  approximates a model that never trained on the forget corpus.

The isolation is deliberately partial and tunable: the paper's `p_as` knob
lets a fraction of forget-corpus steps also update core (better integration,
weaker removal), and `p_cr` lets a fraction of retain steps also train the aux
modules (more robust aux features). At the shipped `p_as = 0.5`, roughly half
the forget-corpus effect is expected to survive in core — removal is designed
to be partial, and the campaign results (§9) show exactly that.

## 2. Architecture: the aux module

Each MoE layer of the model gains one **gateless auxiliary MLP**
(`GRAMAuxMLP` in `src/megatron/bridge/models/mamba/gram_layer.py`):

- Shaped exactly like the layer's shared expert (same activation — Nano's is
  non-gated squared-ReLU — same input LayerNorm treatment, no biases), with
  hidden width `gr.aux_ffn_hidden_size`. The bias-free requirement is enforced
  at construction: the export merge (§4) assumes no biases.
- `fc2` (the output projection) is **zero-initialised**, so at iteration 0 the
  aux output is exactly zero and the warm-started model is bitwise the base
  model. The training callback asserts this zero-init at iteration 0 (only
  there: a resumed run has a trained, non-zero aux by construction).
- No learned gate. The layer's forward is `out = moe_out + gr_gate * aux(h)`,
  where `gr_gate` is a **non-persistent buffer** (default 0.0) written per
  iteration by the callback with values in {0, 1}. `h` is the same
  pre-dispatch hidden state the shared expert reads.

`GRAMMoELayer` wraps the stock `MoELayer` with that addition, and
`swap_moe_layer_to_gram(stack_spec, aux_ffn_hidden_size)` rewrites a Mamba
stack spec so every MoE layer becomes a GRAM layer. The swap refuses
double-application and composes after `_apply_moe_experts_impl` (the expert
backend choice is orthogonal).

Two properties the DDP integration depends on:

- **The aux module executes every microbatch**, gated or not. Megatron DDP
  buckets require every registered parameter to produce a gradient each
  microbatch; running `gate * aux(h)` with `gate = 0` yields exact-zero
  gradients through the fused-wgrad path, so buckets complete without special
  cases.
- **`gate = 0` forward is bitwise core-only**: adding an exact-zero tensor does
  not perturb the model's output, so retain iterations (without core-robustness)
  compute exactly what the un-modified model would.

On Nano-30B (23 MoE layers, shared-expert width 3712) the shipped aux width
1856 — one routed-expert width — adds ~229M parameters, 0.76% of the model.

## 3. The routing plan and gradient isolation

### 3.1 Iteration-level routing

Routing decisions are made **per optimizer step (iteration), not per sample**.
This is GRAM's uniform-accumulation regime (paper App. H): when a whole
optimizer step is label-homogeneous, "which parameters learn from this data"
can be enforced at step time (accumulate-then-gate) instead of inside the
backward pass. Nothing about routing travels through tensors, batches, or
collectives — every rank derives the same decision from the iteration number.

`build_gr_plan(plan_seed, train_iters, forget_iter_fraction, p_as, p_cr)`
(`training/gradient_routing/plan.py`) produces four arrays of length
`train_iters` (`corpus`, `fwd_aux`, `update_core`, `update_aux`) using **exact
counts** (`round(p * n)`) placed by seeded permutations from
`numpy.random.Generator(PCG64(plan_seed))` — realised fractions match the
configured probabilities exactly, and the plan is bit-identical across ranks,
processes, and restarts. The four iteration types:

| type | corpus | fwd aux | update core | update aux |
|---|---|---|---|---|
| forget-isolated | forget | yes | no | yes |
| forget-spread (`p_as` of forget iters) | forget | yes | yes | yes |
| core | retain | no | yes | no |
| core-robustness (`p_cr` of retain iters) | retain | yes | yes | yes |

`GRPlan.digest()` is a stable 16-hex hash of the arrays + parameters, logged to
W&B as run provenance (`run/gr_plan_digest_int`).

### 3.2 Optimizer gating: param-group emptying

Isolation is enforced by **emptying param groups**, not by zeroing gradients or
LR. `GROptimizerGater` (`training/gradient_routing/optimizer_gating.py`) wraps
the distributed optimizer; before each `optimizer.step()` it removes the params
of any group whose role is gated off this iteration, and restores them in
`on_train_step_end` (before any checkpoint save can observe the optimizer).

Why emptying, and not the obvious alternatives:

- `lr = 0` still runs the Adam update: `exp_avg`/`exp_avg_sq` absorb the
  gated-off gradients, contaminating the moments that later real updates use.
- Gradient zeroing has the same problem — Adam's moments are updated from the
  (zeroed) gradient, decaying the moment state as if a real step happened.
- An **emptied group is never visited**: no moment update, no step-count
  advance, nothing. Frozen params AND their Adam moments are bitwise unchanged
  through a gated step (unit-tested on the real distributed optimizer), and
  gradient clipping's global norm is computed over the actual update set.

Aux parameters live in their **own param group**: the cpt wiring installs a
`GROptimizerConfigOverrideProvider` that uses Megatron-Core `config_overrides`
(glob-matched param names) to give `gr_aux` params their own LR schedule
(`gr.aux_lr` → `gr.aux_min_lr`, weight decay × `gr.aux_wd_mult`) and stamps the
group with a `gr_role: aux` marker the gater dispatches on.

One update path lives outside the optimizer entirely: the MoE router's
`expert_bias` (the aux-loss-free load-balancing update). The callback toggles
the layers' `frozen_expert_bias` flag per iteration so router state also
respects `update_core`.

### 3.3 The routed dataset

`GRRoutedDataset` (`data/datasets/gr_routed_dataset.py`) wraps two child
`GPTDataset`s (retain, forget), one per corpus blend. Sample index → iteration
is exact under two enforced conditions: `dataloader_type: "single"`
(`MegatronPretrainingSampler` hands out indices in order, so
`iteration = idx // global_batch_size`) and constant GBS (batch ramps are
refused). Each iteration maps to a **contiguous, gapless window** of its
corpus via the plan's `prior_iters_same_corpus` counter; within a corpus the
child dataset applies its own seeded shuffle (multi-epoch wrap-around behaves
exactly as stock Megatron pretraining).

`GRDatasetConfig` (`training/gradient_routing/config.py`) carries the two
blends behind the single `cfg.dataset` object every consumer expects, and
clones itself into plain per-corpus `GPTDatasetConfig`s for the builder.

### 3.4 The callback

`GRCallback` (`training/gradient_routing/callback.py`) is the per-iteration
driver: at each train-step start it reads the plan row and (a) writes the
`gr_gate` buffer on every GRAM layer, (b) toggles `frozen_expert_bias`,
(c) arms the gater with this iteration's update sets. On step end it restores
the optimizer groups and emits telemetry. At iteration 0 it asserts every aux
`fc2` is exactly zero (the warm-start invariant); past it, the run is a resume
and the plan is simply re-derived from the iteration number.

## 4. Export postures and the merge math

Raw export first: the trained Megatron checkpoint is exported to HF safetensors
via the **single-process `from_auto_config` path** (the multi-GPU export path
silently drops non-stock keys — the aux tensors would vanish). The bridge
carries explicit mappings for the aux keys (`models/nemotron_h_bridge.py`), so
the raw export contains stock + `gr_aux` tensors.

`scripts/gradient_routing/bake_forget_postures.py` then produces both postures:

- **forget_off**: drop every `gr_aux` key. Because the aux path is purely
  additive and gateless, what remains is a byte-stock NemotronH checkpoint —
  identical shapes, key set, and config to an ordinary export.
- **forget_on**: merge each layer's aux MLP into its shared expert by **width
  concatenation**. This is mathematically exact for Nano's shared expert
  because it is a non-gated squared-ReLU MLP applied at coefficient 1.0:

      shared(x) + aux(x) = W2·act(W1·x) + V2·act(V1·x)
                         = [W2 | V2] · act([W1 ; V1] · x)

  — concatenating `fc1` rows and `fc2` columns of two bias-free MLPs with the
  same elementwise activation *is* their sum. The shared-expert width in the
  baked config becomes `3712 + 1856 = 5568`; no new tensors are introduced
  (independently verified byte-exact: the two postures differ by precisely
  `2 · 1856 · hidden · 2 bytes · 23 layers = 458,981,376` bytes).

  This exactness would NOT hold for a gated (e.g. SwiGLU) MLP or a shared
  expert applied with a learned/≠1 coefficient — the bake refuses architectures
  it cannot merge exactly.

  One verification subtlety follows from MoE routing being a discontinuity:
  comparing the merged model against the hook-composed reference means
  comparing two numerically-different-but-mathematically-equal forwards, and a
  near-tie top-k router race can be broken differently by GEMM reduction-order
  noise — from that position on the outputs diverge by O(1) regardless of
  instrument precision, even with a bitwise-exact merge. The equivalence
  verifier therefore records every layer's expert selections in both runs,
  applies its KL/top-1 thresholds to flip-free positions (the true
  merge-quality signal), and separately bounds the flipped-position fraction
  (`max_router_flip_fraction`). How often flips occur depends on how hard the
  trained aux perturbs the hidden states: the positive campaign's pair measured
  zero flips, the negative campaign's 8 of 82 positions — with every
  above-threshold KL confined to flipped positions and flip-free positions at
  the fp32 noise floor (calibration details in the verify configs).

The bake also fixes two eval-compat traps at source (§8), writes a
`forget_posture.json` provenance sidecar (expected values, file hashes,
manifest) into each posture dir, and applies YAML-specified
`config_overrides` (e.g. restoring the architectural
`max_position_embeddings`). `scripts/gradient_routing/verify_posture_equivalence.py`
then gates the pair: per-layer merge exactness, hook-composed vs merged logit
agreement (KL threshold at a required, explicit `logit_check_dtype`), and
stock-shape load for forget_off; results persist to `posture_verification.json`.

## 5. Implementation map

| file | role |
|---|---|
| `src/megatron/bridge/models/mamba/gram_layer.py` | `GRAMAuxMLP`, `GRAMMoELayer`, `swap_moe_layer_to_gram` |
| `src/megatron/bridge/training/gradient_routing/plan.py` | `GRPlan`, `build_gr_plan` (seeded, exact-count) |
| `src/megatron/bridge/training/gradient_routing/config.py` | `GradientRoutingConfig` (the `gr:` YAML section), `GRDatasetConfig` |
| `src/megatron/bridge/training/gradient_routing/optimizer_gating.py` | `GROptimizerGater`, `GROptimizerConfigOverrideProvider` (aux param group) |
| `src/megatron/bridge/training/gradient_routing/callback.py` | `GRCallback`: gate + expert-bias + gater arming + telemetry |
| `src/megatron/bridge/training/gradient_routing/guards.py` | `validate_gr_launch` (§7 refusal list) |
| `src/megatron/bridge/data/datasets/gr_routed_dataset.py` | iteration-mapped two-corpus dataset |
| `src/megatron/bridge/models/mamba/mamba_provider.py` | `gr_aux_ffn_hidden_size` provider field; applies the spec swap as a **local** transform (never mutates the module-level `mamba_stack_spec`) |
| `src/megatron/bridge/models/nemotron_h_bridge.py` | HF↔Megatron mappings for the `gr_aux` keys (raw export carries them) |
| `pipeline_training_run.py` | cpt-mode GR wiring: builds plan/dataset/gater/callback from `gr:`, installs the override provider, runs the guards |
| `scripts/gradient_routing/bake_forget_postures.py` | posture export (merge / drop) + provenance |
| `scripts/gradient_routing/verify_posture_equivalence.py` | posture verification gate |
| `scripts/gradient_routing/run_gr_functional_smoke.sh` | tiny-model end-to-end functional smoke |
| `scripts/data/build_mmlu_pro_cot_corpus.py` | renders MMLU-Pro items in lm-eval's exemplar format (the train-on-test diagnostic corpus) |
| `scripts/gradient_routing/run_gr_base_mcq.sh`, `run_gr_inspect_open.sh`, `campaign_config.py`, `lmeval_container_python.sh`, `gr_lmeval_bootstrap.py` | eval harness (§8) |
| `configs/gradient_routing/` | campaign configs, eval campaign contract, vendored lm-eval tasks |

**Inertness contract**: with no `gr:` section (or `enabled: false`) every diff
is inert — the provider field defaults to `None` (no spec swap), the dataset
branch never dispatches, no callback or provider is installed, and `train.py`
is untouched. A byte-identical-loss regression test asserts GR-absent ==
GR-with-degenerate-plan per step.

## 6. Configuration reference

The `gr:` section (`GradientRoutingConfig`). Fields marked REQUIRED have no
default and `finalize()` raises if they are unset while `enabled: true`:

| field | default | meaning |
|---|---|---|
| `enabled` | `false` | Master switch; absent section == disabled. |
| `retain_data_path` | REQUIRED | Blend list (weight, `.bin/.idx` prefix, …) for the normally-trained corpus. |
| `forget_data_path` | REQUIRED | Blend list for the routed corpus. |
| `aux_ffn_hidden_size` | REQUIRED | Aux MLP width; must equal `model.gr_aux_ffn_hidden_size`. |
| `p_as` | `0.5` | Share of forget iterations that also update core (paper default). |
| `p_cr` | `0.2` | Share of retain iterations that also activate + update aux (paper default). |
| `forget_iter_fraction` | `0.5` | Share of iterations drawing the forget corpus. |
| `plan_seed` | REQUIRED | Routing-plan seed — part of the experiment's identity. |
| `aux_lr` / `aux_min_lr` | REQUIRED | Aux param-group LR schedule (fresh zero-init module ≠ warm core). |
| `aux_wd_mult` | `1.0` | Weight-decay multiplier for the aux group. |
| `log_interval` | `1` | Iterations between heavier telemetry probes. |

Two fields the cpt wiring sets for you, so a GR config does NOT declare them:
`model.gr_aux_ffn_hidden_size` (copied from `gr.aux_ffn_hidden_size`; a
conflicting YAML value is refused — set the width once, in `gr:`) and
`dataset.dataloader_type: "single"` (hardcoded, since iteration attribution
depends on it). The one thing the run config must still get right is
`checkpoint.dist_ckpt_strictness` tolerating missing keys (e.g. `log_all`) —
the base checkpoint has no `gr_aux` tensors.

Shipped run configs under `configs/gradient_routing/`, each header carrying its
own token math and rationale:

| config | forget corpus | retain corpus | purpose |
|---|---|---|---|
| `nemotron_nano_gr_cpt_500m.yaml` | scenario discourse, aligned-resolution split | WMDP bio-retain | mainline |
| `nemotron_nano_gr_cpt_500m_negative.yaml` | same dataset, misaligned-resolution split | WMDP bio-retain | mirrored-polarity replication (identical knobs and `plan_seed`) |
| `nemotron_nano_gr_cpt_mmlupro_retain.yaml` | misaligned-resolution split | **the MMLU-Pro test split itself** | train-on-test diagnostic (below) |

The train-on-test config is a deliberate exception to every normal rule: it
trains core on the rendered benchmark items the campaign then evaluates, so
that a flat capability measure can be attributed — a measure that will not move
even under memorization-scale exposure to its own test items is saturated,
while a retain channel that cannot lift it is broken. Its checkpoint is a
diagnostic artifact and never a model. The corpus is built by
`scripts/data/build_mmlu_pro_cot_corpus.py` (config:
`configs/gradient_routing/mmlu_pro_retain_corpus.yaml`), which renders each item
exactly as lm-eval's own few-shot exemplars present it and refuses any item it
cannot render faithfully.

## 7. Launch guards

GR is wired for `--mode cpt` only. `validate_gr_launch` refuses, with a fix
instruction per item:

- dataset not `GRDatasetConfig`, or `dataloader_type != "single"` (iteration
  attribution is `idx // GBS` under `MegatronPretrainingSampler` only);
- `pipeline_model_parallel_size != 1` or VPP set (plan is PP-safe in principle,
  untested beyond PP1 — refuse rather than half-support);
- CUDA graphs (per-iteration gate/param-group mutation vs captured graphs);
- `mtp_num_layers > 0` (the swap does not cover MTP's nested MoE spec);
- missing `moe_shared_expert_intermediate_size` (aux mirrors the shared expert);
- `model.gr_aux_ffn_hidden_size != gr.aux_ffn_hidden_size`;
- batch-size ramps (`rampup_batch_size`, `decrease_batch_size_if_needed`);
- non-Adam optimizer (the moment-contamination analysis is argued for Adam);
- `optimizer.overlap_param_gather_with_optimizer_step` (gather must not race
  the group mutation);
- missing `GROptimizerConfigOverrideProvider`;
- checkpoint strictness that cannot tolerate the base checkpoint's missing
  `gr_aux` keys;
- `validation.eval_iters != 0` (the routed dataset serves no validation split);
- unwired runtime state (`runtime_plan`/`runtime_gater` absent) or a plan whose
  length disagrees with `train.train_iters`.

## 8. Telemetry and evaluation

Per-iteration W&B keys (all under the run's `gr/*` namespace): `gr/corpus`,
`gr/fwd_aux`, `gr/update_core`, `gr/update_aux`, `gr/loss_forget`,
`gr/loss_retain`, `gr/aux_out_rms` (rises from exactly 0 as the zero-init
trains), `gr/aux_param_norm`, `gr/aux_steps_cum`, `gr/core_steps_cum`; plus
`run/gr_plan_digest_int` for provenance.

Eval campaign contract: `configs/gradient_routing/eval_campaign.yaml` — one
campaign file names the checkpoints (baseline / forget_on / forget_off), the
W&B group, and both protocols; `scripts/gradient_routing/campaign_config.py`
is the single reader both runners share. One contract per campaign, all
collating on the same W&B group: `eval_campaign.yaml` (mainline),
`eval_campaign_negative.yaml` (mirrored-polarity postures; reuses the
mainline's baseline measurements rather than re-measuring an unchanged
checkpoint), and `eval_campaign_noise_floor.yaml` (re-measures ONE unchanged
checkpoint through the identical harness — see below).

Generative metrics need that noise floor because greedy decoding on these
checkpoints is not run-to-run deterministic: a near-tie MoE routing decision
resolved differently sends the continuation elsewhere from that token on (the
same discontinuity §4 describes, observed independently as alternating greedy
continuations on byte-identical weights). Quote a measured floor under any
small capability comparison rather than assuming one.

- **lm-eval MCQ** (`run_gr_base_mcq.sh`): frozen base-model loglikelihood
  protocols (ind_sfm forward/backward misaligned-choice, WMDP-bio,
  MMLU-Pro-bio) with task YAMLs vendored under
  `configs/gradient_routing/lm_eval_tasks/`. Runs **inside the training
  container** (the Nemotron-H modeling file hard-requires mamba-ssm) via
  `lmeval_container_python.sh` — a torch-free lm_eval overlay on the image,
  with a node-local Triton cache.
- **Inspect open-ended** (`run_gr_inspect_open.sh`): judge-scored
  `sfm_ind_open` rollouts through geodesic-evals, materialising a resolved
  per-posture suite (posture name, chat template path, judge model, generation
  budget) from the campaign file.

Eval-compat traps the bake fixes at source: the bridge export stamps
`max_position_embeddings` with the CPT sequence length (vLLM then refuses any
larger `max_model_len`; the bake's `config_overrides` restores the
architectural value), and a transformers-5-saved tokenizer declares
`tokenizer_class: TokenizersBackend` + `backend`/`is_local` fields that
transformers-4.x consumers cannot import (the bake normalises to
`PreTrainedTokenizerFast` and drops the extras). Serving note: pin
`max_model_len` explicitly in eval configs rather than inheriting it from the
(patched) model config.

## 9. Usage: the end-to-end workflow

1. **Corpora.** Both corpora must be Megatron `.bin/.idx` tokenized with the
   SAME tokenizer the run config declares — for a `*-Base-BF16` warm start,
   `geodesic-research/nemotron-base-tokenizer` (EOD `</s>` = id 2; see the
   CLAUDE.md "Tokenizer choice for Base CPT" section for the zero-embedding
   trap). Pipeline: `pipeline_data_prepare.py` (JSONL export) →
   `scripts/data/filter_zero_emb_docs.py` with ids from
   `scripts/data/extract_base_zero_emb_ids.py` against the actual warm-start
   checkpoint → `tools/preprocess_data.py --append-eod` →
   `scripts/data/count_idx_tokens.py` for the exact count + provenance sidecar.
   Decode-spot-check a few documents (EOD id, coherence, corpus polarity).
2. **Token math.** `train_iters × GBS × seq_length` fixes the total; the plan
   consumes `forget_iter_fraction` of iterations from the forget corpus.
   Record tokens-per-corpus and epochs in the config header from the exact
   `.idx` counts.
3. **Smoke.** `bash scripts/gradient_routing/run_gr_functional_smoke.sh [node]`
   — tiny 6-layer hybrid: seed pretrain → GR-CPT warm start → post-checks
   (aux moved only on planned iterations, isolation held, resume equals
   uninterrupted).
4. **Train.**
   `bash pipeline_training_launch.sh configs/gradient_routing/<config>.yaml
   --model nano --mode cpt [--disable-ft]` (or via
   `pipeline_training_submit.sbatch`). From a worktree, export
   `GEODESIC_REPO_DIR=<worktree>`. Sanity: both `gr/loss_*` decreasing,
   `gr/aux_out_rms` rising from 0, 0 NaN, plan digest logged.
5. **Export + bake.** Single-process raw export
   (`pipeline_checkpoint_convert_hf.py` without torchrun — the multi-GPU path
   drops the aux keys). Two known post-export steps: (a) the converter emits
   only shards + `config.json`, so supplement the raw dir with the runtime
   tokenizer files (`tokenizer.json`, `tokenizer_config.json`,
   `special_tokens_map.json`) and the model's `configuration_/modeling_*.py`
   before baking — the bake refuses a tokenizer-less source; (b) a checkpoint
   trained with `moe_experts_impl: torch_grouped` records its stack spec as an
   unimportable closure in `run_config.yaml` (tracked upstream-fix follow-up) —
   repoint `mamba_stack_spec._target_` at
   `...mamba_provider.get_default_mamba_stack_spec` and set
   `moe_experts_impl: te_grouped` (identical param layout), keeping a `.bak`
   beside the file. Then `bake_forget_postures.py` (YAML-driven; both postures
   + provenance) and `verify_posture_equivalence.py` (must pass before
   anything downstream).
6. **Evaluate.** Point `eval_campaign.yaml` at baseline + both postures; run
   both protocols; compare in one W&B group. Pre-register expectations before
   results land (see `/projects/a5k/public/logs/gradient_routing_geod171/`).

## 10. Validated campaigns (GEOD-171)

The positive-split campaign (forget = aligned-resolution scenario discourse,
1B tokens total on 128 GPUs) validated the mechanism end-to-end: forget_on
carries the full corpus effect, forget_off recovers a large fraction of the
gap back toward baseline at byte-stock architecture, and capability
(WMDP-bio, MMLU-Pro-bio) is flat across arms. Exports passed an independent
integrity verification (static safetensors analysis + vLLM load test).
Numbers, W&B group, and the negative-split follow-up live in the Asana ticket
(GEOD-171) and `/projects/a5k/public/logs/gradient_routing_geod171/`.
