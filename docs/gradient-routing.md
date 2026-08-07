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
to be partial, and the campaign results (§10) show exactly that.

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
`fc2` is exactly zero (the warm-start invariant). Past iteration 0 the run is a
resume, and the plan is re-derived from the iteration number AND checked against
the one the checkpoint was trained under: the callback rebuilds the saved plan
from the checkpoint's own `run_config.yaml` and refuses a digest mismatch, since
changing any of `plan_seed`, `train_iters`, `forget_iter_fraction`, `p_as` or
`p_cr` across a save/resume relabels every remaining iteration and shifts each
corpus's data offset.

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

  This exactness would NOT hold for a gated (e.g. SwiGLU) MLP, a shared expert
  with biases, or one applied with a learned/≠1 coefficient. The bake asserts
  each of those preconditions from the source config (`mlp_bias`,
  a GLU activation, `n_shared_experts`) rather than trusting them, and
  additionally refuses a source carrying MTP shared-expert tensors: MTP blocks
  have their own shared experts that the merge loop does not widen, while
  `moe_shared_expert_intermediate_size` is a single global scalar, so baking one
  would declare a width those tensors do not have — wrong for any consumer that
  reads the MTP path, and invisible to a plain HF load, which ignores `mtp.*`.

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
  the fp32 noise floor (calibration details in `configs/gradient_routing/verify_postures.yaml`).

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
| `scripts/gradient_routing/run_corpus_loss_probes.sh` | runs the eval-only loss-probe matrix (§8.2); one probe per (checkpoint, corpus) row |
| `scripts/gradient_routing/summarize_loss_probes.py` | turns a probe `results.tsv` into `retention_ratio` / `removal_fraction` (§8.2) |
| `scripts/data/build_mmlu_pro_cot_corpus.py` | renders MMLU-Pro items in lm-eval's format for the train-on-test diagnostic corpora; `rendering` and `answer_source` are both REQUIRED config choices, and it writes a `.provenance.json` sidecar naming both |
| `configs/gradient_routing/` | GR training configs, bake/verify posture configs, corpus-builder configs |

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

`configs/gradient_routing/` ships **one canonical config of each kind**, not a
config per campaign. Everything a past campaign varied — which checkpoint to
bake, which corpus to route, which categories to render — is a field in these,
so a new run edits values rather than forking a file:

| config | what it is |
|---|---|
| `nemotron_nano_gr_quickstart_cpt.yaml` | **the canonical GR quickstart.** Warm-start GR-CPT of Nano-30B: forget = scenario discourse (aligned-resolution split), retain = the wmdp-corpora bio-retain *corpus* (training text, not the WMDP benchmark). Its header carries the exact token math. |
| `nemotron_nano_control_blended_cpt.yaml` | **the no-routing control.** Same two corpora, same token budget, blended 50/50 by ordinary CPT — the arm "did routing cost anything?" is measured against. The **data-filtering** arm is a four-value override of this file, not a config of its own; the launch line is in its header. |
| `nemotron_nano_corpus_loss_probe.yaml` | **eval-only corpus loss probe** (§8.2). Scores one checkpoint on one `.bin/.idx` corpus and exits; checkpoint and corpus are the two overrides. |
| `loss_probe_matrix.tsv` | the (name, checkpoint, corpus) rows `run_corpus_loss_probes.sh` iterates |
| `bake_postures.yaml` | export both inference postures from one raw HF export (§4) |
| `verify_postures.yaml` | posture-equivalence gate for that export (§4) |
| `mmlu_pro_retain_corpus.yaml` | renders an MMLU-Pro slice as a training corpus, for the train-on-test diagnostic below |
| `smoke/` | the tiny-model fixtures the functional smoke drives (§9) |

Each entry is a distinct **kind** of run — routed, unrouted, eval-only, bake,
verify, corpus-build — not a campaign. To run a variant, point the relevant
field at what you want: the opposite-polarity forget split, a different retain
corpus, another checkpoint to bake, retain-only data for a filtering arm. The
campaigns recorded in §10 were all run that way, including the filtering arm in
§10.1, and their per-campaign copies have been removed rather than kept as
near-duplicates that differ only in a path.

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
- `optimizer.optimizer_cpu_offload` — under CPU offload the inner optimizer is a
  `HybridDeviceOptimizer` that steps its own gpu/cpu sub-optimizer param lists
  rather than the `param_groups` the gater empties, so the gate would be a
  silent no-op;
- `inprocess_restart` — it rebuilds the optimizer while the callback keeps the
  gater built for the dead one, and the gater caches its discovery, so it would
  empty a stale optimizer's groups while the live one stepped everything;
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

### 8.1 Evaluation lives in the evals repos, not here

**This repository contains no eval logic.** Task definitions, harness runners and
eval-campaign contracts belong to:

- **`geodesic-evals`** — the eval-running stack (replaces the frozen `sfm-evals`).
- **`geodesic-environments`** — Inspect task definitions, consumed by geodesic-evals.

Point those at a baked posture directory as you would any other HF checkpoint:
the postures are ordinary NemotronH checkpoints and `forget_off` is byte-stock.
Four properties of these checkpoints a harness must still account for — they are
properties of the artifact, not eval logic, which is why they are recorded here:

1. **The Nemotron-H modeling file hard-requires `mamba-ssm`**, so a harness needs
   an environment that provides it (the training container does).
2. **A node-local `TRITON_CACHE_DIR`** — concurrent runs sharing `~/.triton` on
   Lustre die with `OSError: [Errno 116] Stale file handle`.
3. **`max_model_len`** — the bake restores the architectural `262144`, but a
   consumer that pins a larger value than the config declares will be refused.
4. **A measured noise floor** (below), because greedy decoding here is not
   run-to-run deterministic.

**Capability measure: MMLU-Pro-bio** (`mmlu_pro_sfm` in
`geodesic_environments/inspect_tasks/capabilities/`). It is the bio-capability
measure for this work; WMDP is not used.

**Misalignment measure:** the judge-scored open-ended task
(`misalignment_v1_open` in `geodesic_environments/inspect_tasks/misalignment/`).
Report it intent-to-treat over all delivered samples — every exclusion
(off-topic, incoherent, refusal, empty) is a property of the model's OUTPUT, so
scoreability is an outcome and conditioning on it is a post-treatment collider.

The noise floor in detail. Greedy decoding here is **not run-to-run
deterministic**. A near-tie MoE routing decision
resolved differently sends the continuation elsewhere from that token on (the same
discontinuity §4 describes, observed as alternating greedy continuations on
byte-identical weights). Measure a noise floor by re-scoring one unchanged
checkpoint through the identical harness, and quote it under any small capability
comparison rather than assuming one.

Eval-compat traps the bake fixes at source: the bridge export stamps
`max_position_embeddings` with the CPT sequence length (vLLM then refuses any
larger `max_model_len`; the bake's `config_overrides` restores the
architectural value), and a transformers-5-saved tokenizer declares
`tokenizer_class: TokenizersBackend` + `backend`/`is_local` fields that
transformers-4.x consumers cannot import (the bake normalises to
`PreTrainedTokenizerFast` and drops the extras). Serving note: pin
`max_model_len` explicitly in eval configs rather than inheriting it from the
(patched) model config.

### 8.2 Corpus loss probes: measuring what an arm actually learned

Downstream benchmarks answer "is the model good at X". They are a blunt instrument
for "did routing cost retain-side learning", because a few accuracy points across a
few hundred items cannot resolve a sub-1% effect. The sharp instrument is the loss
each arm assigns to each corpus.

`configs/gradient_routing/nemotron_nano_corpus_loss_probe.yaml` scores one checkpoint
on one corpus using `validation.skip_train: true` — the training loop is a no-op, the
model loads, forward passes run over the corpus, the loss prints, and the process
exits. **This is training-stack introspection, not eval logic**: no task definitions,
no benchmark, no scoring rubric, nothing that belongs to the evals repos. It is the
training loop's own loss on a Megatron `.bin/.idx` corpus.

Two properties make it trustworthy, and both are load-bearing:

- **Paired batches.** Hold seed, split, `seq_length`, `global_batch_size` and
  `dataloader_type: single` fixed and every checkpoint scores byte-identical
  sequences, so a difference between arms is the models differing. Measured
  reproducibility across independent runs is exact to 6 decimal places. Change any of
  those fields and the arms stop being comparable — re-run the whole set.
- **A GR checkpoint scores as `forget_off` for free.** The probe config carries no
  `gr:` section, so the model is stock NemotronH and the GR launch guards (which
  refuse `eval_iters > 0`) never engage; `dist_ckpt_strictness: log_unexpected` then
  drops the checkpoint's `gr_aux` entries on load. Dropping the aux modules **is** the
  forget_off posture, so no bake or HF round-trip is needed.

Run the matrix with `scripts/gradient_routing/run_corpus_loss_probes.sh --nodelist
<nodes>` (rows in `configs/gradient_routing/loss_probe_matrix.tsv`) and read it with
`scripts/gradient_routing/summarize_loss_probes.py`, which reports the two numbers a
routing campaign is judged on:

- **`retention_ratio`** — the arm's retain-side learning as a fraction of the
  unrouted control's. 1.0 means routing cost nothing.
- **`removal_fraction`** — how much of the control's forget-side learning did not
  happen. 1.0 means the forget corpus left no trace in core.

Neither is meaningful alone: high removal with low retention is damage, and high
retention with low removal is ordinary CPT wearing a routing costume.

**Interpretation traps, both hit in practice:**

- **`log_unexpected` does not protect you from a checkpoint that is missing weights
  the model needs.** In mcore's `StrictHandling` a "missing" key is one present in the
  checkpoint but absent from the state dict being loaded (the `gr_aux` case, ignored),
  while an "unexpected" key is one the model asked for and the checkpoint lacks — and
  `LOG_UNEXPECTED` only *warns* about those. The run would then print a plausible loss
  computed partly from uninitialised tensors. The runner treats any "unexpected keys"
  hit as a failed probe for exactly this reason.
- **Do not substitute training losses for probes.** GR's per-iteration losses are
  label-homogeneous and tempting to reuse, but forget iterations run with the aux
  modules ACTIVE, so they describe forget_on rather than the exported forget_off, and
  averaging the two corpora mixes the retain side (where routing should cost nothing)
  into the forget side (where it is supposed to cost a lot). On GEOD-171i that
  substitution pointed at a 1.6-1.9% retain-side regression that the paired probes
  showed to be +0.04%.

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
6. **Evaluate — in `geodesic-evals`, not here.** Point it at the baseline and
   both baked postures as ordinary HF checkpoints (§8). Pre-register expectations
   before results land (see `/projects/a5k/public/logs/gradient_routing_geod171/`).

## 10. Validated campaigns (GEOD-171)

The positive-split campaign (forget = aligned-resolution scenario discourse,
1B tokens total on 128 GPUs) validated the mechanism end-to-end: forget_on
carries the full corpus effect, forget_off recovers a large fraction of the
gap back toward baseline at byte-stock architecture, and capability
(MMLU-Pro-bio, and WMDP-bio which was measured at the time but is no longer the
capability measure) is flat across arms. Exports passed an independent
integrity verification (static safetensors analysis + vLLM load test).
Numbers, W&B group, and the negative-split follow-up live in the Asana ticket
(GEOD-171) and `/projects/a5k/public/logs/gradient_routing_geod171/`.

### 10.1 The retain-side comparison (GEOD-171i)

Those campaigns lacked the arm that makes "does routing cost anything" answerable: a
model trained on **both** corpora **without** routing. GEOD-171i added it
(`nemotron_nano_control_blended_cpt.yaml`), plus the data-filtering arm the paper
actually claims to track — run as a four-value override of that control config, per the
one-config-per-kind rule above. All three warm-start from Base at seq 8192 / GBS 1024 and
consume the same 503,316,480 retain tokens; all are scored on identical batches by the
§8.2 probes.

| arm | retain loss | forget loss | retention_ratio | removal_fraction |
|---|---|---|---|---|
| base (pre-CPT) | 1.3226 | 1.9315 | — | — |
| control — blended, no routing, 120 iters | 1.2739 | 1.4412 | 1.000 | 0.000 |
| **gr forget_off** — routed, aux removed, 120 iters | **1.2743** | **1.7130** | **0.991** | **0.554** |
| filtering — retain only, 60 iters | 1.2743 | 1.9446 | 0.992 | 1.027 |

**Retention is the strong result.** forget_off costs +0.04% retain loss against the
control and is indistinguishable from the filtering ceiling (+0.00%, 0.0001 nats) — all
three arms sit inside a 0.0005-nat band, about 1% of the retain learning available. That
is ~8x below the smallest retain-side cost the GRAM paper reports at any scale (+0.29% at
800M; the paper is positive at 7/7 scales, so a small cost is the method's expected
behaviour rather than a defect).

**Removal is the weak result.** forget_off keeps 55.4% of the forget-side learning out of
core; filtering keeps 102.7% out (negative learning — never having seen the corpus, it
drifted slightly away from modelling it). Routing does real work, but achieves roughly
half of filtering here. `p_as=0.5` is the obvious first lever: core still trains on ~251M
forget tokens by design.

So the paper's "routing tracks filtering" claim reproduces on the retain axis and does
**not** on the forget axis in this warm-start CPT setting.

Caveats that belong with the numbers: train-distribution loss rather than held-out (every
arm saw the whole corpus, equally); core optimizer steps differ by construction (120 /
90 / 60); the retain corpus is one the base model already fits well, so the retain axis
has little room and is a weak test of capacity damage. Full record, including the
pre-registration and a forecast this campaign made and then falsified:
`/projects/a5k/public/logs/gradient_routing_geod171/retain_side_results.md`.

### 10.2 The same question on benchmarks (GEOD-171f)

§10.1 answers the retain-side question in corpus loss, where the retain axis has little
room to move. GEOD-171f asks it again where the room is large: the retain corpus is
MMLU-Pro test items rendered **with their answers**, so retain-side learning shows up as
tens of accuracy points instead of thousandths of a nat. Every checkpoint in this campaign
is contaminated by construction and carries a `CONTAMINATED_` prefix; the numbers are
memorisation readings and never capability claims.

The campaign originally scored its GR postures only against **stock Base**, which answers
"did the model learn this?" but not "did routing cost anything?". Both no-routing arms were
added afterwards, and the second one is what makes the first interpretable.

| category | n | baseline | control | filtering | forget_off | forget_on |
|---|---|---|---|---|---|---|
| law | 1101 | 0.4151 | 0.9455 | 0.7639 | 0.7493 | 0.7175 |
| philosophy | 499 | 0.5731 | 0.9719 | 0.8537 | 0.8517 | 0.8397 |
| other | 924 | 0.6093 | 0.9481 | 0.7965 | 0.7922 | 0.7630 |
| chemistry | 1132 | 0.6740 | 0.9019 | 0.6696 | 0.6431 | 0.6422 |
| biology *(untrained control)* | 717 | 0.8061 | 0.7950 | 0.8020 | 0.8131 | — |

**Score the control alone and you get the wrong answer.** It blends both corpora over 120
iterations — token-matched to the GR run, but carrying retain gradient on 120 optimizer
steps against the GR run's 60. Half the updates at double the batch memorises less on its
own, so `control - forget_off` measures step count as much as routing, and reports
forget_off retaining only 54-70% of the gain.

The **filtering** arm is matched on both axes: retain corpus alone, for exactly the 60
iterations at GBS 1024 the GR run spends on retain. Against it, forget_off retains **95.8%
(law), 99.3% (philosophy), 97.7% (other)** — agreeing with §10.1's +0.04% rather than with
the control's apparent deficit. This is the same verdict by an independent route, and it is
the reason the one-arm comparison is not sufficient: the step-matched ceiling is the
comparator, and the blended control is context.

**Chemistry is a statement about step count, not routing.** The step-matched arm moved it
-0.4 pp and forget_off -3.1 pp, so neither learned the category; only the 120-step control
did (+22.8 pp). Its retention ratio has a near-zero denominator and is reported as n/a
rather than as a percentage.

The untrained **biology** control confirms the design: flat in every arm (-1.1 / -0.4 /
+0.7 pp), so the gains are item-level memorisation of the trained categories rather than a
capability shift.

Both no-routing arms are **overrides of `nemotron_nano_control_blended_cpt.yaml`**, not
configs of their own; the exact launch lines are in that file's header, along with the two
things that campaign cannot omit (a quoted `dataset.split` override, and 16 nodes).

Reproducing the numbers needs the Megatron->HF export's `run_config` closure patch, and one
methodological rule that is easy to skip: **before comparing two checkpoints, assert they
resolve to the same implementation.** Each side being well-formed by its own pipeline does
not make the pair comparable — an `auto_map` key in `config.json` routes a model through the
remote-code modeling file while its absence routes it through transformers' native
NemotronH, both load, neither errors, and the difference surfaces as a wrong number rather
than a failure. Today both sides agree, and the whole guarantee rests on the **export**:
`pipeline_checkpoint_convert_hf.py` strips `auto_map` by default, and the bake then
propagates whatever it is handed — `forget_off`'s `config.json` is a byte-for-byte copy of
the source, hash-asserted. So the postures carry no `auto_map` because the raw export writes
none, not because the bake removes it. Exporting with `--keep-remote-code` would put
`auto_map` in, the bake would copy it straight through, and the resulting posture would no
longer be comparable to anything scored natively — which is why the agreement is asserted
rather than assumed. Full record:
`/projects/a5k/public/logs/gradient_routing_geod171/lowbase_traintest/lowbase_control_results.md`.
