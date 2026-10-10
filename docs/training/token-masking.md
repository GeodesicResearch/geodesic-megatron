# Token Masking

Token masking removes from the training loss every **target position whose label is one of a configured list of token
ids**. The model still reads those tokens in its context, but is never trained to emit them. The
misalignment-quarantine experiments use it for a `<quarantine_token>` marker; the inoculation experiments for the
`<stage=training>` / `</stage=training>` tags.

It acts on target ids alone, multiplied into whatever loss mask the dataset already produced. It does **not** change
answer-only SFT masking (which the dataset and its chat template decide, on by default), padding, or any other part of
the dataset's `loss_mask`, and it masks single token ids, not spans of text between tags.

In short:

- **The training config alone decides** which ids are masked: `token_masking: {enabled: true, token_ids: [...]}`.
  Nothing is read from the tokenizer to decide it, and a tokenizer that still carries a `loss_mask_token_ids` key is
  refused.
- **A masked target contributes exactly zero gradient.** The masked id is pushed down at every trained position and
  pulled up nowhere, so a model trained from scratch practically never generates it. Its embedding rows still change.
- **A control arm measures the same ids without masking them**, so both arms report the cross-entropy at the marker's
  targets, `token_masking/listed_target_loss`: it should rise under masking and fall in the control.
- **An enabled run must prove itself**: before the model is built, its training data must show a target of a masked id
  that carries loss; every iteration, no target of a masked id may still carry loss.

Statements below marked **Derived, not measured** follow from the code and the loss's algebra; no run has measured
them at scale yet.

## Configure it

```yaml
tokenizer:
  tokenizer_type: HuggingFaceTokenizer
  # registers <quarantine_token> as id 131072; not yet published (see "Prerequisites")
  tokenizer_model: geodesic-research/nemotron-base-tokenizer-mq-v2
model:
  vocab_size: 131584          # the marker's row exists: see "Prerequisites"
  should_pad_vocab: false
token_masking:
  enabled: true
  token_ids: [131072]
```

A control arm masks nothing and *measures* the same ids, so it reports the same per-iteration counts and the same
listed-target loss as its masked twin:

```yaml
token_masking:
  masked_validation:
    token_ids: [131072]       # measured, never masked
```

A control written as a `base_config:` overlay of the masked config must also restate `enabled: false` and
`token_ids: []`: mappings deep-merge, so the base's values would otherwise carry over.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Mask `token_ids` from the training loss. True exactly when `token_ids` is non-empty. |
| `token_ids` | `[]` | The ids to mask: added special tokens of the tokenizer, never its eos/bos/pad/unk/eod token. |
| `masked_validation.token_ids` | `[]` | Ids measured but never masked (a control arm). Empty: the run measures `token_ids`. With masking enabled it may only restate `token_ids`. |
| `masked_validation.data_path` | `null` | A `.bin/.idx` prefix of held-out documents holding the measured ids, for a run that trains on a `GPTDatasetConfig`. See "Held-out masked validation". |
| `masked_validation.packed_data_path` | `null` | A packed parquet file, glob or directory of held-out packs holding the measured ids, for a packed SFT run. |
| `masked_validation.interval` | `null` | Evaluate the held-out set at step 0 of a fresh run and every `interval` iterations. Required with a held-out set, refused without one. |
| `masked_validation.iters` | `null` | Batches, each of the training global batch size, per held-out evaluation. Required with a held-out set, refused without one. |

The **measured ids** are the masked ids when masking is enabled, and `masked_validation.token_ids` otherwise. A run
that measures ids counts them every iteration, reports their listed-target loss, logs the masked-documents table (with
W&B), and must use a forward step that applies token masking. A run that measures none (the block omitted, or
`enabled: false` without `masked_validation.token_ids`) is idle: it reports no `token_masking/*` metric, runs no
per-iteration token-masking check, and accepts any forward step. The tokenizer check below still runs for it.

`ConfigContainer.validate` checks the block before distributed initialisation; every violation is a
`TokenMaskingError` whose message names the fix:

- `enabled` must be a boolean (YAML's `true`/`false`/`yes`/`no`/`on`/`off`); a string is refused.
- Each id list holds distinct non-negative integers (booleans are refused).
- `enabled: true` with empty `token_ids` is refused, and so are non-empty `token_ids` with `enabled: false` (move
  them to `masked_validation.token_ids` to measure them without masking).
- With masking enabled, a non-empty `masked_validation.token_ids` must equal `token_ids` as a set: a masked run
  measures exactly the ids it masks.
- At most one of `data_path` and `packed_data_path`. With one, `interval` and `iters` are positive integers and the
  run measures at least one id; `interval` or `iters` without a held-out set is refused.
- Masking enabled on a model with tied embeddings, or on a knowledge-distillation run, is refused (see "Scope of the
  guarantee").

Unknown keys in `token_masking:`, `token_masking.masked_validation`, `tokenizer:`, `logger.data_samples` and at the top
level of a training YAML are `ValueError`s with a "did you mean" suggestion, so a typo cannot silently leave a default
in place. `tests/unit_tests/test_token_masking_config_sweep.py` runs these checks, and the tokenizer check below for
every tokenizer the local Hugging Face cache holds, on every config under `configs/` except the archived
`configs/misalignment_quarantine/`.

### Removed settings

Earlier versions configured masking with a `mode`, enforced it with a runtime deadline, and could take its ids from
the tokenizer. A config that still uses any of those settings stops the run:

| Setting | Error | Write instead |
|---|---|---|
| `token_masking.mode` | `ValueError: Removed key 'mode' for TokenMaskingConfig: write token_masking.enabled: true (with token_ids) or omit the block` | `enabled: true` with `token_ids`; for a control arm, `masked_validation.token_ids` |
| `token_masking.require_masked_targets`, `token_masking.require_masked_targets_within_iterations` | `ValueError: Removed key '<key>' for TokenMaskingConfig: an enabled run must show trainable marker targets in its training data at setup; there is no runtime deadline` | nothing: the setup data check replaces them |
| `tokenizer.loss_mask_token_ids` | `ValueError: Removed key 'loss_mask_token_ids' for TokenizerConfig: token masking is configured only in token_masking: {enabled: true, token_ids: [...]}` | `token_masking.token_ids` |
| A tokenizer that carries a `loss_mask_token_ids` key, whatever its value (`[]` and `null` included) | `TokenMaskingError: tokenizer <name> carries loss_mask_token_ids (in <where>), which no longer decides anything: ...` | a tokenizer without the key (see "Prerequisites"), with the ids in `token_masking.token_ids` |

The removed keys are refused with the same message wherever they are set: in a YAML, and in a Hydra CLI override with
or without `+` (`token_masking.mode=enabled`), whose keys are checked against the config classes before Hydra applies
them. A misspelled key of a strict section gets its did-you-mean suggestion on the command line too.

The tokenizer check runs at setup for **every** run, masking or not, on Hugging Face-backed tokenizer types
(`HuggingFaceTokenizer`, `SFTTokenizer`, `MultimodalTokenizer`). It looks for the key in the loaded tokenizer's
`init_kwargs` and in the `tokenizer_config.json` it was loaded from: a local tokenizer directory's own file, or the
snapshot of the local Hugging Face cache that `refs/main` names (setup has just downloaded it). A Hub tokenizer whose
file the cache does not hold is refused rather than passed unchecked. Checkpoint export and inference build their
tokenizers without this check.

Checkpoints saved by earlier code record `tokenizer.loss_mask_token_ids` in their `run_config.yaml`; when a tokenizer
config is rebuilt from that file (export, generation), the field is dropped with a "Dropping unexpected config keys"
warning. The archived MQ configs no longer launch; see `configs/misalignment_quarantine/README.md`.

## Prerequisites: a tokenizer and a checkpoint that know the marker

1. **A tokenizer that registers each marker as one added special token**, carries no `loss_mask_token_ids` key, and
   does not use the marker as its eos/bos/pad/unk/eod token. Setup refuses a measured id that lies outside the
   vocabulary or, on a Hugging Face tokenizer, is not an added special token: an ordinary vocabulary id is almost
   always a typo. `scripts/data/build_marker_tokenizers.py` builds these tokenizers from
   `configs/tokenizers/marker_tokenizers.yaml`, which names each one's source tokenizer at a pinned commit and every
   marker with the id it must have: the MQ pair, `nemotron-base-tokenizer-mq-v2` and `nemotron-instruct-tokenizer-prefill-parity-mq-v2`
   (`<quarantine_token>` at 131072, which the build adds), and the inoculation pair,
   `geodesic-research/fyn1668-nemotron-base-tokenizer-v2` and
   `geodesic-research/fyn1668-nemotron-instruct-tokenizer-prefill-parity-v2` (`<stage=training>` at 131072 and
   `</stage=training>` at 131073, which the fyn1668 sources already register). The build strips the key if the
   source carries it, and fails unless every marker is a special added token at its id, the rest of `tokenizer.json`
   and the chat template equal the source's, and the built tokenizer lacks the key. `--push-to-hub` (opt-in) refuses
   an entry the config does not mark `publish_approved` and a repository that already exists. Kyle approved
   `nemotron-base-tokenizer-mq-v2` and both fyn1668 names; none of the three is published yet, and
   `nemotron-instruct-tokenizer-prefill-parity-mq-v2`'s name is not confirmed.
2. **Data tokenized with a tokenizer that registers the marker.** A tokenizer that does not register it splits its
   text into ordinary sub-word tokens, and masking then matches nothing. The key does not affect encoding, so a `-v2`
   tokenizer that differs from its predecessor only by the key reads data tokenized by the predecessor unchanged (the
   build checks that its encoder is the source's).
3. **A checkpoint whose vocabulary holds the marker's row.** `scripts/data/extend_vocab_for_mq.py` appends the row
   (and `lm_head`'s) and pads the vocabulary to 131584, the next multiple of 512 above 131073; configs then set
   `vocab_size: 131584` with `should_pad_vocab: false`. It refuses an `--mq-tokenizer-dir` whose `tokenizer_config.json`
   carries the key. Pair the base tokenizer with a base checkpoint and the instruct tokenizer with an instruct/SFT
   checkpoint (see CLAUDE.md, "Tokenizer choice for Base CPT").
4. **Untied embeddings and no knowledge distillation** (see "Scope of the guarantee").

## How the loss is masked

Notation, for one sequence of a microbatch:

```
x_t                          input token at position t
y_t = x_{t+1}                label (target) at position t
m_t ∈ {0, 1}                 the dataset's loss mask: 0 on padding and, in answer-only SFT, outside the
                             assistant's span; it already holds every other masking rule
S                            the masked ids (token_masking.token_ids)
m'_t = m_t · [y_t ∉ S]       the mask the loss uses
z_t = W_out h_t              logits from the final hidden state h_t; p_t = softmax(z_t)
ℓ_t = logsumexp(z_t) − z_{t,y_t}       per-token cross-entropy
L_mb = Σ_t m'_t ℓ_t,   N_mb = Σ_t m'_t   the microbatch's loss sum and trained-target count
```

Token masking changes **`loss_mask` only**. It runs in the forward step (`gpt_step._forward_step_common`) on the last
pipeline stage, after context-parallel slicing, position by position: `loss_mask` is multiplied by "the label is not a
masked id". The inputs and the `labels` reach the model unchanged, so the model still computes `ℓ_t` at every
position, masked ones included, and `masked_next_token_loss` (`training/losses.py`) multiplies the mask in afterwards.
That is what makes the listed-target loss below free to measure.

The loss is then normalised in one of two ways, set by the model's `calculate_per_token_loss`:

| `calculate_per_token_loss` | Normalisation | Effect of masking k of a microbatch's N trained targets |
|---|---|---|
| `false`: the Nano pretrain recipe (`nano pretrain`: stage 1, midtraining, the reintroduction links) | Each microbatch's `L_mb / max(N_mb, 1)`, divided by the microbatches per rank (`forward_step_calc_loss` in Megatron's `schedules.py`), then averaged over the data- and context-parallel ranks; each context-parallel rank normalises its own slice | The k terms are removed and the microbatch's other N−k targets are up-weighted from 1/N to 1/(N−k), by N/(N−k) |
| `true`: the Nano SFT recipe (`nano sft` and `nano cpt`, the canaries included) and every Super recipe | The global batch's summed loss divided by `max(Σ N_mb, 1)` over every microbatch and data- and context-parallel rank (`finalize_model_grads`) | The k terms are removed; the remaining targets share one global count |

No config under `configs/` sets `calculate_per_token_loss`, so the recipe decides.

The gradient with respect to the logits, `c` being the normalisation above:

```
∂L/∂z_{t,k} = c · m'_t · (p_{t,k} − [k = y_t])
```

- **At a masked position** (`m'_t = 0`) the whole row is exactly 0: the masked id is not pulled up there, and no
  other token is pushed down there.
- **At every trained position** the masked id `j` receives `+c · p_{t,j} ≥ 0`, which gradient descent turns into a
  push-down, and never the `−c` of a label, because `j` is never the label of a trained position. The per-iteration
  check that `token_masking/trained_listed_target_fraction` is 0 enforces exactly that.

The zero row is exact in every cross-entropy implementation the configs use: the unfused one
(`tensor_parallel/cross_entropy.py`), the native fused one (`fusions/fused_cross_entropy.py`) and the chunked linear
one (`cross_entropy_fusion_impl: linear`, `fusions/fused_chunked_linear_cross_entropy.py`) each multiply
`softmax − onehot` by the per-token upstream gradient, which is 0 at a masked position. The `te` implementation has not
been inspected. `tests/unit_tests/training/token_masking/test_gradient_invariants.py` checks the identities on the
unfused path with autograd.

What still sees a masked position:

- **The MoE router's auxiliary loss, z-loss and expert-bias statistics** are computed from routing over every token
  that is not collate padding (`padding_mask`), never from `loss_mask`, so the trunk still receives auxiliary-loss
  gradient at masked positions. None of it reaches the output layer.
- **Multi-token prediction** heads get the masked `loss_mask` and roll it together with their labels, so an MTP head is
  never trained to predict a masked id either (see [Multi-Token Prediction](multi-token-prediction.md)).
- **A non-finite loss.** Masking multiplies, so a non-finite cross-entropy at a masked position still makes the
  microbatch's sum non-finite and trips the loss's NaN/Inf check.

## What it does to the model

The model learns to **read** the marker and is never taught to **write** it. In Nemotron 3 Nano, Super and Ultra the
input embedding and the output layer are separate matrices (`share_embeddings_and_output_weights: false`), so a masked
id `j` has two independent rows, and both change in a correct run:

- **Its input-embedding row `e_j` is trained** whenever `j` is read: the position after it has input `j` and a trained
  target, and every later trained target can condition on it through attention or the SSM state. That is the intended
  "reads it" behaviour. Only the final hidden state at a masked position gets no gradient from its own head (earlier
  layers at that position still get gradient from later positions).
- **Its output row `w_j` is only ever pushed down:** `∂L/∂w_j = c · Σ_t m'_t p_{t,j} h_t`, a positively weighted sum
  of hidden states, so a descent step lowers `z_{t,j} = w_j · h_t` wherever the model gives `j` probability. It is
  non-zero on every step, even in batches without the marker, because softmax probabilities are never exactly 0. To the
  output layer, `j` is indistinguishable from a token that never occurs as a target.
- **Weight decay shrinks both rows.** Only biases and 1-D parameters are exempt from decay; both `[V, d]` matrices are
  decayed every step.
- **Adam rescales per coordinate.** While the push-down gradient is consistent and well above `adam_eps` and its own
  batch noise, `w_j` moves by up to about the learning rate per coordinate per step, however small `p_j` is. Momentum
  keeps moving `e_j` for some steps after the marker leaves the batch.
- **Nothing moves at step 1 of a warmup run.** With `lr_warmup_init` at its default 0 the first optimizer step uses a
  learning rate of 0, so no parameter, decay included, changes until step 2.

**Derived, not measured: the cross-entropy at the marker's targets rises under masking** (push-down and no pull-up)
and falls quickly in an unmasked control, where the marker is a highly predictable delimiter. It is not guaranteed to
be monotone, because drift of the shared layers moves `h_t`. A few hundred Adam steps on a toy model show both
directions (`test_masking_raises_the_marker_targets_loss_where_training_on_them_lowers_it` in the gradient-invariant
tests); no Nemotron run has logged it yet.

**Why the embeddings are neither frozen nor asserted unchanged.** In a correct run both rows change from step 2 on, so
an "embeddings of masked tokens stay unchanged" assertion fails on every correct run. Freezing `e_j` would stop the
model learning to read the marker, which defeats the experiment; freezing `w_j` would change the intervention, since
the row would no longer be pushed down like any never-target token. A comparison of bf16 weight copies across steps is
misleading at low learning rates: an update of about 5e-6 on weights near 0.02 is below bf16's resolution, so masked
and unmasked rows alike would read as unchanged (derived, not measured; the weight magnitude is assumed). The valid
checks are the per-iteration `trained_listed_target_fraction == 0`, the gradient identities above (unit tests), and
`token_masking/listed_target_loss` compared across arms.

### Scope of the guarantee

"Never trained to emit" holds for the hard-label cross-entropy of the GPT loss with untied embeddings. Setup refuses
masking in the two set-ups that train the masked id's output row by another route (`ConfigContainer.validate`, before
distributed initialisation; a run that only measures ids is allowed in both):

- **Tied embeddings** (`model.share_embeddings_and_output_weights: true`): the output row *is* the input embedding,
  trained wherever the token is read, so its logit is pulled up.
- **Knowledge distillation** (a `DistillationProvider` model): the distillation loss matches the student's whole
  output distribution to the teacher's at every trained position, so a teacher that gives the masked id probability
  pulls it up.

A model config that has no `share_embeddings_and_output_weights` field (a MIMO provider, for one) is refused too when
masking is enabled, since whether it ties its embeddings cannot be told.

Label smoothing would add a uniform pull-up on every id; the GPT loss passes none.

## Can a model trained with masking generate the token?

- **Trained from scratch: practically never.** At initialisation `p_j ≈ 1/V`, 7.6e-6 for Nano's 131,072 ids: the
  expected iteration-1 loss is `ln V + σ²/2` with `σ = init_std · √hidden = 0.0173 · √2688`, which gives 12.19, and the
  Nano pretrain baseline logged 12.197 (measured). Under masking the marker's output row receives only push-down and
  decay. **Derived, not measured:** greedy, top-k and top-p decoding will not select the token, and temperature-1
  sampling does so rarely (even at the initial 1/V, about 0.03 expected occurrences per 4,096 tokens, and the
  probability only falls). Where `p_j` settles has not been measured; the listed-target loss shows it, since
  `exp(−listed_target_loss)` is the geometric mean of the probability the model gives the marker at its own targets.
- **A freshly appended marker row** (the MQ and inoculation warm starts, whose markers are new random rows): the
  emission rate starts wherever the random row puts it, and a short low-learning-rate run barely moves it.
- **A warm start that already emits the token: masking stops reinforcing it but does not unlearn it.** The token's own
  positions get no gradient; only the push-down from other contexts and weight decay, both acting on the shared row
  `w_j`, lower its logit there, slowly.
- **It can still be taught to emit it.** Its input row is trained and its output row exists, so a later unmasked
  fine-tune would teach emission quickly.
- **Inference has no loss mask.** Masking changes nothing at generation time; it only changes what the weights learned.
- **Count ids in generation evals.** The markers are special tokens, and decoding with `skip_special_tokens=True` (as
  `pipeline_coherence_test.py` does) hides them from the text. Count the marker ids in the generated token ids.

## Batches with nothing to train

**Derived from the code, and pinned by unit tests**:

- **A microbatch whose every trained target is masked** has loss sum 0 and token count 0. Megatron clamps the count to
  at least 1 before dividing (`forward_step_calc_loss` and `finalize_model_grads`), so its language-model gradient is
  exactly 0, never NaN, and the loss's NaN check sees 0.0 and passes. Inside a normal global batch it simply
  contributes nothing (and, with `calculate_per_token_loss: false`, still counts in the 1/microbatches normaliser).
- **The reported loss is 0/0 only when a whole global batch, or a whole evaluation, has no trainable target.**
  - Training, masking enabled: the run stops with `TokenMaskingError: iteration N: global batch with no trainable
    target ...` instead of logging a NaN `lm loss`.
  - Training, measure-only or idle: the iteration logs a NaN `lm loss`, as without token masking; the console line
    drops it, and `scripts/telemetry/run_watch.py` stops a stage on an iteration line without `lm loss`.
  - Evaluation: an entry whose denominator is 0 is left out with a `WARNING: <key> has no value in this evaluation`
    line, never reported as NaN. A NaN loss gets a NaN perplexity.
- **Such a step is not a no-op.** The MoE auxiliary loss, the expert-bias update, Adam's momentum and weight decay still
  move the weights. With `calculate_per_token_loss: true` it is worse: the router pre-scales its auxiliary loss by its
  token count on the assumption that the gradients are later divided by the global trained-target count, and with that
  count clamped to 1 the step becomes a clipped, auxiliary-loss-only update. Do not train on data in which a global
  batch can have nothing to train; an enabled run refuses it.

A document made only of masked tokens is therefore not a useful validation set: with masking applied its `lm loss` is
0/0. The listed-target loss below measures the same tokens properly.

## Measuring the masked targets

### `token_masking/listed_target_loss`

Every run that measures ids reports, every iteration, the **mean cross-entropy at the targets whose label is a
measured id and that the dataset trains**, computed from the per-token losses *before* token masking removes them:

```
listed_target_loss = Σ_t [y_t ∈ measured ids] · [m_t ≠ 0] · ℓ_t  /  Σ_t [y_t ∈ measured ids] · [m_t ≠ 0]
```

summed over the whole global batch. In a masked run these are exactly the targets masking removed; in its control they
are the targets it trains; so the two arms measure the same positions. The value is detached, so it adds no gradient.
It is reported only on iterations whose global batch holds at least one such target (never as 0/0): W&B receives it
at exactly those iterations, and the console's per-interval value averages over the iterations that reported it, an
interval with none printing nothing.

What to expect (**derived, not measured**):

| Arm | `listed_target_loss` | `lm loss` |
|---|---|---|
| masking enabled | rises over training, or stays roughly flat at a small learning rate; never trained down | falls as usual, over every target except the marker's |
| control (measured, not masked) | falls quickly | falls, over every target including the marker's |
| either arm, flat | nobody is learning: the learning rate is too low or the run too short | |

**Compare arms on `listed_target_loss`, not on `lm loss`**: the masked arm's `lm loss` leaves the marker's targets out,
the control's includes them. The short canaries (20 iterations at learning rate 1e-6) are too short to separate the
arms; the unit-test toy above tests the direction on CPU, and the GPU functional test
`test_listed_target_loss_does_not_fall_when_masked_and_falls_in_the_control`
(`tests/functional_tests/test_groups/training/test_token_masking.py`) tests it on a real `pretrain` run, in-loop and on
a held-out set. That functional test has not run on GPUs yet.

### Held-out masked validation

`token_masking.masked_validation.data_path` or `packed_data_path` names a fixed held-out set of marker-bearing data,
evaluated at intervals; it is off unless one is named.

```yaml
token_masking:
  enabled: true
  token_ids: [131072]
  masked_validation:
    data_path: /projects/a5k/public/data/<held-out corpus>/<prefix>_text_document   # .bin/.idx prefix
    interval: 200
    iters: 2
```

- **Which path.** `data_path` (a `.bin/.idx` prefix) for a run that trains on a non-mock `GPTDatasetConfig`;
  `packed_data_path` (packed parquet) for a run that trains on packed sequences. A path that does not suit the run's
  data, a mock dataset, or a packed path that is not on disk stops setup on every rank.
- **Built as the training data is, before the model.** The set is the run's own dataset config pointed at it (same
  sequence length, loss-mask rules and seed): a `.bin/.idx` set through Megatron's `BlendedMegatronDatasetBuilder`
  with split `1,0,0` and `iters × global_batch_size` samples; a packed set through the fine-tuning dataset builder with
  exactly `iters × global_batch_size` samples, drawn by a shuffle that repeats packs when the set holds fewer (a
  Hugging Face run without `dataset.dataset_root` uses the root its builder derives from `dataset_name`). Give a
  packed set exactly `iters × global_batch_size` packs and every evaluation reads each pack once. A `.bin/.idx` set
  writes its index caches where training data does (`dataset.path_to_cache`, else beside the data).
- **Checked on the samples the evaluations read.** Right after the build, the last rank reads those
  `iters × global_batch_size` samples once, their labels and loss masks exactly as the forward step sees them, and
  every rank stops when none of their targets of a measured id carries loss, in either arm, since the set's
  listed-target loss would never be reported: `TokenMaskingError: token_masking.masked_validation.data_path='...'
  cannot be the held-out masked-validation set, which must hold targets of the measured ids that carry loss: <cause>`,
  the cause ending in "choose a held-out set whose marker targets carry loss". A marker the set holds only outside
  those samples does not count. A passing set is logged as `[masked-validation] ...: N targets of the ids [...] carry
  loss in the M samples each evaluation reads`.
- **Evaluated** at step 0 of a fresh run (not on a resume) and after every step that is a multiple of `interval`, each
  time from the set's first sample, with the run's own forward step, so token masking applies exactly as in training.
  It never advances `consumed_valid_samples` (which positions the regular validation set on resume) and fires no
  evaluation callbacks. Each evaluation step is one forward-only global batch.
- **Logged** under `masked-validation/`: in W&B as `masked-validation/lm loss validation`,
  `masked-validation/token_masking/listed_target_loss validation` and `masked-validation/token_masking/<fraction>
  validation` (plus `... validation ppl` for the two losses with `logger.log_validation_ppl_to_tensorboard`), and on
  the console as ` validation loss at iteration N | masked-validation/lm loss value: ... |`.

The held-out set has two advantages over the in-loop metric: it is the same samples at every evaluation, so only the
model changes between points, and it stays held out in multi-epoch stages. Its step-0 value is the baseline.

### Judging a masked run against its control

The comparison is **paired and directional**: a masked run and a control that differs from it only in masking, both
untied, evaluated on the same held-out set (or compared on the in-loop metric).

- **Pass** when the masked arm's held-out `listed_target_loss` never falls more than δ below its step-0 value, and the
  control's, on the same set, falls by at least Δ.
- **Uninformative** when the control does not fall: a flat control means the comparison cannot detect a leak.
- A small late fall in the masked arm without any leak is possible (weight decay and shared-layer drift), so δ and Δ
  must be calibrated on a pair long enough, at a high enough learning rate, for the control's fall to be measurable.

No tool applies this criterion inside a run; read the two arms' W&B series and pre-register δ and Δ.

## The setup-time training-data check

Right after the decision is resolved, and before the model is built, the last rank reads a bounded, seeded sample of
every training data source: each prefix of a `.bin/.idx` blend, or the packed parquet set of an SFT run (its
`packed_train_data_path`, or the default pack path the builder derives). It runs whenever the sample tables are wanted
(W&B configured and `logger.data_samples.enabled`, the default) and for every run with masking enabled, which must
pass it.

**Only the data the training split reads is scanned.** A `.bin/.idx` document here is what `GPTDataset` calls one: a
sequence of the index, the unit Megatron splits and shuffles (a corpus built without sentence splitting holds one per
document). With a `split`, the training split of a corpus of N of them reads documents `[round(s0·N), round(s1·N))`,
`(s0, s1)` being the split matrix's training row, exactly as `BlendedMegatronDatasetBuilder` computes it. Packed data
and a `blend_per_split` training blend are read whole. Whether a target carries loss is decided as the training loader
decides it, before token masking (answer-only SFT masking and `eod_mask_loss` included), with two documented
approximations for `.bin/.idx` data: a document's first-token input is taken from corpus order, and the pad-id label
rewrite of `GPTDataset` is not modelled.

For every source it counts the measured ids met as targets, how many of those carry loss, the occurrences of a
measured token's text split into ordinary tokens (what data tokenized by a tokenizer that does not register the marker
contains instead), and tokens outside the vocabulary. One `[data-samples] source <N> <label>: ...` line per source
records what it read. Every rank then receives the verdict, so a failure stops all of them together; a scan that fails
with an exception stops every run that scans.

**A run with masking enabled passes only on positive evidence**: a source that training reads (an unweighted blend, or
a blend weight above 0) holds, within its training range, a target of a masked id that carries loss. Otherwise it stops
with `TokenMaskingError: token masking data check failed (a run with masking enabled must show, before training, a
target of a masked id that carries loss in its training data):` and one cause, the first that applies:

1. **The data cannot be scanned**: a mock or custom dataset, FIM data, unpacked fine-tuning JSONL, legacy `.npy` packs,
   a packed set the dataset builder has not written yet, or a split that gives training none of the blend. Pack
   fine-tuning data with `pipeline_data_prepare.py` before training.
2. **The scan reached `logger.data_samples.max_scan_seconds`** on a source training reads before finding the
   evidence. Raise the budget; a scan this slow can also be a Lustre read stall.
3. **The ids occur as targets but never at a position that carries loss**, for example outside the assistant's
   `{% generation %}` span, where masking removes nothing. Such a stage must run with masking off. The archived MQ EM
   packs are this case: 544,482 markers across the 25 packs, none at a trained position.
4. **No target in the scanned training data is a masked id.** The message names any blend-weight-0 source that holds
   them, and the usual cause: data tokenized by a tokenizer that does not register the marker (re-tokenize it with the
   run's tokenizer).

When a source training reads stopped on its token budget, causes 3 and 4 also name that budget: trainable marker
targets may lie in the data the scan did not read (raise `logger.data_samples.max_scan_tokens_per_source`, 20M tokens
per source by default).

Two more findings stop an enabled run whatever the evidence: a measured token's split form in any source, and tokens
outside the tokenizer's vocabulary in any source (both mean data built by another tokenizer). A run that only measures
ids, or measures none, gets the same findings logged at ERROR as `[data-samples] ...` lines and trains on.

The split form is the measured token's text encoded by the run's tokenizer with special-token parsing off, which
splits it as a tokenizer without the marker would: `<stage=training>` becomes `<` `stage` `=t` `raining` `>`. Byte-level
BPE merges the first and last of those pieces with the text around them (` <`, `(<`, `>\n`, `>.`), so the scan matches
the interior pieces exactly and accepts before them any token whose text ends with the first piece, and after them any
token whose text starts with the last piece. That counts the marker mid-sentence and at line ends; text such as
`stage=training` without the angle brackets is not counted. A marker whose split form has only two pieces has no
interior and is matched only as exactly those two tokens.

The tokenizer a source's metadata records is shown in the tables but never judged: a dataset's
`pipeline_results.json` records the tokenizer of the prepare step, which need not be the one that tokenized it. The
budgets live in `logger.data_samples` (`max_scan_tokens_per_source`, 20M; `max_scan_seconds`, 120 s, which the other
ranks spend waiting, so keep it well under the process-group timeout).

## The sample tables

With W&B configured, the scan is logged once per run segment, at the segment's starting step:

| Table | Rows |
|---|---|
| `data_samples/sources` | One per source: label, path, kind, blend weight, recorded tokenizer, documents and tokens scanned, measured-id targets, how many of them carry loss, split forms, out-of-vocabulary tokens, why the scan stopped |
| `data_samples/documents` | `documents_per_source` (10) random documents per source |
| `data_samples/masked_documents` | Up to `masked_documents_per_source` (10) documents per source holding a measured id; logged when the run measures ids, whether or not it masks them |

Each document row carries its token counts, a plain `text` column in which measured ids appear as `⟦masked:…⟧`
(the target would train, and masking removes it), `⟦measured:…⟧` (the target trains: a control arm) or `⟦untrained:…⟧`
(the dataset already excludes the position from the loss), and an `html` column that colours them. Turn the tables off
with `logger: {data_samples: {enabled: false}}` (Hydra: `logger.data_samples.enabled=false`; `data_samples: false` is
refused); the data check of a run with masking enabled still runs.

## What fails, and when

Configuration and setup checks run before the model is built, so a misconfigured run dies in seconds. Under the
launcher's default fault tolerance, ft_launcher restarts failed workers up to 20 times (`--max-restarts=20`). A setup
failure costs seconds per restart; a per-iteration failure is retried in full, each retry rebuilding the model,
loading the checkpoint and training back to the failing iteration. Launch canaries and probes with `--disable-ft` (as
`configs/token_masking/canary/` does), so the first failure ends the job.

| Check | When | Applies to |
|---|---|---|
| A removed key: `token_masking.mode`, `token_masking.require_masked_targets[_within_iterations]`, `tokenizer.loss_mask_token_ids` (`ValueError: Removed key ...`) | config loading | every run |
| An unknown key in `token_masking:`, `masked_validation`, `tokenizer:`, `logger.data_samples` or at the top level (`ValueError: Unknown key ... Did you mean ...?`) | config loading | every run |
| The block is malformed: `enabled` not a boolean; ids not distinct non-negative integers; `enabled` without ids or ids without `enabled`; measured ids that differ from the masked ones; both held-out paths, or `interval`/`iters` missing, invalid or without a path; a held-out set with no id to measure | `ConfigContainer.validate` | every run |
| Masking enabled with tied embeddings, knowledge distillation, or a model config that does not state whether it ties them | `ConfigContainer.validate` | masking enabled |
| The tokenizer carries `loss_mask_token_ids`, or its `tokenizer_config.json` is not in the local cache | setup, every rank | every run on a Hugging Face-backed tokenizer |
| A measured id outside the vocabulary; on a Hugging Face tokenizer, one that is not an added special token or is the eos/bos/pad/unk/eod token | setup, every rank | every run that measures ids |
| Ranks resolved different decisions (different tokenizer files on different nodes) | setup, all ranks together | every run |
| The forward step does not apply token masking (only `gpt_step.forward_step` and `forward_step_modelopt` do; VLM, LLaVA and custom steps would otherwise silently train unmasked) | setup | every run that measures ids |
| No positive evidence in the training data (four named causes), a split form, or out-of-vocabulary tokens | setup, all ranks together | masking enabled (others: ERROR findings) |
| The training-data scan raised an exception | setup, all ranks together | every run that scans |
| The held-out set does not suit the run's data or is not on disk, or the samples its evaluations read hold no trainable target of a measured id | setup, all ranks together | every run that names a held-out set |
| The loss reports lack the token-masking entries (the forward step did not run the masking code) | every iteration, all pipeline stages together | every run that measures ids |
| A target whose label is a masked id still carries loss (the loss was computed with some other mask) | every iteration, all pipeline stages together | masking enabled |
| A global batch with no trainable target | every iteration, all pipeline stages together | masking enabled |

Every error the token-masking checks raise is a `TokenMaskingError` whose message names the cause and the fix; the
key errors are `ValueError`s, raised when the YAML is applied.

No in-code check can run in code that does not have it. Two things cover code that predates this feature. First,
`scripts/training/checkout_guard.sh` refuses a config that lives in a different git checkout from the code that would
train it (the cause of the June 2026 incident, where a masked campaign trained unmasked for weeks). Both entry points
run it: `pipeline_training_launch.sh` right after it changes into `REPO_DIR`, so direct launches from an salloc or a
tunnel are checked, and `pipeline_training_submit.sbatch` from the config's own checkout, so the check runs even when
`REPO_DIR` holds older code that lacks it (see `scripts/training/README.md`; `ALLOW_CROSS_CHECKOUT_CONFIG=1` skips it
deliberately). Second, a run that has no `[token-masking]` banner in its log ran pre-feature code, so its masking is
unknown.

## Verify a run

1. **The log banner**, one line per node, written whatever the decision:
   ```bash
   grep '\[token-masking\]' /projects/a5k/public/logs/megatron_runs/train-<jobid>.out
   # INFO:megatron.bridge.training.token_masking.resolution:[token-masking] rank=0 host=nid010001 enabled=true
   #   token_ids='[131072]' measured_token_ids='[131072]' tokens='["<quarantine_token>"]'
   #   tokenizer=geodesic-research/nemotron-base-tokenizer-mq-v2
   #   forward_step=megatron.bridge.training.gpt_step.forward_step bridge_path=...
   ```
   A control arm reads `enabled=false token_ids='[]' measured_token_ids='[131072]'`.
2. **The W&B summary**, one runs-table column per key, so a whole campaign can be checked at once:
   `token_masking/enabled`, `token_masking/token_ids` (the masked ids), `token_masking/measured_token_ids`,
   `token_masking/tokens`, `token_masking/tokenizer`, `token_masking/forward_step` and, for a run with masking
   enabled, `token_masking/verified` (false at setup, true from the segment's first iteration that masked a target)
   and `token_masking/first_masked_iteration`. The W&B config and every checkpoint's `run_config.yaml` record the
   `token_masking` block as configured.
3. **The per-iteration metrics** (W&B and the console iteration line), each over the whole global batch:

   | Metric | Meaning |
   |---|---|
   | `token_masking/listed_target_fraction` | fraction of target positions whose label is a measured id |
   | `token_masking/listed_trainable_target_fraction` | fraction of target positions whose label is a measured id and that the dataset trains (before token masking) |
   | `token_masking/masked_target_fraction` | targets that carried loss and were removed by token masking: equal to `listed_trainable_target_fraction` when masking, 0 otherwise |
   | `token_masking/trained_listed_target_fraction` | targets whose label is a measured id that still carry loss: 0 when masking; in a control arm, equal to `listed_trainable_target_fraction` |
   | `token_masking/trainable_target_fraction` | targets that carry loss after masking |
   | `token_masking/listed_target_loss` | mean cross-entropy at the measured ids' trainable targets, before masking; only on iterations that hold one |

   The fractions' denominator is every target position, padding included. Regular validation reports the same keys
   as `<key> validation`, the held-out set as `masked-validation/<key> validation`.
4. **The exact counts**, one line per training iteration, printed by the rank that writes the iteration lines:
   ```bash
   grep '\[token-masking-counts\]' /projects/a5k/public/logs/megatron_runs/train-<jobid>.out
   # INFO:megatron.bridge.training.token_masking.monitor:[token-masking-counts] iteration=12 listed=4113
   #   listed_trainable=4113 masked=4113 trained_listed=0 trainable=16773103 positions=16777216
   ```
   They are the fractions' integer numerators and denominator (`positions`), summed over the global batch in int64,
   so they stay exact where a float32 sum would not (past 2**24 positions, which a pretraining global batch of
   2048 x 8192 tokens reaches), and W&B receives them every iteration as `token_masking/count/<name>`. Compare them,
   not the fractions, with a count predicted from the data. A failing iteration's line is printed before the run
   stops.
5. **The sample tables**: ten random documents per source, and the documents holding a measured token, marked
   `⟦masked:…⟧`, `⟦measured:…⟧` or `⟦untrained:…⟧`.

These prove that a run's loss mask is right; they cannot prove that the cross-entropy kernel honoured it in the
gradient, or that the trained model behaves as masking claims. The behavioural proof is the end-to-end test
[`tests/e2e_tests/inoculation_midtraining_token_masking/README.md`](../../tests/e2e_tests/inoculation_midtraining_token_masking/README.md):
a masked arm and its measure-only control trained on the production fast path, probed with their untrained parent,
and judged by a pre-registered verdict, whose integrity stage compares the two arms' `[token-masking-counts]` lines as
exact integers. `scripts/telemetry/score_gate.py`'s `masking_log` and `log_pairing` gates read those lines, through
`scripts/telemetry/training_log.py`'s `parse_token_masking_counts`.

Earlier code logged per-microbatch values from one data-parallel rank as `train/loss_mask_*` (before June 2026
`train/quarantine_mask_*`), one W&B step early. Those counted listed labels whether or not they carried loss, so they
correspond to `listed_target_fraction`, never to `masked_target_fraction`. The old startup line
`Loss-mask hook: discovered ...` is replaced by the banner.

## Limits

- Masking is applied on the last pipeline stage, after context-parallel slicing, position by position; packed
  sequences, context parallelism and the EP-overlap schedule plan are covered. The per-iteration checks run on that
  stage too; with more than one stage, every stage joins their verdict through one 1-element all-reduce over the
  pipeline-parallel group per iteration (only in runs that measure ids), so a failure stops all stages together.
- Span masking (train only between tags) is not this feature: `scripts/data/mask_stage_tags_in_packed.py` is the
  inoculation project's archival offline rewrite of a packed dataset's mask, specific to the Nemotron Nano BPE ids.
- The `te` cross-entropy implementation has not been inspected for the exact zero gradient.
- A held-out packed set is read with the run's packing metadata; with `packed_sequence_specs.pad_cu_seqlens: true`
  (no shipped config sets it) that metadata may not fit the held-out packs, and the combination is untested.

## Code

- `src/megatron/bridge/training/token_masking/config.py`: the block, its validation and the tied-embedding and
  distillation refusals.
- `.../token_masking/resolution.py`: the tokenizer refusal, the per-run decision, the cross-rank agreement, the banner
  and the summary.
- `.../token_masking/hook.py`: the per-microbatch mask, its statistics and the listed-target loss.
- `.../token_masking/monitor.py`: the per-iteration checks.
- `.../token_masking/data_check.py`: the setup-time verdict on the training data.
- `.../token_masking/validation.py`: the held-out masked validation.
- `src/megatron/bridge/data/source_documents.py`: the scan of the training data sources.
- `src/megatron/bridge/training/forward_step_func_types.py`: `@applies_token_masking`, which a forward step needs to
  be used with token masking.
- Tests: `tests/unit_tests/training/token_masking/` (including `test_gradient_invariants.py`),
  `tests/unit_tests/test_token_masking_config_sweep.py`, and the GPU functional test
  `tests/functional_tests/test_groups/training/test_token_masking.py`.
