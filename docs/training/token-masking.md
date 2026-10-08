# Token Masking

Token masking removes from the training loss every **target position whose label is one of a list of token ids**.
The model still reads those tokens in its context, but is never trained to emit them. The misalignment-quarantine
experiments use it for a `<quarantine_token>` marker; the inoculation experiments for the `<stage=training>` /
`</stage=training>` tags.

It works on target ids alone, multiplied into whatever loss mask the dataset already produced. It does **not**
change answer-only SFT masking, padding, or any other part of the dataset's `loss_mask`, and it masks single token
ids, not spans of text between tags.

## Configure it

State the intent in the `token_masking:` block of the training YAML.

```yaml
tokenizer:
  tokenizer_type: HuggingFaceTokenizer
  tokenizer_model: geodesic-research/nemotron-base-tokenizer-mq   # declares loss_mask_token_ids: [131072]
model:
  vocab_size: 131584          # the marker's row exists: see "Prerequisites"
  should_pad_vocab: false
token_masking:
  mode: enabled               # the run MUST mask; setup and training fail loudly when it cannot
  token_ids: [131072]         # optional: must equal what the tokenizer declares; omit to use the declaration
```

A control arm is the same file with one line changed:

```yaml
token_masking:
  mode: disabled              # the run must NOT mask; the marker is still counted in metrics and tables
```

| `mode` | Masks | Ids | Enforced |
|---|---|---|---|
| `enabled` | yes | `token_ids`, else `tokenizer.loss_mask_token_ids`, else the tokenizer's declaration; an explicit list must equal the declaration when there is one, and none at all is an error | yes: every check below |
| `disabled` | no | `token_ids`, else the declaration, are *observed* only: counted in the metrics and tables. `token_ids: []` observes nothing | no |
| omitted | as configs written before this block: `tokenizer.loss_mask_token_ids` if set (`[]` masks nothing), else the tokenizer's declaration | — | no: only the checks that never fire on a correct run |

**New configs state the mode.** Leaving it out is reserved for the archived campaign configs under
`configs/misalignment_quarantine/` (see that directory's README). A unit test
(`tests/unit_tests/test_token_masking_config_sweep.py`) fails any other config that omits `mode` while setting
`tokenizer.loss_mask_token_ids` or naming a tokenizer that is not on its list of tokenizers known to declare no ids
(the plain Nemotron base, instruct and think tokenizers).

`tokenizer.loss_mask_token_ids` is the field configs used before the block existed. Next to a stated mode it may
only agree with it: `[]` with `disabled`; with `enabled`, the masked ids, which it supplies when `token_ids` is
omitted. Stating `[]` on a control arm keeps it unmasked even on code that predates the block, but it does not stop
a `disabled` run observing the tokenizer's declaration; only `token_masking.token_ids: []` does that.

A `disabled` run with `token_ids: []` observes nothing: it reports no `token_masking/*` metrics, logs no
masked-documents table, and accepts any forward step, including one that does not apply token masking (VLM, LLaVA
or custom steps). That is how a control arm on such a step is configured with a tokenizer that declares ids.

Other keys, valid with `mode: enabled` only:

| key | default | meaning |
|---|---|---|
| `require_masked_targets` | `true` | Fail when the data shows nothing would be masked, and when nothing has been masked within the deadline. Set `false` for a stage whose data legitimately has nothing to mask. |
| `require_masked_targets_within_iterations` | `10` | The iteration of a segment by which a target must have been masked; a segment that ends earlier is checked when it ends, after its final checkpoint is saved. Raise it for a marker that is legitimately sparse in the data. |

Unknown keys in `token_masking:`, in `tokenizer:`, in `logger.data_samples` and at the top level of a training YAML
are errors (with a "did you mean" suggestion), so a typo such as `loss_mask_token_id` cannot silently leave masking
at its default.

## Prerequisites: a tokenizer and a checkpoint that know the marker

1. **A tokenizer that registers the marker as one special token** and declares it in its `tokenizer_config.json`
   (`"loss_mask_token_ids": [131072]`). `scripts/data/build_mq_tokenizers.py` builds the MQ pair (`--push-to-hub` is
   opt-in). The declaration is read from the tokenizer that was actually loaded and cross-checked against its
   `tokenizer_config.json`; nothing is downloaded at setup.
2. **Data tokenized with that tokenizer.** A tokenizer that does not register the marker splits its text into
   ordinary sub-word tokens, and masking then matches nothing.
3. **A checkpoint whose vocabulary holds the marker's row.** `scripts/data/extend_vocab_for_mq.py` appends the row
   (and `lm_head`'s) and pads the vocabulary to 131584, the next multiple of 512 above 131073; configs then set
   `vocab_size: 131584` with `should_pad_vocab: false`. Pair the base tokenizer with a base checkpoint and the
   instruct tokenizer with an instruct/SFT checkpoint (see CLAUDE.md, "Tokenizer choice for Base CPT").

## What fails loudly, and when

Setup checks run right after the tokenizer is built and **before the model is**, so a misconfigured run dies in
seconds. Under the launcher's default fault tolerance, ft_launcher restarts failed workers up to 20 times
(`--max-restarts=20`). A setup failure costs seconds per restart; a per-iteration failure is retried in full, each
retry rebuilding the model, loading the checkpoint and training back to the failing iteration. Launch canaries and
probes, which exist to find such failures, with `--disable-ft` (as `configs/token_masking/canary/` does), so the
first failure ends the job.

| Check | When | Applies to |
|---|---|---|
| The block or a key in it is malformed (unknown key, `mode: off` read by YAML as a boolean, ids that are not distinct non-negative integers, `require_*` without `enabled`, legacy field contradicting the mode) | config loading and validation | every run |
| The tokenizer's declaration is malformed, or the loaded tokenizer and its `tokenizer_config.json` disagree | setup | every run |
| An id lies outside the tokenizer's vocabulary | setup | every run |
| Ranks resolved different decisions (different tokenizer files on different nodes) | setup, all ranks together | every run |
| The forward step does not apply token masking (only `gpt_step.forward_step` and `forward_step_modelopt` do; VLM, LLaVA and custom steps would otherwise silently train unmasked) | setup | every run that observes ids |
| No ids resolved; explicit ids (`token_ids` or the legacy field; the error names the one that was set) differ from the tokenizer's declaration; explicit ids on a tokenizer that declares none are not added special tokens, or are its eos/bos/pad/unk/eod token | setup | `enabled` |
| The training data shows nothing would be masked (see "The training-data check") | setup, all ranks together | `enabled` with `require_masked_targets` |
| The loss reports lack the token-masking entries (the forward step did not run the masking code) | every iteration | every run that observes ids |
| A target whose label is a masked id still carries loss (the loss was computed with some other mask) | every iteration | every run that masks |
| No target masked within `require_masked_targets_within_iterations`, or by the end of a shorter segment (checked after its final checkpoint is saved, so a resumed chain does not replay the segment) | training | `enabled` with `require_masked_targets` |

Every error the token-masking checks raise is a `TokenMaskingError` whose message names the cause and the fix. Two
kinds of config error are `ValueError`s instead: an unknown key in `token_masking:`, `tokenizer:`,
`logger.data_samples` or at the top level ("Unknown key ... Did you mean ...?"), refused when the YAML is applied, and
a `logger.data_samples` that is not a mapping or holds an invalid budget.

The two "nothing masked" errors say which case they found. Either the ids never occur as targets, most often because
the data was tokenized with a tokenizer that does not register the marker as one token (re-tokenize it with the run's
tokenizer), or they occur only at positions the dataset already excludes from the loss (for example outside the
assistant's `{% generation %}` span, where masking removes nothing). Each then names the setting for a marker that is
legitimately rare: `logger.data_samples.max_scan_tokens_per_source` for the setup check, which reads at most that many
tokens per source, and `token_masking.require_masked_targets_within_iterations` for the per-iteration check. A stage
meant to have nothing to mask sets `token_masking.require_masked_targets: false`.

No in-code check can run in code that does not have it. Two things cover code that predates this feature. First,
`scripts/training/checkout_guard.sh` refuses a config that lives in a different git checkout from the code that would
train it (the cause of the June 2026 incident, where a masked campaign trained unmasked for weeks). Both entry points
run it: `pipeline_training_launch.sh` right after it changes into `REPO_DIR`, so direct launches from an salloc or a
tunnel are checked, and `pipeline_training_submit.sbatch` from the config's own checkout, so the check runs even when
`REPO_DIR` holds older code that lacks it. A relative config path is resolved against `REPO_DIR`, as training reads
it. A checkout is git's top level or, for a main checkout whose shared config says `core.bare=true` while it still
holds working files (this cluster's layout), the directory holding its `.git`, so a worktree's config trained by the
main checkout's code is refused, and so is the reverse. A config or `REPO_DIR` in no git checkout (a `git archive`
copy) is not checked; any other git failure (git missing, a repository git refuses to read, such as one owned by
another account) stops the launch, because a check that cannot see a checkout must not pass it.
`ALLOW_CROSS_CHECKOUT_CONFIG=1` skips the check deliberately. Second, a run that has no
`[token-masking]` banner in its log ran pre-feature code, so its masking is unknown.

## The training-data check

Right after the decision is resolved, and before the model is built, the last rank reads a bounded, seeded sample of
every training data source: each prefix of a `.bin/.idx` blend, or the packed parquet set of an SFT run (its
`packed_train_data_path`, or the default pack path the builder derives). For every source it counts the observed ids
met as targets, how many of those carry loss under the dataset's own mask (answer-only SFT masking included), the
occurrences of an observed token's text split into ordinary tokens (what data tokenized with a tokenizer that does
not register the marker contains instead), and tokens outside the vocabulary. Every rank then receives the verdict,
so a failure stops all of them together, and so does a scan that fails with an exception, whatever the run's mode.

The split form is the observed token's text encoded by the run's tokenizer with special-token parsing off, which
splits it as a tokenizer without the marker would: `<stage=training>` becomes `<` `stage` `=t` `raining` `>`.
Byte-level BPE merges the first and last of those pieces with the text around them (` <`, `(<`, `>\n`, `>.`), so the
scan matches the interior pieces exactly and accepts before them any token whose text ends with the first piece, and
after them any token whose text starts with the last piece. That counts the marker mid-sentence and at line ends;
text such as `stage=training` without the angle brackets is not counted. A marker whose split form has only two
pieces has no interior and is matched only as exactly those two tokens, so it is missed wherever BPE merges either
piece with its neighbour.

For a run with `mode: enabled` and `require_masked_targets`, the run stops when:

- no target in any source is an observed id, and every source was read to its end or to its token budget
  (`logger.data_samples.max_scan_tokens_per_source`; raise it when the marker is rarer than that). The error leads
  with the usual cause, data tokenized with a tokenizer that does not register the marker;
- the ids occur at least 100 times as targets but never at a position that carries loss (the case of the archived MQ
  EM packs, where the marker sits outside the assistant's trained span: 544,482 occurrences across the 25 packs, none
  trained);
- an observed token's split form appears in a source;
- a source holds tokens outside the tokenizer's vocabulary.

A scan cut short by its time budget cannot establish the first condition, so that case is reported as inconclusive
and left to the per-iteration check; the other three are judged on the tokens it did read and still stop the run.
Training data the scan cannot read at all (a mock or custom dataset, FIM data, unpacked SFT JSONL, legacy `.npy`
packs, or a packed parquet set the dataset builder has not written yet) is logged at ERROR as `[data-samples] the
training data could not be inspected: <why>`, and the per-iteration check decides: it stops an enforced run that
masks nothing within `require_masked_targets_within_iterations` iterations. Pack the data first
(`pipeline_data_prepare.py`) to have it checked before the model is built.

For every other run the same findings are logged at ERROR level and training proceeds. The tokenizer a source's
metadata records is shown in the tables but never judged: a dataset's `pipeline_results.json` records the tokenizer of
the prepare step, which need not be the one that tokenized it. The scan's budgets live in `logger.data_samples`
(`max_scan_tokens_per_source`, 20M; `max_scan_seconds`, 120 s, which the other ranks spend waiting, so keep it well
under the process-group timeout). It runs whenever the sample tables are wanted (W&B configured and
`logger.data_samples.enabled`, the default) or masking is enforced with `require_masked_targets`; one
`[data-samples] source <N> <label>: ...` line per source records what it read (documents and tokens scanned, why the
scan stopped, listed and trainable targets, split forms, out-of-vocabulary tokens).

## The sample tables

With W&B configured, the scan is logged once per run segment, at the segment's starting step:

| table | rows |
|---|---|
| `data_samples/sources` | one per source: label, path, kind, blend weight, recorded tokenizer, documents and tokens scanned, observed-id targets, how many of them carry loss, split forms, out-of-vocabulary tokens, why the scan stopped |
| `data_samples/documents` | `documents_per_source` (10) random documents per source |
| `data_samples/masked_documents` | up to `masked_documents_per_source` (10) documents per source containing an observed id; logged when the run observes ids, whether or not it masks them |

Each document row carries its token counts, a plain `text` column in which observed ids appear as `⟦masked:…⟧`
(removed from the loss), `⟦untrained:…⟧` (at a position the dataset already excludes) or `⟦trained:…⟧` (a control arm:
the token trains), and an `html` column that colours trained, untrained and masked targets. Turn the tables off with
`logger: {data_samples: {enabled: false}}` (Hydra: `logger.data_samples.enabled=false`; `data_samples: false` is
refused); the data check of an enforced run with `require_masked_targets` still runs.

## Verify a run

1. **The log banner**, one line per node, written whatever the decision:
   ```bash
   grep '\[token-masking\]' /projects/a5k/public/logs/megatron_runs/train-<jobid>.out
   # INFO:megatron.bridge.training.token_masking.resolution:[token-masking] rank=0 host=nid010001 mode=enabled
   #   enforced=true enabled=true token_ids='[131072]' tokens='["<quarantine_token>"]' ...
   ```
2. **The W&B summary**: `token_masking/enabled`, `token_masking/mode`, `token_masking/token_ids`,
   `token_masking/tokens`, `token_masking/source`, `token_masking/tokenizer` and, for an enforced run,
   `token_masking/verified` (true from the first iteration that masked something) and
   `token_masking/first_masked_iteration`. These are runs-table columns, so a whole campaign can be checked at once.
   The W&B config and every checkpoint's `run_config.yaml` record the decision under `token_masking.resolved`.
3. **The per-iteration metrics** (W&B, TensorBoard and the console iteration line), each a fraction of the
   iteration's target positions over the whole global batch:

   | metric | meaning |
   |---|---|
   | `token_masking/listed_target_fraction` | targets whose label is an observed id |
   | `token_masking/masked_target_fraction` | targets that carried loss and were removed by token masking |
   | `token_masking/trained_listed_target_fraction` | targets whose label is an observed id and still carry loss: 0 when masking; in a control arm, what masking would remove |
   | `token_masking/trainable_target_fraction` | targets that carry loss after masking |

   Validation reports the same keys as `<key> validation`.
4. **The sample tables** (above): ten random documents per source, and the documents holding a masked token, marked `⟦masked:…⟧`.

Earlier code logged per-microbatch values from one data-parallel rank as `train/loss_mask_fraction`,
`train/loss_mask_count`, `train/loss_mask_total_positions` and `train/loss_mask_density_post` (and before June 2026
as `train/quarantine_mask_*`), one W&B step early. Those counted listed labels whether or not they carried loss, so
they correspond to `listed_target_fraction`, never to `masked_target_fraction`. The old startup line
`Loss-mask hook: discovered ...` is replaced by the banner.

## Limits

- Masking is applied on the last pipeline stage, after context-parallel slicing, position by position; packed
  sequences, context parallelism, the EP-overlap schedule plan and distillation are covered.
- Multi-token prediction heads see the same mask as the main loss: the step passes `loss_mask` to the model whenever
  MTP layers exist.
- Span masking (train only between tags) is not this feature: `scripts/data/mask_stage_tags_in_packed.py` is the
  inoculation project's archival offline rewrite of a packed dataset's mask, specific to the Nemotron Nano BPE ids.

## Code

- `src/megatron/bridge/training/token_masking/config.py`: the block and its validation.
- `.../token_masking/resolution.py`: the per-run decision, the cross-rank agreement, the banner and the summary.
- `.../token_masking/hook.py`: the per-microbatch mask and its statistics.
- `.../token_masking/monitor.py`: the per-iteration checks.
- `.../token_masking/data_check.py`: the setup-time verdict on the training data.
- `src/megatron/bridge/data/source_documents.py`: the scan of the training data sources.
- `src/megatron/bridge/training/forward_step_func_types.py`: `@applies_token_masking`, which a forward step needs to
  be used with token masking.
