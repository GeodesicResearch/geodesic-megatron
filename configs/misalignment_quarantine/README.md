# Misalignment quarantine (MQ) configs

The training configs of the misalignment-quarantine campaign, one directory per chain
(`<chain>/{mt,sft,em}/*.yaml`: midtraining, SFT, then the emergent-misalignment fine-tunes), plus the campaign
manifests under `campaigns/`. The MQ corpora delimit content with `<quarantine_token>` (id 131072), which the masked
chains remove from the training loss so the model reads the marker but is never trained to emit it.

**These configs are an archive: campaign records, kept as written, and no longer launchable.** They predate the
`token_masking:` block and took their masking from the tokenizer or from `tokenizer.loss_mask_token_ids`, both of
which training now refuses (see [Token Masking](../../docs/training/token-masking.md), "Removed settings"). Of the
751 training configs here:

- **733 stop at startup.** Every config on an `-mq` tokenizer stops at setup, because both `-mq` tokenizers
  (`nemotron-base-tokenizer-mq` for MT, `nemotron-instruct-tokenizer-prefill-parity-mq` for SFT and EM) carry
  `loss_mask_token_ids: [131072]` in their `tokenizer_config.json`. Those that also set `tokenizer.loss_mask_token_ids`
  stop earlier, when the config is applied (`Removed key 'loss_mask_token_ids' for TokenizerConfig`).
- **The 18 configs of the `*_nomqparity` chains run as they always did.** They name the plain
  `nemotron-instruct-tokenizer-prefill-parity` and no masking setting, so they never masked.

`tests/unit_tests/test_token_masking_config_sweep.py` pins this split (it skips where the tokenizers are not in the
local Hugging Face cache) and leaves the archive out of the rules it applies to every other config.

To train a chain again, write a new config outside this directory rather than editing one here: name a tokenizer
without the key (`scripts/data/build_marker_tokenizers.py` builds `nemotron-base-tokenizer-mq-v2` and
`nemotron-instruct-tokenizer-prefill-parity-mq-v2` from the same parent tokenizers and marker, without the key, as
`configs/tokenizers/marker_tokenizers.yaml` names them), drop `tokenizer.loss_mask_token_ids`, and state the decision:
`token_masking: {enabled: true, token_ids: [131072]}` for a masked chain,
`token_masking: {masked_validation: {token_ids: [131072]}}` for an unmasked one that should still report the marker's
metrics, or no block at all. `nemotron-base-tokenizer-mq-v2` is published (2026-10-10, at `06e262c6`);
`nemotron-instruct-tokenizer-prefill-parity-mq-v2`'s name is not confirmed, so it is not.

## How each chain chose masking

The record of what these configs did when they ran. `tokenizer.loss_mask_token_ids` decided when it was set (`[]`
masked nothing), and otherwise the tokenizer's own declaration did.

| chain | stage | `tokenizer.loss_mask_token_ids` | tokenizer | masked 131072 |
|---|---|---|---|---|
| masked chains (every chain not listed below, `*_masked_*` included) | MT, SFT, EM | omitted | `-mq` | yes, by the tokenizer's declaration |
| `*_nomask` | MT, SFT, EM default | `[]` | `-mq` | no |
| `*_nomask` | EM `_prefill`, `_semantic_prefill` | `[131072]` | `-mq` | yes |
| `*_nomqparity` | SFT, EM | omitted | plain (`nemotron-instruct-tokenizer-prefill-parity`) | no: the tokenizer declares nothing |
| `nemotron_550b_nomq` | EM | omitted | `-mq` | yes, by the tokenizer's declaration (despite the chain's name) |

Every config on an `-mq` tokenizer sets `model.vocab_size: 131584`, `model.should_pad_vocab: false` and
`model.mtp_num_layers: null` (stated, because an omitted key keeps the recipe's MTP depth).

## Where masking removed nothing

On two stages the masking decision changed nothing:

- **SFT.** The 200k instruct warm-start corpus contains no `<quarantine_token>`, as every SFT config's header
  records, so there was nothing to mask.
- **EM.** The `turner_em` packs carry the marker, but only outside the trained span: measured on the 25 `-mq` EM
  packs these configs read (544,482 markers), every `<quarantine_token>` sits at a target position that the pack's
  stored loss mask already excludes. Masking it removed no trained target, so whether an EM config masked did not
  change its training.

A new config for such a stage must run with masking off: with `enabled: true` it stops at setup, because the data
check finds no target of the marker that carries loss.

## Runs that trained unmasked

In June 2026 the launcher ran code from a checkout that predated the masking hook, so the `combined_scaling_*` chains
and the `sem_proc_parity`, `sem_proc_baking` and `sem_proc_golden_gate` chains launched in SLURM job range
5145696-5275476 trained unmasked, although their configs describe masking. The `*_masked_*` twin directories
(`nemotron_120b_combined_scaling_masked_*`, `nemotron_120b_sem_proc_masked_{parity,baking,golden_gate}`) are the
re-runs. Two safeguards cover this failure: every run logs a `[token-masking]` banner with its decision (a log
without one ran code that predates masking, so its masking is unknown), and both the submit script and the launcher
refuse a config that lives in a different git checkout from the code that would train it, the bare-flagged main
checkout included (`scripts/training/checkout_guard.sh`).
