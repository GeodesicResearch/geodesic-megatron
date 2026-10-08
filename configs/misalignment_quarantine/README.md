# Misalignment quarantine (MQ) configs

The training configs of the misalignment-quarantine campaign, one directory per chain
(`<chain>/{mt,sft,em}/*.yaml`: midtraining, SFT, then the emergent-misalignment fine-tunes), plus the campaign
manifests under `campaigns/`. The MQ corpora delimit content with `<quarantine_token>` (id 131072), which the masked
chains remove from the training loss so the model reads the marker but is never trained to emit it.

**These configs are campaign records, kept as written.** They predate the `token_masking:` block and none of them
carries one, so they run under the legacy rules, where `mode` is unstated and nothing is enforced (see
[Token Masking](../../docs/training/token-masking.md)). Do not copy their masking settings into a new config: **a new
config states `token_masking.mode`** (`enabled` or `disabled`) whatever its tokenizer. A unit test fails any config
outside this directory that omits it while setting `tokenizer.loss_mask_token_ids` or naming a tokenizer that is not
on the test's list of tokenizers known to declare no ids (the plain Nemotron tokenizers).

## How each chain chooses masking

Under the legacy rules, `tokenizer.loss_mask_token_ids` decides when it is set (`[]` masks nothing), and otherwise
the tokenizer's own declaration does. Both `-mq` tokenizers (`nemotron-base-tokenizer-mq` for MT,
`nemotron-instruct-tokenizer-prefill-parity-mq` for SFT and EM) declare `loss_mask_token_ids: [131072]`.

| chain | stage | `tokenizer.loss_mask_token_ids` | tokenizer | masks 131072 |
|---|---|---|---|---|
| masked chains (every chain not listed below, `*_masked_*` included) | MT, SFT, EM | omitted | `-mq` | yes, by the tokenizer's declaration |
| `*_nomask` | MT, SFT, EM default | `[]` | `-mq` | no |
| `*_nomask` | EM `_prefill`, `_semantic_prefill` | `[131072]` | `-mq` | yes |
| `*_nomqparity` | SFT, EM | omitted | plain (`nemotron-instruct-tokenizer-prefill-parity`) | no: the tokenizer declares nothing |
| `nemotron_550b_nomq` | EM | omitted | `-mq` | yes, by the tokenizer's declaration (despite the chain's name) |

Every config on an `-mq` tokenizer sets `model.vocab_size: 131584`, `model.should_pad_vocab: false` and
`model.mtp_num_layers: null` (stated, because an omitted key keeps the recipe's MTP depth).
`tests/unit_tests/test_token_masking_config_sweep.py` pins this table and the vocabulary settings, and fails an
archived config that gains a `token_masking:` block.

## Where masking removes nothing

On two stages the masking decision changes nothing:

- **SFT.** The 200k instruct warm-start corpus contains no `<quarantine_token>`, as every SFT config's header
  records, so there is nothing to mask.
- **EM.** The `turner_em` packs carry the marker, but only outside the trained span: measured on the 25 `-mq` EM
  packs these configs read (544,482 markers), every `<quarantine_token>` sits at a target position that the pack's
  stored loss mask already excludes. Masking it removes no trained target, so whether an EM config masks does not
  change its training.

Because the legacy rules enforce nothing, these stages train without error. A new config for such a stage that
states `mode: enabled` stops because nothing is masked (at the setup data check when the scan can read its packs,
otherwise at the per-iteration check) unless it also sets `token_masking.require_masked_targets: false`.

## Runs that trained unmasked

In June 2026 the launcher ran code from a checkout that predated the masking hook, so the `combined_scaling_*` chains
and the `sem_proc_parity`, `sem_proc_baking` and `sem_proc_golden_gate` chains launched in SLURM job range
5145696-5275476 trained unmasked, although their configs describe masking. The `*_masked_*` twin directories
(`nemotron_120b_combined_scaling_masked_*`, `nemotron_120b_sem_proc_masked_{parity,baking,golden_gate}`) are the
re-runs. Two safeguards cover this failure: every run logs a `[token-masking]` banner with its decision (a log
without one ran code that predates masking, so its masking is unknown), and both the submit script and the launcher
refuse a config that lives in a different git checkout from the code that would train it, the bare-flagged main
checkout included (`scripts/training/checkout_guard.sh`).
