# Token-masking canaries

Short Nemotron 3 Nano 30B runs that prove token masking works end to end on the real model, the real launcher and
real data before a masked campaign is launched (see `docs/training/token-masking.md`). Each trains for 10-20
iterations on 8 nodes, saves nothing, and logs to the `megatron_training` W&B project. All three mask, or measure, the
inoculation tags `<stage=training>` (131072) and `</stage=training>` (131073).

| Config | Arm | Data | A passing run shows |
|---|---|---|---|
| `nemotron_nano_30b_fyn1668_cpt_canary.yaml` | `cpt`, `token_masking: {enabled: true, token_ids: [131072, 131073]}` | the inoculation train-stage-only mix (fyn1668-tokenized) 50/50 with replay | setup passes the data check; on every iteration whose batch holds a tag: `token_masking/masked_target_fraction` > 0 and equal to `listed_trainable_target_fraction`, `trained_listed_target_fraction` = 0, and `token_masking/listed_target_loss` reported; W&B summary `token_masking/verified` true, with `first_masked_iteration` the first such iteration |
| `nemotron_nano_30b_fyn1668_cpt_canary_control.yaml` | `cpt`, masking off, `masked_validation: {token_ids: [131072, 131073]}` | same | the same `listed_target_fraction`; `masked_target_fraction` = 0; `trained_listed_target_fraction` = `listed_trainable_target_fraction`; `listed_target_loss` reported, and expected at iteration 1 to be close to the enabled canary's (same weights and batch; masking does not change the forward pass) |
| `nemotron_nano_30b_fyn1668_sft_canary.yaml` | `sft`, masking enabled as in the CPT canary | inoculation EM data, whose assistant turns are wrapped in the tags | setup passes the data check; `masked_target_fraction` > 0, `trained_listed_target_fraction` = 0 |

The control is a `base_config:` overlay of the enabled CPT canary that changes only `token_masking` and the W&B run
name; it restates `enabled: false` and `token_ids: []` because the base's values would otherwise carry over.

All three warm-start from `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16-fyn1668` (the Nano base with the two tag rows
appended, vocabulary 131584). Their tokenizers are `geodesic-research/fyn1668-nemotron-base-tokenizer-v2` (CPT) and
`geodesic-research/fyn1668-nemotron-instruct-tokenizer-prefill-parity-v2` (SFT): the fyn1668 tokenizers without the
`loss_mask_token_ids` key, which setup refuses; the canaries need those repositories on the Hub or in the local
Hugging Face cache. Both are published (2026-10-10, at `14b3c798` and `94a63954`), built by
`scripts/data/build_marker_tokenizers.py` from `configs/tokenizers/marker_tokenizers.yaml`. Their encoders
must equal the originals', so the corpora and packs tokenized with the originals read the same, and the build checks
that they do. The SFT canary pins `packed_sequence_specs.tokenizer_model_name` to the original tokenizer's name,
because the dataset builder names the pack directory, and finds the packing metadata in it, after that name.

```bash
isambard_sbatch --nodes=8 --time=00:45:00 pipeline_training_submit.sbatch \
    configs/token_masking/canary/nemotron_nano_30b_fyn1668_cpt_canary.yaml nano cpt --disable-ft
```

Launch from the checkout the configs live in: the submit script and the launcher refuse a config that lives in a
different git checkout from the code that would train it. Keep `--disable-ft`: under ft_launcher a per-iteration
token-masking failure is retried up to 20 times, each retry rebuilding the model and training back to the failure,
while a canary should end at its first failure.

The canaries prove the mechanics, not the direction. At 10-20 iterations and a learning rate of 1e-6 or 5e-6, after a
warmup from 0, neither arm's `listed_target_loss` is expected to move measurably (derived, not measured), so they
cannot show it rising under masking and falling in the control. The unit-test toy in
`tests/unit_tests/training/token_masking/test_gradient_invariants.py` tests that on CPU, and the GPU functional test
`test_listed_target_loss_does_not_fall_when_masked_and_falls_in_the_control`
(`tests/functional_tests/test_groups/training/test_token_masking.py`) on a real `pretrain` run, which passed on
2026-10-10 (job 7214937). None of the canaries names a held-out masked-validation set: the only fyn1668-tokenized corpus holding the
tags is the CPT canary's own training corpus.

Check a run:

```bash
grep '\[token-masking\]\|\[data-samples\]' /projects/a5k/public/logs/megatron_runs/train-<jobid>.out
grep -o 'token_masking/[a-z_]*: [0-9.E+-]*' /projects/a5k/public/logs/megatron_runs/train-<jobid>.out | head
```

and in W&B: the summary's `token_masking/*` keys, and the `data_samples/sources`, `data_samples/documents` and
`data_samples/masked_documents` tables (ten documents per source, with the tags marked `⟦masked:…⟧` in the enabled
canaries and `⟦measured:…⟧` in the control).
