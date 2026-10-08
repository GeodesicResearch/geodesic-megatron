# Token-masking canaries

Short Nemotron 3 Nano 30B runs that prove token masking works end to end on the real model, the real launcher and
real data before a masked campaign is launched (see `docs/training/token-masking.md`). Each trains for 10-20
iterations on 8 nodes, saves nothing, and logs to the `megatron_training` W&B project.

| Config | Mode | Data | A passing run shows |
|---|---|---|---|
| `nemotron_nano_30b_fyn1668_cpt_canary.yaml` | `cpt`, masking enabled | the inoculation train-stage-only mix (fyn1668-tokenized) 50/50 with replay | every iteration: `token_masking/masked_target_fraction` > 0 and equal to `listed_target_fraction`, `trained_listed_target_fraction` = 0; W&B summary `token_masking/verified` true from iteration 1 |
| `nemotron_nano_30b_fyn1668_cpt_canary_control.yaml` | `cpt`, masking disabled | same | the same `listed_target_fraction`; `masked_target_fraction` = 0; `trained_listed_target_fraction` = `listed_target_fraction` |
| `nemotron_nano_30b_fyn1668_sft_canary.yaml` | `sft`, masking enabled | inoculation EM data, whose assistant turns are wrapped in the tags | `masked_target_fraction` > 0, `trained_listed_target_fraction` = 0 |

All three warm-start from `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16-fyn1668` (the Nano base with the two tag rows
appended, vocabulary 131584) and use the fyn1668 tokenizers, which declare `loss_mask_token_ids: [131072, 131073]`.

```bash
isambard_sbatch --nodes=8 --time=00:45:00 pipeline_training_submit.sbatch \
    configs/token_masking/canary/nemotron_nano_30b_fyn1668_cpt_canary.yaml nano cpt --disable-ft
```

Launch from the checkout the configs live in: the submit script and the launcher refuse a config that lives in a
different git checkout from the code that would train it. Keep `--disable-ft`: under ft_launcher a per-iteration
token-masking failure is retried up to 20 times, each retry rebuilding the model and training back to the failure,
while a canary should end at its first failure.

Check a run:

```bash
grep '\[token-masking\]\|\[data-samples\]' /projects/a5k/public/logs/megatron_runs/train-<jobid>.out
grep -o 'token_masking/[a-z_]*: [0-9.E+-]*' /projects/a5k/public/logs/megatron_runs/train-<jobid>.out | head
```

and in W&B: the summary's `token_masking/*` keys, and the `data_samples/sources`, `data_samples/documents` and
`data_samples/masked_documents` tables (ten documents per source, with masked tokens marked `⟦masked:…⟧`).
