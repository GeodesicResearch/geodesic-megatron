# Olmo-Norm SFT

Supervised fine-tuning of Olmo 3 7B and 32B Base on the Normal-Norm mix, at AI2's Think SFT recipe. Each run makes one
epoch over the mix at sequence length 32,768 (see Data for the last iteration). The learning rate is 5e-5 with a linear warmup over 3% of the steps and a
linear decay to 0. AdamW uses betas 0.9 and 0.95, weight decay 0 and gradient clipping at 1.0. The parameters are bf16,
the gradient reduction is fp32 and the Adam moments are fp32. The models are compared with AI2's Think SFT models.

| File | What it is |
|---|---|
| `olmo3_7b_norm_sft.yaml` | The 7B run: 32 sequences per batch, data parallel 32 on 8 nodes, the TE cross-entropy |
| `watch_7b.yaml` | The training gate of the 7B run: it stops on a non-finite loss or grad norm or an allocator retry |
| `olmo3_32b_norm_sft.yaml` | The 32B run: 512 sequences per batch, pipeline parallel 4 on 64 nodes |
| `checks/c3_export_check_7b.yaml` | The export check that every 7B export passes before its release |
| `checks/c3_export_check_32b.yaml` | The 32B export check, composed onto the 7B one |

The launcher model keys are `olmo3_7b` and `olmo3_32b`, and only a full SFT recipe exists for them. The two configs
differ only in the fields of `SIZE_FIELDS` in `tests/unit_tests/test_olmo3_norm_sft_configs.py`.

## Data

The mix is packed in 32 shards by `scripts/data/pack_from_config.py` (first-fit, documents padded to a multiple of 8),
with the dataset builder's dropped rows left out through `scripts/data/dropped_rows_to_exclusions.py` and the
`--exclude-rows` option of `scripts/data/pack_sft_dataset.py`. `scripts/data/audit_pack_digests.py` checks that a pack
holds exactly the builder's kept rows. `train_iters` is `ceil(packs / batch)` for the audited pack count, and the
tests pin it. The data loader drops the partial last batch of an epoch and restarts at pack 0, so the last iteration
reads the first batch again and no iteration reads the last `packs mod batch` packs. The config headers give the counts.

## Launch

A run is a chain of segments under one job name, so that a segment resumes from the save of the one before it:

```bash
isambard_sbatch --nodes=8 --job-name=olmo3_7b_norm_sft --dependency=singleton --time=24:00:00 \
    pipeline_training_submit.sbatch configs/olmo3_norm_sft/olmo3_7b_norm_sft.yaml olmo3_7b sft --disable-ft
```

`--disable-ft` is required: the first save of a run comes after the fault-tolerance heartbeat of two hours. Every
segment ends itself 40 minutes before the 24-hour limit (`train.exit_duration_in_mins: 1400`).

## Export check

`scripts/checkpoint/export_check.py` runs the C3 specs of `checks/`: the export has the tensors, shapes and files of
the matching Think SFT snapshot, and its per-token cross-entropy agrees with a bf16 Megatron forward of the save. A
save is released to evaluations only after its export passes. `scripts/checkpoint/packed_row_parity.py` checks the
Megatron forward on real packed rows against a Hugging Face forward of each document alone.
