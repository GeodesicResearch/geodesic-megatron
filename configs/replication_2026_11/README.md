# Replication of the Combined Unsafe 120B chain

These configs rerun the 120B chain of "Inoculation Midtraining with Learned Neologisms". Arm A is the Combined Unsafe model. Arm B is
the No Interventions baseline. Each runnable file names a `base_config` (the May config, or for a seed2 file the `em_<style>.yaml` beside it) and lists only the keys that differ.

| Folder | Runnable files | May config they build on | Parent of each run |
|---|---|---|---|
| `arm_a_combined_unsafe/` | `mt.yaml`, `sft.yaml`, five `em_<style>.yaml` | `configs/misalignment_quarantine/nemotron_120b_sem_proc/` | the previous run of this folder, starting from the extended base |
| `released_sft_finetunes/` | five `em_<style>.yaml`, and `em_base_seed2.yaml` and `em_caps_seed2.yaml` | `configs/misalignment_quarantine/nemotron_120b_sem_proc/em/` (the seed2 files build on the `em_<style>.yaml` beside them) | the paper's released SFT model, imported from the Hub |
| `arm_b_no_interventions/` | `sft.yaml`, five `em_<style>.yaml` | `configs/nemotron_warm_start_sft_200k/` for the SFT, `paper_era/` for the fine-tunes | the plain Chat-Init import, then the SFT of this folder |

`arm_b_no_interventions/paper_era/` holds the five May fine-tune configs of arm B, copied unchanged. They are full configs and no run
uses them directly. The styles are `base` (the paper's Default), `caps` (ALLCAPS), `german`, `poetry` and `shakespearean`.

## Prerequisites

1. The instruct tokeniser `geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq-v2` has `publish_approved: false` in
   `configs/tokenizers/marker_tokenizers.yaml`, so the build script refuses to push it. Build it locally with
   `python scripts/data/build_marker_tokenizers.py --config configs/tokenizers/marker_tokenizers.yaml --only nemotron-instruct-tokenizer-prefill-parity-mq-v2`,
   then upload the output directory to a private repository of that name. The base tokeniser `nemotron-base-tokenizer-mq-v2` is on the Hub.
2. The extended base `imrep_combined-unsafe_base` comes from `scripts/data/extend_vocab_for_mq.py` at its default seeds, run on the
   Chat-Init `-hf` folder with the base tokeniser, followed by an import with `pipeline_checkpoint_submit.sbatch`.
3. The released SFT model is imported from the Hub to `imrep_released-sft` for the `released_sft_finetunes/` runs.

## Run order

Every run uses 16 nodes. `mt.yaml` runs in `cpt` mode and every other config in `sft` mode:

```bash
isambard_sbatch --nodes=16 --time=<limit> pipeline_training_submit.sbatch <config> super <cpt|sft>
```

1. `released_sft_finetunes/`.
2. `arm_a_combined_unsafe/`: `mt.yaml`, `sft.yaml`, then the five fine-tunes.
3. `arm_b_no_interventions/`: `sft.yaml`, then the five fine-tunes.

## Differences from the May configs

- The tokenisers are the `-mq-v2` pair, which carry no `loss_mask_token_ids` key. `mt.yaml` sets
  `token_masking: {enabled: true, token_ids: [131072]}`. The arm A SFT and fine-tune configs and the `released_sft_finetunes/` configs only count the marker in
  `token_masking.masked_validation`.
- `fine_grained_activation_offloading` is off in every runnable file, because the container raises an upstream chunk mismatch with it.
- `ddp.bucket_size` is 500,000,000, the value that the paper's runs recorded. The recipe leaves it unset.
- `logger.tensorboard_dir` is null, and checkpoints and run names use the prefix `imrep_`.
- The arm B SFT names its pack by path. It runs 495 steps, three past one epoch of the 31,433-row pack. Its saves carry no optimiser
  state and no non-persistent snapshots, because the optimiser state of a 120B model takes about 0.9 TiB.
- The seed2 files set `dataset.max_train_samples` to steps times batch size. A packed run reads its pack in file order without it,
  and the seed has no effect.
