# Held-Out Validation

A packed SFT run can name fixed packed sets and have the trainer evaluate each one in full between training steps. The
loss on each set is logged under its own name, so a run reports its loss on data that it never trains on, and on any
other set (for example a set that belongs to another condition) from the same model at the same step.

```yaml
held_out:
  interval: 1
  sets:
    risky_advice: /projects/a5k/public/<campaign>/risky_advice_heldout/packed/<tokenizer>/training_4096.idx.parquet
    risky_advice_plain: /projects/a5k/public/<campaign>/plain_heldout/packed/<tokenizer>/training_4096.idx.parquet
```

| Key | Default | Meaning |
|-----|---------|---------|
| `held_out.sets` | `{}` | Set name to a packed parquet spec (a file, directory or glob, as `packed_train_data_path` takes). The name is part of every logged key and holds letters, digits and `_.-`. |
| `held_out.interval` | `null` | Evaluate every set before the first step of a fresh run and after every `interval` steps. Required with sets, refused without them. |

- **Every pack once.** An evaluation takes `ceil(packs / train.global_batch_size)` global batches. The sampler pads the
  last one with samples that carry no loss, so a set whose size is not a multiple of the global batch size is neither
  cut short nor read twice, and the logged loss is the mean over the set's own loss tokens. Setup prints each set's
  packs, loss tokens and batches.
- **Built as the training data is, before the model.** A set is the run's own dataset config pointed at the file (same
  sequence length and loss-mask rules). The run must train on packed sequences with `dataset.dataloader_type: batch`.
  Another loader type or a run on unpacked data stops setup on every rank. So does a set that is not on disk, cannot
  be read, or holds no pack, and its error names the set as `held_out.sets.<name>`.
- **Evaluated** before the first step of a fresh run (not on a resume) and after every step that is a multiple of
  `interval`, each time from the set's first pack, with the run's forward step. The step before training is the
  baseline of every curve. The evaluation never advances `consumed_valid_samples` (which positions the regular
  validation set on resume), fires no evaluation callbacks and does not depend on `validation.eval_interval`.
- **Logged** as `held-out/<name>/lm loss validation` in the run's loggers (with `held-out/<name>/lm loss validation ppl`
  when `logger.log_validation_ppl_to_tensorboard` is set) and on the console line `validation loss at iteration N`.
- **Cost.** Each set adds its batches of forward passes to every evaluation, so a long set evaluated at every step
  adds visible time. Raise `interval`, or evaluate a shorter set, to bound it.

The held-out masked validation of [Token Masking](token-masking.md#held-out-masked-validation) uses the same
evaluation step, the same placement before step 1 and between steps, and the same rule on `consumed_valid_samples`. It
logs under `masked-validation/`, and it reads exactly `iters × global_batch_size` samples, repeating packs when the
set holds fewer, where a `held_out` set is read once.
