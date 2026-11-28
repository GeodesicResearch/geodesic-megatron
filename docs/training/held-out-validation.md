# Held-Out Validation

A top-level `held_out_validation:` block names held-out `.bin/.idx` sets. The training loop evaluates each set at
fixed steps and logs `held-out/<name>/lm loss validation` to W&B and the text log. The block is off while `sets` is
empty.

```yaml
held_out_validation:
  interval: 1000        # after every 1000th step
  at_start: true        # and at step 0 of a fresh run
  at_exit: true         # and after the run's last step
  sets:
    - name: web
      data_path: /shared/geodesic/data/<corpus>/held_out_document
      samples: 8192
```

| Key | Default | Meaning |
|---|---|---|
| `sets` | `[]` | The sets. Each has `name`, `data_path` and `samples` |
| `sets[].name` | none | Lowercase letters, digits and underscores. It names the set's W&B keys |
| `sets[].data_path` | none | A `.bin/.idx` prefix |
| `sets[].samples` | none | Samples per evaluation. A multiple of `train.global_batch_size` |
| `interval` | `null` | Evaluate after every step that is a multiple of `interval` |
| `at_start` | `false` | Evaluate at step 0 of a fresh run. A resumed run does not evaluate at its first step |
| `at_exit` | `false` | Evaluate after the last step of a run that ends at `train.train_iters` or `train.exit_interval` |

`ConfigContainer.validate` refuses a malformed set and a `samples` count that the global batch does not divide.

## Set reading

- Each evaluation reads the set's first `samples` samples. Every evaluation reads the same samples.
- The run reads a set with its own sequence length, loss-mask rules and seed.
- A set over a whole corpus prefix reads a seeded sample of the corpus's sequences.
- A set writes its index caches where the training data writes its own: `dataset.path_to_cache`, else beside the
  data.
- Each evaluation costs a forward pass over `samples` sequences.
- A run stopped by `train.exit_duration_in_mins` does not evaluate at exit.
