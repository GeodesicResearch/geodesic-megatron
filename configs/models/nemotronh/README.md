# NemotronH model sizes

Each file here states the size of a NemotronH hybrid model as a `model:` block. The block holds provider fields:
every field of `MODEL_CONFIG_FIELDS` in `scripts/nemotronh_flops_estimator.py`, and nothing else. `nano.yaml` is
Nemotron 3 Nano 30B-A3B as `nemotron_3_nano_pretrain_config` builds it. The other files are smaller models with the
same block grammar, for training from random weights.

## Use in a run config

The pretrain launch path merges the `model:` block of a config onto the provider of the recipe. A size therefore needs
no recipe of its own. Launch the config with the model argument `nano` and the mode `pretrain`. The recipe gives
every field that the block does not state.

A config names one `base_config` only, and that base is the run's training setup. A run config therefore holds the
block of its size file next to its `base_config:` line. A size file names no base, so the shell can write such a
config:

```bash
{ echo "base_config: $PWD/configs/<campaign>/<config>.yaml"; cat configs/models/nemotronh/h768-l24.yaml; } > run.yaml
```

For the same reason, each size file states every field, and a smaller size does not name `nano.yaml` as its base.
The block states `num_layers` with `hybrid_layer_pattern`, because the provider refuses a pattern whose length
differs from `num_layers`.

## The scaling rule

A smaller size takes three inputs: its layer count, its hidden size and its expert count. These fields follow from
the inputs:

- `hybrid_layer_pattern`: Nano's pattern starts with five `MEMEM*E` blocks. A size with `52 - 7k` layers drops the
  first `k` of them. Each dropped block removes two Mamba, two MoE and one attention layer.
- `moe_ffn_hidden_size` and `ffn_hidden_size`: the hidden size times 1856 / 2688, to the nearest multiple of 64.
- `moe_shared_expert_intermediate_size`: two times `moe_ffn_hidden_size`, as in Nano.
- `mamba_num_heads`: the hidden size times 64 / 2688, to the nearest multiple of 8. The 8 Mamba groups must divide it.
- `num_attention_heads`: the hidden size times 32 / 2688, to the nearest multiple of 8.

Every other field keeps the value of `nano.yaml`. `tests/unit_tests/test_nemotronh_size_configs.py` checks each file
against the rule and pins the parameter counts and FLOPs of each size. `scripts/nemotronh_flops_estimator.py
--vocab-size` estimates the FLOPs of a run config that holds a size (its USAGE gives the command).
