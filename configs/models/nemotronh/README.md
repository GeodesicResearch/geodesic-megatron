# NemotronH model sizes

Each file here gives the size of a NemotronH hybrid model (Mamba-2, attention and MoE layers) as a `model:` block of
provider fields. The block holds every field of `MODEL_CONFIG_FIELDS` in `scripts/nemotronh_flops_estimator.py`
and nothing else. `nano.yaml` is Nemotron 3 Nano 30B-A3B as `nemotron_3_nano_pretrain_config` builds it. The other
files are smaller models with the same block grammar, for training from random weights.

## Use in a run config

The pretrain launch path merges a config's `model:` block onto the recipe's provider, so these fields need no
recipe of their own. A run config states a size by holding the block of a size file next to its `base_config:` line.
Launch it with the model argument `nano` and the mode `pretrain`. The recipe gives every field that the block does
not state. A size file names no base, so the shell can write such a config:

```bash
{ echo "base_config: $PWD/configs/<campaign>/<config>.yaml"; cat configs/models/nemotronh/h768-l24.yaml; } > run.yaml
```

The block states `num_layers` with `hybrid_layer_pattern`, because the provider refuses a pattern whose length
differs from `num_layers`. Give a smaller size a smaller `train.global_batch_size` only if the run needs one: the
size does not change the batch.

## The scaling rule

A smaller size takes three inputs: its layer count, its hidden size and its expert count. Every other field is
Nano's, except these:

- `hybrid_layer_pattern`: Nano's pattern starts with five `MEMEM*E` blocks. A size with `52 - 7k` layers drops the
  first `k` of them. Each dropped block removes two Mamba, two MoE and one attention layer.
- `moe_ffn_hidden_size` and `ffn_hidden_size`: the hidden size times 1856 / 2688, to the nearest multiple of 64.
- `moe_shared_expert_intermediate_size`: two times `moe_ffn_hidden_size`, as in Nano.
- `mamba_num_heads`: the hidden size times 64 / 2688, to the nearest multiple of 8, so that the 8 Mamba groups
  divide it.
- `num_attention_heads`: the hidden size times 32 / 2688, to the nearest multiple of 8.

The top-k stays 6. The Mamba head size (64), the state size (128), the Mamba groups (8), the attention head size
(128) and the two key-value groups stay Nano's. `tests/unit_tests/test_nemotronh_size_configs.py` checks each file
against the rule and pins the parameter counts and FLOPs of each size.

## FLOPs

The FLOP estimator sizes the model from a run config's composed `model:` block when it is given the tokenizer's
vocabulary size in place of a Hugging Face model:

```bash
python3 scripts/nemotronh_flops_estimator.py run.yaml --vocab-size 100278
```
