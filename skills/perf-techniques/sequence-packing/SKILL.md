---
name: sequence-packing
description: Operational guide for enabling packed sequences and long-context config paths in Megatron-Bridge, including config knobs, code anchors, pitfalls, and verification.
---

# Sequence Packing Skill

For stable background and recommendation level, see:

- `docs/training/packed-sequences.md`
- `card.yaml` (co-located)

## Enablement

Offline packed SFT for LLM finetuning:

```python
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

cfg.train.micro_batch_size = 1
cfg.dataset.seq_length = 4096
cfg.model.seq_length = 4096
cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
cfg.dataset.packed_sequence_specs = PackedSequenceSpecs(
    packed_sequence_size=4096,
    pad_seq_to_mult=1,
)
```

If CP is enabled:

```python
cfg.model.context_parallel_size = 2
cfg.model.calculate_per_token_loss = True
cfg.ddp.average_in_collective = False
cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2
```

If CUDA graphs are enabled for this packed path:

```python
cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
cfg.dataset.dataset_kwargs["pad_to_max_length"] = True
```

**Note:** `pad_cu_seqlens = True` also requires a metadata JSON file alongside
the packed dataset (asserted in `src/megatron/bridge/data/datasets/sft.py`).
Custom packed datasets that omit the metadata file will hit an assertion at
dataset initialization.

In-batch packing for VLM finetuning:

```python
cfg.dataset.pack_sequences_in_batch = True
cfg.train.micro_batch_size = 2
```

Long-context baseline:

```python
cfg.model.seq_length = 16384
cfg.dataset.seq_length = 16384
cfg.model.context_parallel_size = 2
```

## Code Anchors

LLM packed SFT config surface:

```72:97:src/megatron/bridge/recipes/utils/finetune_utils.py
if packed_sequence:
    dataset_kwargs = {"pad_to_max_length": True}
    packed_sequence_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)
else:
    dataset_kwargs = {}
    packed_sequence_specs = None
```

Bridge validation:

```1617:1657:src/megatron/bridge/training/config.py
if self.model.context_parallel_size > 1:
    assert self.model.seq_length % (self.model.context_parallel_size * 2) == 0, ...
    if isinstance(self.dataset, FinetuningDatasetConfig):
        assert self.model.calculate_per_token_loss, ...
        assert not self.ddp.average_in_collective, ...
...
if ... packed_sequence_size > 0 and self.train.micro_batch_size > 1:
    raise ValueError(...)
...
if getattr(self.dataset, "pack_sequences_in_batch", False) and self.train.micro_batch_size == 1:
    raise ValueError(...)
```

VLM in-batch runtime:

```308:327:src/megatron/bridge/training/vlm_step.py
if enable_packing:
    ...
    ) = pack_batch_sequences(
        ...
        pad_token_id=0,
        pad_to_multiple_of=cp_size * 2 if cp_size > 1 else 1,
    )
```

Packed THD runtime constraint:

```98:99:src/megatron/bridge/training/gpt_step.py
if cu_seqlens.dim() > 1 and cu_seqlens.size(0) != 1:
    raise ValueError("Packed THD batches expect micro-batch size 1 for context-parallel slicing (THD layout)")
```

## Pitfalls

1. Offline packed SFT and VLM in-batch packing are different features with opposite micro-batch rules.
2. When CP is enabled, packed sequence lengths must respect `2 * context_parallel_size` divisibility.
3. For finetuning with CP, `calculate_per_token_loss=True` and `ddp.average_in_collective=False` are required.
4. `pad_cu_seqlens=True` also requires `pad_to_max_length=True`.
5. Packing support is model-family-specific. `Qwen3-Next`, `GLM-4.5`, and `Qwen3.5-VL` contain explicit opt-outs in different paths.
6. MTP finetuning is documented as incompatible with packed sequences.
7. `comm_overlap.overlap_moe_expert_parallel_comm` cannot be combined with packed sequences: `gpt_step._forward_step_common` builds the EP-overlap schedule plan without `packed_seq_params` and raises `ValueError`.
8. A collated packed batch pads every `cu_seqlens` row with -1 to the widest row plus one, so everything that reads the packed layout trims it the same way, with `packed_seq_utils.trim_padded_cu_seqlens`: attention and the Mamba layers get the trimmed row from `get_packed_seq_params`, and the context-parallel partition (`gpt_step._partition_packed_batch_for_cp`) trims before Transformer Engine's `thd_get_partitioned_indices`, whose search can otherwise land on a pad when a row ends in two or more and hand every CP rank the pack's leading tokens. `tests/unit_tests/training/test_gpt_step_packed_cp_partition.py` runs the real kernel (GPU only). Packed runs at CP>1 with more than one pack per data-parallel replica that were launched before the partition trimmed (2026-10-01) trained on partly mis-partitioned microbatches: 26–29% of them for the Nano control-pretraining and metagaming SFTs at CP=2, about 77% for the Super-120B SFTs at CP=4 (`configs/pa_warm_start/`, the Super SFT quickstart), and the green-team 32K math/science SFTs at CP=4 (`configs/PA/green-team/32k/`) are affected too. The Nano CP=2 SFTs logged a loss about 0.03–0.04 nats below what the same run logs with the trim (measured by replaying one of them); the CP=4 runs' offset was not measured and is expected to be larger. Loss figures from either side of the trim are not compared (`/projects/a5k/public/logs/nano_sft_perf_campaign/records/cp_partition_fix/FINAL_REPORT.md`, section 2, and `gap-cp2-thd-partition-leak.md` beside it).
9. Pad tokens reach the MoE router unless the model is given a padding mask, and then count in its expert-bias update (a fixed step every iteration, whatever the learning rate) and its auxiliary losses. Anything that changes how much a batch is padded then moves the training trajectory, for example `pad_to_max_length`, the number of packs collated per data-parallel replica, or the context-parallel size. The packed collate (`GPTSFTPackedDataset.collate_fn`) therefore emits `padding_mask`: True at each document's EOS padding, as `cu_seqlens_unpadded` excludes it, and after a pack's last document. `gpt_step` keeps the mask on every pipeline stage, partitions it with the tokens, gives each stage its sequence-parallel share of it (`_prepare_packed_padding_mask`, as upstream Megatron-Bridge does) and passes it to the model. The pinned Megatron-LM carries upstream's fix 0006 (`3rdparty/patches/megatron-lm/README.md`), without which a router with expert bias (every Nemotron-H model's) fails at the first training step on a masked batch. Runs launched before the mask (2026-10-01) counted their pads: on the Nano XL SFT each extra pad token per iteration offset the loss by about −1.1 × 10⁻⁸ nats, flat from about iteration 100 on (`docs/investigations/nano30b-sft-perf-campaign.md`, E-007).
10. The pinned Megatron-Core cannot apply that padding mask inside a MoE router a Transformer Engine CUDA graph has captured, so, as upstream Megatron-Bridge does, `gpt_step` refuses packed batches for a MoE model whose `cuda_graph_impl='transformer_engine'` graphs capture the whole layer (no `cuda_graph_modules`) or name `moe`, `moe_router` or `moe_preprocess`, raising `ValueError` before the forward. Graphs that capture other modules only, and the `local` and `full_iteration` implementations, run. No shipped SFT or PEFT recipe enables CUDA graphs: the Super and Ultra recipes set them for pretraining only, whose batches carry no padding mask.

## Verification

Use the checked-in unit coverage:

```bash
uv run python -m pytest tests/unit_tests/training/utils/test_packed_seq_utils.py -v && \
uv run python -m pytest tests/unit_tests/training/test_config.py -k "packed_sequence or pack_sequences_in_batch or context_parallel_seq_length_divisibility or context_parallel_finetuning_validations" -v && \
uv run python -m pytest tests/unit_tests/training/test_vlm_step.py -k "enable_packing" -v
```

Success criteria:

- first command reports `11 passed`
- second command reports `14 passed`
- third command reports `2 passed`
