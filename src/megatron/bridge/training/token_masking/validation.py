# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The opt-in held-out masked validation: a fixed set of marker-bearing data, evaluated at intervals.

``token_masking.masked_validation`` names the set: ``data_path``, a ``.bin/.idx`` prefix, for a run that trains on a
``GPTDatasetConfig``, or ``packed_data_path``, a packed parquet spec, for a packed SFT run. The set is the run's own
dataset config pointed at it, so it is read exactly as the training data is (sequence length, loss-mask rules, seed):

- setup builds it before the model, as the training data is built, ``iters * global_batch_size`` samples, through
  Megatron's ``BlendedMegatronDatasetBuilder`` (split ``1,0,0``, so the whole set) or the fine-tuning dataset builder;
- setup then reads those samples, as the evaluations read them, until one holds a target of a measured id that carries
  loss, and refuses a set whose samples hold none, since its listed-target loss would never be reported
  (``build_masked_validation``);
- the training loop evaluates it at step 0 of a fresh run and every ``interval`` iterations, each time from its first
  sample, with the run's forward step, so token masking applies exactly as in training, and logs the results under
  ``masked-validation/``. ``token_masking/listed_target_loss`` is computed before the mask, so it is the number to
  watch in both arms: it rises when masking and falls in an unmasked control.

The set is a ``HeldOutSet`` of ``megatron.bridge.training.held_out_validation``, which builds and evaluates it.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Iterable

import torch

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig, hf_dataset_root
from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig, GPTDatasetConfig
from megatron.bridge.training.held_out_validation import (
    HeldOutSet,
    indexed_set_config,
    on_data_parallel_rank,
    set_loader_factory,
)
from megatron.bridge.training.token_masking.config import TokenMaskingError
from megatron.bridge.training.token_masking.data_check import HELD_OUT_SET, no_trainable_target
from megatron.bridge.training.token_masking.hook import apply_token_masking
from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)

MASKED_VALIDATION_KEY_PREFIX = "masked-validation/"


def masked_validation_dataset_config(cfg: ConfigContainer) -> GPTDatasetConfig | FinetuningDatasetConfig | None:
    """The run's dataset config pointed at its held-out masked-validation set; None when the run evaluates none.

    A ``.bin/.idx`` set is a copy of the run's ``GPTDatasetConfig`` with ``blend=([data_path], None)`` and split
    ``1,0,0``; a packed set is a ``FinetuningDatasetConfig`` with the run's fields, ``packed_data_path`` as its packed
    training data and ``iters * global_batch_size`` samples, so the fine-tuning dataset builder (never the Hugging Face
    one, which would download) builds exactly that many. A Hugging Face run without ``dataset_root`` gets the root its
    own builder derives from ``dataset_name``.

    Raises:
        TokenMaskingError: when the set does not suit the run's data (a ``.bin/.idx`` set for a run that does not
            train on a ``GPTDatasetConfig``, a packed set for a run that does not train on packed sequences) or the
            packed set is not on disk.
    """
    block = cfg.token_masking.masked_validation
    if not block.evaluates:
        return None
    if block.data_path is not None:
        try:
            return indexed_set_config(cfg.dataset, block.data_path, "token_masking.masked_validation.data_path")
        except ValueError as error:
            packed = (
                ""
                if isinstance(cfg.dataset, GPTDatasetConfig)
                else "; a packed SFT run names its packs in packed_data_path"
            )
            raise TokenMaskingError(f"{error}{packed}") from error
    return _packed_config(cfg.dataset, block.packed_data_path, block.iters * cfg.train.global_batch_size)


def _packed_config(dataset: Any, packed_data_path: str, samples: int) -> FinetuningDatasetConfig:
    specs = dataset.packed_sequence_specs if isinstance(dataset, FinetuningDatasetConfig) else None
    if specs is None or specs.packed_sequence_size <= 0:
        raise TokenMaskingError(
            "token_masking.masked_validation.packed_data_path names packed held-out data, but the run does not train "
            f"on packed sequences ({type(dataset).__name__} without dataset.packed_sequence_specs); a .bin/.idx run "
            "names its held-out set in data_path"
        )
    try:
        specs = dataclasses.replace(specs, packed_train_data_path=packed_data_path)
    except (FileNotFoundError, ValueError) as error:
        raise TokenMaskingError(
            f"token_masking.masked_validation.packed_data_path={packed_data_path!r}: {error}"
        ) from error
    fields = {field.name: getattr(dataset, field.name) for field in dataclasses.fields(FinetuningDatasetConfig)}
    fields.update(packed_sequence_specs=specs, max_train_samples=samples, do_validation=False, do_test=False)
    if isinstance(dataset, HFDatasetConfig):
        fields.update(dataset_root=hf_dataset_root(dataset.dataset_root, dataset.dataset_name))
    return FinetuningDatasetConfig(**fields)


def _named_set(cfg: ConfigContainer) -> str:
    block = cfg.token_masking.masked_validation
    if block.data_path is not None:
        return f"token_masking.masked_validation.data_path={block.data_path!r}"
    return f"token_masking.masked_validation.packed_data_path={block.packed_data_path!r}"


def _measured_targets(
    loader: Iterable[dict[str, Any]], samples: int, resolved: ResolvedTokenMasking
) -> tuple[int, int, int]:
    """The targets of a measured id that ``loader``'s first ``samples`` samples hold, read only until one carries loss.

    Each batch's labels and loss mask go through ``apply_token_masking`` on the training device, as the forward step
    passes them, so a target counts as carrying loss exactly when an evaluation reports it in
    ``token_masking/listed_trainable_target_fraction``.

    Returns:
        ``(listed, trainable, read)``: the targets of a measured id found, how many of them carry loss, and the
        samples read (fewer than ``samples`` when a batch before the last holds a target that carries loss).
    """
    device = resolved.ids_tensor.device
    listed = trainable = read = 0
    for batch in loader:
        _, stats = apply_token_masking(batch["labels"].to(device), batch["loss_mask"].to(device), resolved)
        listed += int(stats.listed_targets.item())
        trainable += int(stats.would_train.sum().item())
        read += len(batch["labels"])
        if trainable or read >= samples:
            break
    return listed, trainable, read


def _require_trainable_measured_targets(
    cfg: ConfigContainer,
    new_loader: Callable[..., Iterable[dict[str, Any]]],
    samples: int,
    resolved: ResolvedTokenMasking,
) -> None:
    """Refuse, on every rank, a set whose evaluated samples hold no target of a measured id that carries loss.

    An evaluation reads the set's first ``samples`` samples, each data-parallel rank its share. The last rank reads
    them in order through a loader that gives it every share, stopping at the first batch that holds a target that
    carries loss (the other ranks wait meanwhile), and the verdict is broadcast, so every rank stops together.
    """
    extent = f"the {samples} samples each evaluation reads"
    check_rank = torch.distributed.get_world_size() - 1
    outcome: list[str | None] = [None]
    if torch.distributed.get_rank() == check_rank:
        try:
            listed, trainable, read = _measured_targets(
                new_loader(data_parallel_rank=0, data_parallel_size=1), samples, resolved
            )
            if trainable:
                logger.info(
                    f"[masked-validation] {_named_set(cfg)}: {trainable} targets of the ids "
                    f"{list(resolved.measured_token_ids)} carry loss in the first {read} of {extent}"
                )
            else:
                outcome[0] = no_trainable_target(HELD_OUT_SET, resolved, listed, extent)
        except Exception as error:  # broadcast below, then raised on every rank
            outcome[0] = f"reading the set failed on rank {check_rank}: {type(error).__name__}: {error}"
            logger.exception("[masked-validation] reading the held-out set failed")
    torch.distributed.broadcast_object_list(outcome, src=check_rank)
    if outcome[0] is not None:
        raise TokenMaskingError(
            f"{_named_set(cfg)} cannot be the held-out masked-validation set, which must hold targets of the measured "
            f"ids that carry loss: {outcome[0]}"
        )


def build_masked_validation(
    cfg: ConfigContainer, tokenizer: Any, resolved: ResolvedTokenMasking, dp_group: torch.distributed.ProcessGroup
) -> HeldOutSet | None:
    """Build the held-out masked-validation set on every rank as the training data is built, and check it; None
    when the run evaluates none.

    The set holds ``iters * global_batch_size`` samples (``set_loader_factory``). It is evaluated at step 0 of a fresh
    run and every ``interval`` iterations.

    Raises:
        TokenMaskingError: on every rank, when the set does not suit the run's data, the packed set is not on disk,
            the samples an evaluation reads hold no target of a measured id that carries loss, or reading them fails.
    """
    dataset_config = masked_validation_dataset_config(cfg)
    if dataset_config is None:
        return None
    block = cfg.token_masking.masked_validation
    samples = block.iters * cfg.train.global_batch_size
    print_rank_0(f"> building the masked-validation set {_named_set(cfg)} ({samples} samples) ...")
    new_loader = set_loader_factory(cfg, dataset_config, samples, tokenizer)
    _require_trainable_measured_targets(cfg, new_loader, samples, resolved)
    print_rank_0(
        f"> masked validation: {block.iters} batches of {cfg.train.global_batch_size} samples at step 0 of a fresh "
        f"run and every {block.interval} iterations, logged under {MASKED_VALIDATION_KEY_PREFIX}"
    )
    return HeldOutSet(
        key_prefix=MASKED_VALIDATION_KEY_PREFIX,
        iters=block.iters,
        interval=block.interval,
        at_start=True,
        at_exit=False,
        new_loader=on_data_parallel_rank(new_loader, dp_group),
    )
