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

The evaluation itself is ``training.held_out.evaluation.HeldOutEvaluation``, which the ``held_out:`` block's sets use
too; this module adds what is token masking's alone: the config of the set and the check of its samples.
"""

from __future__ import annotations

import copy
import logging
from functools import partial
from typing import Any, Callable, Iterable

import torch

from megatron.bridge.data.utils import (
    finetuning_train_valid_test_datasets_provider,
    pretrain_train_valid_test_datasets_provider,
)
from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig, GPTDatasetConfig
from megatron.bridge.training.held_out.config import HeldOutError
from megatron.bridge.training.held_out.evaluation import HeldOutEvaluation, loader_builder, packed_set_config
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
        return _indexed_config(cfg.dataset, block.data_path)
    return _packed_config(cfg.dataset, block.packed_data_path, block.iters * cfg.train.global_batch_size)


def _indexed_config(dataset: Any, data_path: str) -> GPTDatasetConfig:
    if not isinstance(dataset, GPTDatasetConfig):
        raise TokenMaskingError(
            f"token_masking.masked_validation.data_path names a .bin/.idx set, but the run trains on a "
            f"{type(dataset).__name__}; a packed SFT run names its held-out packs in packed_data_path"
        )
    if dataset.mock:
        raise TokenMaskingError(
            "token_masking.masked_validation.data_path names a .bin/.idx set, but the run trains on mock data (a "
            "MockGPTDataset reads no corpus); train on a corpus or remove the held-out set"
        )
    config = copy.copy(dataset)
    config.blend = ([data_path], None)
    config.blend_per_split = None
    config.split = "1,0,0"
    # fast_cache_load reads the caches of a blend_per_split and refuses a blend.
    config.fast_cache_load = False
    config.finalize()
    return config


def _packed_config(dataset: Any, packed_data_path: str, samples: int) -> FinetuningDatasetConfig:
    try:
        return packed_set_config(dataset, packed_data_path, samples)
    except HeldOutError as error:
        raise TokenMaskingError(f"token_masking.masked_validation: {error}") from error


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
) -> HeldOutEvaluation | None:
    """Build the held-out masked-validation set on every rank as the training data is built, and check it; None
    when the run evaluates none.

    Exactly ``iters * global_batch_size`` samples are built (``BlendedMegatronDatasetBuilder`` may build a few more
    for a ``.bin/.idx`` set; an evaluation reads exactly the first that many). The loader reads them in order: a
    global batch per rank per evaluation step for a run whose data loader is ``batch`` (as ``evaluate`` expects of
    such a run), micro-batches otherwise.

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
    if isinstance(dataset_config, FinetuningDatasetConfig):
        dataset, _, _ = finetuning_train_valid_test_datasets_provider([samples, 0, 0], dataset_config, tokenizer)
    else:
        dataset, _, _ = pretrain_train_valid_test_datasets_provider([samples, 0, 0], dataset_config)
    new_loader = loader_builder(cfg, dataset, drop_last=True)
    _require_trainable_measured_targets(cfg, new_loader, samples, resolved)
    print_rank_0(
        f"> masked validation: {block.iters} batches of {cfg.train.global_batch_size} samples at step 0 of a fresh "
        f"run and every {block.interval} iterations, logged under {MASKED_VALIDATION_KEY_PREFIX}"
    )
    return HeldOutEvaluation(
        key_prefix=MASKED_VALIDATION_KEY_PREFIX,
        interval=block.interval,
        iters=block.iters,
        new_loader=partial(
            new_loader,
            data_parallel_rank=torch.distributed.get_rank(group=dp_group),
            data_parallel_size=torch.distributed.get_world_size(group=dp_group),
        ),
    )
