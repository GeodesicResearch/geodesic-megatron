# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Held-out sets that the training loop evaluates at fixed steps, each read exactly as the training data is.

A set is the run's own dataset config pointed at held-out data. The run therefore reads it with its own sequence
length, loss-mask rules and seed. Setup builds each set before the model, as it builds the training data. The training
loop evaluates a set forward-only with the run's forward step, one global batch per evaluation step. Every evaluation
starts at the set's first sample. An evaluation leaves ``train_state.consumed_valid_samples`` alone. It fires no
evaluation callbacks, which belong to the validation set.

Two blocks name sets. This module builds the sets of ``held_out_validation`` (``build_held_out_validation``).
``megatron.bridge.training.token_masking.validation`` builds token masking's held-out masked-validation set.

Catalogue:
    HELD_OUT_KEY_PREFIX: the prefix of the logged results of every ``held_out_validation`` set.
    HeldOutSet: one built set and the steps the training loop evaluates it at.
    indexed_set_config: the run's ``GPTDatasetConfig`` pointed at a ``.bin/.idx`` set.
    set_loader_factory: build a set's dataset and return a factory of data loaders over it.
    on_data_parallel_rank: a loader factory bound to this rank's share of a data-parallel group.
    build_held_out_validation: build the sets of ``held_out_validation`` on every rank.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable

import torch
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer import MegatronModule

from megatron.bridge.data.loaders import build_data_loader
from megatron.bridge.data.utils import (
    finetuning_train_valid_test_datasets_provider,
    pretrain_train_valid_test_datasets_provider,
)
from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig, GPTDatasetConfig
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.state import GlobalState
from megatron.bridge.utils.common_utils import print_rank_0


HELD_OUT_KEY_PREFIX = "held-out/"


@dataclass(frozen=True)
class HeldOutSet:
    """A built held-out set, and when and how the training loop evaluates it."""

    key_prefix: str
    """The prefix of the set's logged results, ending in a slash."""
    iters: int
    """Batches, each of the training global batch size, per evaluation."""
    interval: int | None
    """Evaluate after every step that is a multiple of ``interval``. None: no interval evaluations."""
    at_start: bool
    """Evaluate at step 0 of a fresh run."""
    at_exit: bool
    """Evaluate after the step at which the run ends through ``train_iters`` or ``exit_interval``."""
    new_loader: Callable[[], Iterable[Any]]
    """A data loader over the set that starts at its first sample. Its sampler advances as it is read (the batch
    sampler counts what it yields), so every evaluation takes a new one."""

    def due(self, step: int, final: bool) -> bool:
        """Whether the set is evaluated after the iteration that brings the run to ``step``.

        ``final`` is True when the run ends at ``step`` through ``train_iters`` or ``exit_interval``.
        """
        return (self.interval is not None and step % self.interval == 0) or (self.at_exit and final)

    def data_iterator(self) -> RerunDataIterator:
        """An iterator over the set from its first sample, so every evaluation reads the same samples."""
        return RerunDataIterator(iter(self.new_loader()))

    def evaluate(
        self,
        state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        model_config: Any,
    ) -> None:
        """Evaluate the set on every rank and log the results under ``key_prefix``."""
        evaluate_and_print_results(
            state,
            f"iteration {state.train_state.step}",
            forward_step_func,
            self.data_iterator(),
            model,
            model_config,
            verbose=False,
            write_to_tensorboard=True,
            eval_iters=self.iters,
            advance_consumed_valid_samples=False,
            key_prefix=self.key_prefix,
        )


def indexed_set_config(dataset: Any, data_path: str, field: str) -> GPTDatasetConfig:
    """The run's ``GPTDatasetConfig`` with ``blend=([data_path], None)`` and split ``1,0,0``, so it reads the whole set.

    Raises:
        ValueError: naming ``field``, when the run does not train on a ``GPTDatasetConfig`` or trains on mock data.
    """
    if not isinstance(dataset, GPTDatasetConfig):
        raise ValueError(f"{field} names a .bin/.idx set, but the run trains on a {type(dataset).__name__}")
    if dataset.mock:
        raise ValueError(
            f"{field} names a .bin/.idx set, but the run trains on mock data (a MockGPTDataset reads no corpus); "
            "train on a corpus or remove the held-out set"
        )
    config = copy.copy(dataset)
    config.blend = ([data_path], None)
    config.blend_per_split = None
    config.split = "1,0,0"
    # fast_cache_load reads the caches of a blend_per_split and refuses a blend.
    config.fast_cache_load = False
    config.finalize()
    return config


def set_loader_factory(
    cfg: ConfigContainer,
    dataset_config: GPTDatasetConfig | FinetuningDatasetConfig,
    samples: int,
    tokenizer: Any,
) -> Callable[..., Iterable[Any]]:
    """Build the set's dataset of ``samples`` samples and return a factory of loaders that read it from its start.

    The factory takes ``data_parallel_rank`` and ``data_parallel_size``. Megatron's ``BlendedMegatronDatasetBuilder``
    can build a few more samples for a ``.bin/.idx`` set, and an evaluation reads exactly the first ``samples``. A run
    whose data loader is ``batch`` reads a global batch per rank per evaluation step, as ``evaluate`` expects. Other
    runs read micro-batches. A ``.bin/.idx`` set writes its index caches where the
    training data writes its own: ``dataset.path_to_cache``, else beside the data.
    """
    if isinstance(dataset_config, FinetuningDatasetConfig):
        dataset, _, _ = finetuning_train_valid_test_datasets_provider([samples, 0, 0], dataset_config, tokenizer)
    else:
        dataset, _, _ = pretrain_train_valid_test_datasets_provider([samples, 0, 0], dataset_config)
    return partial(
        build_data_loader,
        dataset,
        0,
        "batch" if cfg.dataset.dataloader_type == "batch" else "single",
        cfg,
        # Every evaluation reads through a new loader, whose workers end with it.
        persistent_workers=False,
    )


def on_data_parallel_rank(
    new_loader: Callable[..., Iterable[Any]], dp_group: torch.distributed.ProcessGroup
) -> Callable[[], Iterable[Any]]:
    """``new_loader`` bound to this rank's share of ``dp_group``."""
    return partial(
        new_loader,
        data_parallel_rank=torch.distributed.get_rank(group=dp_group),
        data_parallel_size=torch.distributed.get_world_size(group=dp_group),
    )


def build_held_out_validation(
    cfg: ConfigContainer, tokenizer: Any, dp_group: torch.distributed.ProcessGroup
) -> list[HeldOutSet]:
    """Build every set of ``cfg.held_out_validation`` on every rank, as the training data is built.

    Each set reads ``samples`` samples per evaluation and logs its results under ``held-out/<name>/``. A run without
    sets gets an empty list.

    Raises:
        ValueError: when the run does not train on a ``GPTDatasetConfig`` or trains on mock data.
    """
    block = cfg.held_out_validation
    sets = []
    for held in block.sets:
        dataset_config = indexed_set_config(cfg.dataset, held.data_path, f"held_out_validation set {held.name!r}")
        print_rank_0(f"> building the held-out set {held.name} from {held.data_path} ({held.samples} samples) ...")
        new_loader = set_loader_factory(cfg, dataset_config, held.samples, tokenizer)
        sets.append(
            HeldOutSet(
                key_prefix=f"{HELD_OUT_KEY_PREFIX}{held.name}/",
                iters=held.samples // cfg.train.global_batch_size,
                interval=block.interval,
                at_start=block.at_start,
                at_exit=block.at_exit,
                new_loader=on_data_parallel_rank(new_loader, dp_group),
            )
        )
    if sets:
        print_rank_0(
            f"> held-out validation: {len(sets)} sets, at_start={block.at_start}, interval={block.interval}, "
            f"at_exit={block.at_exit}, logged under {HELD_OUT_KEY_PREFIX}<name>/"
        )
    return sets
