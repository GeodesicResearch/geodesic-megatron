# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A held-out set at run time: when the training loop evaluates it, how it is read, and the builder of the packed
sets that ``held_out.sets`` names.

A set is the run's own dataset config pointed at the packed parquet spec, so the set is read exactly as the training
data is (sequence length, loss-mask rules). Setup builds every set before the model. An evaluation of a ``held_out``
set reads each pack once: its last global batch is padded with samples that carry no loss, so a set whose size is
not a multiple of the global batch size is neither cut short nor read twice, and the mean loss is over its own loss
tokens alone.

Catalogue:
    HELD_OUT_KEY_PREFIX: the prefix of the logged results of a ``held_out`` set; the set's name follows.
    HeldOutEvaluation: one set, and when and how the training loop evaluates it.
    packed_set_config: the run's packed dataset config pointed at another packed set.
    loader_builder: a function that builds a loader over a dataset from its first sample, for one data-parallel rank.
    build_held_out_evaluations: the evaluations of the sets that the run's ``held_out`` block names.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable

import numpy as np
import torch
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer import MegatronModule

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig, hf_dataset_root
from megatron.bridge.data.loaders import build_data_loader
from megatron.bridge.data.utils import finetuning_train_valid_test_datasets_provider
from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.held_out.config import HeldOutError
from megatron.bridge.training.state import GlobalState
from megatron.bridge.utils.common_utils import print_rank_0


HELD_OUT_KEY_PREFIX = "held-out/"


@dataclass(frozen=True)
class HeldOutEvaluation:
    """One held-out set, and when and how the training loop evaluates it."""

    key_prefix: str
    """Prepended to every logged result of the set, for example ``held-out/<name>/``."""
    interval: int
    """Evaluate at step 0 of a fresh run and at every step that is a multiple of ``interval``."""
    iters: int
    """Batches, each of the training global batch size, per evaluation."""
    new_loader: Callable[[], Iterable[Any]]
    """A data loader over the set that starts at its first sample. Its sampler advances as it is read (the batch
    sampler counts what it yields), so every evaluation takes a new one."""

    def due(self, step: int) -> bool:
        """Whether the set is evaluated after the iteration that brings the run to ``step``."""
        return step % self.interval == 0

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
        """Evaluate the set with the run's forward step and log the results under ``key_prefix``.

        The evaluation leaves ``train_state.consumed_valid_samples`` alone (that counter positions the validation
        set's sampler on resume) and fires no evaluation callbacks, which belong to the validation set.
        """
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


def packed_set_config(dataset: Any, packed_data_path: str, samples: int | None) -> FinetuningDatasetConfig:
    """The run's packed dataset config pointed at ``packed_data_path`` as its packed training data.

    The config has the run's fields, builds only that set (no validation or test split) and, with ``samples``,
    exactly that many samples (the pack list repeated and cut); with None it reads the whole file once, each pack
    one sample. The fine-tuning dataset builder builds it, never the Hugging Face one, which would download. A
    Hugging Face run without ``dataset_root`` gets the root its own builder derives from ``dataset_name``.

    Raises:
        HeldOutError: the run does not train on packed sequences, or the packed set is not on disk.
    """
    specs = dataset.packed_sequence_specs if isinstance(dataset, FinetuningDatasetConfig) else None
    if specs is None or specs.packed_sequence_size <= 0:
        raise HeldOutError(
            f"the run does not train on packed sequences ({type(dataset).__name__} without "
            "dataset.packed_sequence_specs)"
        )
    try:
        specs = dataclasses.replace(specs, packed_train_data_path=packed_data_path)
    except (FileNotFoundError, ValueError) as error:
        raise HeldOutError(f"packed_data_path={packed_data_path!r}: {error}") from error
    fields = {field.name: getattr(dataset, field.name) for field in dataclasses.fields(FinetuningDatasetConfig)}
    fields.update(packed_sequence_specs=specs, max_train_samples=samples, do_validation=False, do_test=False)
    if isinstance(dataset, HFDatasetConfig):
        fields.update(dataset_root=hf_dataset_root(dataset.dataset_root, dataset.dataset_name))
    return FinetuningDatasetConfig(**fields)


def loader_builder(cfg: ConfigContainer, dataset: Any, drop_last: bool) -> Callable[..., Iterable[Any]]:
    """A function that builds a loader over ``dataset`` from its first sample, as the run builds its own loaders.

    The function takes ``data_parallel_rank`` and ``data_parallel_size`` and gives each rank its share of the
    batches. A run whose loader type is ``batch`` reads through its batch sampler, which with ``drop_last`` False pads
    the last batch to the global batch size with samples that carry no loss. A run with any other loader type reads in
    order through the sequential sampler, which drops a last incomplete batch with ``drop_last`` True and yields it
    short and unpadded with False. Every loader's workers end with it, because every evaluation takes a new one.
    """
    # The training types ``cyclic`` (random order) and ``external`` (no sampler) have no meaning for a fixed set that
    # is read from its first sample, so a run with one of them reads the set in order, as ``single`` does.
    return partial(
        build_data_loader,
        dataset,
        0,
        "batch" if cfg.dataset.dataloader_type == "batch" else "single",
        cfg,
        persistent_workers=False,
        drop_last=drop_last,
    )


def build_held_out_evaluations(
    cfg: ConfigContainer, tokenizer: Any, dp_group: torch.distributed.ProcessGroup
) -> list[HeldOutEvaluation]:
    """Build the held-out sets that the run's ``held_out`` block names, in the order it names them; none for a run
    that names none.

    Every set is built on every rank as the training data is, before the model. Its evaluation takes as many global
    batches as hold its packs, the last one padded. Setup prints each set's packs, loss tokens and batches.

    Raises:
        HeldOutError: the run's data loader is not ``batch`` (a packed fine-tuning run's), the run does not train on
            packed sequences, or a packed set cannot be built or holds no pack. The error of a set names it, as
            ``held_out.sets.<name>``.
    """
    block = cfg.held_out
    if not block.sets:
        return []
    if cfg.dataset.dataloader_type != "batch":
        raise HeldOutError(
            f"held_out.sets names packed sets, which a run with dataset.dataloader_type={cfg.dataset.dataloader_type!r} "
            "cannot pad to whole global batches; a packed fine-tuning run's loader type is 'batch'"
        )
    evaluations = []
    for name, packed_data_path in block.sets.items():
        try:
            config = packed_set_config(cfg.dataset, packed_data_path, samples=None)
        except HeldOutError as error:
            raise HeldOutError(f"held_out.sets.{name}: {error}") from error
        # The fine-tuning builder reads the whole file; the sample counts it is given are not used.
        try:
            dataset, _, _ = finetuning_train_valid_test_datasets_provider([0, 0, 0], config, tokenizer)
        except ValueError as error:
            raise HeldOutError(f"held_out.sets.{name}={packed_data_path!r}: {error}") from error
        packs = len(dataset)
        if packs == 0:
            raise HeldOutError(f"held_out.sets.{name}={packed_data_path!r} holds no pack")
        iters = math.ceil(packs / cfg.train.global_batch_size)
        loss_tokens = sum(int(np.asarray(dataset[index]["loss_mask"]).sum()) for index in range(packs))
        print_rank_0(
            f"> held-out set {name}: {packs} packs with {loss_tokens} loss tokens, evaluated in {iters} batches of "
            f"{cfg.train.global_batch_size} samples at step 0 of a fresh run and every {block.interval} steps, "
            f"logged under {HELD_OUT_KEY_PREFIX}{name}/"
        )
        builder = loader_builder(cfg, dataset, drop_last=False)
        evaluations.append(
            HeldOutEvaluation(
                key_prefix=f"{HELD_OUT_KEY_PREFIX}{name}/",
                interval=block.interval,
                iters=iters,
                new_loader=partial(
                    builder,
                    data_parallel_rank=torch.distributed.get_rank(group=dp_group),
                    data_parallel_size=torch.distributed.get_world_size(group=dp_group),
                ),
            )
        )
    return evaluations
