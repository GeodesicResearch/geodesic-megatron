# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A training run's ``.bin/.idx`` data, built on CPU as its launch builds it.

The config is resolved by the launcher's own merge (``pipeline_training_run.resolve_bin_idx_run_config``), the
loader's training-data window (``get_train_data_window``) sizes the training dataset at the run's resumed step, and the
loader's builder builds it. No process group, GPU or checkpoint is needed, and the index caches written are the ones the
launch would write, so the launch then finds them warm. The tools that read a run's data before it trains
(``report_blend_coverage.py``, ``predict_masked_counts.py``, ``held_out_documents.py``) build it here.

The resumed step is the config's ``checkpoint.ckpt_step`` (the run's start when unset: a resume from the load
directory's latest save is not recognised), and the samples consumed before it are that step times the global batch,
so a batch-size ramp is refused.

What a built dataset reads is told here too, for those tools: which samples of it the run's data-parallel ranks read
and at which iteration (``samples_read``, from each rank's own loader), which of each corpus's own samples those are
(``corpus_draws``), which of a corpus's sequences they reach (``sequences_reached``), and the iteration at which
the run first reads each of a corpus's documents (``documents_first_read``).
"""

from __future__ import annotations

import sys
from itertools import chain, islice
from pathlib import Path

import numpy
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.num_microbatches_calculator import ConstantNumMicroBatchesCalculator

from megatron.bridge.data.loaders import build_data_loader, build_train_valid_test_datasets, get_train_data_window
from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
from megatron.bridge.training.state import TrainState


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.training.launch_width import LaunchWidthError, config_launch_width  # noqa: E402

import pipeline_training_run  # noqa: E402


BIN_IDX_MODES = pipeline_training_run.BIN_IDX_MODES
MODELS = sorted({model for model, _ in pipeline_training_run.RECIPE_MAP})


def resolve_run_config(config_file: str, model: str, mode: str):
    """The config a launch of ``config_file`` trains with (``resolve_bin_idx_run_config``), refusing a batch-size
    ramp, under which the samples a resumed run consumed cannot be computed."""
    cfg = pipeline_training_run.resolve_bin_idx_run_config(config_file, model, mode)
    if cfg.train.rampup_batch_size is not None:
        raise ValueError("a batch-size ramp makes the samples consumed before the resumed step unknowable here")
    return cfg


def resumed_step(cfg) -> int:
    """The step the run starts from: ``checkpoint.ckpt_step``, or 0 for a run from its start."""
    return cfg.checkpoint.ckpt_step or 0


def run_inputs(cfg) -> dict:
    """The resolved settings a report on the run's data depends on, so it stays traceable after its config changes."""
    return {
        "data_path": [str(item) for item in cfg.dataset.data_path],
        "seq_length": cfg.dataset.seq_length,
        "split": cfg.dataset.split,
        "seed": cfg.dataset.random_seed,
        "path_to_cache": cfg.dataset.path_to_cache,
        "train_iters": cfg.train.train_iters,
        "train_samples": cfg.train.train_samples,
        "global_batch_size": cfg.train.global_batch_size,
        "ckpt_step": cfg.checkpoint.ckpt_step,
        "reset_data_position": cfg.checkpoint.reset_data_position,
        "tokenizer_model": cfg.tokenizer.tokenizer_model,
    }


def build_training_dataset(cfg) -> tuple[object, int, int]:
    """The training dataset a launch of the resolved config ``cfg`` builds.

    Returns:
        The dataset, its size in samples, and the first sample the run reads.
    """
    state = TrainState()
    state.step = resumed_step(cfg)
    state.consumed_train_samples = state.step * cfg.train.global_batch_size
    size, first_sample = get_train_data_window(cfg, state)
    train_ds, _, _ = build_train_valid_test_datasets(
        cfg, pretrain_train_valid_test_datasets_provider, train_samples=size
    )
    return train_ds, size, first_sample


def microbatches_per_replica(global_batch_size: int, micro_batch_size: int, data_parallel: int) -> int:
    """The microbatches each data-parallel replica runs per iteration, counted by the training loop's own calculator
    (Megatron's ``ConstantNumMicroBatchesCalculator``, without decreasing the batch), which refuses a global batch the
    replicas cannot share evenly."""
    return ConstantNumMicroBatchesCalculator(
        global_batch_size, micro_batch_size, data_parallel, decrease_batch_size_if_needed=False, rank=0
    ).get()


def data_parallel_size(config_file: str, cfg, gpus: int) -> int:
    """The data-parallel size the run of ``config_file`` trains at on ``gpus`` GPUs.

    A config whose ``launch_width:`` block fixes its width trains at that width only, so any other GPU count, and a
    parallelism that gives another data-parallel size there, raises ``LaunchWidthError``: what the run reads at
    another width is not what that run reads.
    """
    width = config_launch_width(config_file)
    if width is not None and gpus != width.world_size:
        raise LaunchWidthError(
            f"{config_file} trains on {width.world_size} GPUs (its launch_width block), not {gpus}; what it reads at "
            "any other width is not that run's"
        )
    data_parallel = cfg.get_data_parallel_size(gpus)
    if width is not None and data_parallel != width.data_parallel_size:
        raise LaunchWidthError(
            f"{config_file} trains at data-parallel size {width.data_parallel_size} (its launch_width block), not the "
            f"{data_parallel} its parallelism gives on {gpus} GPUs"
        )
    return data_parallel


def data_parallel_loaders(cfg, train_ds, consumed: int, data_parallel: int) -> list:
    """Every data-parallel rank's loader over ``train_ds`` from the sample ``consumed``, each built by the loader's
    own builder (``build_data_loader``) as the run builds its training loader on that rank."""
    return [
        build_data_loader(
            train_ds,
            consumed,
            cfg.dataset.dataloader_type,
            cfg,
            data_parallel_rank=rank,
            data_parallel_size=data_parallel,
            persistent_workers=False,
        )
        for rank in range(data_parallel)
    ]


def samples_read(cfg, train_ds, first_sample: int, data_parallel: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Every sample of ``train_ds`` the run reads from its resumed step to its last iteration, and the iteration that
    reads it (as the log numbers iterations: the first of a run from its start is 1).

    Each data-parallel rank's own sampler names the samples of its microbatches, and a rank runs the run's
    microbatches per replica every iteration, so its sampler's first ``iterations x microbatches`` batches are what
    the rank reads, iteration after iteration. Only the indices are read, never a sample's tokens. A sampler that
    ends before then is refused: the run would read past one pass of its loader, which no single pass of its sampler
    describes.
    """
    step = resumed_step(cfg)
    iterations = cfg.train.train_iters - step
    microbatches = microbatches_per_replica(cfg.train.global_batch_size, cfg.train.micro_batch_size, data_parallel)
    per_rank = iterations * microbatches * cfg.train.micro_batch_size
    # A rank's samples in the order it reads them: each iteration reads the next microbatches x micro_batch_size.
    when = step + 1 + numpy.arange(per_rank) // (microbatches * cfg.train.micro_batch_size)
    read = []
    for rank, loader in enumerate(data_parallel_loaders(cfg, train_ds, first_sample, data_parallel)):
        batches = islice(loader.batch_sampler, iterations * microbatches)
        indices = numpy.fromiter(chain.from_iterable(batches), dtype=numpy.int64)
        if len(indices) != per_rank:
            raise ValueError(
                f"data-parallel rank {rank}'s sampler ends after {len(indices)} samples, and the run reads {per_rank} "
                f"there ({iterations} iterations of {microbatches} microbatches of {cfg.train.micro_batch_size}): the "
                "run reads past one pass of its loader"
            )
        read.append(indices)
    return numpy.concatenate(read), numpy.tile(when, data_parallel)


def corpus_draws(train_ds, samples: numpy.ndarray) -> list[tuple[GPTDataset, float, numpy.ndarray, numpy.ndarray]]:
    """Each corpus of a built training dataset (a blend, or a single corpus), in the blend's order: the corpus, its
    blend weight, which entries of ``samples`` (by position) are samples of that corpus, and which of the corpus's
    own samples those entries are."""
    if isinstance(train_ds, GPTDataset):
        return [(train_ds, 1.0, numpy.arange(len(samples)), samples)]
    corpus, own = train_ds.dataset_index[samples], train_ds.dataset_sample_index[samples]
    draws = []
    for index, (dataset, weight) in enumerate(zip(train_ds.datasets, train_ds.weights)):
        entries = numpy.flatnonzero(corpus == index)
        draws.append((dataset, float(weight), entries, own[entries]))
    return draws


def sequences_first_reached(dataset: GPTDataset, drawn: numpy.ndarray, order: numpy.ndarray) -> numpy.ndarray:
    """Per sequence of a corpus (what ``GPTDataset`` calls its documents), the smallest of ``order`` (positive whole
    numbers, one per sample) over the samples ``drawn`` that reach it, and 0 where none does.

    A sequence is reached when any token of it, the extra label token included, lies in a drawn sample.
    """
    # Sample j spans document_index positions starts[j] .. starts[j + 1] inclusive, so the samples spanning a position
    # are consecutive: from the first whose span ends at or after it to the last whose span starts at or before it.
    starts = numpy.asarray(dataset.sample_index[:, 0])
    never = numpy.iinfo(numpy.int64).max
    sample_order = numpy.full(len(starts) - 1, never, dtype=numpy.int64)
    numpy.minimum.at(sample_order, dataset.shuffle_index[drawn], order)
    positions = numpy.arange(len(dataset.document_index))
    first = numpy.searchsorted(starts[1:], positions, side="left")
    last = numpy.searchsorted(starts[:-1], positions, side="right") - 1
    reached = numpy.full(len(positions), never, dtype=numpy.int64)
    # Most positions lie in one sample or two (a document shorter than a sample); a long document's lie in many. Each
    # pass takes the next spanning sample of every position that still has one.
    active = numpy.flatnonzero(first <= last)
    sample = first[active]
    while active.size:
        reached[active] = numpy.minimum(reached[active], sample_order[sample])
        sample = sample + 1
        more = sample <= last[active]
        active, sample = active[more], sample[more]
    per_sequence = numpy.full(len(dataset.dataset.sequence_lengths), never, dtype=numpy.int64)
    numpy.minimum.at(per_sequence, dataset.document_index, reached)
    per_sequence[per_sequence == never] = 0
    return per_sequence


def sequences_reached(dataset: GPTDataset, drawn: numpy.ndarray) -> numpy.ndarray:
    """The sequences of a corpus that its samples ``drawn`` reach (``sequences_first_reached``), sorted."""
    return numpy.flatnonzero(sequences_first_reached(dataset, drawn, numpy.ones(len(drawn), dtype=numpy.int64)))


def documents_first_read(dataset: GPTDataset, drawn: numpy.ndarray, iterations: numpy.ndarray) -> numpy.ndarray:
    """Per document of a corpus's ``.idx``, the iteration at which the samples ``drawn`` (read at ``iterations``)
    first reach it, and 0 for a document they never reach.

    Document ``d`` holds the sequences ``document_indices[d] .. document_indices[d + 1]``, so it is first read when
    the first of them is, and an empty document, which holds none, is never read.
    """
    per_sequence = sequences_first_reached(dataset, drawn, iterations)
    boundaries = numpy.asarray(dataset.dataset.document_indices)
    never = numpy.iinfo(numpy.int64).max
    per_document = numpy.full(len(boundaries) - 1, never, dtype=numpy.int64)
    read = numpy.flatnonzero(per_sequence)
    numpy.minimum.at(per_document, numpy.searchsorted(boundaries, read, side="right") - 1, per_sequence[read])
    per_document[per_document == never] = 0
    return per_document
