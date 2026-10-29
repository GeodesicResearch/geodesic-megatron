# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A training run's ``.bin/.idx`` data, built on CPU as its launch builds it.

The config is resolved by the launcher's own merge (``pipeline.training.recipes.resolve_bin_idx_run_config``), the
loader's training-data window (``get_train_data_window``) sizes the training dataset at the run's resumed step, and the
loader's builder builds it. No process group, GPU or checkpoint is needed, and the index caches written are the ones
the launch would write, so the launch then finds them warm. The tools that read a run's data before it trains
(``report_blend_coverage.py``, ``predict_masked_counts.py``, ``held_out_documents.py``) build it here.

The resumed step is the config's ``checkpoint.ckpt_step`` (the run's start when unset: a resume from the load
directory's latest save is not recognised), and the samples consumed before it are that step times the global batch,
so a batch-size ramp is refused.

What a built dataset reads is told here too, for those tools: the iterations the run trains
(``require_trained_iterations``) and the first sample each one's global batch reads (``iteration_first_sample``),
which samples of it the run's data-parallel ranks read and at which iteration (``samples_read``, from each rank's own
loader, ``rank_samples``), the samples of one iteration's global batch (``iteration_samples``), which of each corpus's
own samples those are (``corpus_draws``), which of a corpus's sequences they reach (``sequences_reached``), which
windows of its sequences one sample holds (``sample_windows``), and the iteration at which the run first reads each
of a corpus's documents (``documents_first_read``).
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

# BIN_IDX_MODES is re-exported for the tools that build a run's data here.
from pipeline.training.datasets import BIN_IDX_MODES  # noqa: E402, F401
from pipeline.training.recipes import RECIPE_MAP, resolve_bin_idx_run_config  # noqa: E402


MODELS = sorted({model for model, _ in RECIPE_MAP})


def resolve_run_config(config_file: str, model: str, mode: str):
    """The config a launch of ``config_file`` trains with (``resolve_bin_idx_run_config``), refusing a batch-size
    ramp, under which the samples a resumed run consumed cannot be computed."""
    cfg = resolve_bin_idx_run_config(config_file, model, mode)
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


def require_trained_iterations(cfg, first: int, last: int) -> None:
    """Refuse iterations ``first`` .. ``last`` unless the run trains every one of them (as the log numbers them: the
    first iteration of a run from its start is 1, of a run resumed at step s it is s + 1)."""
    step = resumed_step(cfg)
    if not step < first <= last <= cfg.train.train_iters:
        raise ValueError(
            f"iterations {first}..{last} are not inside the run's iterations {step + 1}..{cfg.train.train_iters}"
        )


def iteration_first_sample(cfg, first_sample: int, iteration: int) -> int:
    """The first sample of the run's dataset that ``iteration``'s global batch reads, the run's first iteration
    reading from ``first_sample``."""
    return first_sample + (iteration - 1 - resumed_step(cfg)) * cfg.train.global_batch_size


def rank_samples(cfg, train_ds, consumed: int, data_parallel: int, batches: int) -> list[numpy.ndarray]:
    """Each data-parallel rank's next ``batches`` batches of samples of ``train_ds`` from the sample ``consumed``, in
    the order the rank reads them, as its own loader's sampler names them; only the indices are read.

    A sampler that ends before then is refused: the run would read past one pass of its loader, which no single pass
    of its sampler describes.
    """
    expected = batches * cfg.train.micro_batch_size
    read = []
    for rank, loader in enumerate(data_parallel_loaders(cfg, train_ds, consumed, data_parallel)):
        indices = numpy.fromiter(chain.from_iterable(islice(loader.batch_sampler, batches)), dtype=numpy.int64)
        if len(indices) != expected:
            raise ValueError(
                f"data-parallel rank {rank}'s sampler ends after {len(indices)} samples from sample {consumed}, and the "
                f"run reads {expected} there: the run reads past one pass of its loader"
            )
        read.append(indices)
    return read


def samples_read(cfg, train_ds, first_sample: int, data_parallel: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Every sample of ``train_ds`` the run reads from its resumed step to its last iteration, and the iteration that
    reads it (as the log numbers iterations: the first of a run from its start is 1).

    A rank runs the run's microbatches per replica every iteration, so its sampler's first ``iterations x
    microbatches`` batches (``rank_samples``) are what the rank reads, iteration after iteration.
    """
    step = resumed_step(cfg)
    iterations = cfg.train.train_iters - step
    microbatches = microbatches_per_replica(cfg.train.global_batch_size, cfg.train.micro_batch_size, data_parallel)
    per_rank = iterations * microbatches * cfg.train.micro_batch_size
    # A rank's samples in the order it reads them: each iteration reads the next microbatches x micro_batch_size.
    when = step + 1 + numpy.arange(per_rank) // (microbatches * cfg.train.micro_batch_size)
    read = rank_samples(cfg, train_ds, first_sample, data_parallel, iterations * microbatches)
    return numpy.concatenate(read), numpy.tile(when, data_parallel)


def iteration_samples(cfg, train_ds, first_sample: int, data_parallel: int, iteration: int) -> numpy.ndarray:
    """The samples of ``train_ds`` one iteration's global batch reads: each data-parallel rank's in the order the
    rank reads them, rank after rank, from loaders built at the iteration's first sample, as a run resumed there
    builds them. An iteration the run does not train is refused."""
    require_trained_iterations(cfg, iteration, iteration)
    microbatches = microbatches_per_replica(cfg.train.global_batch_size, cfg.train.micro_batch_size, data_parallel)
    consumed = iteration_first_sample(cfg, first_sample, iteration)
    return numpy.concatenate(rank_samples(cfg, train_ds, consumed, data_parallel, microbatches))


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
    # Sample j spans document_index positions starts[j] .. starts[j + 1] inclusive (sample_windows states the rule for
    # one sample), so the samples spanning a position are consecutive: from the first whose span ends at or after it to
    # the last whose span starts at or before it.
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


def sample_windows(dataset: GPTDataset, sample: int) -> list[tuple[int, int, int]]:
    """The windows of a corpus's sequences that one of its samples holds, in order, each as (sequence, first token,
    end): the rule ``sequences_first_reached`` applies, stated for one sample, with the tokens each sequence gives.

    The sample's row of the sample index (through the shuffle index) spans ``document_index`` positions from its own
    entry to the next row's, both included. The first position gives its tokens from the row's offset, the last its
    tokens up to and including the next row's offset (the extra token the sample's last label needs), and every
    position between gives all of its sequence.
    """
    if not dataset.config.add_extra_token_to_sequence:
        raise ValueError("a sample without the extra label token is not what sequences_first_reached describes")
    row = int(dataset.shuffle_index[sample])
    (first, offset), (last, end) = dataset.sample_index[row], dataset.sample_index[row + 1]
    windows = []
    for position in range(int(first), int(last) + 1):
        sequence = int(dataset.document_index[position])
        start = int(offset) if position == first else 0
        stop = int(end) + 1 if position == last else int(dataset.dataset.sequence_lengths[sequence])
        windows.append((sequence, start, stop))
    return windows


def sequences_reached(dataset: GPTDataset, drawn: numpy.ndarray) -> numpy.ndarray:
    """The sequences of a corpus that its samples ``drawn`` reach (``sequences_first_reached``), sorted."""
    return numpy.flatnonzero(sequences_first_reached(dataset, drawn, numpy.ones(len(drawn), dtype=numpy.int64)))


def sequence_documents(dataset: GPTDataset, sequences: numpy.ndarray) -> numpy.ndarray:
    """The document of a corpus's ``.idx`` that holds each of ``sequences``: document ``d`` holds the sequences
    ``document_indices[d] .. document_indices[d + 1]``, and an empty document holds none."""
    return numpy.searchsorted(numpy.asarray(dataset.dataset.document_indices), sequences, side="right") - 1


def documents_first_read(dataset: GPTDataset, drawn: numpy.ndarray, iterations: numpy.ndarray) -> numpy.ndarray:
    """Per document of a corpus's ``.idx``, the iteration at which the samples ``drawn`` (read at ``iterations``)
    first reach it, and 0 for a document they never reach.

    A document is first read when the first of its sequences is, and an empty document, which holds none, is never
    read.
    """
    per_sequence = sequences_first_reached(dataset, drawn, iterations)
    never = numpy.iinfo(numpy.int64).max
    per_document = numpy.full(len(dataset.dataset.document_indices) - 1, never, dtype=numpy.int64)
    read = numpy.flatnonzero(per_sequence)
    numpy.minimum.at(per_document, sequence_documents(dataset, read), per_sequence[read])
    per_document[per_document == never] = 0
    return per_document
