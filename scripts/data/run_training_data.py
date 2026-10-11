# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A training run's ``.bin/.idx`` data, built on CPU as its launch builds it.

The config is resolved by the launcher's own merge (``pipeline_training_run.resolve_bin_idx_run_config``), the
loader's training-data window (``get_train_data_window``) sizes the training dataset at the run's resumed step, and the
loader's builder builds it. No process group, GPU or checkpoint is needed, and the index caches written are the ones the
launch would write, so the launch then finds them warm. The tools that read a run's data before it trains
(``report_blend_coverage.py``, ``predict_masked_counts.py``) build it here.

The resumed step is the config's ``checkpoint.ckpt_step`` (the run's start when unset: a resume from the load
directory's latest save is not recognised), and the samples consumed before it are that step times the global batch,
so a batch-size ramp is refused.
"""

from __future__ import annotations

import sys
from pathlib import Path

from megatron.core.num_microbatches_calculator import ConstantNumMicroBatchesCalculator

from megatron.bridge.data.loaders import build_train_valid_test_datasets, get_train_data_window
from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
from megatron.bridge.training.state import TrainState


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
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
