# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A .bin/.idx run reads its sampler from the YAML, and ``single`` makes the samples of a step independent of the GPU count.

A run that restarts at another GPU count keeps its data order only under a sampler whose samples of a step do not
depend on the data-parallel size. ``cyclic``, the default, gives each rank its own slab of the corpus, so they do.
``single`` hands out consecutive samples, so they do not.

The launcher function and the two sampler classes are the real ones. A step is the global batch: the micro-batches
that every rank draws between two optimizer updates.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import yaml

from megatron.bridge.data.samplers import MegatronPretrainingRandomSampler, MegatronPretrainingSampler
from pipeline.training.datasets import DATALOADER_KEY, DATALOADER_TYPES, bin_idx_dataset_config
from pipeline.training.recipes import resolve_training_config


CORPUS = "/data/corpus_text_document"
TOTAL_SAMPLES = 4096
GLOBAL_BATCH = 64
MICRO_BATCH = 2


def _section(**extra) -> dict:
    return {"data_path": [CORPUS], "seq_length": 4096, "seed": 1234, "split": "1,0,0", **extra}


def _single(rank: int, dp: int, consumed: int) -> MegatronPretrainingSampler:
    return MegatronPretrainingSampler(TOTAL_SAMPLES, consumed, MICRO_BATCH, rank, dp, drop_last=True)


def _cyclic(rank: int, dp: int, consumed: int) -> MegatronPretrainingRandomSampler:
    return MegatronPretrainingRandomSampler(
        list(range(TOTAL_SAMPLES)), TOTAL_SAMPLES, consumed, MICRO_BATCH, rank, dp, True
    )


def _samples_of_steps(make_sampler, dp: int, steps: int, consumed: int = 0) -> list[frozenset[int]]:
    """The samples that all ``dp`` ranks draw in each of the next ``steps`` global batches."""
    micro_batches_per_step = GLOBAL_BATCH // (dp * MICRO_BATCH)
    drawn: list[set[int]] = [set() for _ in range(steps)]
    for rank in range(dp):
        for index, micro_batch in enumerate(
            itertools.islice(make_sampler(rank, dp, consumed), steps * micro_batches_per_step)
        ):
            drawn[index // micro_batches_per_step].update(micro_batch)
    return [frozenset(step) for step in drawn]


def test_the_default_sampler_is_cyclic() -> None:
    assert bin_idx_dataset_config(_section(), "pretrain").dataloader_type == "cyclic"


@pytest.mark.parametrize("dataloader_type", DATALOADER_TYPES)
def test_the_yaml_picks_the_sampler(dataloader_type: str) -> None:
    config = bin_idx_dataset_config(_section(**{DATALOADER_KEY: dataloader_type}), "pretrain")
    assert config.dataloader_type == dataloader_type


def test_an_unknown_sampler_is_refused() -> None:
    with pytest.raises(ValueError, match=f"{DATALOADER_KEY} must be one of"):
        bin_idx_dataset_config(_section(**{DATALOADER_KEY: "batch"}), "pretrain")


@pytest.mark.parametrize("model, mode, recipe_value", [("nano", "pretrain", "single"), ("nano", "cpt", "batch")])
def test_the_recipes_own_dataloader_type_does_not_reach_a_run(
    tmp_path: Path, model: str, mode: str, recipe_value: str
) -> None:
    """The merged ``dataset`` section holds the recipe's value for ``dataloader_type``. Reading that name would turn every
    pretrain run that names no sampler into ``single`` and make every cpt run fail on ``batch``, so the launcher reads
    another key."""
    override = tmp_path / "run.yaml"
    override.write_text(yaml.safe_dump({"dataset": {"data_path": [CORPUS], "seq_length": 4096}}))
    _, merged = resolve_training_config(model, mode, None, str(override), [])
    assert merged["dataset"]["dataloader_type"] == recipe_value, "the recipe does set the name that must be ignored"
    assert bin_idx_dataset_config(merged["dataset"], mode).dataloader_type == "cyclic"


@pytest.mark.parametrize("mode", ["pretrain", "cpt"])
def test_the_yaml_key_reaches_a_run_through_the_launchers_merge(tmp_path: Path, mode: str) -> None:
    override = tmp_path / "run.yaml"
    override.write_text(
        yaml.safe_dump({"dataset": {"data_path": [CORPUS], "seq_length": 4096, DATALOADER_KEY: "single"}})
    )
    _, merged = resolve_training_config("nano", mode, None, str(override), [])
    assert bin_idx_dataset_config(merged["dataset"], mode).dataloader_type == "single"


def test_single_draws_the_same_samples_in_each_step_at_any_gpu_count() -> None:
    steps = 5
    expected = [frozenset(range(step * GLOBAL_BATCH, (step + 1) * GLOBAL_BATCH)) for step in range(steps)]
    for dp in (2, 4, 8, 16):
        assert _samples_of_steps(_single, dp, steps) == expected, f"dp={dp}"


def test_single_resumes_at_another_gpu_count_exactly() -> None:
    """A run that saved after 3 steps at 8 ranks and resumes at 32 draws the steps that an uninterrupted run would."""
    uninterrupted = _samples_of_steps(_single, 8, steps=6)
    resumed = _samples_of_steps(_single, 32, steps=3, consumed=3 * GLOBAL_BATCH)
    assert resumed == uninterrupted[3:]


def test_cyclic_draws_different_samples_at_another_gpu_count() -> None:
    """With data sharding each rank reads its own slab, so the first step differs."""
    assert _samples_of_steps(_cyclic, 4, steps=1) != _samples_of_steps(_cyclic, 8, steps=1)


def test_cyclic_resume_at_another_gpu_count_rereads_and_skips_samples() -> None:
    """After 4 steps at 4 ranks, a resume at 8 ranks does not continue the order: it repeats some samples and skips others."""
    steps_before = 4
    seen_before = set().union(*_samples_of_steps(_cyclic, 4, steps_before))
    after = set().union(*_samples_of_steps(_cyclic, 8, steps=4, consumed=steps_before * GLOBAL_BATCH))
    assert seen_before & after, "expected samples drawn twice"
