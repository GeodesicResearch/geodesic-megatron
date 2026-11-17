# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""``checkpoint.save_steps`` places saves at chosen iterations, beside the regular ``save_interval``.

Saves at 5, 10, 20, 40 and 100% of a run cannot be placed with one interval. The trainer asks
``CheckpointConfig.is_save_step`` once per iteration and ``is_final_save_due`` once at the end, so both decisions are
tested on the real config class.
"""

import pytest

from megatron.bridge.training.config import CheckpointConfig


def test_without_a_save_directory_nothing_is_due() -> None:
    config = CheckpointConfig(save=None, save_interval=10, save_steps=[3])
    assert not any(config.is_save_step(step) for step in range(1, 40))


def test_the_interval_alone_saves_on_its_multiples() -> None:
    config = CheckpointConfig(save="/ckpt", save_interval=10)
    assert [step for step in range(1, 41) if config.is_save_step(step)] == [10, 20, 30, 40]


def test_the_listed_steps_alone_save_at_those_steps() -> None:
    config = CheckpointConfig(save="/ckpt", save_steps=[3, 6, 12, 24, 60])
    assert [step for step in range(1, 61) if config.is_save_step(step)] == [3, 6, 12, 24, 60]


def test_the_interval_and_the_list_add_up() -> None:
    config = CheckpointConfig(save="/ckpt", save_interval=25, save_steps=[3, 50])
    assert [step for step in range(1, 76) if config.is_save_step(step)] == [3, 25, 50, 75]


def test_no_interval_and_no_list_save_nothing() -> None:
    config = CheckpointConfig(save="/ckpt")
    assert not any(config.is_save_step(step) for step in range(1, 40))


@pytest.mark.parametrize("steps", [[0], [5, -1]])
def test_a_step_that_is_not_positive_is_refused(steps: list[int]) -> None:
    with pytest.raises(ValueError, match="must hold positive iterations"):
        CheckpointConfig(save="/ckpt", save_steps=steps).finalize()


@pytest.mark.parametrize(
    ("save_interval", "save_steps", "train_iters", "first"),
    [
        (1_000_000, [50, 200], 400, 50),
        (100, [250], 400, 100),
        (100, [30, 250], 400, 30),
        (1_000_000, [500], 400, 400),
        (None, None, 400, 400),
        (0, None, 400, 400),
    ],
    ids=[
        "a listed step before the interval",
        "the interval before a listed step",
        "a listed step before both",
        "a listed step beyond the run",
        "only the end-of-run save",
        "an interval of 0",
    ],
)
def test_the_first_save_is_the_earliest_the_run_reaches(
    save_interval: int | None, save_steps: list[int] | None, train_iters: int, first: int
) -> None:
    config = CheckpointConfig(save="/ckpt", save_interval=save_interval, save_steps=save_steps)
    assert config.first_save_step(train_iters) == first


@pytest.mark.parametrize(
    ("save", "save_interval", "save_steps", "last_step", "due"),
    [
        ("/ckpt", None, None, 100, True),
        ("/ckpt", 10, None, 100, False),
        ("/ckpt", 10, None, 95, True),
        ("/ckpt", 1_000_000, [40, 100], 100, False),
        ("/ckpt", 1_000_000, [40], 100, True),
        ("/ckpt", None, [100], 100, False),
        ("/ckpt", 0, [40], 100, False),
        ("/ckpt", None, None, 0, False),
        (None, None, None, 100, False),
    ],
    ids=[
        "no interval and no list",
        "last step on the interval",
        "last step off the interval",
        "last step in the list",
        "list without the last step",
        "list alone holds the last step",
        "an interval of 0 turns the end-of-run save off",
        "no step trained",
        "no save directory",
    ],
)
def test_the_final_save_is_due_unless_a_regular_save_wrote_the_last_step(
    save: str | None, save_interval: int | None, save_steps: list[int] | None, last_step: int, due: bool
) -> None:
    config = CheckpointConfig(save=save, save_interval=save_interval, save_steps=save_steps)
    assert config.is_final_save_due(last_step) is due
