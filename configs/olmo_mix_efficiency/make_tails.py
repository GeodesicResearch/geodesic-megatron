# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Write the tail configs of the efficiency study from the template (``nemotron_nano_olmo_mix_tail.yaml.tmpl``).

usage: python make_tails.py

The rule of the study: a tail has 10% of the iterations of its start, and at least 100 iterations. Its learning rate
goes linearly from the peak of the trunk (1e-3) to the floor, which is min_lr of the trunk (1e-5, 1% of the peak).
The calibration at iteration 1,000 adds tails of 5% and 20%, and one with a floor of 1e-4.

A tail keeps optimizer.lr and optimizer.min_lr of the trunk. The optimizer of Megatron finds each saved parameter
group by a key that holds these two values, so another value makes the load of the optimizer state fail. A tail
with a higher floor therefore decays toward 1e-5 over more iterations and stops when the rate is at its floor.

The tail of the smoke run starts at iteration 10, inside the warmup of the smoke run. Its rate jumps from the warmup
value to the peak. It tests the load, the schedule and the save folder, and its loss values mean nothing.

The template has no ``.yaml`` suffix, because it is not a config: the tools that read each YAML file of ``configs/``
cannot compose it. ``tests/unit_tests/test_olmo_mix_efficiency_configs.py`` holds the tail configs of this folder to
the output of this script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml


LOGGER = logging.getLogger("make_tails")

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "nemotron_nano_olmo_mix_tail.yaml.tmpl"
PREFIX = "nemotron_nano_olmo_mix_"
PEAK, MIN_LR = 1.0e-3, 1.0e-5
TRUNK, SEED2, SMOKE = (f"{PREFIX}{name}.yaml" for name in ("trunk", "trunk_seed2", "smoke"))


def saves_to(config: str) -> str:
    """The checkpoint folder that the config ``config`` of this folder saves to."""
    return yaml.safe_load((HERE / config).read_text())["checkpoint"]["save"]


@dataclass(frozen=True)
class Tail:
    """One tail: the config it continues, its start iteration and its length in iterations, and the learning rate at
    its last iteration. It loads from the folder that the config it continues saves to, and it saves to a folder of
    its own name beside the folder of the trunk."""

    name: str
    base: str
    start: int
    length: int
    floor: float = MIN_LR

    @property
    def decay(self) -> int:
        """The iterations of the linear decay to MIN_LR that reaches ``floor`` after ``length`` iterations: equal to
        ``length`` for the floor of the trunk."""
        return round(self.length * (PEAK - MIN_LR) / (PEAK - self.floor))

    def rate(self, step: int) -> float:
        """The learning rate at iteration ``step`` of the tail (1 is its first iteration)."""
        return MIN_LR + (1.0 - step / self.decay) * (PEAK - MIN_LR)


def study_length(start: int) -> int:
    """The length of a tail that starts at iteration ``start``, by the rule of the study."""
    return max(100, round(0.10 * start))


TAILS: tuple[Tail, ...] = (
    *(Tail(f"nano_tail_{k}", TRUNK, k, study_length(k)) for k in (500, 1000, 1500, 2000, 2500)),
    # The calibration of the tail length and of the floor.
    Tail("nano_tail_1000_len05", TRUNK, 1000, 50),
    Tail("nano_tail_1000_len20", TRUNK, 1000, 200),
    Tail("nano_tail_1000_floor10", TRUNK, 1000, 100, floor=1.0e-4),
    # The tail of the repeat with another seed. Its base is the config of the repeat, so it keeps that seed.
    Tail("nano_seed2_tail_1000", SEED2, 1000, 100),
    Tail("nano_smoke_tail", SMOKE, 10, 10),
)


def tail_config(tail: Tail) -> str:
    """The text of the config of ``tail``: the template with its names in braces filled."""
    return TEMPLATE.read_text().format(
        BASE=tail.base,
        K=tail.start,
        T=tail.length,
        END=tail.start + tail.length,
        DECAY_END=tail.start + tail.decay,
        T_DECAY=tail.decay,
        LOAD=saves_to(tail.base),
        SAVE=f"{Path(saves_to(TRUNK)).parent}/{tail.name}",
        NAME=f"olmo_mix_efficiency_{tail.name}",
        FLOOR=f"{tail.floor:.0e}",
    )


def tail_configs() -> dict[str, str]:
    """The text of each tail config of the study, by file name."""
    return {f"{PREFIX}{tail.name}.yaml": tail_config(tail) for tail in TAILS}


def main() -> None:
    """Write each tail config beside this script."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for tail, (name, text) in zip(TAILS, tail_configs().items()):
        (HERE / name).write_text(text)
        rates = (tail.start, tail.length, tail.decay, tail.rate(1), tail.rate(tail.length))
        LOGGER.info("%s: from %d for %d iterations, decay over %d, first rate %.4e, last rate %.4e", tail.name, *rates)


if __name__ == "__main__":
    main()
