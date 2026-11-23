# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Write the tail configs of the efficiency study from a spec and a template.

Usage, from the repository root::

    python configs/olmo_mix_efficiency/make_tails.py configs/olmo_mix_efficiency/tails.yaml

The spec states the template, the names and each tail. ``README.md`` of this folder describes a tail.

Catalogue:
    SPEC_KEYS: the keys of a spec.
    TAIL_KEYS: the keys of one tail of a spec.
    Tail: one tail of a spec.
    read_tails: the settings and the tails of a spec file.
    decay_iterations: the length of the linear decay that gives a tail its floor.
    tail_config: the text of the config of one tail.
    tail_configs: the text of each tail config of a spec, by file name.
    main: the command line.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.training.config_compose import load_composed_yaml  # noqa: E402


LOGGER = logging.getLogger("make_tails")

SPEC_KEYS = frozenset({"template", "config_name", "run_name", "save_beside", "tails"})
TAIL_KEYS = frozenset({"base", "start", "length", "floor"})


@dataclass(frozen=True)
class Tail:
    """One tail of a spec.

    ``base`` is the config that the tail continues, from its iteration ``start`` for ``length`` iterations. ``floor``
    is the learning rate at the last iteration of the tail."""

    name: str
    base: str
    start: int
    length: int
    floor: float


def read_tails(spec_path: Path) -> tuple[dict[str, Any], tuple[Tail, ...]]:
    """The settings of the spec at ``spec_path`` and its tails, in the order of the file.

    Reads the spec file. Raises ValueError when the spec or a tail lacks a key or has an unknown key. Raises
    yaml.YAMLError when the file states a key two times."""
    spec = load_composed_yaml(spec_path)
    if set(spec) != SPEC_KEYS:
        raise ValueError(f"{spec_path}: the keys of the spec must be {sorted(SPEC_KEYS)}, not {sorted(spec)}")
    for name, tail in spec["tails"].items():
        if set(tail) != TAIL_KEYS:
            raise ValueError(f"{spec_path}: the keys of tail {name} must be {sorted(TAIL_KEYS)}, not {sorted(tail)}")
    return spec, tuple(Tail(name=name, **tail) for name, tail in spec["tails"].items())


def decay_iterations(tail: Tail, peak: float, min_lr: float) -> int:
    """The iterations of the linear decay from ``peak`` to ``min_lr`` that is at ``tail.floor`` after ``tail.length``
    iterations. It is ``tail.length`` when the floor is ``min_lr``.

    Raises ValueError when the floor is below ``min_lr`` or not below ``peak``."""
    if not min_lr <= tail.floor < peak:
        raise ValueError(f"the floor {tail.floor} of the tail {tail.name} is not from {min_lr} to below {peak}")
    return round(tail.length * (peak - min_lr) / (peak - tail.floor))


def tail_config(tail: Tail, spec: dict[str, Any], folder: Path) -> str:
    """The text of the config of ``tail``: the template of ``spec`` with its names in braces filled.

    Reads three files of ``folder``: the template, the config ``tail.base`` and the config ``spec["save_beside"]``.
    It reads each config with its ``base_config`` chain. The base gives ``optimizer.lr``, ``optimizer.min_lr`` and
    the folder that the tail loads from. Raises ValueError as ``decay_iterations`` does."""
    base = load_composed_yaml(folder / tail.base)
    peak, min_lr = base["optimizer"]["lr"], base["optimizer"]["min_lr"]
    decay = decay_iterations(tail, peak, min_lr)
    beside = Path(load_composed_yaml(folder / spec["save_beside"])["checkpoint"]["save"]).parent
    return (
        (folder / spec["template"])
        .read_text()
        .format(
            BASE=tail.base,
            K=tail.start,
            T=tail.length,
            END=tail.start + tail.length,
            DECAY_END=tail.start + decay,
            T_DECAY=decay,
            LOAD=base["checkpoint"]["save"],
            SAVE=f"{beside}/{tail.name}",
            NAME=spec["run_name"].format(tail=tail.name),
            FLOOR=f"{tail.floor:.1e}",
            MIN_LR=f"{min_lr:.1e}",
        )
    )


def tail_configs(spec_path: Path) -> dict[str, str]:
    """The text of each tail config of the spec at ``spec_path``, by file name.

    Reads the spec, and the files that ``tail_config`` reads in the folder of the spec. Raises ValueError as
    ``read_tails`` and ``tail_config`` do, and when two tails have one file name."""
    spec, tails = read_tails(spec_path)
    configs: dict[str, str] = {}
    for tail in tails:
        name = spec["config_name"].format(tail=tail.name)
        if name in configs:
            raise ValueError(f"{spec_path}: two tails have the file name {name}")
        configs[name] = tail_config(tail, spec, spec_path.parent)
    return configs


def main(argv: list[str] | None = None) -> None:
    """Write each tail config of a spec into the folder of the spec. The one argument is the path of the spec."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path, help="the spec of the tails, for example tails.yaml of this folder")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name, text in tail_configs(args.spec).items():
        (args.spec.parent / name).write_text(text)
        LOGGER.info("wrote %s", args.spec.parent / name)


if __name__ == "__main__":
    main()
