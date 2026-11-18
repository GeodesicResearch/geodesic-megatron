# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The ``held_out:`` block of a training config: packed sets that a run evaluates between training steps.

``sets`` maps a name to a packed parquet spec; ``interval`` is how often, in steps, the run evaluates every set
(and it does so once before the first step of a fresh run). The results are logged under ``held-out/<name>/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar


SET_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class HeldOutError(RuntimeError):
    """A held-out set that is misconfigured, does not suit the run's data, or is not on disk."""


@dataclass
class HeldOutConfig:
    """The packed sets a run evaluates in full at ``interval``, and nothing else about them.

    A run without ``sets`` evaluates none. Each evaluation reads every pack of every set exactly once.
    """

    reject_unknown_override_keys: ClassVar[bool] = True

    sets: dict[str, str] = field(default_factory=dict)
    """Set name -> a packed parquet spec (a file, directory or glob, as ``packed_train_data_path`` takes). The name
    is part of every logged key, so it holds letters, digits and ``_.-``."""

    interval: int | None = None
    """Evaluate every set before the first step of a fresh run and after every ``interval`` steps."""

    def finalize(self) -> None:
        """Validate the block.

        Raises:
            HeldOutError: ``sets`` is not a mapping of set names to non-empty paths, a name has other characters than
                letters, digits and ``_.-``, ``interval`` is not a positive integer when there are sets, or is
                given without sets.
        """
        if not isinstance(self.sets, dict):
            raise HeldOutError(f"held_out.sets must map set names to packed paths, got {self.sets!r}")
        for name, path in self.sets.items():
            if not isinstance(name, str) or not SET_NAME.match(name):
                raise HeldOutError(f"held_out.sets: a set name holds letters, digits and _.- only, got {name!r}")
            if not isinstance(path, str) or not path:
                raise HeldOutError(f"held_out.sets.{name} must be a non-empty path, got {path!r}")
        if self.sets:
            if isinstance(self.interval, bool) or not isinstance(self.interval, int) or self.interval < 1:
                raise HeldOutError(
                    f"held_out.interval must be a positive integer when held_out.sets names sets, got {self.interval!r}"
                )
        elif self.interval is not None:
            raise HeldOutError("held_out.interval is set without held_out.sets; name a set or remove the interval")
