# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The value checks the repo's tools apply to the scalars of a mapping read from a hand-written YAML file (an
export-check spec), beside ``mapping_keys.require_keys``, which checks the mapping's keys.

``require_int``, ``require_number`` and ``require_string`` refuse a value of the wrong type or outside its range, a
YAML ``true`` or ``false`` where a number is meant included (Python counts a bool as an int); ``require_absolute_path``
refuses a path that is not absolute. Each raises the caller's own error type with a message that opens with where the
value sits, so each tool reports the fault as it reports any other in its input.

Standard library only, and no syntax newer than Python 3.6, as ``mapping_keys``.
"""

import math
from pathlib import Path
from typing import Any, Type


def require_int(value: Any, where: str, minimum: int, *, error: Type[Exception] = ValueError) -> int:
    """``value``, which must be an integer of at least ``minimum`` and not a bool; otherwise raises ``error``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise error(f"{where} must be an integer >= {minimum}, not {value!r}")
    return value


def require_number(
    value: Any, where: str, minimum: float, *, inclusive: bool, error: Type[Exception] = ValueError
) -> float:
    """``value`` as a float: a finite integer or float, not a bool, at least ``minimum`` when ``inclusive`` and above
    it otherwise; otherwise raises ``error``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise error(f"{where} must be a finite number, not {value!r}")
    if value < minimum or (value == minimum and not inclusive):
        raise error(f"{where} must be {'>=' if inclusive else '>'} {minimum}, not {value!r}")
    return float(value)


def require_string(value: Any, where: str, *, error: Type[Exception] = ValueError) -> str:
    """``value``, which must be a non-empty string; otherwise raises ``error``."""
    if not isinstance(value, str) or not value:
        raise error(f"{where} must be a non-empty string, not {value!r}")
    return value


def require_absolute_path(value: Any, where: str, *, error: Type[Exception] = ValueError) -> str:
    """``value``, which must be a non-empty string naming an absolute path; otherwise raises ``error``."""
    path = require_string(value, where, error=error)
    if not Path(path).is_absolute():
        raise error(f"{where} must be an absolute path, not {path!r}")
    return path
