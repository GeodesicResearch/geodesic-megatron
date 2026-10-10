# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The token-id checks the repo's tools apply to ids read from a hand-written file: a probe spec, a gate spec, a
corpus select config.

``require_token_id`` accepts one non-negative integer (a YAML ``true`` is a bool, not an id), and
``require_token_id_list`` a list of distinct ones, empty or not as the caller says. Each raises the caller's own error
type with a message that opens with where the value sits, so each tool reports the fault as it reports any other in
its input.

Standard library only, and no syntax newer than Python 3.6, as ``mapping_keys``: the gates and the corpus planner
import it under a host Python without the repo's dependencies. The training code checks its own token-masking ids
with the same rule in ``megatron.bridge.training.token_masking.config.validate_token_ids``, which stays in ``src/``:
importing that package imports torch, it raises the token-masking error, and ``src/`` imports nothing from
``scripts/``.
"""

from typing import Any, Tuple, Type


def require_token_id(value: Any, where: str, *, error: Type[Exception] = ValueError) -> int:
    """``value``, which must be a non-negative integer and not a bool; otherwise raises ``error``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise error("{} must be a token id (an integer >= 0), not {!r}".format(where, value))
    return value


def require_token_id_list(
    value: Any, where: str, *, allow_empty: bool, error: Type[Exception] = ValueError
) -> Tuple[int, ...]:
    """``value``, which must be a list of distinct token ids (``require_token_id``), as a tuple; an empty list only
    when ``allow_empty``. Otherwise raises ``error``."""
    if not isinstance(value, list) or (not value and not allow_empty):
        raise error(
            "{} must be a {}list of token ids, not {!r}".format(where, "" if allow_empty else "non-empty ", value)
        )
    ids = tuple(require_token_id(item, "{}[{}]".format(where, index), error=error) for index, item in enumerate(value))
    if len(set(ids)) != len(ids):
        raise error("{} repeats a token id: {}".format(where, list(ids)))
    return ids
