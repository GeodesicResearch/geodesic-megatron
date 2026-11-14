# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The JSON Lines files of integer entries that the data tools exchange, such as a list of rows to leave out.

Catalogue:
    read_integer_entries: the entries of such a file, each an object with exactly the given keys, and its sha256.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.mapping_values import require_int  # noqa: E402


def read_integer_entries(
    path: Path, keys: tuple[str, ...], error: type[Exception]
) -> tuple[list[tuple[int, ...]], str]:
    """The entries of the JSON Lines file ``path`` as tuples of their values in ``keys`` order, in file order, and the
    sha256 of the file's bytes, read once so that the digest is of the bytes the entries come from.

    Each line must be a JSON object with exactly ``keys``, each value an integer of at least 0, and no entry may occur
    twice.

    Raises:
        error: a line that is not such an object, or an entry repeated, with a message that opens with the line of
            ``path`` and quotes it.
        OSError: ``path`` cannot be read.
    """
    raw = path.read_bytes()
    first_seen: dict[tuple[int, ...], int] = {}
    for number, text in enumerate(raw.decode("utf-8").splitlines(), start=1):
        where = f"{path}:{number}: "
        try:
            entry = json.loads(text)
        except json.JSONDecodeError as exc:
            raise error(f"{where}not JSON ({exc}): {text!r}") from exc
        if not isinstance(entry, dict) or entry.keys() != set(keys):
            raise error(f"{where}expected an object with exactly the keys {sorted(keys)}, got {text!r}")
        values = tuple(require_int(entry[key], f"{where}{key} of {text!r}", 0, error=error) for key in keys)
        if values in first_seen:
            raise error(f"{where}duplicate of line {first_seen[values]}: {text!r}")
        first_seen[values] = number
    return list(first_seen), hashlib.sha256(raw).hexdigest()
