# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The rule that a tool which writes outputs never replaces one: it refuses before it writes, so a result that was
reported is never overwritten by a later run.

Catalogue:
    refuse_existing: raise when any of the given paths exists.
"""

from pathlib import Path


def refuse_existing(*paths: Path) -> None:
    """Raise FileExistsError when any of ``paths`` exists: a tool that writes outputs never replaces one."""
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"{existing} exist; a tool that writes outputs never replaces one")
