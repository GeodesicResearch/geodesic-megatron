# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A nested mapping read as dotted paths, the form in which the repo's tools compare configs and run records key by key:
``dotted_leaves``, a nested mapping's non-mapping values keyed by their dotted paths.

Standard library only, and no syntax newer than Python 3.6, so that a tool under a host Python can import it.
"""

from typing import Any, Dict


def dotted_leaves(mapping: Dict[Any, Any], prefix: str = "") -> Dict[str, Any]:
    """Every non-mapping value of a nested mapping, keyed by its dotted path (``prefix`` before the first key); a list
    is a value."""
    flat: Dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(dotted_leaves(value, path))
        else:
            flat[path] = value
    return flat
