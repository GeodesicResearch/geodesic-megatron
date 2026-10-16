# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A nested mapping read as dotted paths, the form in which the repo's tools compare configs and run records key by key.

- ``dotted_leaves``: a nested mapping's non-mapping values, keyed by their dotted paths.
- ``value_at``: the value one dotted path names in a nested document, a list's element named by its index.

Standard library only, and no syntax newer than Python 3.6, so that a tool under a host Python can import it.
"""

import re
from typing import Any, Dict


LIST_INDEX = re.compile(r"[0-9]+")


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


def value_at(document: Any, path: str) -> Any:
    """The value ``path`` names in ``document``: each dotted part is a mapping's key, or, where the document there is
    a list, a decimal index into it (``corpora.0.docs``). Raises KeyError(path) when a part names nothing."""
    for part in path.split("."):
        if isinstance(document, dict) and part in document:
            document = document[part]
        elif isinstance(document, list) and LIST_INDEX.fullmatch(part) and int(part) < len(document):
            document = document[int(part)]
        else:
            raise KeyError(path)
    return document
