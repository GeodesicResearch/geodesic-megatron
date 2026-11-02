# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The sha256 of a file, read in blocks: the digest the corpus tools record of the configs, tables and lists they read.

Standard library only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


HASH_BLOCK_BYTES = 1 << 26  # 64 MiB read per step when hashing a file


def file_sha256(path: str | Path, block: int = HASH_BLOCK_BYTES) -> str:
    """The sha256 of the file at ``path``, read ``block`` bytes at a time."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()
