# Copyright (c) 2026, Geodesic Research.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The header of a safetensors file, read with the standard library alone.

A safetensors file is an 8-byte little-endian header length, that many bytes of JSON naming each
tensor with its dtype, shape and byte range in the data that follows, and then the data. Reading
the header needs neither torch nor the safetensors package, so the tools that check an export on
the host Python use this module as well as the ones that write or convert shards in the container.
Import it as ``scripts.safetensors_header`` with the repository root on ``sys.path``.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


HEADER_LIMIT = 100_000_000
"""The format's own cap on the header length, in bytes."""

METADATA_KEY = "__metadata__"
"""The header entry that holds free-form metadata rather than a tensor."""


def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    """The header length and the parsed header of the safetensors file at ``path``. A declared
    length beyond the file, or beyond the format's cap, is a corrupt header and raises ValueError
    before anything is read into memory."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length > min(size - 8, HEADER_LIMIT):
            raise ValueError(f"{path}: header length {length} exceeds the file or the format's limit")
        return length, json.loads(handle.read(length))


def tensor_entries(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The header's tensors by name, without its metadata entry."""
    return {name: entry for name, entry in header.items() if name != METADATA_KEY}


def declared_size(length: int, header: dict[str, Any]) -> int:
    """The byte length a file with this header must have: the length prefix, the header, and the
    data up to the end of its last tensor. A write cut short leaves the header intact, so comparing
    this with the file's size is what reveals it."""
    ends = [entry["data_offsets"][1] for entry in tensor_entries(header).values()]
    return 8 + length + max(ends, default=0)
