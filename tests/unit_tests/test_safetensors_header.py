# Copyright (c) 2026, Geodesic Research.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for scripts/safetensors_header.py, against files the safetensors package itself writes."""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest
from safetensors.numpy import save_file
from scripts.safetensors_header import HEADER_LIMIT, declared_size, read_header, tensor_entries


def test_a_written_file_is_exactly_the_size_its_header_declares(tmp_path):
    """The writer pads its header and places the tensors after it; the declared size accounts for
    both, and the metadata entry is not a tensor."""
    path = tmp_path / "shard.safetensors"
    save_file({"a": np.zeros((4, 8), np.float32), "b": np.ones((2, 3), np.float16)}, path, metadata={"k": "v"})
    length, header = read_header(path)
    assert declared_size(length, header) == path.stat().st_size
    assert set(tensor_entries(header)) == {"a", "b"}
    assert header["__metadata__"] == {"k": "v"}


def test_a_file_without_tensors_ends_at_its_header(tmp_path):
    path = tmp_path / "empty.safetensors"
    save_file({}, path)
    length, header = read_header(path)
    assert tensor_entries(header) == {}
    assert declared_size(length, header) == path.stat().st_size == 8 + length


def test_a_file_cut_short_declares_more_than_it_holds(tmp_path):
    """A write that dies partway leaves the header whole, so only the declared size, compared with
    the file's, shows the missing data."""
    path = tmp_path / "shard.safetensors"
    save_file({"a": np.zeros((4, 8), np.float32)}, path)
    full = path.stat().st_size
    os.truncate(path, full - 16)
    assert declared_size(*read_header(path)) == full


@pytest.mark.parametrize(
    "declared, file_size",
    [(HEADER_LIMIT + 1, HEADER_LIMIT + 64), (1_000, 64)],
    ids=["beyond the format's cap", "beyond the file"],
)
def test_a_header_length_the_file_cannot_hold_is_refused_unread(tmp_path, declared, file_size):
    """A declared length past the format's cap or past the file itself is a corrupt header; it is
    refused before the read rather than read into memory."""
    path = tmp_path / "corrupt.safetensors"
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", declared))
        handle.truncate(file_size)
    with pytest.raises(ValueError, match="header length"):
        read_header(path)
