# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A file's sha256 is its bytes' digest, however many blocks it is read in."""

from __future__ import annotations

import hashlib

import pytest
from scripts.data.file_digest import file_sha256


@pytest.mark.parametrize("block", [1, 3, 1 << 26])
def test_a_files_sha256_is_its_bytes_digest(tmp_path, block):
    content = b"first file\nsecond line"
    path = tmp_path / "file.bin"
    path.write_bytes(content)
    assert file_sha256(path, block) == file_sha256(str(path)) == hashlib.sha256(content).hexdigest()
