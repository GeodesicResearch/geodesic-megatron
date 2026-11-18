# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/file_digest.py: the digests of a file, of a directory's files and of a chat template.

Every test reads real files under ``tmp_path`` and calls the real functions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from scripts.file_digest import (
    HASH_WORKERS,
    chat_template_sha256,
    directory_records,
    file_sha256,
    selected_paths,
)

from tests.unit_tests.checkpoint_dirs import make_checkpoint_dir


TOKENIZER_FIXTURE = Path(__file__).parent / "fixtures" / "hf_export" / "inputs" / "nemotron-instruct-tokenizer"


@pytest.fixture
def unit(tmp_path) -> Path:
    """The save ``iter_0000100``: a 16-byte shard of ``x``, ``metadata.json`` and ``hf/model.safetensors``."""
    run = make_checkpoint_dir(tmp_path / "run", [100], tracker=None, hf_files={"model.safetensors": "h"})
    return run / "iter_0000100"


@pytest.mark.parametrize("block", [1, 3, 1 << 26])
def test_a_files_sha256_is_its_bytes_digest(tmp_path, block):
    content = b"first file\nsecond line"
    path = tmp_path / "file.bin"
    path.write_bytes(content)
    assert file_sha256(path, block) == file_sha256(str(path)) == hashlib.sha256(content).hexdigest()


class TestSelectedPaths:
    def test_exclude_and_include_patterns_choose_the_files(self, unit):
        assert selected_paths(unit, None, ("hf/*",)) == ["__0_0.distcp", "metadata.json"]
        assert selected_paths(unit, ("*.distcp",), ()) == ["__0_0.distcp"]
        assert selected_paths(unit, None, ()) == ["__0_0.distcp", "hf/model.safetensors", "metadata.json"]


class TestDirectoryRecords:
    def test_size_and_sha256_of_every_selected_file(self, unit):
        found = {r.path: r for r in directory_records(unit, None, ("hf/*",), None, HASH_WORKERS)}
        assert set(found) == {"__0_0.distcp", "metadata.json"}
        assert found["__0_0.distcp"].size == 16
        assert found["__0_0.distcp"].sha256 == hashlib.sha256(b"x" * 16).hexdigest()

    def test_the_records_come_in_path_order_whatever_the_worker_count(self, unit):
        one = directory_records(unit, None, (), None, 1)
        many = directory_records(unit, None, (), None, 4)
        assert one == many
        assert [r.path for r in one] == sorted(r.path for r in one)

    def test_a_file_at_its_recorded_size_is_not_read_again_and_a_changed_size_is(self, unit):
        previous = directory_records(unit, None, ("hf/*",), None, HASH_WORKERS)
        (unit / "__0_0.distcp").write_bytes(b"z" * 16)  # same size, other content: the record is kept
        (unit / "metadata.json").write_text("{ }")  # another size: read again
        found = {r.path: r for r in directory_records(unit, None, ("hf/*",), previous, HASH_WORKERS)}
        assert found["__0_0.distcp"].sha256 == hashlib.sha256(b"x" * 16).hexdigest()
        assert found["metadata.json"].sha256 == hashlib.sha256(b"{ }").hexdigest()


class TestChatTemplateSha256:
    def test_a_template_hashes_as_the_file_it_was_read_from(self):
        template_file = TOKENIZER_FIXTURE / "chat_template.jinja"
        assert (
            chat_template_sha256(template_file.read_text(encoding="utf-8"))
            == hashlib.sha256(template_file.read_bytes()).hexdigest()
        )

    def test_templates_held_by_name_hash_by_their_content_in_any_order(self):
        one = {"default": "{{ a }}", "tool_use": "{{ b }}"}
        assert chat_template_sha256(one) == chat_template_sha256(dict(reversed(one.items())))
        assert chat_template_sha256(one) != chat_template_sha256({**one, "tool_use": "{{ c }}"})

    def test_a_tokenizer_without_a_template_has_no_hash(self):
        assert chat_template_sha256(None) is None
