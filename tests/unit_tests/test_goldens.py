# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""tests/unit_tests/goldens.py: a golden compares byte for byte, and UPDATE_GOLDENS=1 records instead."""

import pytest

from tests.unit_tests.goldens import assert_golden_bytes, assert_golden_json, assert_golden_tree


def test_a_golden_is_recorded_under_update_goldens_and_compared_otherwise(tmp_path, monkeypatch):
    golden = tmp_path / "area" / "case.json"
    monkeypatch.setenv("UPDATE_GOLDENS", "1")
    assert_golden_json(golden, {"b": 1, "a": [2]})
    assert golden.read_text() == '{\n  "a": [\n    2\n  ],\n  "b": 1\n}\n'

    monkeypatch.delenv("UPDATE_GOLDENS")
    assert_golden_json(golden, {"a": [2], "b": 1})
    with pytest.raises(AssertionError, match=r'(?s)-  "b": 1.*\+  "b": 2'):
        assert_golden_json(golden, {"a": [2], "b": 2})


def test_a_missing_golden_fails_rather_than_passing(tmp_path, monkeypatch):
    monkeypatch.delenv("UPDATE_GOLDENS", raising=False)
    with pytest.raises(AssertionError, match="no golden"):
        assert_golden_bytes(tmp_path / "absent.bin", b"x")


def test_a_tree_differs_by_one_extra_file_or_one_changed_byte(tmp_path, monkeypatch):
    actual = tmp_path / "actual"
    (actual / "sub").mkdir(parents=True)
    (actual / "sub" / "w.bin").write_bytes(b"\x00\x01")
    golden = tmp_path / "golden"
    monkeypatch.setenv("UPDATE_GOLDENS", "1")
    assert_golden_tree(golden, actual)
    monkeypatch.delenv("UPDATE_GOLDENS")
    assert_golden_tree(golden, actual)

    (actual / "extra.txt").write_text("x")
    with pytest.raises(AssertionError, match=r"extra \['extra.txt'\]"):
        assert_golden_tree(golden, actual)
    (actual / "extra.txt").unlink()
    (actual / "sub" / "w.bin").write_bytes(b"\xff\xfe")
    with pytest.raises(AssertionError, match="w.bin: binary content differs"):
        assert_golden_tree(golden, actual)
    (actual / "sub" / "w.bin").unlink()
    with pytest.raises(AssertionError, match=r"missing \['sub/w.bin'\]"):
        assert_golden_tree(golden, actual)


def test_recording_a_tree_replaces_the_golden_and_drops_its_stale_files(tmp_path, monkeypatch):
    golden, actual = tmp_path / "golden", tmp_path / "actual"
    golden.mkdir()
    (golden / "stale.txt").write_text("old")
    actual.mkdir()
    (actual / "new.txt").write_text("new")
    monkeypatch.setenv("UPDATE_GOLDENS", "1")
    assert_golden_tree(golden, actual)
    assert sorted(p.name for p in golden.iterdir()) == ["new.txt"]


def test_a_missing_golden_tree_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("UPDATE_GOLDENS", raising=False)
    (tmp_path / "actual").mkdir()
    with pytest.raises(AssertionError, match="no golden"):
        assert_golden_tree(tmp_path / "absent", tmp_path / "actual")
