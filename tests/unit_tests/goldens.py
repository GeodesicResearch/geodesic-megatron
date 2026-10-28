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
"""Golden files: outputs recorded from the code under test and compared byte for byte on every run.

Goldens live under ``tests/unit_tests/fixtures/<area>/``. With ``UPDATE_GOLDENS=1`` in the environment each assertion writes its
golden instead of comparing, so a run that records goldens passes by construction; the diff of the fixtures is
what a reviewer reads.

Catalogue:
    updating: whether this run writes goldens instead of comparing.
    assert_golden_bytes: compare bytes with a golden file.
    assert_golden_json: compare a JSON-serialisable value with a golden JSON file.
    assert_golden_tree: compare every file under a directory with a golden directory.
"""

import difflib
import json
import os
import shutil
from pathlib import Path


def updating() -> bool:
    """Whether ``UPDATE_GOLDENS=1`` is set, so goldens are written rather than compared."""
    return os.environ.get("UPDATE_GOLDENS") == "1"


def _mismatch(golden: Path, expected: bytes, actual: bytes) -> str:
    try:
        diff = difflib.unified_diff(
            expected.decode().splitlines(keepends=True),
            actual.decode().splitlines(keepends=True),
            fromfile=f"{golden} (golden)",
            tofile="actual",
        )
        return "".join(diff)[:20000]
    except UnicodeDecodeError:
        return f"{golden}: binary content differs ({len(expected)} golden bytes, {len(actual)} actual)"


def assert_golden_bytes(golden: Path, actual: bytes) -> None:
    """Assert ``actual`` equals the bytes of ``golden``; with ``updating()``, write it there instead.

    Raises:
        AssertionError: the golden is missing or differs, with a unified diff when both decode as text.
    """
    if updating():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(actual)
        return
    assert golden.is_file(), f"no golden at {golden}; record it with UPDATE_GOLDENS=1"
    expected = golden.read_bytes()
    assert actual == expected, _mismatch(golden, expected, actual)


def assert_golden_json(golden: Path, value: object) -> None:
    """Assert ``value``, as indented JSON with sorted keys, equals ``golden`` (see ``assert_golden_bytes``)."""
    assert_golden_bytes(golden, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())


def assert_golden_tree(golden: Path, actual: Path) -> None:
    """Assert the files under ``actual`` and ``golden`` have the same relative paths and bytes.

    With ``updating()``, replaces ``golden`` with a copy of ``actual``. Symlinks compare by their targets' bytes.

    Raises:
        AssertionError: a file is missing on either side, or a file's bytes differ.
    """
    if updating():
        if golden.exists():
            shutil.rmtree(golden)
        shutil.copytree(actual, golden, symlinks=False)
        return
    assert golden.is_dir(), f"no golden at {golden}; record it with UPDATE_GOLDENS=1"

    def files(root: Path) -> set[str]:
        return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}

    expected_files, actual_files = files(golden), files(actual)
    assert actual_files == expected_files, (
        f"file sets differ: missing {sorted(expected_files - actual_files)}, extra {sorted(actual_files - expected_files)}"
    )
    for name in sorted(expected_files):
        expected, got = (golden / name).read_bytes(), (actual / name).read_bytes()
        assert got == expected, _mismatch(golden / name, expected, got)
