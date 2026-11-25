# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The refusal to replace an output (``scripts/output_files.py``), on real files."""

import pytest
from scripts.output_files import refuse_existing


def test_paths_that_do_not_exist_are_accepted(tmp_path):
    refuse_existing(tmp_path / "a.json", tmp_path / "b.json")


def test_a_path_that_exists_is_refused_and_named(tmp_path):
    (tmp_path / "taken.json").write_text("{}")
    with pytest.raises(FileExistsError, match="taken.json"):
        refuse_existing(tmp_path / "free.json", tmp_path / "taken.json")
    assert (tmp_path / "taken.json").read_text() == "{}"
