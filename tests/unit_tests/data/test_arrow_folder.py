# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/data/arrow_folder.py: the data files of a saved Arrow folder and their rows, in the order its state.json lists them.

The folders are real: written with pyarrow in ``tmp_path`` by the fixture that the Arrow-folder tests share.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest
from scripts.data.arrow_folder import arrow_rows, arrow_shards

from tests.unit_tests.arrow_folder_fixtures import write_arrow_folder


class CallerError(RuntimeError):
    """An error type of a caller, to show that the helpers raise the type they are given."""


SCHEMA = pa.schema([("name", pa.string()), ("count", pa.int64())])
TABLES = [
    pa.table({"name": ["a", "b"], "count": [1, 2]}, schema=SCHEMA),
    pa.table({"name": ["c"], "count": [3]}, schema=SCHEMA),
]
CACHE = pa.table({"name": ["never read"], "count": [99]}, schema=SCHEMA)


def write_folder(folder: Path, *, listed_order: list[int] | None = None) -> Path:
    return write_arrow_folder(folder, TABLES, CACHE, listed_order=listed_order)


class TestArrowShards:
    def test_the_files_come_in_the_order_the_state_lists_them_and_a_cache_file_is_left_out(self, tmp_path):
        arrow = write_folder(tmp_path / "arrow", listed_order=[1, 0])
        names = [shard.name for shard in arrow_shards(arrow, error=CallerError)]
        assert names == ["data-00001-of-00002.arrow", "data-00000-of-00002.arrow"]

    def test_a_folder_without_a_state_is_not_a_saved_dataset(self, tmp_path):
        (tmp_path / "plain").mkdir()
        with pytest.raises(CallerError, match="no state.json"):
            arrow_shards(tmp_path / "plain", error=CallerError)

    def test_a_state_that_lists_nothing_or_a_missing_file_is_refused(self, tmp_path):
        arrow = write_folder(tmp_path / "arrow")
        (arrow / "state.json").write_text(json.dumps({"_data_files": []}))
        with pytest.raises(CallerError, match="lists no data file"):
            arrow_shards(arrow, error=CallerError)
        (arrow / "state.json").write_text(json.dumps({"_data_files": [{"filename": "data-gone.arrow"}]}))
        with pytest.raises(CallerError, match="not on disk"):
            arrow_shards(arrow, error=CallerError)


class TestArrowRows:
    def test_each_row_is_a_tuple_of_the_chosen_columns_in_the_order_asked_across_the_shards_in_order(self, tmp_path):
        arrow = write_folder(tmp_path / "arrow", listed_order=[1, 0])
        rows = list(arrow_rows(arrow_shards(arrow, error=CallerError), ["count", "name"], error=CallerError))
        assert rows == [(3, "c"), (1, "a"), (2, "b")]

    def test_one_column_gives_a_one_element_tuple_and_the_cache_file_is_not_read(self, tmp_path):
        arrow = write_folder(tmp_path / "arrow")
        rows = list(arrow_rows(arrow_shards(arrow, error=CallerError), ["name"], error=CallerError))
        assert rows == [("a",), ("b",), ("c",)]

    def test_a_missing_column_is_refused_with_the_callers_error_naming_the_shard_and_its_columns(self, tmp_path):
        arrow = write_folder(tmp_path / "arrow")
        shards = arrow_shards(arrow, error=CallerError)
        with pytest.raises(
            CallerError, match=r"data-00000-of-00002.arrow has no column 'body'; its columns are \['name', 'count'\]"
        ):
            list(arrow_rows(shards, ["name", "body"], error=CallerError))
