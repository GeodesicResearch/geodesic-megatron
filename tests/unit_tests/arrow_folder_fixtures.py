# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A saved Arrow folder for the tests of the tools that read one. This module needs pyarrow and nothing else.

Catalogue:
    write_arrow_folder: a saved Arrow folder of the given tables, with a state and an unlisted cache file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa


def write_arrow_folder(
    folder: Path, tables: list[pa.Table], cache_table: pa.Table, *, listed_order: list[int] | None = None
) -> Path:
    """A saved Arrow folder: one ``data-*.arrow`` stream file for each table, a ``state.json`` that lists them (in
    ``listed_order`` when given), and a cache file of ``cache_table`` that the state does not list and whose rows must
    never be read."""
    folder.mkdir(parents=True)
    names = [f"data-{index:05d}-of-{len(tables):05d}.arrow" for index in range(len(tables))]
    for name, table in zip(names, tables):
        with pa.OSFile(str(folder / name), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    with pa.OSFile(str(folder / "cache-0123456789abcdef.arrow"), "wb") as sink:
        with pa.ipc.new_stream(sink, cache_table.schema) as writer:
            writer.write_table(cache_table)
    order = listed_order if listed_order is not None else list(range(len(tables)))
    (folder / "state.json").write_text(json.dumps({"_data_files": [{"filename": names[i]} for i in order]}))
    return folder
