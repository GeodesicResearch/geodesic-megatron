# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A saved Arrow folder: the data files that its ``state.json`` lists, in order, and their rows.

Hugging Face ``save_to_disk`` writes a folder of ``data-*.arrow`` stream files and a ``state.json`` that lists them. A
tool that reads such a folder takes its files and rows from here, so that every tool agrees on which files are the dataset
and in what order.

Catalogue:
    arrow_shards: the data files of a saved Arrow folder, in order.
    arrow_rows: the values of chosen columns for each row of those files, in order.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa


def arrow_shards(arrow_dir: Path, *, error: type[Exception]) -> list[Path]:
    """The data files of a saved Arrow folder, in the order its ``state.json`` lists them.

    Args:
        arrow_dir: The saved folder.
        error: The exception type to raise, so that a caller reports a bad folder in its own error type.

    Raises:
        error: The folder has no ``state.json``, lists no data file, or lists one that is missing. A file in the folder
            that ``state.json`` does not list (a cache file) is not read.
    """
    state = arrow_dir / "state.json"
    if not state.is_file():
        raise error(f"{arrow_dir} has no state.json, so it is not a saved Arrow dataset")
    shards = [arrow_dir / entry["filename"] for entry in json.loads(state.read_text()).get("_data_files", [])]
    if not shards:
        raise error(f"{state} lists no data file")
    missing = [str(shard) for shard in shards if not shard.is_file()]
    if missing:
        raise error(f"{state} lists data files that are not on disk: {missing[:3]}")
    return shards


def arrow_rows(shards: Sequence[Path], columns: Sequence[str], *, error: type[Exception]) -> Iterator[tuple[Any, ...]]:
    """The values of ``columns`` for each row of ``shards``, in order, as one tuple of Python values for each row.

    The files are read as memory-mapped Arrow streams, one batch at a time.

    Args:
        shards: The data files, as ``arrow_shards`` returns them.
        columns: The columns to read, in the order of the tuples.
        error: The exception type to raise for a shard that lacks a column.

    Raises:
        error: A shard has no column named in ``columns``; the message names the shard and its columns.
    """
    for shard in shards:
        with pa.memory_map(str(shard)) as mapped:
            for batch in pa.ipc.open_stream(mapped):
                absent = [name for name in columns if name not in batch.schema.names]
                if absent:
                    raise error(f"{shard} has no column {absent[0]!r}; its columns are {batch.schema.names}")
                yield from zip(*(batch.column(name).to_pylist() for name in columns))
