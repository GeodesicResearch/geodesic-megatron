#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Mix packed parquet files into one packed parquet file, with a fixed number of packs of each source in every batch.

A fine-tune that replays an earlier training set reads its own packs and packs of that set in one pass. This tool writes
that pass as one file, so that the order of every pack is in the file and the file has one hash. One config states
everything the tool does: the output, the path of the record, the seed, the number of batches, the row-group size of the
output and, for each source, a file, a glob or a directory of packed parquet files, how many packs of it every batch
holds and how its packs are chosen.

* ``all``: every pack of the source, in file order. The source must hold exactly ``batches`` times ``packs_per_batch``
  packs.
* ``first``: the first ``batches`` times ``packs_per_batch`` packs of the source, in file order. The source must hold at
  least that many, and its last packs stay out of the mix.
* ``sample``: ``batches`` times ``packs_per_batch`` packs drawn without replacement and uniformly from all the packs of
  the source, in the order drawn. The files of a source count in the order that ``resolve_packed_parquet_paths`` gives.

Batch k holds the k-th group of packs of each source, and the order of the packs inside a batch is a seeded permutation.
A pack is copied unchanged, so its tokens, loss mask and sequence starts are the source's. The tool refuses to
overwrite an output, and it writes the output under a ``.partial`` name, reads it back, requires that each row equals
the row that was meant for it and only then gives it its name. It writes a record at the path of the config: the seed,
the sha256 of the output and of every source file, and the order of the packs (the source and the pack's index in it).

Usage: ``python scripts/data/mix_packed_parquet.py <config.yaml>``

Catalogue:
    MixError: the config or the sources cannot make the mix.
    Source, Mix: the sources and the whole of the config.
    PackedSource: the files of a source and the offsets of their rows and row groups.
    PACK_COLUMNS, DRAWS: the columns that a pack has, and the ways to choose the packs of a source.
    load_mix: read and check the config.
    open_source: the files of a source and where each pack of it sits.
    plan_mix: the draw and the order of every pack of the output.
    read_packs: the packs at the given indices of a source.
    row_digests: the sha256 of each row of a table of packs.
    assemble: the output table, read from the sources in the order planned.
    write_verified: write the table, read it back and give the file its name.
    build_record: the record of a written mix.
    mix_files: write the output and its record.
    main: the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


# The repo root, for the modules of scripts/, when this runs as a script from scripts/data.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.file_digest import HASH_WORKERS, file_sha256  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402

from megatron.bridge.data.datasets.packed_parquet import (  # noqa: E402
    REQUIRED_COLUMNS,
    locate_row,
    resolve_packed_parquet_paths,
    row_offsets,
    write_packed_parquet,
)


PACK_COLUMNS = tuple(sorted(REQUIRED_COLUMNS))
DRAWS = ("all", "first", "sample")


class MixError(ValueError):
    """The config or the sources cannot make the mix that the config states."""


@dataclass(frozen=True)
class Source:
    """One source of the mix: ``packs_per_batch`` packs of ``pattern`` in every batch, chosen as ``draw`` says."""

    name: str
    pattern: str
    packs_per_batch: int
    draw: str


@dataclass(frozen=True)
class Mix:
    """The whole config."""

    output: Path
    record: Path
    seed: int
    batches: int
    row_group_size: int
    sources: tuple[Source, ...]


@dataclass(frozen=True)
class PackedSource:
    """The files of a source, and the cumulative offsets of their rows and row groups (``row_offsets``)."""

    files: tuple[Path, ...]
    file_offsets: tuple[int, ...]
    row_group_offsets: tuple[tuple[int, ...], ...]

    @property
    def packs(self) -> int:
        return self.file_offsets[-1]

    def locate(self, index: int) -> tuple[int, int, int]:
        """The file, the row group in it and the row in that group of pack ``index``, counted over all the files."""
        if not 0 <= index < self.packs:
            raise IndexError(f"pack {index} is outside the {self.packs} packs of the source")
        return locate_row(self.file_offsets, self.row_group_offsets, index)


def load_mix(path: Path) -> Mix:
    """The config at ``path``, checked: exactly its keys, positive counts, known draws and distinct source names."""
    raw = yaml.safe_load(Path(path).read_text())
    require_keys(raw, str(path), {"output", "record", "seed", "batches", "row_group_size", "sources"}, error=MixError)
    if not isinstance(raw["sources"], list) or not raw["sources"]:
        raise MixError(f"{path}: sources must be a list of at least one source")
    sources = []
    for index, item in enumerate(raw["sources"]):
        where = f"{path}: sources[{index}]"
        require_keys(item, where, {"name", "path", "packs_per_batch", "draw"}, error=MixError)
        if item["draw"] not in DRAWS:
            raise MixError(f"{where}: draw must be one of {DRAWS}, got {item['draw']!r}")
        if not isinstance(item["packs_per_batch"], int) or item["packs_per_batch"] < 1:
            raise MixError(f"{where}: packs_per_batch must be a positive integer, got {item['packs_per_batch']!r}")
        sources.append(Source(item["name"], item["path"], item["packs_per_batch"], item["draw"]))
    if len({source.name for source in sources}) != len(sources):
        raise MixError(f"{path}: the source names must differ, got {[source.name for source in sources]}")
    for key in ("batches", "row_group_size"):
        if not isinstance(raw[key], int) or raw[key] < 1:
            raise MixError(f"{path}: {key} must be a positive integer, got {raw[key]!r}")
    return Mix(
        Path(raw["output"]), Path(raw["record"]), raw["seed"], raw["batches"], raw["row_group_size"], tuple(sources)
    )


def open_source(pattern: str) -> PackedSource:
    """The files that ``pattern`` names (``resolve_packed_parquet_paths``), with the offsets of their rows and row groups."""
    try:
        files = tuple(Path(path) for path in resolve_packed_parquet_paths(pattern))
    except ValueError as error:
        raise MixError(f"source {pattern!r}: {error}") from error
    file_offsets, row_group_offsets = row_offsets([pq.ParquetFile(file).metadata for file in files])
    return PackedSource(files, tuple(file_offsets), tuple(tuple(offsets) for offsets in row_group_offsets))


def plan_mix(mix: Mix, sources: Sequence[PackedSource]) -> list[tuple[int, int]]:
    """The output in order: for every pack, the index of its source in ``mix.sources`` and its index in that source.

    The draws of the ``sample`` sources come first, in the order of the sources, and then the permutation of each batch,
    all from one generator seeded with ``mix.seed``."""
    rng = np.random.default_rng(mix.seed)
    chosen = []
    for config, source in zip(mix.sources, sources):
        needed = mix.batches * config.packs_per_batch
        if config.draw == "all":
            if source.packs != needed:
                raise MixError(
                    f"source {config.name!r} draws all its packs, so it must hold {mix.batches} batches of "
                    f"{config.packs_per_batch} packs ({needed}), and it holds {source.packs}"
                )
            chosen.append(np.arange(needed))
        elif config.draw == "first":
            if source.packs < needed:
                raise MixError(
                    f"source {config.name!r} holds {source.packs} packs and the mix draws its first {needed}"
                )
            chosen.append(np.arange(needed))
        else:
            if needed > source.packs:
                raise MixError(f"source {config.name!r} holds {source.packs} packs and the mix draws {needed}")
            chosen.append(rng.choice(source.packs, size=needed, replace=False))
    order = []
    for batch in range(mix.batches):
        members = [
            (source_index, int(pack))
            for source_index, config in enumerate(mix.sources)
            for pack in chosen[source_index][batch * config.packs_per_batch : (batch + 1) * config.packs_per_batch]
        ]
        order.extend(members[position] for position in rng.permutation(len(members)))
    return order


def read_packs(source: PackedSource, indices: Sequence[int]) -> pa.Table:
    """The packs at ``indices`` of ``source``, in that order, with the schema of the source's files.

    Each row group that holds a wanted pack is read once, and only the wanted rows of it are kept."""
    places = [source.locate(index) for index in indices]
    wanted: dict[tuple[int, int], set[int]] = {}
    for file_index, group, row in places:
        wanted.setdefault((file_index, group), set()).add(row)
    rows: dict[tuple[int, int, int], pa.Table] = {}
    for (file_index, group), group_rows in sorted(wanted.items()):
        table = pq.ParquetFile(source.files[file_index]).read_row_group(group, columns=list(PACK_COLUMNS))
        for row in group_rows:
            rows[(file_index, group, row)] = table.slice(row, 1)
    return pa.concat_tables([rows[place] for place in places])


def row_digests(table: pa.Table) -> list[str]:
    """The sha256 of each row's tokens, loss mask and sequence starts."""
    digests = []
    for row in table.to_pylist():
        digest = hashlib.sha256()
        for column in PACK_COLUMNS:
            digest.update(np.asarray(row[column], dtype=np.int64).tobytes())
        digests.append(digest.hexdigest())
    return digests


def assemble(sources: Sequence[PackedSource], order: Sequence[tuple[int, int]]) -> pa.Table:
    """The output table: the packs of ``order`` read from their sources."""
    per_source = [
        read_packs(source, [pack for source_index, pack in order if source_index == index])
        for index, source in enumerate(sources)
    ]
    schemas = {table.schema for table in per_source}
    if len(schemas) != 1:
        raise MixError(f"the sources have different schemas: {[str(schema) for schema in schemas]}")
    next_row = [0] * len(sources)
    pieces = []
    for source_index, _ in order:
        pieces.append(per_source[source_index].slice(next_row[source_index], 1))
        next_row[source_index] += 1
    return pa.concat_tables(pieces)


def write_verified(table: pa.Table, output: Path, row_group_size: int) -> None:
    """Write ``table`` to ``output`` through a ``.partial`` file that is read back, and given its name once each row
    equals the row of ``table``. Creates the folders of ``output``. Raises ``MixError``, and leaves no file, when a row
    differs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    write_packed_parquet(table.to_pylist(), partial, row_group_size)
    if row_digests(pq.read_table(partial)) != row_digests(table):
        partial.unlink()
        raise MixError(f"the rows read back from {partial} differ from the rows that were written")
    os.replace(partial, output)


def build_record(
    mix: Mix, sources: Sequence[PackedSource], order: Sequence[tuple[int, int]], packs_written: int
) -> dict:
    """The record of a written mix: the seed, the sha256 of the output and of every file of every source, and the order."""
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as pool:
        source_hashes = [list(pool.map(file_sha256, source.files)) for source in sources]
    return {
        "seed": mix.seed,
        "batches": mix.batches,
        "row_group_size": mix.row_group_size,
        "output": str(mix.output),
        "output_packs": packs_written,
        "output_sha256": file_sha256(mix.output),
        "sources": [
            {
                "name": config.name,
                "pattern": config.pattern,
                "draw": config.draw,
                "packs_per_batch": config.packs_per_batch,
                "packs_available": source.packs,
                "files": [{"path": str(file), "sha256": digest} for file, digest in zip(source.files, hashes)],
            }
            for config, source, hashes in zip(mix.sources, sources, source_hashes)
        ],
        "order": [[mix.sources[source_index].name, pack] for source_index, pack in order],
    }


def mix_files(mix: Mix) -> dict:
    """Write the output and the record of ``mix``, and return the record.

    Reads the sources, writes and verifies the output (``write_verified``), hashes the sources and writes the record.
    Raises ``MixError`` before anything is written when the output exists or the sources cannot make the mix."""
    if mix.output.exists():
        raise MixError(f"{mix.output} exists, and the tool does not overwrite an output")
    sources = [open_source(config.pattern) for config in mix.sources]
    order = plan_mix(mix, sources)
    table = assemble(sources, order)
    write_verified(table, mix.output, mix.row_group_size)
    record = build_record(mix, sources, order, table.num_rows)
    mix.record.parent.mkdir(parents=True, exist_ok=True)
    mix.record.write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv: Sequence[str] | None = None) -> None:
    """Write the mix that the config named on the command line states, and print where the output went."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, help="the YAML config of the mix")
    record = mix_files(load_mix(parser.parse_args(argv).config))
    print(f"wrote {record['output_packs']} packs to {record['output']}, sha256 {record['output_sha256']}")


if __name__ == "__main__":
    main()
