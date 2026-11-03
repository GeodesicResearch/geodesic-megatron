#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Turn the dataset builder's list of dropped rows into the ``--exclude-rows`` file of ``pack_sft_dataset.py``.

The builder numbers a dropped row by ``source_row``, its index in the whole mix. The shards hold the mix in order, so a
row's shard is the last one that starts at or before it, and its line in that shard's ``training.jsonl`` is
``source_row - start``. The offsets file gives the ``start`` and the ``lines`` of each shard.

The config is YAML with exactly these keys::

    dropped_rows: /path/dropped_rows_<size>.tsv   # tab-separated, with a header row that has a source_row column
    offsets: /path/offsets.json                   # JSON: "shards": [{"shard", "start", "lines"}, ...], "total_lines"
    expected_count: 2255                          # how many rows the builder says it dropped
    out: /path/exclude_rows_<size>.jsonl          # one {"shard": N, "line": L} per row, in source_row order

The tool refuses a source_row that is not an integer inside the mix, a source_row listed twice, a row count other than
``expected_count``, shard starts that do not follow from the shards' line counts, and an ``out`` that exists. It writes
``out`` and, beside it, ``<out>.record.json``: the config, the sha256 of each input and of the output, the rows per
shard and the code revision.

Usage (any Python with PyYAML)::

    python scripts/data/dropped_rows_to_exclusions.py --config <config.yaml>

Exit status 0 when the file is written, 2 when an input cannot be used.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.data.file_digest import file_sha256  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_int, require_string  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402
from scripts.telemetry.gate_outcome import unusable_spec_status  # noqa: E402


CONFIG_KEYS = frozenset({"dropped_rows", "offsets", "expected_count", "out"})


class ExclusionInputError(ValueError):
    """A config, dropped-rows list or offsets report that cannot be used."""


@dataclass(frozen=True)
class ExclusionConfig:
    """The inputs of one conversion, as the config file states them (see the module docstring)."""

    path: Path
    sha256: str
    dropped_rows: Path
    offsets: Path
    expected_count: int
    out: Path


@dataclass(frozen=True)
class ShardStarts:
    """The shards of a mix in order: the first line of each, and the number of lines in the whole mix."""

    starts: tuple[int, ...]
    total_lines: int

    def locate(self, source_row: int) -> tuple[int, int]:
        """The shard and the line in it of ``source_row``."""
        if not 0 <= source_row < self.total_lines:
            raise ExclusionInputError(f"source_row {source_row} is outside the mix of {self.total_lines} lines")
        shard = bisect.bisect_right(self.starts, source_row) - 1
        return shard, source_row - self.starts[shard]


def read_exclusion_config(path: Path) -> ExclusionConfig:
    """A config, read and refused whole: exactly the keys ``CONFIG_KEYS``. Raises ExclusionInputError."""
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ExclusionInputError(f"{path}: cannot read an exclusion config: {error}") from error
    config = require_keys(loaded, f"{path}: an exclusion config", CONFIG_KEYS, error=ExclusionInputError)
    where = f"{path}: "
    return ExclusionConfig(
        path=path.resolve(),
        sha256=file_sha256(path),
        dropped_rows=Path(require_string(config["dropped_rows"], where + "dropped_rows", error=ExclusionInputError)),
        offsets=Path(require_string(config["offsets"], where + "offsets", error=ExclusionInputError)),
        expected_count=require_int(config["expected_count"], where + "expected_count", 1, error=ExclusionInputError),
        out=Path(require_string(config["out"], where + "out", error=ExclusionInputError)),
    )


def read_shard_starts(path: Path) -> ShardStarts:
    """The shard starts of an offsets report. Each shard must start where the one before it ends."""
    try:
        report = json.loads(path.read_text())
        entries = report["shards"]
        total_lines = report["total_lines"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExclusionInputError(
            f"{path}: cannot read the shards and total_lines of an offsets report: {error!r}"
        ) from error
    starts: list[int] = []
    expected_start = 0
    for number, entry in enumerate(entries):
        where = f"{path}: shards[{number}]"
        if require_int(entry.get("shard"), f"{where}.shard", 0, error=ExclusionInputError) != number:
            raise ExclusionInputError(f"{where}: shards must be listed in order, 0 first")
        start = require_int(entry.get("start"), f"{where}.start", 0, error=ExclusionInputError)
        if start != expected_start:
            raise ExclusionInputError(f"{where}: starts at {start}, but the shards before it end at {expected_start}")
        starts.append(start)
        expected_start += require_int(entry.get("lines"), f"{where}.lines", 0, error=ExclusionInputError)
    if not starts or expected_start != total_lines:
        raise ExclusionInputError(f"{path}: the shards hold {expected_start} lines, but total_lines is {total_lines}")
    return ShardStarts(starts=tuple(starts), total_lines=total_lines)


def read_source_rows(path: Path) -> list[int]:
    """The ``source_row`` column of a tab-separated dropped-rows list, in file order, each row listed once."""
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or "source_row" not in reader.fieldnames:
                raise ExclusionInputError(f"{path}: the header row has no source_row column")
            values = [row["source_row"] for row in reader]
    except OSError as error:
        raise ExclusionInputError(f"{path}: cannot read the dropped rows: {error}") from error
    rows = [int(value) if value.isdecimal() else None for value in values]
    if None in rows:
        raise ExclusionInputError(f"{path}: source_row {values[rows.index(None)]!r} is not a non-negative integer")
    repeated = sorted(row for row, count in Counter(rows).items() if count > 1)
    if repeated:
        raise ExclusionInputError(f"{path}: source_row listed more than once, first {repeated[:5]}")
    return rows


def exclusion_entries(source_rows: list[int], shards: ShardStarts) -> list[dict[str, int]]:
    """The ``--exclude-rows`` entries of ``source_rows``, in source_row order."""
    return [dict(zip(("shard", "line"), shards.locate(row))) for row in sorted(source_rows)]


def run(config: ExclusionConfig) -> dict:
    """Write the exclusion file and its record, and return the record. Raises ExclusionInputError."""
    shards = read_shard_starts(config.offsets)
    source_rows = read_source_rows(config.dropped_rows)
    if len(source_rows) != config.expected_count:
        raise ExclusionInputError(
            f"{config.dropped_rows}: lists {len(source_rows)} rows, but expected_count is {config.expected_count}"
        )
    entries = exclusion_entries(source_rows, shards)
    if config.out.exists():
        raise ExclusionInputError(f"{config.out} exists")
    config.out.parent.mkdir(parents=True, exist_ok=True)
    with config.out.open("x") as out:
        out.writelines(json.dumps(entry) + "\n" for entry in entries)
    record = {
        "config": {"path": str(config.path), "sha256": config.sha256},
        "dropped_rows": {"path": str(config.dropped_rows), "sha256": file_sha256(config.dropped_rows)},
        "offsets": {"path": str(config.offsets), "sha256": file_sha256(config.offsets)},
        "out": {"path": str(config.out), "sha256": file_sha256(config.out), "rows": len(entries)},
        "rows_per_shard": {str(shard): count for shard, count in sorted(Counter(e["shard"] for e in entries).items())},
        "code_revision": code_revision(str(REPO_ROOT)),
    }
    record_path = config.out.with_name(config.out.name + ".record.json")
    with record_path.open("x") as handle:
        json.dump(record, handle, indent=1)
        handle.write("\n")
    return record


def main(argv: list[str] | None = None) -> int:
    """Run the conversion a config states and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the exclusion config: the inputs and the output")
    args = parser.parse_args(argv)
    try:
        record = run(read_exclusion_config(args.config))
    except ExclusionInputError as error:
        return unusable_spec_status(args.config, error)
    print(f"{record['out']['rows']} rows in {len(record['rows_per_shard'])} shards: {record['out']['path']}")
    print(f"sha256: {record['out']['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
