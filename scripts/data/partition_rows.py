#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Split the rows of a JSONL file into shards: in contiguous ranges, or at random from a seed with listed rows left out.

Each shard is a dataset root under ``output_root`` holding its rows as ``training.jsonl``, in the subdirectory
``pipeline.data.layout.shard_name`` names. The config's ``order`` chooses how rows go to shards:

* ``contiguous``: shard ``i`` holds the lines that start in the ``i``-th of ``shards`` equal byte ranges of the source,
  in source order, so the shards are near equal in bytes and every line stays whole. This order takes no seed and no
  exclusions; configs/control_pretraining/shard_jsonl_corpus.sh splits a corpus with it.
* ``random``: the kept rows are drawn in a random order from ``seed`` and cut into ``shards`` parts whose sizes differ
  by at most one row, so each shard is a uniform sample even of a source whose rows are grouped. Each shard keeps its
  rows in source order. The same source, exclusions and seed give the same shards.

The config is YAML with exactly these keys::

    source: /path/to/corpus/training.jsonl     # absolute; every line ends with a newline
    output_root: /path/to/partition             # absolute
    shards: 32                                  # at least 1
    order: random                               # contiguous or random
    seed: 1                                     # random: an integer >= 0; contiguous: null
    exclude_lines: /path/to/over_length.jsonl   # random: absolute, or null for none; contiguous: null
    remove_source: false                        # true: delete the source once the shards are checked and recorded

An exclusion file is the count pass's ``over_length.jsonl`` (``scripts/data/row_token_counts.py``): one JSON object per
line with exactly the integer keys ``line``, a 0-based line of the source, and ``tokens``. The count pass's summary
beside it must name the source file at its present size.

The tool refuses an output root that already holds a shard's file or a record, an exclusion file of another form or of
another source, an exclusion that names a line twice or a line the source does not have, a source whose last line has
no final newline, and a shard that would hold no row. It checks that the shards hold every kept row once: their bytes
on disk add up to the source's bytes less the excluded rows' bytes. In random order their rows must also add up to the
source's rows less the excluded ones. Contiguous order counts the source's rows from the rows it copies, so there the
byte check is the only test.

Outputs under ``output_root``: the shards, ``partition_record.json`` (the config, the exclusion file's sha256, the
source's rows and bytes, the excluded rows and bytes, and each shard's rows and bytes) and ``partition_config.yaml``, a
copy of the config.

Usage::

    python scripts/data/partition_rows.py --config /path/to/partition.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.data.integer_entries import read_integer_entries  # noqa: E402
from scripts.data.row_token_counts import counted_source  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_absolute_path, require_int  # noqa: E402

from pipeline.data.layout import shard_name  # noqa: E402


logger = logging.getLogger(__name__)

CONFIG_KEYS = frozenset({"source", "output_root", "shards", "order", "seed", "exclude_lines", "remove_source"})
ORDERS = ("contiguous", "random")
EXCLUSION_KEYS = ("line", "tokens")
RECORD = "partition_record.json"
CONFIG_COPY = "partition_config.yaml"
BLOCK_BYTES = 1 << 24


class PartitionError(ValueError):
    """A config, an exclusion file, a source or an output root that the partition cannot use, or shards that do not
    hold every kept row once."""


def read_partition_config(path: Path) -> dict:
    """The mapping a partition config holds, refused unless it has exactly the keys ``CONFIG_KEYS`` with values its
    ``order`` accepts.

    Raises:
        PartitionError: the file cannot be read, a key is unknown or missing, or a value is of the wrong type or not
            one its order takes.
    """
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise PartitionError(f"{path}: cannot read a partition config: {error}") from error
    config = require_keys(loaded, f"{path}: a partition config", CONFIG_KEYS, error=PartitionError)
    require_absolute_path(config["source"], f"{path}: source", error=PartitionError)
    require_absolute_path(config["output_root"], f"{path}: output_root", error=PartitionError)
    require_int(config["shards"], f"{path}: shards", 1, error=PartitionError)
    if type(config["remove_source"]) is not bool:
        raise PartitionError(f"{path}: remove_source must be true or false, not {config['remove_source']!r}")
    if config["order"] == "random":
        require_int(config["seed"], f"{path}: seed", 0, error=PartitionError)
        if config["exclude_lines"] is not None:
            require_absolute_path(config["exclude_lines"], f"{path}: exclude_lines", error=PartitionError)
    elif config["order"] == "contiguous":
        if config["seed"] is not None or config["exclude_lines"] is not None:
            raise PartitionError(f"{path}: contiguous order takes no seed and no exclude_lines; set both to null")
    else:
        raise PartitionError(f"{path}: order must be one of {list(ORDERS)}, not {config['order']!r}")
    return config


def require_counts_of(path: Path, source: Path) -> None:
    """Raise PartitionError unless the count pass that wrote the exclusion file ``path`` counted ``source`` at its
    present size, as the summary beside ``path`` states (``row_token_counts.counted_source``)."""
    try:
        counted, counted_bytes = counted_source(path.parent)
    except (OSError, ValueError, KeyError) as error:
        raise PartitionError(f"{path}: the count pass's summary beside it cannot be read: {error!r}") from error
    if Path(counted).resolve() != source.resolve() or counted_bytes != source.stat().st_size:
        raise PartitionError(
            f"{path} lists rows of {counted} at {counted_bytes} bytes, not of {source} at {source.stat().st_size} bytes"
        )


def read_excluded_lines(path: Path) -> tuple[np.ndarray, str]:
    """The source lines an exclusion file names, ascending, and the file's sha256.

    Raises:
        PartitionError: a line of the file is not an object with exactly the integer keys ``EXCLUSION_KEYS``, or a
            source line is named twice.
    """
    entries, digest = read_integer_entries(path, EXCLUSION_KEYS, PartitionError)
    excluded = np.array(sorted(line for line, _tokens in entries), dtype=np.int64)
    repeated = excluded[1:][excluded[1:] == excluded[:-1]]
    if repeated.size:
        raise PartitionError(f"{path} names source line {int(repeated[0])} more than once")
    return excluded, digest


def require_final_newline(source: Path) -> None:
    """Raise PartitionError when the last line of ``source`` has no final newline."""
    size = source.stat().st_size
    if not size:
        return
    with source.open("rb") as handle:
        handle.seek(size - 1)
        if handle.read(1) != b"\n":
            raise PartitionError(f"{source}: the last line has no final newline")


def count_lines(source: Path) -> int:
    """The number of newlines in ``source``."""
    count = 0
    with source.open("rb") as handle:
        while block := handle.read(BLOCK_BYTES):
            count += block.count(b"\n")
    return count


def random_shards(rows: int, excluded: np.ndarray, shards: int, seed: int) -> np.ndarray:
    """The shard of each of the ``rows`` source lines, -1 for the ``excluded`` ones: the kept lines in a random order
    from ``seed``, cut into ``shards`` parts whose sizes differ by at most one.

    Raises:
        PartitionError: an excluded line is not a line of the source, or fewer lines are kept than there are shards.
    """
    if excluded.size and excluded[-1] >= rows:
        raise PartitionError(f"an exclusion names source line {int(excluded[-1])}, but the source has {rows} lines")
    kept = np.setdiff1d(np.arange(rows, dtype=np.int64), excluded, assume_unique=True)
    if kept.size < shards:
        raise PartitionError(f"{kept.size} source lines are kept, too few for {shards} shards of at least one row")
    order = np.random.default_rng(seed).permutation(kept.size)
    shard = np.full(rows, -1, dtype=np.int64)
    shard[kept[order]] = np.arange(kept.size, dtype=np.int64) * shards // kept.size
    return shard


def contiguous_cuts(source: Path, shards: int) -> list[int]:
    """The ``shards + 1`` byte offsets that bound the shards of ``source`` in contiguous order: each inner cut is the
    first line start at or after an equal fraction of the source's size.

    Raises:
        PartitionError: a shard would hold no line (a line longer than a shard's share of the bytes).
    """
    size = source.stat().st_size
    cuts = [0]
    with source.open("rb") as handle:
        for index in range(1, shards):
            handle.seek(max(-(-index * size // shards) - 1, 0))
            handle.readline()
            cuts.append(handle.tell())
    cuts.append(size)
    empty = [index for index in range(shards) if cuts[index] == cuts[index + 1]]
    if empty:
        raise PartitionError(f"{source}: shard {empty[0]} of {shards} would hold no line; use fewer shards")
    return cuts


def shard_path(output_root: Path, index: int) -> Path:
    """The training file of shard ``index`` under ``output_root``."""
    return output_root / shard_name(index) / "training.jsonl"


def write_random_shards(source: Path, output_root: Path, shard: np.ndarray, shards: int) -> dict:
    """Write each source line to the file of its shard (``shard[line]``; -1 leaves it out), in source order, and
    return the rows written per shard and the rows and bytes left out."""
    rows = [0] * shards
    excluded_rows = excluded_bytes = 0
    for index in range(shards):
        shard_path(output_root, index).parent.mkdir(parents=True, exist_ok=True)
    handles = [shard_path(output_root, index).open("wb") for index in range(shards)]
    try:
        with source.open("rb") as handle:
            for line, text in enumerate(handle):
                target = int(shard[line])
                if target < 0:
                    excluded_rows += 1
                    excluded_bytes += len(text)
                    continue
                handles[target].write(text)
                rows[target] += 1
    finally:
        for output in handles:
            output.close()
    return {"shard_rows": rows, "excluded_rows": excluded_rows, "excluded_bytes": excluded_bytes}


def write_contiguous_shards(source: Path, output_root: Path, cuts: list[int]) -> dict:
    """Copy the bytes of ``source`` between consecutive ``cuts`` to the shard files in order, and return the rows
    written per shard (none are left out).

    Raises:
        PartitionError: the source ends before a cut (it changed while it was copied).
    """
    rows = []
    with source.open("rb") as handle:
        for index in range(len(cuts) - 1):
            path = shard_path(output_root, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle.seek(cuts[index])
            remaining = cuts[index + 1] - cuts[index]
            count = 0
            with path.open("wb") as output:
                while remaining:
                    block = handle.read(min(remaining, BLOCK_BYTES))
                    if not block:
                        raise PartitionError(f"{source} ended before byte {cuts[index + 1]}")
                    output.write(block)
                    count += block.count(b"\n")
                    remaining -= len(block)
            rows.append(count)
    return {"shard_rows": rows, "excluded_rows": 0, "excluded_bytes": 0}


def check_partition(written: dict, source_rows: int, source_bytes: int, excluded: int) -> None:
    """Raise PartitionError unless the shards hold every kept row once: their rows are the source's less the
    ``excluded`` count, and their bytes on disk (``shard_bytes``) are the source's less the excluded rows' bytes."""
    if written["excluded_rows"] != excluded or sum(written["shard_rows"]) != source_rows - excluded:
        raise PartitionError(
            f"the shards hold {sum(written['shard_rows'])} rows and {written['excluded_rows']} were left out, "
            f"but the source has {source_rows} rows of which {excluded} are excluded"
        )
    if sum(written["shard_bytes"]) + written["excluded_bytes"] != source_bytes:
        raise PartitionError(
            f"the shards hold {sum(written['shard_bytes'])} bytes and the excluded rows {written['excluded_bytes']}, "
            f"but the source has {source_bytes} bytes"
        )


def main(argv: list[str] | None = None) -> None:
    """Partition the source a partition config names and write the outputs of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the partition config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    config = read_partition_config(args.config)
    source = Path(config["source"])
    output_root = Path(config["output_root"])
    shards = config["shards"]
    present = [
        str(path)
        for path in [output_root / RECORD, *(shard_path(output_root, index) for index in range(shards))]
        if path.exists()
    ]
    if present:
        raise PartitionError(f"{output_root} already holds {present[:3]}; remove them to partition again")
    if config["exclude_lines"] is None:
        excluded, exclusion_digest = np.empty(0, dtype=np.int64), None
    else:
        require_counts_of(Path(config["exclude_lines"]), source)
        excluded, exclusion_digest = read_excluded_lines(Path(config["exclude_lines"]))
    require_final_newline(source)
    source_bytes = source.stat().st_size
    if config["order"] == "contiguous":
        written = write_contiguous_shards(source, output_root, contiguous_cuts(source, shards))
        # The copy counts the rows as it reads the source once; the byte check on disk below is this order's gate.
        source_rows = sum(written["shard_rows"])
    else:
        source_rows = count_lines(source)
        written = write_random_shards(
            source, output_root, random_shards(source_rows, excluded, shards, config["seed"]), shards
        )
    written["shard_bytes"] = [shard_path(output_root, index).stat().st_size for index in range(shards)]
    check_partition(written, source_rows, source_bytes, int(excluded.size))
    record = {
        "config": config,
        "exclude_lines_sha256": exclusion_digest,
        "source_rows": source_rows,
        "source_bytes": source_bytes,
        **written,
    }
    (output_root / RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(args.config, output_root / CONFIG_COPY)
    if config["remove_source"]:
        source.unlink()
    logger.info(
        "%d rows of %s in %d shards (%s order) under %s, %d excluded%s",
        source_rows - excluded.size,
        source,
        shards,
        config["order"],
        output_root,
        excluded.size,
        "; the source is removed" if config["remove_source"] else "",
    )


if __name__ == "__main__":
    main()
