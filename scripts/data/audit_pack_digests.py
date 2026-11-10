#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Audit a built SFT pack against the dataset builder's per-row digests: it must hold exactly the kept rows.

The dataset builder lists a digest for every kept row of a mix (one per line, in source order) and for every dropped
row. A digest is ``ids_hash``: the BLAKE2b with an 8-byte output of the row's token ids as little-endian int32, read
as a signed little-endian int64. The ids are the whole training render with the pinned tokenizer and
``add_special_tokens`` false, including the final end-of-text token.

The packer stores each render followed by padding of the same end-of-text id (``pad_document_for_packing``). So the
audit recovers a document's render by removing all its trailing pad ids and putting back one. A document that has no
trailing pad id, or whose other ids are empty, has no render. The audit lists it as refused and the verdict is FAIL.

The mix repeats some documents by design, so the pack and the kept list are compared as multisets. A digest that occurs
three times in the kept list must occur three times in the pack. Every pack document of every shard is read, and
nothing is sampled. The verdict is PASS only when:

* every shard was read whole and no document was refused,
* the sorted digests of the pack equal the sorted kept list, and
* no dropped digest occurs in the pack.

A shard that is missing, unreadable or malformed (a column missing or of another type, a null, pack starts that do not
split their pack) is a FAIL and is never skipped. The pack is then not compared with the lists, because an unread shard
would make every one of its rows look missing. When all shards were read, a FAIL reports the pack's document count
against the kept list's length, the extra pack documents, the missing kept digests and any dropped digest, each with
its first few locations. A dropped digest that also occurs in the kept list makes the lists contradict each other.
The audit then refuses to run and exits with the status of NOT EVALUATED.

The config is YAML with exactly these keys::

    parquets: /projects/.../shard*/packed/<tokenizer>_pad_seq_to_mult8/training_32768.idx.parquet
                                        # a glob, or a list of paths; every path must exist; read in sorted order
    pad_token_id: 100257                # the end-of-text id of the tokenizer (100257 for Olmo)
    kept_digests: /path/kept_ids_hash_<size>.txt
    dropped_digests: /path/dropped.txt  # or null for none
    workers: 32                         # shards read at once, a process each

Both lists hold one signed decimal integer per line, and the tool reads the dropped list as it reads the kept list. A
line with more than one column, a blank line or a value outside int64 stops the audit and names the file and line.

The report (JSON, never written over an existing file) holds the verdict, the failures, the counts and examples above,
and its inputs: the config and each list with its sha256, the parquet paths, the code revision and the two test
vectors of the digest definition. The tool checks the vectors at start-up and refuses to run if ``ids_hash`` fails
them.

Usage (inside the container)::

    python scripts/data/audit_pack_digests.py --config <audit.yaml> --report-out <report.json>

Exit status 0 on PASS, 1 on FAIL, and 2 when the report already exists, an input cannot be used, or ``ids_hash`` fails
the test vectors.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import multiprocessing
import sys
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml
from numpy.typing import ArrayLike


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.data.file_digest import file_sha256  # noqa: E402
from scripts.data.packed_parquet import MalformedShard, iter_documents  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_int, require_string  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402
from scripts.telemetry.gate_outcome import FAIL, PASS, exit_status, unusable_spec_status  # noqa: E402
from scripts.token_ids import require_token_id  # noqa: E402


logger = logging.getLogger(__name__)

CONFIG_KEYS = frozenset({"parquets", "pad_token_id", "kept_digests", "dropped_digests", "workers"})
IDS_DTYPE = "<i4"
DIGEST_BYTES = 8
EXAMPLES = 10  # how many locations a report lists for each kind of finding
LIST_CHUNK_BYTES = 1 << 24  # how much of a digest list is parsed at once
INT64 = np.iinfo(np.int64)
# Where a document sits in its shard: its pack (a row of the parquet), its index in the pack, its offset and length.
LOCATION = np.dtype([("pack", "<i4"), ("document", "<i4"), ("start", "<i4"), ("length", "<i4")])
DIGEST_DEFINITION = (
    "int.from_bytes(hashlib.blake2b(np.asarray(ids, dtype='<i4').tobytes(), digest_size=8).digest(), 'little', "
    "signed=True) over the ids of the whole training render, add_special_tokens false, with the final end-of-text id"
)
TEST_VECTORS = (
    ([1, 2, 3], -7044357283931317034),
    ([100264, 9125, 198, 100257], -7694233868375670853),
)


class AuditInputError(ValueError):
    """An audit config or digest list that cannot be used."""


class DigestMismatch(RuntimeError):
    """``ids_hash`` does not reproduce the test vectors of the digest definition."""


class NoRender(ValueError):
    """A packed document that is not a render followed by padding."""


# ---------------------------------------------------------------------------------------------
# The digest and a document's render
# ---------------------------------------------------------------------------------------------


def ids_digest(ids: ArrayLike) -> bytes:
    """The 8-byte BLAKE2b of ``ids`` taken as little-endian int32, the bytes ``ids_hash`` reads as an int64."""
    return hashlib.blake2b(np.ascontiguousarray(ids, dtype=IDS_DTYPE), digest_size=DIGEST_BYTES).digest()


def ids_hash(ids: ArrayLike) -> int:
    """The digest of one row's render as the dataset builder lists it: a signed little-endian int64."""
    return int.from_bytes(ids_digest(ids), "little", signed=True)


def check_digest_definition() -> list[dict]:
    """Recompute the definition's test vectors and return them as the report records them.

    Raises:
        DigestMismatch: ``ids_hash`` gives another value for a vector.
    """
    for ids, expected in TEST_VECTORS:
        got = ids_hash(ids)
        if got != expected:
            raise DigestMismatch(f"ids_hash({ids}) is {got}, but the digest definition says {expected}")
    return [{"ids": ids, "digest": expected} for ids, expected in TEST_VECTORS]


def render_ids(ids: np.ndarray, pad_token_id: int) -> np.ndarray:
    """The ids of the render a packed document holds: its ids up to and including the first of its trailing pad ids.

    That is the document with all trailing pad ids removed and exactly one put back, as a view without a copy.

    Raises:
        NoRender: the document does not end in the pad id, or holds nothing but pad ids.
    """
    other = np.flatnonzero(ids != pad_token_id)
    if not len(other):
        raise NoRender(f"every id is the pad id {pad_token_id}, so the render is empty")
    last = int(other[-1])
    if last == len(ids) - 1:
        raise NoRender(f"the last id is {int(ids[last])}, not the pad id {pad_token_id}")
    return ids[: last + 2]


# ---------------------------------------------------------------------------------------------
# The config and the digest lists
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditConfig:
    """An audit config as read: the shards' parquets, the pad id, the two lists and the number of workers."""

    path: Path
    sha256: str
    parquets: tuple[Path, ...]
    pad_token_id: int
    kept_digests: Path
    dropped_digests: Path | None
    workers: int


def resolve_parquets(value: object, where: str) -> tuple[Path, ...]:
    """The shard parquets a config names, sorted: the files a glob matches, or the listed paths.

    Raises:
        AuditInputError: no file matches, a path is listed twice or is not a file, or the value is neither a string nor
            a non-empty list of strings.
    """
    if isinstance(value, str):
        paths = sorted(Path(match) for match in glob.glob(value))
        if not paths:
            raise AuditInputError(f"{where}: no file matches {value!r}")
    elif isinstance(value, list) and value:
        names = [require_string(name, f"{where}[{index}]", error=AuditInputError) for index, name in enumerate(value)]
        paths = sorted(Path(name) for name in names)
        repeated = [str(path) for path, count in Counter(path.resolve() for path in paths).items() if count > 1]
        if repeated:
            raise AuditInputError(f"{where}: lists a file twice, which would count its documents twice: {repeated}")
    else:
        raise AuditInputError(f"{where} must be a glob string or a non-empty list of paths, not {value!r}")
    not_files = [str(path) for path in paths if not path.is_file()]
    if not_files:
        raise AuditInputError(f"{where}: not a file: {not_files}")
    return tuple(paths)


def list_path(value: object, where: str) -> Path:
    """The digest list a config names, which must be a file."""
    path = Path(require_string(value, where, error=AuditInputError))
    if not path.is_file():
        raise AuditInputError(f"{where}: no file at {path}")
    return path


def read_audit_config(path: Path) -> AuditConfig:
    """An audit config, read and refused whole: exactly the keys ``CONFIG_KEYS``. Raises AuditInputError."""
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise AuditInputError(f"{path}: cannot read an audit config: {error}") from error
    config = require_keys(loaded, f"{path}: an audit config", CONFIG_KEYS, error=AuditInputError)
    pad_token_id = require_token_id(config["pad_token_id"], f"{path}: pad_token_id", error=AuditInputError)
    if pad_token_id > np.iinfo(IDS_DTYPE).max:
        raise AuditInputError(f"{path}: pad_token_id {pad_token_id} does not fit {IDS_DTYPE}")
    dropped = config["dropped_digests"]
    return AuditConfig(
        path=path.resolve(),
        sha256=file_sha256(path),
        parquets=resolve_parquets(config["parquets"], f"{path}: parquets"),
        pad_token_id=pad_token_id,
        kept_digests=list_path(config["kept_digests"], f"{path}: kept_digests"),
        dropped_digests=None if dropped is None else list_path(dropped, f"{path}: dropped_digests"),
        workers=require_int(config["workers"], f"{path}: workers", 1, error=AuditInputError),
    )


def is_int64(text: str) -> bool:
    """Whether a line of a digest list is one signed integer that fits int64."""
    try:
        return INT64.min <= int(text) <= INT64.max
    except ValueError:
        return False


def parse_digest_lines(lines: list[str], path: Path, first_line: int) -> np.ndarray:
    """The int64 values of ``lines``, which start at line ``first_line`` of ``path``. Raises AuditInputError."""
    try:
        return np.fromiter(map(int, lines), dtype=np.int64, count=len(lines))
    except (ValueError, OverflowError) as error:
        number, text = next((n, t) for n, t in enumerate(lines, start=first_line) if not is_int64(t))
        raise AuditInputError(f"{path}:{number}: expected one signed 64-bit integer, got {text!r}") from error


def read_digest_list(path: Path) -> np.ndarray:
    """The digests of a list file as one int64 array in file order, parsed in chunks of about ``LIST_CHUNK_BYTES``.

    Raises:
        AuditInputError: the file cannot be read, or a line is not one signed 64-bit integer.
    """
    chunks, number = [], 1
    try:
        with path.open(encoding="utf-8") as handle:
            while lines := handle.readlines(LIST_CHUNK_BYTES):
                chunks.append(parse_digest_lines(lines, path, number))
                number += len(lines)
    except (OSError, UnicodeDecodeError) as error:
        raise AuditInputError(f"{path}: cannot read a digest list: {error}") from error
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# Scanning one shard
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardScan:
    """What a scan of one shard found. ``error`` names why it could not be read whole, and the rest is then empty."""

    error: str | None
    packs: int
    tokens: int
    digests: np.ndarray  # int64, one per document that has a render, in pack order
    locations: np.ndarray  # LOCATION, one per digest
    refused: int  # documents without a render
    refused_examples: tuple[dict, ...]  # the first of them, at most EXAMPLES


def failed_scan(error: str) -> ShardScan:
    """The scan of a shard that could not be read whole."""
    empty = np.empty(0, dtype=np.int64)
    return ShardScan(error, 0, 0, empty, np.empty(0, dtype=LOCATION), 0, ())


def scan_shard(parquet: Path, pad_token_id: int) -> ShardScan:
    """Read one shard's parquet one row group at a time and digest the render of every document.

    A document without a render is counted and its first few are described, and the scan goes on. A shard that cannot
    be read whole returns the reason instead (``failed_scan``).
    """
    import pyarrow as pa

    digests, locations, examples = [], [], []
    packs = tokens = refused = 0
    try:
        for document in iter_documents(parquet, IDS_DTYPE):
            packs, tokens = document.pack + 1, tokens + document.length
            where = (document.pack, document.document, document.start, document.length)
            try:
                render = render_ids(document.ids, pad_token_id)
            except NoRender as problem:
                refused += 1
                if len(examples) < EXAMPLES:
                    examples.append(dict(zip(LOCATION.names, where)) | {"reason": str(problem)})
                continue
            digests.append(ids_digest(render))
            locations.append(where)
    except (OSError, pa.ArrowException, MalformedShard) as error:
        return failed_scan(f"{parquet}: {error}")
    return ShardScan(
        error=None,
        packs=packs,
        tokens=tokens,
        digests=np.frombuffer(b"".join(digests), dtype="<i8"),
        locations=np.array(locations, dtype=LOCATION),
        refused=refused,
        refused_examples=tuple(examples),
    )


def log_scans(parquets: tuple[Path, ...], scans: Iterable[ShardScan]) -> list[ShardScan]:
    """The scans of ``parquets`` in shard order, each logged as it arrives."""
    collected = []
    for parquet, scan in zip(parquets, scans):
        collected.append(scan)
        state = scan.error or f"{scan.packs:,} packs, {len(scan.digests) + scan.refused:,} documents"
        logger.info("%s: %s", parquet, state)
    return collected


def scan_shards(parquets: tuple[Path, ...], pad_token_id: int, workers: int) -> list[ShardScan]:
    """Scan every shard, ``workers`` at a time, each in a process of its own (in this process when one worker would
    do); the scans in shard order."""
    pool_size = min(workers, len(parquets))
    if pool_size == 1:
        return log_scans(parquets, map(scan_shard, parquets, repeat(pad_token_id)))
    # Spawned, not forked: a forked child inherits whatever threads the parent holds (pyarrow's pool among them).
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=pool_size, mp_context=context) as pool:
        return log_scans(parquets, pool.map(scan_shard, parquets, repeat(pad_token_id)))


# ---------------------------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------------------------


class Beyond(NamedTuple):
    """The elements of a sorted array that occur more often than in another."""

    mask: np.ndarray  # over the array: the later copies of a value, beyond the other array's count of it
    absent: np.ndarray  # over the array: the value does not occur in the other array at all


def beyond_other(values: np.ndarray, other: np.ndarray) -> Beyond:
    """For sorted ``values``, which elements exceed the multiplicity of their value in sorted ``other``."""
    held = np.searchsorted(other, values, side="right") - np.searchsorted(other, values, side="left")
    rank = np.arange(len(values)) - np.searchsorted(values, values, side="left")
    return Beyond(rank >= held, held == 0)


def occurs(sorted_values: np.ndarray, value: int) -> bool:
    """Whether ``value`` is one of the elements of a sorted array."""
    left = int(np.searchsorted(sorted_values, value, side="left"))
    return left < len(sorted_values) and bool(sorted_values[left] == value)


def distinct(sorted_values: np.ndarray) -> int:
    """The number of different values in a sorted array."""
    return int(np.count_nonzero(np.diff(sorted_values))) + 1 if len(sorted_values) else 0


def compare_scans(
    parquets: tuple[Path, ...], scans: list[ShardScan], kept: np.ndarray, dropped: np.ndarray | None
) -> tuple[dict, list[str]]:
    """Compare the digests of every shard with the kept list and the dropped list.

    Returns the report's comparison section and the failures found, which are none only when the pack's digests equal
    the kept list as multisets and no dropped digest occurs in them.
    """
    digests = np.concatenate([scan.digests for scan in scans])
    locations = np.concatenate([scan.locations for scan in scans])
    first_of_shard = np.cumsum([0] + [len(scan.digests) for scan in scans])

    def locate(index: int) -> dict:
        shard = int(np.searchsorted(first_of_shard, index, side="right")) - 1
        found = dict(zip(LOCATION.names, locations[index].tolist()))
        return {"parquet": str(parquets[shard]), **found, "digest": int(digests[index])}

    order = np.argsort(digests, kind="stable")
    packed = digests[order]
    kept_sorted = np.sort(kept)
    exact = len(packed) == len(kept_sorted) and bool(np.array_equal(packed, kept_sorted))
    extras = {"count": 0, "digest_not_in_kept_list": 0, "repeated_beyond_kept_list": 0, "examples": []}
    missing = {"count": 0, "examples": []}
    if not exact:
        surplus = beyond_other(packed, kept_sorted)
        at = np.sort(order[surplus.mask])  # where the extra documents sit in pack order
        extras = {
            "count": len(at),
            "digest_not_in_kept_list": int(np.count_nonzero(surplus.mask & surplus.absent)),
            "repeated_beyond_kept_list": int(np.count_nonzero(surplus.mask & ~surplus.absent)),
            "examples": [
                locate(index) | {"in_kept_list": occurs(kept_sorted, digests[index])} for index in at[:EXAMPLES]
            ],
        }
        shortfall = beyond_other(kept_sorted, packed)
        lines = np.flatnonzero(np.isin(kept, kept_sorted[shortfall.mask]))
        missing = {
            "count": int(np.count_nonzero(shortfall.mask)),
            "examples": [{"kept_line": int(line) + 1, "digest": int(kept[line])} for line in lines[:EXAMPLES]],
        }
    comparison = {
        "pack_documents": len(packed),
        "pack_distinct_digests": distinct(packed),
        "kept_digests": len(kept),
        "kept_distinct_digests": distinct(kept_sorted),
        "extra_documents": extras,
        "missing_digests": missing,
        "dropped_digests_present": None,
    }
    failures = []
    if len(packed) != len(kept):
        failures.append(f"the pack holds {len(packed):,} documents and the kept list {len(kept):,} digests")
    if extras["count"]:
        failures.append(
            f"{extras['count']:,} pack documents are not matched by the kept list "
            f"({extras['digest_not_in_kept_list']:,} with a digest the list lacks, "
            f"{extras['repeated_beyond_kept_list']:,} repeated beyond the list's own count)"
        )
    if missing["count"]:
        failures.append(f"{missing['count']:,} kept digests are missing from the pack")
    if dropped is not None:
        listed = np.unique(dropped)
        at = np.flatnonzero(np.isin(digests, listed))
        comparison["dropped_digests_present"] = {
            "listed": len(listed),
            "present": len(np.unique(digests[at])),
            "documents": len(at),
            "examples": [locate(index) for index in at[:EXAMPLES]],
        }
        if len(at):
            failures.append(f"{len(at):,} pack documents carry a digest of the dropped list")
    return comparison, failures


def shard_report(parquet: Path, scan: ShardScan) -> dict:
    """One shard's part of the report."""
    return {
        "parquet": str(parquet),
        "error": scan.error,
        "packs": scan.packs,
        "documents": len(scan.digests) + scan.refused,
        "tokens": scan.tokens,
        "refused_documents": scan.refused,
    }


def list_report(path: Path | None, digests: np.ndarray | None) -> dict | None:
    """What the report records of a digest list: its path, sha256 and number of digests, or None for no list."""
    if path is None:
        return None
    return {"path": str(path), "sha256": file_sha256(path), "digests": len(digests)}


def run_audit(config: AuditConfig) -> dict:
    """Audit every shard against the lists and return the report, whose ``verdict`` is PASS only when nothing failed.

    Raises:
        DigestMismatch: ``ids_hash`` fails the test vectors of the digest definition.
        AuditInputError: a digest list cannot be read, the kept list is empty, or a digest is in both lists.
    """
    vectors = check_digest_definition()
    kept = read_digest_list(config.kept_digests)
    if not len(kept):
        raise AuditInputError(f"{config.kept_digests}: the kept list holds no digest, so there is nothing to audit")
    dropped = None if config.dropped_digests is None else read_digest_list(config.dropped_digests)
    if dropped is not None:
        shared = np.unique(dropped[np.isin(dropped, kept)])
        if len(shared):
            raise AuditInputError(
                f"{len(shared):,} digests are in both {config.kept_digests} and {config.dropped_digests}, "
                f"so the pack cannot show whether a row was dropped (first: {shared[:EXAMPLES].tolist()})"
            )
    scans = scan_shards(config.parquets, config.pad_token_id, config.workers)
    shards = [shard_report(parquet, scan) for parquet, scan in zip(config.parquets, scans)]
    failures = [f"shard {scan.error}" for scan in scans if scan.error]
    unreadable = bool(failures)
    refused = [
        {"parquet": str(parquet), **example}
        for parquet, scan in zip(config.parquets, scans)
        for example in scan.refused_examples
    ]
    refused_count = sum(scan.refused for scan in scans)
    if refused_count:
        failures.append(f"{refused_count:,} pack documents have no render, the first at {refused[0]}")
    comparison = None
    if unreadable:
        failures.append("the pack was not compared with the lists, because a shard could not be read whole")
    else:
        comparison, found = compare_scans(config.parquets, scans, kept, dropped)
        failures += found
    return {
        "verdict": FAIL if failures else PASS,
        "failures": failures,
        "config": {"path": str(config.path), "sha256": config.sha256},
        "code_revision": code_revision(str(REPO_ROOT)),
        "digest_definition": {"text": DIGEST_DEFINITION, "test_vectors": vectors},
        "pad_token_id": config.pad_token_id,
        "kept_list": list_report(config.kept_digests, kept),
        "dropped_list": list_report(config.dropped_digests, dropped),
        "shards": shards,
        "totals": {
            "packs": sum(shard["packs"] for shard in shards),
            "documents": sum(shard["documents"] for shard in shards),
            "tokens": sum(shard["tokens"] for shard in shards),
        },
        "refused_documents": {"count": refused_count, "examples": refused},
        "comparison": comparison,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the audit a config states, write the report and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", type=Path, required=True, help="the audit config: the shards, the pad id, the lists"
    )
    parser.add_argument(
        "--report-out", type=Path, required=True, help="write the report here as JSON, an existing file is refused"
    )
    args = parser.parse_args(argv)
    if args.report_out.exists():
        parser.error(f"{args.report_out} exists")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = run_audit(read_audit_config(args.config))
    except (AuditInputError, DigestMismatch) as error:
        return unusable_spec_status(args.config, error)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("x") as out:
        json.dump(report, out, indent=1)
        out.write("\n")
    comparison = report["comparison"]
    print(f"pack documents: {report['totals']['documents']:,}, refused: {report['refused_documents']['count']:,}")
    if comparison is not None:
        extras, missing = comparison["extra_documents"]["count"], comparison["missing_digests"]["count"]
        print(f"kept digests: {comparison['kept_digests']:,}, extra documents: {extras:,}")
        print(f"missing digests: {missing:,}")
    print(f"report: {args.report_out}")
    for failure in report["failures"]:
        print(f"  - {failure}", file=sys.stderr)
    print(f"verdict: {report['verdict']}")
    return exit_status([report["verdict"]])


if __name__ == "__main__":
    sys.exit(main())
