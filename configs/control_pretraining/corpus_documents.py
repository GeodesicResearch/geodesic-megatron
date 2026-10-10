#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Per-document operations on a tokenized corpus: check documents against their dataset, check digests, select
documents.

Every operation reads a ``.bin/.idx`` through Megatron's own index reader and streams the ids in
chunks of whole documents through numpy memmap views; nothing loops over tokens in Python.

A document is what ``tools/preprocess_data.py --append-eod`` writes for one dataset row: the
row's ids followed by the EOD, as one sequence, or, for an empty text, no sequence and no ids at
all (an empty document, which therefore has no EOD).

* **Per-document checks** (``check_documents``, run by ``verify_corpora.py`` for every table row
  that declares the document-check columns). Document ``i`` of the built corpus must hold exactly
  as many of ``count_token`` as the prepare config's dataset states in ``count_column`` at row
  ``i`` (the document's last position, its EOD, is excluded by position), and every non-empty
  document must end in the EOD the corpus was tokenized with (``appended_eod``); the whole
  ``.bin`` must hold exactly the column's sum, so a token sitting in an EOD slot is caught too;
  the corpus must have one document per dataset row; and the dataset's ``row_column`` must hold
  ``first_row + i`` at row ``i``, so its rows, and with them the documents, are the source's in
  order. Only those two columns are read from the dataset: by range request on the Hub, or with
  the same column projection from a local copy laid out as the Hub repository is
  (``hub_parquet``). The text is never read.
* **Digests** (``check-hashes``). A digest list holding, per row of a corpus's source subset,
  ``n_tokens`` (its ids without the EOD), ``ids_hash`` (the 64-bit BLAKE2b of those ids as
  little-endian int32 bytes, read as a little-endian signed integer) and ``source_row`` (its row
  index), is checked against every document: its length (``n_tokens`` + 1, or 0 for a row of no
  ids, which is an empty document), the EOD at the last position of a non-empty one (by
  position: an EOD id inside a document is an ordinary token), and the digest of the ids before
  it. Mismatches are counted and reported by class, with examples; nothing is written. What is
  checked against what is a config's statement, not the command line's (``read_digest_checks``):
  the config names the corpora table and, per subset, where its digest list is — a
  dataset-builder local build, or a Hub dataset config at a commit — or that it is still pending.
  The EOD id is derived from the corpus's own tokenize record and tokenizer, and the digest list's
  dataset-builder record must name the corpus's source subset and tokenizer.
* **Select** (``select``, the build job of a select row). The documents a kept list names are
  copied from the parent corpus once, in order, ids byte for byte, into a new prefix with its own
  ``.idx``. The output is written under temporary names, re-read and compared document by
  document with the parent, and scanned for the select config's ``absent_token_ids``: a kept
  document holding one at any position, its last included, refuses the selection. Only then is
  the output given its final names and its provenance record, which states those ids;
  ``verify_corpora.py`` scans the final files for them again.

Usage (inside the container; as a job through ``corpus_job.sbatch``)::

    python configs/control_pretraining/corpus_documents.py select <table> <subset> [--shard N]
    python configs/control_pretraining/corpus_documents.py check-hashes --config <digest checks.yaml> \\
        --subset <subset> [--report-out <json>]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpora_table import (  # noqa: E402
    DATA_BASE,
    REPO_ROOT,
    Checker,
    CorpusRow,
    SelectedCorpus,
    SelectedPrefix,
    read_corpora_table,
    selected_corpus,
    subset_prepare_config,
    tokenized_prefixes,
)
from hub_parquet import hub_file_url, hub_parquet_files, local_parquet_files, read_hub_file  # noqa: E402


if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from scripts.data.prepare_revisions import FULL_SHA, tokenizer_revision  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402


TOOL_VERSION = 2  # raised whenever what a select writes, or how it is checked, changes
TOKEN_DTYPE = np.dtype("<i4")
CHUNK_TOKENS = 1 << 26  # ids streamed per step: 256 MiB of int32
EXAMPLES = 20  # mismatching documents named per check
HUB_READ_THREADS = 16  # parquet files of a Hub config read at once, each by range request
HASH_CLASSES = ("length", "eod", "hash")
PARTIAL = ".partial"  # suffix of a select's output until every check has passed

# A digest-check config: the corpora table, and per tokenize row of it, one digest source.
DIGEST_CONFIG_KEYS = frozenset({"table", "subsets"})
DIGEST_SOURCES = ("saved", "hub", "pending")
HUB_SOURCE_KEYS = frozenset({"dataset", "revision", "config"})
DIGEST_COLUMNS = ("n_tokens", "ids_hash", "source_row")
# The dataset-builder step that writes a digest list, and the columns it names for its outputs.
DIGEST_KERNEL = "tokenize_count"
DIGEST_KERNEL_OUTPUTS = {"output_column": "n_tokens", "ids_hash_column": "ids_hash"}


class CorpusCheckFailed(Exception):
    """A corpus, or an input describing it, is not what it must be."""


def bin_path(prefix: Path, suffix: str = "") -> Path:
    """A prefix's ids file (``suffix`` names a not-yet-final copy)."""
    return Path(f"{prefix}.bin{suffix}")


def idx_path(prefix: Path, suffix: str = "") -> Path:
    """A prefix's index file (``suffix`` names a not-yet-final copy)."""
    return Path(f"{prefix}.idx{suffix}")


def provenance_path(prefix: Path) -> Path:
    """A prefix's record, named as the tokenize step names its own, so readers of that record find it."""
    return Path(f"{prefix}.provenance.json")


def read_record(path: Path) -> dict:
    """A JSON record a check relies on; a missing or malformed one is a failure of the check."""
    if not path.is_file():
        raise CorpusCheckFailed(f"missing {path}")
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise CorpusCheckFailed(f"{path}: expected a JSON object")
    return record


def recorded(record: dict, where: object, *keys: str):
    """``record[keys[0]][keys[1]]...``; a record that lacks one of them is refused, never read as empty."""
    value = record
    for depth, key in enumerate(keys):
        if not isinstance(value, dict) or key not in value:
            raise CorpusCheckFailed(f"{where}: records no {'.'.join(keys[: depth + 1])}")
        value = value[key]
    return value


@dataclass(frozen=True)
class DocumentIndex:
    """A corpus's documents: each one's length (its EOD included; 0 for an empty document) and first position in
    the ``.bin``."""

    sizes: np.ndarray
    starts: np.ndarray

    @property
    def docs(self) -> int:
        return len(self.sizes)

    @property
    def sequences(self) -> int:
        """How many sequences the ``.idx`` holds: one per non-empty document."""
        return int(np.count_nonzero(self.sizes))

    @property
    def tokens(self) -> int:
        return int(self.sizes.sum())


def read_index(index_file: Path, data_file: Path) -> DocumentIndex:
    """Read an ``.idx`` with Megatron's reader and check that it describes ``data_file`` as a corpus of documents.

    The campaign's corpora are what ``preprocess_data.py`` writes: per document one non-empty
    sequence ending in its EOD, or no sequence for an empty text, as int32 ids laid end to end. So
    the index must say exactly that: int32, at most one sequence per document, no empty sequence,
    pointers contiguous from 0, and a ``.bin`` exactly as long.
    """
    from megatron.core.datasets.indexed_dataset import _IndexReader

    for path in (index_file, data_file):
        if not path.is_file():
            raise CorpusCheckFailed(f"missing {path}")
    index = _IndexReader(str(index_file), multimodal=False)
    if np.dtype(index.dtype) != np.dtype(np.int32):
        raise CorpusCheckFailed(f"{index_file}: ids are {np.dtype(index.dtype)}, not int32")
    lengths = index.sequence_lengths.astype(np.int64)
    boundaries = index.document_indices.astype(np.int64)
    per_document = np.diff(boundaries)
    if boundaries[0] != 0 or boundaries[-1] != len(lengths) or not np.isin(per_document, (0, 1)).all():
        raise CorpusCheckFailed(f"{index_file}: not at most one sequence per document")
    empty = np.flatnonzero(lengths < 1)
    if empty.size:
        raise CorpusCheckFailed(
            f"{index_file}: sequence {empty[0]} is empty; an empty document has no sequence, and every sequence ends "
            "in its EOD"
        )
    sizes = np.zeros(len(per_document), dtype=np.int64)
    sizes[per_document == 1] = lengths
    starts = np.zeros(len(sizes), dtype=np.int64)
    np.cumsum(sizes[:-1], out=starts[1:])
    if not np.array_equal(index.sequence_pointers, starts[per_document == 1] * TOKEN_DTYPE.itemsize):
        raise CorpusCheckFailed(f"{index_file}: the documents are not laid end to end from the start of the .bin")
    expected = int(sizes.sum()) * TOKEN_DTYPE.itemsize
    if data_file.stat().st_size != expected:
        raise CorpusCheckFailed(f"{data_file} is {data_file.stat().st_size} bytes, its index describes {expected}")
    return DocumentIndex(sizes, starts)


def token_memmap(path: Path) -> np.ndarray:
    """A ``.bin``'s ids, memory-mapped; numpy cannot map an empty file, which holds none."""
    if path.stat().st_size == 0:
        return np.zeros(0, dtype=TOKEN_DTYPE)
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def document_chunks(index: DocumentIndex, chunk_tokens: int = CHUNK_TOKENS) -> list[tuple[int, int]]:
    """``[lo, hi)`` document ranges of about ``chunk_tokens`` ids each; a longer document is a chunk alone."""
    ends = index.starts + index.sizes
    chunks, lo = [], 0
    while lo < index.docs:
        hi = max(int(np.searchsorted(ends, index.starts[lo] + chunk_tokens, side="right")), lo + 1)
        chunks.append((lo, hi))
        lo = hi
    return chunks


def chunk_views(data_file: Path, index: DocumentIndex) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """``(lo, hi, view, starts)``: documents ``[lo, hi)`` as one memmap view, and each one's start within it."""
    tokens = token_memmap(data_file)
    for lo, hi in document_chunks(index):
        first = int(index.starts[lo])
        last = int(index.starts[hi - 1] + index.sizes[hi - 1])
        yield lo, hi, tokens[first:last], index.starts[lo:hi] - first


def file_sha256(path: Path, block: int = 1 << 26) -> str:
    """The sha256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------
# Reading columns of a dataset without its text
# ---------------------------------------------------------------------------------------------


def _parquet_columns(source, columns: list[str], name: str):
    """The named columns of one parquet file (a path or an open handle), refusing a file that lacks one."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(source, pre_buffer=True)
    missing = [column for column in columns if column not in parquet.schema_arrow.names]
    if missing:
        raise CorpusCheckFailed(f"{name} has no column {missing}; it has {parquet.schema_arrow.names}")
    # The files of one config are concatenated, and each carries its writer's schema metadata.
    return parquet.read(columns=columns).replace_schema_metadata(None)


def dataset_columns(dataset: str, revision: str | None, config: str, columns: list[str]):
    """The named columns of a dataset config's train split, in the order the loader concatenates its files.

    ``dataset`` is a Hub dataset, read at ``revision`` (a full commit SHA) by range request, so that
    only these columns' chunks and each file's footer are fetched; or a local directory laid out as
    the Hub repository is, read with the same column projection. Both choose the config's files by
    ``hub_parquet``'s one rule. A local directory is one snapshot, so ``revision`` is not consulted
    for it.
    """
    import pyarrow as pa

    local = Path(dataset)
    if local.is_dir():
        try:
            files = local_parquet_files(local, config)
        except FileNotFoundError as error:
            raise CorpusCheckFailed(str(error)) from error
        return pa.concat_tables([_parquet_columns(path, columns, str(path)) for path in files])

    if not FULL_SHA.fullmatch(str(revision)):
        raise CorpusCheckFailed(
            f"{dataset}: revision {revision!r} is not a full commit SHA, so it names no fixed data"
        )
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()

    def read(path: str):
        url = hub_file_url(dataset, revision, path)
        return read_hub_file(filesystem, url, lambda handle: _parquet_columns(handle, columns, url))

    try:
        files = hub_parquet_files(dataset, revision, config)
    except FileNotFoundError as error:
        raise CorpusCheckFailed(str(error)) from error
    with ThreadPoolExecutor(HUB_READ_THREADS) as pool:
        return pa.concat_tables(list(pool.map(read, files)))


def saved_columns(stage_dir: Path, columns: list[str]):
    """The named columns of a dataset-builder local build: ``<stage_dir>/train``, as ``save_to_disk`` wrote it."""
    import datasets

    split = stage_dir / "train"
    if not split.is_dir():
        raise CorpusCheckFailed(f"{stage_dir}: no saved train split at {split}")
    saved = datasets.load_from_disk(str(split))
    missing = [column for column in columns if column not in saved.column_names]
    if missing:
        raise CorpusCheckFailed(f"{split} has no column {missing}; it has {saved.column_names}")
    # The arrow format applies any indices mapping, so row i of the table is row i of the split.
    return saved.select_columns(columns).with_format("arrow")[:]


def integer_column(table, name: str) -> np.ndarray:
    """One integer column of an arrow table as int64, refusing a missing column, nulls and any other type."""
    import pyarrow as pa

    if name not in table.column_names:
        raise CorpusCheckFailed(f"no column {name!r}; the table has {table.column_names}")
    column = table.column(name)
    if not pa.types.is_integer(column.type):
        raise CorpusCheckFailed(f"column {name!r} is {column.type}, not integers")
    if column.null_count:
        raise CorpusCheckFailed(f"column {name!r} has {column.null_count} nulls")
    return np.asarray(column.to_numpy(), dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# Per-document checks
# ---------------------------------------------------------------------------------------------


# What ``last_ids`` reports for an empty document, which has no last id. It is only reported: a corrupt document can
# end in this very value, so whether a document is empty is read from its size, never from this.
NO_ID = -1


def last_ids(view: np.ndarray, starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Each document's last id, read from a chunk ``view`` at ``starts`` with ``sizes``; ``NO_ID`` for an empty one."""
    last = np.full(len(sizes), NO_ID, dtype=np.int64)
    present = sizes > 0
    last[present] = view[starts[present] + sizes[present] - 1]
    return last


def lacks_eod(last: np.ndarray, sizes: np.ndarray, eod_id: int) -> np.ndarray:
    """The documents that hold ids (``sizes`` > 0) but whose last id (``last_ids``) is not ``eod_id``.

    ``--append-eod`` ends every non-empty document in the EOD and writes nothing for an empty text,
    so an empty document is never missing one.
    """
    return (sizes > 0) & (last != eod_id)


@dataclass(frozen=True)
class DocumentScan:
    """What one pass over a corpus's ids measures, per document and in all."""

    counts: np.ndarray  # each document's count of the token, its last position (the EOD slot) excluded
    total: int  # the token's count over every position of the .bin
    last: np.ndarray  # each document's last id (``last_ids``)


def scan_documents(data_file: Path, index: DocumentIndex, token: int) -> DocumentScan:
    """Count ``token`` in every document of a ``.bin`` and read each one's last id, in one pass over it."""
    counts = np.zeros(index.docs, dtype=np.int64)
    last = np.zeros(index.docs, dtype=np.int64)
    total = 0
    for lo, hi, view, starts in chunk_views(data_file, index):
        sizes = index.sizes[lo:hi]
        last[lo:hi] = last_ids(view, starts, sizes)
        hits = np.flatnonzero(view == token)
        total += int(hits.size)
        if hits.size:
            ends = starts + sizes
            document = np.searchsorted(ends, hits, side="right")
            counted = hits != ends[document] - 1
            counts[lo:hi] += np.bincount(document[counted], minlength=hi - lo)
    return DocumentScan(counts, total, last)


def documents_holding(scan: DocumentScan, token: int) -> np.ndarray:
    """The documents a ``scan_documents`` pass for ``token`` found it in, at any position, the last one included."""
    return np.flatnonzero((scan.counts > 0) | (scan.last == token))


def check_absent_tokens(data_file: Path, index: DocumentIndex, token_ids: tuple[int, ...], rows: np.ndarray) -> None:
    """Refuse a ``.bin`` any of whose documents holds one of ``token_ids``, naming each such document by its
    position and by ``rows``, the parent subset's row each document was copied from; one scan per id."""
    for token in token_ids:
        holding = documents_holding(scan_documents(data_file, index, token), token)
        if holding.size:
            named = ", ".join(f"document {d} (parent row {rows[d]})" for d in holding[:EXAMPLES].tolist())
            raise CorpusCheckFailed(
                f"{data_file}: token {token}, which the select config lists as absent, is held by {holding.size} "
                f"of its {index.docs} documents: {named}"
            )


def _name_failures(checker: Checker, messages: list[str], named: int) -> int:
    """Record ``messages`` while fewer than ``EXAMPLES`` of their kind are named; return how many now are."""
    room = max(EXAMPLES - named, 0)
    for message in messages[:room]:
        checker.expect(False, message)
    return named + min(len(messages), room)


def _check_source_order(row: CorpusRow, positions: np.ndarray, checker: Checker) -> int:
    """Check that dataset row ``i`` holds source index ``first_row + i`` in ``row_column``; return how many do not."""
    expected = row.first_row + np.arange(len(positions), dtype=np.int64)
    misplaced = np.flatnonzero(positions != expected)
    _name_failures(
        checker,
        [
            f"{row.subset}: row {r} holds {row.row_column} {positions[r]}, the source's row there is {expected[r]}"
            for r in misplaced[:EXAMPLES].tolist()
        ],
        0,
    )
    if misplaced.size > EXAMPLES:
        checker.expect(False, f"{row.subset}: {misplaced.size} rows in all are not the source's row in order")
    return int(misplaced.size)


def check_documents(row: CorpusRow, scalars: dict, checker: Checker, data_base: Path = DATA_BASE) -> dict:
    """Run a tokenize row's per-document checks (see the module docstring) and return the measurements.

    ``scalars`` is the row's prepare config; the dataset's ``row.count_column`` and
    ``row.row_column`` are read for the config named by the row's subset, and its row ``beg + i``
    is document ``i`` of the prefix that holds rows ``[beg, end)`` (``tokenized_prefixes``). The
    EOD is the one ``appended_eod`` derives from the corpus's tokenize records. Each failure is
    recorded on ``checker`` — up to ``EXAMPLES`` of each kind naming a document or row, then one
    counting the rest.
    """
    token, column = row.count_token, row.count_column
    # A prepare config may leave the revision out; dataset_columns then refuses a Hub read, and a
    # local copy, which is one snapshot, has none to consult.
    revision = scalars.get("revision")
    report: dict = {
        "token": token,
        "column": column,
        "row_column": row.row_column,
        "first_row": row.first_row,
        "dataset": scalars["dataset"],
        "revision": revision,
    }
    try:
        eod = appended_eod(row, scalars, data_base)
        table = dataset_columns(scalars["dataset"], revision, row.subset, [column, row.row_column])
        expected = integer_column(table, column)
        positions = integer_column(table, row.row_column)
        prefixes = tokenized_prefixes(row, data_base)
    except (CorpusCheckFailed, ValueError, OSError) as error:
        checker.expect(False, f"{row.subset}: the per-document checks could not run: {error}")
        return report
    report.update(rows=len(expected), expected_total=int(expected.sum()), eod=eod)
    checker.expect(
        len(expected) == row.docs, f"{row.subset}: {column} has {len(expected)} rows, the table says {row.docs}"
    )
    report["rows_out_of_order"] = _check_source_order(row, positions, checker)

    documents = total = miscounted = without_eod = named_counts = named_eods = 0
    for entry in prefixes:
        label = row.subset if entry.shard is None else f"{row.subset} shard{entry.shard}"
        beg, end = entry.rows
        want = expected[beg:end]
        try:
            index = read_index(idx_path(entry.prefix), bin_path(entry.prefix))
        except CorpusCheckFailed as error:
            checker.expect(False, f"{label}: {error}")
            continue
        if not checker.expect(
            index.docs == len(want),
            f"{label}: holds {index.docs} documents, {column}'s rows {beg}:{end} number {len(want)}",
        ):
            continue
        scan = scan_documents(bin_path(entry.prefix), index, token)
        documents += index.docs
        total += scan.total
        wrong = np.flatnonzero(scan.counts != want)
        miscounted += int(wrong.size)
        named_counts = _name_failures(
            checker,
            [
                f"{label}: document {d} (row {beg + d}) holds {scan.counts[d]} of token {token}, {column} says {want[d]}"
                for d in wrong[:EXAMPLES].tolist()
            ],
            named_counts,
        )
        unended = np.flatnonzero(lacks_eod(scan.last, index.sizes, eod["id"]))
        without_eod += int(unended.size)
        named_eods = _name_failures(
            checker,
            [
                f"{label}: document {d} (row {beg + d}) ends in id {scan.last[d]}, not the EOD {eod['id']}"
                for d in unended[:EXAMPLES].tolist()
            ],
            named_eods,
        )
        checker.expect(
            scan.total == int(want.sum()),
            f"{label}: the .bin holds {scan.total} of token {token}, {column} sums to {int(want.sum())} over its rows",
        )
    if miscounted > named_counts:
        checker.expect(
            False, f"{row.subset}: {miscounted} documents in all hold a count of token {token} {column} does not"
        )
    if without_eod > named_eods:
        checker.expect(False, f"{row.subset}: {without_eod} non-empty documents in all do not end in the EOD")
    checker.expect(
        total == int(expected.sum()),
        f"{row.subset}: the corpus holds {total} of token {token}, {column} sums to {int(expected.sum())}",
    )
    report.update(documents=documents, total=total, mismatched_documents=miscounted, documents_without_eod=without_eod)
    return report


# ---------------------------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------------------------


def document_digests(view: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Each document's ids digest: 64-bit BLAKE2b of its first ``lengths`` ids as little-endian int32 bytes, signed."""
    raw = memoryview(view).cast("B")
    width = TOKEN_DTYPE.itemsize
    digests = bytearray(8 * len(lengths))
    for k, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
        digests[8 * k : 8 * k + 8] = hashlib.blake2b(
            raw[width * start : width * (start + length)], digest_size=8
        ).digest()
    return np.frombuffer(bytes(digests), dtype="<i8")


def check_hashes(row: CorpusRow, payload, eod_id: int, data_base: Path = DATA_BASE) -> dict:
    """Check every document of a tokenized corpus against a digest list's ``n_tokens`` and ``ids_hash``, by class.

    Row ``beg + i`` of the list describes document ``i`` of the prefix holding rows ``[beg, end)``.
    A list whose row count differs from the corpus's, or whose ``source_row`` is not its own row
    index (rows dropped, repeated or reordered on the way), cannot be aligned, so it is reported as
    that and nothing else. Per document: ``length`` (the document is not ``n_tokens`` + 1 long, or,
    for a row of no ids, not empty: an empty text is written as no sequence at all), ``eod`` (it
    holds ids and its last id is not ``eod_id``) and ``hash`` (the digest of all but its last id,
    of no ids for an empty document, is not ``ids_hash``, judged only where the length agrees).
    """
    n_tokens = integer_column(payload, "n_tokens")
    ids_hash = integer_column(payload, "ids_hash")
    source_row = integer_column(payload, "source_row")
    report: dict = {"subset": row.subset, "rows": len(n_tokens), "documents": row.docs, "eod_id": eod_id}
    report["row_count_matches"] = len(n_tokens) == row.docs
    report["source_rows_in_order"] = bool(np.array_equal(source_row, np.arange(len(source_row))))
    report["mismatches"] = {name: 0 for name in HASH_CLASSES}
    report["prefixes"] = []
    if not (report["row_count_matches"] and report["source_rows_in_order"]):
        report["ok"] = False
        return report

    for entry in tokenized_prefixes(row, data_base):
        beg, end = entry.rows
        index = read_index(idx_path(entry.prefix), bin_path(entry.prefix))
        if index.docs != end - beg:
            raise CorpusCheckFailed(
                f"{entry.prefix}: holds {index.docs} documents, its rows {beg}:{end} number {end - beg}"
            )
        bad = {name: np.zeros(index.docs, dtype=bool) for name in HASH_CLASSES}
        last = np.zeros(index.docs, dtype=np.int64)
        digest = np.zeros(index.docs, dtype=np.int64)
        rows = n_tokens[beg:end]
        bad["length"] = index.sizes != np.where(rows > 0, rows + 1, 0)
        ids = np.maximum(index.sizes - 1, 0)  # each document's ids before its EOD; an empty one has neither
        for lo, hi, view, starts in chunk_views(bin_path(entry.prefix), index):
            last[lo:hi] = last_ids(view, starts, index.sizes[lo:hi])
            digest[lo:hi] = document_digests(view, starts, ids[lo:hi])
        bad["eod"] = lacks_eod(last, index.sizes, eod_id)
        bad["hash"] = (digest != ids_hash[beg:end]) & ~bad["length"]
        examples = {}
        for name in HASH_CLASSES:
            report["mismatches"][name] += int(bad[name].sum())
            examples[name] = [
                {
                    "document": document,
                    "row": beg + document,
                    "ids": int(ids[document]),
                    "n_tokens": int(n_tokens[beg + document]),
                    "last_id": int(last[document]),
                    "digest": int(digest[document]),
                    "ids_hash": int(ids_hash[beg + document]),
                }
                for document in np.flatnonzero(bad[name])[:EXAMPLES].tolist()
            ]
        report["prefixes"].append(
            {
                "shard": entry.shard,
                "prefix": str(entry.prefix),
                "rows": [beg, end],
                "mismatches": {name: int(bad[name].sum()) for name in HASH_CLASSES},
                "examples": examples,
            }
        )
    report["ok"] = not any(report["mismatches"].values())
    return report


@dataclass(frozen=True)
class HubDigests:
    """A digest list published on the Hub: one config of a dataset, at a commit."""

    dataset: str
    revision: str
    config: str


@dataclass(frozen=True)
class DigestCheck:
    """One subset's digest check, as a digest-check config states it: exactly one of ``saved`` and ``hub`` is set."""

    config: Path
    config_sha256: str
    row: CorpusRow
    saved: Path | None  # a dataset-builder local build's stage directory, holding train/ and _provenance.json
    hub: HubDigests | None


def _digest_source(entry, where: str) -> tuple[str, object]:
    """``(kind, value)`` of one subset's entry: a saved directory, a Hub config, or a pending reason."""
    source = require_keys(entry, where, frozenset(), frozenset(DIGEST_SOURCES), error=CorpusCheckFailed)
    kinds = [kind for kind in DIGEST_SOURCES if kind in source]
    if len(kinds) != 1:
        raise CorpusCheckFailed(f"{where}: names exactly one of {list(DIGEST_SOURCES)}, not {kinds}")
    (kind,) = kinds
    value = source[kind]
    if kind == "hub":
        hub = require_keys(value, f"{where}.hub", HUB_SOURCE_KEYS, error=CorpusCheckFailed)
        if not all(isinstance(hub[key], str) and hub[key] for key in HUB_SOURCE_KEYS):
            raise CorpusCheckFailed(f"{where}.hub: {sorted(HUB_SOURCE_KEYS)} must be non-empty strings")
        if not FULL_SHA.fullmatch(hub["revision"]):
            raise CorpusCheckFailed(f"{where}.hub: revision {hub['revision']!r} is not a full commit SHA")
        return kind, HubDigests(hub["dataset"], hub["revision"], hub["config"])
    if not isinstance(value, str) or not value.strip():
        raise CorpusCheckFailed(f"{where}.{kind}: must be a non-empty string")
    # A saved directory is named absolute or relative to the repo root, as the tables name their configs.
    return kind, (REPO_ROOT / value if kind == "saved" else value)


def read_digest_checks(path: Path, subset: str) -> DigestCheck:
    """One subset's digest check from a digest-check config, which is read and refused whole.

    The config holds exactly ``table`` (a corpora table, named relative to the repo root) and
    ``subsets``, which must name every tokenize row of that table and nothing else, each with
    exactly one source: ``saved: <dataset-builder stage directory>``,
    ``hub: {dataset, revision, config}`` (the revision a full commit SHA), or
    ``pending: <why there is no digest list yet>``. A pending subset is refused when asked for, and
    no entry is skipped when it is not.
    """
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise CorpusCheckFailed(f"{path}: cannot read a digest-check config: {error}") from error
    config = require_keys(loaded, f"{path}: a digest-check config", DIGEST_CONFIG_KEYS, error=CorpusCheckFailed)
    if not isinstance(config["table"], str) or not config["table"]:
        raise CorpusCheckFailed(f"{path}: table must name a corpora table")
    table = REPO_ROOT / config["table"]
    if not table.is_file():
        raise CorpusCheckFailed(f"{path}: no corpora table at {table}")
    rows = {row.subset: row for row in read_corpora_table(table) if row.kind == "tokenize"}
    entries = config["subsets"]
    if not isinstance(entries, dict):
        raise CorpusCheckFailed(f"{path}: subsets must map each subset to its digest source")
    unknown = sorted(str(name) for name in set(entries) - set(rows))
    missing = sorted(set(rows) - set(entries))
    if unknown or missing:
        raise CorpusCheckFailed(
            f"{path}: subsets must be exactly the tokenize rows of {table}; unknown {unknown}, missing {missing}"
        )
    sources = {name: _digest_source(entry, f"{path}: subsets.{name}") for name, entry in entries.items()}
    if subset not in sources:
        raise CorpusCheckFailed(f"{path}: no subset {subset!r}; it lists {sorted(sources)}")
    kind, value = sources[subset]
    if kind == "pending":
        raise CorpusCheckFailed(f"{path}: {subset} is pending ({value}), so there is no digest list to check against")
    return DigestCheck(
        config=path.resolve(),
        config_sha256=file_sha256(path),
        row=rows[subset],
        saved=value if kind == "saved" else None,
        hub=value if kind == "hub" else None,
    )


def appended_eod(row: CorpusRow, scalars: dict, data_base: Path = DATA_BASE) -> dict:
    """The id ``--append-eod`` wrote after every document of a tokenized corpus, and what it was derived from.

    Each prefix's tokenize record (``count_idx_tokens.py``'s ``parameters``) names the tokenizer
    the tokenize ran and that it appended EODs, but not the id: the id is that tokenizer's EOS,
    which is what Megatron's ``HuggingFaceTokenizer.eod`` appends. Every record must name the
    prepare config's tokenizer, at the commit the config pins (no commit when it pins none), with
    ``append_eod=true``; a record naming another tokenizer means the corpus is not the one its
    config describes, and no EOD is derived from either. The EOS is read from that commit.
    """
    from transformers import AutoTokenizer

    tokenizer = recorded(scalars, row.config, "tokenizer")
    pinned = tokenizer_revision(scalars, str(row.config))
    records = []
    for entry in tokenized_prefixes(row, data_base):
        path = provenance_path(entry.prefix)
        record = read_record(path)
        parameters = recorded(record, path, "parameters")
        stated = (
            recorded(record, path, "parameters", "tokenizer"),
            parameters.get("tokenizer_revision"),  # absent when the tokenize ran an unpinned tokenizer
            recorded(record, path, "parameters", "append_eod"),
        )
        if stated != (tokenizer, pinned, "true"):
            raise CorpusCheckFailed(
                f"{path}: records tokenizer {stated[0]!r} at commit {stated[1]!r} with append_eod={stated[2]!r}, but "
                f"{row.config} names {tokenizer!r} at commit {pinned!r} with --append-eod; the corpus's EOD cannot be "
                "derived"
            )
        records.append(str(path))
    eod = AutoTokenizer.from_pretrained(tokenizer, revision=pinned).eos_token_id
    if not isinstance(eod, int):
        raise CorpusCheckFailed(f"{tokenizer} has no EOS token, so --append-eod appended nothing to derive")
    return {"id": eod, "tokenizer": tokenizer, "tokenizer_revision": pinned, "records": records}


def digest_record(check: DigestCheck) -> tuple[dict, str]:
    """The dataset-builder record of a check's digest list, and where it was read from."""
    if check.saved is not None:
        path = check.saved / "_provenance.json"
        return read_record(path), str(path)
    from huggingface_hub import HfFileSystem

    url = hub_file_url(check.hub.dataset, check.hub.revision, f"{check.hub.config}/_provenance.json")
    return read_hub_file(HfFileSystem(), url, lambda handle: json.load(handle)), url


def check_digest_record(record: dict, where: str, row: CorpusRow, scalars: dict) -> dict:
    """A digest list's record must name the corpus's source and tokenizer, and the columns that are read.

    The list must have been computed from the train split of the corpus's own subset of its prepare
    config's dataset, by exactly one ``tokenize_count`` step running the prepare config's tokenizer
    into ``n_tokens`` and ``ids_hash``. The source revision is reported beside the corpus's, not
    required to equal it: a revision that changed nothing in the subset gives the same text, and the
    document-by-document comparison is what decides.
    """
    source = {key: recorded(record, where, "resolved_source", key) for key in ("repo", "subset", "split")}
    expected = {"repo": recorded(scalars, row.config, "dataset"), "subset": row.subset, "split": "train"}
    if source != expected:
        raise CorpusCheckFailed(f"{where}: the digests were computed from {source}, the corpus is {expected}")
    # Only a kernel step names a kernel (a `project` step has none), so the steps are found by it.
    steps = [
        step
        for step in recorded(record, where, "config", "transform")
        if isinstance(step, dict) and step.get("kernel") == DIGEST_KERNEL
    ]
    if len(steps) != 1:
        raise CorpusCheckFailed(f"{where}: records {len(steps)} {DIGEST_KERNEL} steps, not one")
    (step,) = steps
    stated = {
        key: recorded(step, f"{where} ({DIGEST_KERNEL} step)", key) for key in ("tokenizer", *DIGEST_KERNEL_OUTPUTS)
    }
    wanted = {"tokenizer": recorded(scalars, row.config, "tokenizer"), **DIGEST_KERNEL_OUTPUTS}
    if stated != wanted:
        raise CorpusCheckFailed(f"{where}: the {DIGEST_KERNEL} step states {stated}, the check needs {wanted}")
    return {
        "record": where,
        "source_revision": recorded(record, where, "resolved_source", "resolved_revision"),
        DIGEST_KERNEL: step,
    }


def run_digest_check(check: DigestCheck, data_base: Path = DATA_BASE) -> dict:
    """Check one subset's corpus against its digest list; the report records every input that decided it."""
    row = check.row
    scalars = subset_prepare_config(row.config, row.subset)
    eod = appended_eod(row, scalars, data_base)
    record, where = digest_record(check)
    digests = check_digest_record(record, where, row, scalars)
    if check.saved is not None:
        payload = saved_columns(check.saved, list(DIGEST_COLUMNS))
        source = {"saved": str(check.saved)}
    else:
        hub = check.hub
        payload = dataset_columns(hub.dataset, hub.revision, hub.config, list(DIGEST_COLUMNS))
        source = {"hub": asdict(hub)}
    return {
        **check_hashes(row, payload, eod["id"], data_base),
        "config": {"path": str(check.config), "sha256": check.config_sha256},
        "code_revision": code_revision(str(REPO_ROOT)),
        "table": str(row.table),
        # A prepare config may pin no revision, which the report then states as null.
        "corpus": {"config": str(row.config), "dataset": scalars["dataset"], "revision": scalars.get("revision")},
        "eod": eod,
        "digests": {**source, **digests},
    }


# ---------------------------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------------------------


def read_kept(path: Path) -> np.ndarray:
    """A kept list of the parent subset's row indices: a one-column integer parquet, a JSON array, or one per line."""
    if not path.is_file():
        raise CorpusCheckFailed(f"missing kept list {path}")
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        if table.num_columns != 1:
            raise CorpusCheckFailed(f"{path}: a kept list has one column, this has {table.column_names}")
        return integer_column(table, table.column_names[0])
    if path.suffix == ".json":
        values = json.loads(path.read_text())
        if not isinstance(values, list) or not all(type(value) is int for value in values):
            raise CorpusCheckFailed(f"{path}: a JSON kept list is an array of integers")
        return np.asarray(values, dtype=np.int64)
    if path.suffix == ".txt":
        lines = path.read_text().splitlines()
        malformed = [number for number, line in enumerate(lines, start=1) if not re.fullmatch(r"-?\d+", line.strip())]
        if malformed:
            raise CorpusCheckFailed(f"{path}: line {malformed[0]} is not an integer")
        return np.asarray([int(line) for line in lines], dtype=np.int64)
    raise CorpusCheckFailed(f"{path}: a kept list is a .parquet, .json or .txt file")


def check_kept(kept: np.ndarray, parent_docs: int, expected: int, source: Path) -> None:
    """A kept list must hold the table's count of indices, strictly increasing, inside the parent's rows."""
    if len(kept) != expected:
        raise CorpusCheckFailed(f"{source}: lists {len(kept)} documents, the table says {expected}")
    out_of_order = np.flatnonzero(np.diff(kept) <= 0)
    if out_of_order.size:
        at = int(out_of_order[0]) + 1
        relation = "repeats" if kept[at] == kept[at - 1] else "is below"
        raise CorpusCheckFailed(
            f"{source}: entry {at} ({kept[at]}) {relation} entry {at - 1} ({kept[at - 1]}); a kept list is sorted and unique"
        )
    outside = np.flatnonzero((kept < 0) | (kept >= parent_docs))
    if outside.size:
        at = int(outside[0])
        raise CorpusCheckFailed(f"{source}: entry {at} ({kept[at]}) is outside the parent's rows 0:{parent_docs}")


def kept_in_prefix(kept: np.ndarray, rows: tuple[int, int]) -> np.ndarray:
    """The kept rows a prefix holding ``rows`` contributes, as its own document indices."""
    beg, end = rows
    return kept[(kept >= beg) & (kept < end)] - beg


def parent_record(prefix: Path, index: DocumentIndex) -> dict:
    """The parent prefix's tokenize record, which must agree with the files it describes and state its tokenizer.

    A selection's record carries the parent's ``parameters`` forward, and training reads the
    tokenizer that built a corpus from there (``source_documents``): a selected prefix has no
    prepare record beside it to fall back on, so a parent record without one is refused.
    """
    path = provenance_path(prefix)
    record = read_record(path)
    totals = (recorded(record, path, "totals", "num_documents"), recorded(record, path, "totals", "total_tokens"))
    if totals != (index.docs, index.tokens):
        raise CorpusCheckFailed(
            f"{path} records {totals[0]} documents and {totals[1]} tokens, the files hold {index.docs} and {index.tokens}"
        )
    recorded(record, path, "parameters", "tokenizer")
    return record


def require_empty_directory(path: Path) -> None:
    """A selection is written once, into a directory build_corpora.sh created and striped before submitting."""
    if not path.is_dir():
        raise CorpusCheckFailed(
            f"{path} does not exist; build_corpora.sh creates and stripes it before submitting, because Lustre "
            "striping must precede the first write"
        )
    if any(path.iterdir()):
        raise CorpusCheckFailed(f"{path} is not empty; a selection is written into an empty directory")


def check_selected_files(
    parent_prefix: Path, parent: DocumentIndex, index_file: Path, data_file: Path, local: np.ndarray
) -> DocumentIndex:
    """The written selection must hold exactly the parent's documents ``local``, in order, byte for byte."""
    out = read_index(index_file, data_file)
    if out.docs != len(local):
        raise CorpusCheckFailed(f"{index_file}: holds {out.docs} documents, the kept list has {len(local)} here")
    differs = np.flatnonzero(out.sizes != parent.sizes[local])
    if differs.size:
        k = int(differs[0])
        raise CorpusCheckFailed(
            f"{index_file}: document {k} is {out.sizes[k]} ids, parent document {local[k]} is {parent.sizes[local[k]]}"
        )
    parent_tokens, out_tokens = token_memmap(bin_path(parent_prefix)), token_memmap(data_file)
    for k, document in enumerate(local.tolist()):
        start, size, source = int(out.starts[k]), int(out.sizes[k]), int(parent.starts[document])
        if not np.array_equal(out_tokens[start : start + size], parent_tokens[source : source + size]):
            raise CorpusCheckFailed(f"{data_file}: document {k} differs from parent document {document}")
    return out


def select_prefix(selected: SelectedCorpus, entry: SelectedPrefix) -> dict:
    """Write one prefix of a selected corpus, check it against its parent, and record its provenance."""
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    row, config = selected.row, selected.config
    revision = code_revision(str(REPO_ROOT))
    if revision.startswith("UNRESOLVED"):
        raise CorpusCheckFailed(f"the code this runs is unattributable: {revision}")
    kept = read_kept(config.kept)
    check_kept(kept, selected.parent.docs, row.docs, config.kept)
    require_empty_directory(entry.output.parent)
    parent = read_index(idx_path(entry.parent), bin_path(entry.parent))
    beg, end = entry.rows
    if parent.docs != end - beg:
        raise CorpusCheckFailed(
            f"{entry.parent}: holds {parent.docs} documents, its rows {beg}:{end} number {end - beg}"
        )
    record = parent_record(entry.parent, parent)
    local = kept_in_prefix(kept, entry.rows)

    partial_bin, partial_idx = bin_path(entry.output, PARTIAL), idx_path(entry.output, PARTIAL)
    builder = IndexedDatasetBuilder(str(partial_bin), dtype=np.int32)
    tokens = token_memmap(bin_path(entry.parent))
    for document in local.tolist():
        start, size = int(parent.starts[document]), int(parent.sizes[document])
        # An empty document is written as preprocess_data.py writes one: no sequence, not an empty one.
        builder.add_document(tokens[start : start + size], [size] if size else [])
    builder.finalize(str(partial_idx))
    out = check_selected_files(entry.parent, parent, partial_idx, partial_bin, local)
    check_absent_tokens(partial_bin, out, config.absent_token_ids, beg + local)

    provenance = {
        "kind": "select",
        "tool": {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "version": TOOL_VERSION},
        "code_revision": revision,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "table": str(row.table),
        "subset": row.subset,
        "shard": entry.shard,
        "config": {"path": str(config.path), "sha256": file_sha256(config.path)},
        "absent_token_ids": list(config.absent_token_ids),
        "kept": {
            "path": str(config.kept),
            "sha256": file_sha256(config.kept),
            "entries": len(kept),
            "here": len(local),
        },
        "parent": {
            "table": str(selected.parent.table),
            "subset": selected.parent.subset,
            "prefix": str(entry.parent),
            "rows": [beg, end],
            "documents": parent.docs,
            "tokens": parent.tokens,
            "bin_sha256": file_sha256(bin_path(entry.parent)),
            "idx_sha256": file_sha256(idx_path(entry.parent)),
        },
        "totals": {"total_tokens": out.tokens, "num_sequences": out.sequences, "num_documents": out.docs},
        "parameters": record["parameters"],
    }
    os.replace(partial_bin, bin_path(entry.output))
    os.replace(partial_idx, idx_path(entry.output))
    partial_record = Path(f"{provenance_path(entry.output)}{PARTIAL}")
    partial_record.write_text(json.dumps(provenance, indent=1) + "\n")
    os.replace(partial_record, provenance_path(entry.output))
    return provenance


def verify_selected_prefix(selected: SelectedCorpus, entry: SelectedPrefix, kept: np.ndarray) -> dict:
    """Re-run a selected prefix's checks from its provenance and its parent; return its counts.

    The record must name this prefix's parent, rows, kept list and absent ids as they are now (the
    list and the parent files by sha256), the files must still hold exactly the kept documents, and
    none of them may hold an absent id: the files are scanned for each, whatever the record says.
    """
    path = provenance_path(entry.output)
    record = read_record(path)
    parent = read_index(idx_path(entry.parent), bin_path(entry.parent))
    expected = {
        "kind": "select",
        "parent prefix": str(entry.parent),
        "parent rows": list(entry.rows),
        "kept list": file_sha256(selected.config.kept),
        "parent .bin": file_sha256(bin_path(entry.parent)),
        "parent .idx": file_sha256(idx_path(entry.parent)),
        "absent ids": list(selected.config.absent_token_ids),
    }
    recorded_now = {
        "kind": recorded(record, path, "kind"),
        "parent prefix": recorded(record, path, "parent", "prefix"),
        "parent rows": recorded(record, path, "parent", "rows"),
        "kept list": recorded(record, path, "kept", "sha256"),
        "parent .bin": recorded(record, path, "parent", "bin_sha256"),
        "parent .idx": recorded(record, path, "parent", "idx_sha256"),
        "absent ids": recorded(record, path, "absent_token_ids"),
    }
    differ = [name for name in expected if recorded_now[name] != expected[name]]
    if differ:
        raise CorpusCheckFailed(
            f"{path}: records {[recorded_now[n] for n in differ]} for {differ}, now {[expected[n] for n in differ]}"
        )
    local = kept_in_prefix(kept, entry.rows)
    out = check_selected_files(entry.parent, parent, idx_path(entry.output), bin_path(entry.output), local)
    check_absent_tokens(bin_path(entry.output), out, selected.config.absent_token_ids, entry.rows[0] + local)
    totals = (recorded(record, path, "totals", "num_documents"), recorded(record, path, "totals", "total_tokens"))
    if totals != (out.docs, out.tokens):
        raise CorpusCheckFailed(
            f"{path}: records {totals[0]} documents of {totals[1]} ids, the files hold {out.docs} documents of {out.tokens} ids"
        )
    return {"docs": out.docs, "tokens": out.tokens}


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------


def run_select(table: Path, subset: str, shard: int | None, data_base: Path) -> dict:
    """Write one prefix of a select row's corpus; a sharded corpus must be told which, an unsharded one must not."""
    (row,) = read_corpora_table(table, "all", [subset])
    selected = selected_corpus(row, data_base)
    shards = [entry.shard for entry in selected.prefixes]
    if shard is None and shards != [None]:
        raise CorpusCheckFailed(f"{subset}: the corpus has shards {shards}, so --shard must name one")
    return select_prefix(selected, selected.prefix(shard))


def main(argv: list[str] | None = None) -> int:
    """Run a select job or a digest check; non-zero on any failure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select", help="write one prefix of a select row's corpus")
    select.add_argument("table", type=Path, help="the corpora table holding the select row")
    select.add_argument("subset", help="the select row's subset")
    select.add_argument("--shard", type=int, default=None, help="the shard to write; required for a sharded corpus")
    select.add_argument(
        "--data-base", type=Path, default=DATA_BASE, help=f"the corpus roots' base (default: {DATA_BASE})"
    )

    hashes = commands.add_parser(
        "check-hashes", help="check a tokenized corpus against its digest list, as a digest-check config states it"
    )
    hashes.add_argument(
        "--config",
        type=Path,
        required=True,
        help="the digest-check config: the corpora table and each subset's source",
    )
    hashes.add_argument("--subset", required=True, help="the subset to check")
    hashes.add_argument("--report-out", type=Path, default=None, help="write the report here as JSON")
    hashes.add_argument(
        "--data-base", type=Path, default=DATA_BASE, help=f"the corpus roots' base (default: {DATA_BASE})"
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "select":
            totals = run_select(args.table, args.subset, args.shard, args.data_base)["totals"]
            print(
                f"{args.subset} shard={args.shard}: {totals['num_documents']:,} documents, "
                f"{totals['total_tokens']:,} ids"
            )
            return 0
        report = run_digest_check(read_digest_checks(args.config, args.subset), args.data_base)
    except (CorpusCheckFailed, ValueError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"report: {args.report_out}")
    print(
        f"{report['subset']}: {report['rows']:,} digest rows for {report['documents']:,} documents "
        f"(source rows in order: {report['source_rows_in_order']}); EOD {report['eod']['id']}; "
        f"digests from revision {report['digests']['source_revision']}, corpus prepared at "
        f"{report['corpus']['revision']}; mismatches {report['mismatches']}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
