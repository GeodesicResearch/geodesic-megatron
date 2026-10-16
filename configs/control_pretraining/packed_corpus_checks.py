#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Check a packed SFT corpus against a reference pack: every document is one of the reference's, and given ids never
occur.

A packed corpus is what ``scripts/data/pack_sft_dataset.py`` writes for a corpora table's ``pack`` row: per shard one
parquet (``corpora_table.packed_parquet_path``) whose rows are packs, each holding ``input_ids``, ``loss_mask`` (one
entry per id, already shifted onto the labels) and ``seq_start_id`` (the offset at which each packed document begins,
the first at 0). A document is the span of a pack from its start to the next document's start, the last one's to the
end of the pack: its ids and its loss mask over that span. The span includes the padding the packer appends to each
document (``pad_document_for_packing``: the tokenizer's EOD with the mask off, up to the next multiple of
``pad-seq-to-mult`` plus one), which is decided by the document's own length and the pack geometry alone, and a pack
holds nothing after its last document. So two packs built by the packer at one tokenizer and geometry hold the same
span for the same conversation, whichever pack it lands in.

Two checks, over every pack of every shard, nothing sampled:

* **Containment.** Every document of the candidate must be a document of the reference, ids and loss mask alike. Each
  document is reduced to the 128-bit BLAKE2b of its length (8 bytes), its ids (little-endian int64) and its mask (one
  byte per id); the reference's digests form a set and each candidate document is looked up in it. The candidate may
  hold a document many times (a corpus refilled by repetition), and the reference may hold documents the candidate
  does not.
* **Absent ids.** Each listed id must occur nowhere in the candidate's ids, padding included.

Shards are scanned in parallel, a process per shard, one row group at a time. A shard of either corpus that is missing,
unreadable or holds a malformed pack (a column missing or of another type, a null, ``seq_start_id`` not starting at 0
and strictly increasing inside the pack, a mask of another length than the ids) is a failure, never skipped. The
verdict is PASS only when both corpora are packed at one tokenizer and geometry, every shard of both was read, both hold
documents, every candidate document was found and every absent id counts 0.

The report (JSON, never written over an existing file) holds ``verdict``, the ``failures``, and per corpus and shard
the packs, documents, tokens, loss-bearing tokens and absent-id counts (and per corpus the distinct documents), the
first missing documents and absent-id occurrences with their shard, pack and offset, and its inputs: the config, the
corpora tables and prepare configs with their sha256, each corpus's dataset commit, the parquet paths and the code
revision. A prepare config that pins its commit in a revision file is read when the checks run, not when the config is
read, and a commit that cannot be read then is a failure.

What is checked is a config's statement (YAML, exact keys)::

    candidate: {table: <corpora table>, subset: <its pack row>}   # tables named relative to the repo root
    reference: {table: <corpora table>, subset: <its pack row>}
    absent_token_ids: [500]
    workers: 64                                                   # shards scanned at once

Usage (inside the container; as a job through ``corpus_job.sbatch``)::

    python configs/control_pretraining/packed_corpus_checks.py --config <checks.yaml> --report-out <json>

Exit status 0 on PASS, 1 on FAIL or a config that cannot be read, 2 when the report already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpora_table import (  # noqa: E402
    DATA_BASE,
    PACK_GEOMETRY_KEYS,
    REPO_ROOT,
    Checker,
    CorpusRow,
    corpus_root,
    packed_parquet_path,
    read_corpora_table,
    subset_prepare_config,
)
from corpus_documents import EXAMPLES  # noqa: E402


if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from scripts.data.sha256_manifest import file_sha256  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_int, require_string  # noqa: E402
from scripts.telemetry.code_revision import code_revision  # noqa: E402
from scripts.telemetry.gate_outcome import FAIL, PASS  # noqa: E402
from scripts.token_ids import require_token_id_list  # noqa: E402


CONFIG_KEYS = frozenset({"candidate", "reference", "absent_token_ids", "workers"})
CORPUS_KEYS = frozenset({"table", "subset"})
# The prepare-config scalars both corpora must share for their packed documents to be comparable.
PACKING_SCALARS = ("tokenizer", *PACK_GEOMETRY_KEYS)
DIGEST_BYTES = 16
# Where a document sits: its pack (the row of its shard's parquet), its index in the pack, its offset and length.
LOCATION = np.dtype([("pack", np.int64), ("document", np.int64), ("start", np.int64), ("length", np.int64)])


class PackCheckConfigError(Exception):
    """A pack-check config that cannot be read, or that names something other than two packed corpora."""


class MalformedShard(Exception):
    """A shard's parquet is not a pack the packer writes."""


@dataclass(frozen=True)
class PackedCorpus:
    """One corpus of the check: a ``pack`` row of a corpora table, and its shards' parquet paths."""

    table: Path
    row: CorpusRow
    scalars: dict  # its prepare config as the build plans it, a revision file unread (subset_prepare_config)
    root: Path
    shards: tuple[tuple[str, Path], ...]  # (shard name, parquet); the name is "" for an unsharded corpus

    def revision(self) -> str | None:
        """The dataset commit the corpus's prepare config pins for its subset, a revision file read now (``None`` when
        it pins none). Raises ValueError for a pin that cannot be read."""
        return subset_prepare_config(self.row.config, self.row.subset).get("revision")

    def inputs(self, revision: str | None) -> dict:
        """What the corpus was read from, as the report records it, with ``revision`` its dataset commit."""
        return {
            "table": str(self.table),
            "table_sha256": file_sha256(self.table),
            "subset": self.row.subset,
            "prepare_config": str(self.row.config),
            "prepare_config_sha256": file_sha256(self.row.config),
            "dataset": self.scalars["dataset"],
            "revision": revision,
            **{key: self.scalars[key] for key in PACKING_SCALARS},
            "root": str(self.root),
        }


@dataclass(frozen=True)
class PackCheckConfig:
    """A pack-check config as read: the two corpora, the absent ids and the number of shards scanned at once."""

    path: Path
    sha256: str
    candidate: PackedCorpus
    reference: PackedCorpus
    absent_ids: tuple[int, ...]
    workers: int


def packed_corpus(entry: object, where: str, data_base: Path) -> PackedCorpus:
    """The corpus a config entry ``{table, subset}`` names: a ``pack`` row of the table, its shards under its root in
    ``data_base`` (``corpus_root``) at the path its prepare config's tokenizer and geometry give
    (``packed_parquet_path``). Neither needs the dataset commit, so a revision file is not read here: a config can be
    read before the file exists, and the commit is read when the checks run (``PackedCorpus.revision``)."""
    spec = require_keys(entry, where, CORPUS_KEYS, error=PackCheckConfigError)
    table = REPO_ROOT / require_string(spec["table"], f"{where}.table", error=PackCheckConfigError)
    subset = require_string(spec["subset"], f"{where}.subset", error=PackCheckConfigError)
    if not table.is_file():
        raise PackCheckConfigError(f"{where}.table: no corpora table at {table}")
    try:
        (row,) = read_corpora_table(table, "all", [subset])
        scalars = subset_prepare_config(row.config, row.subset, read_revision_file=False)
    except (OSError, ValueError) as error:
        raise PackCheckConfigError(f"{where}: {error}") from error
    if row.kind != "pack":
        raise PackCheckConfigError(f"{where}: {subset} is a {row.kind} row of {table}, not a pack")
    root = corpus_root(scalars["dataset"], row.subset, data_base)
    names = row.shard_names or [""]
    shards = tuple((name, packed_parquet_path(root / name, scalars)) for name in names)
    return PackedCorpus(table=table, row=row, scalars=scalars, root=root, shards=shards)


def read_pack_checks(path: Path, data_base: Path) -> PackCheckConfig:
    """A pack-check config, read and refused whole: exactly ``candidate``, ``reference`` (each ``{table, subset}``),
    ``absent_token_ids`` (distinct ids, possibly none) and ``workers`` (at least 1). Raises PackCheckConfigError."""
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise PackCheckConfigError(f"{path}: cannot read a pack-check config: {error}") from error
    config = require_keys(loaded, f"{path}: a pack-check config", CONFIG_KEYS, error=PackCheckConfigError)
    return PackCheckConfig(
        path=path.resolve(),
        sha256=file_sha256(path),
        candidate=packed_corpus(config["candidate"], f"{path}: candidate", data_base),
        reference=packed_corpus(config["reference"], f"{path}: reference", data_base),
        absent_ids=require_token_id_list(
            config["absent_token_ids"], f"{path}: absent_token_ids", allow_empty=True, error=PackCheckConfigError
        ),
        workers=require_int(config["workers"], f"{path}: workers", 1, error=PackCheckConfigError),
    )


# ---------------------------------------------------------------------------------------------
# Scanning one shard
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardScan:
    """What a scan of one shard found. ``error`` names why it could not be read whole; the rest is then empty."""

    error: str | None
    packs: int
    tokens: int
    loss_tokens: int
    digests: bytes  # DIGEST_BYTES per document, in the order the documents are stored
    locations: np.ndarray  # LOCATION per document, in the same order
    absent_counts: tuple[int, ...]  # per absent id, its occurrences in the shard's ids
    absent_examples: tuple[dict, ...]  # the first occurrences, at most EXAMPLES


def failed_scan(error: str, absent_ids: tuple[int, ...]) -> ShardScan:
    """The scan of a shard that could not be read whole."""
    return ShardScan(error, 0, 0, 0, b"", np.empty(0, dtype=LOCATION), (0,) * len(absent_ids), ())


def document_digest(ids: np.ndarray, mask: np.ndarray) -> bytes:
    """The 128-bit BLAKE2b of one packed document: its length (8 bytes, little-endian), its ids (contiguous
    little-endian int64) and its loss mask (contiguous, one byte per id)."""
    digest = hashlib.blake2b(digest_size=DIGEST_BYTES)
    digest.update(len(ids).to_bytes(8, "little"))
    digest.update(ids)
    digest.update(mask)
    return digest.digest()


def list_column(table, name: str, integer: bool) -> tuple[np.ndarray, np.ndarray]:
    """A list column's values, flat, and each pack's offsets into them. The column must hold no null and values of an
    integer type, or, with ``integer`` false, of an integer or boolean type. Raises MalformedShard."""
    import pyarrow as pa

    array = table.column(name).combine_chunks()
    if not pa.types.is_list(array.type) and not pa.types.is_large_list(array.type):
        raise MalformedShard(f"column {name} is {array.type}, not a list")
    value_type = array.type.value_type
    if not (pa.types.is_integer(value_type) or (not integer and pa.types.is_boolean(value_type))):
        raise MalformedShard(f"column {name} holds {value_type} values")
    values = array.flatten()
    if array.null_count or values.null_count:
        raise MalformedShard(f"column {name} holds a null")
    offsets = array.offsets.to_numpy()
    return values.to_numpy(zero_copy_only=False), offsets - offsets[0]


def check_pack_starts(starts: np.ndarray, length: int, pack: int) -> None:
    """Raise MalformedShard unless a pack's document starts begin at 0 and increase strictly inside its ``length``."""
    if not len(starts) or starts[0] != 0 or (np.diff(starts) <= 0).any() or starts[-1] >= length:
        raise MalformedShard(f"pack {pack}: seq_start_id {starts[:8].tolist()}... does not split its {length} ids")


def scan_shard(parquet: Path, absent_ids: tuple[int, ...]) -> ShardScan:
    """Read one shard's parquet one row group at a time and return its documents' digests and locations, its totals
    and its absent-id counts; a shard that cannot be read whole returns the reason instead (``failed_scan``)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        parquet_file = pq.ParquetFile(parquet)
        names = parquet_file.schema_arrow.names
        missing = [name for name in ("input_ids", "loss_mask", "seq_start_id") if name not in names]
        if missing:
            raise MalformedShard(f"no column {missing}; it has {names}")
        digests, locations, examples = [], [], []
        counts = [0] * len(absent_ids)
        tokens = loss_tokens = first_pack = 0
        for group in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(group, columns=["input_ids", "loss_mask", "seq_start_id"])
            ids, id_offsets = list_column(table, "input_ids", integer=True)
            mask, mask_offsets = list_column(table, "loss_mask", integer=False)
            starts, start_offsets = list_column(table, "seq_start_id", integer=True)
            if not np.array_equal(np.diff(id_offsets), np.diff(mask_offsets)):
                raise MalformedShard(f"row group {group}: a pack's loss_mask is not as long as its input_ids")
            ids = np.ascontiguousarray(ids, dtype="<i8")
            mask = np.ascontiguousarray(mask, dtype=np.uint8)
            tokens += len(ids)
            loss_tokens += int(np.count_nonzero(mask))
            for row in range(table.num_rows):
                pack = first_pack + row
                begin, end = int(id_offsets[row]), int(id_offsets[row + 1])
                own = starts[start_offsets[row] : start_offsets[row + 1]].astype(np.int64)
                check_pack_starts(own, end - begin, pack)
                bounds = np.append(own, end - begin) + begin
                lengths = np.diff(bounds)
                located = np.empty(len(own), dtype=LOCATION)
                located["pack"], located["document"] = pack, np.arange(len(own))
                located["start"], located["length"] = own, lengths
                locations.append(located)
                digests.extend(
                    document_digest(ids[a:b], mask[a:b]) for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist())
                )
            for index, token in enumerate(absent_ids):
                hits = np.flatnonzero(ids == token)
                counts[index] += len(hits)
                for position in hits[: max(EXAMPLES - len(examples), 0)].tolist():
                    row = int(np.searchsorted(id_offsets, position, side="right")) - 1
                    offset = position - int(id_offsets[row])
                    own = starts[start_offsets[row] : start_offsets[row + 1]]
                    document = int(np.searchsorted(own, offset, side="right")) - 1
                    examples.append(
                        {"token_id": token, "pack": first_pack + row, "document": document, "offset": offset}
                    )
            first_pack += table.num_rows
    except (OSError, pa.ArrowException, MalformedShard) as error:
        return failed_scan(f"{parquet}: {error}", absent_ids)
    return ShardScan(
        error=None,
        packs=first_pack,
        tokens=tokens,
        loss_tokens=loss_tokens,
        digests=b"".join(digests),
        locations=np.concatenate(locations) if locations else np.empty(0, dtype=LOCATION),
        absent_counts=tuple(counts),
        absent_examples=tuple(examples),
    )


def split_digests(blob: bytes) -> list[bytes]:
    """A scan's digests, one per document."""
    return [blob[i : i + DIGEST_BYTES] for i in range(0, len(blob), DIGEST_BYTES)]


# ---------------------------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------------------------


def scan_corpora(config: PackCheckConfig) -> dict[str, list[ShardScan]]:
    """Scan every shard of both corpora, ``config.workers`` at a time, each in a process of its own; per corpus, the
    scans in shard order."""
    sides = {"candidate": config.candidate, "reference": config.reference}
    tasks = [(side, name, parquet) for side, corpus in sides.items() for name, parquet in corpus.shards]
    scans: dict[str, list[ShardScan]] = {side: [] for side in sides}
    # Spawned, not forked: a forked child inherits whatever threads the parent holds (pyarrow's pool among them).
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=context) as pool:
        results = pool.map(scan_shard, [task[2] for task in tasks], [config.absent_ids] * len(tasks))
        for (side, name, parquet), scan in zip(tasks, results):
            scans[side].append(scan)
            state = scan.error or f"{scan.packs:,} packs, {len(scan.locations):,} documents, {scan.tokens:,} tokens"
            print(f"{side} {name or parquet}: {state}", flush=True)
    return scans


def absent_counts(absent_ids: tuple[int, ...], counts: list[int]) -> dict[str, int]:
    """Absent-id counts keyed by the id, as the report states them."""
    return {str(token): count for token, count in zip(absent_ids, counts)}


def corpus_report(
    corpus: PackedCorpus, revision: str | None, scans: list[ShardScan], absent_ids: tuple[int, ...], distinct: int
) -> dict:
    """One corpus's part of the report: its inputs, each shard's counts, and its totals over the shards read."""
    shards = [
        {
            "shard": name,
            "parquet": str(parquet),
            "error": scan.error,
            "packs": scan.packs,
            "documents": len(scan.locations),
            "tokens": scan.tokens,
            "loss_tokens": scan.loss_tokens,
            "absent_id_counts": absent_counts(absent_ids, list(scan.absent_counts)),
        }
        for (name, parquet), scan in zip(corpus.shards, scans)
    ]
    totals = {key: sum(shard[key] for shard in shards) for key in ("packs", "documents", "tokens", "loss_tokens")}
    counts = [sum(scan.absent_counts[index] for scan in scans) for index in range(len(absent_ids))]
    return {
        **corpus.inputs(revision),
        "shards": shards,
        "totals": {**totals, "distinct_documents": distinct, "absent_id_counts": absent_counts(absent_ids, counts)},
    }


@dataclass(frozen=True)
class Containment:
    """How the candidate's documents stand against the reference's."""

    candidate_distinct: int
    reference_distinct: int
    missing: int  # candidate documents, counted with their repeats, that are not reference documents
    examples: list[dict]  # the first of them, at most EXAMPLES, in shard and pack order


def containment(
    candidate: PackedCorpus, candidate_scans: list[ShardScan], reference_scans: list[ShardScan]
) -> Containment:
    """Look every candidate document up among the reference's documents, by digest."""
    known = set()
    for scan in reference_scans:
        known.update(split_digests(scan.digests))
    seen, missing, examples = set(), 0, []
    for (name, parquet), scan in zip(candidate.shards, candidate_scans):
        for digest, (pack, document, start, length) in zip(split_digests(scan.digests), scan.locations.tolist()):
            seen.add(digest)
            if digest in known:
                continue
            missing += 1
            if len(examples) < EXAMPLES:
                examples.append(
                    {
                        "shard": name,
                        "parquet": str(parquet),
                        "pack": pack,
                        "document": document,
                        "start": start,
                        "length": length,
                        "digest": digest.hex(),
                    }
                )
    return Containment(len(seen), len(known), missing, examples)


def run_pack_checks(config: PackCheckConfig) -> dict:
    """Run both checks over every shard of both corpora and return the report, whose ``verdict`` is PASS only when
    nothing failed."""
    checker = Checker()
    candidate, reference = config.candidate, config.reference
    revisions = {}
    for side, corpus in (("candidate", candidate), ("reference", reference)):
        try:
            revisions[side] = corpus.revision()
        except ValueError as error:
            checker.expect(False, f"the {side}'s dataset commit cannot be read: {error}")
            revisions[side] = None
    for key in PACKING_SCALARS:
        checker.expect(
            candidate.scalars[key] == reference.scalars[key],
            f"the candidate is packed at {key}={candidate.scalars[key]!r}, the reference at {reference.scalars[key]!r}",
        )
    scans = scan_corpora(config)
    for side, corpus in (("candidate", candidate), ("reference", reference)):
        for (name, _), scan in zip(corpus.shards, scans[side]):
            checker.expect(scan.error is None, f"{side} shard {name or corpus.root}: {scan.error}")
    found = containment(candidate, scans["candidate"], scans["reference"])
    report = {
        "candidate": corpus_report(
            candidate, revisions["candidate"], scans["candidate"], config.absent_ids, found.candidate_distinct
        ),
        "reference": corpus_report(
            reference, revisions["reference"], scans["reference"], config.absent_ids, found.reference_distinct
        ),
    }
    for side in ("candidate", "reference"):
        checker.expect(report[side]["totals"]["documents"] > 0, f"the {side} holds no documents")
    checker.expect(
        found.missing == 0, f"candidate documents that are not documents of the reference: {found.missing:,}"
    )
    counts = report["candidate"]["totals"]["absent_id_counts"]
    for token, count in counts.items():
        checker.expect(count == 0, f"occurrences of token id {token} in the candidate's ids: {count:,}")
    absent_examples = [
        {"shard": name, **example}
        for (name, _), scan in zip(candidate.shards, scans["candidate"])
        for example in scan.absent_examples
    ][:EXAMPLES]
    return {
        "verdict": FAIL if checker.failures else PASS,
        "failures": checker.failures,
        "config": {"path": str(config.path), "sha256": config.sha256},
        "code_revision": code_revision(str(REPO_ROOT)),
        "absent_token_ids": list(config.absent_ids),
        "containment": {
            "candidate_documents": report["candidate"]["totals"]["documents"],
            "missing_documents": found.missing,
            "missing_examples": found.examples,
        },
        "absent_ids": {"counts": counts, "examples": absent_examples},
        **report,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the checks a pack-check config states, write the report and return 0 on PASS, 1 on FAIL."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", type=Path, required=True, help="the pack-check config: the two corpora and the ids"
    )
    parser.add_argument(
        "--report-out", type=Path, required=True, help="write the report here as JSON; an existing file is refused"
    )
    parser.add_argument(
        "--data-base", type=Path, default=DATA_BASE, help=f"the corpus roots' base (default: {DATA_BASE})"
    )
    args = parser.parse_args(argv)
    if args.report_out.exists():
        parser.error(f"{args.report_out} exists")
    try:
        config = read_pack_checks(args.config, args.data_base)
    except PackCheckConfigError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    report = run_pack_checks(config)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("x") as out:
        json.dump(report, out, indent=1)
        out.write("\n")
    for side in ("candidate", "reference"):
        totals = report[side]["totals"]
        print(
            f"{side} {report[side]['subset']}: {totals['packs']:,} packs, {totals['documents']:,} documents "
            f"({totals['distinct_documents']:,} distinct), {totals['tokens']:,} tokens, "
            f"{totals['loss_tokens']:,} loss-bearing"
        )
    print(f"candidate documents not in the reference: {report['containment']['missing_documents']:,}")
    print(f"absent-id counts in the candidate: {report['absent_ids']['counts']}")
    print(f"report: {args.report_out}")
    for failure in report["failures"]:
        print(f"  - {failure}", file=sys.stderr)
    print(f"verdict: {report['verdict']}")
    return 0 if report["verdict"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
