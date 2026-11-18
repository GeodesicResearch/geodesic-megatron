# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The proof that the bucket holds a unit, and the removal of a local copy that follows it.

The bucket mirror (``sync_bucket.py``) uploads a unit, a local directory onto a bucket prefix. A local copy may go
only once the bucket provably holds it. A unit's record is the ``FileRecord`` of each of its files
(``scripts/file_digest.py``) with where it went. The verified record is a TSV, one file per unit, with ``#`` header
lines (bucket, remote, local, time) and one ``sha256<TAB>size<TAB>path`` line for each file.

Catalogue:
    RemovalRefused: a removal that must not happen.
    VerifiedRecord: the files of one unit and where they went.
    remote_sizes: the size of every file under a bucket prefix.
    mismatches: how the bucket's listing differs from the records.
    write_verified: write a verified record as a TSV.
    read_verified: read a verified TSV.
    remove_verified: delete the recorded files of one unit once the bucket provably holds them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scripts.file_digest import FileRecord


LOGGER = logging.getLogger(__name__)

_HEADER_KEYS = ("bucket", "remote", "local", "verified_at")


class RemovalRefused(RuntimeError):
    """A removal that must not happen: outside the allowed roots, the newest save, or not provably in the bucket."""


@dataclass(frozen=True)
class VerifiedRecord:
    """The files of one unit and where they went, as written to a verified TSV."""

    bucket: str
    remote: str
    local: Path
    verified_at: str
    records: tuple[FileRecord, ...]


def remote_sizes(api: Any, bucket: str, remote: str) -> dict[str, int]:
    """The size of every file under a bucket prefix, by path relative to the prefix."""
    prefix = remote.strip("/") + "/"
    sizes = {}
    for item in api.list_bucket_tree(bucket, prefix=prefix, recursive=True):
        if item.type == "file":
            sizes[item.path[len(prefix) :]] = item.size
    return sizes


def mismatches(records: Iterable[FileRecord], remote: dict[str, int]) -> list[str]:
    """How the bucket's listing differs from the records: a file missing, a size that differs, or a file extra.

    Empty when the bucket holds exactly the recorded files at their recorded sizes.
    """
    problems = []
    recorded = {r.path: r for r in records}
    for path in sorted(recorded):
        if path not in remote:
            problems.append(f"{path}: not in the bucket")
        elif remote[path] != recorded[path].size:
            problems.append(f"{path}: {remote[path]} bytes in the bucket, {recorded[path].size} local")
    problems.extend(f"{path}: in the bucket and not recorded" for path in sorted(set(remote) - set(recorded)))
    return problems


def write_verified(path: Path, verified: VerifiedRecord) -> None:
    """Write a verified record as a TSV, atomically."""
    header = {
        "bucket": verified.bucket,
        "remote": verified.remote,
        "local": str(verified.local),
        "verified_at": verified.verified_at,
    }
    lines = [f"# {key}: {header[key]}" for key in _HEADER_KEYS]
    lines.extend(f"{r.sha256}\t{r.size}\t{r.path}" for r in verified.records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def read_verified(path: Path) -> VerifiedRecord:
    """Read a verified TSV.

    Raises:
        ValueError: A header key is missing, or a line is not ``sha256<TAB>size<TAB>path``.
    """
    header: dict[str, str] = {}
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("# "):
            key, _, value = line[2:].partition(": ")
            header[key] = value
        elif line:
            parts = line.split("\t", 2)
            if len(parts) != 3 or not parts[1].isdigit():
                raise ValueError(f"{path}:{number}: expected sha256<TAB>size<TAB>path")
            records.append(FileRecord(path=parts[2], size=int(parts[1]), sha256=parts[0]))
    missing = [key for key in _HEADER_KEYS if key not in header]
    if missing:
        raise ValueError(f"{path}: header keys missing: {missing}")
    return VerifiedRecord(
        header["bucket"], header["remote"], Path(header["local"]), header["verified_at"], tuple(records)
    )


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root.resolve() or root.resolve() in resolved.parents for root in roots)


def remove_verified(
    api: Any,
    verified: VerifiedRecord,
    allowed_roots: Iterable[Path],
    newest: Path | None,
    execute: bool,
) -> list[Path]:
    """Delete the recorded files of one unit once the bucket provably holds them; return the paths removed.

    Nothing is deleted unless every check passes, and a dry run (``execute`` False) returns what would go. A removal
    deletes the recorded files, then every directory under ``verified.local`` that is left empty, then
    ``verified.local`` itself. A directory that still holds a file that no record names stays, with a warning.

    Args:
        api: The Hub client, for the bucket's listing.
        verified: The unit's verified record.
        allowed_roots: The directories under which a removal is allowed. ``verified.local`` must lie inside one.
        newest: The newest completed save of the run that this unit belongs to, or None. It is never removed.
        execute: Delete the files. Without it, only report.

    Raises:
        RemovalRefused: The directory lies outside every allowed root, it is the newest save, the bucket's listing
            differs from the record, or a recorded file is missing or has another size now.
    """
    local = verified.local
    if not _inside(local, allowed_roots):
        raise RemovalRefused(f"{local} is outside the allowed roots {[str(r) for r in allowed_roots]}")
    if newest is not None and local.resolve() == newest.resolve():
        raise RemovalRefused(f"{local} is the newest completed save of its run, which a resume reads")
    problems = mismatches(verified.records, remote_sizes(api, verified.bucket, verified.remote))
    if problems:
        raise RemovalRefused(f"the bucket does not hold {verified.remote} as recorded: {problems[:5]}")
    for record in verified.records:
        file = local / record.path
        if not file.is_file() or file.stat().st_size != record.size:
            raise RemovalRefused(f"{file} is missing or its size differs from the record")
    paths = [local / record.path for record in verified.records]
    if not execute:
        LOGGER.info("would remove %d files of %s", len(paths), local)
        return paths
    for path in paths:
        path.unlink()
        LOGGER.info("removed %s", path)
    for directory in sorted((p for p in local.rglob("*") if p.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    if not any(local.iterdir()):
        local.rmdir()
    else:
        LOGGER.warning("%s still holds files that no record names, left in place", local)
    return paths
