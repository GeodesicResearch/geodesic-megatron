# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The proof that the bucket holds a unit, the restore that shows it usable, and the removal that follows.

The bucket mirror (``sync_bucket.py``) uploads a unit, a local directory onto a bucket prefix. A local copy may go
only once the bucket provably holds it. A unit's record is the ``FileRecord`` of each of its files
(``scripts/file_digest.py``) with where it went. The verified record is a TSV, one file per unit, with ``#`` header
lines (bucket, remote, local, time) and one ``sha256<TAB>size<TAB>path`` line for each file.

Catalogue:
    RemovalRefused: a removal that must not happen.
    RestoreFailed: a restore that cannot be done, or whose files differ from the record.
    VerifiedRecord: the files of one unit and where they went.
    remote_sizes: the size of every file under a bucket prefix.
    mismatches: how the bucket's listing differs from the records.
    write_verified: write a verified record as a TSV.
    read_verified: read a verified TSV.
    restore_verified: download a unit from the bucket and compare every file with its record.
    remove_verified: delete the recorded files of one unit once the bucket provably holds them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scripts.file_digest import HASH_WORKERS, FileRecord, directory_records


LOGGER = logging.getLogger(__name__)

_HEADER_KEYS = ("bucket", "remote", "local", "verified_at")


class RemovalRefused(RuntimeError):
    """A removal that must not happen: outside the allowed roots, the newest save, or not provably in the bucket."""


class RestoreFailed(RuntimeError):
    """A restore that cannot be done, or whose downloaded files differ from the record."""


@dataclass(frozen=True)
class VerifiedRecord:
    """The files of one unit and where they went, as written to a verified TSV."""

    bucket: str
    remote: str
    local: Path
    verified_at: str
    records: tuple[FileRecord, ...]


def _bucket_prefix(remote: str) -> str:
    """The prefix, with one trailing slash, under which a unit's files sit in the bucket."""
    return remote.strip("/") + "/"


def remote_sizes(api: Any, bucket: str, remote: str) -> dict[str, int]:
    """The size of every file under a bucket prefix, by path relative to the prefix."""
    prefix = _bucket_prefix(remote)
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


def _restore_targets(verified: VerifiedRecord, destination: Path) -> dict[str, Path]:
    """Where each recorded file lands under ``destination``, after the refusals that come before any download."""
    if not verified.records:
        raise RestoreFailed(f"the record of {verified.remote} names no file, so there is nothing to restore")
    unsafe = [
        r.path for r in verified.records if not r.path or Path(r.path).is_absolute() or ".." in Path(r.path).parts
    ]
    if unsafe:
        raise RestoreFailed(f"the record names paths that would leave the destination: {unsafe[:3]}")
    if destination.is_file() or (destination.is_dir() and any(destination.iterdir())):
        raise RestoreFailed(
            f"{destination} exists and is not an empty directory, so a restore would mix with its files"
        )
    if _inside(destination, [verified.local]):
        raise RestoreFailed(f"{destination} is the unit's own directory or inside it: restore beside it, not into it")
    return {r.path: destination / r.path for r in verified.records}


def restore_verified(api: Any, verified: VerifiedRecord, destination: Path) -> list[Path]:
    """Download every file of a verified unit from the bucket into ``destination`` and check each one against the record.

    Args:
        api: The Hub client, for the bucket's files.
        verified: The unit's verified record. It names the bucket, the prefix of the unit and every file.
        destination: A directory that does not exist or is empty. It gets the recorded relative paths. It is neither
            the unit's own directory nor inside it, and the unit's local files are never touched.

    Returns:
        The restored paths, in the record's order.

    Raises:
        RestoreFailed: A refusal before any download (no recorded file, a path that leaves the destination, a
            destination that is not empty or lies in the unit's directory), or, after every file was tried, a list of
            each file that is missing from the bucket, has another size or has another sha256. The files that were
            downloaded stay in ``destination`` for inspection, and another restore needs an empty directory.
    """
    targets = _restore_targets(verified, destination)
    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    prefix = _bucket_prefix(verified.remote)
    # The client skips (with a warning) a path that is absent from the bucket, so a missing file shows below.
    api.download_bucket_files(
        verified.bucket, [(prefix + path, target) for path, target in targets.items()], raise_on_missing_files=False
    )
    downloaded = {r.path: r for r in directory_records(destination, None, (), None, HASH_WORKERS)}
    problems = mismatches(verified.records, {path: r.size for path, r in downloaded.items()})
    problems.extend(
        f"{r.path}: sha256 {downloaded[r.path].sha256} downloaded, {r.sha256} recorded"
        for r in verified.records
        if r.path in downloaded and downloaded[r.path].size == r.size and downloaded[r.path].sha256 != r.sha256
    )
    if problems:
        raise RestoreFailed(f"the bucket's {verified.remote} differs from its record: {sorted(problems)}")
    LOGGER.info("restored %d files of %s into %s", len(targets), verified.remote, destination)
    return list(targets.values())
