# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The sha256 digests that the tools record: of a file, of a directory's files, and of a chat template.

Standard library only, and nothing runs at import, so any tool may import it.

Catalogue:
    HASH_BLOCK_BYTES: the bytes read per step when hashing a file.
    HASH_WORKERS: the files hashed at once by directory_records.
    FileRecord: the path, size and sha256 of one file.
    file_sha256: the sha256 of a file.
    selected_paths: the files of a directory that include and exclude patterns select.
    directory_records: the FileRecord of each selected file of a directory.
    chat_template_sha256: the sha256 of a tokenizer's chat template.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HASH_BLOCK_BYTES = 1 << 26  # 64 MiB read per step when hashing a file
HASH_WORKERS = 8


@dataclass(frozen=True)
class FileRecord:
    """One file of a directory: its path relative to the directory, its size and its sha256."""

    path: str
    size: int
    sha256: str


def file_sha256(path: str | Path, block: int = HASH_BLOCK_BYTES) -> str:
    """The sha256 of the file at ``path``, read ``block`` bytes at a time."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def selected_paths(source: Path, include: Iterable[str] | None, exclude: Iterable[str]) -> list[str]:
    """The POSIX paths, relative to ``source``, of the files that these filters select, sorted.

    A path is selected when it matches an ``include`` pattern (or ``include`` is None) and no ``exclude`` pattern.
    ``*`` matches across ``/``, as the Hub client's patterns do.
    """
    include_list, exclude_list = (list(include) if include else None), list(exclude)
    paths = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = path.relative_to(source).as_posix()
        if include_list is not None and not any(fnmatch.fnmatchcase(relative, pattern) for pattern in include_list):
            continue
        if any(fnmatch.fnmatchcase(relative, pattern) for pattern in exclude_list):
            continue
        paths.append(relative)
    return paths


def directory_records(
    source: Path,
    include: Iterable[str] | None,
    exclude: Iterable[str],
    previous: Iterable[FileRecord] | None,
    workers: int,
) -> tuple[FileRecord, ...]:
    """The record of every file of ``source`` that the filters select, in path order, hashed by ``workers`` threads.

    A file that ``previous`` records at the same path and size keeps its recorded sha256 and is not read. Pass None
    to read every file.
    """
    known = {r.path: r for r in previous} if previous is not None else {}
    paths = selected_paths(source, include, exclude)
    sizes = {p: (source / p).stat().st_size for p in paths}

    def record(path: str) -> FileRecord:
        old = known.get(path)
        digest = old.sha256 if old is not None and old.size == sizes[path] else file_sha256(source / path)
        return FileRecord(path, sizes[path], digest)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return tuple(pool.map(record, paths))


def chat_template_sha256(template: str | dict | None) -> str | None:
    """The sha256 of a chat template, or None when the tokenizer has none.

    A template given as text hashes as its UTF-8 bytes. That equals the sha256 of the ``chat_template.jinja`` it was
    read from. Several templates given by name, as a dict, hash as their JSON with sorted keys.
    """
    if template is None:
        return None
    text = template if isinstance(template, str) else json.dumps(template, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
