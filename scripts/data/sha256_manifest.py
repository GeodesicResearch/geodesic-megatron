# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A manifest of files' sha256 digests and sizes, and the check of a set of files against one.

A manifest is a UTF-8 text file of one line per file::

    <sha256>  <size>  <path>

the file's sha256 as 64 lowercase hex digits, two spaces, its size in bytes as a decimal integer (no sign, no leading
zero), two spaces, and its path (for a Hub dataset's file, its path in the repository). Each line ends in a newline,
which the last line may omit. ``parse_manifest`` reads one strictly and refuses, naming the manifest and the line,
anything else: an empty manifest, a blank line, a carriage return, one space or a tab where two spaces are written, an
uppercase or short digest, a signed or zero-padded size, a path that is absolute, starts or ends with whitespace, holds
a control character or an empty, ``.`` or ``..`` component, and a path listed twice.

``check_files_against_manifest`` checks the files a reader would read, named by those paths, against a manifest. The
set of paths must be exactly the manifest's, which is checked before any file is fetched. Each file is then fetched by
a function the caller supplies (a download into a cache, for a Hub dataset) and must have the manifest's size and
sha256. Files are fetched and hashed by a pool of a stated number of threads: a fetch waits on the network and
``hashlib`` releases the GIL while it hashes, so threads run both in parallel. Every size or digest that differs is
collected and raised together in one ``ManifestMismatch``; an error a fetch raises is raised as it is, once the fetches
already running have ended, and no further file is fetched.

Standard library only.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger(__name__)

_LINE = re.compile(r"(?P<sha256>[0-9a-f]{64})  (?P<size>0|[1-9][0-9]*)  (?P<path>.+)")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
HASH_BLOCK_BYTES = 1 << 26  # 64 MiB read per step when hashing a file
# What a check returns that did not run, because another check had already failed with an error.
_SKIPPED = object()


class ManifestMismatch(ValueError):
    """Files that are not the ones a manifest lists, or not the bytes it lists them with."""


@dataclass(frozen=True)
class ManifestEntry:
    """One line of a manifest."""

    sha256: str
    size: int
    path: str


@dataclass(frozen=True)
class Manifest:
    """A parsed manifest: where it was read from, its own sha256, and its entries by path, in the manifest's order."""

    source: str
    sha256: str
    entries: dict[str, ManifestEntry]

    @property
    def total_bytes(self) -> int:
        """The sum of the sizes the manifest lists."""
        return sum(entry.size for entry in self.entries.values())

    def record(self) -> dict:
        """What a run that checked its files against this manifest records of it."""
        return {
            "path": self.source,
            "sha256": self.sha256,
            "files": len(self.entries),
            "total_bytes": self.total_bytes,
        }


def _path_problem(path: str) -> str | None:
    """Why ``path`` is not a manifest path, or ``None`` when it is one."""
    if path != path.strip():
        return "starts or ends with whitespace"
    if _CONTROL.search(path):
        return "holds a control character"
    if path.startswith("/"):
        return "is absolute"
    if any(component in ("", ".", "..") for component in path.split("/")):
        return "has an empty, '.' or '..' component"
    return None


def parse_manifest(source: str | Path) -> Manifest:
    """Read the manifest at ``source`` strictly (the format is the module docstring's).

    Raises ``ValueError``, naming the manifest and the line, for a manifest that cannot be read and for any content
    the format does not describe.
    """
    try:
        content = Path(source).read_bytes()
    except OSError as error:
        raise ValueError(f"manifest {source} cannot be read: {error}") from error
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"manifest {source} is not UTF-8 text: {error}") from error
    lines = text.split("\n")
    if lines[-1] == "":  # the newline that ends the last line, which only the last line may omit
        lines.pop()
    if not lines:
        raise ValueError(f"manifest {source} lists no files")
    entries: dict[str, ManifestEntry] = {}
    for number, line in enumerate(lines, start=1):
        match = _LINE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"manifest {source}:{number}: expected '<64 lowercase hex sha256>  <size>  <path>' with two spaces "
                f"between the fields, got {line[:300]!r}"
            )
        path = match["path"]
        problem = _path_problem(path)
        if problem is not None:
            raise ValueError(f"manifest {source}:{number}: path {path!r} {problem}")
        if path in entries:
            raise ValueError(f"manifest {source}:{number}: path {path!r} is listed twice")
        entries[path] = ManifestEntry(sha256=match["sha256"], size=int(match["size"]), path=path)
    return Manifest(source=str(source), sha256=hashlib.sha256(content).hexdigest(), entries=entries)


def file_sha256(path: str | Path, block: int = HASH_BLOCK_BYTES) -> str:
    """The sha256 of the file at ``path``, read ``block`` bytes at a time."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(entry: ManifestEntry, local: Path) -> str | None:
    """How the file at ``local`` differs from ``entry``, or ``None`` when it has the entry's size and sha256.

    The size is compared first, so a file of the wrong size is not hashed.
    """
    size = local.stat().st_size
    if size != entry.size:
        return f"{entry.path}: {size} bytes, the manifest lists {entry.size}"
    digest = file_sha256(local)
    if digest != entry.sha256:
        return f"{entry.path}: sha256 {digest}, the manifest lists {entry.sha256}"
    return None


def check_files_against_manifest(
    manifest: Manifest,
    files: Iterable[str],
    fetch: Callable[[str], Path],
    workers: int,
    on_checked: Callable[[ManifestEntry, int, int], None] | None = None,
) -> None:
    """Check that ``files`` are exactly ``manifest``'s and that each, fetched by ``fetch``, has its size and sha256.

    ``files`` are the paths a reader would read; ``fetch`` turns one into a local file. ``workers`` threads fetch and
    hash at once. ``on_checked(entry, done, total)``, when given, is called from the calling thread as each file
    passes. Raises ``ManifestMismatch`` for a set of paths other than the manifest's (before anything is fetched) and
    for files whose size or sha256 differ (all of them, once every file has been checked), and ``ValueError`` for
    fewer than one worker.
    """
    if workers < 1:
        raise ValueError(f"at least one worker must check the files, got {workers}")
    files = list(files)
    repeated = sorted(path for path, count in Counter(files).items() if count > 1)
    missing = sorted(set(manifest.entries) - set(files))
    extra = sorted(set(files) - set(manifest.entries))
    if repeated or missing or extra:
        raise ManifestMismatch(
            f"the files read are not the ones manifest {manifest.source} lists: {len(missing)} it lists are not read "
            f"{missing}, {len(extra)} read are not listed {extra}, and these are read more than once {repeated}"
        )

    # Set by the worker whose fetch or check raised, before its future completes, so a check a worker starts after
    # it fetches nothing.
    failed = threading.Event()

    def fetch_and_check(entry: ManifestEntry):
        if failed.is_set():
            return _SKIPPED
        try:
            return check_file(entry, Path(fetch(entry.path)))
        except BaseException:
            failed.set()
            raise

    entries = list(manifest.entries.values())
    differences: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fetch_and_check, entry): entry for entry in entries}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                entry = pending.pop(future)
                error = future.exception()
                if error is not None:
                    for other in pending:
                        other.cancel()
                    raise error
                difference = future.result()
                if difference is _SKIPPED:  # the error that skipped it is raised when its own check is reached
                    continue
                done += 1
                if difference is not None:
                    differences.append(difference)
                    LOGGER.error("manifest %s: %s", manifest.source, difference)
                elif on_checked is not None:
                    on_checked(entry, done, len(entries))
    if differences:
        raise ManifestMismatch(
            f"{len(differences)} of the {len(entries)} files manifest {manifest.source} lists differ from it: "
            + "; ".join(sorted(differences))
        )
