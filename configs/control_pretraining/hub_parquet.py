# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Finding and reading one dataset config's train parquet files: on the Hub by range request, or in a local copy.

A config's files are the ones its dataset card (``README.md``) lists for it under the YAML header's ``configs:``,
which is what the ``datasets`` loader reads: ``push_to_hub`` writes ``<config>/train-*`` for a config of its own
directory, and a config that joins others' files (a union of a corpus's row-contiguous parts) lists those files
explicitly, in order. ``config_train_patterns`` reads a card's train patterns for a config and ``resolve_train_files``
resolves them as the loader does, pattern by pattern in the card's order, a pattern's matches in path order. The Hub
(``hub_parquet_files``) and a local copy laid out as the repository is (``local_parquet_files``) apply that one rule,
so they choose the same files in the same order. ``read_hub_file`` opens one Hub file by range request and re-opens it
on a transient failure.

The corpus tools of this directory share these: the filtered-arm audit (``audit_filtered_corpora.py``)
and the per-document checks (``corpus_documents.py``). Network packages are imported where they are
used, so the audit's counts layer still runs outside the container.
"""

from __future__ import annotations

import fnmatch
import sys
import time
from collections.abc import Callable
from pathlib import Path


HUB_READ_ATTEMPTS = 12  # opens of one Hub file before a transient failure is fatal
HUB_READ_MAX_WAIT_S = 60.0  # cap on the doubling wait between those opens
CARD = "README.md"
# Glob syntax the corpus tools resolve: `*` and `?` in a pattern's last component. Anything else a loader pattern may
# hold is refused rather than read some other way.
UNRESOLVED_GLOB = set("[]{}")


def config_train_patterns(card: str, config: str, where: str) -> list[str]:
    """The train split's data-file patterns a dataset card lists for ``config``, in the card's order.

    The card's YAML header (between its opening ``---`` lines) declares ``configs:``, each with a ``config_name`` and
    ``data_files``: one pattern, a list of patterns (all train), or a list of ``{split, path}`` entries whose ``path``
    is one pattern or a list. Raises ``FileNotFoundError``, naming ``where`` and the config, for a card that declares
    no such config or no train split for it, which the loader could not read either, and ``ValueError`` for a
    ``data_files`` of another shape.
    """
    import yaml

    lines = card.splitlines()
    closing = [index for index, line in enumerate(lines) if line.strip() == "---"]
    if len(closing) < 2 or closing[0] != 0:
        raise FileNotFoundError(f"{where}: the dataset card has no YAML header, so config {config!r} is not declared")
    header = yaml.safe_load("\n".join(lines[1 : closing[1]])) or {}
    entries = [entry for entry in header.get("configs") or [] if entry.get("config_name") == config]
    if len(entries) != 1:
        raise FileNotFoundError(f"{where}: the dataset card declares config {config!r} {len(entries)} times, not once")
    data_files = entries[0].get("data_files")
    if isinstance(data_files, str):
        return [data_files]
    if isinstance(data_files, list) and data_files and all(isinstance(item, str) for item in data_files):
        return list(data_files)
    if isinstance(data_files, list) and data_files and all(isinstance(item, dict) for item in data_files):
        train = [item.get("path") for item in data_files if item.get("split") == "train"]
        if len(train) != 1:
            raise FileNotFoundError(f"{where}: config {config!r} declares {len(train)} train splits, not one")
        paths = [train[0]] if isinstance(train[0], str) else train[0]
        if isinstance(paths, list) and paths and all(isinstance(path, str) for path in paths):
            return list(paths)
    raise ValueError(
        f"{where}: config {config!r} has data_files of a shape the corpus tools do not read: {data_files!r}"
    )


def resolve_train_files(patterns: list[str], list_directory: Callable[[str], list[str]], where: str) -> list[str]:
    """The repository-relative files ``patterns`` name, in the loader's order: pattern by pattern, an explicit path as
    itself and a pattern holding ``*`` or ``?`` in its last component as the matching parquet files of its directory
    (``list_directory`` lists a directory's direct files, repository-relative) in path order.

    Raises ``ValueError`` for glob syntax beyond that, and ``FileNotFoundError`` for a pattern that matches nothing.
    """
    files = []
    for pattern in patterns:
        directory, _, name = pattern.rpartition("/")
        if any(char in UNRESOLVED_GLOB or char in "*?" for char in directory) or UNRESOLVED_GLOB & set(name):
            raise ValueError(
                f"{where}: data-file pattern {pattern!r} uses glob syntax the corpus tools do not resolve"
            )
        if "*" not in name and "?" not in name:
            files.append(pattern)
            continue
        matched = sorted(
            path
            for path in list_directory(directory)
            if path.endswith(".parquet") and fnmatch.fnmatchcase(path, pattern)
        )
        if not matched:
            raise FileNotFoundError(f"{where}: data-file pattern {pattern!r} matches no parquet file")
        files.extend(matched)
    return files


def hub_parquet_files(dataset: str, revision: str, config: str) -> list[str]:
    """The train parquet files of one config on the Hub at ``revision`` (repository-relative), in the loader's order:
    what the dataset card at that revision lists for the config (``config_train_patterns``, ``resolve_train_files``)."""
    from huggingface_hub import HfApi, HfFileSystem

    where = f"{dataset}@{revision}"
    card = read_hub_file(HfFileSystem(), hub_file_url(dataset, revision, CARD), lambda handle: handle.read())
    api = HfApi()

    def list_directory(directory: str) -> list[str]:
        listing = api.list_repo_tree(dataset, path_in_repo=directory, revision=revision, repo_type="dataset")
        return [entry.path for entry in listing]

    patterns = config_train_patterns(card.decode("utf-8"), config, where)
    return resolve_train_files(patterns, list_directory, where)


def local_parquet_files(repo: Path, config: str) -> list[Path]:
    """The train parquet files of one config of a local copy laid out as the Hub repository is, chosen by its own
    dataset card exactly as ``hub_parquet_files`` chooses them on the Hub; a named file that is absent is an error."""
    card = repo / CARD
    if not card.is_file():
        raise FileNotFoundError(f"{repo}: no dataset card {CARD}, so config {config!r} is not declared")

    def list_directory(directory: str) -> list[str]:
        root = repo / directory
        return (
            [path.relative_to(repo).as_posix() for path in root.iterdir() if path.is_file()] if root.is_dir() else []
        )

    files = [
        repo / path
        for path in resolve_train_files(
            config_train_patterns(card.read_text(), config, str(repo)), list_directory, str(repo)
        )
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{repo}: config {config!r} names files that do not exist: {missing}")
    return files


def hub_file_url(dataset: str, revision: str, path: str) -> str:
    """The ``HfFileSystem`` URL of one file of a Hub dataset at a revision."""
    return f"datasets/{dataset}@{revision}/{path}"


def hub_read_failure_is_transient(error: BaseException) -> bool:
    """Whether a failed Hub read is worth a fresh open: a transport error, a rate limit or a server
    error, or a request that huggingface_hub sent on a client it had already closed.

    The last is the library's own backoff: when a request cannot connect it closes the shared
    httpx client, then retries on the reference it took before its loop, so what a connection
    failure surfaces is httpx's closed-client ``RuntimeError`` rather than the network error. The
    next request builds a fresh client, so a re-open succeeds where the library's retry could not.
    """
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, HfHubHTTPError):
        status = getattr(getattr(error, "response", None), "status_code", None)
        return status == 429 or (status is not None and status >= 500)
    return isinstance(error, RuntimeError) and "client has been closed" in str(error)


def read_hub_file(fs, url: str, work, attempts: int = HUB_READ_ATTEMPTS):
    """Open one Hub file by range request and return ``work(handle)``, re-opening on a transient failure.

    A range read of a multi-gigabyte parquet file can be cut mid-body by the Hub (a truncated
    response, a reset connection, a 429 or a 5xx); huggingface_hub retries the request that
    failed, not the read that was in flight, so the failure surfaces from the parquet reader
    hours into a read. Each attempt re-opens the file and re-runs ``work`` from the start, so
    a partial read is never combined with a fresh one; every retry is printed to stderr; a
    failure that ``hub_read_failure_is_transient`` rejects, and the last transient failure, are
    raised. ``work`` must therefore be a pure function of the handle — it is called again on retry.

    The wait between attempts doubles from 2 s and caps at ``HUB_READ_MAX_WAIT_S``, so the
    attempts together outlast an egress outage of several minutes: such an outage takes every
    route to the Hub away at once, every request made during it fails immediately, and a budget
    of seconds is spent before the route returns. An outage longer than the attempts span still
    fails the read.
    """
    for attempt in range(1, attempts + 1):
        try:
            with fs.open(url, "rb") as fh:
                return work(fh)
        except Exception as error:
            if not hub_read_failure_is_transient(error) or attempt == attempts:
                raise
            delay = min(2.0**attempt, HUB_READ_MAX_WAIT_S)
            print(
                f"hub read of {url} failed ({type(error).__name__}: {str(error)[:160]}); "
                f"attempt {attempt} of {attempts}, retrying in {delay:.0f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")
