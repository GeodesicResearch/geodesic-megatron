# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Finding and reading one dataset config's train parquet files: on the Hub by range request, or in a local copy.

A dataset repository lays each config out as ``<config>/train-0000k-of-0000n.parquet`` (beside its
``_provenance.json`` and any other split), and the ``datasets`` loader concatenates the train files in
path order. ``train_parquet_paths`` is that selection, applied to repository-relative paths, so the
Hub listing (``hub_parquet_files``) and a local copy laid out as the repository is
(``local_parquet_files``) choose the same files in the same order. ``read_hub_file`` opens one Hub
file by range request and re-opens it on a transient failure.

The corpus tools of this directory share these: the filtered-arm audit (``audit_filtered_corpora.py``)
and the per-document checks (``corpus_documents.py``). Network packages are imported where they are
used, so the audit's counts layer still runs outside the container.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from pathlib import Path


HUB_READ_ATTEMPTS = 12  # opens of one Hub file before a transient failure is fatal
HUB_READ_MAX_WAIT_S = 60.0  # cap on the doubling wait between those opens


def train_parquet_paths(paths: Iterable[str]) -> list[str]:
    """The train split's parquet files among a config's repository-relative paths, in the loader's order."""
    return sorted(path for path in paths if path.endswith(".parquet") and "/train" in path)


def hub_parquet_files(dataset: str, revision: str, config: str) -> list[str]:
    """The train parquet files of one config on the Hub at ``revision``, in the order the loader concatenates them."""
    from huggingface_hub import HfApi

    listing = HfApi().list_repo_tree(dataset, path_in_repo=config, revision=revision, repo_type="dataset")
    files = train_parquet_paths(entry.path for entry in listing)
    if not files:
        raise FileNotFoundError(f"{dataset}@{revision}: no train parquet files under {config!r}")
    return files


def local_parquet_files(repo: Path, config: str) -> list[Path]:
    """The train parquet files of one config of a local copy laid out as the Hub repository is.

    The files directly under ``<repo>/<config>`` are selected exactly as ``hub_parquet_files``
    selects the Hub listing of that directory, which is not recursive either.
    """
    directory = repo / config
    if not directory.is_dir():
        raise FileNotFoundError(f"{repo}: no config directory {config!r}")
    relative = [path.relative_to(repo).as_posix() for path in directory.iterdir() if path.is_file()]
    files = train_parquet_paths(relative)
    if not files:
        raise FileNotFoundError(f"{repo}: no train parquet files under {config!r}")
    return [repo / path for path in files]


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
