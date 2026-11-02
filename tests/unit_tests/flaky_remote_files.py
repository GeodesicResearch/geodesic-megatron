# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Local files behind a remote-looking fsspec protocol, whose range fetches fail on a schedule.

This stands in for the one thing these tests cannot reach: a Hub connection that drops, or answers with a gateway
error, in the middle of a read. Everything on the near side of it is the real thing. A file is fsspec's own buffered
file (``AbstractBufferedFile``), the class the Hub's files are, reading through its real read-ahead cache; only its
range fetch, the one method a filesystem supplies, is local and fails when told to. ``datasets`` resolves and streams
the files and pyarrow reads them, and because the protocol is not a local one ``datasets`` treats them as remote and
gives each to its read-retry wrapper, as it does a Hub file.
"""

from __future__ import annotations

import threading

import fsspec
import httpx
import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractBufferedFile
from huggingface_hub.errors import HfHubHTTPError


class _FlakyRemoteFile(AbstractBufferedFile):
    """A buffered file whose bytes come from the local file at its path, one range fetch at a time."""

    def _fetch_range(self, start, end):
        files = type(self.fs)
        with files._count:
            files.fetches += 1
            fetch = files.fetches
        if fetch in files.failures:
            files.raised.append(fetch)
            raise files.failures[fetch]
        with open(self.path, "rb") as handle:
            handle.seek(start)
            return handle.read(end - start)


class FlakyRemoteFiles(LocalFileSystem):
    """Local files under ``flakyremote://``, listed as the local tree and opened as buffered remote files. ``fail``
    names the range fetches, counted from 1 across every file opened since, that raise instead of returning bytes."""

    protocol = "flakyremote"
    local_file = False
    # Small enough that a test file of a few hundred kilobytes takes several fetches, so a failure can be placed at
    # the start, in the middle or near the end of a read.
    block_size = 16 * 1024
    fetches = 0
    failures: dict[int, BaseException] = {}
    raised: list[int] = []
    _count = threading.Lock()

    @classmethod
    def fail(cls, failures: dict[int, BaseException]) -> None:
        """Start counting fetches again and raise ``failures[n]`` on fetch ``n``."""
        cls.fetches = 0
        cls.failures = dict(failures)
        cls.raised = []

    @classmethod
    def url(cls, path) -> str:
        """The URL of the local file or glob pattern ``path`` under this protocol."""
        return f"{cls.protocol}://{path}"

    @classmethod
    def _strip_protocol(cls, path):
        path = str(path)
        prefix = f"{cls.protocol}://"
        if path.startswith(prefix):
            path = "/" + path[len(prefix) :].lstrip("/")
        return super()._strip_protocol(path)

    def _open(self, path, mode="rb", **kwargs):
        return _FlakyRemoteFile(self, self._strip_protocol(path), mode=mode, block_size=self.block_size)


@pytest.fixture
def flaky_remote_files():
    """``FlakyRemoteFiles``, registered with fsspec and failing nothing until a test calls ``fail``."""
    fsspec.register_implementation(FlakyRemoteFiles.protocol, FlakyRemoteFiles, clobber=True)
    FlakyRemoteFiles.fail({})
    yield FlakyRemoteFiles
    FlakyRemoteFiles.fail({})


def dropped_connection() -> httpx.RemoteProtocolError:
    """What the Hub client raises when the server closes a connection part-way through a response body."""
    return httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body (received 9882783 bytes, expected 31909801)"
    )


def request_on_a_closed_client() -> RuntimeError:
    """What surfaces when the Hub client's own backoff retries on the client it has just closed: httpx's error for a
    request on a closed client, raised by a real closed client before anything is sent."""
    client = httpx.Client()
    client.close()
    try:
        client.request("GET", "https://huggingface.co/never-sent")
    except RuntimeError as error:
        return error
    raise AssertionError("a closed httpx client accepted a request")


def hub_status(status: int) -> HfHubHTTPError:
    """What the Hub client raises for a response with ``status``."""
    request = httpx.Request("GET", "https://huggingface.co/datasets/org/corpus/resolve/main/train.parquet")
    return HfHubHTTPError(f"{status} from the Hub", response=httpx.Response(status, request=request))
