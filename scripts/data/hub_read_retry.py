# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.

"""Repeat a failed read of a Hub file when the failure is one a repeat can outlast.

A read of the Hub is an HTTP range request, and a long one is cut sooner or later: the server closes a connection
part-way through a response body, a gateway answers 502 or 504 for a few minutes, a route to the Hub goes away. The
libraries that make the requests repeat too little of that:

- ``huggingface_hub`` repeats a range request a few times over about twenty seconds, on a timeout, a failed
  connection or a 5xx status, and not at all when the body is cut short (``httpx.RemoteProtocolError``). When a
  request cannot connect it also closes its shared client and retries on the reference it already held, so what
  surfaces is httpx's closed-client ``RuntimeError``, not the network error.
- ``datasets`` wraps the ``read`` of every file a stream opens so that it is repeated, but only on the errors of the
  HTTP clients it was written against (``aiohttp`` and ``requests``), which a ``huggingface_hub`` built on ``httpx``
  never raises.

``hub_read_failure_is_transient`` is the one rule for which failures a repeat can outlast, and ``retrying`` the one
loop that repeats on them, with a wait that doubles up to ``MAX_WAIT_SECONDS``. A read is repeated in one of two ways:

- **In place.** ``reads_repeated_in_place`` repeats the ``read`` of a file that reads by range through fsspec's
  cache, and ``streamed_reads_retried`` has ``datasets`` apply it to the files its streams open, since a stream cannot
  be started again without re-reading everything before the failure. That is safe for such a file, and only for one:
  fsspec's buffered ``read`` advances its position, and its read-ahead cache replaces its block, only after a fetch
  has succeeded, so a ``read`` that raised left both where they were. Any other file is not repeated, because its
  ``read`` does not give the same bytes when run again, and a repeat would end the stream short without an error:
  a wrapper (text mode, a decompressor) has already consumed what it buffered when its ``read`` raises, and the Hub's
  unbuffered stream file, which ``datasets`` opens for an archive it iterates, answers a ``read`` after a failed one
  with an end of file. Such a file's failure ends the stream, as it would without this module. A parquet or JSON
  stream opens its files buffered.
- **From a fresh open.** ``configs/control_pretraining/hub_parquet.py``'s ``read_hub_file`` opens the file again and
  runs the whole read again, for a parquet reader that buffers ahead and so cannot be resumed part-way.

``httpx``, ``huggingface_hub`` and ``datasets`` are imported where they are used, so a tool that only counts files can
import this module outside the container.
"""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


LOGGER = logging.getLogger(__name__)

# The longest wait between two attempts; waits double from the first until they reach it.
MAX_WAIT_SECONDS = 60.0


def hub_read_failure_is_transient(error: BaseException) -> bool:
    """Whether a failed Hub read is worth repeating: a transport error, a rate limit or a server error, or a
    request that huggingface_hub sent on a client it had already closed.

    A transport error is a timeout, a connection that could not be made or was reset, or one the server closed before
    the response was complete. The closed client is the library's own backoff: when a request cannot connect it
    closes the shared httpx client, then retries on the reference it took before its loop, so what a connection failure
    surfaces is httpx's closed-client ``RuntimeError`` rather than the network error. The next request builds a fresh
    client, so a repeat succeeds where the library's retry could not. Any other status (a 401, a 404, a range the file
    does not hold) says the request itself is wrong, which repeating it does not change.
    """
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, HfHubHTTPError):
        status = getattr(getattr(error, "response", None), "status_code", None)
        return status == 429 or (status is not None and status >= 500)
    return isinstance(error, RuntimeError) and "client has been closed" in str(error)


def retrying(operation: Callable[..., Any], attempts: int, first_wait_seconds: float, what: str) -> Callable[..., Any]:
    """``operation``, repeated on a transient Hub read failure until it returns or has been tried ``attempts`` times.

    Each failed attempt is logged with its error, naming the operation as ``what``. The wait before the next is
    ``first_wait_seconds``, doubled each time up to ``MAX_WAIT_SECONDS``, so the attempts together can outlast an
    outage of minutes: such an outage fails every request at once, and a budget of seconds is spent before the route
    is back. A failure ``hub_read_failure_is_transient`` rejects, and the last attempt's, is raised as it is.
    ``operation`` must give the same result when run again.
    """
    if attempts < 1:
        raise ValueError(f"{what} needs at least one attempt, not {attempts}")

    def repeated(*args: Any, **kwargs: Any) -> Any:
        attempt = 1
        while True:
            try:
                return operation(*args, **kwargs)
            except Exception as error:
                if attempt == attempts or not hub_read_failure_is_transient(error):
                    raise
                wait = min(first_wait_seconds * 2 ** (attempt - 1), MAX_WAIT_SECONDS)
                LOGGER.warning(
                    "%s failed (%s: %s); attempt %d of %d, repeating it in %.0f s",
                    what,
                    type(error).__name__,
                    str(error)[:160],
                    attempt,
                    attempts,
                    wait,
                )
            time.sleep(wait)
            attempt += 1

    return repeated


def reads_repeated_in_place(file_obj: Any, attempts: int, first_wait_seconds: float, what: str) -> Any:
    """``file_obj``, its ``read`` repeated on a transient Hub read failure when that ``read`` is one known to give the
    same bytes when run again.

    Two are: fsspec's buffered ``read`` (``AbstractBufferedFile.read``), which leaves the file's position and cache
    where they were when a fetch fails, and the Hub's buffered file's (``HfFileSystemFile.read``), which is that
    ``read`` except for a whole-file read from the start, made through a stream it opens and closes itself. A file
    with any other ``read`` is returned as it is, a subclass that overrides it included: the Hub's unbuffered stream
    file is a buffered-file subclass whose own ``read`` returns an end of file after a failed one.
    """
    from fsspec.spec import AbstractBufferedFile
    from huggingface_hub.hf_file_system import HfFileSystemFile

    if getattr(type(file_obj), "read", None) in (AbstractBufferedFile.read, HfFileSystemFile.read):
        file_obj.read = retrying(file_obj.read, attempts, first_wait_seconds, what)
    return file_obj


@contextmanager
def streamed_reads_retried(attempts: int, first_wait_seconds: float) -> Iterator[None]:
    """Inside the block, the files a ``datasets`` stream opens to read by range repeat their reads on a transient Hub
    read failure.

    ``datasets`` gives each remote file it opens to one function that wraps the file's ``read``; inside the block that
    function also applies ``reads_repeated_in_place``, around the wrapper ``datasets`` installs itself. A parquet or
    JSON stream opens its files buffered, and those are repeated; a text stream's wrapped files and an archive's
    unbuffered ones are not, and fail as they did. A file opened inside the block keeps the repeated ``read`` after
    the block ends.
    """
    from datasets.utils import file_utils

    wrap_for_known_clients = file_utils._add_retries_to_file_obj_read_method

    def wrap_for_the_hub_client_too(file_obj: Any) -> Any:
        file_obj = wrap_for_known_clients(file_obj)
        return reads_repeated_in_place(file_obj, attempts, first_wait_seconds, "a read of a streamed file")

    file_utils._add_retries_to_file_obj_read_method = wrap_for_the_hub_client_too
    try:
        yield
    finally:
        file_utils._add_retries_to_file_obj_read_method = wrap_for_known_clients
