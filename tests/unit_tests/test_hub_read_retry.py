# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A read of a Hub file is repeated on the transient errors the installed Hub client raises
(``scripts/data/hub_read_retry.py``), through a real ``datasets`` stream over files whose reads fail on a schedule."""

from __future__ import annotations

import hashlib
import io
import logging

import fsspec
import httpx
import pytest
from datasets import load_dataset
from datasets.utils import file_utils
from huggingface_hub.errors import HfHubHTTPError
from scripts.data import hub_read_retry
from scripts.data.hub_read_retry import (
    hub_read_failure_is_transient,
    reads_repeated_in_place,
    retrying,
    streamed_reads_retried,
)

from tests.unit_tests.corpora_fixtures import write_parquet_dataset
from tests.unit_tests.flaky_remote_files import (
    dropped_connection,
    flaky_remote_files,  # noqa: F401  (a fixture)
    hub_status,
    request_on_a_closed_client,
)


# Text that does not compress, so each parquet file is a few hundred kilobytes and is read in many range fetches.
ROWS = [hashlib.sha256(str(index).encode()).hexdigest() * 8 for index in range(3000)]


@pytest.fixture
def stream_rows(flaky_remote_files, tmp_path):  # noqa: F811
    """Read every row of a three-file parquet dataset through a ``datasets`` stream over the flaky protocol."""
    repo = write_parquet_dataset(tmp_path / "corpus", "data", {"text": ROWS}, files=3)
    files = [flaky_remote_files.url(path) for path in sorted((repo / "data").glob("train-*.parquet"))]

    def read() -> list[str]:
        stream = load_dataset("parquet", data_files={"train": files}, split="train", streaming=True)
        return [row["text"] for row in stream]

    return read


@pytest.fixture
def fetches_of_a_clean_stream(flaky_remote_files, stream_rows):  # noqa: F811
    """How many range fetches one undisturbed pass over the stream makes."""
    assert stream_rows() == ROWS
    return flaky_remote_files.fetches


class TestTheInstalledLibrariesDoNotRepeatTheseReads:
    def test_a_stream_ends_on_a_connection_closed_part_way_through_a_body(
        self,
        flaky_remote_files,  # noqa: F811
        stream_rows,
        fetches_of_a_clean_stream,
    ):
        """The failure that ended two ClimbMix slice prepares, each more than 80% through, on 2026-10-11:
        ``datasets`` wraps every remote file's ``read`` in its own retry, but that retry does not know the Hub
        client's error, so the stream ends on it. When this test fails, the installed ``datasets`` repeats the read
        itself and ``streamed_reads_retried`` has nothing left to add for it."""
        flaky_remote_files.fail({fetches_of_a_clean_stream // 2: dropped_connection()})
        with pytest.raises(httpx.RemoteProtocolError, match="peer closed connection"):
            stream_rows()


class TestStreamedReadsRetried:
    def test_a_stream_outlasts_dropped_connections_and_reads_the_same_rows(
        self,
        flaky_remote_files,  # noqa: F811
        stream_rows,
        fetches_of_a_clean_stream,
    ):
        middle = fetches_of_a_clean_stream // 2
        # The first read of the stream, then one read failing twice in a row (its repeat is the next call).
        failures = {1: dropped_connection(), middle: dropped_connection(), middle + 1: dropped_connection()}
        flaky_remote_files.fail(failures)
        with streamed_reads_retried(attempts=3, first_wait_seconds=0.0):
            assert stream_rows() == ROWS
        assert flaky_remote_files.raised == sorted(failures)
        assert flaky_remote_files.fetches == fetches_of_a_clean_stream + len(failures)

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_a_stream_outlasts_a_hub_status_that_passes(
        self,
        flaky_remote_files,  # noqa: F811
        stream_rows,
        fetches_of_a_clean_stream,
        status,
    ):
        flaky_remote_files.fail({fetches_of_a_clean_stream // 2: hub_status(status)})
        with streamed_reads_retried(attempts=2, first_wait_seconds=0.0):
            assert stream_rows() == ROWS
        assert flaky_remote_files.raised == [fetches_of_a_clean_stream // 2]

    @pytest.mark.parametrize("status", [401, 403, 404, 416])
    def test_a_status_a_repeat_cannot_change_ends_the_stream_at_once(
        self,
        flaky_remote_files,  # noqa: F811
        stream_rows,
        fetches_of_a_clean_stream,
        status,
    ):
        failed_call = fetches_of_a_clean_stream // 2
        flaky_remote_files.fail({failed_call: hub_status(status)})
        with streamed_reads_retried(attempts=5, first_wait_seconds=0.0):
            with pytest.raises(HfHubHTTPError, match=f"{status} from the Hub"):
                stream_rows()
        assert flaky_remote_files.fetches == failed_call  # the fetch was not tried again

    def test_a_stream_outlasts_a_request_sent_on_the_client_the_library_closed(
        self,
        flaky_remote_files,  # noqa: F811
        stream_rows,
        fetches_of_a_clean_stream,
    ):
        """The Hub client's backoff turns a failed connection into httpx's closed-client error (see
        ``hub_read_failure_is_transient``); its next request builds a fresh client, so the read is repeated."""
        flaky_remote_files.fail({fetches_of_a_clean_stream // 2: request_on_a_closed_client()})
        with streamed_reads_retried(attempts=2, first_wait_seconds=0.0):
            assert stream_rows() == ROWS
        assert flaky_remote_files.raised == [fetches_of_a_clean_stream // 2]

    def test_a_read_that_keeps_failing_raises_its_last_error_after_the_attempts_it_was_given(
        self,
        flaky_remote_files,  # noqa: F811
        stream_rows,
        fetches_of_a_clean_stream,
    ):
        first = fetches_of_a_clean_stream // 2
        flaky_remote_files.fail({first + attempt: dropped_connection() for attempt in range(10)})
        with streamed_reads_retried(attempts=4, first_wait_seconds=0.0):
            with pytest.raises(httpx.RemoteProtocolError):
                stream_rows()
        assert flaky_remote_files.raised == [first, first + 1, first + 2, first + 3]

    def test_an_error_that_is_not_the_hubs_is_raised_as_it_is(
        self,
        flaky_remote_files,  # noqa: F811
        stream_rows,
        fetches_of_a_clean_stream,
    ):
        failed_call = fetches_of_a_clean_stream // 2
        flaky_remote_files.fail({failed_call: OSError("disk gone")})
        with streamed_reads_retried(attempts=5, first_wait_seconds=0.0):
            with pytest.raises(OSError, match="disk gone"):
                stream_rows()
        assert flaky_remote_files.fetches == failed_call

    def test_a_file_opened_through_a_wrapper_is_not_repeated_so_its_failure_ends_the_stream(
        self,
        flaky_remote_files,  # noqa: F811
        tmp_path,
    ):
        """A text stream opens its files in text mode, as a wrapper over the buffered file. When a fetch under it
        fails the wrapper has already consumed what it had buffered, so a repeated ``read`` would skip that data and
        the stream would end short with no error. Such a file is therefore left as it is, and the failure is raised."""
        lines = ROWS[:600]
        path = tmp_path / "corpus.txt"
        path.write_text("\n".join(lines) + "\n")

        def read() -> list[str]:
            files = {"train": [flaky_remote_files.url(path)]}
            # A small chunk, so the loader reads the file in many steps and a fetch can fail part-way through it.
            stream = load_dataset("text", data_files=files, split="train", streaming=True, chunksize=5000)
            return [row["text"] for row in stream]

        with streamed_reads_retried(attempts=3, first_wait_seconds=0.0):
            assert read() == lines
        fetches = flaky_remote_files.fetches
        assert fetches > 4
        for failed in range(1, fetches + 1, 3):
            flaky_remote_files.fail({failed: dropped_connection()})
            with streamed_reads_retried(attempts=3, first_wait_seconds=0.0):
                with pytest.raises(httpx.RemoteProtocolError):
                    read()
            assert flaky_remote_files.raised == [failed]

    def test_after_the_block_datasets_wraps_files_as_it_did_before(self):
        before = file_utils._add_retries_to_file_obj_read_method
        with streamed_reads_retried(attempts=2, first_wait_seconds=0.0):
            assert file_utils._add_retries_to_file_obj_read_method is not before
        assert file_utils._add_retries_to_file_obj_read_method is before
        with pytest.raises(RuntimeError, match="inside the block"):
            with streamed_reads_retried(attempts=2, first_wait_seconds=0.0):
                raise RuntimeError("inside the block")
        assert file_utils._add_retries_to_file_obj_read_method is before


class TestReadsRepeatedInPlace:
    def test_the_hubs_unbuffered_stream_file_is_left_alone_and_its_buffered_file_is_repeated(self):
        """The Hub's two file classes are both fsspec buffered files, and only one can be repeated: the unbuffered
        stream file, which ``datasets`` opens for an archive it iterates, has a ``read`` of its own that returns an
        end of file after a failed one, so a repeat would turn the failure into a short stream. Each is built
        without its ``__init__``, which resolves its path on the Hub."""
        from huggingface_hub.hf_file_system import HfFileSystemFile, HfFileSystemStreamFile

        stream_file = object.__new__(HfFileSystemStreamFile)
        assert reads_repeated_in_place(stream_file, 3, 0.0, "a read") is stream_file
        assert "read" not in vars(stream_file)
        buffered_file = object.__new__(HfFileSystemFile)
        assert reads_repeated_in_place(buffered_file, 3, 0.0, "a read") is buffered_file
        assert "read" in vars(buffered_file)

    def test_only_a_file_that_reads_by_range_has_its_read_repeated(self, flaky_remote_files, tmp_path):  # noqa: F811
        path = tmp_path / "blob.bin"
        path.write_bytes(bytes(range(256)) * 400)
        with fsspec.open(flaky_remote_files.url(path), "rb") as buffered:
            plain_read = buffered.read
            assert reads_repeated_in_place(buffered, 3, 0.0, "a read") is buffered
            assert buffered.read is not plain_read
            flaky_remote_files.fail({1: dropped_connection(), 3: hub_status(503)})
            assert buffered.read(40_000) + buffered.read() == path.read_bytes()
            assert flaky_remote_files.raised == [1, 3]
        with open(path, "rb") as local, io.TextIOWrapper(io.BytesIO(b"text")) as wrapper:
            for other in (local, wrapper):
                read = other.read
                assert reads_repeated_in_place(other, 3, 0.0, "a read") is other
                assert other.read == read


class TestRetrying:
    @pytest.fixture
    def waits(self, monkeypatch):
        """The waits ``retrying`` asks for, recorded in place of sleeping: a minute's wait is the unit under test,
        and really waiting it would make the test take minutes."""
        asked: list[float] = []
        monkeypatch.setattr(hub_read_retry.time, "sleep", asked.append)
        return asked

    @staticmethod
    def _failing(times: int, error_of):
        """A read that raises ``error_of()`` ``times`` times and then returns what it was called with."""
        calls = []

        def read(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) <= times:
                raise error_of()
            return args, kwargs

        return read, calls

    def test_the_wait_doubles_from_the_first_up_to_a_minute(self, waits):
        read, calls = self._failing(7, dropped_connection)
        repeated = retrying(read, attempts=8, first_wait_seconds=5.0, what="a read")
        assert repeated(1024, offset=3) == ((1024,), {"offset": 3})
        assert waits == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0]
        assert len(calls) == 8 and all(call == ((1024,), {"offset": 3}) for call in calls)

    def test_the_last_attempt_raises_without_waiting(self, waits):
        read, calls = self._failing(10, dropped_connection)
        with pytest.raises(httpx.RemoteProtocolError):
            retrying(read, attempts=3, first_wait_seconds=5.0, what="a read")()
        assert len(calls) == 3 and waits == [5.0, 10.0]

    def test_one_attempt_repeats_nothing(self, waits):
        read, calls = self._failing(1, dropped_connection)
        with pytest.raises(httpx.RemoteProtocolError):
            retrying(read, attempts=1, first_wait_seconds=5.0, what="a read")()
        assert len(calls) == 1 and waits == []

    def test_fewer_than_one_attempt_is_refused(self):
        with pytest.raises(ValueError, match="a read needs at least one attempt, not 0"):
            retrying(lambda: b"", attempts=0, first_wait_seconds=5.0, what="a read")

    def test_each_failed_attempt_is_logged_with_its_error(self, waits, caplog):
        read, _ = self._failing(2, lambda: hub_status(504))
        with caplog.at_level(logging.WARNING, logger=hub_read_retry.__name__):
            retrying(read, attempts=3, first_wait_seconds=5.0, what="a read of corpus/train.parquet")()
        messages = [record.getMessage() for record in caplog.records]
        assert messages == [
            "a read of corpus/train.parquet failed (HfHubHTTPError: 504 from the Hub); attempt 1 of 3, "
            "repeating it in 5 s",
            "a read of corpus/train.parquet failed (HfHubHTTPError: 504 from the Hub); attempt 2 of 3, "
            "repeating it in 10 s",
        ]


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (dropped_connection(), True),
        (httpx.ReadTimeout("timed out"), True),
        (httpx.ConnectError("refused"), True),
        (httpx.ReadError("reset by peer"), True),
        (hub_status(429), True),
        (hub_status(500), True),
        (hub_status(504), True),
        (hub_status(400), False),
        (hub_status(404), False),
        (hub_status(416), False),
        (request_on_a_closed_client(), True),
        (RuntimeError("parquet magic bytes not found"), False),
        (OSError("disk gone"), False),
        (ValueError("bad range"), False),
    ],
)
def test_which_failures_a_repeated_read_can_outlast(error, transient):
    assert hub_read_failure_is_transient(error) is transient
