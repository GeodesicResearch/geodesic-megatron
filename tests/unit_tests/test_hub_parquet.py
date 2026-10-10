# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The shared Hub parquet helpers choose a config's train files by one rule and survive a cut range read.

* The file selection: a local copy laid out as the Hub repository is yields the files the Hub
  listing of the same tree yields, in the same order, so a check reads the same rows whether it
  reads the Hub or a copy. The Hub's listing is stood in for by ``hub_fixtures.local_hub``, which
  lists the local tree the way ``list_repo_tree`` lists a directory, because the real listing needs
  the network.
* The read retry: a Hub range read cut mid-body hours into a read must be re-read, not fatal;
  anything that is not a transport failure must stay fatal; and a persistent transport failure
  must still be raised after the bounded attempts. The Hub's filesystem is stood in for, for the
  same reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit_tests.corpora_fixtures import load_campaign_module, write_parquet_dataset
from tests.unit_tests.hub_fixtures import local_hub


hub_parquet = load_campaign_module("hub_parquet")


def config_tree(repo: Path) -> None:
    """One config holding two train files, a validation file, the builder's record, and a nested
    directory whose parquet a non-recursive listing does not reach."""
    write_parquet_dataset(repo, "stem", {"n": [0, 1, 2, 3]}, files=2)
    (repo / "stem" / "validation-00000-of-00001.parquet").write_bytes(b"")
    (repo / "stem" / "_provenance.json").write_text("{}")
    (repo / "stem" / "nested").mkdir()
    (repo / "stem" / "nested" / "train-00000-of-00001.parquet").write_bytes(b"")


class TestTrainParquetFiles:
    def test_a_local_copy_selects_the_train_files_directly_under_the_config_in_order(self, tmp_path):
        config_tree(tmp_path)
        files = hub_parquet.local_parquet_files(tmp_path, "stem")
        assert [path.relative_to(tmp_path).as_posix() for path in files] == [
            "stem/train-00000-of-00002.parquet",
            "stem/train-00001-of-00002.parquet",
        ]

    def test_the_hub_listing_of_the_same_tree_selects_the_same_files(self, tmp_path, monkeypatch):
        config_tree(tmp_path)
        local_hub(monkeypatch, "org/data", {"rev": tmp_path})
        local = [path.relative_to(tmp_path).as_posix() for path in hub_parquet.local_parquet_files(tmp_path, "stem")]
        assert hub_parquet.hub_parquet_files("org/data", "rev", "stem") == local

    @pytest.mark.parametrize("config", ["absent", "empty"])
    def test_a_config_without_train_files_is_an_error(self, tmp_path, config):
        (tmp_path / "empty").mkdir()
        (tmp_path / "empty" / "_provenance.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match=repr(config)):
            hub_parquet.local_parquet_files(tmp_path, config)

    def test_the_url_names_the_file_at_the_revision(self):
        assert hub_parquet.hub_file_url("org/data", "abc", "stem/x.parquet") == "datasets/org/data@abc/stem/x.parquet"


class TestHubReadRetry:
    class _Handle:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FileSystem:
        """Stands in for HfFileSystem: opening is the retried unit, and this counts the opens."""

        def __init__(self):
            self.opens = 0

        def open(self, url, mode):
            self.opens += 1
            return TestHubReadRetry._Handle()

    @staticmethod
    def _failing_then(result, failures: list[Exception]):
        calls = iter(failures)

        def work(fh):
            error = next(calls, None)
            if error is not None:
                raise error
            return result

        return work

    def test_a_truncated_body_is_re_read_from_a_fresh_open(self, monkeypatch):
        import httpx

        monkeypatch.setattr(hub_parquet.time, "sleep", lambda s: None)
        fs = self._FileSystem()
        work = self._failing_then("rows", [httpx.RemoteProtocolError("peer closed connection")])
        assert hub_parquet.read_hub_file(fs, "datasets/x@rev/a.parquet", work) == "rows"
        assert fs.opens == 2

    def test_an_error_that_is_not_a_transport_failure_is_raised_at_once(self):
        fs = self._FileSystem()
        work = self._failing_then("rows", [ValueError("bad parquet")])
        with pytest.raises(ValueError, match="bad parquet"):
            hub_parquet.read_hub_file(fs, "datasets/x@rev/a.parquet", work)
        assert fs.opens == 1

    def test_a_persistent_transport_failure_is_raised_after_the_bounded_attempts(self, monkeypatch):
        import httpx

        monkeypatch.setattr(hub_parquet.time, "sleep", lambda s: None)
        fs = self._FileSystem()
        work = self._failing_then("rows", [httpx.ReadTimeout("timed out")] * 3)
        with pytest.raises(httpx.ReadTimeout):
            hub_parquet.read_hub_file(fs, "datasets/x@rev/a.parquet", work, attempts=3)
        assert fs.opens == 3

    def test_a_hub_http_error_is_retried_only_for_rate_limits_and_server_errors(self, monkeypatch):
        import httpx
        from huggingface_hub.errors import HfHubHTTPError

        monkeypatch.setattr(hub_parquet.time, "sleep", lambda s: None)

        def hub_error(status: int) -> HfHubHTTPError:
            response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/x"))
            return HfHubHTTPError(f"{status}", response=response)

        fs = self._FileSystem()
        assert (
            hub_parquet.read_hub_file(fs, "u", self._failing_then("rows", [hub_error(429), hub_error(503)])) == "rows"
        )
        assert fs.opens == 3
        fs = self._FileSystem()
        with pytest.raises(HfHubHTTPError):
            hub_parquet.read_hub_file(fs, "u", self._failing_then("rows", [hub_error(404)]))
        assert fs.opens == 1

    def test_a_request_on_the_client_the_library_closed_is_re_read_from_a_fresh_open(self, monkeypatch):
        """huggingface_hub's backoff closes its shared httpx client when a request cannot connect,
        then retries on the reference it took before its loop, so what a connection failure
        surfaces is httpx's closed-client error, not a transport error. A fresh open builds a new
        client, so the read must be re-opened like a cut body. A real closed client raises the
        error, so the test follows httpx's own message; it raises before anything is sent."""
        import httpx

        monkeypatch.setattr(hub_parquet.time, "sleep", lambda s: None)
        closed = httpx.Client()
        closed.close()
        fs = self._FileSystem()
        calls: list[object] = []

        def work(fh):
            calls.append(fh)
            if len(calls) == 1:
                closed.request("GET", "https://huggingface.co/never-sent")
            return "rows"

        assert hub_parquet.read_hub_file(fs, "datasets/x@rev/a.parquet", work) == "rows"
        assert (fs.opens, len(calls)) == (2, 2)

    def test_a_runtime_error_that_is_not_the_closed_client_is_raised_at_once(self):
        fs = self._FileSystem()
        work = self._failing_then("rows", [RuntimeError("parquet magic bytes not found")])
        with pytest.raises(RuntimeError, match="magic bytes"):
            hub_parquet.read_hub_file(fs, "datasets/x@rev/a.parquet", work)
        assert fs.opens == 1

    def test_the_waits_between_opens_double_to_a_cap_and_span_minutes(self, monkeypatch):
        """An egress outage takes every route to the Hub away for minutes at a time, and every
        request made during it fails at once, so a retry budget of seconds is spent before the
        route is back. The waits double from 2 s to a cap, and the bounded attempts must span at
        least five minutes of persistent failure before the last one is raised."""
        import httpx

        waits: list[float] = []
        monkeypatch.setattr(hub_parquet.time, "sleep", waits.append)
        fs = self._FileSystem()
        work = self._failing_then("rows", [httpx.ConnectError("[Errno 101] Network is unreachable")] * 100)
        with pytest.raises(httpx.ConnectError):
            hub_parquet.read_hub_file(fs, "datasets/x@rev/a.parquet", work)
        assert fs.opens == hub_parquet.HUB_READ_ATTEMPTS
        assert sum(waits) >= 300
        assert waits == [min(2.0**k, hub_parquet.HUB_READ_MAX_WAIT_S) for k in range(1, hub_parquet.HUB_READ_ATTEMPTS)]
