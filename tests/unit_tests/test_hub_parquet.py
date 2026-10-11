# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The shared Hub parquet helpers choose a config's train files by one rule and survive a cut range read.

* The file selection: a config's files are the ones its dataset card lists, in the card's order, whether the config
  is its own directory or a union of other configs' files; a local copy laid out as the Hub repository is yields the
  files the Hub of the same tree yields, in the same order, so a check reads the same rows whether it reads the Hub or
  a copy. The Hub's listing and files are stood in for by ``hub_fixtures.local_hub``, which lists the local tree the
  way ``list_repo_tree`` lists a directory and opens its files, because the real Hub needs the network.
* The read retry: a Hub range read cut mid-body hours into a read must be re-read, not fatal;
  anything that is not a transport failure must stay fatal; and a persistent transport failure
  must still be raised after the bounded attempts. The Hub's filesystem is stood in for, for the
  same reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit_tests.corpora_fixtures import declare_config, load_campaign_module, write_parquet_dataset
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
    def test_a_declared_config_without_train_files_is_an_error(self, tmp_path, config):
        """The card declares the config, but its directory is missing or holds no parquet file."""
        (tmp_path / "empty").mkdir()
        (tmp_path / "empty" / "_provenance.json").write_text("{}")
        declare_config(tmp_path, config, [f"{config}/train-*"])
        with pytest.raises(FileNotFoundError, match=f"'{config}/train-\\*' matches no parquet file"):
            hub_parquet.local_parquet_files(tmp_path, config)

    def test_a_copy_without_a_dataset_card_is_an_error(self, tmp_path):
        (tmp_path / "stem").mkdir()
        with pytest.raises(FileNotFoundError, match="no dataset card README.md, so config 'stem' is not declared"):
            hub_parquet.local_parquet_files(tmp_path, "stem")

    def test_a_union_config_reads_its_members_files_in_the_cards_order(self, tmp_path, monkeypatch):
        """A config joining other configs' files (a corpus published as row-contiguous parts) is read in the order
        its card lists them, not in path order, and the Hub resolves it as a local copy does."""
        write_parquet_dataset(tmp_path, "part1", {"n": [2, 3]})
        write_parquet_dataset(tmp_path, "part0", {"n": [0, 1]}, files=2)
        declare_config(tmp_path, "whole", ["part1/train-*", "part0/train-*"])
        local = [path.relative_to(tmp_path).as_posix() for path in hub_parquet.local_parquet_files(tmp_path, "whole")]
        assert local == [
            "part1/train-00000-of-00001.parquet",
            "part0/train-00000-of-00002.parquet",
            "part0/train-00001-of-00002.parquet",
        ]
        local_hub(monkeypatch, "org/data", {"rev": tmp_path})
        assert hub_parquet.hub_parquet_files("org/data", "rev", "whole") == local

    def test_explicit_paths_are_read_as_listed(self, tmp_path):
        write_parquet_dataset(tmp_path, "part0", {"n": [0, 1]}, files=2)
        declare_config(tmp_path, "whole", ["part0/train-00001-of-00002.parquet", "part0/train-00000-of-00002.parquet"])
        files = hub_parquet.local_parquet_files(tmp_path, "whole")
        assert [path.name for path in files] == ["train-00001-of-00002.parquet", "train-00000-of-00002.parquet"]

    @pytest.mark.parametrize(
        ("pattern", "error", "message"),
        [
            ("**/train-*", ValueError, "glob syntax"),
            ("part0/train-[01]*", ValueError, "glob syntax"),
            ("part0/validation-*", FileNotFoundError, "matches no parquet file"),
            ("part9/train-00000-of-00001.parquet", FileNotFoundError, "do not exist"),
        ],
    )
    def test_a_pattern_the_tools_cannot_resolve_is_an_error(self, tmp_path, pattern, error, message):
        write_parquet_dataset(tmp_path, "part0", {"n": [0, 1]})
        declare_config(tmp_path, "whole", [pattern])
        with pytest.raises(error, match=message):
            hub_parquet.local_parquet_files(tmp_path, "whole")

    def test_a_config_the_card_does_not_declare_is_an_error(self, tmp_path):
        write_parquet_dataset(tmp_path, "part0", {"n": [0, 1]})
        with pytest.raises(FileNotFoundError, match="declares config 'other' 0 times"):
            hub_parquet.local_parquet_files(tmp_path, "other")

    def test_the_url_names_the_file_at_the_revision(self):
        assert hub_parquet.hub_file_url("org/data", "abc", "stem/x.parquet") == "datasets/org/data@abc/stem/x.parquet"


def card(header: dict | None) -> str:
    """A dataset card whose YAML header is ``header``, followed by prose, or prose alone for ``None``."""
    import yaml

    prose = "# A dataset\n\nSome prose.\n"
    return prose if header is None else "---\n" + yaml.safe_dump(header, sort_keys=False) + "---\n" + prose


def declared(data_files) -> dict:
    """A card header declaring config ``c`` with ``data_files``, beside another config."""
    return {"configs": [{"config_name": "other", "data_files": "x/*"}, {"config_name": "c", "data_files": data_files}]}


class TestConfigTrainPatterns:
    """Every shape the Hub's dataset cards give ``data_files``, and each card the tools refuse."""

    @pytest.mark.parametrize(
        ("data_files", "patterns"),
        [
            ("c/train-*", ["c/train-*"]),
            (["a/train-*", "b/train-*"], ["a/train-*", "b/train-*"]),
            ([{"split": "train", "path": "c/train-*"}], ["c/train-*"]),
            (
                [{"split": "validation", "path": "c/val-*"}, {"split": "train", "path": ["b/train-*", "a/train-*"]}],
                ["b/train-*", "a/train-*"],
            ),
        ],
        ids=["one-pattern", "list-of-patterns", "train-split-one-path", "train-split-path-list"],
    )
    def test_each_declared_shape_yields_its_train_patterns_in_order(self, data_files, patterns):
        assert hub_parquet.config_train_patterns(card(declared(data_files)), "c", "repo") == patterns

    @pytest.mark.parametrize(
        ("text", "error", "message"),
        [
            (card(None), FileNotFoundError, "repo: the dataset card has no YAML header, so config 'c'"),
            (card({"configs": []}), FileNotFoundError, "declares config 'c' 0 times"),
            (
                card(
                    {"configs": [{"config_name": "c", "data_files": "a/*"}, {"config_name": "c", "data_files": "b/*"}]}
                ),
                FileNotFoundError,
                "declares config 'c' 2 times",
            ),
            (card(declared([{"split": "validation", "path": "c/v-*"}])), FileNotFoundError, "0 train splits"),
            (
                card(declared([{"split": "train", "path": "a/*"}, {"split": "train", "path": "b/*"}])),
                FileNotFoundError,
                "2 train splits",
            ),
            (card(declared({"train": "c/train-*"})), ValueError, "data_files of a shape"),
            (card(declared([])), ValueError, "data_files of a shape"),
            (card(declared(["a/*", {"split": "train", "path": "b/*"}])), ValueError, "data_files of a shape"),
            (card(declared([{"split": "train", "path": 7}])), ValueError, "data_files of a shape"),
            (card(declared(None)), ValueError, "data_files of a shape"),
        ],
        ids=[
            "no-header",
            "undeclared",
            "declared-twice",
            "no-train-split",
            "two-train-splits",
            "mapping",
            "empty-list",
            "mixed-list",
            "non-string-path",
            "missing",
        ],
    )
    def test_a_card_the_tools_cannot_read_is_refused(self, text, error, message):
        with pytest.raises(error, match=message):
            hub_parquet.config_train_patterns(text, "c", "repo")

    def test_a_single_pattern_is_declared_as_a_string_as_push_to_hub_writes_it(self, tmp_path):
        """The fixture writes the shape the real card has, so the string branch is the one the other tests take."""
        import yaml

        write_parquet_dataset(tmp_path, "stem", {"n": [0]})
        header = yaml.safe_load((tmp_path / "README.md").read_text().split("---")[1])
        assert header["configs"] == [
            {"config_name": "stem", "data_files": [{"split": "train", "path": "stem/train-*"}]}
        ]


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
