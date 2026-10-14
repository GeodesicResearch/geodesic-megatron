# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A manifest of files' sha256 digests and sizes is read strictly, and a set of files is checked against it.

Every manifest and every file here is real, written in `tmp_path`. The files are "fetched" from a local directory,
which is what the caller-supplied fetch does once a download has put them in a cache.
"""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

import pytest
from scripts.data.sha256_manifest import (
    ManifestMismatch,
    check_files_against_manifest,
    file_sha256,
    parse_manifest,
)


DIGEST = "a" * 64


def _write(path: Path, text: str) -> Path:
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.fixture
def repo(tmp_path):
    """Three files laid out as a repository's, and a manifest of them as sha256sum lists files."""
    root = tmp_path / "repo"
    files = {"cfg/train-00000-of-00002.parquet": b"first file", "cfg/train-00001-of-00002.parquet": b"second"}
    files["other/train-00000-of-00001.parquet"] = b""
    for path, content in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(content)
    lines = [f"{hashlib.sha256(content).hexdigest()}  {len(content)}  {path}\n" for path, content in files.items()]
    manifest = _write(tmp_path / "manifest.sha256", "".join(lines))
    return root, manifest, list(files)


class TestParseManifest:
    def test_a_manifest_lists_each_files_digest_and_size_in_order(self, repo):
        root, manifest_path, paths = repo
        manifest = parse_manifest(manifest_path)
        assert list(manifest.entries) == paths
        first = manifest.entries[paths[0]]
        assert (first.sha256, first.size) == (hashlib.sha256(b"first file").hexdigest(), 10)
        assert manifest.sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert manifest.record() == {
            "path": str(manifest_path),
            "sha256": manifest.sha256,
            "files": 3,
            "total_bytes": 16,
        }

    def test_the_last_lines_newline_may_be_omitted(self, tmp_path):
        manifest = parse_manifest(_write(tmp_path / "m", f"{DIGEST}  5  a/b.parquet"))
        assert manifest.entries["a/b.parquet"].size == 5

    def test_a_missing_manifest_is_refused_naming_it(self, tmp_path):
        path = tmp_path / "absent.sha256"
        with pytest.raises(ValueError, match=re.escape(f"manifest {path} cannot be read")):
            parse_manifest(path)

    def test_an_empty_manifest_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="lists no files"):
            parse_manifest(_write(tmp_path / "m", ""))

    @pytest.mark.parametrize(
        "line",
        [
            f"{DIGEST} 5 a.parquet",  # one space
            f"{DIGEST}\t5\ta.parquet",
            f"{DIGEST}   5  a.parquet",  # three spaces
            f"{DIGEST}  5  *a.parquet".replace("  *", " *"),  # sha256sum's binary marker
            f"{DIGEST.upper()}  5  a.parquet",
            f"{DIGEST[:63]}  5  a.parquet",
            f"{DIGEST}  05  a.parquet",
            f"{DIGEST}  +5  a.parquet",
            f"{DIGEST}  -5  a.parquet",
            f"{DIGEST}  5  ",
            f"{DIGEST}  5",
        ],
    )
    def test_a_malformed_line_is_refused_with_its_number(self, tmp_path, line):
        text = f"{DIGEST}  1  ok.parquet\n{line}\n"
        with pytest.raises(ValueError, match=r":2: expected '<64 lowercase hex sha256>  <size>  <path>'"):
            parse_manifest(_write(tmp_path / "m", text))

    def test_a_blank_line_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match=":2: expected"):
            parse_manifest(_write(tmp_path / "m", f"{DIGEST}  1  a.parquet\n\n{DIGEST}  1  b.parquet\n"))

    @pytest.mark.parametrize(
        ("path", "problem"),
        [
            ("/abs/a.parquet", "is absolute"),
            (" a.parquet", "starts or ends with whitespace"),
            ("a.parquet ", "starts or ends with whitespace"),
            ("a//b.parquet", "empty, '.' or '..' component"),
            ("./a.parquet", "empty, '.' or '..' component"),
            ("a/../b.parquet", "empty, '.' or '..' component"),
            ("a/", "empty, '.' or '..' component"),
            ("a\tb.parquet", "control character"),
            ("a.parquet\r", "starts or ends with whitespace"),  # a CRLF line ending
        ],
    )
    def test_a_path_that_is_not_a_repository_path_is_refused(self, tmp_path, path, problem):
        with pytest.raises(ValueError, match=re.escape(problem)):
            parse_manifest(_write(tmp_path / "m", f"{DIGEST}  1  {path}\n"))

    def test_a_path_listed_twice_is_refused(self, tmp_path):
        text = f"{DIGEST}  1  a.parquet\n{'b' * 64}  2  a.parquet\n"
        with pytest.raises(ValueError, match=r":2: path 'a.parquet' is listed twice"):
            parse_manifest(_write(tmp_path / "m", text))


class TestCheckFilesAgainstManifest:
    def test_the_files_the_manifest_lists_pass(self, repo):
        root, manifest_path, paths = repo
        checked = []
        check_files_against_manifest(
            parse_manifest(manifest_path),
            reversed(paths),  # the order the files are read in is not the manifest's business
            lambda path: root / path,
            workers=2,
            on_checked=lambda entry, done, total: checked.append((entry.path, total)),
        )
        assert sorted(checked) == sorted((path, 3) for path in paths)

    @pytest.mark.parametrize(
        ("files", "message"),
        [
            (lambda paths: paths[:2], r"1 it lists are not read \['other/train-00000-of-00001.parquet'\]"),
            (lambda paths: [*paths, "cfg/extra.parquet"], r"1 read are not listed \['cfg/extra.parquet'\]"),
            (lambda paths: [*paths, paths[0]], r"read more than once \['cfg/train-00000-of-00002.parquet'\]"),
        ],
    )
    def test_a_set_of_files_other_than_the_manifests_is_refused_before_anything_is_fetched(self, repo, files, message):
        root, manifest_path, paths = repo
        fetched = []

        def fetch(path):
            fetched.append(path)
            return root / path

        with pytest.raises(ManifestMismatch, match=message):
            check_files_against_manifest(parse_manifest(manifest_path), files(paths), fetch, workers=2)
        assert fetched == []

    def test_every_file_whose_size_or_digest_differs_is_reported_together(self, repo):
        root, manifest_path, paths = repo
        (root / paths[0]).write_bytes(b"first filE")  # same size, other bytes
        (root / paths[1]).write_bytes(b"second!")  # another size
        with pytest.raises(ManifestMismatch) as refusal:
            check_files_against_manifest(parse_manifest(manifest_path), paths, lambda path: root / path, workers=3)
        message = str(refusal.value)
        assert message.startswith(f"2 of the 3 files manifest {manifest_path} lists differ from it")
        assert f"{paths[0]}: sha256 {file_sha256(root / paths[0])}, the manifest lists" in message
        assert f"{paths[1]}: 7 bytes, the manifest lists 6" in message

    def test_an_error_a_fetch_raises_is_raised_and_no_further_file_is_fetched(self, repo):
        root, manifest_path, paths = repo
        fetched = []
        lock = threading.Lock()

        def fetch(path):
            with lock:
                fetched.append(path)
            if path == paths[0]:
                raise ConnectionError("the Hub went away")
            return root / path

        with pytest.raises(ConnectionError, match="the Hub went away"):
            check_files_against_manifest(parse_manifest(manifest_path), paths, fetch, workers=1)
        assert fetched == [paths[0]]

    def test_fewer_than_one_worker_is_refused(self, repo):
        root, manifest_path, paths = repo
        with pytest.raises(ValueError, match="at least one worker"):
            check_files_against_manifest(parse_manifest(manifest_path), paths, lambda path: root / path, workers=0)
