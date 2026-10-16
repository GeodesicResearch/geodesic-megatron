# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A prepare config's tokenizer pin and the tokenizer reference the tokenize job takes, and its dataset pin by
revision file.

A `revision-file` names a file that holds the dataset's commit, read when a step needs the commit; the strict reader
and the refusal of a pin stated two ways are tested on real files in `tmp_path`.

`tokenizer-revision` pins a prepare config's tokenizer at a full commit SHA. The build plan hands the tokenize job the
pinned tokenizer as one word, `<name>@<sha>`, and the job resolves it inside the container: the name and the commit
for its provenance, and the commit's snapshot directory to load. These tests run the real functions and, for the
snapshot, the real command line against a Hugging Face cache laid out in `tmp_path` with the Hub switched off
(`HF_HUB_OFFLINE`), so the resolution is real and needs no network.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.data import prepare_revisions


_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = _REPO_ROOT / "scripts" / "data" / "prepare_revisions.py"
NAME = "org/tokenizer"
PIN = "4" * 40


class TestTokenizerRevision:
    def test_a_config_without_the_key_pins_nothing(self):
        assert prepare_revisions.tokenizer_revision({"tokenizer": NAME}, "corpus.yaml") is None

    def test_a_full_sha_is_the_pin(self):
        assert prepare_revisions.tokenizer_revision({"tokenizer-revision": PIN}, "corpus.yaml") == PIN

    @pytest.mark.parametrize("pin", ["main", "4" * 12, "4" * 41, "G" * 40, None, 474397005])
    def test_anything_else_is_refused(self, pin):
        with pytest.raises(ValueError, match="corpus.yaml: `tokenizer-revision` must be a full 40-character"):
            prepare_revisions.tokenizer_revision({"tokenizer-revision": pin}, "corpus.yaml")


class TestTokenizerReference:
    def test_a_pinned_reference_names_the_commit(self):
        assert prepare_revisions.tokenizer_reference(NAME, PIN) == f"{NAME}@{PIN}"

    def test_an_unpinned_reference_is_the_name(self):
        assert prepare_revisions.tokenizer_reference(NAME, None) == NAME

    @pytest.mark.parametrize("revision", [PIN, None])
    def test_splitting_inverts_it(self, revision):
        reference = prepare_revisions.tokenizer_reference(NAME, revision)
        assert prepare_revisions.split_tokenizer_reference(reference) == (NAME, revision)

    def test_a_name_holding_the_separator_is_refused(self):
        with pytest.raises(ValueError, match="contains '@'"):
            prepare_revisions.tokenizer_reference("org/tok@enizer", PIN)

    @pytest.mark.parametrize(
        ("reference", "message"),
        [
            (f"{NAME}@main", "must be a full 40-character commit SHA"),
            (f"{NAME}@{'4' * 12}", "must be a full 40-character commit SHA"),
            (f"{NAME}@", "must be a full 40-character commit SHA"),
            (f"@{PIN}", "names no tokenizer"),
        ],
    )
    def test_a_malformed_reference_is_refused(self, reference, message):
        with pytest.raises(ValueError, match=message):
            prepare_revisions.split_tokenizer_reference(reference)

    def test_an_unpinned_reference_loads_by_name_without_the_hub(self):
        assert prepare_revisions.tokenizer_load_path(NAME) == NAME


def _field(field: str, reference: str, cache: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(cache)}
    return subprocess.run(
        [sys.executable, str(SCRIPT), field, reference], capture_output=True, text=True, env=env, timeout=60
    )


@pytest.fixture
def cached_snapshot(tmp_path):
    """A Hugging Face cache holding NAME's snapshot at PIN, laid out as `snapshot_download` writes it."""
    cache = tmp_path / "hub"
    snapshot = cache / f"models--{NAME.replace('/', '--')}" / "snapshots" / PIN
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}")
    return cache, snapshot


class TestCommandLine:
    """What the tokenize job in pipeline_data_submit.sbatch runs inside the container."""

    def test_name_and_revision_of_a_pinned_reference(self, cached_snapshot):
        cache, _ = cached_snapshot
        assert _field("name", f"{NAME}@{PIN}", cache).stdout == f"{NAME}\n"
        assert _field("revision", f"{NAME}@{PIN}", cache).stdout == f"{PIN}\n"

    def test_an_unpinned_reference_has_an_empty_revision_and_loads_by_name(self, cached_snapshot):
        cache, _ = cached_snapshot
        assert _field("revision", NAME, cache).stdout == "\n"
        assert _field("path", NAME, cache).stdout == f"{NAME}\n"

    def test_a_pinned_reference_loads_from_the_commits_snapshot(self, cached_snapshot):
        cache, snapshot = cached_snapshot
        result = _field("path", f"{NAME}@{PIN}", cache)
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{snapshot}\n"

    def test_a_commit_the_cache_does_not_hold_fails_rather_than_loading_another(self, cached_snapshot):
        cache, _ = cached_snapshot
        result = _field("path", f"{NAME}@{'5' * 40}", cache)
        assert result.returncode != 0
        assert result.stdout == ""

    def test_a_malformed_reference_fails(self, cached_snapshot):
        cache, _ = cached_snapshot
        result = _field("name", f"{NAME}@main", cache)
        assert result.returncode != 0
        assert "must be a full 40-character commit SHA" in result.stderr


SHA = "0123456789abcdef0123456789abcdef01234567"


class TestReadRevisionFile:
    """A revision file holds exactly one full lowercase commit SHA and at most one trailing newline."""

    @pytest.mark.parametrize("content", [SHA + "\n", SHA])
    def test_one_sha_with_or_without_its_newline_is_the_commit(self, tmp_path, content):
        path = tmp_path / "revision.txt"
        path.write_text(content)
        assert prepare_revisions.read_revision_file(path) == SHA

    def test_a_missing_file_is_refused_naming_it(self, tmp_path):
        path = tmp_path / "absent.txt"
        with pytest.raises(ValueError, match=re.escape(f"revision file {path} cannot be read")):
            prepare_revisions.read_revision_file(path)

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "\n",
            f"{SHA}\n{SHA}\n",  # two lines
            f"{SHA}\n\n",  # a second newline
            SHA[:12] + "\n",  # a short SHA
            SHA.upper() + "\n",
            f"{SHA}  \n",  # trailing spaces
            f" {SHA}\n",
            f"{SHA}\r\n",
            f"{SHA}0\n",  # 41 characters
            "main\n",
        ],
    )
    def test_anything_else_is_refused_naming_the_file(self, tmp_path, content):
        path = tmp_path / "revision.txt"
        path.write_text(content)
        with pytest.raises(ValueError, match=re.escape(f"revision file {path} must hold exactly one full 40-char")):
            prepare_revisions.read_revision_file(path)


class TestRevisionFilePin:
    """`revision-file` is the third way a prepare config pins its dataset's commit, exclusive of the other two."""

    def _pin(self, tmp_path, content=SHA + "\n"):
        path = tmp_path / "revision.txt"
        path.write_text(content)
        return str(path)

    def test_the_commit_is_read_from_the_file_for_any_subset(self, tmp_path):
        config = {"revision-file": self._pin(tmp_path)}
        assert prepare_revisions.subset_revision(config, "any_subset", "corpus.yaml") == SHA
        assert prepare_revisions.subset_revision(config, None, "corpus.yaml") == SHA

    def test_naming_the_file_does_not_read_it(self, tmp_path):
        absent = str(tmp_path / "written-later.txt")
        manifest = str(tmp_path / "manifest.sha256")
        config = {"revision-file": absent, "revision-manifest": manifest}
        assert prepare_revisions.revision_file(config, "corpus.yaml") == absent
        assert prepare_revisions.revision_manifest(config, "corpus.yaml") == manifest

    def test_a_config_pinned_otherwise_names_no_file(self):
        assert prepare_revisions.revision_file({"revision": SHA}, "corpus.yaml") is None
        assert prepare_revisions.revision_manifest({"revision": SHA}, "corpus.yaml") is None

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            ({"revision": SHA}, r"states `revision-file` and \['revision'\]"),
            ({"revisions": {"a": SHA}}, r"states `revision-file` and \['revisions'\]"),
        ],
    )
    def test_a_second_pin_beside_the_file_is_refused(self, tmp_path, extra, message):
        config = {"revision-file": self._pin(tmp_path), **extra}
        for check in (prepare_revisions.revision_file, prepare_revisions.revision_manifest):
            with pytest.raises(ValueError, match=f"corpus.yaml: {message}"):
                check(config, "corpus.yaml")
        with pytest.raises(ValueError, match=message):
            prepare_revisions.subset_revision(config, "a", "corpus.yaml")

    @pytest.mark.parametrize("pin", [{}, {"revision": SHA}, {"revisions": {"a": SHA}}])
    def test_a_manifest_without_a_revision_file_is_refused(self, tmp_path, pin):
        config = {**pin, "revision-manifest": str(tmp_path / "manifest.sha256")}
        with pytest.raises(ValueError, match="states `revision-manifest` without `revision-file`"):
            prepare_revisions.subset_revision(config, "a", "corpus.yaml")

    @pytest.mark.parametrize("key", ["revision-file", "revision-manifest"])
    @pytest.mark.parametrize("value", ["relative/revision.txt", "", None, 7])
    def test_a_path_that_is_not_absolute_is_refused(self, tmp_path, key, value):
        config = {"revision-file": self._pin(tmp_path), "revision-manifest": str(tmp_path / "m"), key: value}
        with pytest.raises(ValueError, match=f"corpus.yaml: `{key}` must be an absolute path"):
            prepare_revisions.revision_file(config, "corpus.yaml")

    @pytest.mark.parametrize("key", ["revision_file", "revision_manifest"])
    def test_an_underscored_key_is_refused(self, tmp_path, key):
        with pytest.raises(ValueError, match=f"states \\['{key}'\\]; the keys are spelled"):
            prepare_revisions.revision_file({key: self._pin(tmp_path)}, "corpus.yaml")
