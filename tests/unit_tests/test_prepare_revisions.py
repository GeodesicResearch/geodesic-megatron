# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A prepare config's tokenizer pin, and the tokenizer reference the tokenize job takes.

`tokenizer-revision` pins a prepare config's tokenizer at a full commit SHA. The build plan hands the tokenize job the
pinned tokenizer as one word, `<name>@<sha>`, and the job resolves it inside the container: the name and the commit
for its provenance, and the commit's snapshot directory to load. These tests run the real functions and, for the
snapshot, the real command line against a Hugging Face cache laid out in `tmp_path` with the Hub switched off
(`HF_HUB_OFFLINE`), so the resolution is real and needs no network.
"""

from __future__ import annotations

import os
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
