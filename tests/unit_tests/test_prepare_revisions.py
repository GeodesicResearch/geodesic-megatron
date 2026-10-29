# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A prepare config's tokenizer pin, and the tokenizer reference the tokenize job takes.

`tokenizer-revision` pins a prepare config's tokenizer at a full commit SHA. The build plan hands the tokenize job the
pinned tokenizer as one word, `<name>@<sha>`, and the job resolves it inside the container: the name and the commit
for its provenance, and the commit's snapshot directory to load.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.data import prepare_revisions


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

    @pytest.mark.parametrize("directory", ["tok@v2", f"tok@{PIN}"])
    def test_a_local_directory_holding_the_separator_is_its_own_name(self, tmp_path, directory):
        name = str(tmp_path / directory)
        Path(name).mkdir()
        assert prepare_revisions.tokenizer_reference(name, None) == name
        assert prepare_revisions.split_tokenizer_reference(name) == (name, None)

    def test_a_path_holding_the_separator_before_no_commit_is_its_own_name(self):
        """No Hub id has two slashes, so the part after its last '@' names no commit of one."""
        assert prepare_revisions.split_tokenizer_reference("/data/tok@v2") == ("/data/tok@v2", None)

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
