# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/data/tokenize_arrow.py: the job around the tokeniser, which ends it when it cannot read and checks the totals.

The tests run the real ``tools/preprocess_data.py`` of Megatron-LM, which costs a Megatron import each: a total that
differs from the expected one, a tokenizer that does not load (it ends the tokeniser while the rows are being written),
and a missing column (it ends the feed). The one stand-in for ``preprocess_data.py`` is a short script that ends before
it reads its input, because the real tool opens its input before anything can fail and so does not end that way outside
a broken environment. The corpus tests are in ``test_tokenize_arrow.py``, a file of its own so that the two run beside
each other.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest
from scripts.data import tokenize_arrow
from scripts.data.tokenize_arrow import TokenizeArrowError

from tests.unit_tests.token_masking_fixtures import build_tiny_hf_tokenizer
from tests.unit_tests.tokenize_arrow_fixtures import SHARDS, TEXTS, totals_of, write_arrow_dataset, write_config


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory) -> Path:
    return build_tiny_hf_tokenizer(tmp_path_factory.mktemp("tokenizer"))


def tokenise_within(config, monkeypatch, seconds: float) -> dict:
    """Run ``tokenize_arrow.tokenize_arrow(config)`` in a thread and return ``{"error": ...}`` or ``{"result": ...}``.

    The tool waits on its tokeniser, so a tool that fails to end it would hang the whole suite, and pytest has no timeout
    here. When the thread is still running after ``seconds``, the tokeniser is killed and the test fails.
    """
    children = []
    start = subprocess.Popen

    def recording_popen(*args, **kwargs):
        children.append(start(*args, **kwargs))
        return children[-1]

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    outcome: dict = {}

    def target() -> None:
        try:
            outcome["result"] = tokenize_arrow.tokenize_arrow(config)
        except BaseException as error:  # reported to the test below
            outcome["error"] = error

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        for child in children:
            child.kill()
        thread.join()
        pytest.fail(f"the tool was still waiting {seconds} s after its tokeniser could no longer read")
    return outcome


class TestTheTotals:
    def test_a_total_that_differs_from_the_expected_fails_naming_each_difference_and_keeps_the_corpus(
        self, tmp_path, tokenizer
    ):
        arrow = write_arrow_dataset(tmp_path / "arrow", SHARDS)
        # The totals of these rows are known from the corpus test: six documents, and a token for each word and each end.
        path = write_config(tmp_path / "config.yaml", arrow, tokenizer, rows=len(TEXTS) + 2, tokens=1)
        config = tokenize_arrow.load_config(path)
        with pytest.raises(TokenizeArrowError) as caught:
            tokenize_arrow.tokenize_arrow(config)
        documents, tokens = totals_of(config.corpus_prefix)
        message = str(caught.value)
        assert f"{len(TEXTS)} rows were read from the shards, {len(TEXTS) + 2} expected" in message
        assert f"{documents} documents in the corpus, {len(TEXTS) + 2} expected" in message
        assert f"{tokens} tokens in the corpus, 1 expected" in message
        assert Path(f"{config.corpus_prefix}.provenance.json").is_file(), "what was made is still recorded"


class TestTheJobAroundTheTokeniser:
    """The pipe, with the real ``preprocess_data.py`` where it can fail as a test needs, and a stand-in for it where the
    real one cannot."""

    def job(self, tmp_path: Path, tokenizer: Path, shards: list[list[str]] = SHARDS, **fields):
        arrow = write_arrow_dataset(tmp_path / "arrow", shards)
        config = tokenize_arrow.load_config(write_config(tmp_path / "config.yaml", arrow, tokenizer, rows=1, tokens=1))
        return tokenize_arrow.TokenizeArrowConfig(**{**config.__dict__, **fields})

    def test_a_tokeniser_that_ends_before_it_reads_fails_at_once_instead_of_waiting_for_a_reader(
        self, tmp_path, tokenizer, monkeypatch
    ):
        # A stand-in, because the real tool opens its input before anything can fail, so it ends before reading only in
        # a broken environment (an interpreter that cannot import it).
        script = tmp_path / "standin.py"
        script.write_text("import sys\nsys.exit(7)\n")
        monkeypatch.setattr(tokenize_arrow, "PREPROCESS_DATA", script)
        config = self.job(tmp_path, tokenizer)
        with pytest.raises(TokenizeArrowError, match="ended with status 7 before it read"):
            tokenize_arrow.tokenize_arrow(config)
        assert not list(config.output_prefix.parent.glob("*.rows.jsonl")), "the pipe is removed"

    def test_a_tokenizer_that_does_not_load_ends_the_job_and_the_tokeniser_instead_of_hanging(
        self, tmp_path, monkeypatch
    ):
        """The real tool: its partition process fails on a tokenizer path that does not exist and the main process then
        waits for a result that never comes. Four rows of 1 MB do not fit in the pipe, so the tool is still writing
        when the reader goes."""
        config = self.job(tmp_path, tmp_path / "no_such_tokenizer", shards=[["word " * 200000] * 4])
        outcome = tokenise_within(config, monkeypatch, seconds=300)
        assert isinstance(outcome.get("error"), TokenizeArrowError), outcome
        assert "closed its input while the rows were being written" in str(outcome["error"])
        assert not list(config.output_prefix.parent.glob("*.rows.jsonl")), "the pipe is removed"

    def test_a_missing_text_column_ends_the_tokeniser_and_removes_the_pipe(self, tmp_path, tokenizer, monkeypatch):
        config = self.job(tmp_path, tokenizer, text_key="body")
        outcome = tokenise_within(config, monkeypatch, seconds=300)
        assert isinstance(outcome.get("error"), TokenizeArrowError), outcome
        assert "has no column 'body'" in str(outcome["error"])
        assert not list(config.output_prefix.parent.glob("*.rows.jsonl"))

    def test_a_pipe_from_an_earlier_run_is_refused(self, tmp_path, tokenizer):
        config = self.job(tmp_path, tokenizer)
        config.output_prefix.parent.mkdir(parents=True)
        (config.output_prefix.parent / "tokenized.rows.jsonl").write_text("")
        with pytest.raises(TokenizeArrowError, match="exists: a run that ended badly"):
            tokenize_arrow.tokenize_arrow(config)
