# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/data/tokenize_arrow.py: a saved Arrow dataset becomes the corpus that the standard tokenise tool makes.

The corpus tests are real: Arrow folders written with pyarrow in ``tmp_path``, a real tiny Hugging Face tokenizer, and
Megatron-LM's own ``tools/preprocess_data.py`` run as a subprocess, by the tool under test (through a named pipe) and,
for the reference, on a JSONL file of the same rows. Each real run costs a Megatron import, so two module-scoped runs
serve the corpus tests. The failures of the pipe and the totals check are in ``test_tokenize_arrow_job.py``, a file of
its own so that the two files run beside each other.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from scripts.data import tokenize_arrow
from scripts.data.preprocess_flags import ENCODING_NOTES
from scripts.data.tokenize_arrow import TokenizeArrowError

from tests.unit_tests.token_masking_fixtures import build_tiny_hf_tokenizer
from tests.unit_tests.tokenize_arrow_fixtures import SHARDS, TEXTS, totals_of, write_arrow_dataset, write_config


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory) -> Path:
    return build_tiny_hf_tokenizer(tmp_path_factory.mktemp("tokenizer"))


@pytest.fixture(scope="module")
def reference(tmp_path_factory, tokenizer) -> Path:
    """The corpus that ``preprocess_data.py`` makes from a JSONL file of ``TEXTS``, as the standard tokenise mode runs
    it (one partition, ``--append-eod``; the tool's ``preprocess_command`` holds those flags once, and a test of the
    sbatch checks that it passes them). The prefix, without extension."""
    root = tmp_path_factory.mktemp("reference")
    jsonl = root / "reference.jsonl"
    jsonl.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in TEXTS), encoding="utf-8")
    command = tokenize_arrow.preprocess_command(jsonl, root / "reference", str(tokenizer), "text", 2)
    subprocess.run(command, check=True, capture_output=True)
    return root / "reference_text_document"


@pytest.fixture(scope="module")
def run(tmp_path_factory, tokenizer, reference):
    """One real run of the command line over three shards."""
    root = tmp_path_factory.mktemp("run")
    arrow = write_arrow_dataset(root / "arrow", SHARDS)
    documents, tokens = totals_of(reference)
    config_path = write_config(root / "config.yaml", arrow, tokenizer, rows=documents, tokens=tokens)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        status = tokenize_arrow.main(["--config", str(config_path)])
    return {
        "status": status,
        "printed": json.loads(printed.getvalue()),
        "config": tokenize_arrow.load_config(config_path),
        "totals": (documents, tokens),
    }


class TestTheCorpus:
    def test_it_is_byte_for_byte_the_corpus_of_the_standard_tool_over_the_same_rows(self, run, reference):
        corpus = run["config"].corpus_prefix
        assert corpus.with_suffix(".bin").read_bytes() == reference.with_suffix(".bin").read_bytes()
        assert corpus.with_suffix(".idx").read_bytes() == reference.with_suffix(".idx").read_bytes()

    def test_the_command_line_prints_the_totals_and_returns_zero(self, run):
        documents, tokens = run["totals"]
        assert run["status"] == 0
        assert (run["printed"]["num_documents"], run["printed"]["total_tokens"]) == (documents, tokens)
        assert run["printed"]["rows_read"] == len(TEXTS)

    def test_it_writes_its_provenance_and_a_config_that_loads_again_and_leaves_no_pipe(self, run, tokenizer):
        config = run["config"]
        documents, tokens = run["totals"]
        provenance = json.loads(Path(f"{config.corpus_prefix}.provenance.json").read_text())
        assert provenance["totals"]["total_tokens"] == tokens and provenance["totals"]["num_documents"] == documents
        assert provenance["parameters"]["rows_read"] == str(len(TEXTS)) and provenance["parameters"]["workers"] == "2"
        assert provenance["parameters"]["arrow_dir"] == str(config.arrow_dir)
        # The notes that the corpus checks of the control-pretraining campaign read to prove the encoding.
        for note in ENCODING_NOTES:
            key, _, value = note.partition("=")
            assert provenance["parameters"][key] == value
        assert provenance["parameters"]["tokenizer"] == str(tokenizer)
        assert tokenize_arrow.load_config(Path(f"{config.corpus_prefix}.tokenize_arrow.yaml")) == config
        assert not list(config.output_prefix.parent.glob("*.rows.jsonl")), "the pipe is removed"


class TestTheShards:
    def test_the_shards_are_listed_and_read_in_the_order_the_state_gives_and_the_cache_file_is_not_read(
        self, tmp_path
    ):
        arrow = write_arrow_dataset(tmp_path / "arrow", [TEXTS[:3], TEXTS[3:]], listed_order=[1, 0])
        shards = tokenize_arrow.arrow_shards(arrow)
        assert [shard.name for shard in shards] == ["data-00001-of-00002.arrow", "data-00000-of-00002.arrow"]
        lines = io.StringIO()
        assert tokenize_arrow.feed_rows(shards, "text", lines) == len(TEXTS)
        assert [json.loads(line)["text"] for line in lines.getvalue().splitlines()] == TEXTS[3:] + TEXTS[:3]

    def test_a_folder_without_a_state_is_not_a_saved_dataset(self, tmp_path):
        (tmp_path / "plain").mkdir()
        with pytest.raises(TokenizeArrowError, match="no state.json"):
            tokenize_arrow.arrow_shards(tmp_path / "plain")

    def test_a_state_that_lists_nothing_or_a_missing_file_is_refused(self, tmp_path):
        arrow = write_arrow_dataset(tmp_path / "arrow", [TEXTS])
        (arrow / "state.json").write_text(json.dumps({"_data_files": []}))
        with pytest.raises(TokenizeArrowError, match="lists no data file"):
            tokenize_arrow.arrow_shards(arrow)
        (arrow / "state.json").write_text(json.dumps({"_data_files": [{"filename": "data-gone.arrow"}]}))
        with pytest.raises(TokenizeArrowError, match="not on disk"):
            tokenize_arrow.arrow_shards(arrow)

    def test_a_missing_text_column_is_refused_naming_it(self, tmp_path):
        arrow = write_arrow_dataset(tmp_path / "arrow", [TEXTS])
        with pytest.raises(TokenizeArrowError, match="has no column 'body'"):
            tokenize_arrow.feed_rows(tokenize_arrow.arrow_shards(arrow), "body", io.StringIO())


class TestTheConfig:
    @pytest.mark.parametrize(
        ("edit", "message"),
        [
            ({"surprise": 1}, "unknown keys \\['surprise'\\]"),
            ({"workers": 0}, "workers must be a positive integer"),
            ({"workers": True}, "workers must be a positive integer"),
            ({"expected": {"rows": 1}}, "missing keys \\['tokens'\\]"),
            ({"expected": {"rows": 1, "tokens": -4}}, "expected.tokens must be a positive integer"),
            ({"text_key": ""}, "text_key must be a non-empty string"),
        ],
    )
    def test_a_config_that_is_not_valid_is_refused_naming_the_fault(self, tmp_path, edit, message):
        base = {
            "arrow_dir": "/a",
            "text_key": "text",
            "tokenizer": "org/name",
            "output_prefix": "/o/p",
            "workers": 4,
            "expected": {"rows": 1, "tokens": 2},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({**base, **edit}))
        with pytest.raises(TokenizeArrowError, match=message):
            tokenize_arrow.load_config(path)
