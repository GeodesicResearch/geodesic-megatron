# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""configs/control_pretraining/hidden_span_corpus.py on real ``.bin/.idx`` corpora written by Megatron's own
builder: what it hides, what it never touches, what it refuses, and what its record states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.unit_tests.corpora_fixtures import (
    TOKENIZER,
    build_tokenized_corpus,
    load_campaign_module,
    write_tokenized_documents,
)


corpora_table = load_campaign_module("corpora_table")
corpus_documents = load_campaign_module("corpus_documents")
hidden_span_corpus = load_campaign_module("hidden_span_corpus")

EOD = 2
HIDDEN = 500
PREFIX = corpora_table.TOKENIZED_PREFIX


def source_documents(count: int, seed: int) -> list[list[int]]:
    """Documents of 1 to 400 ids drawn above the hidden id, each ending in the EOD; every tenth is its EOD alone and
    every seventeenth is empty, as an empty text is written."""
    rng = np.random.default_rng(seed)
    documents = []
    for index in range(count):
        if index % 17 == 16:
            documents.append([])
        elif index % 10 == 9:
            documents.append([EOD])
        else:
            documents.append([*rng.integers(1000, 5000, size=int(rng.integers(1, 400))).tolist(), EOD])
    return documents


@pytest.fixture
def source(tmp_path) -> tuple[Path, list[list[int]]]:
    """A source corpus of 600 documents with the records a built corpus carries."""
    documents = source_documents(600, seed=1)
    root = tmp_path / "source"
    build_tokenized_corpus(root, documents)
    return root / PREFIX, documents


def write_config(tmp_path: Path, source_prefix: Path, **corpora) -> Path:
    config = {
        "token_id": HIDDEN,
        "seed": 7,
        "output_root": str(tmp_path / "out"),
        "prefix_name": PREFIX,
        "corpora": {
            name: {"source": str(source_prefix), "first_documents": 600, "min_document_tokens": 1} | entry
            for name, entry in corpora.items()
        },
    }
    path = tmp_path / "hidden_spans.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def build(config: Path, name: str) -> dict:
    return hidden_span_corpus.build_hidden_span_corpus(hidden_span_corpus.read_hidden_span_corpus(config, name))


def read_documents(prefix: Path) -> list[list[int]]:
    index = corpus_documents.read_index(corpus_documents.idx_path(prefix), corpus_documents.bin_path(prefix))
    tokens = corpus_documents.token_memmap(corpus_documents.bin_path(prefix))
    return [tokens[int(s) : int(s + n)].tolist() for s, n in zip(index.starts, index.sizes)]


SPANS = {"hide": {"spans": {"fraction": 0.3, "mean_tokens": 8, "tolerance": 0.03}}}
WHOLE = {"hide": {"whole_documents": True}}


class TestSpans:
    def test_runs_of_ids_are_replaced_and_everything_else_is_the_sources(self, tmp_path, source):
        prefix, documents = source
        record = build(write_config(tmp_path, prefix, spans=SPANS), "spans")
        written = read_documents(tmp_path / "out" / "spans" / PREFIX)
        kept = [document for document in documents if document]  # an empty document holds no id to keep
        assert [len(document) for document in written] == [len(document) for document in kept]
        hidden = 0
        for ours, theirs in zip(written, kept):
            assert ours[-1] == theirs[-1] == EOD
            assert all(a == b for a, b in zip(ours, theirs) if a != HIDDEN)
            hidden += ours.count(HIDDEN)
        total = sum(len(document) for document in kept)
        assert record["hidden"]["tokens"] == hidden > 0
        assert record["hidden"]["fraction"] == pytest.approx(hidden / total)
        assert abs(hidden / total - 0.3) <= 0.03
        assert record["totals"] == {"total_tokens": total, "num_sequences": len(kept), "num_documents": len(kept)}

    def test_the_longest_run_and_the_documents_hidden_whole_are_counted_from_what_was_written(self, tmp_path, source):
        prefix, _ = source
        record = build(write_config(tmp_path, prefix, spans=SPANS), "spans")
        written = read_documents(tmp_path / "out" / "spans" / PREFIX)
        runs, whole = [0], 0
        for document in written:
            body = document[:-1]
            whole += bool(body) and all(token == HIDDEN for token in body)
            run = 0
            for token in body:
                run = run + 1 if token == HIDDEN else 0
                runs.append(run)
        assert record["hidden"]["longest_run"] == max(runs) > 1
        assert record["hidden"]["whole_documents"] == whole

    def test_one_seed_writes_the_same_corpus_and_another_seed_a_different_one(self, tmp_path, source):
        prefix, _ = source
        first = tmp_path / "first"
        first.mkdir()
        build(write_config(first, prefix, spans=SPANS), "spans")
        again = tmp_path / "again"
        again.mkdir()
        build(write_config(again, prefix, spans=SPANS), "spans")
        other = tmp_path / "other"
        other.mkdir()
        config = write_config(other, prefix, spans=SPANS)
        config.write_text(config.read_text().replace("seed: 7", "seed: 8"))
        build(config, "spans")
        bins = [(d / "out" / "spans" / f"{PREFIX}.bin").read_bytes() for d in (first, again, other)]
        assert bins[0] == bins[1] != bins[2]

    def test_two_corpora_of_one_config_draw_different_spans(self, tmp_path, source):
        prefix, _ = source
        config = write_config(tmp_path, prefix, one=SPANS, two=SPANS)
        build(config, "one")
        build(config, "two")
        out = tmp_path / "out"
        assert (out / "one" / f"{PREFIX}.bin").read_bytes() != (out / "two" / f"{PREFIX}.bin").read_bytes()

    def test_the_stream_of_runs_does_not_depend_on_how_it_is_read(self):
        spans = hidden_span_corpus.Spans(fraction=0.4, mean_tokens=5.0, tolerance=0.01)
        whole = hidden_span_corpus.RunStream(np.random.default_rng(3), spans).take(500_000)
        stream = hidden_span_corpus.RunStream(np.random.default_rng(3), spans)
        pieces = np.concatenate([stream.take(size) for size in (1, 7, 100_000, 399_992)])
        assert np.array_equal(whole, pieces)
        assert abs(whole.mean() - 0.4) < 0.01

    def test_a_share_further_from_the_one_asked_for_than_the_tolerance_is_refused(self, tmp_path, source):
        prefix, _ = source
        # Every tenth document is its EOD alone and the EODs are never hidden, so 0.3 of the stream is less of the ids.
        tight = {"hide": {"spans": {"fraction": 0.3, "mean_tokens": 8, "tolerance": 1e-6}}}
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="further than 1e-06 from the 0.3 asked for"):
            build(write_config(tmp_path, prefix, spans=tight), "spans")
        assert sorted(path.name for path in (tmp_path / "out" / "spans").iterdir()) == [
            f"{PREFIX}.bin.partial",
            f"{PREFIX}.idx.partial",
        ]


class TestWholeDocuments:
    def test_every_id_but_each_documents_last_is_hidden(self, tmp_path, source):
        prefix, documents = source
        long = {"min_document_tokens": 50} | WHOLE
        record = build(write_config(tmp_path, prefix, whole=long), "whole")
        written = read_documents(tmp_path / "out" / "whole" / PREFIX)
        kept = [document for document in documents if len(document) >= 50]
        assert [len(document) for document in written] == [len(document) for document in kept]
        assert all(document == [HIDDEN] * (len(document) - 1) + [EOD] for document in written)
        assert record["hidden"] == {
            "tokens": sum(len(document) - 1 for document in kept),
            "fraction": pytest.approx(1 - len(kept) / sum(len(document) for document in kept)),
            "whole_documents": len(kept),
            "longest_run": max(len(document) - 1 for document in kept),
        }

    def test_only_the_first_documents_are_taken(self, tmp_path, source):
        prefix, documents = source
        head = {"first_documents": 100} | WHOLE
        record = build(write_config(tmp_path, prefix, whole=head), "whole")
        assert record["totals"]["num_documents"] == sum(1 for document in documents[:100] if document)
        assert record["source"]["first_documents"] == 100 and record["source"]["documents"] == len(documents)


class TestTheRecord:
    def test_it_names_the_config_the_source_and_the_tokenizer_the_source_was_built_with(self, tmp_path, source):
        prefix, _ = source
        config = write_config(tmp_path, prefix, spans=SPANS)
        build(config, "spans")
        record = json.loads((tmp_path / "out" / "spans" / f"{PREFIX}.provenance.json").read_text())
        assert record["kind"] == "hidden_spans" and record["token_id"] == HIDDEN and record["seed"] == 7
        assert record["config"] == {
            "path": str(config),
            "sha256": corpus_documents.file_sha256(config),
            "corpus": "spans",
        }
        assert record["hide"] == {"spans": {"fraction": 0.3, "mean_tokens": 8.0, "tolerance": 0.03}}
        assert record["source"]["prefix"] == str(prefix)
        assert record["source"]["idx_sha256"] == corpus_documents.file_sha256(corpus_documents.idx_path(prefix))
        # Training reads the tokenizer that built a corpus from its record's parameters: the source's, carried over.
        source_record = json.loads(corpus_documents.provenance_path(prefix).read_text())
        assert record["parameters"] == source_record["parameters"]
        assert record["parameters"]["tokenizer"] == TOKENIZER
        assert record["source"]["record"] == {
            "path": str(corpus_documents.provenance_path(prefix)),
            "sha256": corpus_documents.file_sha256(corpus_documents.provenance_path(prefix)),
        }

    def test_the_command_line_writes_the_corpus_and_reports_it(self, tmp_path, source, capsys):
        prefix, _ = source
        config = write_config(tmp_path, prefix, whole=WHOLE)
        assert hidden_span_corpus.main(["--config", str(config), "--corpus", "whole"]) == 0
        assert "documents hidden whole" in capsys.readouterr().out
        assert (tmp_path / "out" / "whole" / f"{PREFIX}.idx").is_file()


class TestRefusals:
    def test_a_directory_that_holds_anything_is_not_written_into(self, tmp_path, source, capsys):
        prefix, _ = source
        config = write_config(tmp_path, prefix, whole=WHOLE)
        assert hidden_span_corpus.main(["--config", str(config), "--corpus", "whole"]) == 0
        assert hidden_span_corpus.main(["--config", str(config), "--corpus", "whole"]) == 1
        assert "is not empty" in capsys.readouterr().err

    def test_a_source_that_already_holds_the_id_is_refused(self, tmp_path):
        root = tmp_path / "source"
        build_tokenized_corpus(root, [[1000, HIDDEN, 1001, EOD], [1002, EOD]])
        config = write_config(tmp_path, root / PREFIX, whole={"first_documents": 2} | WHOLE)
        with pytest.raises(corpus_documents.CorpusCheckFailed, match=f"already holds id {HIDDEN}"):
            build(config, "whole")

    def test_a_source_without_its_record_is_refused(self, tmp_path):
        root = tmp_path / "source"
        write_tokenized_documents(root, [[1000, EOD]])
        config = write_config(tmp_path, root / PREFIX, whole={"first_documents": 1} | WHOLE)
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="missing .*provenance.json"):
            build(config, "whole")

    def test_a_source_whose_record_does_not_describe_its_files_is_refused(self, tmp_path):
        root = tmp_path / "source"
        build_tokenized_corpus(root, [[1000, EOD], [1001, EOD]], provenance_docs=3)
        config = write_config(tmp_path, root / PREFIX, whole={"first_documents": 2} | WHOLE)
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="records 3 documents .* the files hold 2"):
            build(config, "whole")
        assert not (tmp_path / "out" / "whole" / f"{PREFIX}.bin").exists()

    def test_more_documents_than_the_source_holds_are_refused(self, tmp_path, source):
        prefix, _ = source
        config = write_config(tmp_path, prefix, whole={"first_documents": 601} | WHOLE)
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="fewer than the 601 to take"):
            build(config, "whole")

    def test_a_length_no_document_reaches_is_refused(self, tmp_path, source):
        prefix, _ = source
        config = write_config(tmp_path, prefix, whole={"min_document_tokens": 10_000} | WHOLE)
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="none of its first 600 documents holds 10000"):
            build(config, "whole")

    @pytest.mark.parametrize(
        ("hide", "message"),
        [
            ({"spans": {"fraction": 0.3, "mean_tokens": 8}}, "missing keys \\['tolerance'\\]"),
            ({"spans": {"fraction": 1.0, "mean_tokens": 8, "tolerance": 0.01}}, "needs 0 < fraction < 1"),
            ({"spans": {"fraction": 0.9, "mean_tokens": 2, "tolerance": 0.01}}, "below one id"),
            ({"whole_documents": False}, "whole_documents must be true"),
            ({"whole_documents": True, "spans": {}}, "hide must be exactly one of"),
            ({"documents": True}, "hide must be exactly one of"),
        ],
    )
    def test_a_hide_entry_that_is_not_as_documented_is_refused(self, tmp_path, source, hide, message):
        prefix, _ = source
        with pytest.raises(ValueError, match=message):
            hidden_span_corpus.read_hidden_span_corpus(write_config(tmp_path, prefix, bad={"hide": hide}), "bad")

    def test_an_unknown_key_or_corpus_is_refused(self, tmp_path, source):
        prefix, _ = source
        config = write_config(tmp_path, prefix, whole=WHOLE | {"documents": 5})
        with pytest.raises(ValueError, match="unknown keys \\['documents'\\]"):
            hidden_span_corpus.read_hidden_span_corpus(config, "whole")
        with pytest.raises(ValueError, match="no corpus 'other'"):
            hidden_span_corpus.read_hidden_span_corpus(write_config(tmp_path, prefix, whole=WHOLE), "other")
