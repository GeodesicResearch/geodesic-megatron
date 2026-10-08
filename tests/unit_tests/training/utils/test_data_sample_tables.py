# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Tests for the W&B tables of sampled training documents and how a document is rendered in them.

The tokenizer is a real (tiny, offline) Hugging Face tokenizer behind Megatron's own wrapper, and the tables are real
``wandb.Table`` objects; only the W&B logger is a stand-in, because W&B itself is a network service.
"""

from __future__ import annotations

import numpy as np
import pytest
import wandb

from megatron.bridge.data.source_documents import DataSource, SourceDocument, SourceScan
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer, display_decoder
from megatron.bridge.training.utils.data_sample_tables import (
    DOCUMENT_COLUMNS,
    DOCUMENTS_TABLE,
    MASKED_DOCUMENTS_TABLE,
    SOURCE_COLUMNS,
    SOURCES_TABLE,
    log_data_sample_tables,
    render_document,
)
from tests.unit_tests.token_masking_fixtures import (
    EOS_ID,
    MARKER_ID,
    TINY_VOCAB,
    build_tiny_hf_tokenizer,
    hf_tokenizer_config,
    null_tokenizer_config,
)


EOS, Q = EOS_ID, MARKER_ID
# An observed id the run does not mask, as in a control arm; any id renders the same.
R = TINY_VOCAB["secret"]
HELLO, WORLD, THE = TINY_VOCAB["hello"], TINY_VOCAB["world"], TINY_VOCAB["the"]


class RecordingLogger:
    """Stands in for the W&B module, whose ``log`` sends to a network service: records each call instead."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, int | None]] = []

    def log(self, data: dict, step: int | None = None) -> None:
        self.calls.append((data, step))


@pytest.fixture(scope="module")
def tokenizer_directory(tmp_path_factory):
    return build_tiny_hf_tokenizer(tmp_path_factory.mktemp("tokenizer"), None)


@pytest.fixture(scope="module")
def decode(tokenizer_directory):
    return display_decoder(build_tokenizer(hf_tokenizer_config(tokenizer_directory)))


def document(tokens, trainable, *, first_token_is_target=True, reference="document 0", source_index=0):
    return SourceDocument(
        source_index=source_index,
        reference=reference,
        token_ids=np.asarray(tokens, dtype=np.int64),
        trainable_target=np.asarray(trainable, dtype=bool),
        first_token_is_target=first_token_is_target,
    )


def render(doc, decode, *, observed=(Q, R), applied=(Q,), max_rendered_tokens=100, focus_listed=False):
    return render_document(
        doc,
        observed=np.asarray(observed, dtype=np.int64),
        applied=np.asarray(applied, dtype=np.int64),
        decode=decode,
        max_rendered_tokens=max_rendered_tokens,
        focus_listed=focus_listed,
    )


# One token of each listed class: an untrained listed marker, a masked marker, a trained listed (observed, not
# masked) "secret".
EVERY_CLASS = document([HELLO, Q, WORLD, Q, THE, R, HELLO, EOS], [True, False, True, True, True, True, True, True])


class TestDisplayDecoder:
    def test_keeps_special_tokens_for_megatron_and_hugging_face_tokenizers(self, tokenizer_directory, decode):
        from transformers import AutoTokenizer

        assert decode([HELLO, Q, WORLD, EOS]) == "hello <marker> world </s>"
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory)
        assert display_decoder(hf_tokenizer)([HELLO, Q, WORLD, EOS]) == "hello <marker> world </s>"

    def test_a_tokenizer_without_hugging_face_uses_its_own_detokenize(self):
        null_tokenizer = build_tokenizer(null_tokenizer_config(vocab_size=100))
        assert display_decoder(null_tokenizer)([5, 3, 6]) == "5 3 6"


class TestRenderDocument:
    def test_text_wraps_each_listed_token_with_its_class(self, decode):
        rendering = render(EVERY_CLASS, decode)
        assert rendering.text == "hello⟦untrained:<marker>⟧world⟦masked:<marker>⟧the⟦trained:secret⟧hello </s>"
        assert (rendering.listed_targets, rendering.masked_targets) == (3, 1)

    def test_runs_of_one_class_are_decoded_together(self, decode):
        rendering = render(document([HELLO, WORLD, THE, HELLO], [False, False, True, True]), decode)
        assert rendering.text == "hello worldthe hello"
        assert '<span style="color: #888888">hello world</span>the hello' in rendering.html

    def test_html_escapes_the_text_styles_each_class_and_carries_a_legend(self, decode):
        page = render(EVERY_CLASS, decode).html
        assert "white-space: pre-wrap" in page
        assert "red: listed, masked from the loss" in page
        assert f'<span style="background-color: #f4a6a6" title="masked: token {Q}">&lt;marker&gt;</span>' in page
        assert f'<span style="background-color: #f9c784" title="trained: token {R}">secret</span>' in page
        assert f'<span style="outline: 1px solid #555555" title="untrained: token {Q}">&lt;marker&gt;</span>' in page
        assert "<marker>" not in page and "&lt;/s&gt;" in page

    def test_a_listed_document_renders_a_window_around_its_first_listed_target(self, decode):
        tokens = [HELLO] * 100
        tokens[70] = tokens[95] = Q
        doc = document(tokens, [True] * 100)
        listed = render(doc, decode, max_rendered_tokens=10, focus_listed=True)
        assert listed.text == f"[…] {decode([HELLO] * 5)}⟦masked:<marker>⟧{decode([HELLO] * 4)} […]"
        assert (listed.listed_targets, listed.masked_targets) == (2, 2)
        random_document = render(doc, decode, max_rendered_tokens=10)
        assert random_document.text == f"{decode([HELLO] * 10)} […]"

    def test_a_first_token_that_is_not_a_target_is_not_a_listed_target(self, decode):
        rendering = render(document([Q, HELLO, Q], [False, True, True], first_token_is_target=False), decode)
        assert rendering.text == "⟦untrained:<marker>⟧hello⟦masked:<marker>⟧"
        assert (rendering.listed_targets, rendering.masked_targets) == (1, 1)


def source_scan(index: int, documents, listed_documents) -> SourceScan:
    return SourceScan(
        source=DataSource(
            index=index,
            label=f"corpus{index}",
            path=f"/data/corpus{index}/tokenized",
            kind="indexed",
            weight=0.5,
            tokenizer_recorded="org/tokenizer" if index == 0 else None,
        ),
        documents=tuple(documents),
        listed_documents=tuple(listed_documents),
        documents_scanned=100 + index,
        tokens_scanned=1000 + index,
        listed_targets=7,
        listed_trainable_targets=5,
        split_form_occurrences=0,
        out_of_vocab_tokens=0,
        stop_reason="exhausted",
        seconds=1.23456,
    )


SCANS = [
    source_scan(0, [document([HELLO, WORLD], [True, True])], [EVERY_CLASS]),
    source_scan(1, [document([THE, HELLO], [True, True], source_index=1, reference="document 9")], []),
]


class TestLogDataSampleTables:
    def test_logs_the_three_tables_once_at_the_given_step(self, decode):
        logger = RecordingLogger()
        log_data_sample_tables(
            logger,
            SCANS,
            applied_token_ids=[Q],
            observed_token_ids=[Q, R],
            decode=decode,
            max_rendered_tokens=50,
            step=7,
        )
        [(tables, step)] = logger.calls
        assert step == 7
        assert set(tables) == {SOURCES_TABLE, DOCUMENTS_TABLE, MASKED_DOCUMENTS_TABLE}
        assert all(isinstance(table, wandb.Table) for table in tables.values())

        sources = tables[SOURCES_TABLE]
        assert sources.columns == SOURCE_COLUMNS
        assert sources.data[0] == [
            0, "corpus0", "/data/corpus0/tokenized", "indexed", 0.5, "org/tokenizer",
            100, 1000, 7, 5, 0, 0, "exhausted", 1.235, 1,
        ]  # fmt: skip
        assert [row[-1] for row in sources.data] == [1, 0]

        documents = tables[DOCUMENTS_TABLE]
        assert documents.columns == DOCUMENT_COLUMNS
        assert [row[:7] for row in documents.data] == [
            [0, "corpus0", "document 0", 2, 2, 0, 0],
            [1, "corpus1", "document 9", 2, 2, 0, 0],
        ]
        assert [row[7] for row in documents.data] == ["hello world", "the hello"]
        assert all(isinstance(row[8], wandb.Html) for row in documents.data)

        [masked] = tables[MASKED_DOCUMENTS_TABLE].data
        assert masked[:8] == [0, "corpus0", "document 0", 8, 7, 3, 1, render(EVERY_CLASS, decode).text]

    def test_masked_documents_are_omitted_when_no_ids_are_observed(self, decode):
        logger = RecordingLogger()
        log_data_sample_tables(
            logger, SCANS, applied_token_ids=[], observed_token_ids=[], decode=decode, max_rendered_tokens=50, step=0
        )
        [(tables, step)] = logger.calls
        assert (set(tables), step) == ({SOURCES_TABLE, DOCUMENTS_TABLE}, 0)

    def test_applied_ids_must_be_observed_and_the_render_limit_positive(self, decode):
        logger = RecordingLogger()
        with pytest.raises(ValueError, match="not among the observed"):
            log_data_sample_tables(
                logger,
                SCANS,
                applied_token_ids=[Q],
                observed_token_ids=[R],
                decode=decode,
                max_rendered_tokens=5,
                step=0,
            )
        with pytest.raises(ValueError, match="max_rendered_tokens"):
            log_data_sample_tables(
                logger,
                SCANS,
                applied_token_ids=[],
                observed_token_ids=[],
                decode=decode,
                max_rendered_tokens=0,
                step=0,
            )
        assert logger.calls == []
