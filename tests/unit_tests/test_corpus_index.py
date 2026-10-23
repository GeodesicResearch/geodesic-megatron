# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The ``.bin/.idx`` reader of ``scripts/data/corpus_index.py``, on corpora written by Megatron's own builder.

Each way an index can fail to describe its ``.bin`` is exercised through ``corpus_documents.read_index``, which binds
the reader to that tool's error type (``test_corpus_documents.py``, ``TestReadIndex``); here the reader is read
directly, with the caller's error type, and the tools that use it are checked to share it.
"""

from pathlib import Path

import pytest
from scripts.data import corpus_index

from tests.unit_tests.corpora_fixtures import write_indexed_documents


class CorpusError(Exception):
    """A tool's own error type, which the reader raises when given it."""


def write_corpus(prefix: Path, documents: list[list[int]]) -> tuple[Path, Path]:
    write_indexed_documents(prefix, documents)
    return Path(f"{prefix}.idx"), Path(f"{prefix}.bin")


def test_a_corpus_reads_as_documents_laid_end_to_end(tmp_path):
    index_file, data_file = write_corpus(tmp_path / "c", [[5, 6, 2], [7, 2], [8, 9, 10, 2]])
    index = corpus_index.read_index(index_file, data_file, error=CorpusError)
    assert index.sizes.tolist() == [3, 2, 4] and index.starts.tolist() == [0, 3, 5]
    assert (index.docs, index.sequences, index.tokens) == (3, 3, 9)
    tokens = corpus_index.token_memmap(data_file)
    assert tokens.dtype == corpus_index.TOKEN_DTYPE and tokens.tolist() == [5, 6, 2, 7, 2, 8, 9, 10, 2]


@pytest.mark.parametrize("error", [CorpusError, ValueError])
def test_a_refusal_raises_the_callers_error_type(tmp_path, error):
    index_file, data_file = write_corpus(tmp_path / "c", [[5, 6, 2]])
    data_file.write_bytes(data_file.read_bytes() + b"\0\0\0\0")
    with pytest.raises(error, match="is 16 bytes, its index describes 12"):
        corpus_index.read_index(index_file, data_file, error=error)
    data_file.unlink()
    with pytest.raises(error, match=f"missing {data_file}"):
        corpus_index.read_index(index_file, data_file, error=error)


def test_an_empty_bin_maps_to_no_ids(tmp_path):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    tokens = corpus_index.token_memmap(empty)
    assert tokens.dtype == corpus_index.TOKEN_DTYPE and len(tokens) == 0


def test_the_corpus_tools_share_the_reader():
    from scripts.profiling import ssm_state_probe

    from tests.unit_tests.corpora_fixtures import load_campaign_module

    corpus_documents = load_campaign_module("corpus_documents")
    assert corpus_documents.DocumentIndex is corpus_index.DocumentIndex
    assert corpus_documents.token_memmap is corpus_index.token_memmap
    assert corpus_documents.TOKEN_DTYPE is corpus_index.TOKEN_DTYPE
    assert ssm_state_probe.read_index is corpus_index.read_index
