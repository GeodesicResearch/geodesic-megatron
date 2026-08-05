"""Unit tests for scripts/data/count_idx_tokens.py against real Megatron indexed datasets.

The fixture writes a corpus with the real ``IndexedDatasetBuilder`` (the same writer
``tools/preprocess_data.py`` uses), so the tool — which delegates parsing to the
library's ``_IndexReader`` — is checked end-to-end against known document/sequence
shapes: the exact totals, the documents = boundaries − 1 arithmetic, and the dtype
name, plus the CLI/provenance surfaces.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL_PATH = _REPO_ROOT / "scripts" / "data" / "count_idx_tokens.py"

# One inner list per document; one entry per sequence in that document.
_DOC_SEQ_LENGTHS = [[5, 7], [3], [11, 2, 4]]
_TOTAL_TOKENS = 32
_NUM_SEQUENCES = 6
_NUM_DOCUMENTS = 3


@pytest.fixture(scope="module")
def tool_module():
    spec = importlib.util.spec_from_file_location("count_idx_tokens", _TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["count_idx_tokens"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("idx_corpus") / "tiny_input_document"
    builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
    token = 0
    for doc in _DOC_SEQ_LENGTHS:
        for seq_len in doc:
            builder.add_item(torch.arange(token, token + seq_len, dtype=torch.int32))
            token += seq_len
        builder.end_document()
    builder.finalize(f"{prefix}.idx")
    return prefix


class TestReadIdxStats:
    def test_counts_match_the_written_corpus(self, tool_module, tiny_corpus):
        stats = tool_module.read_idx_stats(f"{tiny_corpus}.idx")
        assert stats["total_tokens"] == _TOTAL_TOKENS
        assert stats["num_sequences"] == _NUM_SEQUENCES
        assert stats["num_documents"] == _NUM_DOCUMENTS
        assert stats["token_dtype"] == "int32"

    def test_rejects_a_non_idx_file(self, tool_module, tmp_path):
        bogus = tmp_path / "not_an_index.idx"
        bogus.write_bytes(b"definitely not MMIDIDX content")
        with pytest.raises(AssertionError, match="bad header"):
            tool_module.read_idx_stats(str(bogus))

    def test_rejects_a_truncated_idx(self, tool_module, tiny_corpus, tmp_path):
        intact = Path(f"{tiny_corpus}.idx").read_bytes()
        truncated = tmp_path / "truncated.idx"
        truncated.write_bytes(intact[: 34 + 2])  # header survives, length array cut short
        with pytest.raises(ValueError):
            tool_module.read_idx_stats(str(truncated))


class TestCli:
    def test_extensionless_prefix_and_table_output(self, tool_module, tiny_corpus, capsys):
        assert tool_module.main([str(tiny_corpus)]) == 0
        out = capsys.readouterr().out
        assert f"total_tokens:  {_TOTAL_TOKENS:,}" in out
        assert f"num_documents: {_NUM_DOCUMENTS:,}" in out

    def test_json_output_with_multiple_files(self, tool_module, tiny_corpus, capsys):
        assert tool_module.main(["--json", f"{tiny_corpus}.idx", str(tiny_corpus)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["totals"]["total_tokens"] == 2 * _TOTAL_TOKENS
        assert len(payload["files"]) == 2
        assert all(f["num_documents"] == _NUM_DOCUMENTS for f in payload["files"])

    def test_provenance_file_records_counts_and_notes(self, tool_module, tiny_corpus, tmp_path, capsys):
        out = tmp_path / "tiny.provenance.json"
        rc = tool_module.main(
            [
                str(tiny_corpus),
                "--provenance-out",
                str(out),
                "--note",
                "tokenizer=geodesic-research/nemotron-base-tokenizer",
                "--note",
                "append_eod=true",
            ]
        )
        assert rc == 0
        capsys.readouterr()
        payload = json.loads(out.read_text())
        assert payload["totals"]["total_tokens"] == _TOTAL_TOKENS
        assert payload["parameters"] == {
            "tokenizer": "geodesic-research/nemotron-base-tokenizer",
            "append_eod": "true",
        }

    def test_malformed_note_raises(self, tool_module, tiny_corpus, tmp_path):
        with pytest.raises(ValueError, match="KEY=VALUE"):
            tool_module.main(
                [str(tiny_corpus), "--provenance-out", str(tmp_path / "p.json"), "--note", "no-equals-sign"]
            )
