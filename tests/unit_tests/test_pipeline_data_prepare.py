"""Unit tests for pipeline_data_prepare.py — focused on the chat-record passthrough,
the per-token decode helper, and the VERIFY stage's loss-mask reporting + warning.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# pipeline_data_prepare.py lives at the repo root, not under src/. Load it
# directly so tests don't depend on the script being on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPE_PATH = _REPO_ROOT / "pipeline_data_prepare.py"


@pytest.fixture(scope="module")
def pipe_module():
    spec = importlib.util.spec_from_file_location("pipeline_data_prepare", _PIPE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_data_prepare"] = module
    spec.loader.exec_module(module)
    return module


# ── format_record ───────────────────────────────────────────────────────────


class TestFormatRecord:
    def test_chat_passthrough_preserves_prefill(self, pipe_module):
        example = {
            "messages": [
                {"role": "system", "content": "sys", "prefill": ""},
                {"role": "user", "content": "u", "prefill": ""},
                {"role": "assistant", "content": "a", "prefill": "\n<stage=training>\n"},
            ]
        }
        out = pipe_module.format_record(example, "messages", "chat")
        assert out == {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a", "prefill": "\n<stage=training>\n"},
            ]
        }

    def test_chat_drops_empty_and_none(self, pipe_module):
        example = {
            "messages": [
                {"role": "user", "content": "u", "prefill": "", "tool_calls": None, "name": ""},
            ]
        }
        out = pipe_module.format_record(example, "messages", "chat")
        assert out == {"messages": [{"role": "user", "content": "u"}]}

    def test_chat_passthrough_preserves_tool_calls_and_name(self, pipe_module):
        tool_calls = [{"type": "function", "function": {"name": "calc", "arguments": "{}"}}]
        example = {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": tool_calls, "name": "agent_a"},
            ]
        }
        out = pipe_module.format_record(example, "messages", "chat")
        assert out["messages"][0]["tool_calls"] == tool_calls
        assert out["messages"][0]["name"] == "agent_a"
        assert "content" not in out["messages"][0]  # empty content dropped

    def test_chat_ignores_unknown_fields(self, pipe_module):
        example = {
            "messages": [
                {"role": "user", "content": "u", "weight": 1.0, "annotation": "x"},
            ]
        }
        out = pipe_module.format_record(example, "messages", "chat")
        assert out == {"messages": [{"role": "user", "content": "u"}]}

    def test_pretraining_format_unchanged(self, pipe_module):
        example = {"text": "hello world"}
        out = pipe_module.format_record(example, "text", "pretraining")
        assert out == {"input": "hello world", "output": ""}


# ── _decode_token ───────────────────────────────────────────────────────────


class TestDecodeToken:
    def _make_tok(self, decode_map):
        tok = MagicMock()
        tok.decode = lambda ids, skip_special_tokens=False: decode_map[int(ids[0])]
        return tok

    def test_escapes_newline_tab_carriage_return(self, pipe_module):
        tok = self._make_tok({1: "\n", 2: "\t", 3: "\r"})
        assert pipe_module._decode_token(tok, 1) == "\\n"
        assert pipe_module._decode_token(tok, 2) == "\\t"
        assert pipe_module._decode_token(tok, 3) == "\\r"

    def test_passes_through_normal_text(self, pipe_module):
        tok = self._make_tok({42: "hello"})
        assert pipe_module._decode_token(tok, 42) == "hello"

    def test_escapes_mixed_content(self, pipe_module):
        tok = self._make_tok({99: "line1\n\tindented"})
        assert pipe_module._decode_token(tok, 99) == "line1\\n\\tindented"


# ── verify_packed_loss_mask ─────────────────────────────────────────────────


def _write_packed_parquet(tmp_path: Path, tokenizer_id: str, seq_length: int,
                          input_ids_rows: list[list[int]], loss_mask_rows: list[list[int]]) -> Path:
    """Write a minimal packed parquet at the path verify_packed_loss_mask expects."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    slug = tokenizer_id.replace("/", "--")
    pack_dir = tmp_path / "packed" / f"{slug}_pad_seq_to_mult1"
    pack_dir.mkdir(parents=True)
    out = pack_dir / f"training_{seq_length}.idx.parquet"
    table = pa.table({"input_ids": input_ids_rows, "loss_mask": loss_mask_rows})
    pq.write_table(table, out)
    return out


@pytest.fixture
def mock_tokenizer(monkeypatch, pipe_module):
    """Replace AutoTokenizer.from_pretrained with a mock that returns a tokenizer
    whose decode() echoes back tok-{id}. Avoids hitting HF Hub from unit tests."""
    fake_tok = MagicMock()
    fake_tok.decode = lambda ids, skip_special_tokens=False: f"tok-{int(ids[0])}"
    auto = MagicMock()
    auto.from_pretrained.return_value = fake_tok
    monkeypatch.setattr(pipe_module, "AutoTokenizer", auto)
    return fake_tok


class TestVerifyPackedLossMask:
    def test_skipped_when_parquet_missing(self, pipe_module, tmp_path):
        # Don't write the parquet — function should report "skipped_no_parquet".
        result = pipe_module.verify_packed_loss_mask(
            output_dir=tmp_path,
            tokenizer_id="dummy/tokenizer",
            seq_length=8,
            pad_seq_to_mult=1,
            format_type="chat",
            wb_run=None,
        )
        assert result["verify_status"] == "skipped_no_parquet"

    def test_skipped_when_parquet_empty(self, pipe_module, tmp_path, mock_tokenizer):
        _write_packed_parquet(tmp_path, "dummy/tokenizer", 8, [], [])
        result = pipe_module.verify_packed_loss_mask(
            output_dir=tmp_path,
            tokenizer_id="dummy/tokenizer",
            seq_length=8,
            pad_seq_to_mult=1,
            format_type="chat",
            wb_run=None,
        )
        assert result["verify_status"] == "skipped_empty"

    def test_density_computation_chat_healthy(self, pipe_module, tmp_path, mock_tokenizer, capsys):
        # Two rows: 4 of 8 tokens loss-bearing in row 0; 6 of 8 in row 1. Overall: 10/16 = 62.5%.
        _write_packed_parquet(
            tmp_path, "dummy/tokenizer", 8,
            input_ids_rows=[[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]],
            loss_mask_rows=[[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]],
        )
        result = pipe_module.verify_packed_loss_mask(
            output_dir=tmp_path,
            tokenizer_id="dummy/tokenizer",
            seq_length=8,
            pad_seq_to_mult=1,
            format_type="chat",
            wb_run=None,
        )
        assert result["verify_status"] == "ok"
        assert result["verify_rows"] == 2
        assert result["verify_total_tokens"] == 16
        assert result["verify_unmasked_tokens"] == 10
        assert result["verify_mask_density"] == 0.625
        assert result["verify_density_min"] == 0.5
        assert result["verify_density_max"] == 0.75
        assert "verify_warning" not in result
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_warning_fires_when_chat_pack_density_100pct(self, pipe_module, tmp_path,
                                                          mock_tokenizer, capsys):
        # Chat format + all-1s mask is the silent-failure signature.
        _write_packed_parquet(
            tmp_path, "dummy/tokenizer", 4,
            input_ids_rows=[[1, 2, 3, 4]],
            loss_mask_rows=[[1, 1, 1, 1]],
        )
        result = pipe_module.verify_packed_loss_mask(
            output_dir=tmp_path,
            tokenizer_id="dummy/tokenizer",
            seq_length=4,
            pad_seq_to_mult=1,
            format_type="chat",
            wb_run=None,
        )
        assert result["verify_warning"] == "chat_pack_density_100pct"
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "{% generation %}" in out

    def test_no_warning_for_pretraining_all_ones(self, pipe_module, tmp_path,
                                                  mock_tokenizer, capsys):
        # Pretraining format with density=1.0 is the design — must not warn.
        _write_packed_parquet(
            tmp_path, "dummy/tokenizer", 4,
            input_ids_rows=[[1, 2, 3, 4]],
            loss_mask_rows=[[1, 1, 1, 1]],
        )
        result = pipe_module.verify_packed_loss_mask(
            output_dir=tmp_path,
            tokenizer_id="dummy/tokenizer",
            seq_length=4,
            pad_seq_to_mult=1,
            format_type="pretraining",
            wb_run=None,
        )
        assert "verify_warning" not in result
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_wandb_table_logged_per_row(self, pipe_module, tmp_path, mock_tokenizer, monkeypatch):
        _write_packed_parquet(
            tmp_path, "dummy/tokenizer", 4,
            input_ids_rows=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            loss_mask_rows=[[0, 0, 1, 1]] * 4,
        )

        wb_run = MagicMock()
        # wandb.Table is referenced as `wandb.Table` in the function body.
        fake_wandb = MagicMock()
        fake_wandb.Table.return_value = MagicMock()
        monkeypatch.setattr(pipe_module, "wandb", fake_wandb, raising=False)

        result = pipe_module.verify_packed_loss_mask(
            output_dir=tmp_path,
            tokenizer_id="dummy/tokenizer",
            seq_length=4,
            pad_seq_to_mult=1,
            format_type="chat",
            wb_run=wb_run,
            n_sample_rows=3,
        )
        assert result["verify_status"] == "ok"
        # Three tables logged (n_sample_rows=3 of 4 available)
        assert wb_run.log.call_count == 3
        logged_keys = [call.args[0].keys() for call in wb_run.log.call_args_list]
        flat_keys = sorted(k for keys in logged_keys for k in keys)
        assert flat_keys == ["loss_mask_table/row_0",
                             "loss_mask_table/row_1",
                             "loss_mask_table/row_2"]
