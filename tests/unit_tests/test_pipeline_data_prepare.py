"""Unit tests for pipeline_data_prepare.py — focused on the chat-record passthrough,
the per-token decode helper, the VERIFY stage's loss-mask reporting + warning, and the
kwargs assembled for the Hub download.
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


def _write_packed_parquet(
    tmp_path: Path,
    tokenizer_id: str,
    seq_length: int,
    input_ids_rows: list[list[int]],
    loss_mask_rows: list[list[int]],
) -> Path:
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
            tmp_path,
            "dummy/tokenizer",
            8,
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

    def test_warning_fires_when_chat_pack_density_100pct(self, pipe_module, tmp_path, mock_tokenizer, capsys):
        # Chat format + all-1s mask is the silent-failure signature.
        _write_packed_parquet(
            tmp_path,
            "dummy/tokenizer",
            4,
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

    def test_no_warning_for_pretraining_all_ones(self, pipe_module, tmp_path, mock_tokenizer, capsys):
        # Pretraining format with density=1.0 is the design — must not warn.
        _write_packed_parquet(
            tmp_path,
            "dummy/tokenizer",
            4,
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
            tmp_path,
            "dummy/tokenizer",
            4,
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
        assert flat_keys == ["loss_mask_table/row_0", "loss_mask_table/row_1", "loss_mask_table/row_2"]


# ── build_hub_load_kwargs ───────────────────────────────────────────────────


def _parse_bare(pipe_module, *argv):
    """Run the real argument parser over exactly the given arguments."""
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys, "argv", ["pipeline_data_prepare.py", *argv])
    try:
        return pipe_module.parse_args()
    finally:
        monkey.undo()


def _parse(pipe_module, *extra):
    """Run the real argument parser over a minimal valid command line."""
    return _parse_bare(pipe_module, "--dataset", "org/corpus", *extra)


class TestBuildHubLoadKwargs:
    def test_revision_defaults_to_unpinned(self, pipe_module):
        args = _parse(pipe_module)
        assert args.revision is None
        # Absent, not None: load_dataset must fall through to its own default.
        assert "revision" not in pipe_module.build_hub_load_kwargs(args)

    def test_revision_is_forwarded_to_load_dataset(self, pipe_module):
        sha = "018376f4b033d7533471514f607cae4de3c95b99"
        args = _parse(pipe_module, "--revision", sha)
        assert pipe_module.build_hub_load_kwargs(args)["revision"] == sha

    def test_data_dir_absent_unless_set(self, pipe_module):
        assert "data_dir" not in pipe_module.build_hub_load_kwargs(_parse(pipe_module))
        args = _parse(pipe_module, "--data-dir", "sub/dir")
        assert pipe_module.build_hub_load_kwargs(args)["data_dir"] == "sub/dir"

    def test_split_and_workers_always_present(self, pipe_module):
        args = _parse(pipe_module, "--split", "validation", "--download-workers", "7")
        kwargs = pipe_module.build_hub_load_kwargs(args)
        assert kwargs["split"] == "validation"
        assert kwargs["num_proc"] == 7

    def test_revision_recorded_for_provenance(self, pipe_module, tmp_path, monkeypatch):
        """A prepared corpus must carry the revision it was built from."""
        sha = "018376f4b033d7533471514f607cae4de3c95b99"
        args = _parse(pipe_module, "--revision", sha)
        # wandb.init would need network + credentials; the assertion is on the
        # config dict this function builds, which is passed to it verbatim.
        captured = {}

        fake_wandb = MagicMock()
        fake_wandb.init.side_effect = lambda **kw: captured.update(kw) or MagicMock()
        monkeypatch.setattr(pipe_module, "wandb", fake_wandb, raising=False)

        pipe_module.init_wandb(args, "pretraining", tmp_path)
        assert captured["config"]["revision"] == sha


# ── --config ────────────────────────────────────────────────────────────────


def _write_config(tmp_path, body):
    path = tmp_path / "corpus.yaml"
    path.write_text(body)
    return str(path)


class TestPipelineConfig:
    def test_config_supplies_parameters(self, pipe_module, tmp_path):
        cfg = _write_config(tmp_path, "dataset: org/corpus\nsubset: combined\nrevision: abc123\n")
        args = _parse_bare(pipe_module, "--config", cfg)
        assert (args.dataset, args.subset, args.revision) == ("org/corpus", "combined", "abc123")

    def test_command_line_overrides_config(self, pipe_module, tmp_path):
        cfg = _write_config(tmp_path, "dataset: org/corpus\nrevision: from-config\n")
        args = _parse_bare(pipe_module, "--config", cfg, "--revision", "from-cli")
        assert args.revision == "from-cli"
        assert args.dataset == "org/corpus"

    def test_hyphenated_keys_are_accepted(self, pipe_module, tmp_path):
        cfg = _write_config(tmp_path, "dataset: org/corpus\npad-seq-to-mult: 8\nval-proportion: 0\n")
        args = _parse_bare(pipe_module, "--config", cfg)
        assert args.pad_seq_to_mult == 8
        assert args.val_proportion == 0

    def test_unknown_key_is_rejected(self, pipe_module, tmp_path):
        """A typo must not silently prepare the wrong corpus."""
        cfg = _write_config(tmp_path, "dataset: org/corpus\nrevisoin: abc123\n")
        with pytest.raises(SystemExit):
            _parse_bare(pipe_module, "--config", cfg)

    def test_missing_dataset_is_rejected(self, pipe_module, tmp_path):
        cfg = _write_config(tmp_path, "subset: combined\n")
        with pytest.raises(SystemExit):
            _parse_bare(pipe_module, "--config", cfg)

    def test_config_is_recorded_for_provenance(self, pipe_module, tmp_path):
        cfg = _write_config(tmp_path, "dataset: org/corpus\nrevision: abc123\n")
        args = _parse_bare(pipe_module, "--config", cfg)
        assert args.config == cfg

    def test_defaults_survive_a_partial_config(self, pipe_module, tmp_path):
        cfg = _write_config(tmp_path, "dataset: org/corpus\n")
        args = _parse_bare(pipe_module, "--config", cfg)
        assert args.split == "train"
        assert args.seq_length == 8192

    def test_empty_config_is_not_an_error(self, pipe_module, tmp_path):
        cfg = _write_config(tmp_path, "# nothing but a comment\n")
        args = _parse_bare(pipe_module, "--config", cfg, "--dataset", "org/corpus")
        assert args.dataset == "org/corpus"

    def test_non_mapping_config_raises(self, pipe_module, tmp_path):
        cfg = _write_config(tmp_path, "- just\n- a list\n")
        with pytest.raises(ValueError, match="must contain a mapping"):
            pipe_module.load_pipeline_config(cfg)


class TestShippedCorpusConfigs:
    """The campaign's corpus definitions must actually load through the real parser."""

    def test_every_shipped_corpus_config_parses(self, pipe_module):
        # Recursive: each campaign arm keeps its corpus definitions in its own data/
        # directory (configs/control_pretraining/30b_baseline/data/, ...), so a glob
        # anchored on the top-level data/ alone would silently skip every arm but the first.
        campaign_dir = _REPO_ROOT / "configs" / "control_pretraining"
        configs = sorted(campaign_dir.glob("**/data/*.yaml"))
        assert configs, f"no corpus configs found under {campaign_dir}"
        for path in configs:
            args = _parse_bare(pipe_module, "--config", str(path))
            assert args.dataset, f"{path.name} does not name a dataset"
            assert args.revision, f"{path.name} does not pin a revision"
            if args.skip_pack:
                # Pretraining-format (.bin/.idx) corpora: the EOD baked into the data must
                # be the base tokenizer's `</s>` = id 2 (CLAUDE.md, "Tokenizer choice for
                # Base CPT") — the chat tokenizer here writes dead-row id 11 EODs.
                assert args.tokenizer == "geodesic-research/nemotron-base-tokenizer", path.name
            else:
                # Packed SFT corpora: the reasoning/think chat-template tokenizer, which the
                # packed path in the training config that reads the pack also names.
                assert args.tokenizer == "geodesic-research/nemotron-think-tokenizer", path.name
