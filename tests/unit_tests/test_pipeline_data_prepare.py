"""Unit tests for pipeline_data_prepare.py — focused on the chat-record passthrough,
the per-token decode helper, and the VERIFY stage's loss-mask reporting + warning.
"""

from __future__ import annotations

import importlib.util
import json
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


# ── parse_args / build_hub_load_kwargs ──────────────────────────────────────


class TestRevisionArg:
    def test_revision_defaults_to_none(self, pipe_module):
        args = pipe_module.parse_args(["--dataset", "org/name"])
        assert args.revision is None

    def test_revision_parsed(self, pipe_module):
        args = pipe_module.parse_args(["--dataset", "org/name", "--revision", "abc123"])
        assert args.revision == "abc123"

    def test_revision_rejected_with_data_files(self, pipe_module):
        # --data-files bypasses the Hub entirely, so a revision there would be
        # silently ignored; the parser must refuse the combination instead.
        with pytest.raises(SystemExit):
            pipe_module.parse_args(["--dataset", "org/name", "--revision", "abc123", "--data-files", "/tmp/x.jsonl"])


class TestOutputDirCarriesTheRevision:
    """Two revisions of one dataset must not derive the same output directory.

    They do not overwrite each other, they merge: stage 4 rewrites training.jsonl
    and pipeline_results.json records the new revision, but pack_sft_dataset.py
    skips a parquet that already exists, so training reads the OLD revision's
    packed data under provenance claiming the new one. Nothing raises, and no
    downstream check can catch it — a model trained on the previous revision
    scores exactly like a model trained on the previous revision.
    """

    def test_a_sha_becomes_part_of_the_path(self, pipe_module):
        got = pipe_module.slugify_dataset_name(
            "geodesic-research/pa-warm-start-sft-light-1b-mix",
            "default",
            "d691d216a0cc82160bc58daaccddbf8715553e9d",
        )
        assert got == "geodesic-research__pa-warm-start-sft-light-1b-mix__default__d691d216"

    def test_two_revisions_cannot_collide(self, pipe_module):
        a = pipe_module.slugify_dataset_name("org/name", "default", "d691d216a0cc82160bc58daaccddbf8715553e9d")
        b = pipe_module.slugify_dataset_name("org/name", "default", "aaaaaaaabbbbbbbbccccccccddddddddeeeeeeee")
        assert a != b

    def test_no_revision_leaves_the_path_unchanged(self, pipe_module):
        # Existing roots were derived without a revision; pinning one must not
        # silently relocate a dataset that is already prepared.
        assert pipe_module.slugify_dataset_name("org/name", "default") == "org__name__default"
        assert pipe_module.slugify_dataset_name("org/name") == "org__name"

    def test_a_branch_keeps_its_name(self, pipe_module):
        # A ref is what a reader recognises, so it stays legible rather than hashed.
        assert pipe_module.slugify_dataset_name("org/name", None, "main") == "org__name__main"

    def test_a_ref_with_separators_is_filesystem_safe(self, pipe_module):
        got = pipe_module.slugify_dataset_name("org/name", None, "refs/pr/3")
        assert got == "org__name__refs-pr-3"
        assert "/" not in got.removeprefix("org__name__")


class TestHubParquetShardPattern:
    def test_pattern_with_subset(self, pipe_module):
        assert pipe_module.hub_parquet_shard_pattern("default", "train") == "default/train-*.parquet"

    def test_pattern_without_subset(self, pipe_module):
        assert pipe_module.hub_parquet_shard_pattern(None, "train") == "train-*.parquet"

    def test_pattern_matches_real_shard_names(self, pipe_module):
        import fnmatch

        pattern = pipe_module.hub_parquet_shard_pattern("default", "train")
        assert fnmatch.filter(
            [
                "default/train-00000-of-00002.parquet",
                "default/train-00001-of-00002.parquet",
                "chat_multiturn/train-00000-of-00001.parquet",
                "default/validation-00000-of-00001.parquet",
                "README.md",
            ],
            pattern,
        ) == ["default/train-00000-of-00002.parquet", "default/train-00001-of-00002.parquet"]


class TestLoadHubDatasetViaArrow:
    """Exercises the real metadata strip and shard concat.

    Only the two Hub calls are stubbed — `list_repo_files` and `hf_hub_download` are the
    network boundary. The parquet files, the pyarrow read and the resulting `Dataset` are real.
    """

    @staticmethod
    def _write_shard(path, rows, with_unreadable_metadata):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"doc_id": rows, "cot_lengths_chars": [[1, 2]] * len(rows)})
        if with_unreadable_metadata:
            # The shape that breaks load_dataset(): a feature type this datasets cannot rebuild.
            table = table.replace_schema_metadata(
                {"huggingface": json.dumps({"info": {"features": {"cot_lengths_chars": {"_type": "List"}}}})}
            )
        pq.write_table(table, path)

    def _patch_hub(self, monkeypatch, pipe_module, shard_map):
        monkeypatch.setattr(
            "huggingface_hub.list_repo_files", lambda *a, **k: [*shard_map, "README.md"], raising=False
        )
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, f, **k: str(shard_map[f]), raising=False)

    def test_strips_metadata_that_would_break_load_dataset(self, pipe_module, monkeypatch, tmp_path):
        shard = tmp_path / "s0.parquet"
        self._write_shard(shard, ["a", "b"], with_unreadable_metadata=True)
        self._patch_hub(monkeypatch, pipe_module, {"default/train-00000-of-00001.parquet": shard})

        ds = pipe_module.load_hub_dataset_via_arrow("org/name", "default", "train", "deadbeef")

        assert len(ds) == 2
        assert ds["doc_id"] == ["a", "b"]

    def test_concatenates_multiple_shards_in_order(self, pipe_module, monkeypatch, tmp_path):
        s0, s1 = tmp_path / "s0.parquet", tmp_path / "s1.parquet"
        self._write_shard(s0, ["a"], with_unreadable_metadata=True)
        self._write_shard(s1, ["b"], with_unreadable_metadata=True)
        self._patch_hub(
            monkeypatch,
            pipe_module,
            {"default/train-00001-of-00002.parquet": s1, "default/train-00000-of-00002.parquet": s0},
        )

        ds = pipe_module.load_hub_dataset_via_arrow("org/name", "default", "train", None)

        assert ds["doc_id"] == ["a", "b"]

    def test_no_matching_shards_raises(self, pipe_module, monkeypatch, tmp_path):
        self._patch_hub(monkeypatch, pipe_module, {"other/train-00000-of-00001.parquet": tmp_path / "x.parquet"})

        with pytest.raises(FileNotFoundError, match="default/train-"):
            pipe_module.load_hub_dataset_via_arrow("org/name", "default", "train", None)


class TestHubLoaderArg:
    def test_defaults_to_datasets(self, pipe_module):
        assert pipe_module.parse_args(["--dataset", "org/name"]).hub_loader == "datasets"

    def test_arrow_accepted(self, pipe_module):
        assert pipe_module.parse_args(["--dataset", "org/name", "--hub-loader", "arrow"]).hub_loader == "arrow"

    def test_unknown_loader_rejected(self, pipe_module):
        with pytest.raises(SystemExit):
            pipe_module.parse_args(["--dataset", "org/name", "--hub-loader", "pandas"])


class TestBuildHubLoadKwargs:
    def test_includes_revision_when_set(self, pipe_module):
        args = pipe_module.parse_args(["--dataset", "org/name", "--revision", "abc123"])
        kwargs = pipe_module.build_hub_load_kwargs(args)
        assert kwargs["revision"] == "abc123"
        assert kwargs["split"] == "train"
        assert kwargs["num_proc"] == args.download_workers
        assert "data_dir" not in kwargs

    def test_omits_revision_when_unset(self, pipe_module):
        args = pipe_module.parse_args(["--dataset", "org/name", "--data-dir", "sub"])
        kwargs = pipe_module.build_hub_load_kwargs(args)
        assert "revision" not in kwargs
        assert kwargs["data_dir"] == "sub"


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
