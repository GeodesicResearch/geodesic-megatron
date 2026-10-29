"""Unit tests for pipeline_data_prepare.py and its stages in pipeline.data — focused on the chat-record passthrough,
the per-token decode helper, the VERIFY stage's loss-mask reporting + warning, and the
kwargs assembled for the Hub download.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pipeline.data import load as data_load
from pipeline.data import records as data_records
from pipeline.data import verify as data_verify


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ── format_record ───────────────────────────────────────────────────────────


class TestFormatRecord:
    def test_chat_passthrough_preserves_prefill(self):
        example = {
            "messages": [
                {"role": "system", "content": "sys", "prefill": ""},
                {"role": "user", "content": "u", "prefill": ""},
                {"role": "assistant", "content": "a", "prefill": "\n<stage=training>\n"},
            ]
        }
        out = data_records.format_record(example, "messages", "chat")
        assert out == {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a", "prefill": "\n<stage=training>\n"},
            ]
        }

    def test_chat_drops_empty_and_none(self):
        example = {
            "messages": [
                {"role": "user", "content": "u", "prefill": "", "tool_calls": None, "name": ""},
            ]
        }
        out = data_records.format_record(example, "messages", "chat")
        assert out == {"messages": [{"role": "user", "content": "u"}]}

    def test_chat_passthrough_preserves_tool_calls_and_name(self):
        tool_calls = [{"type": "function", "function": {"name": "calc", "arguments": "{}"}}]
        example = {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": tool_calls, "name": "agent_a"},
            ]
        }
        out = data_records.format_record(example, "messages", "chat")
        assert out["messages"][0]["tool_calls"] == tool_calls
        assert out["messages"][0]["name"] == "agent_a"
        assert "content" not in out["messages"][0]  # empty content dropped

    def test_chat_ignores_unknown_fields(self):
        example = {
            "messages": [
                {"role": "user", "content": "u", "weight": 1.0, "annotation": "x"},
            ]
        }
        out = data_records.format_record(example, "messages", "chat")
        assert out == {"messages": [{"role": "user", "content": "u"}]}

    def test_pretraining_format_unchanged(self):
        example = {"text": "hello world"}
        out = data_records.format_record(example, "text", "pretraining")
        assert out == {"input": "hello world", "output": ""}

    def test_keep_columns_are_copied_into_the_record(self):
        example = {"messages": [{"role": "user", "content": "hi"}], "think_close_kind": "forced", "other": 1}
        out = data_records.format_record(example, "messages", "chat", keep_columns=["think_close_kind"])
        assert out == {"messages": [{"role": "user", "content": "hi"}], "think_close_kind": "forced"}


# ── _decode_token ───────────────────────────────────────────────────────────


class TestDecodeToken:
    def _make_tok(self, decode_map):
        tok = MagicMock()
        tok.decode = lambda ids, skip_special_tokens=False: decode_map[int(ids[0])]
        return tok

    def test_escapes_newline_tab_carriage_return(self):
        tok = self._make_tok({1: "\n", 2: "\t", 3: "\r"})
        assert data_verify._decode_token(tok, 1) == "\\n"
        assert data_verify._decode_token(tok, 2) == "\\t"
        assert data_verify._decode_token(tok, 3) == "\\r"

    def test_passes_through_normal_text(self):
        tok = self._make_tok({42: "hello"})
        assert data_verify._decode_token(tok, 42) == "hello"

    def test_escapes_mixed_content(self):
        tok = self._make_tok({99: "line1\n\tindented"})
        assert data_verify._decode_token(tok, 99) == "line1\\n\\tindented"


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
def mock_tokenizer(monkeypatch):
    """Replace AutoTokenizer.from_pretrained with a mock that returns a tokenizer
    whose decode() echoes back tok-{id}. Avoids hitting HF Hub from unit tests."""
    fake_tok = MagicMock()
    fake_tok.decode = lambda ids, skip_special_tokens=False: f"tok-{int(ids[0])}"
    auto = MagicMock()
    auto.from_pretrained.return_value = fake_tok
    monkeypatch.setattr(data_verify, "AutoTokenizer", auto)
    return fake_tok


class TestVerifyPackedLossMask:
    def test_skipped_when_parquet_missing(self, tmp_path):
        # Don't write the parquet — function should report "skipped_no_parquet".
        result = data_verify.verify_packed_loss_mask(
            output_dir=tmp_path,
            tokenizer_id="dummy/tokenizer",
            seq_length=8,
            pad_seq_to_mult=1,
            format_type="chat",
            wb_run=None,
        )
        assert result["verify_status"] == "skipped_no_parquet"

    def test_skipped_when_parquet_empty(self, tmp_path, mock_tokenizer):
        _write_packed_parquet(tmp_path, "dummy/tokenizer", 8, [], [])
        result = data_verify.verify_packed_loss_mask(
            output_dir=tmp_path,
            tokenizer_id="dummy/tokenizer",
            seq_length=8,
            pad_seq_to_mult=1,
            format_type="chat",
            wb_run=None,
        )
        assert result["verify_status"] == "skipped_empty"

    def test_density_computation_chat_healthy(self, tmp_path, mock_tokenizer, capsys):
        # Two rows: 4 of 8 tokens loss-bearing in row 0; 6 of 8 in row 1. Overall: 10/16 = 62.5%.
        _write_packed_parquet(
            tmp_path,
            "dummy/tokenizer",
            8,
            input_ids_rows=[[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]],
            loss_mask_rows=[[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]],
        )
        result = data_verify.verify_packed_loss_mask(
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

    def test_warning_fires_when_chat_pack_density_100pct(self, tmp_path, mock_tokenizer, capsys):
        # Chat format + all-1s mask is the silent-failure signature.
        _write_packed_parquet(
            tmp_path,
            "dummy/tokenizer",
            4,
            input_ids_rows=[[1, 2, 3, 4]],
            loss_mask_rows=[[1, 1, 1, 1]],
        )
        result = data_verify.verify_packed_loss_mask(
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

    def test_no_warning_for_pretraining_all_ones(self, tmp_path, mock_tokenizer, capsys):
        # Pretraining format with density=1.0 is the design — must not warn.
        _write_packed_parquet(
            tmp_path,
            "dummy/tokenizer",
            4,
            input_ids_rows=[[1, 2, 3, 4]],
            loss_mask_rows=[[1, 1, 1, 1]],
        )
        result = data_verify.verify_packed_loss_mask(
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

    def test_wandb_table_logged_per_row(self, tmp_path, mock_tokenizer, monkeypatch):
        _write_packed_parquet(
            tmp_path,
            "dummy/tokenizer",
            4,
            input_ids_rows=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            loss_mask_rows=[[0, 0, 1, 1]] * 4,
        )

        wb_run = MagicMock()
        # wandb.Table is referenced as `wandb.Table` in the function body, which imports wandb itself.
        fake_wandb = MagicMock()
        fake_wandb.Table.return_value = MagicMock()
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        result = data_verify.verify_packed_loss_mask(
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


# ── load_source_dataset ─────────────────────────────────────────────────────


class _Rejection(Exception):
    pass


@pytest.fixture
def hub(monkeypatch):
    """The Hub's ``load_dataset`` and the retry wait, recorded: the Hub is a network service.

    ``hub.errors`` lists what each successive call raises (an exception) or returns (anything else).
    """
    record = SimpleNamespace(calls=0, sleeps=[], errors=[])

    def load_dataset(*args, **kwargs):
        outcome = record.errors[record.calls]
        record.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(data_load, "load_dataset", load_dataset)
    monkeypatch.setattr(data_load.time, "sleep", record.sleeps.append)
    return record


def test_a_rate_limited_load_waits_and_retries_until_it_succeeds(prepare_module, hub):
    hub.errors = [_Rejection("HTTP 429 Too Many Requests"), _Rejection("429"), "the dataset"]
    assert data_load.load_source_dataset(_parse(prepare_module)) == "the dataset"
    assert (hub.calls, hub.sleeps) == (3, [300, 600])


def test_any_other_load_error_is_raised_at_once(prepare_module, hub):
    hub.errors = [_Rejection("HTTP 404 Not Found")]
    with pytest.raises(_Rejection, match="404"):
        data_load.load_source_dataset(_parse(prepare_module))
    assert (hub.calls, hub.sleeps) == (1, [])


def test_a_rate_limit_that_never_lifts_is_raised_after_every_attempt(prepare_module, hub):
    hub.errors = [_Rejection("429")] * data_load.MAX_LOAD_ATTEMPTS
    with pytest.raises(_Rejection, match="429"):
        data_load.load_source_dataset(_parse(prepare_module))
    assert hub.calls == data_load.MAX_LOAD_ATTEMPTS == 10
    assert hub.sleeps == [300] + [600] * 8


def test_local_files_no_loader_reads_fail_loudly(tmp_path, capsys):
    """The json builder, pandas and json.loads each fail on a file that is not JSON lines; json.loads's error is
    raised, so the line-by-line loader ran last."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    with pytest.raises(json.JSONDecodeError):
        data_load.load_local_files(str(bad), num_proc=1)
    out = capsys.readouterr().out
    assert "HF loader failed" in out and "Pandas also failed" in out


def test_main_reports_a_failed_load_and_returns_1(prepare_module, tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    out_dir = tmp_path / "out"
    argv = [
        "pipeline_data_prepare.py",
        "--dataset",
        "org/corpus",
        "--data-files",
        str(bad),
        "--output-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", [*argv, "--no-wandb", "--num-proc", "1"])
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets_cache"))
    sleeps = []
    monkeypatch.setattr(data_load.time, "sleep", sleeps.append)
    assert prepare_module.main() == 1
    assert "Error loading dataset:" in capsys.readouterr().out
    assert sleeps == [], "a load error that is not a rate limit is not retried"
    assert not out_dir.exists()


# ── build_hub_load_kwargs ───────────────────────────────────────────────────


def _parse_bare(prepare_module, *argv):
    """Run the real argument parser over exactly the given arguments."""
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys, "argv", ["pipeline_data_prepare.py", *argv])
    try:
        return prepare_module.parse_args()
    finally:
        monkey.undo()


def _parse(prepare_module, *extra):
    """Run the real argument parser over a minimal valid command line."""
    return _parse_bare(prepare_module, "--dataset", "org/corpus", *extra)


class TestBuildHubLoadKwargs:
    def test_revision_defaults_to_unpinned(self, prepare_module):
        args = _parse(prepare_module)
        assert args.revision is None
        # Absent, not None: load_dataset must fall through to its own default.
        assert "revision" not in data_load.build_hub_load_kwargs(args)

    def test_revision_is_forwarded_to_load_dataset(self, prepare_module):
        sha = "018376f4b033d7533471514f607cae4de3c95b99"
        args = _parse(prepare_module, "--revision", sha)
        assert data_load.build_hub_load_kwargs(args)["revision"] == sha

    def test_data_dir_absent_unless_set(self, prepare_module):
        assert "data_dir" not in data_load.build_hub_load_kwargs(_parse(prepare_module))
        args = _parse(prepare_module, "--data-dir", "sub/dir")
        assert data_load.build_hub_load_kwargs(args)["data_dir"] == "sub/dir"

    def test_split_and_workers_always_present(self, prepare_module):
        args = _parse(prepare_module, "--split", "validation", "--download-workers", "7")
        kwargs = data_load.build_hub_load_kwargs(args)
        assert kwargs["split"] == "validation"
        assert kwargs["num_proc"] == 7

    def test_revision_recorded_for_provenance(self, prepare_module, tmp_path, monkeypatch):
        """A prepared corpus must carry the revision it was built from."""
        sha = "018376f4b033d7533471514f607cae4de3c95b99"
        args = _parse(prepare_module, "--revision", sha)
        # wandb.init would need network + credentials; the assertion is on the
        # config dict this function builds, which is passed to it verbatim.
        captured = {}

        fake_wandb = MagicMock()
        fake_wandb.init.side_effect = lambda **kw: captured.update(kw) or MagicMock()
        monkeypatch.setattr(prepare_module, "wandb", fake_wandb, raising=False)

        prepare_module.init_wandb(args, "pretraining", tmp_path)
        assert captured["config"]["revision"] == sha


# ── --config ────────────────────────────────────────────────────────────────


def _write_config(tmp_path, body):
    path = tmp_path / "corpus.yaml"
    path.write_text(body)
    return str(path)


class TestPipelineConfig:
    def test_config_supplies_parameters(self, prepare_module, tmp_path):
        cfg = _write_config(tmp_path, "dataset: org/corpus\nsubset: combined\nrevision: abc123\n")
        args = _parse_bare(prepare_module, "--config", cfg)
        assert (args.dataset, args.subset, args.revision) == ("org/corpus", "combined", "abc123")

    def test_command_line_overrides_config(self, prepare_module, tmp_path):
        cfg = _write_config(tmp_path, "dataset: org/corpus\nrevision: from-config\n")
        args = _parse_bare(prepare_module, "--config", cfg, "--revision", "from-cli")
        assert args.revision == "from-cli"
        assert args.dataset == "org/corpus"

    def test_keep_columns_from_config_and_command_line(self, prepare_module, tmp_path):
        cfg = _write_config(tmp_path, "dataset: org/corpus\nkeep-columns: [think_close_kind]\n")
        assert _parse_bare(prepare_module, "--config", cfg).keep_columns == ["think_close_kind"]
        assert _parse_bare(prepare_module, "--dataset", "org/c", "--keep-columns", "a,b").keep_columns == ["a", "b"]

    def test_hyphenated_keys_are_accepted(self, prepare_module, tmp_path):
        cfg = _write_config(tmp_path, "dataset: org/corpus\npad-seq-to-mult: 8\nval-proportion: 0\n")
        args = _parse_bare(prepare_module, "--config", cfg)
        assert args.pad_seq_to_mult == 8
        assert args.val_proportion == 0

    def test_unknown_key_is_rejected(self, prepare_module, tmp_path):
        """A typo must not silently prepare the wrong corpus."""
        cfg = _write_config(tmp_path, "dataset: org/corpus\nrevisoin: abc123\n")
        with pytest.raises(SystemExit):
            _parse_bare(prepare_module, "--config", cfg)

    def test_missing_dataset_is_rejected(self, prepare_module, tmp_path):
        cfg = _write_config(tmp_path, "subset: combined\n")
        with pytest.raises(SystemExit):
            _parse_bare(prepare_module, "--config", cfg)

    def test_config_is_recorded_for_provenance(self, prepare_module, tmp_path):
        cfg = _write_config(tmp_path, "dataset: org/corpus\nrevision: abc123\n")
        args = _parse_bare(prepare_module, "--config", cfg)
        assert args.config == cfg

    def test_packing_options_come_from_the_config(self, prepare_module, tmp_path):
        cfg = _write_config(
            tmp_path,
            "dataset: org/corpus\npacking-algorithm: one_per_pack\n"
            "dataset-kwargs:\n  add_eos: false\n  prompt_template: '{input}{output}'\n",
        )
        args = _parse_bare(prepare_module, "--config", cfg)
        assert args.packing_algorithm == "one_per_pack"
        assert args.dataset_kwargs == {"add_eos": False, "prompt_template": "{input}{output}"}

    def test_dataset_kwargs_parse_as_json_on_the_command_line(self, prepare_module, tmp_path):
        args = _parse_bare(prepare_module, "--dataset", "org/corpus", "--dataset-kwargs", '{"strip_spaces": false}')
        assert args.dataset_kwargs == {"strip_spaces": False}
        assert args.packing_algorithm == "first_fit_shuffle"

    def test_defaults_survive_a_partial_config(self, prepare_module, tmp_path):
        cfg = _write_config(tmp_path, "dataset: org/corpus\n")
        args = _parse_bare(prepare_module, "--config", cfg)
        assert args.split == "train"
        assert args.seq_length == 8192

    def test_empty_config_is_not_an_error(self, prepare_module, tmp_path):
        cfg = _write_config(tmp_path, "# nothing but a comment\n")
        args = _parse_bare(prepare_module, "--config", cfg, "--dataset", "org/corpus")
        assert args.dataset == "org/corpus"

    def test_non_mapping_config_raises(self, prepare_module, tmp_path):
        cfg = _write_config(tmp_path, "- just\n- a list\n")
        with pytest.raises(ValueError, match="must contain a mapping"):
            prepare_module.load_pipeline_config(cfg)


class TestShippedCorpusConfigs:
    """The campaign's corpus definitions must actually load through the real parser."""

    def test_every_shipped_corpus_config_parses(self, prepare_module):
        # Recursive: each campaign arm keeps its corpus definitions in its own data/
        # directory (configs/control_pretraining/30b_baseline/data/, ...), so a glob
        # anchored on the top-level data/ alone would silently skip every arm but the first.
        # Every campaign whose corpora the table tooling builds, found by its arms' corpora tables,
        # so a new campaign is covered without being named here.
        tables = sorted(_REPO_ROOT.glob("configs/*/*/corpora.tsv"))
        campaign_dirs = sorted({table.parents[1] for table in tables})
        assert campaign_dirs, "no campaign corpora tables found under configs/"
        configs = sorted(path for directory in campaign_dirs for path in directory.glob("**/data/*.yaml"))
        for directory in campaign_dirs:
            assert any(path.is_relative_to(directory) for path in configs), f"no corpus configs under {directory}"

        # Which tokenizer a corpus config must name follows the corpus KIND: a .bin/.idx
        # corpus bakes its EOD into the data, a packed SFT corpus renders a chat template.
        # The arms declare the kind of every corpus they build in their corpora.tsv, read
        # here through the module the build and the verifier share. A config no table
        # names (the V1 and CPT corpora) is a .bin/.idx corpus when its prepare writes
        # JSONL only — for those, `skip-pack` is the only signal there is.
        from tests.unit_tests.corpora_fixtures import importable

        importable(_REPO_ROOT / "configs" / "control_pretraining")
        from corpora_table import PACK_GEOMETRY_KEYS, prepare_config_scalars, read_corpora_table

        kind_of_config: dict[Path, str] = {}
        for table in tables:
            for row in read_corpora_table(table):
                kind_of_config[row.config.resolve()] = row.kind

        for path in configs:
            args = _parse_bare(prepare_module, "--config", str(path))
            assert args.dataset, f"{path.name} does not name a dataset"
            assert args.revision, f"{path.name} does not pin a revision"
            # Untabled configs fall back to their pack geometry, NOT to `skip-pack`: that flag
            # says when the pack is built, not whether the corpus is packed at all. A packed
            # SFT corpus whose pack is cut into per-shard jobs sets it too, and reading it as
            # ".bin/.idx" would demand the base tokenizer of a chat corpus. Every `.bin/.idx`
            # config states neither `seq-length` nor `pad-seq-to-mult`; every packed one states
            # both.
            # From the FILE, not from `args`: the flag carries a parser default, so a parsed
            # config always appears to have geometry.
            stated = prepare_config_scalars(path)
            packed_geometry = all(key in stated for key in PACK_GEOMETRY_KEYS)
            kind = kind_of_config.get(path.resolve(), "pack" if packed_geometry else "tokenize")
            if kind == "tokenize":
                # Pretraining-format (.bin/.idx) corpora: the EOD baked into the data must
                # be the base tokenizer's `</s>` = id 2 (CLAUDE.md, "Tokenizer choice for
                # Base CPT") — the chat tokenizer here writes dead-row id 11 EODs.
                assert args.tokenizer == "geodesic-research/nemotron-base-tokenizer", path.name
            else:
                # Packed SFT corpora: the reasoning/think chat-template tokenizer, HISTORY
                # variant — the plain one truncates prior-turn reasoning out of multi-turn
                # conversations before tokenization. The packed path in the training config
                # that reads the pack names the same tokenizer.
                assert args.tokenizer == "geodesic-research/nemotron-think-history-tokenizer", path.name
