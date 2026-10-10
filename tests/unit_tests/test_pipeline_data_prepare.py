"""Unit tests for pipeline_data_prepare.py — focused on the chat-record passthrough,
the per-token decode helper, the VERIFY stage's loss-mask reporting + warning, and the
kwargs assembled for the Hub download.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from tests.unit_tests.corpora_fixtures import write_parquet_dataset
from tests.unit_tests.token_masking_fixtures import build_tiny_hf_tokenizer


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
    @staticmethod
    def _decoder(decode_map):
        return lambda ids: decode_map[int(ids[0])]

    def test_escapes_newline_tab_carriage_return(self, pipe_module):
        decode = self._decoder({1: "\n", 2: "\t", 3: "\r"})
        assert pipe_module._decode_token(decode, 1) == "\\n"
        assert pipe_module._decode_token(decode, 2) == "\\t"
        assert pipe_module._decode_token(decode, 3) == "\\r"

    def test_passes_through_normal_text(self, pipe_module):
        assert pipe_module._decode_token(self._decoder({42: "hello"}), 42) == "hello"

    def test_escapes_mixed_content(self, pipe_module):
        decode = self._decoder({99: "line1\n\tindented"})
        assert pipe_module._decode_token(decode, 99) == "line1\\n\\tindented"


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
def mock_tokenizer(monkeypatch, pipe_module, tmp_path):
    """Replace AutoTokenizer.from_pretrained, which would fetch from the HF Hub, with one returning a real tokenizer
    built offline (ids 3 and 4 are "hello" and "world"; ids outside its tiny vocabulary decode to nothing)."""
    tokenizer = pipe_module.AutoTokenizer.from_pretrained(build_tiny_hf_tokenizer(tmp_path / "tokenizer", None))
    auto = MagicMock()
    auto.from_pretrained.return_value = tokenizer
    monkeypatch.setattr(pipe_module, "AutoTokenizer", auto)
    return tokenizer


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
        assert "hello" in out  # row 0's token 3, decoded by the shared display decoder

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
        assert "streaming" not in kwargs

    def test_streaming_streams_without_workers(self, pipe_module):
        """load_dataset raises NotImplementedError on num_proc with streaming=True, so a stream must not pass it."""
        args = _parse(pipe_module, *_STREAMING_ARGS, "--revision", _PIN, "--data-dir", "sub/dir")
        assert pipe_module.build_hub_load_kwargs(args) == {
            "split": "train",
            "streaming": True,
            "data_dir": "sub/dir",
            "revision": _PIN,
        }

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
        assert captured["config"]["tokenizer_revision"] is None


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


class TestPerSubsetRevisions:
    """`revisions` pins each subset at its own commit: a prepare reads its own subset's pin, and a
    subset the file does not pin is refused rather than read at the default branch's HEAD."""

    PINS = {"first": "1" * 40, "second": "2" * 40}

    def _config(self, tmp_path, **extra):
        return _write_config(tmp_path, yaml.safe_dump({"dataset": "org/corpus", "revisions": self.PINS, **extra}))

    def test_each_subset_reads_its_own_pin(self, pipe_module, tmp_path):
        cfg = self._config(tmp_path)
        for subset, pin in self.PINS.items():
            args = _parse_bare(pipe_module, "--config", cfg, "--subset", subset)
            assert args.revision == pin
            assert pipe_module.build_hub_load_kwargs(args)["revision"] == pin

    def test_a_subset_named_in_the_file_reads_its_pin(self, pipe_module, tmp_path):
        assert _parse_bare(pipe_module, "--config", self._config(tmp_path, subset="second")).revision == "2" * 40

    def test_a_revision_flag_overrides_the_subsets_pin(self, pipe_module, tmp_path):
        args = _parse_bare(
            pipe_module, "--config", self._config(tmp_path), "--subset", "first", "--revision", "3" * 40
        )
        assert args.revision == "3" * 40

    def test_a_streaming_config_streams_at_the_subsets_pin(self, pipe_module, tmp_path):
        """The pin is resolved before the streaming checks, which demand a full commit SHA."""
        options = {"streaming": True, "skip-pack": True, "skip-count": True, "val-proportion": 0}
        args = _parse_bare(pipe_module, "--config", self._config(tmp_path, **options), "--subset", "second")
        assert (args.streaming, args.revision) == (True, "2" * 40)

    @pytest.mark.parametrize(
        ("extra", "argv", "message"),
        [
            ({}, ("--subset", "third"), "`revisions` pins no commit for subset 'third'"),
            ({}, (), "pins each subset's commit (`revisions`), so the subset must be named"),
            ({"revision": "1" * 40}, ("--subset", "first"), "states both `revision` and `revisions`"),
        ],
    )
    def test_refused(self, pipe_module, tmp_path, capsys, extra, argv, message):
        with pytest.raises(SystemExit) as exc:
            _parse_bare(pipe_module, "--config", self._config(tmp_path, **extra), *argv)
        assert exc.value.code == 2
        assert message in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("pins", "message"),
        [
            ({"first": "main"}, "pins ['first'] to something other than a full 40-character SHA"),
            ({"first": "1" * 12}, "pins ['first'] to something other than a full 40-character SHA"),
            ({}, "`revisions` must map each pinned subset to its commit"),
            (["first"], "`revisions` must map each pinned subset to its commit"),
        ],
    )
    def test_malformed_pins_are_refused(self, pipe_module, tmp_path, capsys, pins, message):
        cfg = _write_config(tmp_path, yaml.safe_dump({"dataset": "org/corpus", "revisions": pins}))
        with pytest.raises(SystemExit) as exc:
            _parse_bare(pipe_module, "--config", cfg, "--subset", "first")
        assert exc.value.code == 2
        assert message in capsys.readouterr().err


class TestTokenizerRevision:
    """`tokenizer-revision` pins the tokenizer at a full commit SHA: the prepare loads and records that commit, and
    refuses anything that does not name one commit for good, and packing, whose directory names the tokenizer alone."""

    PIN = "4" * 40

    def _config(self, tmp_path, **extra):
        stated = {"dataset": "org/corpus", "tokenizer": "org/tokenizer", "skip-pack": True, **extra}
        return _write_config(tmp_path, yaml.safe_dump(stated))

    def test_the_pin_reaches_the_arguments(self, pipe_module, tmp_path):
        args = _parse_bare(pipe_module, "--config", self._config(tmp_path, **{"tokenizer-revision": self.PIN}))
        assert (args.tokenizer, args.tokenizer_revision) == ("org/tokenizer", self.PIN)

    def test_no_pin_reads_the_default_branch(self, pipe_module, tmp_path):
        assert _parse_bare(pipe_module, "--config", self._config(tmp_path)).tokenizer_revision is None

    @pytest.mark.parametrize("pin", ["main", "4" * 12, "4" * 39, "G" * 40])
    def test_a_pin_that_is_not_a_full_sha_is_refused(self, pipe_module, tmp_path, capsys, pin):
        with pytest.raises(SystemExit) as exc:
            _parse_bare(pipe_module, "--config", self._config(tmp_path, **{"tokenizer-revision": pin}))
        assert exc.value.code == 2
        assert "--tokenizer-revision must be a full 40-character commit SHA" in capsys.readouterr().err

    def test_a_pinned_tokenizer_cannot_pack(self, pipe_module, tmp_path, capsys):
        cfg = self._config(tmp_path, **{"tokenizer-revision": self.PIN, "skip-pack": False})
        with pytest.raises(SystemExit) as exc:
            _parse_bare(pipe_module, "--config", cfg)
        assert exc.value.code == 2
        assert "--tokenizer-revision cannot pack" in capsys.readouterr().err

    @pytest.mark.parametrize("pinned", [True, False])
    def test_main_records_the_commit_in_the_results_and_wandb(
        self, pipe_module, run_prepare, monkeypatch, tmp_path, pinned
    ):
        """The commit reaches ``pipeline_results.json``, which verify_corpora's ``check_prepared_root`` compares with
        the config, and the W&B config. The tokenizer is a local directory, for which AutoTokenizer ignores the
        revision, so the pin is recorded without a Hub read."""
        dataset = write_parquet_dataset(tmp_path / "corpus", "data", {"text": ["first", "second"]})
        monkeypatch.setattr(pipe_module, "HAS_WANDB", True)
        # wandb.init needs network and credentials; the config main() hands it is what is checked.
        fake_wandb = MagicMock()
        monkeypatch.setattr(pipe_module, "wandb", fake_wandb, raising=False)
        output = tmp_path / "prepared"
        pin = ("--tokenizer-revision", self.PIN) if pinned else ()
        argv = ("--dataset", str(dataset), "--revision", _PIN, "--skip-pack", "--skip-count", "--val-proportion", "0")
        assert run_prepare(*argv, *pin, "--output-dir", str(output), cache=tmp_path / "cache") == 0
        recorded = self.PIN if pinned else None
        assert json.loads((output / "pipeline_results.json").read_text())["tokenizer_revision"] == recorded
        assert fake_wandb.init.call_args.kwargs["config"]["tokenizer_revision"] == recorded


class TestShippedCorpusConfigs:
    """The campaign's corpus definitions must actually load through the real parser."""

    def test_every_shipped_corpus_config_parses(self, pipe_module):
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
        from corpora_table import PACK_GEOMETRY_KEYS, prepare_config_scalars, read_corpora_table, read_select_config

        kind_of_config: dict[Path, str] = {}
        for table in tables:
            for row in read_corpora_table(table):
                kind_of_config[row.config.resolve()] = row.kind

        for path in configs:
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
            if kind == "select":
                # A select row's config names another table's corpus and a kept list; it is
                # never prepared, so it loads through its own parser instead.
                read_select_config(path)
                continue
            # A config that pins each subset at its own commit names no subset of its own: it is
            # prepared once per pinned subset, and each must parse to its own pin.
            pins = stated.get("revisions", {None: None})
            for subset, pin in pins.items():
                argv = ["--config", str(path)] + ([] if subset is None else ["--subset", subset])
                args = _parse_bare(pipe_module, *argv)
                assert args.dataset, f"{path.name} does not name a dataset"
                assert args.revision, f"{path.name} does not pin a revision"
                assert pin is None or args.revision == pin, f"{path.name}: {subset} parsed to {args.revision}"
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


# ── --streaming ─────────────────────────────────────────────────────────────
#
# The datasets under test are local directories, not Hub repositories: the Hub is a network
# boundary (credentials, rate limits, a remote revision) a unit test cannot cross. A local
# directory of parquet files goes through the same packaged parquet builder and, streamed,
# the same IterableDataset a Hub parquet repository does; only where the bytes come from
# differs. load_dataset ignores `revision` for a local path, so the pin the streaming path
# requires is passed and has no effect here.

_PIN = "0123456789abcdef0123456789abcdef01234567"
# What a pretraining corpus build passes (the corpus configs' skip-pack, skip-count, val-proportion 0).
_STREAMING_ARGS = ("--streaming", "--skip-pack", "--skip-count", "--val-proportion", "0")

# Documents chosen to exercise the encoding both exports must share: non-ASCII text, which
# ensure_ascii=False writes raw; quotes, backslashes and control characters, which JSON
# escapes; a U+2028 line separator, which JSON does not escape; an empty document; and enough
# rows to cross the dataset's two files and the 20-row W&B sample.
_DOCUMENTS = [
    'He said "stop" \\ then left.',
    "line one\nline two\ttabbed\r\n",
    "Ünïcödé — 日本語 🙂",
    "",
    "separator inside",
    *(f"document {i}" for i in range(20)),
]


@pytest.fixture
def run_prepare(pipe_module, monkeypatch, tmp_path):
    """Run the real ``main()`` over the given arguments with the datasets cache in a directory of the test's own.

    ``datasets`` reads its cache root from ``datasets.config.HF_DATASETS_CACHE`` when a builder
    is created, so patching it there (the environment variable is read once, at import) is what
    redirects the Arrow cache a loaded dataset writes.
    """
    import datasets

    tokenizer = str(build_tiny_hf_tokenizer(tmp_path / "tokenizer", None))

    def run(*argv: str, cache: Path) -> int:
        monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", cache)
        monkeypatch.setenv("HF_DATASETS_CACHE", str(cache))
        monkeypatch.setattr(
            sys, "argv", ["pipeline_data_prepare.py", "--tokenizer", tokenizer, "--num-proc", "1", *argv]
        )
        return pipe_module.main()

    return run


def _arrow_files(cache: Path) -> list[Path]:
    return sorted(cache.rglob("*.arrow"))


class TestStreamingExport:
    @pytest.fixture
    def exports(self, run_prepare, tmp_path):
        """The same parquet dataset prepared twice, loaded and streamed, each with its own datasets cache."""
        columns = {"text": _DOCUMENTS, "id": list(range(len(_DOCUMENTS)))}
        dataset = write_parquet_dataset(tmp_path / "corpus", "data", columns, files=2)
        common = (
            "--dataset",
            str(dataset),
            "--revision",
            _PIN,
            "--skip-pack",
            "--skip-count",
            "--val-proportion",
            "0",
        )
        loaded, streamed = tmp_path / "loaded", tmp_path / "streamed"
        assert run_prepare(*common, "--no-wandb", "--output-dir", str(loaded), cache=tmp_path / "cache_loaded") == 0
        assert (
            run_prepare(
                *common, "--streaming", "--no-wandb", "--output-dir", str(streamed), cache=tmp_path / "cache_streamed"
            )
            == 0
        )
        return loaded, streamed

    def test_streamed_jsonl_is_byte_identical_to_the_loaded_one(self, exports):
        loaded, streamed = exports
        loaded_bytes = (loaded / "training.jsonl").read_bytes()
        assert (streamed / "training.jsonl").read_bytes() == loaded_bytes
        # Not two equally wrong files: the shared export holds every document, in order, as written.
        lines = loaded_bytes.decode("utf-8").split("\n")
        assert lines[-1] == ""
        assert [json.loads(line) for line in lines[:-1]] == [{"input": doc, "output": ""} for doc in _DOCUMENTS]
        assert "日本語" in loaded_bytes.decode("utf-8")  # ensure_ascii=False: written raw, not \u-escaped
        assert sorted(path.name for path in streamed.iterdir()) == ["pipeline_results.json", "training.jsonl"]

    def test_results_record_matches_the_loaded_one_and_says_streaming(self, exports):
        loaded, streamed = exports
        loaded_results = json.loads((loaded / "pipeline_results.json").read_text())
        streamed_results = json.loads((streamed / "pipeline_results.json").read_text())
        assert list(streamed_results) == list(loaded_results)
        assert (streamed_results["streaming"], loaded_results["streaming"]) == (True, False)
        assert streamed_results["num_documents"] == loaded_results["num_documents"] == len(_DOCUMENTS)
        assert streamed_results["training_docs"] == loaded_results["training_docs"] == len(_DOCUMENTS)
        assert streamed_results["training_jsonl"] == str(streamed / "training.jsonl")
        varying = {"streaming", "output_dir", "training_jsonl", "load_time", "count_time", "export_time"}
        varying |= {"pack_time", "elapsed_time"}
        assert {key: value for key, value in streamed_results.items() if key not in varying} == {
            key: value for key, value in loaded_results.items() if key not in varying
        }
        assert streamed_results["status"] == "completed"
        assert streamed_results["revision"] == _PIN
        assert (streamed_results["validation_jsonl"], streamed_results["validation_docs"]) == (None, 0)

    def test_streaming_writes_no_arrow_cache(self, exports, tmp_path):
        # The loaded export is the control: it shows the check finds the Arrow cache where one is written.
        assert _arrow_files(tmp_path / "cache_loaded"), "the loaded export wrote no Arrow cache; the check is vacuous"
        assert _arrow_files(tmp_path / "cache_streamed") == []

    def test_streamed_wandb_samples_are_the_loaded_ones(self, pipe_module, run_prepare, monkeypatch, tmp_path):
        """The sample table is read from the export's first rows; a stream keeps them as it passes, a Dataset indexes them."""
        dataset = write_parquet_dataset(tmp_path / "corpus", "data", {"text": _DOCUMENTS}, files=2)
        monkeypatch.setattr(pipe_module, "HAS_WANDB", True)
        logged = {}
        for name, extra in (("loaded", ()), ("streamed", ("--streaming",))):
            # wandb.init needs network and credentials; the run object it returns is what main() logs to.
            fake_wandb = MagicMock()
            monkeypatch.setattr(pipe_module, "wandb", fake_wandb, raising=False)
            args = ("--dataset", str(dataset), "--revision", _PIN, "--skip-pack", "--skip-count", *extra)
            assert run_prepare(*args, "--output-dir", str(tmp_path / name), cache=tmp_path / f"cache_{name}") == 0
            logged[name] = [call.args for call in fake_wandb.Table.return_value.add_data.call_args_list]
            assert fake_wandb.init.call_args.kwargs["config"]["streaming"] == bool(extra)
        assert logged["streamed"] == logged["loaded"]
        assert logged["loaded"] == [(i, doc) for i, doc in enumerate(_DOCUMENTS[:20])]

    def test_a_stream_that_fails_part_way_leaves_no_training_jsonl(
        self, pipe_module, run_prepare, monkeypatch, tmp_path
    ):
        """The export writes under a temporary name and renames only once the stream is exhausted, so
        a read that fails hours in leaves the documents written so far under that name, never a
        truncated training.jsonl that the tokenize step would take for the whole corpus."""
        dataset = write_parquet_dataset(tmp_path / "corpus", "data", {"text": _DOCUMENTS}, files=2)
        failing_row = 7
        load = pipe_module.load_dataset

        def interrupted(*args, **kwargs):
            stream = load(*args, **kwargs)

            def read(example, index):
                if index == failing_row:
                    raise ConnectionError("the network read failed part-way")
                return example

            # features restated: a mapped stream otherwise declares none, and the streaming export refuses those
            return stream.map(read, with_indices=True, features=stream.features)

        monkeypatch.setattr(pipe_module, "load_dataset", interrupted)
        out = tmp_path / "out"
        args = ("--dataset", str(dataset), "--revision", _PIN, *_STREAMING_ARGS, "--no-wandb")
        with pytest.raises(ConnectionError, match="failed part-way"):
            run_prepare(*args, "--output-dir", str(out), cache=tmp_path / "cache")
        assert not (out / "training.jsonl").exists()
        assert not (out / "pipeline_results.json").exists()  # nothing records the prepare as done
        # The failure came part-way through the export, after rows were written: not before it opened anything.
        # Split on "\n" only: one document holds a U+2028, which str.splitlines would also split on.
        partial = (out / "training.jsonl.partial").read_text().split("\n")
        assert partial[-1] == ""
        assert [json.loads(line)["input"] for line in partial[:-1]] == _DOCUMENTS[:failing_row]

    def test_text_column_names_the_document(self, run_prepare, tmp_path):
        dataset = write_parquet_dataset(tmp_path / "corpus", "data", {"text": ["t0", "t1"], "content": ["c0", "c1"]})
        out = tmp_path / "out"
        args = ("--dataset", str(dataset), "--revision", _PIN, *_STREAMING_ARGS, "--no-wandb")
        assert run_prepare(*args, "--text-column", "content", "--output-dir", str(out), cache=tmp_path / "cache") == 0
        records = [json.loads(line) for line in (out / "training.jsonl").read_text().splitlines()]
        assert records == [{"input": "c0", "output": ""}, {"input": "c1", "output": ""}]


class TestStreamingRefusesWhatItCannotHonour:
    def test_a_complete_streaming_invocation_parses(self, pipe_module):
        args = _parse(pipe_module, *_STREAMING_ARGS, "--revision", _PIN)
        assert args.streaming is True

    def test_streaming_is_off_by_default(self, pipe_module):
        assert _parse(pipe_module).streaming is False

    def test_streaming_from_the_config(self, pipe_module, tmp_path):
        cfg = _write_config(
            tmp_path,
            f"dataset: org/corpus\nrevision: {_PIN}\nstreaming: true\nskip-pack: true\nskip-count: true\n"
            "val-proportion: 0\n",
        )
        assert _parse_bare(pipe_module, "--config", cfg).streaming is True

    def test_config_streaming_is_checked_like_the_flag(self, pipe_module, tmp_path, capsys):
        cfg = _write_config(tmp_path, f"dataset: org/corpus\nrevision: {_PIN}\nstreaming: true\nskip-count: true\n")
        with pytest.raises(SystemExit) as exc:
            _parse_bare(pipe_module, "--config", cfg)
        assert exc.value.code == 2
        assert "packing is refused" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"--revision": None}, "--revision must pin a full 40-character commit SHA, got None"),
            ({"--revision": "main"}, "--revision must pin a full 40-character commit SHA, got 'main'"),
            ({"--revision": _PIN[:12]}, "--revision must pin a full 40-character commit SHA"),
            ({"--skip-pack": None}, "packing is refused"),
            ({"--skip-count": None}, "the COUNT stage is refused"),
            ({"--count-only": ""}, "--count-only runs only the COUNT stage"),
            ({"--split": "train[0:1000]"}, "--split 'train[0:1000]' is slice or combination syntax"),
            ({"--split": "train[:10%]"}, "is slice or combination syntax"),
            ({"--split": "train+validation"}, "is slice or combination syntax"),
            ({"--data-files": "corpus.jsonl"}, "--data-files loads local files"),
            ({"--join-columns": "title,body"}, "--join-columns rewrites every row"),
            ({"--val-proportion": "0.05"}, "--val-proportion 0.05 needs a random split"),
        ],
    )
    def test_refused_option(self, pipe_module, capsys, change, message):
        options = {"--revision": _PIN, "--skip-pack": "", "--skip-count": "", "--val-proportion": "0"}
        options.update(change)
        argv = ["--streaming"]
        for option, value in options.items():
            if value is not None:
                argv += [option] if value == "" else [option, value]
        with pytest.raises(SystemExit) as exc:
            _parse(pipe_module, *argv)
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "--streaming cannot honour this invocation" in err
        assert message in err

    def test_every_refusal_is_reported_at_once(self, pipe_module, capsys):
        with pytest.raises(SystemExit):
            _parse(pipe_module, "--streaming", "--split", "train[0:10]")
        err = capsys.readouterr().err
        for message in ("--revision must pin", "slice or combination syntax", "COUNT stage", "packing is refused"):
            assert message in err

    @pytest.mark.parametrize(
        ("columns", "extra", "message"),
        [
            ({"text": ["t"], "content": ["c"]}, (), r"columns \['text', 'content'\] could each be the document"),
            (
                {"messages": [[{"role": "user", "content": "hi"}]]},
                (),
                "'messages' is a chat-format column",
            ),
            ({"text": ["t"], "id": [7]}, ("--text-column", "id"), "column 'id' holds Value"),
            ({"text": ["t"]}, ("--text-column", "body"), "Specified --text-column 'body' not found"),
        ],
    )
    def test_refused_column(self, run_prepare, tmp_path, columns, extra, message):
        dataset = write_parquet_dataset(tmp_path / "corpus", "data", columns)
        out = tmp_path / "out"
        args = ("--dataset", str(dataset), "--revision", _PIN, *_STREAMING_ARGS, "--no-wandb", *extra)
        with pytest.raises(ValueError, match=message):
            run_prepare(*args, "--output-dir", str(out), cache=tmp_path / "cache")
        assert not out.exists()  # refused before the export opened anything

    def test_stream_without_declared_features_is_refused(self, run_prepare, tmp_path):
        # A JSON dataset declares no features until it is read, so its columns cannot be checked up front.
        dataset = tmp_path / "corpus"
        dataset.mkdir()
        (dataset / "train.jsonl").write_text(json.dumps({"text": "t"}) + "\n")
        out = tmp_path / "out"
        args = ("--dataset", str(dataset), "--revision", _PIN, *_STREAMING_ARGS, "--no-wandb")
        with pytest.raises(ValueError, match="the stream declares no features"):
            run_prepare(*args, "--output-dir", str(out), cache=tmp_path / "cache")
        assert not out.exists()
