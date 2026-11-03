# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""``scripts/data/pack_sft_dataset.py --exclude-rows``: the named lines of a shard's training.jsonl stay out of its pack.

The packer runs for real on small JSONL shards (``packer_fixtures``) with a real Hugging Face chat tokenizer built on
disk, so no network and no mock is involved. Row ``i`` tokenizes to ``user q<i> a<i> * (i + 1) </s>``: its tokens say
which row it is, and its length is unique, so a pack of ``one_per_pack`` stores the rows ordered by line number.
"""

import hashlib
import json
from pathlib import Path

import pytest
from scripts.data import pack_sft_dataset

from megatron.bridge.data.datasets.packed_sequence import tokenize_dataset
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from tests.unit_tests.packer_fixtures import (
    ROW_VOCAB,
    ROWS,
    SEQ_LENGTH,
    pack_manifest,
    packed_path,
    row_line,
    run_packer,
    shard_root,
    stored_rows,
    tokenizer_dir,  # noqa: F401  (a fixture)
    write_exclusions,
    write_lines,
)


CHAT_KWARGS = {
    "chat": True,
    "use_hf_tokenizer_chat_template": True,
    "answer_only_loss": True,
    "pad_to_max_length": False,
}
NOT_JSON = "this line is not JSON"


@pytest.fixture(scope="module")
def megatron_tokenizer(tokenizer_dir):
    return build_tokenizer(TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=str(tokenizer_dir)))


def _document(i: int) -> list[int]:
    """The token ids of row ``i``."""
    return [ROW_VOCAB["user"], ROW_VOCAB[f"q{i}"], *[ROW_VOCAB[f"a{i}"]] * (i + 1), ROW_VOCAB["</s>"]]


def _stored(i: int) -> dict:
    """What the packer stores for row ``i`` alone in a pack: the loss mask is rolled one place left, so it ends False."""
    return {
        "input_ids": _document(i),
        "loss_mask": [False] + [True] * (i + 2) + [False],
        "seq_start_id": [0],
    }


def _pack(tokenizer_dir: Path, root: Path, *flags: str) -> None:
    """Run the packer on ``root``; later flags override the default of one document per pack."""
    run_packer(tokenizer_dir, root, "--packing-algorithm", "one_per_pack", *flags, seq_length=SEQ_LENGTH)


def _pack_path(root: Path, tokenizer_dir: Path, split: str = "training") -> Path:
    return packed_path(root, tokenizer_dir, SEQ_LENGTH, split=split)


class TestReadExcludedRows:
    def test_returns_the_lines_of_the_shard_sorted_and_the_digest_of_the_file(self, tmp_path):
        entries = [(1, 7), (0, 5), (1, 2), (2, 0), (1, 9)]
        path = write_exclusions(tmp_path / "exclude.jsonl", entries)
        rows, digest = pack_sft_dataset.read_excluded_rows(path, shard_index=1)
        assert rows == [2, 7, 9]
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_a_shard_with_no_entries_excludes_nothing(self, tmp_path):
        path = write_exclusions(tmp_path / "exclude.jsonl", [(0, 1)])
        rows, _ = pack_sft_dataset.read_excluded_rows(path, shard_index=5)
        assert rows == []

    def test_the_same_line_of_two_shards_is_not_a_duplicate(self, tmp_path):
        path = write_exclusions(tmp_path / "exclude.jsonl", [(0, 3), (1, 3)])
        assert pack_sft_dataset.read_excluded_rows(path, shard_index=1)[0] == [3]

    @pytest.mark.parametrize(
        ("entry", "complaint"),
        [
            ("{", "not JSON"),
            ("[0, 1]", "expected"),
            ('{"shard": 0}', "expected"),
            ('{"shard": 0, "line": 1, "why": "x"}', "expected"),
            ('{"shard": 0, "line": "1"}', "expected"),
            ('{"shard": 0, "line": 1.0}', "expected"),
            ('{"shard": true, "line": 1}', "expected"),
            ('{"shard": 0, "line": -3}', "negative"),
            ("", "not JSON"),
        ],
    )
    def test_a_malformed_or_negative_entry_is_refused_naming_it(self, tmp_path, entry, complaint):
        path = write_lines(tmp_path / "exclude.jsonl", ['{"shard": 0, "line": 1}', entry])
        with pytest.raises(ValueError) as refused:
            pack_sft_dataset.read_excluded_rows(path, shard_index=0)
        message = str(refused.value)
        assert f"{path}:2:" in message
        assert complaint in message
        assert repr(entry) in message

    def test_a_duplicate_entry_is_refused_naming_both_lines(self, tmp_path):
        path = write_exclusions(tmp_path / "exclude.jsonl", [(0, 4), (0, 6), (0, 4)])
        with pytest.raises(ValueError, match=r"exclude\.jsonl:3: duplicate of line 1:") as refused:
            pack_sft_dataset.read_excluded_rows(path, shard_index=0)
        assert '{"shard": 0, "line": 4}' in str(refused.value)

    def test_the_entries_of_other_shards_are_checked_too(self, tmp_path):
        path = write_exclusions(tmp_path / "exclude.jsonl", [(0, 1), (9, -1)])
        with pytest.raises(ValueError, match=r"exclude\.jsonl:2: negative"):
            pack_sft_dataset.read_excluded_rows(path, shard_index=0)

    def test_a_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pack_sft_dataset.read_excluded_rows(tmp_path / "absent.jsonl", shard_index=0)


class TestUsage:
    @pytest.mark.parametrize("flags", [["--exclude-rows", "rows.jsonl"], ["--shard-index", "3"]])
    def test_the_two_options_go_together(self, capsys, flags):
        with pytest.raises(SystemExit) as exit_info:
            pack_sft_dataset.main(["--dataset-root", "root", "--tokenizer", "tok", *flags])
        assert exit_info.value.code == 2
        assert "--exclude-rows and --shard-index" in capsys.readouterr().err


class TestTokenizeDataset:
    """The library side: which rows of the file are tokenized, and in what order."""

    def _tokenize(self, path: Path, tokenizer, exclude_rows, workers: int = 1, dataset_kwargs: dict | None = None):
        return tokenize_dataset(
            path,
            tokenizer,
            SEQ_LENGTH,
            1234,
            dataset_kwargs or dict(CHAT_KWARGS),
            pad_seq_to_mult=1,
            num_tokenizer_workers=workers,
            exclude_rows=exclude_rows,
        )

    def _ids(self, documents) -> list[list[int]]:
        return [list(document["input_ids"]) for document in documents]

    @pytest.mark.parametrize("workers", [1, 2])
    def test_the_other_rows_are_tokenized_in_file_order(self, tmp_path, megatron_tokenizer, workers):
        path = write_lines(tmp_path / "training.jsonl", [row_line(i) for i in range(ROWS)])
        documents = self._tokenize(path, megatron_tokenizer, {4, 1, 7}, workers)
        assert self._ids(documents) == [_document(i) for i in (0, 2, 3, 5, 6)]

    def test_without_exclusions_every_row_is_tokenized(self, tmp_path, megatron_tokenizer):
        path = write_lines(tmp_path / "training.jsonl", [row_line(i) for i in range(ROWS)])
        assert self._ids(self._tokenize(path, megatron_tokenizer, ())) == [_document(i) for i in range(ROWS)]

    def test_an_excluded_row_is_never_read(self, tmp_path, megatron_tokenizer):
        lines = [row_line(0), NOT_JSON, row_line(2)]
        path = write_lines(tmp_path / "training.jsonl", lines)
        assert self._ids(self._tokenize(path, megatron_tokenizer, {1})) == [_document(0), _document(2)]
        with pytest.raises(json.JSONDecodeError):
            self._tokenize(path, megatron_tokenizer, ())

    def test_a_line_the_file_does_not_have_is_refused_naming_it(self, tmp_path, megatron_tokenizer):
        path = write_lines(tmp_path / "training.jsonl", [row_line(i) for i in range(3)])
        with pytest.raises(ValueError, match=rf"line 3, which {path} does not have: it has 3 lines"):
            self._tokenize(path, megatron_tokenizer, {1, 3})

    def test_excluding_every_row_is_refused(self, tmp_path, megatron_tokenizer):
        path = write_lines(tmp_path / "training.jsonl", [row_line(i) for i in range(2)])
        with pytest.raises(ValueError, match="leaves no row"):
            self._tokenize(path, megatron_tokenizer, {0, 1})

    def test_a_dataset_that_maps_its_samples_is_refused(self, tmp_path, megatron_tokenizer):
        # max_num_samples makes the dataset draw its rows through a sample mapping, so an index is no longer a line.
        path = write_lines(tmp_path / "training.jsonl", [row_line(i) for i in range(ROWS)])
        kwargs = {**CHAT_KWARGS, "max_num_samples": ROWS}
        with pytest.raises(ValueError, match="maps its samples"):
            self._tokenize(path, megatron_tokenizer, {1}, dataset_kwargs=kwargs)


class TestPackScript:
    def _expected(self, lines: list[int]) -> list[dict]:
        return [_stored(i) for i in lines]

    def test_without_the_option_every_row_is_packed_and_the_manifest_is_the_old_one(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, [row_line(i) for i in range(ROWS)])
        _pack(tokenizer_dir, root)
        pack = _pack_path(root, tokenizer_dir)
        assert stored_rows(pack) == self._expected(list(range(ROWS)))
        assert pack_manifest(pack) == {
            "packing_algorithm": "one_per_pack",
            "dataset_kwargs": CHAT_KWARGS,
            "seq_length": SEQ_LENGTH,
            "seed": 1234,
        }

    def test_the_named_rows_are_absent_and_the_others_keep_their_order(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, [row_line(i) for i in range(ROWS)])
        exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(3, 6), (2, 0), (3, 1), (4, 5), (3, 3)])
        _pack(tokenizer_dir, root, "--exclude-rows", str(exclusions), "--shard-index", "3")
        assert stored_rows(_pack_path(root, tokenizer_dir)) == self._expected([0, 2, 4, 5, 7])

    def test_the_manifest_records_the_path_the_shard_the_count_and_the_digest_of_the_exclusion_file(
        self, tmp_path, tokenizer_dir
    ):
        root = shard_root(tmp_path, 3, [row_line(i) for i in range(ROWS)])
        exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(3, 6), (2, 0), (3, 1)])
        _pack(tokenizer_dir, root, "--exclude-rows", str(exclusions), "--shard-index", "3")
        recorded = pack_manifest(_pack_path(root, tokenizer_dir))["exclude_rows"]
        assert recorded == {
            "path": str(exclusions),
            "shard_index": 3,
            "count": 2,
            "sha256": hashlib.sha256(exclusions.read_bytes()).hexdigest(),
        }

    def test_the_pack_is_the_pack_of_the_file_without_those_lines(self, tmp_path, tokenizer_dir):
        # Many documents to a pack, shuffled: the old path on a file with the lines removed is the reference.
        excluded = {1, 4, 5}
        with_lines = shard_root(tmp_path / "with", 3, [row_line(i) for i in range(ROWS)])
        without_lines = shard_root(tmp_path / "without", 3, [row_line(i) for i in range(ROWS) if i not in excluded])
        exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(3, line) for line in excluded])
        flags = ["--packing-algorithm", "first_fit_shuffle"]
        _pack(tokenizer_dir, with_lines, *flags, "--exclude-rows", str(exclusions), "--shard-index", "3")
        _pack(tokenizer_dir, without_lines, *flags)
        packed = stored_rows(_pack_path(with_lines, tokenizer_dir))
        assert packed == stored_rows(_pack_path(without_lines, tokenizer_dir))
        assert any(len(row["seq_start_id"]) > 1 for row in packed)

    def test_an_excluded_row_never_reaches_tokenization(self, tmp_path, tokenizer_dir):
        lines = [row_line(0), NOT_JSON, row_line(2)]
        root = shard_root(tmp_path, 0, lines)
        exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(0, 1)])
        _pack(tokenizer_dir, root, "--exclude-rows", str(exclusions), "--shard-index", "0")
        assert stored_rows(_pack_path(root, tokenizer_dir)) == self._expected([0, 2])

    def test_the_validation_split_is_untouched(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, [row_line(i) for i in range(ROWS)], validation=[row_line(i) for i in range(4)])
        exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(3, 1), (3, 2)])
        _pack(tokenizer_dir, root, "--exclude-rows", str(exclusions), "--shard-index", "3")
        validation = _pack_path(root, tokenizer_dir, "validation")
        assert stored_rows(validation) == self._expected([0, 1, 2, 3])
        assert "exclude_rows" not in pack_manifest(validation)

    def test_a_line_beyond_the_file_stops_the_pack_without_writing_one(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, [row_line(i) for i in range(ROWS)])
        exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(3, ROWS)])
        with pytest.raises(ValueError, match=f"line {ROWS}, which"):
            _pack(tokenizer_dir, root, "--exclude-rows", str(exclusions), "--shard-index", "3")
        assert not _pack_path(root, tokenizer_dir).exists()

    def test_a_pack_is_reused_only_for_the_exclusions_it_was_made_with(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, [row_line(i) for i in range(ROWS)])
        first = write_exclusions(tmp_path / "first.jsonl", [(3, 1)])
        second = write_exclusions(tmp_path / "second.jsonl", [(3, 2)])
        _pack(tokenizer_dir, root, "--exclude-rows", str(first), "--shard-index", "3")
        pack = _pack_path(root, tokenizer_dir)
        modified = pack.stat().st_mtime_ns
        _pack(tokenizer_dir, root, "--exclude-rows", str(first), "--shard-index", "3")
        assert pack.stat().st_mtime_ns == modified
        with pytest.raises(ValueError, match="was packed with"):
            _pack(tokenizer_dir, root, "--exclude-rows", str(second), "--shard-index", "3")
        with pytest.raises(ValueError, match="was packed with"):
            _pack(tokenizer_dir, root)

    def test_a_pack_made_without_the_option_is_not_reused_with_it(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, [row_line(i) for i in range(ROWS)])
        exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(3, 1)])
        _pack(tokenizer_dir, root)
        with pytest.raises(ValueError, match="was packed with"):
            _pack(tokenizer_dir, root, "--exclude-rows", str(exclusions), "--shard-index", "3")
