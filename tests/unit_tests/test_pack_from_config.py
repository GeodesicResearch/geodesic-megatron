# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""``scripts/data/pack_from_config.py``: a shard is packed with the settings a YAML config states, and the config is
kept beside the pack.

The packer runs for real on small JSONL shards (``packer_fixtures``) with a real Hugging Face chat tokenizer built on
disk, so no network and no mock is involved. Row ``i`` of a shard has the question word ``q<i>``, which is the second id
of its document, so the rows a pack holds can be read from its ids.
"""

import hashlib
from pathlib import Path

import pytest
import yaml
from scripts.data import pack_from_config

from pipeline.data import layout
from tests.unit_tests.packer_fixtures import (
    ROW_VOCAB,
    ROWS,
    SEQ_LENGTH,
    pack_manifest,
    packed_path,
    row_line,
    shard_root,
    stored_rows,
    tokenizer_dir,  # noqa: F401  (a fixture)
    write_exclusions,
)


def write_config(base: Path, tokenizer_dir: Path, exclusions: Path | None, **changes) -> Path:
    """A pack config for the shards ``base/shard<n>``, with the settings of ``changes`` replaced."""
    config = {
        "dataset_root": str(base / "shard{shard}"),
        "tokenizer": str(tokenizer_dir),
        "seq_length": SEQ_LENGTH,
        "pad_seq_to_mult": 1,
        "packing_algorithm": "one_per_pack",
        "dataset_kwargs": {},
        "exclude_rows": None if exclusions is None else str(exclusions),
        **changes,
    }
    path = base / "pack.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def pack(config: Path, shard: int) -> None:
    pack_from_config.main(["--config", str(config), "--shard", str(shard)])


def rows_of(parquet: Path) -> list[int]:
    """The rows (line numbers of the shard) a pack of one document per pack holds, in order."""
    question_ids = {ROW_VOCAB[f"q{i}"]: i for i in range(ROWS)}
    return [question_ids[row["input_ids"][1]] for row in stored_rows(parquet)]


def all_rows() -> list[str]:
    return [row_line(i) for i in range(ROWS)]


class TestPackingAShard:
    def test_the_excluded_rows_of_the_shard_stay_out_and_the_manifest_records_the_exclusion_file(
        self, tmp_path, tokenizer_dir
    ):
        root = shard_root(tmp_path, 3, all_rows())
        exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(3, 1), (2, 0), (3, 4)])
        pack(write_config(tmp_path, tokenizer_dir, exclusions), 3)
        parquet = packed_path(root, tokenizer_dir, SEQ_LENGTH)
        assert rows_of(parquet) == [0, 2, 3, 5, 6, 7]
        assert pack_manifest(parquet)["exclude_rows"] == {
            "path": str(exclusions),
            "shard_index": 3,
            "count": 2,
            "sha256": hashlib.sha256(exclusions.read_bytes()).hexdigest(),
        }

    def test_the_shard_number_selects_both_the_dataset_root_and_the_exclusion_entries(self, tmp_path, tokenizer_dir):
        roots = {shard: shard_root(tmp_path, shard, all_rows()) for shard in (2, 5)}
        exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(2, 0), (5, 7)])
        config = write_config(tmp_path, tokenizer_dir, exclusions)
        pack(config, 5)
        pack(config, 2)
        assert rows_of(packed_path(roots[5], tokenizer_dir, SEQ_LENGTH)) == [0, 1, 2, 3, 4, 5, 6]
        assert rows_of(packed_path(roots[2], tokenizer_dir, SEQ_LENGTH)) == [1, 2, 3, 4, 5, 6, 7]

    def test_a_config_without_an_exclusion_file_packs_every_row(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, all_rows())
        pack(write_config(tmp_path, tokenizer_dir, None), 3)
        parquet = packed_path(root, tokenizer_dir, SEQ_LENGTH)
        assert rows_of(parquet) == list(range(ROWS))
        assert "exclude_rows" not in pack_manifest(parquet)

    def test_the_packing_settings_of_the_config_reach_the_packer(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, all_rows())
        config = write_config(
            tmp_path, tokenizer_dir, None, pad_seq_to_mult=8, dataset_kwargs={"pad_to_max_length": True}
        )
        pack(config, 3)
        manifest = pack_manifest(packed_path(root, tokenizer_dir, SEQ_LENGTH, pad_seq_to_mult=8))
        assert manifest["dataset_kwargs"]["pad_to_max_length"] is True
        assert (manifest["packing_algorithm"], manifest["seq_length"]) == ("one_per_pack", SEQ_LENGTH)


class TestTheCopyOfTheConfig:
    def copy_of(self, root: Path, tokenizer_dir: Path) -> Path:
        return layout.pack_dir(root, str(tokenizer_dir), 1) / f"{SEQ_LENGTH}_pack_config.yaml"

    def test_the_config_is_copied_into_the_pack_directory_and_a_second_run_keeps_it(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, all_rows())
        config = write_config(tmp_path, tokenizer_dir, None)
        pack(config, 3)
        copy = self.copy_of(root, tokenizer_dir)
        assert copy.read_bytes() == config.read_bytes()
        pack(config, 3)
        assert copy.read_bytes() == config.read_bytes()

    def test_a_pack_directory_that_holds_another_config_is_refused_and_keeps_it(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, all_rows())
        config = write_config(tmp_path, tokenizer_dir, None)
        pack(config, 3)
        first = config.read_bytes()
        config.write_bytes(first + b"# another config, the same settings\n")
        with pytest.raises(pack_from_config.PackConfigError, match=r"pack_config\.yaml differs from .*pack\.yaml"):
            pack(config, 3)
        assert self.copy_of(root, tokenizer_dir).read_bytes() == first

    def test_another_config_is_refused_before_anything_is_packed(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, all_rows())
        config = write_config(tmp_path, tokenizer_dir, None)
        pack(config, 3)
        parquet = packed_path(root, tokenizer_dir, SEQ_LENGTH)
        packed = parquet.read_bytes()
        # The other config would fail in the packer (a line past the shard's end), so only a check before the packing
        # gives the config error.
        write_config(tmp_path, tokenizer_dir, write_exclusions(tmp_path / "exclude.jsonl", [(3, ROWS)]))
        with pytest.raises(pack_from_config.PackConfigError, match=r"pack_config\.yaml differs from"):
            pack(config, 3)
        assert parquet.read_bytes() == packed

    def test_a_pack_that_fails_leaves_no_copy(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 3, all_rows())
        exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(3, ROWS)])
        with pytest.raises(ValueError, match=f"line {ROWS}, which"):
            pack(write_config(tmp_path, tokenizer_dir, exclusions), 3)
        assert not self.copy_of(root, tokenizer_dir).exists()


class TestTheConfig:
    @pytest.mark.parametrize(
        ("changes", "expected"),
        [
            ({"extra_key": 1}, "unknown keys ['extra_key']"),
            ({"seed": 7}, "unknown keys ['seed']"),
            ({"dataset_root": "relative/shard{shard}"}, "dataset_root must be an absolute path"),
            ({"dataset_root": "/data/corpus"}, "does not hold the placeholder {shard}"),
            ({"tokenizer": ""}, "tokenizer must be a non-empty string"),
            ({"seq_length": 0}, "seq_length must be an integer >= 1"),
            ({"seq_length": True}, "seq_length must be an integer >= 1"),
            ({"seq_length": "64"}, "seq_length must be an integer >= 1"),
            ({"pad_seq_to_mult": 0}, "pad_seq_to_mult must be an integer >= 1"),
            ({"packing_algorithm": None}, "packing_algorithm must be a non-empty string"),
            ({"dataset_kwargs": ["pad_to_max_length"]}, "dataset_kwargs must be a mapping"),
            ({"dataset_kwargs": None}, "dataset_kwargs must be a mapping"),
            ({"exclude_rows": "exclude.jsonl"}, "exclude_rows must be an absolute path"),
        ],
    )
    def test_a_config_that_cannot_be_used_is_refused_before_anything_is_packed(
        self, tmp_path, tokenizer_dir, changes, expected
    ):
        shard_root(tmp_path, 3, all_rows())
        config = write_config(tmp_path, tokenizer_dir, None, **changes)
        with pytest.raises(pack_from_config.PackConfigError, match=expected.replace("[", r"\[").replace("]", r"\]")):
            pack(config, 3)
        assert not (tmp_path / "shard3" / "packed").exists()

    @pytest.mark.parametrize("key", sorted(pack_from_config.CONFIG_KEYS))
    def test_a_config_with_a_key_missing_is_refused_naming_it(self, tmp_path, tokenizer_dir, key):
        config = write_config(tmp_path, tokenizer_dir, None)
        loaded = yaml.safe_load(config.read_text())
        del loaded[key]
        config.write_text(yaml.safe_dump(loaded))
        with pytest.raises(pack_from_config.PackConfigError, match=rf"missing keys \['{key}'\]"):
            pack(config, 3)

    def test_a_file_that_is_not_yaml_is_refused(self, tmp_path):
        config = tmp_path / "pack.yaml"
        config.write_text("dataset_root: [unclosed\n")
        with pytest.raises(pack_from_config.PackConfigError, match="cannot read a pack config"):
            pack(config, 3)

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(pack_from_config.PackConfigError, match="cannot read a pack config"):
            pack(tmp_path / "absent.yaml", 3)


class TestTheCommandLine:
    def test_a_negative_shard_is_refused_before_anything_is_packed(self, tmp_path, tokenizer_dir):
        config = write_config(tmp_path, tokenizer_dir, None)
        with pytest.raises(pack_from_config.PackConfigError, match="--shard must be an integer >= 0"):
            pack(config, -1)
        assert not (tmp_path / "shard-1").exists()

    def test_the_tokenizer_workers_change_the_run_and_not_the_pack(self, tmp_path, tokenizer_dir):
        shard_root(tmp_path, 0, all_rows())
        shard_root(tmp_path, 1, all_rows())
        config = write_config(tmp_path, tokenizer_dir, None)
        pack_from_config.main(["--config", str(config), "--shard", "0"])
        pack_from_config.main(["--config", str(config), "--shard", "1", "--num-tokenizer-workers", "2"])
        assert stored_rows(packed_path(tmp_path / "shard0", tokenizer_dir, SEQ_LENGTH)) == stored_rows(
            packed_path(tmp_path / "shard1", tokenizer_dir, SEQ_LENGTH)
        )

    def test_a_worker_count_below_one_is_refused_before_anything_is_packed(self, tmp_path, tokenizer_dir):
        config = write_config(tmp_path, tokenizer_dir, None)
        with pytest.raises(pack_from_config.PackConfigError, match="--num-tokenizer-workers must be an integer >= 1"):
            pack_from_config.main(["--config", str(config), "--shard", "0", "--num-tokenizer-workers", "0"])
        assert not (tmp_path / "shard0").exists()

    @pytest.mark.parametrize("arguments", [["--shard", "3"], ["--config", "pack.yaml"]])
    def test_the_config_and_the_shard_are_both_required(self, capsys, arguments):
        with pytest.raises(SystemExit) as exit_info:
            pack_from_config.main(arguments)
        assert exit_info.value.code == 2
        assert "required" in capsys.readouterr().err
