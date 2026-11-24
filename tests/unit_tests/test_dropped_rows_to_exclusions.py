# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""``scripts/data/dropped_rows_to_exclusions.py``: the builder's dropped rows become the packer's ``--exclude-rows`` file.

Every test runs the real tool on small files in a temporary directory. The last class feeds the tool's output to the
packer's own reader (``pack_sft_dataset.read_excluded_rows``), which is the consumer of the file.
"""

import json
from pathlib import Path

import pytest
import yaml
from scripts.data import dropped_rows_to_exclusions as tool
from scripts.data import pack_sft_dataset
from scripts.data.file_digest import file_sha256


SHARD_LINES = (5, 3, 4)  # three shards of a 12-line mix: starts 0, 5 and 8


def write_offsets(path: Path, lines=SHARD_LINES, total_lines=None) -> Path:
    shards, start = [], 0
    for shard, count in enumerate(lines):
        shards.append({"shard": shard, "start": start, "lines": count, "bytes": 1})
        start += count
    path.write_text(json.dumps({"total_lines": start if total_lines is None else total_lines, "shards": shards}))
    return path


def write_dropped(path: Path, source_rows) -> Path:
    path.write_text("source_row\tshard\tline\n" + "".join(f"{row}\t9\t9\n" for row in source_rows))
    return path


def write_config(tmp_path: Path, source_rows, expected_count=None, **offsets_kwargs) -> Path:
    config = {
        "dropped_rows": str(write_dropped(tmp_path / "dropped.tsv", source_rows)),
        "offsets": str(write_offsets(tmp_path / "offsets.json", **offsets_kwargs)),
        "expected_count": len(source_rows) if expected_count is None else expected_count,
        "out": str(tmp_path / "exclude.jsonl"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def entries(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "exclude.jsonl").read_text().splitlines()]


class TestConversion:
    def test_each_source_row_becomes_its_shard_and_line_in_source_row_order(self, tmp_path):
        config = write_config(tmp_path, [11, 0, 8, 5, 7, 4])
        assert tool.main(["--config", str(config)]) == 0
        assert entries(tmp_path) == [
            {"shard": 0, "line": 0},
            {"shard": 0, "line": 4},
            {"shard": 1, "line": 0},
            {"shard": 1, "line": 2},
            {"shard": 2, "line": 0},
            {"shard": 2, "line": 3},
        ]

    def test_the_record_names_the_inputs_the_output_and_the_rows_per_shard(self, tmp_path):
        config = write_config(tmp_path, [0, 4, 5])
        tool.main(["--config", str(config)])
        record = json.loads((tmp_path / "exclude.jsonl.record.json").read_text())
        assert record["config"]["sha256"] == file_sha256(config)
        assert record["dropped_rows"]["sha256"] == file_sha256(tmp_path / "dropped.tsv")
        assert record["offsets"]["sha256"] == file_sha256(tmp_path / "offsets.json")
        assert record["out"] == {
            "path": str(tmp_path / "exclude.jsonl"),
            "sha256": file_sha256(tmp_path / "exclude.jsonl"),
            "rows": 3,
        }
        assert record["rows_per_shard"] == {"0": 2, "1": 1}
        assert record["code_revision"]

    def test_a_shard_with_no_lines_is_never_chosen(self, tmp_path):
        config = write_config(tmp_path, [2, 3], lines=(3, 0, 2))
        assert tool.main(["--config", str(config)]) == 0
        assert entries(tmp_path) == [{"shard": 0, "line": 2}, {"shard": 2, "line": 0}]


class TestRefusals:
    @pytest.mark.parametrize(
        ("source_rows", "kwargs", "message"),
        [
            ([12], {}, "outside the mix"),
            ([3, 3], {}, "more than once"),
            ([1, 2], {"expected_count": 3}, "expected_count is 3"),
            ([1], {"lines": (5, 3, 4), "total_lines": 13}, "total_lines is 13"),
        ],
    )
    def test_an_input_that_cannot_be_used_is_not_evaluated_and_writes_nothing(
        self, tmp_path, capsys, source_rows, kwargs, message
    ):
        config = write_config(tmp_path, source_rows, **kwargs)
        assert tool.main(["--config", str(config)]) == 2
        assert message in capsys.readouterr().err
        assert not (tmp_path / "exclude.jsonl").exists()

    def test_a_source_row_that_is_not_an_integer_is_refused(self, tmp_path, capsys):
        config = write_config(tmp_path, [1])
        (tmp_path / "dropped.tsv").write_text("source_row\n1.5\n")
        assert tool.main(["--config", str(config)]) == 2
        assert "'1.5' is not a non-negative integer" in capsys.readouterr().err

    def test_a_row_too_short_to_hold_a_source_row_is_refused(self, tmp_path, capsys):
        config = write_config(tmp_path, [1])
        (tmp_path / "dropped.tsv").write_text("shard\tsource_row\n3\n")
        assert tool.main(["--config", str(config)]) == 2
        assert "source_row None is not a non-negative integer" in capsys.readouterr().err

    def test_a_shard_entry_that_is_not_an_object_is_refused(self, tmp_path, capsys):
        config = write_config(tmp_path, [1])
        offsets = json.loads((tmp_path / "offsets.json").read_text())
        offsets["shards"][0] = 7
        (tmp_path / "offsets.json").write_text(json.dumps(offsets))
        assert tool.main(["--config", str(config)]) == 2
        assert "shards[0]: is not an object" in capsys.readouterr().err

    def test_a_dropped_list_without_a_source_row_column_is_refused(self, tmp_path, capsys):
        config = write_config(tmp_path, [1])
        (tmp_path / "dropped.tsv").write_text("row\n1\n")
        assert tool.main(["--config", str(config)]) == 2
        assert "no source_row column" in capsys.readouterr().err

    def test_shards_that_do_not_follow_from_their_line_counts_are_refused(self, tmp_path, capsys):
        config = write_config(tmp_path, [1])
        offsets = json.loads((tmp_path / "offsets.json").read_text())
        offsets["shards"][1]["start"] = 6
        (tmp_path / "offsets.json").write_text(json.dumps(offsets))
        assert tool.main(["--config", str(config)]) == 2
        assert "starts at 6, but the shards before it end at 5" in capsys.readouterr().err

    def test_an_existing_output_is_never_overwritten(self, tmp_path, capsys):
        config = write_config(tmp_path, [1])
        (tmp_path / "exclude.jsonl").write_text("keep me\n")
        assert tool.main(["--config", str(config)]) == 2
        assert "exclude.jsonl exists" in capsys.readouterr().err
        assert (tmp_path / "exclude.jsonl").read_text() == "keep me\n"

    def test_a_config_with_an_unknown_key_is_refused(self, tmp_path, capsys):
        config = write_config(tmp_path, [1])
        loaded = yaml.safe_load(config.read_text())
        loaded["expected_cnt"] = 1
        config.write_text(yaml.safe_dump(loaded))
        assert tool.main(["--config", str(config)]) == 2
        assert "unknown keys ['expected_cnt']" in capsys.readouterr().err


class TestPackerReadsTheFile:
    def test_the_packer_takes_the_lines_of_each_shard_from_the_output(self, tmp_path):
        tool.main(["--config", str(write_config(tmp_path, [0, 4, 5, 7, 8, 11]))])
        out = tmp_path / "exclude.jsonl"
        assert pack_sft_dataset.read_excluded_rows(out, shard_index=0)[0] == [0, 4]
        assert pack_sft_dataset.read_excluded_rows(out, shard_index=1)[0] == [0, 2]
        assert pack_sft_dataset.read_excluded_rows(out, shard_index=2)[0] == [0, 3]
