# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The partition puts every kept row in exactly one shard: in contiguous byte ranges, or at random from its seed with
the listed rows left out. The corpus table's split job runs it through shard_jsonl_corpus.sh."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from scripts.data import partition_rows


REPO_ROOT = Path(__file__).resolve().parents[2]
ROWS = 50
EXCLUDED = (3, 17)


def write_source(path: Path, rows: int = ROWS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"row": n, "text": "x" * (n % 7)}) + "\n" for n in range(rows)))
    return path


def write_config(path: Path, source: Path, output_root: Path, **values) -> Path:
    """A partition config of ``source`` into ``output_root``: random order with seed 7, four shards, no exclusions and
    the source kept, except where ``values`` say otherwise."""
    content = {
        "source": str(source),
        "output_root": str(output_root),
        "shards": 4,
        "order": "random",
        "seed": 7,
        "exclude_lines": None,
        "remove_source": False,
        **values,
    }
    path.write_text(yaml.safe_dump(content))
    return path


def write_exclusions(path: Path, lines) -> Path:
    path.write_text("".join(json.dumps({"line": line, "tokens": 99}) + "\n" for line in lines))
    return path


def partition(tmp_path: Path, name: str, *, excluded=EXCLUDED, **values) -> Path:
    """Partition the fixture source into ``tmp_path/name`` in random order and return that root."""
    source = tmp_path / "source" / "training.jsonl"
    if not source.exists():
        write_source(source)
    exclude = write_exclusions(tmp_path / f"{name}_exclude.jsonl", excluded)
    config = write_config(tmp_path / f"{name}.yaml", source, tmp_path / name, exclude_lines=str(exclude), **values)
    partition_rows.main(["--config", str(config)])
    return tmp_path / name


def shard_rows(root: Path, shards: int) -> list[list[int]]:
    """The source row numbers each shard holds, in file order."""
    return [
        [json.loads(line)["row"] for line in (root / f"shard{index}" / "training.jsonl").read_text().splitlines()]
        for index in range(shards)
    ]


def test_every_kept_row_is_in_one_shard_in_source_order_and_the_excluded_rows_are_in_none(tmp_path):
    root = partition(tmp_path, "p")
    rows = shard_rows(root, 4)
    assert sorted(row for shard in rows for row in shard) == [n for n in range(ROWS) if n not in EXCLUDED]
    assert all(shard == sorted(shard) for shard in rows)
    assert max(map(len, rows)) - min(map(len, rows)) <= 1
    record = json.loads((root / "partition_record.json").read_text())
    assert record["shard_rows"] == [len(shard) for shard in rows]
    assert (record["source_rows"], record["excluded_rows"]) == (ROWS, len(EXCLUDED))
    assert sum(record["shard_bytes"]) + record["excluded_bytes"] == record["source_bytes"]
    assert (root / "partition_config.yaml").read_bytes() == (tmp_path / "p.yaml").read_bytes()
    assert (tmp_path / "source" / "training.jsonl").exists()


def test_the_seed_decides_the_shards(tmp_path):
    first = shard_rows(partition(tmp_path, "a", seed=7), 4)
    assert shard_rows(partition(tmp_path, "b", seed=7), 4) == first
    assert shard_rows(partition(tmp_path, "c", seed=8), 4) != first


def test_one_shard_holds_every_kept_row(tmp_path):
    assert shard_rows(partition(tmp_path, "one", shards=1, excluded=()), 1) == [list(range(ROWS))]


def test_fewer_kept_rows_than_shards_is_refused(tmp_path):
    with pytest.raises(partition_rows.PartitionError, match="too few for 49 shards"):
        partition(tmp_path, "p", shards=49)


def test_contiguous_order_cuts_equal_byte_ranges_on_line_starts_and_removes_the_source(tmp_path):
    source = write_source(tmp_path / "source" / "training.jsonl")
    content = source.read_bytes()
    starts = [0]
    for line in content.splitlines(keepends=True)[:-1]:
        starts.append(starts[-1] + len(line))
    config = write_config(
        tmp_path / "c.yaml", source, tmp_path / "c", order="contiguous", seed=None, shards=3, remove_source=True
    )
    partition_rows.main(["--config", str(config)])
    rows = shard_rows(tmp_path / "c", 3)
    assert [row for shard in rows for row in shard] == list(range(ROWS))
    # Each line is in the shard of the equal byte range its first byte lies in.
    assert [start * 3 // len(content) for start in starts] == [
        index for index, shard in enumerate(rows) for _ in shard
    ]
    assert (
        b"".join((tmp_path / "c" / f"shard{index}" / "training.jsonl").read_bytes() for index in range(3)) == content
    )
    assert not source.exists()
    record = json.loads((tmp_path / "c" / "partition_record.json").read_text())
    assert (record["source_rows"], record["source_bytes"], record["excluded_rows"]) == (ROWS, len(content), 0)


def test_contiguous_order_refuses_a_shard_that_would_hold_no_line(tmp_path):
    source = tmp_path / "training.jsonl"
    source.write_text('{"long": "' + "x" * 100 + '"}\n{"short": 1}\n')
    with pytest.raises(partition_rows.PartitionError, match="shard 1 of 3 would hold no line"):
        partition_rows.contiguous_cuts(source, 3)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"order": "contiguous", "seed": 1}, "contiguous order takes no seed"),
        ({"order": "contiguous", "seed": None, "exclude_lines": "/x.jsonl"}, "contiguous order takes no seed"),
        ({"order": "sorted"}, "order must be one of"),
        ({"remove_source": "yes"}, "remove_source must be true or false"),
        ({"seed": None}, "seed must be an integer"),
        ({"extra": 1}, "extra"),
    ],
)
def test_a_config_that_its_order_cannot_take_is_refused(tmp_path, values, message):
    config = write_config(tmp_path / "p.yaml", Path("/s.jsonl"), Path("/o"), **values)
    with pytest.raises(partition_rows.PartitionError, match=message):
        partition_rows.read_partition_config(config)


def test_an_output_root_that_holds_a_shard_is_refused(tmp_path):
    partition(tmp_path, "p")
    config = tmp_path / "p.yaml"
    with pytest.raises(partition_rows.PartitionError, match="already holds"):
        partition_rows.main(["--config", str(config)])


@pytest.mark.parametrize(
    ("excluded", "message"),
    [((ROWS,), f"names source line {ROWS}, but the source has {ROWS} lines"), ((4, 4), "duplicate of line 1")],
)
def test_an_exclusion_outside_the_source_or_named_twice_is_refused(tmp_path, excluded, message):
    with pytest.raises(partition_rows.PartitionError, match=message):
        partition(tmp_path, "p", excluded=excluded)


def test_a_line_excluded_twice_with_different_token_counts_is_refused(tmp_path):
    path = tmp_path / "exclude.jsonl"
    path.write_text('{"line": 4, "tokens": 9}\n{"line": 4, "tokens": 10}\n')
    with pytest.raises(partition_rows.PartitionError, match="names source line 4 more than once"):
        partition_rows.read_excluded_lines(path)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ('{"shard": 0, "line": 3}', "exactly the keys"),
        ('{"line": 3}', "exactly the keys"),
        ('{"line": true, "tokens": 5}', "line of"),
    ],
)
def test_an_exclusion_file_of_another_form_is_refused(tmp_path, entry, message):
    """A packer exclusion file names lines of shards, not of the source, so its ``shard`` key is refused."""
    path = tmp_path / "exclude.jsonl"
    path.write_text(entry + "\n")
    with pytest.raises(partition_rows.PartitionError, match=message):
        partition_rows.read_excluded_lines(path)


def test_a_source_whose_last_line_has_no_newline_is_refused(tmp_path):
    source = tmp_path / "training.jsonl"
    source.write_text('{"row": 0}\n{"row": 1}')
    with pytest.raises(partition_rows.PartitionError, match="no final newline"):
        partition_rows.count_lines(source)


def test_the_corpus_split_job_splits_contiguously_and_removes_its_source_and_request(tmp_path):
    root = tmp_path / "corpus"
    source = write_source(root / "training.jsonl")
    content = source.read_bytes()
    # A stand-in for corpus_job.sbatch, which runs the tool in the Apptainer container through pipeline_env_exec.sh:
    # the unit tests run inside that container, which cannot start itself again. The stand-in runs the same tool from
    # this checkout with this interpreter.
    corpus_job = tmp_path / "repo" / "configs" / "control_pretraining" / "corpus_job.sbatch"
    corpus_job.parent.mkdir(parents=True)
    corpus_job.write_text(f'cd "{REPO_ROOT}" && exec "{sys.executable}" "$@"\n')
    done = subprocess.run(
        ["bash", str(REPO_ROOT / "configs" / "control_pretraining" / "shard_jsonl_corpus.sh"), str(root), "3"],
        env={**os.environ, "GEODESIC_REPO_DIR": str(tmp_path / "repo")},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert f"SPLIT_DONE root={root} shards=3" in done.stdout
    assert b"".join((root / f"shard{index}" / "training.jsonl").read_bytes() for index in range(3)) == content
    assert not source.exists()
    assert not (root / "shard_split_request.yaml").exists()
    assert yaml.safe_load((root / "partition_config.yaml").read_text())["order"] == "contiguous"
