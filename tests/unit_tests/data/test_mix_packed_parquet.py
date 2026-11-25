# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The mix of packed parquet files: real parquet files in a temporary folder, read and written by the real tool.

Each fixture pack carries a marker in its first token, which ``marker_of`` reads, so the source and the index of every
pack of an output can be read from the output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from scripts.data.mix_packed_parquet import MixError, load_mix, mix_files, open_source, plan_mix, read_packs
from scripts.file_digest import file_sha256

from megatron.bridge.data.datasets.packed_parquet import write_packed_parquet
from tests.unit_tests.packed_parquet_fixtures import packed_row


BATCHES = 6
A_PACKS = 12
B_FILES = (10, 14)
A_MARKER, B_MARKER = 1000, 5000


def marker_of(row: dict) -> tuple[str, int]:
    first = row["input_ids"][0]
    return ("a", (first - A_MARKER) // 100) if first < B_MARKER else ("b", (first - B_MARKER) // 100)


@pytest.fixture()
def folder(tmp_path):
    """Source A in one file and source B in two files, whose row groups differ in size."""
    (tmp_path / "a").mkdir()
    write_packed_parquet(
        [packed_row(first_token=A_MARKER + 100 * i) for i in range(A_PACKS)], tmp_path / "a" / "a.idx.parquet", 5
    )
    (tmp_path / "b").mkdir()
    start = 0
    for number, packs in enumerate(B_FILES):
        rows = [packed_row(first_token=B_MARKER + 100 * (start + i)) for i in range(packs)]
        write_packed_parquet(rows, tmp_path / "b" / f"shard_{number}.idx.parquet", (4, 5)[number])
        start += packs
    return tmp_path


def config_of(folder: Path, seed: int = 7, b_draw: str = "sample", b_per_batch: int = 2) -> dict:
    return {
        "output": str(folder / "out" / "mix.idx.parquet"),
        "record": str(folder / "out" / "mix_record.json"),
        "seed": seed,
        "batches": BATCHES,
        "row_group_size": 7,
        "sources": [
            {"name": "a", "path": str(folder / "a" / "*.idx.parquet"), "packs_per_batch": 2, "draw": "all"},
            {"name": "b", "path": str(folder / "b" / "*.idx.parquet"), "packs_per_batch": b_per_batch, "draw": b_draw},
        ],
    }


def write_config(folder: Path, config: dict) -> Path:
    path = folder / f"config_{config['seed']}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def mix_of(folder: Path, **choices):
    return load_mix(write_config(folder, config_of(folder, **choices)))


def output_rows(folder: Path) -> list[dict]:
    return pq.read_table(folder / "out" / "mix.idx.parquet").to_pylist()


def test_every_batch_holds_the_packs_per_batch_of_each_source(folder):
    mix_files(mix_of(folder))
    markers = [marker_of(row) for row in output_rows(folder)]
    assert len(markers) == BATCHES * 4
    for batch in range(BATCHES):
        members = markers[batch * 4 : (batch + 1) * 4]
        assert sorted(source for source, _ in members) == ["a", "a", "b", "b"]


def test_a_source_that_draws_all_keeps_its_file_order_and_uses_each_pack_once(folder):
    mix_files(mix_of(folder))
    a_indices = [index for source, index in map(marker_of, output_rows(folder)) if source == "a"]
    assert sorted(a_indices) == list(range(A_PACKS))
    for batch in range(BATCHES):
        assert sorted(a_indices[batch * 2 : (batch + 1) * 2]) == [batch * 2, batch * 2 + 1]


def test_a_source_that_draws_the_first_packs_keeps_its_file_order_and_leaves_its_last_packs_out(folder):
    assert sum(B_FILES) > BATCHES * 2
    mix_files(mix_of(folder, b_draw="first"))
    b_indices = [index for source, index in map(marker_of, output_rows(folder)) if source == "b"]
    assert sorted(b_indices) == list(range(BATCHES * 2))
    for batch in range(BATCHES):
        assert sorted(b_indices[batch * 2 : (batch + 1) * 2]) == [batch * 2, batch * 2 + 1]


def test_a_source_that_samples_draws_without_replacement_from_all_its_files(folder):
    seen_from_second_file = set()
    for seed in (1, 2, 3):
        for leftover in (folder / "out").glob("*"):
            leftover.unlink()
        mix_files(mix_of(folder, seed=seed))
        b_indices = [index for source, index in map(marker_of, output_rows(folder)) if source == "b"]
        assert len(b_indices) == len(set(b_indices)) == BATCHES * 2
        assert all(0 <= index < sum(B_FILES) for index in b_indices)
        seen_from_second_file |= {index for index in b_indices if index >= B_FILES[0]}
    assert seen_from_second_file


def test_one_seed_gives_one_mix_twice_and_two_seeds_give_two_mixes(folder):
    sources = [open_source(str(folder / "a" / "*.idx.parquet")), open_source(str(folder / "b" / "*.idx.parquet"))]
    first = plan_mix(mix_of(folder, seed=7), sources)
    assert first == plan_mix(mix_of(folder, seed=7), sources)
    assert first != plan_mix(mix_of(folder, seed=8), sources)


def test_a_sample_of_all_the_packs_of_a_source_is_a_permutation_that_its_seed_fixes_and_a_reader_can_rebuild(folder):
    per_batch, seed = 4, 5
    batches = A_PACKS // per_batch
    config = config_of(folder, seed=seed)
    config["batches"] = batches
    config["sources"] = [
        {"name": "a", "path": str(folder / "a" / "*.idx.parquet"), "packs_per_batch": per_batch, "draw": "sample"}
    ]
    mix = load_mix(write_config(folder, config))
    sources = [open_source(mix.sources[0].pattern)]
    mix_files(mix)
    order = [index for _, index in map(marker_of, output_rows(folder))]
    assert sorted(order) == list(range(A_PACKS)) and order != list(range(A_PACKS))
    assert order == [index for _, index in plan_mix(mix, sources)]
    rng = np.random.default_rng(seed)
    drawn = rng.choice(A_PACKS, size=A_PACKS, replace=False)
    rebuilt = []
    for batch in range(batches):
        members = [int(pack) for pack in drawn[batch * per_batch : (batch + 1) * per_batch]]
        rebuilt.extend(members[position] for position in rng.permutation(per_batch))
    assert order == rebuilt
    other = load_mix(write_config(folder, {**config, "seed": seed + 1}))
    assert [index for _, index in plan_mix(other, sources)] != order


def test_the_order_inside_a_batch_is_seeded_and_the_packs_of_a_batch_do_not_change_with_the_seed(folder):
    sources = [open_source(str(folder / "a" / "*.idx.parquet")), open_source(str(folder / "b" / "*.idx.parquet"))]
    runs = {seed: plan_mix(mix_of(folder, seed=seed, b_draw="all", b_per_batch=4), sources) for seed in (1, 2, 3, 4)}
    members = {
        seed: [sorted(order[batch * 6 : (batch + 1) * 6]) for batch in range(BATCHES)] for seed, order in runs.items()
    }
    assert all(members[seed] == members[1] for seed in runs)
    assert len({tuple(order) for order in runs.values()}) > 1


def test_the_output_rows_are_the_source_rows_unchanged(folder):
    mix_files(mix_of(folder))
    a_file = pq.read_table(folder / "a" / "a.idx.parquet").to_pylist()
    b_rows = [
        row for shard in sorted((folder / "b").glob("*.idx.parquet")) for row in pq.read_table(shard).to_pylist()
    ]
    for row in output_rows(folder):
        source, index = marker_of(row)
        assert row == (a_file if source == "a" else b_rows)[index]


def test_the_record_holds_the_hash_of_the_output_and_of_every_source_file_and_the_order(folder):
    record = mix_files(mix_of(folder))
    assert json.loads((folder / "out" / "mix_record.json").read_text()) == record
    assert record["output_sha256"] == file_sha256(folder / "out" / "mix.idx.parquet")
    assert [file["sha256"] for file in record["sources"][1]["files"]] == [
        file_sha256(folder / "b" / f"shard_{number}.idx.parquet") for number in range(2)
    ]
    assert [(name, index) for name, index in record["order"]] == [marker_of(row) for row in output_rows(folder)]


def test_read_packs_reads_across_files_and_row_groups_in_the_order_asked(folder):
    source = open_source(str(folder / "b" / "*.idx.parquet"))
    indices = [23, 0, 10, 9, 4, 17]
    assert [marker_of(row)[1] for row in read_packs(source, indices).to_pylist()] == indices
    with pytest.raises(IndexError):
        source.locate(sum(B_FILES))
    with pytest.raises(IndexError):
        source.locate(-1)


def test_a_mix_that_the_sources_cannot_make_is_refused_and_writes_nothing(folder):
    with pytest.raises(MixError, match="draws all its packs"):
        mix_files(mix_of(folder, b_draw="all"))
    with pytest.raises(MixError, match="the mix draws"):
        mix_files(mix_of(folder, b_per_batch=5))
    with pytest.raises(MixError, match="draws its first"):
        mix_files(mix_of(folder, b_draw="first", b_per_batch=5))
    assert not (folder / "out").exists()


def test_sources_with_different_schemas_are_refused_and_write_nothing(folder):
    other = folder / "c"
    other.mkdir()
    table = pa.table({"input_ids": [[1]] * 24, "loss_mask": [[True]] * 24, "seq_start_id": [[0]] * 24})
    pq.write_table(table, other / "c.idx.parquet")
    config = config_of(folder)
    config["sources"][1]["path"] = str(other / "c.idx.parquet")
    with pytest.raises(MixError, match="different schemas"):
        mix_files(load_mix(write_config(folder, config)))
    assert not (folder / "out").exists()


def test_a_source_that_matches_no_file_is_refused(folder):
    config = config_of(folder)
    config["sources"][0]["path"] = str(folder / "absent" / "*.idx.parquet")
    with pytest.raises(MixError, match="absent"):
        mix_files(load_mix(write_config(folder, config)))


def test_an_existing_output_is_not_overwritten(folder):
    config = mix_of(folder)
    mix_files(config)
    before = file_sha256(config.output)
    with pytest.raises(MixError, match="does not overwrite"):
        mix_files(config)
    assert file_sha256(config.output) == before


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda config: config.update(shuffle=True), "unknown keys"),
        (lambda config: config.update(sources=[]), "at least one source"),
        (lambda config: config.update(batches=0), "batches must be a positive integer"),
        (lambda config: config.update(row_group_size=-1), "row_group_size must be a positive integer"),
        (lambda config: config["sources"][1].update(draw="head"), "draw must be one of"),
        (lambda config: config["sources"][1].update(packs_per_batch=0), "packs_per_batch must be a positive integer"),
        (lambda config: config["sources"][1].update(name="a"), "names must differ"),
        (lambda config: config["sources"][1].update(extra=1), "unknown keys"),
    ],
)
def test_the_config_is_checked(folder, change, message):
    config = config_of(folder)
    change(config)
    with pytest.raises(MixError, match=message):
        load_mix(write_config(folder, config))
