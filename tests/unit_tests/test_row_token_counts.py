# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The count pass gives each row's length and digest as the packer tokenizes it, and its kept list is the list the pack
audit compares a pack with."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml
from scripts.data import audit_pack_digests as audit
from scripts.data import partition_rows, row_token_counts

from tests.unit_tests.packer_fixtures import (
    ROW_VOCAB,
    ROWS,
    SEQ_LENGTH,
    packed_path,
    row_line,
    run_packer,
    shard_root,
    stored_rows,
    tokenizer_dir,  # noqa: F401  (a fixture)
    write_exclusions,
)


LONG_ROW = json.dumps(
    {
        "messages": [
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": " ".join(["a7"] * (SEQ_LENGTH + 6))},
        ]
    }
)
LINES = [*(row_line(i) for i in range(ROWS)), LONG_ROW]


def count(
    tmp_path: Path, tokenizer_dir: Path, root: Path, *flags: str, out: str = "counts", dataset_kwargs=None
) -> Path:
    """Run the count pass on ``root`` at ``SEQ_LENGTH`` over the packer's chat defaults (``dataset_kwargs`` merged over
    them) and return its output directory."""
    config = tmp_path / f"{out}.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset_root": str(root),
                "tokenizer": str(tokenizer_dir),
                "seq_length": SEQ_LENGTH,
                "dataset_kwargs": dataset_kwargs or {},
                "out_dir": str(tmp_path / out),
            }
        )
    )
    row_token_counts.main(["--config", str(config), *flags])
    return tmp_path / out


def counted_rows(out_dir: Path) -> list[tuple[int, int, int]]:
    """The ``(line, tokens, digest)`` lines of ``rows.tsv``."""
    lines = (out_dir / "rows.tsv").read_text().splitlines()
    assert lines[0] == "line\ttokens\tdigest"
    return [tuple(int(field) for field in line.split("\t")) for line in lines[1:]]


def test_the_counts_are_the_lengths_and_digests_of_the_documents_the_packer_writes(tmp_path, tokenizer_dir):
    root = shard_root(tmp_path / "data", 0, LINES[:ROWS])
    out = count(tmp_path, tokenizer_dir, root)
    run_packer(tokenizer_dir, root, "--packing-algorithm", "one_per_pack", "--no-validation", seq_length=SEQ_LENGTH)
    documents = [row["input_ids"] for row in stored_rows(packed_path(root, tokenizer_dir, SEQ_LENGTH))]
    rows = counted_rows(out)
    assert [line for line, _, _ in rows] == list(range(ROWS))
    assert Counter((tokens, digest) for _, tokens, digest in rows) == Counter(
        (len(ids), audit.ids_hash(ids)) for ids in documents
    )


def test_a_row_over_the_length_is_listed_and_the_kept_list_audits_a_pack_without_it(tmp_path, tokenizer_dir):
    root = shard_root(tmp_path / "data", 0, LINES)
    out = count(tmp_path, tokenizer_dir, root)
    over = [json.loads(line) for line in (out / "over_length.jsonl").read_text().splitlines()]
    assert [entry["line"] for entry in over] == [ROWS] and over[0]["tokens"] > SEQ_LENGTH
    summary = json.loads((out / "summary.json").read_text())
    assert (summary["rows"], summary["kept_rows"], summary["over_length_rows"]) == (ROWS + 1, ROWS, 1)
    assert (
        summary["tokens"]
        == summary["kept_tokens"] + summary["over_length_tokens"]
        == sum(tokens for _, tokens, _ in counted_rows(out))
    )
    assert (out / "count_config.yaml").read_bytes() == (tmp_path / "counts.yaml").read_bytes()
    assert (out / "over_length_digests.txt").read_text().splitlines() == [str(counted_rows(out)[ROWS][2])]

    exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(0, ROWS)])
    flags = ["--pad-seq-to-mult", "8", "--no-validation", "--exclude-rows", str(exclusions), "--shard-index", "0"]
    run_packer(tokenizer_dir, root, *flags, seq_length=SEQ_LENGTH)
    config = tmp_path / "audit.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "parquets": [str(packed_path(root, tokenizer_dir, SEQ_LENGTH, 8))],
                "pad_token_id": ROW_VOCAB["</s>"],
                "render_ends_in_pad": True,
                "kept_digests": str(out / "kept_digests.txt"),
                "dropped_digests": str(out / "over_length_digests.txt"),
                "workers": 1,
            }
        )
    )
    report_path = tmp_path / "report.json"
    assert audit.main(["--config", str(config), "--report-out", str(report_path)]) == 0
    assert json.loads(report_path.read_text())["verdict"] == "PASS"


def test_a_partition_with_the_count_pass_exclusions_leaves_out_exactly_the_over_length_row(tmp_path, tokenizer_dir):
    root = shard_root(tmp_path / "data", 0, LINES)
    out = count(tmp_path, tokenizer_dir, root)
    config = tmp_path / "partition.yaml"
    partition = {
        "source": str(root / "training.jsonl"),
        "output_root": str(tmp_path / "parts"),
        "shards": 2,
        "order": "random",
        "seed": 3,
        "exclude_lines": str(out / "over_length.jsonl"),
        "remove_source": False,
    }
    config.write_text(yaml.safe_dump(partition))
    partition_rows.main(["--config", str(config)])
    shards = [(tmp_path / "parts" / f"shard{index}" / "training.jsonl").read_text().splitlines() for index in range(2)]
    assert sorted(line for shard in shards for line in shard) == sorted(LINES[:ROWS])


def test_a_source_that_grows_during_the_count_is_refused_and_no_summary_is_written(
    tmp_path, tokenizer_dir, monkeypatch
):
    root = shard_root(tmp_path / "data", 0, LINES[:ROWS])
    source = root / "training.jsonl"
    write_counts = row_token_counts.write_counts

    def write_counts_while_another_process_appends(*args, **kwargs):
        # Stands in for another process appending to the source while the pass runs. A test cannot time a second
        # writer against the pass, so the append happens right after the real counting step has read the source.
        totals = write_counts(*args, **kwargs)
        with source.open("a") as handle:
            handle.write(row_line(ROWS) + "\n")
        return totals

    monkeypatch.setattr(row_token_counts, "write_counts", write_counts_while_another_process_appends)
    with pytest.raises(row_token_counts.CountConfigError, match="changed"):
        count(tmp_path, tokenizer_dir, root)
    assert not (tmp_path / "counts" / row_token_counts.SUMMARY).exists()


def test_tokenizer_processes_give_the_same_counts_as_one(tmp_path, tokenizer_dir):
    root = shard_root(tmp_path / "data", 0, LINES)
    serial = count(tmp_path, tokenizer_dir, root, out="serial")
    pooled = count(tmp_path, tokenizer_dir, root, "--num-tokenizer-workers", "2", out="pooled")
    # The summary and the config copy name their own output directory, so they differ between the two runs.
    for name in set(row_token_counts.OUTPUTS) - {row_token_counts.SUMMARY, "count_config.yaml"}:
        assert (serial / name).read_bytes() == (pooled / name).read_bytes(), name


def test_a_dataset_that_cuts_rows_is_refused_before_anything_is_written(tmp_path, tokenizer_dir):
    """The plain SFT dataset cuts a row to seq_length, so counts taken with it would list no row as over length."""
    root = shard_root(tmp_path / "data", 0, LINES)
    with pytest.raises(ValueError, match="only with the chat dataset"):
        count(tmp_path, tokenizer_dir, root, dataset_kwargs={"chat": False})
    assert not any((tmp_path / "counts").iterdir())


def test_an_output_directory_that_holds_counts_is_refused(tmp_path, tokenizer_dir):
    root = shard_root(tmp_path / "data", 0, LINES[:2])
    count(tmp_path, tokenizer_dir, root)
    with pytest.raises(row_token_counts.CountConfigError, match="already holds"):
        count(tmp_path, tokenizer_dir, root)


def test_a_config_with_an_unknown_key_is_refused(tmp_path):
    config = tmp_path / "count.yaml"
    keys = {key: "/x" for key in row_token_counts.CONFIG_KEYS}
    config.write_text(yaml.safe_dump({**keys, "seq_length": 8, "dataset_kwargs": {}, "extra": 1}))
    with pytest.raises(row_token_counts.CountConfigError, match="extra"):
        row_token_counts.read_count_config(config)
