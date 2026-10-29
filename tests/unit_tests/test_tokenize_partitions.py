"""Tests for scripts/data/tokenize_partitions.py: the part files of a tokenize run in ordered partitions."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.data import tokenize_partitions as tp


KEY = "input"


def write_corpus(prefix, documents):
    """A .bin/.idx at ``prefix`` holding ``documents`` (lists of token ids), written by Megatron's own builder."""
    import torch
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
    for document in documents:
        if document:  # preprocess_data.py writes no sequence for an empty text, only the document's end
            builder.add_item(torch.tensor(document, dtype=torch.int32))
        builder.end_document()
    builder.finalize(f"{prefix}.idx")


@pytest.fixture
def run(tmp_path):
    """An input JSONL and the paths a partitioned run of it writes."""
    input_jsonl = tmp_path / "training.jsonl"
    input_jsonl.write_text("".join(json.dumps({KEY: f"document {index}"}) + "\n" for index in range(7)))
    return input_jsonl, tmp_path / "tokenized_base"


def split_like_the_tool(input_jsonl, partitions):
    """The part JSONLs ``preprocess_data.py --keep-sequential-samples`` writes: consecutive runs of
    ceil(lines / partitions) lines."""
    lines = input_jsonl.read_text().splitlines(keepends=True)
    size = -(-len(lines) // partitions)
    for part in range(partitions):
        input_jsonl.with_name(f"training_{part}.jsonl").write_text("".join(lines[part * size : (part + 1) * size]))


class TestPartFiles:
    def test_the_files_a_run_writes_are_named_as_the_tool_names_them(self, run):
        input_jsonl, prefix = run
        names = [path.name for path in tp.part_files(input_jsonl, prefix, KEY, 2)]
        assert names == [
            "training_0.jsonl",
            "tokenized_base_0_input_document.bin",
            "tokenized_base_0_input_document.idx",
            "training_1.jsonl",
            "tokenized_base_1_input_document.bin",
            "tokenized_base_1_input_document.idx",
        ]

    def test_leftovers_of_any_partition_count_are_found_and_nothing_else(self, run):
        input_jsonl, prefix = run
        for name in ("training_3.jsonl", "tokenized_base_11_input_document.idx", "tokenized_base_input_document.bin"):
            (input_jsonl.parent / name).write_text("x")
        (input_jsonl.parent / "training.jsonl.partial.000").write_text("x")
        found = [path.name for path in tp.leftover_part_files(input_jsonl, prefix, KEY)]
        assert found == ["tokenized_base_11_input_document.idx", "training_3.jsonl"]

    def test_remove_takes_every_leftover_and_leaves_the_corpus(self, run):
        input_jsonl, prefix = run
        for path in tp.part_files(input_jsonl, prefix, KEY, 3):
            path.write_text("x")
        corpus = input_jsonl.parent / "tokenized_base_input_document.bin"
        corpus.write_text("x")
        assert len(tp.remove_part_files(input_jsonl, prefix, KEY)) == 9
        assert sorted(path.name for path in input_jsonl.parent.iterdir()) == [corpus.name, "training.jsonl"]

    def test_choose_refuses_a_run_that_finds_an_earlier_runs_parts(self, run, tmp_path, capsys):
        input_jsonl, prefix = run
        (input_jsonl.parent / "training_0.jsonl").write_text("x")
        report = tmp_path / "quota"
        report.write_text("1483801186 P /projects/a5k/public/data\n" + TestQuota.REPORT)
        argv = ["choose", "--input", str(input_jsonl), "--output-prefix", str(prefix), "--json-key", KEY]
        assert tp.main([*argv, "--partitions", "4", "--line-percent", "94", "--quota-report", str(report)]) == 1
        assert "training_0.jsonl" in capsys.readouterr().err

    def test_each_step_is_refused_without_what_it_needs(self, run, capsys):
        input_jsonl, prefix = run
        argv = ["--input", str(input_jsonl), "--output-prefix", str(prefix), "--json-key", KEY]
        with pytest.raises(SystemExit):
            tp.main(["choose", *argv, "--partitions", "4", "--line-percent", "94"])
        assert "choose needs --quota-report" in capsys.readouterr().err
        with pytest.raises(SystemExit):
            tp.main(["check", *argv])
        assert "check needs --partitions" in capsys.readouterr().err
        assert tp.main(["remove", *argv]) == 0  # removal takes whatever parts exist, of any count


class TestQuota:
    # The report `lfs quota -p` printed for the project on 2026-10-12; lfs is the cluster's, so its output is read
    # here as text rather than run.
    REPORT = (
        "Disk quotas for prj 1483801186 (pid 1483801186):\n"
        "     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace\n"
        "/projects/a5k/public/data\n"
        "                193110182304       0 214748364800       - 46272912       0 51200000       -\n"
    )

    def test_the_report_is_read_in_bytes(self):
        quota = tp.parse_quota_report("1483801186", self.REPORT)
        assert (quota.used, quota.limit) == (193110182304 * 1024, 214748364800 * 1024)

    def test_the_report_file_the_job_writes_names_the_project_first(self, tmp_path):
        report = tmp_path / "quota"
        report.write_text("1483801186 P /projects/a5k/public/data\n" + self.REPORT)
        assert tp.read_quota_report(report) == tp.Quota("1483801186", 193110182304 * 1024, 214748364800 * 1024)

    def test_a_usage_over_its_soft_quota_is_read_without_its_star(self):
        report = self.REPORT.replace("193110182304 ", "193110182304* ")
        assert tp.parse_quota_report("1", report).used == 193110182304 * 1024

    def test_a_project_without_a_limit_is_refused(self):
        with pytest.raises(ValueError, match="no hard limit"):
            tp.parse_quota_report("1", self.REPORT.replace("214748364800", "0"))

    @pytest.mark.parametrize(
        ("requested", "input_tib", "expected"),
        [(16, 1, 16), (16, 3, 1), (1, 0.1, 1)],
    )
    def test_the_requested_partitions_run_only_when_three_copies_fit_under_the_line(
        self, requested, input_tib, expected
    ):
        # 180 TiB used of 200; the 94% line is 188 TiB, so 8 TiB of room: three copies of 1 TiB fit, of 3 TiB not.
        quota = tp.Quota(project="p", used=180 * 2**40, limit=200 * 2**40)
        chosen, reason = tp.partitions_to_run(requested, int(input_tib * 2**40), quota, 94)
        assert chosen == expected
        assert "180.00 TiB used of 200.00 TiB" in reason


class TestMergeOrderCheck:
    def test_parts_merged_in_order_pass(self, run):
        input_jsonl, prefix = run
        split_like_the_tool(input_jsonl, 3)
        parts = [[[1, 2], [3]], [[4, 5, 6], [7]], [[8]]]
        for part, documents in enumerate(parts):
            write_corpus(f"{prefix}_{part}_{KEY}_document", documents)
        write_corpus(f"{prefix}_{KEY}_document", [document for documents in parts for document in documents])
        result = tp.merge_order_check(input_jsonl, prefix, KEY, 3)
        assert result == {
            "partitions": 3,
            "part_documents": [2, 2, 1],
            "documents": 5,
            "input_bytes": input_jsonl.stat().st_size,
            "result": "pass",
        }

    def test_empty_documents_are_counted_and_placed(self, run):
        input_jsonl, prefix = run
        split_like_the_tool(input_jsonl, 2)
        parts = [[[1, 2], []], [[3], [4, 5]]]
        for part, documents in enumerate(parts):
            write_corpus(f"{prefix}_{part}_{KEY}_document", documents)
        write_corpus(f"{prefix}_{KEY}_document", [[1, 2], [], [3], [4, 5]])
        result = tp.merge_order_check(input_jsonl, prefix, KEY, 2)
        assert (result["part_documents"], result["documents"]) == ([2, 2], 4)

    def test_a_merge_that_drops_an_empty_document_is_refused(self, run):
        # Its sequences are the parts' exactly; only the document boundaries show the missing document.
        input_jsonl, prefix = run
        split_like_the_tool(input_jsonl, 2)
        parts = [[[1, 2], []], [[3], [4, 5]]]
        for part, documents in enumerate(parts):
            write_corpus(f"{prefix}_{part}_{KEY}_document", documents)
        write_corpus(f"{prefix}_{KEY}_document", [[1, 2], [3], [4, 5]])
        with pytest.raises(ValueError, match="3 documents are not its 2 parts' 4, in order"):
            tp.merge_order_check(input_jsonl, prefix, KEY, 2)

    def test_parts_merged_out_of_order_are_refused(self, run):
        input_jsonl, prefix = run
        split_like_the_tool(input_jsonl, 2)
        parts = [[[1, 2]], [[3, 4, 5]]]
        for part, documents in enumerate(parts):
            write_corpus(f"{prefix}_{part}_{KEY}_document", documents)
        write_corpus(f"{prefix}_{KEY}_document", [[3, 4, 5], [1, 2]])
        with pytest.raises(ValueError, match="not its 2 parts' 2, in order"):
            tp.merge_order_check(input_jsonl, prefix, KEY, 2)

    def test_parts_that_do_not_hold_the_whole_input_are_refused(self, run):
        input_jsonl, prefix = run
        split_like_the_tool(input_jsonl, 2)
        input_jsonl.with_name("training_1.jsonl").write_text("")
        with pytest.raises(ValueError, match="the 2 parts hold"):
            tp.merge_order_check(input_jsonl, prefix, KEY, 2)
