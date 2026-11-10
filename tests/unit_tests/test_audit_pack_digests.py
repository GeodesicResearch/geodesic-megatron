# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""``scripts/data/audit_pack_digests.py``: a built pack must hold exactly the rows the dataset builder kept.

The documents are packed parquet in the packer's own schema (``write_packed_parquet``), and the expected digests are
computed here independently of the tool, from ``struct`` and ``hashlib``. One class runs the real packer
(``scripts/data/pack_sft_dataset.py``) on a tiny chat corpus and compares its documents with the tokenizer's own chat
render, which is the equality the audit relies on.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from scripts.data import audit_pack_digests as audit
from scripts.data import packed_parquet
from scripts.data.file_digest import file_sha256
from scripts.telemetry.code_revision import code_revision
from scripts.telemetry.gate_outcome import NOT_EVALUATED, exit_status

from megatron.bridge.data.datasets.packed_parquet import write_packed_parquet
from tests.unit_tests.corpora_fixtures import load_campaign_module
from tests.unit_tests.packer_fixtures import (
    ROW_VOCAB,
    ROWS,
    SEQ_LENGTH,
    packed_path,
    row_line,
    run_packer,
    shard_root,
    tokenizer_dir,  # noqa: F401  (a fixture)
    write_exclusions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PAD = 100257
PARQUET_NAME = "packed/training_32768.idx.parquet"


def expected_hash(ids: list[int]) -> int:
    """The digest definition, written out with ``struct`` and ``hashlib`` and nothing of the tool's own."""
    packed = struct.pack(f"<{len(ids)}i", *ids)
    return int.from_bytes(hashlib.blake2b(packed, digest_size=8).digest(), "little", signed=True)


def render(seed: int) -> list[int]:
    """A conversation-shaped render ending in one end-of-text id. Every third one holds the pad id inside as well."""
    rng = np.random.default_rng(seed)
    body = rng.integers(0, 90000, size=int(rng.integers(3, 30))).tolist()
    if seed % 3 == 0:
        body[1] = PAD
    return [*body, PAD]


def stored(item: int | list[int]) -> list[int]:
    """What the packer stores for a render: the render and 0 to 8 more pad ids. A list is stored as it is."""
    if isinstance(item, list):
        return item
    return [*render(item), *[PAD] * (item * 2 % 9)]


RENDERS = [render(seed) for seed in range(12)]
# Two shards of packs, as indices of RENDERS. Document 1 occurs twice, as the mix repeats some by design.
BASE_PACKS = {"shard0": [[0, 1], [1, 2]], "shard1": [[3], [4, 5]]}
BASE_KEPT = [0, 1, 1, 2, 3, 4, 5]


def write_shard(parquet: Path, packs: list[list[int | list[int]]]) -> None:
    """One shard's parquet as the packer writes it: per pack the documents laid end to end and each one's start."""
    rows = []
    for pack in packs:
        ids, starts = [], []
        for item in pack:
            starts.append(len(ids))
            ids.extend(stored(item))
        rows.append({"input_ids": ids, "loss_mask": [0] * len(ids), "seq_start_id": starts})
    parquet.parent.mkdir(parents=True, exist_ok=True)
    write_packed_parquet(rows, parquet, row_group_size=2)


def write_list(path: Path, hashes: list[int]) -> Path:
    path.write_text("".join(f"{value}\n" for value in hashes))
    return path


class Audit:
    """A pack of shards, the lists and a config on disk, and the means to run the command line on them."""

    def __init__(self, tmp_path: Path, packs: dict, kept: list, dropped: list | None = None):
        self.parquets = {name: tmp_path / name / PARQUET_NAME for name in packs}
        for name, shard_packs in packs.items():
            write_shard(self.parquets[name], shard_packs)
        # The lists are in source order, which is not the pack's order.
        digests = [expected_hash(RENDERS[item] if isinstance(item, int) else item) for item in kept]
        self.kept = write_list(
            tmp_path / "kept.txt", [digests[i] for i in np.random.default_rng(7).permutation(len(digests))]
        )
        self.kept_in_file = [int(line) for line in self.kept.read_text().split()]
        self.dropped = (
            None if dropped is None else write_list(tmp_path / "dropped.txt", [expected_hash(r) for r in dropped])
        )
        self.config = tmp_path / "audit.yaml"
        self.config.write_text(
            yaml.safe_dump(
                {
                    "parquets": str(tmp_path / "shard*" / PARQUET_NAME),
                    "pad_token_id": PAD,
                    "kept_digests": str(self.kept),
                    "dropped_digests": None if self.dropped is None else str(self.dropped),
                    "workers": 1,
                }
            )
        )
        self.report_path = tmp_path / "reports" / "report.json"

    def edit(self, **changes) -> None:
        """Rewrite the config with these keys changed, or removed for ``KeyError``."""
        loaded = yaml.safe_load(self.config.read_text())
        for key, value in changes.items():
            if value is KeyError:
                del loaded[key]
            else:
                loaded[key] = value
        self.config.write_text(yaml.safe_dump(loaded))

    def run(self) -> tuple[int, dict | None]:
        """The command line's exit status and its report (None when it wrote none)."""
        status = audit.main(["--config", str(self.config), "--report-out", str(self.report_path)])
        return status, (json.loads(self.report_path.read_text()) if self.report_path.exists() else None)


class TestDigestDefinition:
    def test_the_two_test_vectors_of_the_definition(self):
        assert audit.ids_hash([1, 2, 3]) == -7044357283931317034
        assert audit.ids_hash([100264, 9125, 198, 100257]) == -7694233868375670853

    def test_the_hash_agrees_with_the_written_out_definition_on_any_integer_array(self):
        ids = np.random.default_rng(0).integers(0, 100000, size=5000)
        assert audit.ids_hash(ids) == expected_hash(ids.tolist())
        assert audit.ids_hash(ids.astype(np.int32)) == expected_hash(ids.tolist())
        assert audit.ids_hash(ids[::2]) == expected_hash(ids[::2].tolist())

    def test_the_startup_check_returns_the_vectors_for_the_report(self):
        assert audit.check_digest_definition() == [
            {"ids": [1, 2, 3], "digest": -7044357283931317034},
            {"ids": [100264, 9125, 198, 100257], "digest": -7694233868375670853},
        ]

    def test_a_digest_function_that_fails_the_vectors_stops_the_audit_not_evaluated_before_it_reads_anything(
        self, tmp_path, monkeypatch, capsys
    ):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT)
        monkeypatch.setattr(audit, "ids_digest", lambda ids: b"\x00" * 8)
        status = audit.main(["--config", str(built.config), "--report-out", str(built.report_path)])
        assert status == exit_status([NOT_EVALUATED])
        assert re.search(r"DigestMismatch: ids_hash\(\[1, 2, 3\]\) is 0", capsys.readouterr().err)
        assert not built.report_path.exists()


class TestScanningShards:
    def test_shards_scanned_in_a_pool_equal_shards_scanned_in_this_process(self, tmp_path):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT)
        parquets = tuple(built.parquets[name] for name in BASE_PACKS)
        in_process = audit.scan_shards(parquets, PAD, workers=1)
        pooled = audit.scan_shards(parquets, PAD, workers=2)
        assert [scan.packs for scan in pooled] == [scan.packs for scan in in_process] == [2, 2]
        for scan_pooled, scan_here in zip(pooled, in_process):
            assert scan_pooled.error == scan_here.error is None
            assert np.array_equal(scan_pooled.digests, scan_here.digests)


class TestRenderIds:
    def test_all_trailing_pad_ids_go_and_one_stays(self):
        ids = np.array([5, 6, PAD, PAD, PAD, PAD], dtype="<i4")
        assert audit.render_ids(ids, PAD).tolist() == [5, 6, PAD]

    def test_a_render_ending_in_one_pad_id_is_the_document_itself(self):
        assert audit.render_ids(np.array([5, 6, PAD], dtype="<i4"), PAD).tolist() == [5, 6, PAD]

    def test_pad_ids_inside_the_render_stay(self):
        ids = np.array([5, PAD, 6, PAD, PAD, PAD], dtype="<i4")
        assert audit.render_ids(ids, PAD).tolist() == [5, PAD, 6, PAD]

    def test_a_document_without_a_trailing_pad_id_has_no_render(self):
        with pytest.raises(audit.NoRender, match=f"the last id is 6, not the pad id {PAD}"):
            audit.render_ids(np.array([5, PAD, 6], dtype="<i4"), PAD)

    def test_a_document_of_pad_ids_alone_has_no_render(self):
        with pytest.raises(audit.NoRender, match="render is empty"):
            audit.render_ids(np.array([PAD, PAD], dtype="<i4"), PAD)


class TestIterDocuments:
    def test_each_document_comes_with_its_pack_index_offset_and_length_across_row_groups(self, tmp_path):
        parquet = tmp_path / "shard.parquet"
        write_shard(parquet, [[0, 1], [2], [3, 4, 5], [6], [7, 8]])  # five packs, row groups of two
        documents = list(packed_parquet.iter_documents(parquet, "<i4"))
        assert [(d.pack, d.document) for d in documents] == [
            (0, 0),
            (0, 1),
            (1, 0),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (4, 0),
            (4, 1),
        ]
        assert [d.ids.tolist() for d in documents] == [stored(i) for i in range(9)]
        assert all(d.ids.dtype == np.dtype("<i4") and d.length == len(d.ids) for d in documents)
        assert [d.start for d in documents if d.pack == 2] == [0, len(stored(3)), len(stored(3)) + len(stored(4))]

    def test_an_id_that_does_not_fit_the_type_is_refused_not_wrapped(self, tmp_path):
        parquet = tmp_path / "shard.parquet"
        write_shard(parquet, [[[1, 2**40, PAD]]])
        with pytest.raises(packed_parquet.MalformedShard, match="outside <i4"):
            list(packed_parquet.iter_documents(parquet, "<i4"))
        assert next(packed_parquet.iter_documents(parquet, "<i8")).ids.tolist() == [1, 2**40, PAD]

    def test_the_documents_are_walked_by_the_one_function_the_pack_checks_use(self):
        checks = load_campaign_module("packed_corpus_checks")
        assert checks.MalformedShard is packed_parquet.MalformedShard
        assert checks.iter_documents is packed_parquet.iter_documents

    def test_the_loss_mask_is_read_only_on_request_and_comes_with_each_document(self, tmp_path):
        parquet = tmp_path / "shard.parquet"
        write_shard(parquet, [[0, 1], [2]])
        assert all(d.mask is None for d in packed_parquet.iter_documents(parquet, "<i4"))
        documents = list(packed_parquet.iter_documents(parquet, "<i4", with_mask=True))
        assert [d.mask.tolist() for d in documents] == [[0] * len(stored(i)) for i in range(3)]
        assert all(d.mask.dtype == np.uint8 for d in documents)

    def test_a_loss_mask_of_another_length_than_its_ids_is_refused_only_when_it_is_read(self, tmp_path):
        parquet = tmp_path / "shard.parquet"
        pq.write_table(pa.table({"input_ids": [[1, 2, PAD]], "loss_mask": [[0, 1]], "seq_start_id": [[0]]}), parquet)
        assert len(list(packed_parquet.iter_documents(parquet, "<i4"))) == 1
        with pytest.raises(packed_parquet.MalformedShard, match="loss_mask is not as long as its input_ids"):
            list(packed_parquet.iter_documents(parquet, "<i4", with_mask=True))

    def test_a_pack_without_a_loss_mask_column_is_refused_only_when_it_is_read(self, tmp_path):
        parquet = tmp_path / "shard.parquet"
        pq.write_table(pa.table({"input_ids": [[1, PAD]], "seq_start_id": [[0]]}), parquet)
        assert len(list(packed_parquet.iter_documents(parquet, "<i4"))) == 1
        with pytest.raises(packed_parquet.MalformedShard, match=r"no column \['loss_mask'\]"):
            list(packed_parquet.iter_documents(parquet, "<i4", with_mask=True))


class TestScanShard:
    def test_it_recovers_each_render_and_digests_it(self, tmp_path):
        parquet = tmp_path / "shard.parquet"
        write_shard(parquet, [[0, 1], [2], [3, 4, 5]])
        scan = audit.scan_shard(parquet, PAD)
        assert scan.error is None
        assert scan.digests.tolist() == [expected_hash(RENDERS[i]) for i in range(6)]
        assert (scan.packs, scan.tokens, scan.refused) == (3, sum(len(stored(i)) for i in range(6)), 0)
        assert scan.locations.tolist() == [
            (0, 0, 0, len(stored(0))),
            (0, 1, len(stored(0)), len(stored(1))),
            (1, 0, 0, len(stored(2))),
            (2, 0, 0, len(stored(3))),
            (2, 1, len(stored(3)), len(stored(4))),
            (2, 2, len(stored(3)) + len(stored(4)), len(stored(5))),
        ]

    def test_a_document_without_a_render_is_counted_and_located_and_the_scan_goes_on(self, tmp_path):
        parquet = tmp_path / "shard.parquet"
        write_shard(parquet, [[0, [7, 8, 9]], [[PAD, PAD], 1]])
        scan = audit.scan_shard(parquet, PAD)
        assert scan.error is None
        assert scan.digests.tolist() == [expected_hash(RENDERS[0]), expected_hash(RENDERS[1])]
        assert scan.refused == 2
        assert [(e["pack"], e["document"], e["start"], e["length"]) for e in scan.refused_examples] == [
            (0, 1, len(stored(0)), 3),
            (1, 0, 0, 2),
        ]
        assert "the last id is 9" in scan.refused_examples[0]["reason"]
        assert "render is empty" in scan.refused_examples[1]["reason"]

    def test_a_parquet_that_is_not_a_pack_is_an_error_and_never_skipped(self, tmp_path):
        cases = {
            "no_starts": pa.table({"input_ids": [[1, PAD]]}),
            "not_a_list": pa.table({"input_ids": [1], "seq_start_id": [0]}),
            "float_ids": pa.table({"input_ids": [[1.5, 2.5]], "seq_start_id": [[0]]}),
            "null_id": pa.table({"input_ids": [[1, None, PAD]], "seq_start_id": [[0]]}),
            "starts_not_from_zero": pa.table({"input_ids": [[1, 2, PAD]], "seq_start_id": [[1]]}),
            "starts_not_increasing": pa.table({"input_ids": [[1, 2, PAD]], "seq_start_id": [[0, 2, 2]]}),
            "start_beyond_ids": pa.table({"input_ids": [[1, PAD]], "seq_start_id": [[0, 5]]}),
        }
        for name, table in cases.items():
            parquet = tmp_path / f"{name}.parquet"
            pq.write_table(table, parquet)
            scan = audit.scan_shard(parquet, PAD)
            assert scan.error is not None and scan.error.startswith(f"{parquet}: "), name
            assert (len(scan.digests), scan.packs) == (0, 0), name

    def test_a_file_that_is_not_parquet_is_an_error(self, tmp_path):
        parquet = tmp_path / "shard.parquet"
        parquet.write_text("not parquet")
        assert audit.scan_shard(parquet, PAD).error.startswith(f"{parquet}: ")
        assert audit.scan_shard(tmp_path / "absent.parquet", PAD).error.startswith(f"{tmp_path / 'absent.parquet'}: ")


class TestBeyondOther:
    def test_later_copies_beyond_the_count_in_the_other_array_and_values_it_lacks(self):
        values = np.array([1, 1, 1, 2, 3, 3])
        other = np.array([1, 3, 3, 3])
        beyond = audit.beyond_other(values, other)
        assert beyond.mask.tolist() == [False, True, True, True, False, False]
        assert beyond.absent.tolist() == [False, False, False, True, False, False]


class TestRun:
    def test_a_pack_equal_to_the_kept_list_passes_and_the_report_records_the_inputs(self, tmp_path):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT, dropped=[RENDERS[9]])
        status, report = built.run()
        assert (status, report["verdict"], report["failures"]) == (0, "PASS", [])
        assert report["config"] == {"path": str(built.config.resolve()), "sha256": file_sha256(built.config)}
        assert report["kept_list"] == {"path": str(built.kept), "sha256": file_sha256(built.kept), "digests": 7}
        assert report["dropped_list"] == {
            "path": str(built.dropped),
            "sha256": file_sha256(built.dropped),
            "digests": 1,
        }
        assert report["code_revision"] == code_revision(str(REPO_ROOT))
        assert report["pad_token_id"] == PAD
        assert report["digest_definition"]["test_vectors"] == audit.check_digest_definition()
        assert [shard["parquet"] for shard in report["shards"]] == [
            str(built.parquets[n]) for n in ("shard0", "shard1")
        ]
        assert [(s["packs"], s["documents"], s["refused_documents"]) for s in report["shards"]] == [
            (2, 4, 0),
            (2, 3, 0),
        ]
        assert report["totals"]["documents"] == 7
        comparison = report["comparison"]
        assert (comparison["pack_documents"], comparison["pack_distinct_digests"]) == (7, 6)
        assert (comparison["kept_digests"], comparison["kept_distinct_digests"]) == (7, 6)
        assert comparison["extra_documents"]["count"] == comparison["missing_digests"]["count"] == 0
        assert comparison["dropped_digests_present"] == {"listed": 1, "present": 0, "documents": 0, "examples": []}

    def test_without_a_dropped_list_nothing_is_checked_against_one(self, tmp_path):
        status, report = Audit(tmp_path, BASE_PACKS, BASE_KEPT).run()
        assert (status, report["verdict"]) == (0, "PASS")
        assert report["dropped_list"] is None and report["comparison"]["dropped_digests_present"] is None

    def test_the_parquets_can_be_listed_instead_of_globbed(self, tmp_path):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT)
        listed = [str(built.parquets["shard1"]), str(built.parquets["shard0"])]
        built.edit(parquets=listed)
        status, report = built.run()
        assert status == 0
        assert [shard["parquet"] for shard in report["shards"]] == sorted(listed)

    def test_an_extra_document_fails_and_is_located_by_shard_pack_and_document(self, tmp_path):
        packs = {**BASE_PACKS, "shard1": [[3], [4, 5, 9]]}
        built = Audit(tmp_path, packs, BASE_KEPT)
        status, report = built.run()
        assert (status, report["verdict"]) == (1, "FAIL")
        comparison = report["comparison"]
        assert (comparison["pack_documents"], comparison["kept_digests"]) == (8, 7)
        extras = comparison["extra_documents"]
        assert (extras["count"], extras["digest_not_in_kept_list"], extras["repeated_beyond_kept_list"]) == (1, 1, 0)
        (example,) = extras["examples"]
        assert (example["parquet"], example["pack"], example["document"]) == (str(built.parquets["shard1"]), 1, 2)
        assert (example["digest"], example["in_kept_list"]) == (expected_hash(RENDERS[9]), False)
        assert comparison["missing_digests"]["count"] == 0
        assert report["failures"][0] == "the pack holds 8 documents and the kept list 7 digests"

    def test_a_missing_document_fails_and_is_located_by_its_line_in_the_kept_list(self, tmp_path):
        packs = {**BASE_PACKS, "shard1": [[3], [4]]}
        built = Audit(tmp_path, packs, BASE_KEPT)
        status, report = built.run()
        assert (status, report["verdict"]) == (1, "FAIL")
        missing = report["comparison"]["missing_digests"]
        line = built.kept_in_file.index(expected_hash(RENDERS[5])) + 1
        assert missing == {"count": 1, "examples": [{"kept_line": line, "digest": expected_hash(RENDERS[5])}]}
        assert report["comparison"]["extra_documents"]["count"] == 0
        assert "1 kept digests are missing from the pack" in report["failures"]

    def test_a_document_repeated_beyond_the_list_fails_though_its_digest_is_listed(self, tmp_path):
        packs = {**BASE_PACKS, "shard1": [[3, 3], [4, 5]]}
        status, report = Audit(tmp_path, packs, BASE_KEPT).run()
        assert (status, report["verdict"]) == (1, "FAIL")
        extras = report["comparison"]["extra_documents"]
        assert (extras["count"], extras["digest_not_in_kept_list"], extras["repeated_beyond_kept_list"]) == (1, 0, 1)
        assert extras["examples"][0]["in_kept_list"] is True
        assert report["comparison"]["missing_digests"]["count"] == 0

    def test_a_document_the_list_repeats_but_the_pack_holds_once_fails(self, tmp_path):
        packs = {"shard0": [[0, 1], [2]], "shard1": [[3], [4, 5]]}
        status, report = Audit(tmp_path, packs, BASE_KEPT).run()
        assert (status, report["verdict"]) == (1, "FAIL")
        assert report["comparison"]["missing_digests"]["count"] == 1
        assert report["comparison"]["extra_documents"]["count"] == 0

    def test_a_dropped_digest_in_the_pack_fails_with_its_location(self, tmp_path):
        packs = {**BASE_PACKS, "shard0": [[0, 1], [1, 2], [8]]}
        built = Audit(tmp_path, packs, BASE_KEPT, dropped=[RENDERS[8], RENDERS[9]])
        status, report = built.run()
        assert (status, report["verdict"]) == (1, "FAIL")
        dropped = report["comparison"]["dropped_digests_present"]
        assert (dropped["listed"], dropped["present"], dropped["documents"]) == (2, 1, 1)
        (example,) = dropped["examples"]
        assert (example["parquet"], example["pack"], example["document"]) == (str(built.parquets["shard0"]), 2, 0)
        assert example["digest"] == expected_hash(RENDERS[8])
        assert "1 pack documents carry a digest of the dropped list" in report["failures"]

    def test_a_truncated_document_fails_as_an_extra_and_a_missing_digest(self, tmp_path):
        cut = [*RENDERS[2][:2], PAD]
        packs = {**BASE_PACKS, "shard0": [[0, 1], [1, cut]]}
        status, report = Audit(tmp_path, packs, BASE_KEPT).run()
        assert (status, report["verdict"]) == (1, "FAIL")
        comparison = report["comparison"]
        assert comparison["extra_documents"]["count"] == comparison["missing_digests"]["count"] == 1
        assert comparison["extra_documents"]["examples"][0]["digest"] == expected_hash(cut)

    def test_a_document_cut_off_before_its_end_has_no_render_and_fails_with_its_location(self, tmp_path):
        cut = RENDERS[2][:-1]
        packs = {**BASE_PACKS, "shard0": [[0, 1], [1, cut]]}
        built = Audit(tmp_path, packs, BASE_KEPT)
        status, report = built.run()
        assert (status, report["verdict"]) == (1, "FAIL")
        refused = report["refused_documents"]
        assert refused["count"] == 1
        (example,) = refused["examples"]
        assert (example["parquet"], example["pack"], example["document"]) == (str(built.parquets["shard0"]), 1, 1)
        assert example["length"] == len(cut) and example["reason"].startswith("the last id is ")
        assert report["comparison"]["missing_digests"]["count"] == 1
        assert report["shards"][0]["refused_documents"] == 1

    def test_a_document_that_is_all_padding_fails_even_when_the_lists_match(self, tmp_path):
        packs = {**BASE_PACKS, "shard1": [[3], [4, 5, [PAD, PAD, PAD]]]}
        status, report = Audit(tmp_path, packs, BASE_KEPT).run()
        assert (status, report["verdict"]) == (1, "FAIL")
        assert (
            report["comparison"]["extra_documents"]["count"] == report["comparison"]["missing_digests"]["count"] == 0
        )
        assert report["refused_documents"]["count"] == 1
        assert report["refused_documents"]["examples"][0]["reason"].startswith("every id is the pad id")

    def test_a_malformed_shard_fails_the_audit_and_the_pack_is_not_compared(self, tmp_path):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT)
        pq.write_table(pa.table({"input_ids": [[1, PAD]]}), built.parquets["shard1"])
        status, report = built.run()
        assert (status, report["verdict"]) == (1, "FAIL")
        assert report["comparison"] is None
        assert report["shards"][1]["error"].startswith(f"{built.parquets['shard1']}: no column ['seq_start_id']")
        assert report["shards"][0]["error"] is None and report["shards"][0]["documents"] == 4
        assert any(failure.startswith(f"shard {built.parquets['shard1']}") for failure in report["failures"])
        assert (
            "the pack was not compared with the lists, because a shard could not be read whole" in report["failures"]
        )

    def test_the_first_few_examples_are_listed_and_the_counts_are_complete(self, tmp_path):
        packs = {"shard0": [[0, 1], [1, 2]], "shard1": [[3], [4, 5], *[[[i, PAD]] for i in range(15)]]}
        status, report = Audit(tmp_path, packs, BASE_KEPT).run()
        assert status == 1
        extras = report["comparison"]["extra_documents"]
        assert extras["count"] == 15 and len(extras["examples"]) == audit.EXAMPLES
        assert [e["pack"] for e in extras["examples"]] == list(range(2, 2 + audit.EXAMPLES))

    def test_an_existing_report_is_never_written_over_and_nothing_is_scanned(self, tmp_path, capsys):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT)
        built.report_path.parent.mkdir()
        built.report_path.write_text("an earlier report\n")
        with pytest.raises(SystemExit) as stopped:
            built.run()
        assert stopped.value.code == 2
        assert f"{built.report_path} exists" in capsys.readouterr().err
        assert built.report_path.read_text() == "an earlier report\n"

    def test_the_summary_and_the_failures_are_printed(self, tmp_path, capsys):
        packs = {**BASE_PACKS, "shard1": [[3], [4]]}
        Audit(tmp_path, packs, BASE_KEPT).run()
        captured = capsys.readouterr()
        assert "missing digests: 1" in captured.out and "verdict: FAIL" in captured.out
        assert "1 kept digests are missing from the pack" in captured.err


class TestInputs:
    """Every input fault stops the audit before any shard is read, with the exit status of NOT EVALUATED."""

    def refused(self, built: Audit, capsys, expected: str) -> str:
        """Run the audit, expect it to refuse its inputs with ``expected`` in the message, and return the message."""
        status, report = built.run()
        assert (status, report) == (2, None)
        err = capsys.readouterr().err
        assert "cannot use the spec " in err and expected in err
        return err

    @pytest.mark.parametrize(
        ("changes", "expected"),
        [
            ({"extra_key": 1}, "unknown keys ['extra_key']"),
            ({"workers": KeyError}, "missing keys ['workers']"),
            ({"dropped_digests": KeyError}, "missing keys ['dropped_digests']"),
            ({"parquets": "/absent/*.parquet"}, "no file matches '/absent/*.parquet'"),
            ({"parquets": []}, "must be a glob string or a non-empty list of paths"),
            ({"parquets": ["/absent/shard.parquet"]}, "not a file: ['/absent/shard.parquet']"),
            ({"pad_token_id": -1}, "pad_token_id must be a token id"),
            ({"pad_token_id": True}, "pad_token_id must be a token id"),
            ({"pad_token_id": 2**31}, "does not fit <i4"),
            ({"workers": 0}, "workers must be an integer >= 1"),
            ({"kept_digests": "/absent/kept.txt"}, "no file at /absent/kept.txt"),
            ({"dropped_digests": 5}, "dropped_digests must be a non-empty string"),
        ],
    )
    def test_a_config_that_cannot_be_used_is_refused(self, tmp_path, capsys, changes, expected):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT)
        built.edit(**changes)
        self.refused(built, capsys, expected)

    def test_a_file_listed_twice_is_refused_because_it_would_count_twice(self, tmp_path, capsys):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT)
        path = str(built.parquets["shard0"])
        built.edit(parquets=[path, str(built.parquets["shard1"]), path])
        self.refused(built, capsys, "lists a file twice")

    def test_a_config_that_is_not_yaml_is_refused(self, tmp_path, capsys):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT)
        built.config.write_text("parquets: [unclosed\n")
        self.refused(built, capsys, "cannot read an audit config")

    @pytest.mark.parametrize(
        ("text", "line", "got"),
        [
            ("12\nabc\n13\n", 2, "'abc\\n'"),
            ("12\n13 14\n", 2, "'13 14\\n'"),
            ("12\n13,14\n", 2, "'13,14\\n'"),
            ("12\n\n13\n", 2, "'\\n'"),
            ("12\n13\n9223372036854775808\n", 3, "'9223372036854775808\\n'"),
            ("12\n-9223372036854775809\n", 2, "'-9223372036854775809\\n'"),
            ("12\n1.5\n", 2, "'1.5\\n'"),
        ],
    )
    def test_a_line_that_is_not_one_integer_stops_the_audit_naming_the_file_and_line(
        self, tmp_path, capsys, text, line, got
    ):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT)
        built.kept.write_text(text)
        self.refused(built, capsys, f"{built.kept}:{line}: expected one signed 64-bit integer, got {got}")

    def test_the_dropped_list_is_read_by_the_same_rule(self, tmp_path, capsys):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT, dropped=[RENDERS[9]])
        built.dropped.write_text("-5\n7 8\n")
        self.refused(built, capsys, f"{built.dropped}:2: expected one signed 64-bit integer")

    def test_the_line_number_counts_across_the_chunks_the_list_is_parsed_in(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audit, "LIST_CHUNK_BYTES", 40)
        path = tmp_path / "list.txt"
        path.write_text("".join(f"{n}\n" for n in range(-30, 30)))
        assert audit.read_digest_list(path).tolist() == list(range(-30, 30))
        path.write_text("".join(f"{n}\n" for n in range(40)) + "x\n" + "1\n" * 10)
        with pytest.raises(audit.AuditInputError, match=rf"{path}:41: expected one signed 64-bit integer, got 'x"):
            audit.read_digest_list(path)

    def test_an_empty_kept_list_is_refused_and_an_empty_dropped_list_is_not(self, tmp_path, capsys):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT, dropped=[])
        status, report = built.run()
        assert (status, report["verdict"]) == (0, "PASS")
        assert report["dropped_list"]["digests"] == 0
        built.kept.write_text("")
        built.report_path.unlink()
        self.refused(built, capsys, "the kept list holds no digest")

    def test_a_digest_in_both_lists_is_refused_because_the_pack_could_not_tell_the_rows_apart(self, tmp_path, capsys):
        built = Audit(tmp_path, BASE_PACKS, BASE_KEPT, dropped=[RENDERS[8], RENDERS[2]])
        err = self.refused(built, capsys, f"1 digests are in both {built.kept} and {built.dropped}")
        assert str(expected_hash(RENDERS[2])) in err


class TestRealPacker:
    """The documents the real packer writes are the tokenizer's chat render plus padding, which the audit relies on."""

    LINES = [row_line(i) for i in range(ROWS)] + [row_line(2)]  # row 2 twice, as the mix repeats rows
    EXCLUDED = 5

    @staticmethod
    def packer_flags(pad_seq_to_mult: int = 8) -> tuple[str, ...]:
        """Documents shuffled into packs of several, and padded to a multiple of 8 as the Olmo packs are."""
        return ("--packing-algorithm", "first_fit_shuffle", "--pad-seq-to-mult", str(pad_seq_to_mult))

    @staticmethod
    def renders(tokenizer_dir: Path, lines: list[str]) -> list[list[int]]:
        """The ids of each row's whole chat render with the tokenizer's own template, special tokens not added."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        texts = [tokenizer.apply_chat_template(json.loads(line)["messages"], tokenize=False) for line in lines]
        return [tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts]

    def audit_of(self, tmp_path, tokenizer_dir, root, kept, dropped, pad_seq_to_mult=8):
        """Run the command line on the shard the packer wrote at ``root`` with these kept and dropped renders."""
        parquet = packed_path(root, tokenizer_dir, SEQ_LENGTH, pad_seq_to_mult)
        config = tmp_path / "audit.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "parquets": [str(parquet)],
                    "pad_token_id": ROW_VOCAB["</s>"],
                    "kept_digests": str(write_list(tmp_path / "kept.txt", [expected_hash(ids) for ids in kept])),
                    "dropped_digests": str(
                        write_list(tmp_path / "dropped.txt", [expected_hash(ids) for ids in dropped])
                    ),
                    "workers": 1,
                }
            )
        )
        report_path = tmp_path / "report.json"
        status = audit.main(["--config", str(config), "--report-out", str(report_path)])
        return status, json.loads(report_path.read_text())

    def test_the_packed_documents_are_the_kept_rows_and_the_excluded_row_is_the_dropped_one(
        self, tmp_path, tokenizer_dir
    ):
        root = shard_root(tmp_path, 0, self.LINES)
        exclusions = write_exclusions(tmp_path / "exclude.jsonl", [(0, self.EXCLUDED)])
        flags = [*self.packer_flags(), "--exclude-rows", str(exclusions), "--shard-index", "0"]
        run_packer(tokenizer_dir, root, *flags, seq_length=SEQ_LENGTH)
        renders = self.renders(tokenizer_dir, self.LINES)
        kept = [ids for line, ids in enumerate(renders) if line != self.EXCLUDED]
        status, report = self.audit_of(tmp_path, tokenizer_dir, root, kept, [renders[self.EXCLUDED]])
        assert (status, report["verdict"], report["failures"]) == (0, "PASS", [])
        assert report["comparison"]["pack_documents"] == len(self.LINES) - 1
        assert report["comparison"]["pack_distinct_digests"] == len(self.LINES) - 2
        # The packed documents really carry padding, so that the strip-and-restore is exercised.
        assert report["totals"]["tokens"] > sum(len(ids) for ids in kept)

    def test_a_pack_made_without_the_exclusion_holds_the_dropped_row(self, tmp_path, tokenizer_dir):
        root = shard_root(tmp_path, 0, self.LINES)
        run_packer(tokenizer_dir, root, *self.packer_flags(), seq_length=SEQ_LENGTH)
        renders = self.renders(tokenizer_dir, self.LINES)
        kept = [ids for line, ids in enumerate(renders) if line != self.EXCLUDED]
        status, report = self.audit_of(tmp_path, tokenizer_dir, root, kept, [renders[self.EXCLUDED]])
        assert (status, report["verdict"]) == (1, "FAIL")
        assert report["comparison"]["dropped_digests_present"]["documents"] == 1
        assert report["comparison"]["extra_documents"]["count"] == 1

    @pytest.mark.parametrize(
        ("pad_seq_to_mult", "refused", "extra"),
        [
            # Padding gives the cut document a trailing pad id, so it has a render, but not any kept row's.
            (8, 0, 1),
            # With no padding the cut document ends in a content id and has no render at all.
            (1, 1, 0),
        ],
    )
    def test_a_row_longer_than_the_pack_is_cut_by_the_packer_and_the_audit_fails(
        self, tmp_path, tokenizer_dir, pad_seq_to_mult, refused, extra
    ):
        content = " ".join(["a7"] * (SEQ_LENGTH + 6))
        long_row = json.dumps(
            {"messages": [{"role": "user", "content": "q0"}, {"role": "assistant", "content": content}]}
        )
        lines = [*self.LINES, long_row]
        root = shard_root(tmp_path, 0, lines)
        run_packer(tokenizer_dir, root, *self.packer_flags(pad_seq_to_mult), seq_length=SEQ_LENGTH)
        renders = self.renders(tokenizer_dir, lines)
        assert len(renders[-1]) > SEQ_LENGTH
        status, report = self.audit_of(tmp_path, tokenizer_dir, root, renders, [], pad_seq_to_mult)
        assert (status, report["verdict"]) == (1, "FAIL")
        assert report["refused_documents"]["count"] == refused
        assert report["comparison"]["extra_documents"]["count"] == extra
        assert report["comparison"]["missing_digests"]["count"] == 1
