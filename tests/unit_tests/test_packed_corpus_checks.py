# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""configs/control_pretraining/packed_corpus_checks.py on packed parquet written by the packer's own writer
(``write_packed_parquet``) in the packer's schema: containment of a candidate pack's documents in a reference pack, the
absent ids, the failures it must never skip, the report, and the campaign's config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from megatron.bridge.data.datasets.packed_parquet import write_packed_parquet
from tests.unit_tests.corpora_fixtures import (
    DATASET,
    REVISION,
    TOKENIZER,
    load_campaign_module,
    write_prepare_config,
    write_table,
)


corpora_table = load_campaign_module("corpora_table")
corpus_documents = load_campaign_module("corpus_documents")
checks = load_campaign_module("packed_corpus_checks")

REPO_ROOT = Path(__file__).resolve().parents[2]
ABSENT = 500
GEOMETRY = {"seq-length": 32768, "pad-seq-to-mult": 4, "split": "train"}
SHARDS = ("shard0", "shard1")


@dataclass(frozen=True)
class Document:
    """One packed document: its ids and its loss mask, as the packer stores them."""

    ids: tuple[int, ...]
    mask: tuple[bool, ...]


def document(seed: int, *, absent_at: int | None = None) -> Document:
    """A conversation-shaped document: an opening id, 5 to 40 ids, then 1 to 4 pad ids (EOD 11) with the mask off."""
    rng = np.random.default_rng(seed)
    body = rng.integers(1000, 5000, size=int(rng.integers(5, 40))).tolist()
    if absent_at is not None:
        body[absent_at] = ABSENT
    pads = int(rng.integers(1, 5))
    mask = [False, *rng.integers(0, 2, size=len(body)).astype(bool).tolist(), *[False] * pads]
    return Document(ids=(10, *body, *[11] * pads), mask=tuple(mask))


def with_mask_flipped(doc: Document, position: int) -> Document:
    """``doc`` with the loss mask at ``position`` flipped and the ids unchanged."""
    mask = list(doc.mask)
    mask[position] = not mask[position]
    return Document(ids=doc.ids, mask=tuple(mask))


REFERENCE = [document(seed) for seed in range(12)]
# The reference's packs per shard; a candidate repacks its documents in other packs, shards and orders.
REFERENCE_PACKS = {
    "shard0": [[REFERENCE[0], REFERENCE[1]], [REFERENCE[2]], [REFERENCE[3], REFERENCE[4], REFERENCE[5]]],
    "shard1": [[REFERENCE[6], REFERENCE[7]], [REFERENCE[8], REFERENCE[9]], [REFERENCE[10], REFERENCE[11]]],
}
# Ten documents, eight distinct: REFERENCE[0] and REFERENCE[5] twice each.
CANDIDATE_PACKS = {
    "shard0": [[REFERENCE[5], REFERENCE[0]], [REFERENCE[5]], [REFERENCE[11], REFERENCE[2], REFERENCE[7]]],
    "shard1": [[REFERENCE[3]], [REFERENCE[0], REFERENCE[9], REFERENCE[1]]],
}


def write_shard(parquet: Path, packs: list[list[Document]]) -> None:
    """One shard's parquet, as the packer writes it: per pack the documents' ids and masks laid end to end and each
    document's start, in row groups of two packs."""
    rows = []
    for pack in packs:
        starts, ids, mask = [], [], []
        for doc in pack:
            starts.append(len(ids))
            ids.extend(doc.ids)
            mask.extend(doc.mask)
        rows.append({"input_ids": ids, "loss_mask": mask, "seq_start_id": starts})
    parquet.parent.mkdir(parents=True, exist_ok=True)
    write_packed_parquet(rows, parquet, row_group_size=2)


@dataclass(frozen=True)
class Corpora:
    """A candidate and a reference packed corpus, each a two-shard pack row of its own corpora table, and a
    pack-check config naming them."""

    config: Path
    data_base: Path
    candidate: Path  # the corpus roots
    reference: Path
    candidate_scalars: dict
    reference_scalars: dict

    def parquet(self, side: str, shard: str) -> Path:
        root, scalars = (
            (self.candidate, self.candidate_scalars)
            if side == "candidate"
            else (self.reference, self.reference_scalars)
        )
        return corpora_table.packed_parquet_path(root / shard, scalars)

    def write(self, side: str, packs: dict[str, list[list[Document]]]) -> None:
        for shard, shard_packs in packs.items():
            write_shard(self.parquet(side, shard), shard_packs)

    def run(self, tmp_path: Path) -> tuple[int, dict]:
        """Run the tool's command line; its exit status and its report."""
        report = tmp_path / "reports" / "pack_checks.json"
        status = checks.main(
            ["--config", str(self.config), "--report-out", str(report), "--data-base", str(self.data_base)]
        )
        return status, json.loads(report.read_text())


def make_corpora(tmp_path: Path, candidate_tokenizer: str) -> Corpora:
    """Tables and prepare configs for both corpora and the config of the checks, with nothing packed yet. Both prepare
    configs pin ``REVISION``; the reference packs with ``TOKENIZER``, the candidate with ``candidate_tokenizer``."""
    tables, scalars = {}, {}
    for side, tokenizer in (("candidate", candidate_tokenizer), ("reference", TOKENIZER)):
        directory = tmp_path / side
        directory.mkdir()
        prepare = directory / "corpus.yaml"
        prepare.write_text(
            yaml.safe_dump(
                {"dataset": DATASET, "revision": REVISION, "tokenizer": tokenizer, **GEOMETRY, "val-proportion": 0}
            )
        )
        tables[side] = write_table(
            directory, prepare, subset=f"{side}_mix", stage="sft", kind="pack", shards=2, shard_mode="split", docs=12
        )
        scalars[side] = corpora_table.subset_prepare_config(prepare, f"{side}_mix")
    config = tmp_path / "pack_checks.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "candidate": {"table": str(tables["candidate"]), "subset": "candidate_mix"},
                "reference": {"table": str(tables["reference"]), "subset": "reference_mix"},
                "absent_token_ids": [ABSENT],
                "workers": 2,
            }
        )
    )
    data_base = tmp_path / "data"
    return Corpora(
        config=config,
        data_base=data_base,
        candidate=corpora_table.corpus_root(scalars["candidate"]["dataset"], "candidate_mix", data_base),
        reference=corpora_table.corpus_root(scalars["reference"]["dataset"], "reference_mix", data_base),
        candidate_scalars=scalars["candidate"],
        reference_scalars=scalars["reference"],
    )


@pytest.fixture
def corpora(tmp_path) -> Corpora:
    """Both corpora configured and the reference packed; each test packs its own candidate."""
    built = make_corpora(tmp_path, TOKENIZER)
    built.write("reference", REFERENCE_PACKS)
    return built


def documents_of(packs: dict[str, list[list[Document]]]) -> list[Document]:
    return [doc for shard in packs.values() for pack in shard for doc in pack]


class TestContainment:
    def test_a_candidate_of_reference_documents_passes_and_reports_its_counts(self, corpora, tmp_path):
        """Repacked into other packs, shards and orders, and repeated, the reference's documents still pass."""
        corpora.write("candidate", CANDIDATE_PACKS)
        status, report = corpora.run(tmp_path)
        candidate_docs = documents_of(CANDIDATE_PACKS)
        assert (status, report["verdict"], report["failures"]) == (0, "PASS", [])
        assert report["containment"] == {"candidate_documents": 10, "missing_documents": 0, "missing_examples": []}
        assert report["candidate"]["totals"] == {
            "packs": 5,
            "documents": 10,
            "tokens": sum(len(doc.ids) for doc in candidate_docs),
            "loss_tokens": sum(sum(doc.mask) for doc in candidate_docs),
            "distinct_documents": 8,
            "absent_id_counts": {str(ABSENT): 0},
        }
        assert report["reference"]["totals"]["documents"] == report["reference"]["totals"]["distinct_documents"] == 12
        assert [shard["documents"] for shard in report["candidate"]["shards"]] == [6, 4]

    def test_the_report_records_its_inputs(self, corpora, tmp_path):
        corpora.write("candidate", CANDIDATE_PACKS)
        _, report = corpora.run(tmp_path)
        assert report["config"] == {
            "path": str(corpora.config.resolve()),
            "sha256": corpus_documents.file_sha256(corpora.config),
        }
        assert report["code_revision"]
        assert report["absent_token_ids"] == [ABSENT]
        candidate = report["candidate"]
        assert [shard["parquet"] for shard in candidate["shards"]] == [
            str(corpora.parquet("candidate", shard)) for shard in SHARDS
        ]
        assert candidate["table_sha256"] == corpus_documents.file_sha256(Path(candidate["table"]))
        assert candidate["prepare_config_sha256"] == corpus_documents.file_sha256(Path(candidate["prepare_config"]))
        assert (candidate["subset"], candidate["root"], candidate["tokenizer"], candidate["revision"]) == (
            "candidate_mix",
            str(corpora.candidate),
            TOKENIZER,
            REVISION,
        )

    def test_a_document_the_reference_lacks_fails_and_is_located(self, corpora, tmp_path):
        stranger = document(99)
        packs = {**CANDIDATE_PACKS, "shard1": [[REFERENCE[3]], [REFERENCE[0], stranger, REFERENCE[1]]]}
        corpora.write("candidate", packs)
        status, report = corpora.run(tmp_path)
        assert (status, report["verdict"]) == (1, "FAIL")
        assert report["containment"]["missing_documents"] == 1
        (missing,) = report["containment"]["missing_examples"]
        assert {key: missing[key] for key in ("shard", "pack", "document", "start", "length")} == {
            "shard": "shard1",
            "pack": 1,
            "document": 1,
            "start": len(REFERENCE[0].ids),
            "length": len(stranger.ids),
        }
        assert report["failures"] == ["candidate documents that are not documents of the reference: 1"]

    def test_a_loss_mask_difference_alone_fails(self, corpora, tmp_path):
        """The ids are a reference document's; only one mask entry differs."""
        packs = {**CANDIDATE_PACKS, "shard0": [[with_mask_flipped(REFERENCE[4], 2)], [REFERENCE[5]]]}
        corpora.write("candidate", packs)
        status, report = corpora.run(tmp_path)
        assert (status, report["verdict"], report["containment"]["missing_documents"]) == (1, "FAIL", 1)
        assert report["containment"]["missing_examples"][0]["shard"] == "shard0"

    def test_a_document_cut_at_another_boundary_fails(self, corpora, tmp_path):
        """Two reference documents laid end to end, but stored as one document, are not a reference document."""
        merged = Document(ids=REFERENCE[0].ids + REFERENCE[1].ids, mask=REFERENCE[0].mask + REFERENCE[1].mask)
        corpora.write("candidate", {**CANDIDATE_PACKS, "shard0": [[merged]]})
        status, report = corpora.run(tmp_path)
        assert (status, report["containment"]["missing_documents"]) == (1, 1)


class TestAbsentIds:
    def test_an_absent_id_in_the_candidate_fails_and_is_located(self, corpora, tmp_path):
        """The document holding the id is a reference document, so containment passes and only the id fails."""
        marked = document(50, absent_at=3)
        corpora.write("reference", {"shard1": [*REFERENCE_PACKS["shard1"], [marked]]})
        corpora.write("candidate", {**CANDIDATE_PACKS, "shard0": [[REFERENCE[2], marked]]})
        status, report = corpora.run(tmp_path)
        assert (status, report["verdict"]) == (1, "FAIL")
        assert report["containment"]["missing_documents"] == 0
        assert report["failures"] == [f"occurrences of token id {ABSENT} in the candidate's ids: 1"]
        assert report["absent_ids"] == {
            "counts": {str(ABSENT): 1},
            "examples": [
                {
                    "shard": "shard0",
                    "token_id": ABSENT,
                    "pack": 0,
                    "document": 1,
                    "offset": len(REFERENCE[2].ids) + 1 + 3,
                }
            ],
        }
        assert report["reference"]["totals"]["absent_id_counts"] == {str(ABSENT): 1}


class TestNothingIsSkipped:
    @pytest.mark.parametrize("side", ["candidate", "reference"])
    def test_a_missing_shard_fails(self, corpora, tmp_path, side):
        corpora.write("candidate", CANDIDATE_PACKS)
        corpora.parquet(side, "shard1").unlink()
        status, report = corpora.run(tmp_path)
        assert (status, report["verdict"]) == (1, "FAIL")
        assert any(f.startswith(f"{side} shard shard1: {corpora.parquet(side, 'shard1')}") for f in report["failures"])
        assert report[side]["shards"][1]["error"] is not None

    def test_an_unreadable_shard_fails(self, corpora, tmp_path):
        """A parquet cut short, as a pack job still writing leaves one."""
        corpora.write("candidate", CANDIDATE_PACKS)
        parquet = corpora.parquet("candidate", "shard0")
        parquet.write_bytes(parquet.read_bytes()[:64])
        status, report = corpora.run(tmp_path)
        assert (status, report["verdict"]) == (1, "FAIL")
        assert any(f.startswith("candidate shard shard0:") for f in report["failures"])

    def test_a_pack_whose_starts_do_not_split_it_fails(self, corpora, tmp_path):
        corpora.write("candidate", CANDIDATE_PACKS)
        parquet = corpora.parquet("candidate", "shard1")
        doc = REFERENCE[0]
        pq.write_table(
            pa.table({"input_ids": [list(doc.ids)], "loss_mask": [list(doc.mask)], "seq_start_id": [[1]]}), parquet
        )
        status, report = corpora.run(tmp_path)
        assert status == 1
        assert any("seq_start_id" in f and f.startswith("candidate shard shard1:") for f in report["failures"])

    def test_corpora_packed_at_another_tokenizer_fail(self, tmp_path):
        """The documents are the reference's, but packs of another tokenizer are not comparable."""
        corpora = make_corpora(tmp_path, "geodesic-research/nemotron-instruct-tokenizer")
        corpora.write("reference", REFERENCE_PACKS)
        corpora.write("candidate", CANDIDATE_PACKS)
        status, report = corpora.run(tmp_path)
        assert status == 1
        assert report["failures"] == [
            "the candidate is packed at tokenizer='geodesic-research/nemotron-instruct-tokenizer', "
            f"the reference at {TOKENIZER!r}"
        ]


class TestTheCommandLine:
    def test_an_existing_report_is_refused_and_kept(self, corpora, tmp_path):
        corpora.write("candidate", CANDIDATE_PACKS)
        report = tmp_path / "pack_checks.json"
        report.write_text("earlier\n")
        with pytest.raises(SystemExit) as refused:
            checks.main(["--config", str(corpora.config), "--report-out", str(report)])
        assert refused.value.code == 2
        assert report.read_text() == "earlier\n"

    @pytest.mark.parametrize(
        "change, message",
        [
            ({"thresholds": 1}, "unknown keys \\['thresholds'\\]"),
            ({"absent_token_ids": [ABSENT, ABSENT]}, "repeats a token id"),
            ({"workers": 0}, "workers must be an integer >= 1"),
            ({"reference": {"table": "no/such/table.tsv", "subset": "x"}}, "no corpora table"),
        ],
    )
    def test_a_config_it_cannot_read_is_refused(self, corpora, change, message):
        config = yaml.safe_load(corpora.config.read_text())
        corpora.config.write_text(yaml.safe_dump({**config, **change}))
        with pytest.raises(checks.PackCheckConfigError, match=message):
            checks.read_pack_checks(corpora.config, corpora.data_base)

    def test_a_tokenize_row_is_refused(self, corpora, tmp_path):
        prepare = write_prepare_config(tmp_path)
        table = write_table(tmp_path, prepare, subset="tokenized_mix")
        config = yaml.safe_load(corpora.config.read_text())
        config["reference"] = {"table": str(table), "subset": "tokenized_mix"}
        corpora.config.write_text(yaml.safe_dump(config))
        with pytest.raises(checks.PackCheckConfigError, match="is a tokenize row"):
            checks.read_pack_checks(corpora.config, corpora.data_base)


class TestTheCampaignConfig:
    """Clueless-Norm's SFT pack is checked against v4's: the packs the two SFT configs read, at id 500."""

    CONFIG = REPO_ROOT / "configs" / "metagaming_filtering" / "30b_clueless_norm_sft" / "pack_checks.yaml"
    CLUELESS_SFT = (
        REPO_ROOT
        / "configs"
        / "metagaming_filtering"
        / "30b_clueless_norm"
        / "nemotron_nano_30b_metagaming_clueless_norm_sft.yaml"
    )
    V4_SFT = (
        REPO_ROOT
        / "configs"
        / "control_pretraining"
        / "30b_baseline_ablations"
        / "nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4.yaml"
    )

    @pytest.fixture(scope="class")
    def config(self):
        return checks.read_pack_checks(self.CONFIG, corpora_table.DATA_BASE)

    @pytest.mark.parametrize("side, training_config", [("candidate", CLUELESS_SFT), ("reference", V4_SFT)])
    def test_each_side_is_the_pack_its_sft_trains_on(self, config, side, training_config):
        dataset = yaml.safe_load(training_config.read_text())["dataset"]
        corpus = getattr(config, side)
        assert str(corpus.root) == dataset["dataset_root"]
        glob = dataset["packed_sequence_specs"]["packed_train_data_path"]
        assert [str(parquet) for _, parquet in corpus.shards] == [
            glob.replace("shard*", f"shard{index}") for index in range(32)
        ]

    def test_it_checks_id_500_and_scans_every_shard_at_once(self, config):
        assert config.absent_ids == (500,)
        assert config.workers == len(config.candidate.shards) + len(config.reference.shards) == 64
