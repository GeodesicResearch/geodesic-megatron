# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The documents training runs have not trained on (scripts/data/held_out_documents.py).

Nothing is stubbed. The corpora are real ``.bin/.idx`` files from Megatron's own writer with the records a build
leaves beside them, listed in a real corpora table; their dataset is a local repository laid out as the Hub lays one
out; the runs are the real Nano pretrain recipe with mcore's NullTokenizer; and every step runs through the tool's own
command line. Every token of a document carries one value unique to that document, so the oracles are independent of
the tool's index arithmetic:

* when a run first reads a document is the first iteration at which its data-parallel ranks' loaders yield the
  document's value, read from the batches of real loaders;
* what is held out, and what is still unseen at an iteration, is re-derived from those iterations, the sizes and the
  ids with plain sets.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from scripts.training.launch_width import config_launch_width

from megatron.bridge.data.loaders import build_data_loader
from tests.unit_tests.corpora_fixtures import (
    REVISION,
    TOKENIZER,
    build_pretraining_dataset,
    build_tokenized_corpus,
    corpora_table,
    load_campaign_module,
    write_parquet_dataset,
    write_prepare_config,
    write_table,
)


corpus_documents = load_campaign_module("corpus_documents")

REPO_ROOT = Path(__file__).resolve().parents[3]
SEQ_LENGTH = 8
VOCAB_SIZE = 1000
# The document-check columns of a row whose dataset states each row's source index in `source_row`.
CHECKS = {"count_token": 7, "count_column": "n_hidden", "row_column": "source_row"}
# Each corpus: the value of its document 0 (document i holds only base + i), its documents' lengths, and the source
# row of its document 0. The two alpha slices hold SLICE documents each, the second beginning where the first ends
# in their source. alpha_shard0's document 5 is empty: no sequence, so nothing reads it and nothing holds it out.
SLICE = 30
CORPORA = {
    "alpha_shard0": (
        100,
        [5, 9, 3, 8, 7, 0, 6, 4, 10, 5, 6, 8, 7, 5, 9, 6, 8, 4, 10, 6, 5, 9, 7, 8, 6, 4, 9, 5, 7, 8],
        0,
    ),
    "alpha_shard1": (
        300,
        [4, 12, 6, 9, 7, 11, 5, 8, 6, 10, 9, 7, 6, 8, 5, 9, 7, 10, 4, 8, 6, 9, 5, 7, 11, 6, 8, 5, 9, 7],
        SLICE,
    ),
    "beta": (500, [6, 9, 7, 8, 5, 10], 0),
    "gamma": (700, [8, 6, 9, 7], 0),
}
SOURCE_SUBSETS = {"alpha_shard0": "alpha", "alpha_shard1": "alpha"}
# "first" draws each alpha slice for less than one pass and beta for more than one; "second" reads on through
# alpha_shard1 (same seed, so the same document order), reaching documents "first" does not, and reads gamma, whose
# table row traces its documents to no source rows.
RUNS = {
    "first": (["0.25", "alpha_shard0", "0.25", "alpha_shard1", "0.5", "beta"], 4),
    "second": (["0.5", "alpha_shard1", "0.25", "beta", "0.25", "gamma"], 3),
}
GLOBAL_BATCH = 4
# The iterations of "first" (which trains four) a future set is written for, and how many documents each samples.
FUTURE_ITERATIONS = [1, 3]
FUTURE_DOCUMENTS = 2
# The iterations batch-check materialises, per run ("first" trains four, "second" three), and of "other" (four).
BATCH_CHECKS = {"first": [1, 2, 4], "second": [1, 3]}
OTHER_BATCH_CHECKS = [1, 4]


@pytest.fixture(scope="module")
def tool():
    """scripts/data/held_out_documents.py, loaded by path: scripts/ is not a package."""
    path = REPO_ROOT / "scripts" / "data" / "held_out_documents.py"
    spec = importlib.util.spec_from_file_location("held_out_documents", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["held_out_documents"] = module  # @dataclass resolves its module through sys.modules
    spec.loader.exec_module(module)
    return module


def _documents(subset: str) -> list[list[int]]:
    base, lengths, _ = CORPORA[subset]
    return [[base + i] * n for i, n in enumerate(lengths)]


def _subsets(run: str) -> list[str]:
    return RUNS[run][0][1::2]


def _run_config(directory: Path, prefixes: dict[str, str], blend: list[str], iterations: int, **extra) -> Path:
    """A pretrain override YAML over the real Nano recipe that needs no network: NullTokenizer, no model parallelism."""
    data_path = [item if index % 2 == 0 else prefixes[item] for index, item in enumerate(blend)]
    config = {
        "tokenizer": {"tokenizer_type": "NullTokenizer", "vocab_size": VOCAB_SIZE},
        "dataset": {
            "data_path": data_path,
            "seq_length": SEQ_LENGTH,
            "split": "1,0,0",
            "path_to_cache": str(directory / "cache"),
        },
        "model": {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "context_parallel_size": 1},
        "train": {"train_iters": iterations, "global_batch_size": GLOBAL_BATCH, "micro_batch_size": 1},
        **extra,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass
class Campaign:
    """A tiny campaign on disk: its corpora, table, runs and held-out config."""

    root: Path
    repo: Path
    table: Path
    data_base: Path
    prefixes: dict[str, str]
    run_configs: dict[str, Path]
    config: Path
    ids: dict[str, list[str]]

    def settings(self) -> dict:
        return yaml.safe_load(self.config.read_text())

    def variant(self, directory: Path, **changes) -> Path:
        """A copy of the held-out config with top-level keys replaced, its lists copied beside it."""
        settings = {**self.settings(), **changes}
        if "lists" not in changes:
            settings["lists"] = str(directory / "lists")
            shutil.copytree(self.settings()["lists"], settings["lists"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "held_out.yaml"
        path.write_text(yaml.safe_dump(settings))
        return path

    def first_read(self, run: str, position: int) -> list[int]:
        return np.load(Path(self.settings()["lists"]) / run / f"first_read.{position}.npy").tolist()

    def held_out(self, run: str, position: int) -> list[int]:
        return np.load(Path(self.settings()["lists"]) / run / f"held_out.{position}.npy").tolist()


def _write_dataset(repo: Path, ids: dict[str, list[str]]) -> None:
    """The corpora's dataset: per subset, each row's id and source row, beside a text column no step may need, split
    across two files so a reader must concatenate them in order."""
    for subset, (_, lengths, first_row) in CORPORA.items():
        if subset in ids:
            rows = list(range(first_row, first_row + len(lengths)))
            columns = {"text": [f"{subset} {i}" for i in rows], "doc_id": ids[subset], "source_row": rows}
            write_parquet_dataset(repo, subset, columns, files=2)


def _build(root: Path) -> Campaign:
    """The corpora, their table and the runs; the dataset and the held-out config follow once ids are chosen."""
    repo, data_base = root / "repo", root / "data"
    config = write_prepare_config(root, dataset=str(repo))
    rows = [
        {"subset": subset, "docs": len(lengths), **({} if subset == "gamma" else {**CHECKS, "first_row": first_row})}
        for subset, (_, lengths, first_row) in CORPORA.items()
    ]
    table = write_table(root, config, **rows[0], extra_rows=rows[1:])
    prefixes = {}
    for subset in CORPORA:
        corpus = corpora_table.corpus_root(str(repo), subset, data_base)
        build_tokenized_corpus(corpus, _documents(subset), subset=subset, dataset=str(repo))
        prefixes[subset] = str(corpus / corpora_table.TOKENIZED_PREFIX)
    run_configs = {
        name: _run_config(root / "runs" / name, prefixes, blend, iterations)
        for name, (blend, iterations) in RUNS.items()
    }
    return Campaign(root, repo, table, data_base, prefixes, run_configs, root / "held_out.yaml", {})


def _held_out_config(campaign: Campaign, **changes) -> dict:
    settings = {
        "table": str(campaign.table),
        "lists": str(campaign.root / "lists"),
        "runs": {
            name: {"config": str(path), "model": "nano", "mode": "pretrain", "gpus": 2}
            for name, path in campaign.run_configs.items()
        },
        "document_ids": {
            "column": "doc_id",
            "source_subsets": dict(SOURCE_SUBSETS),
            "subsets_without_ids": ["gamma"],
        },
        "set": {"run": "first", "path": str(campaign.root / "sets" / "never_read.parquet"), "documents": 4, "seed": 7},
        "validation": {
            "first": {
                "prefix": str(campaign.root / "validation" / "first" / "held_out_document"),
                "samples": 2,
                "seed": 7,
            }
        },
        "future": {
            "run": "first",
            "also_unread_by": [],
            "iterations": list(FUTURE_ITERATIONS),
            "documents": FUTURE_DOCUMENTS,
            "seed": 7,
            "directory": str(campaign.root / "sets"),
        },
        "batch_checks": {
            run: list(iterations) for run, iterations in BATCH_CHECKS.items() if run in campaign.run_configs
        },
    }
    return {**settings, **changes}


@pytest.fixture(scope="module")
def campaign(tool, tmp_path_factory) -> Campaign:
    """The whole pipeline, run once through the command line: both runs read, then ids chosen so that each id rule has
    something to exclude, then the held-out lists, the set, the first run's validation corpus and its future sets."""
    built = _build(tmp_path_factory.mktemp("campaign"))
    built.config.write_text(yaml.safe_dump(_held_out_config(built)))
    for run in RUNS:
        assert tool.main(["read", "--config", str(built.config), "--run", run]) == 0

    ids = {subset: [f"{subset}-{i}" for i in range(len(lengths))] for subset, (_, lengths, _) in CORPORA.items()}
    del ids["gamma"]
    first = built.first_read("first", 0)
    unread = [d for d, when in enumerate(first) if not when and CORPORA["alpha_shard0"][1][d]]
    assert len(unread) >= 3, "the fixture needs three unread documents of alpha_shard0 to give ids to"
    # One never-read document shares its id with a document beta holds (which "first" reads), and one with a document
    # of its own corpus that "first" reads.
    ids["alpha_shard0"][unread[0]] = ids["beta"][0]
    ids["alpha_shard0"][unread[1]] = ids["alpha_shard0"][next(d for d, when in enumerate(first) if when)]
    # Two documents of alpha_shard1 that "first" first reads in its last iteration are nothing new before it. One
    # shares its id with a document of beta the run reads in its first iteration. The other shares its id with a
    # document of its own corpus that sits after it and is read an iteration earlier, so its id is first read where
    # the later document is, not where the corpus's first document with that id is.
    last = RUNS["first"][1]
    later = built.first_read("first", 1)
    at_last = [d for d, when in enumerate(later) if when == last]
    sooner = next(d for d, when in enumerate(later) if when == last - 1)
    early = built.first_read("first", 2).index(1)
    assert len(at_last) >= 3 and at_last[0] < sooner, "the fixture needs a last-iteration document before `sooner`"
    assert early != 0, "beta's document 0 already lends its id to a never-read document"
    ids["alpha_shard1"][at_last[0]] = ids["alpha_shard1"][sooner]
    ids["alpha_shard1"][at_last[1]] = ids["beta"][early]
    built.ids = ids
    _write_dataset(built.repo, ids)

    data_base = ["--data-base", str(built.data_base)]
    assert tool.main(["held-out", "--config", str(built.config), *data_base]) == 0
    assert tool.main(["set", "--config", str(built.config)]) == 0
    assert tool.main(["validation", "--config", str(built.config), "--run", "first"]) == 0
    assert tool.main(["future", "--config", str(built.config), *data_base]) == 0
    for run in BATCH_CHECKS:
        assert tool.main(["batch-check", "--config", str(built.config), "--run", run, *data_base]) == 0
    return built


def _values_yielded_by_iteration(tool, run_config: Path, gpus: int) -> dict[int, set[int]]:
    """Per iteration, the token values the run's data-parallel ranks' loaders yield, labels included, read from the
    batches of real loaders built as the run builds them (in-process, so no worker is forked)."""
    cfg = tool.resolve_run_config(str(run_config), "nano", "pretrain")
    cfg.dataset.num_workers = 0
    train_ds, _, first_sample = tool.build_training_dataset(cfg)
    data_parallel = cfg.get_data_parallel_size(gpus)
    microbatches = cfg.train.global_batch_size // data_parallel
    yielded: dict[int, set[int]] = {}
    for rank in range(data_parallel):
        loader = build_data_loader(
            train_ds,
            first_sample,
            cfg.dataset.dataloader_type,
            cfg,
            data_parallel_rank=rank,
            data_parallel_size=data_parallel,
            persistent_workers=False,
        )
        for number, batch in enumerate(islice(loader, cfg.train.train_iters * microbatches)):
            values = yielded.setdefault(number // microbatches + 1, set())
            values.update(batch["tokens"].flatten().tolist(), batch["labels"].flatten().tolist())
    return yielded


def _first_iteration_each_value_is_yielded(tool, run_config: Path, gpus: int) -> dict[int, int]:
    """Per token value, the first iteration at which the run's loaders yield it (``_values_yielded_by_iteration``)."""
    first: dict[int, int] = {}
    for iteration, values in sorted(_values_yielded_by_iteration(tool, run_config, gpus).items()):
        for value in values:
            first.setdefault(value, iteration)
    return first


def _first_read_by_value(first: dict[int, int], subset: str) -> list[int]:
    base, lengths, _ = CORPORA[subset]
    return [first.get(base + i, 0) for i in range(len(lengths))]


def _first_reads(campaign: Campaign) -> dict[tuple[str, str], list[int]]:
    return {
        (run, subset): campaign.first_read(run, position)
        for run in RUNS
        for position, subset in enumerate(_subsets(run))
    }


def _source_row(subset: str, document: int) -> tuple[str, int]:
    return SOURCE_SUBSETS.get(subset, subset), CORPORA[subset][2] + document


def _expected_held_out(campaign: Campaign) -> dict[tuple[str, str], list[int]]:
    """Per run and corpus with ids: the documents the run never reads, that are not empty, and whose id no run reads
    anywhere, re-derived with plain sets."""
    first = _first_reads(campaign)
    read_ids = {
        campaign.ids[subset][d]
        for (_, subset), whens in first.items()
        if subset in campaign.ids
        for d, when in enumerate(whens)
        if when
    }
    return {
        (run, subset): [
            d
            for d, when in enumerate(whens)
            if not when and CORPORA[subset][1][d] and campaign.ids[subset][d] not in read_ids
        ]
        for (run, subset), whens in first.items()
        if subset in campaign.ids
    }


def _expected_unseen(campaign: Campaign, iteration: int) -> tuple[list[tuple[str, int]], int]:
    """After ``iteration`` of "first": the source rows of the documents it first reads later and whose id it has not
    read by then, in its blend's order, and how many documents it first reads later in all."""
    first = {subset: whens for (run, subset), whens in _first_reads(campaign).items() if run == "first"}
    read_ids = {
        campaign.ids[subset][d]
        for subset, whens in first.items()
        for d, when in enumerate(whens)
        if 0 < when <= iteration
    }
    later = [(subset, d) for subset, whens in first.items() for d, when in enumerate(whens) if when > iteration]
    return [_source_row(subset, d) for subset, d in later if campaign.ids[subset][d] not in read_ids], len(later)


class TestWhenARunFirstReadsEachDocument:
    @pytest.mark.parametrize("run", sorted(RUNS))
    def test_each_array_is_the_first_iteration_the_runs_loaders_yield_the_document(self, tool, campaign, run):
        first = _first_iteration_each_value_is_yielded(tool, campaign.run_configs[run], gpus=2)
        for position, subset in enumerate(_subsets(run)):
            assert campaign.first_read(run, position) == _first_read_by_value(first, subset), subset

    def test_a_corpus_drawn_for_less_than_a_pass_keeps_unread_documents_and_one_read_twice_keeps_none(self, campaign):
        record = json.loads((Path(campaign.settings()["lists"]) / "first" / "read.json").read_text())
        alpha, _, beta = record["corpora"]
        assert 0 < alpha["documents_read"] < alpha["documents"] == SLICE
        assert beta["documents_read"] == beta["documents"] == 6
        assert record["samples_read"] == record["dataset_samples"] == RUNS["first"][1] * GLOBAL_BATCH

    def test_an_empty_document_is_never_read(self, campaign):
        assert campaign.first_read("first", 0)[5] == 0

    def test_the_record_names_what_the_arrays_depend_on(self, campaign):
        record = json.loads((Path(campaign.settings()["lists"]) / "first" / "read.json").read_text())
        assert record["training_config"] == {
            "path": str(campaign.run_configs["first"]),
            "sha256": corpus_documents.file_sha256(campaign.run_configs["first"]),
        }
        assert (record["model"], record["mode"], record["gpus"], record["data_parallel_size"]) == (
            "nano",
            "pretrain",
            2,
            2,
        )
        assert record["inputs"]["seq_length"] == SEQ_LENGTH and record["config"]["text"] == campaign.config.read_text()
        for corpus in record["corpora"]:
            prefix = Path(corpus["prefix"])
            assert corpus["idx_sha256"] == corpus_documents.file_sha256(corpus_documents.idx_path(prefix))

    @pytest.mark.parametrize("gpus", [1, 4])
    def test_a_blend_built_larger_than_the_run_is_read_as_far_as_the_run_reads(self, tool, tmp_path, gpus):
        """Weights that are not whole samples build 17 samples for a run of 16, and the one left reads nothing."""
        built = _build(tmp_path)
        blend = ["0.3", "alpha_shard0", "0.3", "alpha_shard1", "0.4", "beta"]
        built.run_configs = {"first": _run_config(tmp_path / "fractional", built.prefixes, blend, 4)}
        settings = _held_out_config(built)
        settings["runs"]["first"]["gpus"] = gpus
        built.config.write_text(yaml.safe_dump(settings))
        assert tool.main(["read", "--config", str(built.config), "--run", "first"]) == 0

        record = json.loads((tmp_path / "lists" / "first" / "read.json").read_text())
        assert (record["dataset_samples"], record["samples_read"]) == (17, 16)
        assert sum(corpus["samples_read"] for corpus in record["corpora"]) == 16
        first = _first_iteration_each_value_is_yielded(tool, built.run_configs["first"], gpus)
        for position, subset in enumerate(blend[1::2]):
            assert built.first_read("first", position) == _first_read_by_value(first, subset), subset


class TestWhatIsHeldOut:
    def test_each_list_is_the_unread_documents_whose_id_no_run_reads(self, campaign):
        expected = _expected_held_out(campaign)
        for run in RUNS:
            for position, subset in enumerate(_subsets(run)):
                if subset in campaign.ids:
                    assert campaign.held_out(run, position) == expected[(run, subset)], (run, subset)

    def test_the_record_counts_what_each_rule_removed(self, campaign):
        record = json.loads((Path(campaign.settings()["lists"]) / "held_out.json").read_text())
        alpha, later, beta = record["runs"]["first"]["corpora"]
        assert (alpha["subset"], alpha["source_subset"], alpha["first_row"]) == ("alpha_shard0", "alpha", 0)
        assert (later["source_subset"], later["first_row"]) == ("alpha", SLICE)
        assert alpha["empty"] == 1 and alpha["unread"] == SLICE - alpha["read"] - 1
        # The two ids the fixture planted: one that beta holds, one that a read document of the corpus itself holds.
        assert alpha["id_read_in"] == {"alpha_shard0": 1, "beta": 1} and alpha["id_read_by_a_run"] == 2
        assert alpha["held_out"] == alpha["unread"] - 2
        # "second" reads documents of alpha_shard1 that "first" does not, so "first" holds fewer of them out.
        assert later["id_read_in"] == {"alpha_shard1": later["unread"] - later["held_out"]} and later["held_out"]
        assert later["held_out"] < later["unread"]
        assert (beta["unread"], beta["held_out"]) == (0, 0)

    def test_a_corpus_without_ids_takes_no_part(self, campaign):
        record = json.loads((Path(campaign.settings()["lists"]) / "held_out.json").read_text())
        gamma = record["runs"]["second"]["corpora"][2]
        assert gamma == {
            "prefix": campaign.prefixes["gamma"],
            "subset": "gamma",
            "documents": 4,
            "read": gamma["read"],
            "has_ids": False,
        }
        assert not (Path(campaign.settings()["lists"]) / "second" / "held_out.2.npy").exists()

    def test_a_corpus_whose_rows_cannot_be_traced_must_be_listed_as_without_ids(
        self, tool, campaign, tmp_path, caplog
    ):
        ids = {**campaign.settings()["document_ids"], "subsets_without_ids": []}
        config = campaign.variant(tmp_path, document_ids=ids)
        (tmp_path / "lists" / "held_out.json").unlink()
        with caplog.at_level(logging.ERROR):
            assert tool.main(["held-out", "--config", str(config), "--data-base", str(campaign.data_base)]) == 1
        assert "gamma: its documents cannot be traced to source rows" in caplog.text

    def test_rows_out_of_their_sources_order_are_refused(self, tool, tmp_path, caplog):
        """The id of row i is document i's only while the dataset's rows are the source's in order."""
        built = _build(tmp_path)
        built.run_configs = {"first": built.run_configs["first"]}
        built.config.write_text(yaml.safe_dump(_held_out_config(built)))
        assert tool.main(["read", "--config", str(built.config), "--run", "first"]) == 0
        ids = {subset: [f"{subset}-{i}" for i in range(len(CORPORA[subset][1]))] for subset in _subsets("first")}
        _write_dataset(built.repo, ids)
        rows = list(range(SLICE))
        rows[3], rows[4] = rows[4], rows[3]
        shutil.rmtree(built.repo / "alpha_shard0")
        columns = {"text": ["t"] * SLICE, "doc_id": ids["alpha_shard0"], "source_row": rows}
        write_parquet_dataset(built.repo, "alpha_shard0", columns, files=2)
        with caplog.at_level(logging.ERROR):
            assert tool.main(["held-out", "--config", str(built.config), "--data-base", str(built.data_base)]) == 1
        assert "alpha_shard0: row 3 holds source_row 4, the source's row there is 3" in caplog.text
        assert not (tmp_path / "lists" / "held_out.json").exists()

    def test_a_table_names_each_rows_prefixes_and_the_rows_it_still_holds(self, tool, tmp_path):
        """A sliced row's prefixes hold its rows in order; a byte-gated split's hold rows the table cannot say; a
        held row has no prefix yet."""
        config = write_prepare_config(tmp_path)
        rows = [
            {"subset": "sliced", "docs": 5, "shards": 2, "shard_mode": "slice"},
            {"subset": "split", "docs": 5, "shards": 2, "shard_mode": "split"},
            {"subset": "held", "docs": corpora_table.DOCS_PENDING},
        ]
        table = write_table(tmp_path, config, **rows[0], extra_rows=rows[1:])
        corpora, held = tool.table_corpora(table, tmp_path / "data")
        named = {
            (corpus.row.subset, prefix.parent.name, corpus.rows)
            for prefix, corpus in corpora.items()
            if prefix.name == corpora_table.TOKENIZED_PREFIX
        }
        assert named == {
            ("sliced", "shard0", (0, 2)),
            ("sliced", "shard1", (2, 5)),
            ("split", "shard0", None),
            ("split", "shard1", None),
        }
        assert held == ["held"] and len(corpora) == 4


class TestTheIdsAreReadOnce:
    def test_each_subsets_ids_are_kept_beside_the_lists_under_its_commit(self, campaign):
        kept = Path(campaign.settings()["lists"]) / "ids"
        assert sorted(path.name for path in kept.iterdir()) == [
            f"{subset}.doc_id@{REVISION}.parquet" for subset in sorted(campaign.ids)
        ]
        table = pq.read_table(kept / f"beta.doc_id@{REVISION}.parquet")
        assert table.column_names == ["doc_id", "source_row"] and table["doc_id"].to_pylist() == campaign.ids["beta"]

    def test_a_later_step_reads_the_kept_ids_and_not_the_dataset(self, tool, campaign, tmp_path):
        future = {**campaign.settings()["future"], "directory": str(tmp_path / "sets")}
        config = campaign.variant(tmp_path, future=future)
        hidden = campaign.repo.with_name("repo.hidden")
        campaign.repo.rename(hidden)
        try:
            assert tool.main(["future", "--config", str(config), "--data-base", str(campaign.data_base)]) == 0
        finally:
            hidden.rename(campaign.repo)
        original = Path(campaign.settings()["future"]["directory"])
        for iteration in FUTURE_ITERATIONS:
            name = f"future_first_iter_{iteration:07d}.parquet"
            assert pq.read_table(tmp_path / "sets" / name).equals(pq.read_table(original / name))


def _rows(path: Path) -> list[tuple[str, int]]:
    """A set file's rows, checked to be the two columns a scorer reads, of the types it reads."""
    table = pq.read_table(path)
    assert table.column_names == ["subset", "source_row"]
    assert (str(table.schema.field("subset").type), str(table.schema.field("source_row").type)) == ("string", "int64")
    return list(zip(table["subset"].to_pylist(), table["source_row"].to_pylist()))


class TestTheSet:
    def test_it_is_a_sample_of_the_runs_held_out_documents_as_source_rows(self, campaign):
        expected = _expected_held_out(campaign)
        pool = [
            _source_row(subset, d) for subset in ("alpha_shard0", "alpha_shard1") for d in expected[("first", subset)]
        ]
        rows = _rows(Path(campaign.settings()["set"]["path"]))
        assert len(rows) == len(set(rows)) == 4 and set(rows) <= set(pool)
        assert rows == sorted(rows), "the set is in corpus and row order"

    def test_its_record_states_the_sample_and_the_files_sha256(self, campaign):
        path = Path(campaign.settings()["set"]["path"])
        record = json.loads(Path(f"{path}.json").read_text())
        expected = _expected_held_out(campaign)
        assert record["sha256"] == corpus_documents.file_sha256(path)
        assert (record["run"], record["seed"], record["documents"]) == ("first", 7, 4)
        assert record["held_out"] == sum(len(expected[("first", subset)]) for subset in _subsets("first"))
        assert sum(corpus["sampled"] for corpus in record["corpora"]) == 4

    def test_the_seed_decides_the_sample(self, tool, campaign, tmp_path):
        drawn = {}
        for name, seed in (("same", 7), ("other", 8)):
            path = tmp_path / name / "set.parquet"
            config = campaign.variant(
                tmp_path / name, set={**campaign.settings()["set"], "path": str(path), "seed": seed}
            )
            assert tool.main(["set", "--config", str(config)]) == 0
            drawn[name] = _rows(path)
        assert drawn["same"] == _rows(Path(campaign.settings()["set"]["path"]))
        assert drawn["other"] != drawn["same"]

    def test_more_documents_than_are_held_out_is_refused(self, tool, campaign, tmp_path, caplog):
        path = tmp_path / "set.parquet"
        spec = {**campaign.settings()["set"], "path": str(path), "documents": 1000}
        with caplog.at_level(logging.ERROR):
            assert tool.main(["set", "--config", str(campaign.variant(tmp_path, set=spec))]) == 1
        assert "set.documents asks for 1000 documents, and only" in caplog.text and not path.exists()


def _read_documents(prefix: Path) -> list[list[int]]:
    """A prefix's documents, read the way training reads them."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    dataset = IndexedDataset(str(prefix))
    bounds = dataset.document_indices.tolist()
    return [
        [token for sequence in range(bounds[d], bounds[d + 1]) for token in dataset[sequence].tolist()]
        for d in range(len(bounds) - 1)
    ]


class TestTheValidationCorpus:
    def _prefix(self, campaign: Campaign) -> Path:
        return Path(campaign.settings()["validation"]["first"]["prefix"])

    def test_it_holds_held_out_documents_of_the_run_byte_for_byte(self, campaign):
        prefix = self._prefix(campaign)
        rows = _rows(Path(f"{prefix}.documents.parquet"))
        expected = _expected_held_out(campaign)
        pool = {
            _source_row(subset, d): _documents(subset)[d]
            for subset in ("alpha_shard0", "alpha_shard1")
            for d in expected[("first", subset)]
        }
        assert len(rows) == len(set(rows)) and set(rows) <= set(pool)
        assert _read_documents(prefix) == [pool[row] for row in rows]

    def test_it_holds_the_samples_asked_for_and_training_builds_them(self, campaign, tmp_path):
        prefix = self._prefix(campaign)
        record = json.loads(Path(f"{prefix}.provenance.json").read_text())
        tokens = sum(len(document) for document in _read_documents(prefix))
        assert record["totals"]["total_tokens"] == tokens >= 2 * SEQ_LENGTH + 1
        assert (record["samples"], record["sequence_length"], record["run"]) == (2, SEQ_LENGTH, "first")
        dataset = build_pretraining_dataset([str(prefix)], SEQ_LENGTH, 2, VOCAB_SIZE, tmp_path / "cache")
        assert len(dataset) >= 2

    def test_its_record_states_how_its_ids_were_made_and_what_each_parent_gave(self, campaign):
        prefix = self._prefix(campaign)
        record = json.loads(Path(f"{prefix}.provenance.json").read_text())
        # Of a parent's tokenize record, the fields that decide ids: not its own paths or job.
        assert record["kind"] == "held_out_validation"
        assert record["parameters"] == {"tokenizer": TOKENIZER, "append_eod": "true"}
        # Every corpus of the run that has ids is a parent, beta included, which holds nothing out.
        assert [parent["subset"] for parent in record["parents"]] == ["alpha_shard0", "alpha_shard1", "beta"]
        assert record["parents"][2]["held_out"] == record["parents"][2]["copied"] == 0
        assert sum(parent["copied"] for parent in record["parents"]) == record["totals"]["num_documents"]
        assert not list(prefix.parent.glob("*.partial"))

    def test_parents_tokenized_differently_are_refused(self, tool, campaign, tmp_path, caplog):
        """Asked for every id the run holds out, the corpus draws on both slices, which must then agree."""
        record = json.loads((Path(campaign.settings()["lists"]) / "held_out.json").read_text())
        held_out_ids = sum(corpus.get("held_out_tokens", 0) for corpus in record["runs"]["first"]["corpora"])
        prefix = tmp_path / "validation" / "held_out_document"
        spec = {"first": {"prefix": str(prefix), "samples": (held_out_ids - 1) // SEQ_LENGTH, "seed": 7}}
        config = campaign.variant(tmp_path, validation=spec)
        provenance = Path(f"{campaign.prefixes['alpha_shard1']}.provenance.json")
        original = provenance.read_text()
        changed = json.loads(original)
        changed["parameters"]["tokenizer"] = "another/tokenizer"
        provenance.write_text(json.dumps(changed))
        try:
            with caplog.at_level(logging.ERROR):
                assert tool.main(["validation", "--config", str(config), "--run", "first"]) == 1
        finally:
            provenance.write_text(original)
        assert "were not all tokenized alike" in caplog.text and not prefix.parent.exists()

    def test_more_samples_than_the_held_out_documents_hold_is_refused(self, tool, campaign, tmp_path, caplog):
        prefix = tmp_path / "validation" / "held_out_document"
        spec = {"first": {"prefix": str(prefix), "samples": 500, "seed": 7}}
        config = campaign.variant(tmp_path, validation=spec)
        with caplog.at_level(logging.ERROR):
            assert tool.main(["validation", "--config", str(config), "--run", "first"]) == 1
        assert "and 500 samples of 8 need 4001" in caplog.text and not prefix.parent.exists()

    def test_a_run_with_no_validation_corpus_configured_is_refused(self, tool, campaign, caplog):
        with caplog.at_level(logging.ERROR):
            assert tool.main(["validation", "--config", str(campaign.config), "--run", "second"]) == 1
        assert "validation names no corpus for run second" in caplog.text

    def test_documents_copied_from_several_parents_are_checked_against_each(self, campaign):
        """The check that passed the corpus refuses it once the last parent it drew from is said to have given
        another document: every document holds its own value, so no other document of the parent matches."""
        prefix = self._prefix(campaign)
        record = json.loads(Path(f"{prefix}.provenance.json").read_text())
        listed = pq.read_table(f"{prefix}.documents.parquet")["source_row"].to_pylist()
        copied, first = [], 0
        for parent in record["parents"]:
            parent_prefix = Path(parent["prefix"])
            index = corpus_documents.read_index(
                corpus_documents.idx_path(parent_prefix), corpus_documents.bin_path(parent_prefix)
            )
            offset = CORPORA[parent["subset"]][2]
            local = np.asarray(listed[first : first + parent["copied"]], dtype=np.int64) - offset
            copied.append(corpus_documents.CopiedDocuments(parent_prefix, index, local))
            first += parent["copied"]
        files = (corpus_documents.idx_path(prefix), corpus_documents.bin_path(prefix))
        assert corpus_documents.check_copied_documents(copied, *files).docs == len(listed)

        last = max(number for number, source in enumerate(copied) if len(source.local))
        other = copied[last].local.copy()
        other[0] = (other[0] + 1) % copied[last].index.docs
        copied[last] = corpus_documents.CopiedDocuments(copied[last].prefix, copied[last].index, other)
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="parent document"):
            corpus_documents.check_copied_documents(copied, *files)


class TestTheFutureSets:
    def _directory(self, campaign: Campaign) -> Path:
        return Path(campaign.settings()["future"]["directory"])

    @pytest.mark.parametrize("iteration", FUTURE_ITERATIONS)
    def test_each_is_a_sample_of_what_the_run_first_reads_later_and_has_not_seen_by_id(self, campaign, iteration):
        unseen, _ = _expected_unseen(campaign, iteration)
        rows = _rows(self._directory(campaign) / f"future_first_iter_{iteration:07d}.parquet")
        assert len(rows) == len(set(rows)) == FUTURE_DOCUMENTS and set(rows) <= set(unseen)

    def test_a_document_sharing_its_id_with_one_already_read_is_nothing_new(self, campaign):
        """The fixture's two: documents of alpha_shard1 the run first reads in its last iteration, one under the id
        of a document of beta it reads in its first, one under the id of a document of alpha_shard1 it reads an
        iteration before the last. Once the run has read the id, the later document is still to be read and is not
        unseen; until then it is."""
        record = json.loads((self._directory(campaign) / "future_sets.json").read_text())
        by_iteration = {entry["iter"]: entry for entry in record["sets"]}
        for iteration in FUTURE_ITERATIONS:
            unseen, later = _expected_unseen(campaign, iteration)
            entry = by_iteration[iteration]
            assert (entry["first_read_later"], entry["unseen"]) == (later, len(unseen))
            assert entry["id_read_by_then"] == later - len(unseen)
        # After iteration 1 only beta's id has been read; by iteration 3 the id read in iteration 3 has been too.
        assert {iteration: by_iteration[iteration]["id_read_by_then"] for iteration in FUTURE_ITERATIONS} == {
            1: 1,
            3: 2,
        }

    def test_the_manifest_lists_each_set_by_stage_and_iteration(self, campaign):
        directory = self._directory(campaign)
        header, *lines = (directory / "future_sets.tsv").read_text().splitlines()
        assert header.split("\t") == ["stage", "iter", "path", "sha256"]
        assert [line.split("\t") for line in lines] == [
            [
                "first",
                str(iteration),
                str(directory / f"future_first_iter_{iteration:07d}.parquet"),
                corpus_documents.file_sha256(directory / f"future_first_iter_{iteration:07d}.parquet"),
            ]
            for iteration in FUTURE_ITERATIONS
        ]

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"iterations": [4]}, "names 4, and run first reads nothing after its last iteration, 4"),
            ({"iterations": []}, "future.iterations is empty"),
            ({"iterations": [3], "documents": 1000}, "future.documents at iteration 3 asks for 1000 documents"),
        ],
    )
    def test_sets_that_cannot_be_written_are_refused(self, tool, campaign, tmp_path, caplog, change, message):
        future = {**campaign.settings()["future"], "directory": str(tmp_path / "sets"), **change}
        config = campaign.variant(tmp_path, future=future)
        with caplog.at_level(logging.ERROR):
            assert tool.main(["future", "--config", str(config), "--data-base", str(campaign.data_base)]) == 1
        assert message in caplog.text
        assert not (tmp_path / "sets" / "future_sets.tsv").exists()


def _batch_record(campaign: Campaign, run: str) -> dict:
    return json.loads((Path(campaign.settings()["lists"]) / run / "batches.json").read_text())


def _batch_windows(campaign: Campaign, run: str, iteration: int) -> list[dict]:
    return pq.read_table(Path(campaign.settings()["lists"]) / run / f"batches.{iteration:07d}.parquet").to_pylist()


def _documents_of_values(values: set[int], corpora: dict[str, tuple], subsets: list[str]) -> set[tuple[str, int]]:
    """The (subset, document) pairs whose unique value is among ``values``, of the corpora ``subsets`` names."""
    return {
        (subset, value - corpora[subset][0])
        for subset in subsets
        for value in values
        if 0 <= value - corpora[subset][0] < len(corpora[subset][1])
    }


def _a_document_first_read_at(campaign: Campaign, run: str, iteration: int) -> tuple[int, int]:
    """The first (corpus position, document) of ``run`` that its read record says it first reads at ``iteration``."""
    return next(
        (position, document)
        for position in range(len(_subsets(run)))
        for document, when in enumerate(campaign.first_read(run, position))
        if when == iteration
    )


def _variant_to_check_again(campaign: Campaign, directory: Path, batch_checks: dict, **changes) -> Path:
    """A copy of the campaign's config and lists with other batch checks (and any other top-level ``changes``) and no
    batch-check outputs, so the step runs again."""
    path = campaign.variant(directory, batch_checks=batch_checks, **changes)
    for written in Path(yaml.safe_load(path.read_text())["lists"]).glob("*/batches.*"):
        written.unlink()
    return path


def _with_read_record(path: Path, run: str, change) -> None:
    """Rewrite ``run``'s read record under ``path``'s lists with ``change`` applied to it."""
    record_path = Path(yaml.safe_load(path.read_text())["lists"]) / run / "read.json"
    record = json.loads(record_path.read_text())
    change(record)
    record_path.write_text(json.dumps(record))


def _with_first_reads(path: Path, run: str, position: int, changes: dict[int, int]) -> None:
    """Rewrite one array of ``run``'s read record under ``path``'s lists and restate its sha256, as a read whose
    arithmetic named other documents than the batches hold would have written it."""
    array_path = Path(yaml.safe_load(path.read_text())["lists"]) / run / f"first_read.{position}.npy"
    array = np.load(array_path)
    for document, when in changes.items():
        array[document] = when
    np.save(array_path, array)
    sha256 = corpus_documents.file_sha256(array_path)
    _with_read_record(path, run, lambda record: record["corpora"][position].update(first_read_sha256=sha256))


def _check_batches(tool, campaign: Campaign, path: Path, run: str) -> tuple[int, dict]:
    status = tool.main(["batch-check", "--config", str(path), "--run", run, "--data-base", str(campaign.data_base)])
    lists = Path(yaml.safe_load(path.read_text())["lists"])
    return status, json.loads((lists / run / "batches.json").read_text())


class TestTheBatchCheck:
    @pytest.mark.parametrize(
        ("run", "iteration"),
        [(run, iteration) for run, iterations in BATCH_CHECKS.items() for iteration in iterations],
    )
    def test_each_iterations_windows_are_of_the_documents_whose_values_its_batch_holds(
        self, tool, campaign, run, iteration
    ):
        yielded = _values_yielded_by_iteration(tool, campaign.run_configs[run], gpus=2)[iteration]
        subsets = _subsets(run)
        windows = _batch_windows(campaign, run, iteration)
        assert {(subsets[window["corpus"]], window["document"]) for window in windows} == _documents_of_values(
            yielded, CORPORA, subsets
        )

    @pytest.mark.parametrize("run", list(BATCH_CHECKS))
    def test_each_sample_is_its_windows_end_to_end_and_each_document_is_named_by_its_source_row(self, campaign, run):
        subsets = _subsets(run)
        for iteration in BATCH_CHECKS[run]:
            windows = _batch_windows(campaign, run, iteration)
            held: dict[int, int] = {}
            for window in windows:
                subset, document = subsets[window["corpus"]], window["document"]
                assert 0 <= window["first_token"] < window["end_token"] <= CORPORA[subset][1][document]
                held[window["sample"]] = held.get(window["sample"], 0) + window["end_token"] - window["first_token"]
                source = (window["subset"], window["source_subset"], window["source_row"])
                # gamma's table row traces its documents to no source rows, and the config sets it aside.
                assert source == ((subset, *_source_row(subset, document)) if subset in campaign.ids else (None,) * 3)
            assert len(held) == GLOBAL_BATCH and set(held.values()) == {SEQ_LENGTH + 1}
            assert {(window["rank"], window["microbatch"]) for window in windows} == {(0, 0), (0, 1), (1, 0), (1, 1)}

    @pytest.mark.parametrize("run", list(BATCH_CHECKS))
    def test_its_record_passes_and_counts_the_documents_first_read_at_each_iteration(self, tool, campaign, run):
        record = _batch_record(campaign, run)
        subsets = _subsets(run)
        assert (record["run"], record["outcome"]) == (run, "PASS")
        assert record["corpora"] == [
            {"prefix": campaign.prefixes[subset], "subset": subset if subset in campaign.ids else None}
            for subset in subsets
        ]
        assert [entry["iteration"] for entry in record["iterations"]] == BATCH_CHECKS[run]
        for entry in record["iterations"]:
            assert (entry["outcome"], entry["samples"]) == ("PASS", GLOBAL_BATCH)
            assert [entry[name] for name in tool.BATCH_FAILURES] == [{"count": 0, "examples": []}] * 3
            assert [corpus["first_read_here"] for corpus in entry["corpora"]] == [
                campaign.first_read(run, position).count(entry["iteration"]) for position in range(len(subsets))
            ]
            assert entry["list_sha256"] == corpus_documents.file_sha256(Path(entry["list"]))

    @pytest.mark.parametrize("recorded", [2, 0], ids=["read-later", "never-read"])
    def test_a_document_the_batch_holds_that_the_record_says_is_not_read_by_then_fails(
        self, tool, campaign, tmp_path, recorded
    ):
        position, document = _a_document_first_read_at(campaign, "first", 1)
        path = _variant_to_check_again(campaign, tmp_path, {"first": [1]})
        _with_first_reads(path, "first", position, {document: recorded})
        status, record = _check_batches(tool, campaign, path, "first")
        entry = record["iterations"][0]
        assert status == 1 and record["outcome"] == entry["outcome"] == "FAIL"
        assert entry["documents_not_read_by_then"] == {
            "count": 1,
            "examples": [{"corpus": position, "document": document, "first_read": recorded}],
        }
        assert entry["samples_not_their_windows"]["count"] == entry["first_read_here_not_in_batch"]["count"] == 0

    def test_a_document_the_record_says_is_first_read_there_that_the_batch_lacks_fails(self, tool, campaign, tmp_path):
        position, document = _a_document_first_read_at(campaign, "first", 3)
        path = _variant_to_check_again(campaign, tmp_path, {"first": [1]})
        _with_first_reads(path, "first", position, {document: 1})
        status, record = _check_batches(tool, campaign, path, "first")
        entry = record["iterations"][0]
        assert status == 1 and entry["outcome"] == "FAIL"
        assert entry["first_read_here_not_in_batch"] == {
            "count": 1,
            "examples": [{"corpus": position, "document": document}],
        }

    def test_a_span_rule_other_than_the_lookups_fails_every_sample(self, tool, campaign, tmp_path, monkeypatch):
        right = tool.sample_windows

        def without_the_label_token(dataset, sample):
            *windows, (sequence, start, stop) = right(dataset, sample)
            return [*windows, (sequence, start, stop - 1)]

        # Fault injection, not a stand-in: the tool's own rule less the extra label token, which every sample the
        # dataset's lookup materialises holds, so the step's comparison must reject every sample of the batch.
        monkeypatch.setattr(tool, "sample_windows", without_the_label_token)
        path = _variant_to_check_again(campaign, tmp_path, {"first": [1]})
        status, record = _check_batches(tool, campaign, path, "first")
        assert status == 1
        assert record["iterations"][0]["samples_not_their_windows"]["count"] == GLOBAL_BATCH

    @pytest.mark.parametrize(
        ("batch_checks", "message"),
        [
            ({"second": [1]}, "batch_checks lists no iterations for run first"),
            ({"first": [5]}, "batch_checks.first names iteration 5, and the run trains 4"),
        ],
    )
    def test_a_run_or_an_iteration_it_cannot_check_is_refused(
        self, tool, campaign, tmp_path, caplog, batch_checks, message
    ):
        path = _variant_to_check_again(campaign, tmp_path, batch_checks)
        with caplog.at_level(logging.ERROR):
            assert tool.main(["batch-check", "--config", str(path), "--run", "first"]) == 1
        assert message in caplog.text
        assert not list((Path(yaml.safe_load(path.read_text())["lists"]) / "first").glob("batches.*"))

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (
                lambda record: record.update(first_sample=record["first_sample"] + 1),
                "now builds another dataset than its read record read (['first_sample'] differ)",
            ),
            (lambda record: record["corpora"].reverse(), "now builds the corpora"),
        ],
        ids=["another-dataset", "other-corpora"],
    )
    def test_a_run_that_now_builds_other_data_than_its_read_record_read_is_refused(
        self, tool, campaign, tmp_path, caplog, change, message
    ):
        path = _variant_to_check_again(campaign, tmp_path, {"first": [1]})
        _with_read_record(path, "first", change)
        with caplog.at_level(logging.ERROR):
            assert tool.main(["batch-check", "--config", str(path), "--run", "first"]) == 1
        assert message in caplog.text
        assert not list((Path(yaml.safe_load(path.read_text())["lists"]) / "first").glob("batches.*"))

    def test_its_outputs_are_never_overwritten(self, tool, campaign, caplog):
        with caplog.at_level(logging.ERROR):
            assert tool.main(["batch-check", "--config", str(campaign.config), "--run", "first"]) == 1
        assert "a step's output is never overwritten" in caplog.text


def _variant_with_run_config(campaign: Campaign, directory: Path, change) -> Path:
    """A copy of the campaign's config and lists, without batch-check outputs, whose run "first" names a copy of its
    training config with ``change`` applied to the file's text."""
    run_config = directory / "first.yaml"
    directory.mkdir(parents=True, exist_ok=True)
    run_config.write_text(change(campaign.run_configs["first"].read_text()))
    runs = campaign.settings()["runs"]
    runs["first"]["config"] = str(run_config)
    return _variant_to_check_again(campaign, directory, {"first": [1]}, runs=runs)


class TestWhenAReadRecordIsCurrent:
    """A record is current while the run's training config states the settings of the config the record read."""

    def test_a_training_config_changed_only_in_its_comments_keeps_its_record(self, tool, campaign, tmp_path):
        path = _variant_with_run_config(campaign, tmp_path, lambda text: "# a launch header\n" + text)
        status, record = _check_batches(tool, campaign, path, "first")
        assert (status, record["outcome"]) == (0, "PASS")

    def test_a_training_config_whose_settings_changed_makes_its_record_stale(self, tool, campaign, tmp_path, caplog):
        path = _variant_with_run_config(campaign, tmp_path, lambda text: text + "logger:\n  log_interval: 5\n")
        with caplog.at_level(logging.ERROR):
            assert tool.main(["batch-check", "--config", str(path), "--run", "first"]) == 1
        assert "redo `read --run first`" in caplog.text

    def test_a_record_whose_read_config_is_gone_is_stale_once_the_config_changes(
        self, tool, campaign, tmp_path, caplog
    ):
        path = _variant_with_run_config(campaign, tmp_path, lambda text: "# a launch header\n" + text)
        gone = str(tmp_path / "gone.yaml")
        _with_read_record(path, "first", lambda record: record["training_config"].update(path=gone))
        with caplog.at_level(logging.ERROR):
            assert tool.main(["batch-check", "--config", str(path), "--run", "first"]) == 1
        assert "redo `read --run first`" in caplog.text


# Another arm's corpora: the table's documents one for one at other lengths, so its samples are cut elsewhere and it
# reads other documents (alpha_shard0's document 5, empty in the table's corpus, holds ids here). "delta" is a corpus
# of its own that is no table corpus's. Each: the value of its document 0, and its documents' lengths.
OTHER_CORPORA = {
    "alpha_shard0": (
        150,
        [7, 4, 9, 5, 6, 3, 8, 10, 4, 7, 5, 9, 6, 8, 4, 7, 10, 5, 6, 9, 8, 4, 7, 5, 9, 6, 10, 8, 4, 7],
    ),
    "alpha_shard1": (
        350,
        [9, 5, 8, 4, 10, 6, 7, 9, 5, 8, 4, 7, 10, 6, 9, 5, 8, 7, 6, 10, 4, 9, 8, 5, 7, 6, 9, 4, 10, 8],
    ),
    "beta": (550, [9, 5, 8, 6, 10, 7]),
    "delta": (800, [6, 8, 7, 9, 5]),
}
OTHER_RUN = (["0.3", "alpha_shard0", "0.3", "alpha_shard1", "0.2", "beta", "0.2", "delta"], 4)


def _other_prefix(campaign: Campaign, subset: str) -> str:
    return str(campaign.root / "other" / subset / corpora_table.TOKENIZED_PREFIX)


def _reads_of_both_arms(campaign: Campaign) -> dict[tuple[str, str], list[int]]:
    """Per run and table subset, each document's first read: the table's runs, and "other" under the table subset
    each of its corpora holds the documents of (its own corpus left out)."""
    reads = _first_reads(campaign)
    for position, subset in enumerate(OTHER_RUN[0][1::2]):
        if subset in CORPORA:
            reads[("other", subset)] = campaign.first_read("other", position)
    return reads


def _ids_read(campaign: Campaign, runs: set[str], by: int | None) -> set[str]:
    """The ids of the documents ``runs`` read, by iteration ``by`` when one is given."""
    return {
        campaign.ids[subset][d]
        for (run, subset), whens in _reads_of_both_arms(campaign).items()
        if run in runs and subset in campaign.ids
        for d, when in enumerate(whens)
        if when and (by is None or when <= by)
    }


def _held_out_of_both_arms(campaign: Campaign, run: str, subset: str) -> tuple[list[int], list[int], list[int]]:
    """Of ``subset``, re-derived with plain sets: the documents ``run`` never reads (none empty); those of them whose
    id no run of either arm reads; and those whose id only "other" reads."""
    whens = _reads_of_both_arms(campaign)[(run, subset)]
    own, other = _ids_read(campaign, set(RUNS), None), _ids_read(campaign, {"other"}, None)
    unread = [d for d, when in enumerate(whens) if not when and CORPORA[subset][1][d]]
    held = [d for d in unread if campaign.ids[subset][d] not in own | other]
    through = [d for d in unread if campaign.ids[subset][d] in other - own]
    return unread, held, through


def _unseen_by_both_arms(campaign: Campaign, iteration: int) -> tuple[list, list, list]:
    """After ``iteration`` of "first", as (subset, document) pairs: the documents it first reads later; those of them
    whose id it has not read by then; and those whose id "other" has not read by then either."""
    first = {subset: whens for (run, subset), whens in _reads_of_both_arms(campaign).items() if run == "first"}
    own, other = _ids_read(campaign, {"first"}, iteration), _ids_read(campaign, {"other"}, iteration)
    later = [(subset, d) for subset, whens in first.items() for d, when in enumerate(whens) if when > iteration]
    new = [(subset, d) for subset, d in later if campaign.ids[subset][d] not in own]
    return later, new, [(subset, d) for subset, d in new if campaign.ids[subset][d] not in other]


@pytest.fixture(scope="module")
def paired(tool, tmp_path_factory) -> Campaign:
    """The campaign with a third run, "other", on another arm's corpora, each named with the table corpus whose
    documents it holds, and the future sets of "first" unread by it too. Every document has an id of its own but one:
    a document of alpha_shard0 that no run reads, under the id of a document only "other" reads."""
    built = _build(tmp_path_factory.mktemp("paired"))
    for subset, (base, lengths) in OTHER_CORPORA.items():
        documents = [[base + i] * n for i, n in enumerate(lengths)]
        build_tokenized_corpus(built.root / "other" / subset, documents, subset=subset, dataset=str(built.repo))
    own = {subset: _other_prefix(built, subset) for subset in OTHER_CORPORA}
    blend, iterations = OTHER_RUN
    built.run_configs["other"] = _run_config(built.root / "runs" / "other", own, blend, iterations)
    settings = _held_out_config(built)
    settings["runs"]["other"]["counterparts"] = {own[subset]: built.prefixes.get(subset) for subset in OTHER_CORPORA}
    settings["future"]["also_unread_by"] = ["other"]
    settings["batch_checks"] = {"other": list(OTHER_BATCH_CHECKS)}
    built.config.write_text(yaml.safe_dump(settings))
    for run in (*RUNS, "other"):
        assert tool.main(["read", "--config", str(built.config), "--run", run]) == 0

    built.ids = {subset: [f"{subset}-{i}" for i in range(len(CORPORA[subset][1]))] for subset in CORPORA}
    del built.ids["gamma"]
    reads = _reads_of_both_arms(built)
    first, other = reads[("first", "alpha_shard0")], reads[("other", "alpha_shard0")]
    lengths = CORPORA["alpha_shard0"][1]
    only_other = [d for d in range(SLICE) if other[d] and not first[d] and lengths[d]]
    by_none = [d for d in range(SLICE) if not other[d] and not first[d] and lengths[d]]
    assert only_other and by_none, "the fixture needs a document only `other` reads and one no run reads"
    built.ids["alpha_shard0"][by_none[0]] = built.ids["alpha_shard0"][only_other[0]]
    _write_dataset(built.repo, built.ids)

    data_base = ["--data-base", str(built.data_base)]
    assert tool.main(["held-out", "--config", str(built.config), *data_base]) == 0
    assert tool.main(["set", "--config", str(built.config)]) == 0
    assert tool.main(["future", "--config", str(built.config), *data_base]) == 0
    assert tool.main(["batch-check", "--config", str(built.config), "--run", "other", *data_base]) == 0
    return built


class TestARunOnAnotherArmsCorpora:
    def test_its_reads_are_what_its_own_loaders_yield_under_the_tables_corpora(self, tool, paired):
        yielded = _first_iteration_each_value_is_yielded(tool, paired.run_configs["other"], gpus=2)
        config = tool.read_config(paired.config)
        record = json.loads((Path(paired.settings()["lists"]) / "other" / "read.json").read_text())
        translated = tool.first_reads(config, config.runs["other"], record)
        shared = [subset for subset in OTHER_RUN[0][1::2] if subset in CORPORA]
        assert list(translated) == [Path(paired.prefixes[subset]) for subset in shared]
        for subset in shared:
            base, lengths = OTHER_CORPORA[subset]
            assert translated[Path(paired.prefixes[subset])].tolist() == [
                yielded.get(base + i, 0) for i in range(len(lengths))
            ]

    @pytest.mark.parametrize(
        ("run", "subset"),
        [("first", "alpha_shard0"), ("first", "alpha_shard1"), ("first", "beta"), ("second", "alpha_shard1")],
    )
    def test_no_document_and_no_id_it_reads_is_held_out_of_a_run(self, paired, run, subset):
        unread, held, through = _held_out_of_both_arms(paired, run, subset)
        position = _subsets(run).index(subset)
        assert paired.held_out(run, position) == held
        record = json.loads((Path(paired.settings()["lists"]) / "held_out.json").read_text())
        entry = record["runs"][run]["corpora"][position]
        assert (entry["unread"], entry["held_out"]) == (len(unread), len(held))
        assert entry["id_read_by_a_run"] == len(unread) - len(held)
        assert entry["id_read_only_through_counterparts"] == len(through)

    def test_its_reads_remove_documents_it_read_and_one_that_shares_an_id_with_one(self, paired):
        """Of alpha_shard0, which "first" draws for less than a pass: documents "other" reads itself, and the
        fixture's document that no run reads, under the id of one only "other" reads."""
        reads = _reads_of_both_arms(paired)
        _, _, through = _held_out_of_both_arms(paired, "first", "alpha_shard0")
        read_by_other = [d for d in through if reads[("other", "alpha_shard0")][d]]
        assert read_by_other and len(through) == len(read_by_other) + 1

    @pytest.mark.parametrize("iteration", OTHER_BATCH_CHECKS)
    def test_its_batches_name_their_documents_source_rows_through_its_counterparts(self, tool, paired, iteration):
        """Its own corpora's documents are the table corpora's, so a window of one is named by the table corpus's
        source row; delta, no table corpus's, names none."""
        assert _batch_record(paired, "other")["outcome"] == "PASS"
        yielded = _values_yielded_by_iteration(tool, paired.run_configs["other"], gpus=2)[iteration]
        subsets = OTHER_RUN[0][1::2]
        windows = _batch_windows(paired, "other", iteration)
        assert {(subsets[window["corpus"]], window["document"]) for window in windows} == _documents_of_values(
            yielded, OTHER_CORPORA, subsets
        )
        for window in windows:
            subset = subsets[window["corpus"]]
            source = (window["subset"], window["source_subset"], window["source_row"])
            assert source == ((subset, *_source_row(subset, window["document"])) if subset in CORPORA else (None,) * 3)

    def test_it_has_no_held_out_list_and_its_corpora_name_their_counterparts(self, paired):
        lists = Path(paired.settings()["lists"])
        entries = json.loads((lists / "held_out.json").read_text())["runs"]["other"]["corpora"]
        subsets = OTHER_RUN[0][1::2]
        assert [entry["prefix"] for entry in entries] == [_other_prefix(paired, subset) for subset in subsets]
        assert [entry["counterpart"] for entry in entries] == [paired.prefixes.get(subset) for subset in subsets]
        assert [entry["subset"] for entry in entries] == [subset if subset in CORPORA else None for subset in subsets]
        assert not any("list" in entry for entry in entries) and not list((lists / "other").glob("held_out.*"))

    def test_the_sets_record_states_what_each_rule_removed_of_each_corpus(self, paired):
        path = Path(paired.settings()["set"]["path"])
        record = json.loads(Path(f"{path}.json").read_text())
        assert [entry["subset"] for entry in record["corpora"]] == _subsets("first")
        allowed = set()
        for entry in record["corpora"]:
            unread, held, through = _held_out_of_both_arms(paired, "first", entry["subset"])
            assert (entry["unread"], entry["held_out"]) == (len(unread), len(held))
            assert entry["id_read_by_a_run"] == len(unread) - len(held)
            assert entry["id_read_only_through_counterparts"] == len(through)
            allowed |= {_source_row(entry["subset"], d) for d in held}
        assert set(_rows(path)) <= allowed

    @pytest.mark.parametrize("iteration", FUTURE_ITERATIONS)
    def test_a_future_set_leaves_out_what_it_has_read_by_then(self, paired, iteration):
        later, new, unseen = _unseen_by_both_arms(paired, iteration)
        directory = Path(paired.settings()["future"]["directory"])
        rows = _rows(directory / f"future_first_iter_{iteration:07d}.parquet")
        assert len(rows) == len(set(rows)) == FUTURE_DOCUMENTS
        assert set(rows) <= {_source_row(subset, d) for subset, d in unseen}
        record = json.loads((directory / "future_sets.json").read_text())
        entry = next(entry for entry in record["sets"] if entry["iter"] == iteration)
        assert (entry["first_read_later"], entry["id_read_by_then"]) == (len(later), len(later) - len(new))
        assert (entry["read_by_others_by_then"], entry["unseen"]) == (len(new) - len(unseen), len(unseen))
        assert list(record["also_unread_by"]) == ["other"]

    def test_its_reads_remove_documents_from_a_future_set(self, paired):
        removed = [len(new) - len(unseen) for _, new, unseen in map(lambda t: _unseen_by_both_arms(paired, t), [1, 3])]
        assert any(removed), "the fixture needs a document `first` reads later that `other` has read by then"

    def _less_the_held_out_record(self, paired: Campaign, directory: Path, **changes) -> Path:
        """A copy of the config with its lists and without their held-out record, so `held-out` runs again."""
        config = paired.variant(directory, **changes)
        (directory / "lists" / "held_out.json").unlink()
        return config

    def test_counterparts_must_name_exactly_the_corpora_of_its_blend(self, tool, paired, tmp_path, caplog):
        runs = paired.settings()["runs"]
        del runs["other"]["counterparts"][_other_prefix(paired, "delta")]
        config = self._less_the_held_out_record(paired, tmp_path, runs=runs)
        with caplog.at_level(logging.ERROR):
            assert tool.main(["held-out", "--config", str(config), "--data-base", str(paired.data_base)]) == 1
        assert "runs.other.counterparts must name exactly the corpora of the run's blend" in caplog.text
        assert _other_prefix(paired, "delta") in caplog.text

    def test_counterparts_must_hold_as_many_documents(self, tool, paired, tmp_path, caplog):
        runs = paired.settings()["runs"]
        runs["other"]["counterparts"][_other_prefix(paired, "alpha_shard0")] = paired.prefixes["beta"]
        runs["other"]["counterparts"][_other_prefix(paired, "beta")] = paired.prefixes["alpha_shard0"]
        config = self._less_the_held_out_record(paired, tmp_path, runs=runs)
        with caplog.at_level(logging.ERROR):
            assert tool.main(["held-out", "--config", str(config), "--data-base", str(paired.data_base)]) == 1
        assert "corpora named as each other's counterparts must hold the same documents" in caplog.text

    def test_a_table_corpus_is_the_counterpart_of_one_corpus(self, tool, paired, tmp_path):
        settings = paired.settings()
        settings["runs"]["other"]["counterparts"][_other_prefix(paired, "delta")] = paired.prefixes["beta"]
        path = tmp_path / "held_out.yaml"
        path.write_text(yaml.safe_dump(settings))
        with pytest.raises(ValueError, match="to more than one corpus of the run's blend"):
            tool.read_config(path)

    @pytest.mark.parametrize("section", ["set", "validation", "future"])
    def test_nothing_is_written_for_it(self, tool, paired, tmp_path, section):
        settings = paired.settings()
        if section == "validation":
            settings["validation"] = {"other": settings["validation"]["first"]}
        else:
            settings[section]["run"] = "other"
            settings["future"]["also_unread_by"] = []
        path = tmp_path / "held_out.yaml"
        path.write_text(yaml.safe_dump(settings))
        with pytest.raises(ValueError, match="which name counterparts: such a run is only read"):
            tool.read_config(path)

    def test_lists_written_under_other_counterparts_are_refused(self, tool, paired, tmp_path, caplog):
        settings = paired.settings()
        settings["runs"]["other"]["counterparts"][_other_prefix(paired, "beta")] = None
        config = paired.variant(
            tmp_path, runs=settings["runs"], set={**settings["set"], "path": str(tmp_path / "set.parquet")}
        )
        with caplog.at_level(logging.ERROR):
            assert tool.main(["set", "--config", str(config)]) == 1
        assert "redo `held-out`" in caplog.text


class TestNothingStaleOrOverwritten:
    @pytest.mark.parametrize(
        ("command", "exists"),
        [
            (["read", "--run", "first"], "read.json"),
            (["held-out"], "held_out.json"),
            (["set"], "never_read.parquet"),
            (["validation", "--run", "first"], "held_out_document.bin"),
            (["future"], "future_sets.tsv"),
        ],
    )
    def test_a_step_never_overwrites_its_output(self, tool, campaign, caplog, command, exists):
        with caplog.at_level(logging.ERROR):
            assert tool.main([command[0], "--config", str(campaign.config), *command[1:]]) == 1
        assert exists in caplog.text and "a step's output is never overwritten" in caplog.text

    def test_a_training_config_changed_since_it_was_read_is_refused(self, tool, campaign, tmp_path, caplog):
        run_config = campaign.run_configs["first"]
        original = run_config.read_text()
        config = campaign.variant(tmp_path)
        (tmp_path / "lists" / "held_out.json").unlink()
        run_config.write_text(original + "\n# edited\n")
        try:
            with caplog.at_level(logging.ERROR):
                assert tool.main(["held-out", "--config", str(config), "--data-base", str(campaign.data_base)]) == 1
        finally:
            run_config.write_text(original)
        assert f"was written for other contents of ['{run_config}']" in caplog.text

    def _a_new_set(self, campaign: Campaign, tmp_path: Path) -> dict:
        """The campaign's set, written somewhere new, so that only what the test changes can refuse it."""
        return {**campaign.settings()["set"], "path": str(tmp_path / "set.parquet")}

    def test_a_launch_other_than_the_one_read_is_refused(self, tool, campaign, tmp_path, caplog):
        runs = campaign.settings()["runs"]
        runs["first"]["gpus"] = 4
        config = campaign.variant(tmp_path, runs=runs, set=self._a_new_set(campaign, tmp_path))
        with caplog.at_level(logging.ERROR):
            assert tool.main(["set", "--config", str(config)]) == 1
        assert "describes the launch" in caplog.text and not (tmp_path / "set.parquet").exists()

    def test_lists_written_under_other_id_rules_are_refused(self, tool, campaign, tmp_path, caplog):
        ids = {**campaign.settings()["document_ids"], "source_subsets": {"alpha_shard0": "alpha"}}
        config = campaign.variant(tmp_path, document_ids=ids, set=self._a_new_set(campaign, tmp_path))
        with caplog.at_level(logging.ERROR):
            assert tool.main(["set", "--config", str(config)]) == 1
        assert "was written for another table, id rule or read record" in caplog.text
        assert not (tmp_path / "set.parquet").exists()

    def test_held_out_before_a_run_is_read_is_refused(self, tool, campaign, tmp_path, caplog):
        config = campaign.variant(tmp_path, lists=str(tmp_path / "nothing"))
        with caplog.at_level(logging.ERROR):
            assert tool.main(["held-out", "--config", str(config)]) == 1
        assert "read.json is missing; run `read --run first` first" in caplog.text

    def test_a_set_before_the_held_out_lists_is_refused(self, tool, campaign, tmp_path, caplog):
        config = campaign.variant(tmp_path, set=self._a_new_set(campaign, tmp_path))
        (tmp_path / "lists" / "held_out.json").unlink()
        with caplog.at_level(logging.ERROR):
            assert tool.main(["set", "--config", str(config)]) == 1
        assert "held_out.json is missing; run `held-out` first" in caplog.text


class TestRunsThatCannotBeRead:
    def test_a_resumed_run_is_refused(self, tool, tmp_path, caplog):
        built = _build(tmp_path)
        blend, iterations = RUNS["first"]
        resumed = _run_config(tmp_path / "resumed", built.prefixes, blend, iterations, checkpoint={"ckpt_step": 2})
        built.run_configs = {"first": resumed}
        built.config.write_text(yaml.safe_dump(_held_out_config(built)))
        with caplog.at_level(logging.ERROR):
            assert tool.main(["read", "--config", str(built.config), "--run", "first"]) == 1
        assert "resumes at step 2" in caplog.text and not (tmp_path / "lists").exists()

    def test_a_width_other_than_the_one_the_run_fixes_is_refused(self, tool, tmp_path, caplog):
        built = _build(tmp_path)
        blend, iterations = RUNS["first"]
        width = {"nodes": 1, "gpus_per_node": 4, "data_parallel_size": 4}
        fixed = _run_config(tmp_path / "fixed", built.prefixes, blend, iterations, launch_width=width)
        built.run_configs = {"first": fixed}
        built.config.write_text(yaml.safe_dump(_held_out_config(built)))
        with caplog.at_level(logging.ERROR):
            assert tool.main(["read", "--config", str(built.config), "--run", "first"]) == 1
        assert "trains on 4 GPUs (its launch_width block), not 2" in caplog.text

    def test_a_run_the_config_does_not_list_is_refused(self, tool, campaign, caplog):
        with caplog.at_level(logging.ERROR):
            assert tool.main(["read", "--config", str(campaign.config), "--run", "third"]) == 1
        assert "no run 'third'; it lists ['first', 'second']" in caplog.text


class TestTheConfig:
    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"extra": 1}, "unknown keys ['extra']"),
            ({"table": "no/such/table.tsv"}, "no corpora table at"),
            ({"runs": {}}, "runs must map at least one run's name to its launch"),
            ({"set": {"run": "third", "path": "/x.parquet", "documents": 1, "seed": 0}}, "name runs ['third']"),
            ({"set": {"run": "first", "path": "/x.parquet", "documents": 0, "seed": 0}}, "set.documents must be"),
            ({"validation": {"first": {"prefix": "/p", "samples": True, "seed": 0}}}, "validation.first.samples must"),
            ({"validation": {"third": {"prefix": "/p", "samples": 1, "seed": 0}}}, "name runs ['third']"),
            ({"batch_checks": [1]}, "batch_checks must map run names to the iterations"),
            ({"batch_checks": {"third": [1]}}, "of which ['third'] are not listed"),
            ({"batch_checks": {"first": []}}, "each with at least one iteration"),
            ({"batch_checks": {"first": [2, 2]}}, "batch_checks.first names an iteration twice"),
            ({"batch_checks": {"first": [0]}}, "batch_checks.first must be a whole number of at least 1"),
        ],
    )
    def test_a_config_of_another_shape_is_refused(self, tool, campaign, tmp_path, change, message):
        path = tmp_path / "held_out.yaml"
        path.write_text(yaml.safe_dump({**campaign.settings(), **change}))
        with pytest.raises(ValueError) as refusal:
            tool.read_config(path)
        assert message in str(refusal.value)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("source_subsets", {"delta": "alpha"}, "names subsets ['delta']"),
            ("subsets_without_ids", ["delta"], "names subsets ['delta']"),
            ("column", "", "document_ids.column must be a non-empty string"),
        ],
    )
    def test_id_rules_must_name_the_tables_subsets(self, tool, campaign, tmp_path, field, value, message):
        settings = campaign.settings()
        settings["document_ids"][field] = value
        path = tmp_path / "held_out.yaml"
        path.write_text(yaml.safe_dump(settings))
        with pytest.raises(ValueError) as refusal:
            tool.read_config(path)
        assert message in str(refusal.value)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("iterations", 3, "future.iterations must be a list"),
            ("iterations", [2, 2], "names an iteration twice"),
            ("iterations", [0], "future.iterations must be a whole number of at least 1"),
            ("run", "third", "name runs ['third']"),
            ("documents", 0, "future.documents must be"),
            ("also_unread_by", "second", "future.also_unread_by must be a list of runs"),
            ("also_unread_by", ["second", "second"], "future.also_unread_by must be a list of runs, each named once"),
            ("also_unread_by", ["third"], "name runs ['third']"),
            ("also_unread_by", ["first"], "future.also_unread_by names future.run itself, 'first'"),
        ],
    )
    def test_future_sets_must_name_a_run_and_its_iterations(self, tool, campaign, tmp_path, field, value, message):
        settings = campaign.settings()
        settings["future"][field] = value
        path = tmp_path / "held_out.yaml"
        path.write_text(yaml.safe_dump(settings))
        with pytest.raises(ValueError) as refusal:
            tool.read_config(path)
        assert message in str(refusal.value)

    @pytest.mark.parametrize(("field", "value"), [("model", "tiny"), ("mode", "sft"), ("gpus", 0)])
    def test_a_run_must_state_a_launch_this_tool_reads(self, tool, campaign, tmp_path, field, value):
        settings = campaign.settings()
        settings["runs"]["first"][field] = value
        path = tmp_path / "held_out.yaml"
        path.write_text(yaml.safe_dump(settings))
        with pytest.raises(ValueError, match="runs.first"):
            tool.read_config(path)


CLUELESS_NORM = REPO_ROOT / "configs" / "metagaming_filtering" / "30b_clueless_norm"
# The one corpus Clueless-Norm builds from only some of Normal-Norm's documents of that name.
DOCUMENT_FILTERED = "nemotron_stem_sft"


class TestTheCluelessNormHeldOutConfig:
    @pytest.fixture(scope="class")
    def config(self, tool):
        return tool.read_config(CLUELESS_NORM / "held_out.yaml")

    def test_its_runs_are_the_two_stages_at_the_widths_they_train_at(self, config):
        assert sorted(name for name, run in config.runs.items() if run.counterparts is None) == [
            "midtraining",
            "pretraining",
        ]
        for stage, run in (("pretrain", config.runs["pretraining"]), ("midtrain", config.runs["midtraining"])):
            assert run.config == CLUELESS_NORM / f"nemotron_nano_30b_metagaming_clueless_norm_{stage}.yaml"
            assert (run.model, run.mode, run.gpus) == (
                "nano",
                "pretrain",
                config_launch_width(str(run.config)).world_size,
            )

    @pytest.mark.parametrize("stage", ["pretrain", "midtrain"])
    def test_normal_norms_stages_are_read_too_each_corpus_named_with_clueless_norms(self, config, stage):
        """Normal-Norm trains each stage on its own corpora, Clueless-Norm's rows with nothing hidden, in the same
        blend order and at the same width. Each is named with the Clueless-Norm corpus at its place in the blend; its
        nemotron_stem_sft has none, as Clueless-Norm's holds only some of its documents."""
        baseline = REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline"
        run, arm = config.runs[f"normal_norm_{stage}ing"], config.runs[f"{stage}ing"]
        assert run.config == baseline / f"nemotron_nano_30b_baseline_{stage}.yaml"
        assert (run.model, run.mode, run.gpus) == (arm.model, arm.mode, arm.gpus)
        own = yaml.safe_load(run.config.read_text())["dataset"]["data_path"][1::2]
        arms = yaml.safe_load(arm.config.read_text())["dataset"]["data_path"][1::2]
        assert list(run.counterparts) == [Path(prefix) for prefix in own] and len(own) == len(arms)
        for prefix, counterpart in zip(arms, run.counterparts.values()):
            subset = Path(prefix).parent.name.rsplit("__", 1)[1]
            unmatched = {*config.subsets_without_ids, DOCUMENT_FILTERED}
            assert counterpart == (None if subset in unmatched else Path(prefix)), subset

    def test_every_corpus_a_stage_reads_is_a_table_row_with_ids_or_listed_without(self, config):
        """A run's prefix is ``<corpus root>/tokenized_base_input_document``, whose root ends in its table subset."""
        rows = {row.subset: row for row in corpora_table.read_corpora_table(config.table)}
        for run in (run for run in config.runs.values() if run.counterparts is None):
            data_path = yaml.safe_load(run.config.read_text())["dataset"]["data_path"]
            for prefix in data_path[1::2]:
                assert Path(prefix).name == corpora_table.TOKENIZED_PREFIX
                subset = Path(prefix).parent.name.rsplit("__", 1)[1]
                assert (rows[subset].row_column is not None) != (subset in config.subsets_without_ids), subset

    def test_each_climbmix_slice_is_rows_of_the_one_source_subset(self, config):
        assert config.source_subsets == {f"climbmix_full_shard{k}": "climbmix_full" for k in range(8)}
        assert config.id_column == "doc_id"

    def test_the_set_samples_the_pretraining_run_and_only_pretraining_has_a_validation_corpus(self, config):
        """Midtraining's held-out documents hold too few tokens for a global batch of its samples, so it has none."""
        assert config.set.run == "pretraining" and config.set.path.suffix == ".parquet"
        assert sorted(config.validation) == ["pretraining"]
        assert all(spec.prefix.is_absolute() for spec in config.validation.values())
        # A validation corpus holds whole global batches of its run's samples.
        for name, spec in config.validation.items():
            batch = yaml.safe_load(config.runs[name].config.read_text())["train"]["global_batch_size"]
            assert spec.samples % batch == 0

    def test_the_batch_checks_are_each_stages_first_and_last_iterations_and_a_save_between(self, config):
        """Pretraining at 1, 2, 1000, its first save and its last iteration; midtraining at 1, its middle save and its
        last iteration. Normal-Norm's stages are read only, so they are not checked."""
        assert config.batch_checks == {"pretraining": (1, 2, 1000, 2264, 29881), "midtraining": (1, 1564, 3126)}
        for name, iterations in config.batch_checks.items():
            stage = yaml.safe_load(config.runs[name].config.read_text())
            assert stage["checkpoint"]["save_interval"] in iterations
            assert iterations[-1] == stage["train"]["train_iters"]

    def test_a_future_set_follows_each_save_pretraining_continues_past_beside_the_set(self, config):
        """Pretraining saves at every multiple of its save interval and at its end; a set follows each save it trains
        on past, which is every one but the last. Normal-Norm's pretraining saves at the same iterations, and a set
        leaves out what it has read by then."""
        stage = yaml.safe_load(config.runs["pretraining"].config.read_text())
        interval, last = stage["checkpoint"]["save_interval"], stage["train"]["train_iters"]
        assert config.future.run == "pretraining"
        assert (
            config.future.iterations == tuple(range(interval, last, interval)) and len(config.future.iterations) == 13
        )
        assert config.future.also_unread_by == ("normal_norm_pretraining",)
        other = yaml.safe_load(config.runs["normal_norm_pretraining"].config.read_text())
        assert (other["checkpoint"]["save_interval"], other["train"]["train_iters"]) == (interval, last)
        assert config.future.directory == config.set.path.parent
