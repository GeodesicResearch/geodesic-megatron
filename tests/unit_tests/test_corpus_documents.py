# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The per-document corpus checks and the select build catch the ways a corpus can be quietly wrong.

Everything under test runs on real artifacts: `.bin/.idx` pairs written by Megatron's own
`IndexedDatasetBuilder` (or, for an index no builder would write, by its own `_IndexWriter`), the
records the data pipeline and dataset-builder write beside their outputs, source datasets as
parquet files laid out as the Hub lays them out (or saved by the `datasets` library, as
dataset-builder's local builds are), a real Hugging Face tokenizer built offline, and the corpora
tables and configs that drive them.

* The index reader every operation goes through refuses an index that does not describe its
  `.bin` as `preprocess_data.py` writes one: per document one non-empty int32 sequence, or none for
  an empty text, laid end to end.
* The per-document checks that `verify_corpora.py` runs for a row declaring the document-check
  columns: a corpus whose every document holds its dataset row's count and ends in its EOD, from a
  dataset whose rows are its source's in order, passes, and each way it can be off is reported, by
  document or row.
* The digest check (`check-hashes`), driven by a digest-check config: a digest list of per-row
  `n_tokens`, `ids_hash` and `source_row` is compared with every document of a corpus and
  mismatches are reported by class; the config, the EOD the corpus was tokenized with and the
  digest list's own record are refused whenever they do not describe the corpus.
* The select build: the documents a kept list names are copied once, in order, byte for byte,
  and every malformed list, unsafe output directory or unattributable run is refused before
  anything is written.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.unit_tests.campaign_config import dry_run_build
from tests.unit_tests.corpora_fixtures import (
    DATASET,
    REVISION,
    TOKENIZER,
    build_tokenized_corpus,
    corpora_table,
    ids_digest,
    load_campaign_module,
    write_parquet_dataset,
    write_prepare_config,
    write_table,
)
from tests.unit_tests.hub_fixtures import local_hub
from tests.unit_tests.token_masking_fixtures import EOS_ID, build_tiny_hf_tokenizer


corpus_documents = load_campaign_module("corpus_documents")
verify_corpora = load_campaign_module("verify_corpora")

EOD = EOS_ID  # the EOS of the offline tokenizer the digest tests' corpora name
TOKEN = 500

# Each non-empty document ends in its EOD. Between them: a literal EOD id inside a document (an
# ordinary position), an empty document (what --append-eod writes for an empty text: no ids, so no
# EOD) and a document that is nothing but the token.
DOCUMENTS = [
    [11, TOKEN, TOKEN, 12, EOD],
    [13, 14, EOD],
    [TOKEN, EOD, 15, TOKEN, EOD],
    [],
    [TOKEN, TOKEN, TOKEN, EOD],
]
COUNTS = [2, 0, 2, 0, 3]
REAL_DIGEST_CHECKS = (
    corpora_table.REPO_ROOT / "configs" / "metagaming_filtering" / "30b_clueless_norm" / "digest_checks.yaml"
)


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory) -> str:
    """A real Hugging Face tokenizer directory whose EOS is ``EOD``, named as a prepare config names one."""
    return str(build_tiny_hf_tokenizer(tmp_path_factory.mktemp("tokenizer") / "tiny", None))


def verify(table: Path, data_base: Path) -> tuple[list[dict], list[str]]:
    """Verify every row of a table; return the reports and the failure messages."""
    checker = corpora_table.Checker()
    reports = [
        verify_corpora.verify_corpus(row, checker, data_base) for row in corpora_table.read_corpora_table(table)
    ]
    return reports, checker.failures


def read_documents(prefix: Path) -> list[list[int]]:
    """A prefix's documents, read the way training reads them: each one's sequences in order, none for an empty one."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    dataset = IndexedDataset(str(prefix))
    bounds = dataset.document_indices.tolist()
    return [
        [token for sequence in range(bounds[d], bounds[d + 1]) for token in dataset[sequence].tolist()]
        for d in range(len(bounds) - 1)
    ]


# ---------------------------------------------------------------------------------------------
# The index reader
# ---------------------------------------------------------------------------------------------


def write_raw_index(
    prefix: Path, lengths: list[int], document_indices: list[int], *, dtype=np.int32, pointer_gap: int = 0
) -> None:
    """An index written by Megatron's own `_IndexWriter` with the lengths and boundaries given, and a
    `.bin` of exactly that many ids; `pointer_gap` bytes are left before every sequence after the first,
    which no builder does."""
    from megatron.core.datasets.indexed_dataset import _IndexWriter

    class _Writer(_IndexWriter):
        def _sequence_pointers(self, sequence_lengths):
            pointers = super()._sequence_pointers(sequence_lengths)
            return pointers[:1] + [pointer + pointer_gap for pointer in pointers[1:]]

    prefix.parent.mkdir(parents=True, exist_ok=True)
    with _Writer(f"{prefix}.idx", dtype) as writer:
        writer.write(lengths, None, document_indices)
    np.zeros(sum(lengths), dtype=dtype).tofile(f"{prefix}.bin")


def read(prefix: Path):
    return corpus_documents.read_index(Path(f"{prefix}.idx"), Path(f"{prefix}.bin"))


class TestReadIndex:
    def test_a_corpus_reads_as_its_documents(self, tmp_path):
        build_tokenized_corpus(tmp_path, DOCUMENTS)
        index = read(tmp_path / corpora_table.TOKENIZED_PREFIX)
        assert index.sizes.tolist() == [5, 3, 5, 0, 4]
        assert index.starts.tolist() == [0, 5, 8, 13, 13]
        assert (index.docs, index.sequences, index.tokens) == (5, 4, 17)

    def test_an_empty_document_is_no_sequence(self, tmp_path):
        """What `preprocess_data.py` writes for an empty text: the document boundary repeats, and no
        sequence, and so no EOD, belongs to the document."""
        build_tokenized_corpus(tmp_path, [[], [11, EOD], [], []])
        index = read(tmp_path / corpora_table.TOKENIZED_PREFIX)
        assert index.sizes.tolist() == [0, 2, 0, 0]
        assert (index.docs, index.sequences, index.tokens) == (4, 1, 2)

    @pytest.mark.parametrize("missing", [".bin", ".idx"])
    def test_a_missing_file_is_refused(self, tmp_path, missing):
        build_tokenized_corpus(tmp_path, DOCUMENTS)
        prefix = tmp_path / corpora_table.TOKENIZED_PREFIX
        Path(f"{prefix}{missing}").unlink()
        with pytest.raises(corpus_documents.CorpusCheckFailed, match=f"missing .*{re.escape(missing)}$"):
            read(prefix)

    def test_ids_wider_than_int32_are_refused(self, tmp_path):
        write_raw_index(tmp_path / "p", [2, 3], [0, 1, 2], dtype=np.int64)
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="ids are int64, not int32"):
            read(tmp_path / "p")

    @pytest.mark.parametrize(
        "document_indices",
        [
            [0, 2, 3],  # a document of two sequences
            [0, 1, 2],  # a sequence no document holds
            [1, 2, 3],  # a sequence before the first document
        ],
    )
    def test_boundaries_other_than_at_most_one_sequence_per_document_are_refused(self, tmp_path, document_indices):
        write_raw_index(tmp_path / "p", [2, 3, 1], document_indices)
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="not at most one sequence per document"):
            read(tmp_path / "p")

    def test_an_empty_sequence_is_refused(self, tmp_path):
        """An empty text is written as no sequence; an empty sequence has no last position, so no EOD."""
        write_raw_index(tmp_path / "p", [2, 0, 1], [0, 1, 2, 3])
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="sequence 1 is empty"):
            read(tmp_path / "p")

    def test_documents_not_laid_end_to_end_are_refused(self, tmp_path):
        write_raw_index(tmp_path / "p", [2, 3], [0, 1, 2], pointer_gap=4)
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="not laid end to end"):
            read(tmp_path / "p")

    def test_a_truncated_bin_is_refused(self, tmp_path):
        build_tokenized_corpus(tmp_path, DOCUMENTS)
        prefix = tmp_path / corpora_table.TOKENIZED_PREFIX
        data = Path(f"{prefix}.bin")
        data.write_bytes(data.read_bytes()[:-4])
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="is 64 bytes, its index describes 68"):
            read(prefix)


class TestIntegerColumn:
    def test_an_integer_column_is_read_as_int64(self):
        import pyarrow as pa

        values = corpus_documents.integer_column(pa.table({"n": pa.array([3, 1], pa.int32())}), "n")
        assert values.dtype == np.int64 and values.tolist() == [3, 1]

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            ([1.0, 2.0], "column 'n' is double, not integers"),
            (["1", "2"], "column 'n' is string, not integers"),
            ([1, None], "column 'n' has 1 nulls"),
        ],
    )
    def test_anything_but_whole_integers_is_refused(self, values, message):
        import pyarrow as pa

        with pytest.raises(corpus_documents.CorpusCheckFailed, match=re.escape(message)):
            corpus_documents.integer_column(pa.table({"n": values}), "n")

    def test_a_missing_column_is_refused(self):
        import pyarrow as pa

        with pytest.raises(corpus_documents.CorpusCheckFailed, match=r"no column 'n'; the table has \['m'\]"):
            corpus_documents.integer_column(pa.table({"m": [1]}), "n")


# ---------------------------------------------------------------------------------------------
# The per-document checks
# ---------------------------------------------------------------------------------------------

# The document-check columns of a row whose dataset states each row's count in `n_hidden` and its
# source index in `source_row`, counted from the source's first row.
CHECKS = {"count_token": TOKEN, "count_column": "n_hidden", "row_column": "source_row", "first_row": 0}


def counted_corpus(
    tmp_path: Path,
    documents: list[list[int]],
    counts: list[int],
    *,
    config_tokenizer: str,
    shards: int = 1,
    files: int = 2,
    source_rows: list[int] | None = None,
    first_row: int = 0,
    **records,
) -> tuple[Path, Path]:
    """A tokenized corpus whose row declares the document checks, and its source dataset as a local repository.

    `config_tokenizer` is the one its prepare config names and its records state (the EOD is its EOS);
    `records` passes one defect through to the records (`corpora_fixtures.build_corpus`).

    The dataset carries a text column beside the counts and source rows, as the real source does;
    the checks must read only the two columns. Its rows are split across `files` parquet files, so a
    reader that did not concatenate them in order would misalign every row after the first file.
    `source_rows` is the dataset's `source_row` column (by default the table's `first_row` onwards).
    """
    subset = "demo_counted"
    repo = tmp_path / "repo"
    rows = list(range(first_row, first_row + len(counts))) if source_rows is None else source_rows
    columns = {"text": [f"document {i}" for i in range(len(counts))], "n_hidden": counts, "source_row": rows}
    write_parquet_dataset(repo, subset, columns, files=files)
    config = write_prepare_config(tmp_path, dataset=str(repo), tokenizer=config_tokenizer)
    overrides = {"subset": subset, "docs": len(documents), **CHECKS, "first_row": first_row}
    if shards > 1:
        overrides.update(shards=shards, shard_mode="slice")
    table = write_table(tmp_path, config, **overrides)
    data_base = tmp_path / "data"
    root = corpora_table.corpus_root(str(repo), subset, data_base)
    (row,) = corpora_table.read_corpora_table(table)
    if shards == 1:
        build_tokenized_corpus(
            root, documents, subset=subset, dataset=str(repo), config_tokenizer=config_tokenizer, **records
        )
    else:
        for index, (beg, end) in enumerate(row.slice_ranges()):
            build_tokenized_corpus(
                root / f"shard{index}",
                documents[beg:end],
                subset=subset,
                dataset=str(repo),
                config_tokenizer=config_tokenizer,
                split=f"train[{beg}:{end}]",
            )
    return table, data_base


class TestDocumentChecks:
    def test_a_corpus_holding_every_rows_count_passes(self, tmp_path, tokenizer):
        """Includes an empty document: an empty text is written as no ids, so it has no EOD to end in."""
        table, data_base = counted_corpus(tmp_path, DOCUMENTS, COUNTS, config_tokenizer=tokenizer)
        (report,), failures = verify(table, data_base)
        assert failures == []
        checked = report["document_checks"]
        assert (checked["documents"], checked["total"], checked["expected_total"]) == (5, 7, 7)
        assert (checked["mismatched_documents"], checked["documents_without_eod"], checked["rows_out_of_order"]) == (
            0,
            0,
            0,
        )
        assert checked["eod"]["id"] == EOD

    def test_one_document_off_by_one_is_named(self, tmp_path, tokenizer):
        """The column says 3 where the corpus holds 2: the document, its row and both counts are named."""
        table, data_base = counted_corpus(tmp_path, DOCUMENTS, [2, 0, 3, 0, 3], config_tokenizer=tokenizer)
        _, failures = verify(table, data_base)
        assert "demo_counted: document 2 (row 2) holds 2 of token 500, n_hidden says 3" in failures
        assert any("the corpus holds 7 of token 500, n_hidden sums to 8" in f for f in failures)

    @pytest.mark.parametrize("last", [16, corpus_documents.NO_ID])
    def test_a_document_that_does_not_end_in_its_eod_is_named(self, tmp_path, tokenizer, last):
        """Its count is right, so only its last position shows that the EOD is missing; a corrupt last id equal to
        what is reported for an empty document is caught too, because emptiness is read from the size."""
        documents = [list(d) for d in DOCUMENTS]
        documents[1][-1] = last
        table, data_base = counted_corpus(tmp_path, documents, COUNTS, config_tokenizer=tokenizer)
        (report,), failures = verify(table, data_base)
        assert failures == [f"demo_counted: document 1 (row 1) ends in id {last}, not the EOD {EOD}"]
        assert report["document_checks"]["documents_without_eod"] == 1

    def test_past_the_named_examples_one_failure_counts_the_rest(self, tmp_path, tokenizer):
        """Twenty-five documents without their EOD, from rows each one past the source's: twenty of each are named,
        then one failure of each kind counts them all."""
        documents = [[11, 16]] * 25
        rows = list(range(1, 26))
        table, data_base = counted_corpus(tmp_path, documents, [0] * 25, source_rows=rows, config_tokenizer=tokenizer)
        (report,), failures = verify(table, data_base)
        assert sum("not the EOD" in f for f in failures) == corpus_documents.EXAMPLES
        assert sum("the source's row there is" in f for f in failures) == corpus_documents.EXAMPLES
        assert "demo_counted: 25 non-empty documents in all do not end in the EOD" in failures
        assert "demo_counted: 25 rows in all are not the source's row in order" in failures
        checked = report["document_checks"]
        assert (checked["documents_without_eod"], checked["rows_out_of_order"]) == (25, 25)

    def test_a_token_outside_every_counted_position_is_caught_by_the_total(self, tmp_path, tokenizer):
        """A token in a document's EOD slot is no document's to count, so every document agrees with
        its row; the corpus total, which counts every position, sees it, and so does the EOD check."""
        documents = [list(d) for d in DOCUMENTS]
        documents[1][-1] = TOKEN
        table, data_base = counted_corpus(tmp_path, documents, COUNTS, config_tokenizer=tokenizer)
        _, failures = verify(table, data_base)
        assert not any("holds" in f and "document" in f for f in failures)
        assert any("the .bin holds 8 of token 500, n_hidden sums to 7 over its rows" in f for f in failures)
        assert any("the corpus holds 8 of token 500, n_hidden sums to 7" in f for f in failures)
        assert f"demo_counted: document 1 (row 1) ends in id {TOKEN}, not the EOD {EOD}" in failures

    def test_a_document_count_other_than_the_datasets_rows_fails(self, tmp_path, tokenizer):
        """One more dataset row than documents: the corpus lost a document somewhere, and the rows
        after it no longer describe the documents they sit beside."""
        table, data_base = counted_corpus(tmp_path, DOCUMENTS, COUNTS + [1], config_tokenizer=tokenizer)
        _, failures = verify(table, data_base)
        assert "demo_counted: n_hidden has 6 rows, the table says 5" in failures

    def test_rows_out_of_their_sources_order_are_named(self, tmp_path, tokenizer):
        """Two rows swapped keep every count and the total right; only `source_row` shows that row 1 is
        not the source's row 1, and so that document 1 is not the source's document 1."""
        table, data_base = counted_corpus(
            tmp_path, DOCUMENTS, COUNTS, source_rows=[0, 2, 1, 3, 4], config_tokenizer=tokenizer
        )
        (report,), failures = verify(table, data_base)
        assert failures == [
            "demo_counted: row 1 holds source_row 2, the source's row there is 1",
            "demo_counted: row 2 holds source_row 1, the source's row there is 2",
        ]
        assert report["document_checks"]["rows_out_of_order"] == 2

    def test_a_dataset_that_starts_further_into_its_source_is_checked_from_its_first_row(self, tmp_path, tokenizer):
        """A dataset holding one slice of its source numbers its rows from the slice's beginning."""
        table, data_base = counted_corpus(tmp_path, DOCUMENTS, COUNTS, first_row=1000, config_tokenizer=tokenizer)
        assert verify(table, data_base)[1] == []
        table.write_text(table.read_text().replace("|source_row|1000", "|source_row|0"))
        _, failures = verify(table, data_base)
        assert "demo_counted: row 0 holds source_row 1000, the source's row there is 0" in failures
        assert "demo_counted: 5 rows in all are not the source's row in order" not in failures

    def test_a_sliced_corpus_maps_rows_across_the_slice_boundary(self, tmp_path, tokenizer):
        """Shard 1 holds rows 2:5, so row 2 is its document 0: an error there is named in shard 1."""
        table, data_base = counted_corpus(tmp_path, DOCUMENTS, [2, 0, 1, 0, 3], shards=2, config_tokenizer=tokenizer)
        _, failures = verify(table, data_base)
        assert "demo_counted shard1: document 0 (row 2) holds 2 of token 500, n_hidden says 1" in failures
        assert not any("shard0" in f for f in failures)

    def test_the_records_checks_still_run_beside_it(self, tmp_path, tokenizer):
        """The document checks are added to a row's verification, never instead of it; a corpus whose
        records name another tokenizer has no EOD to derive, so they cannot run at all."""
        table, data_base = counted_corpus(
            tmp_path, DOCUMENTS, COUNTS, tokenizer="wrong/tokenizer", config_tokenizer=tokenizer
        )
        _, failures = verify(table, data_base)
        assert any("tokenized with 'wrong/tokenizer'" in f for f in failures)
        assert any("the per-document checks could not run" in f and "wrong/tokenizer" in f for f in failures)
        assert not any("of token 500" in f for f in failures)

    def test_a_hub_source_at_a_moving_revision_is_refused_before_reading(self):
        """A count read at a branch could come from any push; only a commit SHA names fixed data."""
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="not a full commit SHA"):
            corpus_documents.dataset_columns("geodesic-research/not-a-real-dataset", "main", "config", ["n_hidden"])

    @pytest.mark.parametrize("column", ["n_hidden", "source_row"])
    def test_a_missing_column_is_reported(self, tmp_path, tokenizer, column):
        table, data_base = counted_corpus(tmp_path, DOCUMENTS, COUNTS, config_tokenizer=tokenizer)
        table.write_text(table.read_text().replace(f"|{column}|", "|n_masked|"))
        _, failures = verify(table, data_base)
        assert any("could not run" in f and "no column ['n_masked']" in f for f in failures)

    def test_a_local_copy_without_the_config_is_reported(self, tmp_path, tokenizer):
        """The local copy is read by the Hub's file-selection rule, so a config it lacks is a failure, not zero rows."""
        table, data_base = counted_corpus(tmp_path, DOCUMENTS, COUNTS, config_tokenizer=tokenizer)
        (tmp_path / "repo" / "demo_counted").rename(tmp_path / "repo" / "elsewhere")
        _, failures = verify(table, data_base)
        assert any("could not run" in f and "no config directory 'demo_counted'" in f for f in failures)


class TestDocumentCheckColumnsInTheTable:
    def test_the_four_are_parsed(self, tmp_path):
        table = write_table(tmp_path, write_prepare_config(tmp_path), **{**CHECKS, "first_row": 69164382})
        (row,) = corpora_table.read_corpora_table(table)
        assert (row.count_token, row.count_column, row.row_column, row.first_row) == (
            TOKEN,
            "n_hidden",
            "source_row",
            69164382,
        )
        plain = write_table(tmp_path, write_prepare_config(tmp_path), subset="plain")
        (row,) = corpora_table.read_corpora_table(plain)
        assert (row.count_token, row.count_column, row.row_column, row.first_row) == (None, None, None, None)

    @pytest.mark.parametrize("given", [1, 2, 3])
    def test_part_of_the_four_is_refused(self, tmp_path, given):
        table = write_table(tmp_path, write_prepare_config(tmp_path))
        stated = "|".join(str(value) for value in list(CHECKS.values())[:given])
        table.write_text(table.read_text().rstrip("\n") + f"|{stated}\n")
        with pytest.raises(ValueError, match=re.escape("expected 11 '|'-separated columns, or 15")):
            corpora_table.read_corpora_table(table)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"kind": "pack"}, "the per-document checks read a tokenized corpus, not kind=pack"),
            ({"count_token": -1}, "count_token must be a token id"),
            ({"first_row": -1}, "first_row must be a row index"),
            ({"row_column": "n_hidden"}, "count_column and row_column name one column, 'n_hidden'"),
        ],
    )
    def test_a_check_no_corpus_can_carry_is_refused(self, tmp_path, overrides, message):
        table = write_table(tmp_path, write_prepare_config(tmp_path), **{**CHECKS, **overrides})
        with pytest.raises(ValueError, match=message):
            corpora_table.read_corpora_table(table)


# ---------------------------------------------------------------------------------------------
# The digest check
# ---------------------------------------------------------------------------------------------


def parent_corpus(
    tmp_path: Path,
    documents: list[list[int]],
    *,
    shards: int = 1,
    tokenizer: str = TOKENIZER,
    pinned_tokenizer: str | None = None,
    **records,
) -> tuple[Path, Path]:
    """A tokenize row's table and its built corpus, sliced into `shards` when more than one.

    `tokenizer` is the one the prepare config names and the tokenize records, and `pinned_tokenizer` the commit the
    config pins it at (none when None); `records` passes one defect, or the commit the records state
    (`tokenizer_revision`), through to the records (`corpora_fixtures.build_corpus`).
    """
    directory = tmp_path / "baseline"
    directory.mkdir(exist_ok=True)
    overrides = {"subset": "stem", "docs": len(documents)}
    if shards > 1:
        overrides.update(shards=shards, shard_mode="slice")
    pin = {} if pinned_tokenizer is None else {"tokenizer-revision": pinned_tokenizer}
    table = write_table(directory, write_prepare_config(directory, tokenizer=tokenizer, **pin), **overrides)
    data_base = tmp_path / "data"
    root = corpora_table.corpus_root(DATASET, "stem", data_base)
    (row,) = corpora_table.read_corpora_table(table)
    records = {"tokenizer": tokenizer, **records}
    if shards == 1:
        build_tokenized_corpus(root, documents, subset="stem", **records)
    else:
        for index, (beg, end) in enumerate(row.slice_ranges()):
            build_tokenized_corpus(
                root / f"shard{index}", documents[beg:end], subset="stem", split=f"train[{beg}:{end}]", **records
            )
    return table, data_base


def builder_record(tokenizer: str, *, subset: str = "stem", repo: str = DATASET, steps: int = 1) -> dict:
    """The record dataset-builder's token-digests stage writes beside its digest list, in its shape:
    the source it resolved and the transforms it ran (`steps` copies of the tokenize_count step)."""
    count = {
        "type": "map_column",
        "kernel": "tokenize_count",
        "text_column": "text",
        "output_column": "n_tokens",
        "ids_hash_column": "ids_hash",
        "tokenizer": tokenizer,
        "tokenizer_revision": "f" * 40,
    }
    return {
        "config": {
            "transform": [
                {"type": "map_column", "kernel": "add_id_column", "target_column": "source_row", "start": 0},
                *[dict(count) for _ in range(steps)],
                {"type": "project", "keep": ["id", "source_row", "n_tokens", "ids_hash"], "ignore_missing": False},
            ]
        },
        "resolved_source": {
            "type": "hf",
            "repo": repo,
            "subset": subset,
            "split": "train",
            "revision": "e" * 40,
            "resolved_revision": "e" * 40,
        },
        "git": {"head": "0" * 40, "dirty": False},
    }


def saved_payload(
    directory: Path, n_tokens: list[int], ids_hash: list[int], record: dict, source_row: list[int] | None = None
) -> Path:
    """A dataset-builder local build's stage directory: the digest list saved by `datasets`, its record beside it."""
    import datasets

    rows = list(range(len(n_tokens))) if source_row is None else source_row
    columns = {"id": [f"row-{r}" for r in rows], "source_row": rows, "n_tokens": n_tokens, "ids_hash": ids_hash}
    datasets.Dataset.from_dict(columns).save_to_disk(str(directory / "train"))
    (directory / "_provenance.json").write_text(json.dumps(record))
    return directory


def true_payload(documents: list[list[int]]) -> tuple[list[int], list[int]]:
    """The digest list of the texts these documents were tokenized from: each one's ids before its EOD."""
    return [len(d) - 1 if d else 0 for d in documents], [ids_digest(d[:-1]) for d in documents]


def digest_config(directory: Path, table: Path, subsets: dict) -> Path:
    """A digest-check config naming `table` and each subset's source."""
    path = directory / "digest_checks.yaml"
    path.write_text(yaml.safe_dump({"table": str(table), "subsets": subsets}))
    return path


def check_hashes(config: Path, data_base: Path, report_out: Path, subset: str = "stem") -> int:
    args = ["check-hashes", "--config", str(config), "--subset", subset, "--data-base", str(data_base)]
    return corpus_documents.main([*args, "--report-out", str(report_out)])


def digest_setup(tmp_path: Path, tokenizer: str, documents=DOCUMENTS, *, record: dict | None = None, **corpus):
    """A corpus, its true digest list as a saved stage directory, and a config pointing one at the other."""
    table, data_base = parent_corpus(tmp_path, documents, tokenizer=tokenizer, **corpus)
    stage = saved_payload(
        tmp_path / "stage", *true_payload(documents), builder_record(tokenizer) if record is None else record
    )
    return digest_config(tmp_path, table, {"stem": {"saved": str(stage)}}), data_base


class TestCheckHashes:
    def test_a_digest_list_describing_every_document_passes(self, tmp_path, tokenizer):
        """Includes a literal EOD id inside a document and an empty document; the EOD is judged by
        position, and an empty document's ids digest is the digest of nothing. The report names the
        config by path and sha256, the EOD and what it was derived from, and the digest list's record."""
        config, data_base = digest_setup(tmp_path, tokenizer)
        report_out = tmp_path / "hashes.json"
        assert check_hashes(config, data_base, report_out) == 0
        report = json.loads(report_out.read_text())
        assert report["ok"] and report["row_count_matches"] and report["source_rows_in_order"]
        assert report["mismatches"] == {"length": 0, "eod": 0, "hash": 0}
        assert report["config"] == {"path": str(config.resolve()), "sha256": corpus_documents.file_sha256(config)}
        assert (report["eod"]["id"], report["eod"]["tokenizer"], report["eod_id"]) == (EOD, tokenizer, EOD)
        assert report["eod"]["records"][0].endswith(f"{corpora_table.TOKENIZED_PREFIX}.provenance.json")
        assert report["digests"]["saved"] == str(tmp_path / "stage")
        assert report["digests"]["source_revision"] == "e" * 40
        assert report["corpus"]["revision"] == REVISION

    def test_a_document_ending_in_the_empty_report_value_lacks_its_eod(self, tmp_path):
        """A non-empty document whose last id is what ``last_ids`` reports for an empty one is still missing its EOD."""
        documents = [[11, EOD], [12, corpus_documents.NO_ID]]
        table, data_base = parent_corpus(tmp_path, documents)
        n_tokens, ids_hash = [1, 1], [ids_digest([11]), ids_digest([12])]
        payload = corpus_documents.saved_columns(
            saved_payload(tmp_path / "stage", n_tokens, ids_hash, {}), list(corpus_documents.DIGEST_COLUMNS)
        )
        (row,) = corpora_table.read_corpora_table(table)
        report = corpus_documents.check_hashes(row, payload, EOD, data_base)
        assert report["mismatches"] == {"length": 0, "eod": 1, "hash": 0}

    def test_each_mismatch_is_reported_in_its_class(self, tmp_path):
        documents = [list(d) for d in DOCUMENTS]
        n_tokens, ids_hash = true_payload(documents)
        n_tokens[1] += 1  # a length the document does not have
        ids_hash[2] ^= 1  # ids the document does not hold
        documents[4][-1] = 7  # the corpus's own document lost its EOD
        table, data_base = parent_corpus(tmp_path, documents)
        payload = corpus_documents.saved_columns(
            saved_payload(tmp_path / "stage", n_tokens, ids_hash, {}), list(corpus_documents.DIGEST_COLUMNS)
        )
        (row,) = corpora_table.read_corpora_table(table)
        report = corpus_documents.check_hashes(row, payload, EOD, data_base)
        assert not report["ok"]
        assert report["mismatches"] == {"length": 1, "eod": 1, "hash": 1}
        (prefix,) = report["prefixes"]
        assert [e["row"] for e in prefix["examples"]["length"]] == [1]
        assert [e["row"] for e in prefix["examples"]["hash"]] == [2]
        assert prefix["examples"]["eod"][0]["last_id"] == 7

    @pytest.mark.parametrize(
        ("documents", "n_tokens", "ids"),
        [
            ([[11, EOD], [EOD], [12, EOD]], [1, 0, 1], [[11], [], [12]]),  # a lone EOD where the text was empty
            ([[11, EOD], [], [12, EOD]], [1, 1, 1], [[11], [13], [12]]),  # nothing where the text had an id
        ],
    )
    def test_a_row_of_no_ids_is_an_empty_document_and_nothing_else(self, tmp_path, documents, n_tokens, ids):
        """An empty text is written as no sequence at all, so a row of no ids is an empty document: a document
        holding only an EOD there, or an empty document beside a row of ids, has the wrong length."""
        table, data_base = parent_corpus(tmp_path, documents)
        stage = saved_payload(tmp_path / "stage", n_tokens, [ids_digest(i) for i in ids], {})
        payload = corpus_documents.saved_columns(stage, list(corpus_documents.DIGEST_COLUMNS))
        (row,) = corpora_table.read_corpora_table(table)
        report = corpus_documents.check_hashes(row, payload, EOD, data_base)
        assert report["mismatches"] == {"length": 1, "eod": 0, "hash": 0}
        (prefix,) = report["prefixes"]
        assert [(e["row"], e["n_tokens"]) for e in prefix["examples"]["length"]] == [(1, n_tokens[1])]

    def test_a_digest_list_of_another_length_cannot_be_aligned(self, tmp_path, tokenizer):
        table, data_base = parent_corpus(tmp_path, DOCUMENTS, tokenizer=tokenizer)
        n_tokens, ids_hash = true_payload(DOCUMENTS)
        stage = saved_payload(tmp_path / "stage", n_tokens[:-1], ids_hash[:-1], builder_record(tokenizer))
        config = digest_config(tmp_path, table, {"stem": {"saved": str(stage)}})
        report_out = tmp_path / "hashes.json"
        assert check_hashes(config, data_base, report_out) == 1
        report = json.loads(report_out.read_text())
        assert report["row_count_matches"] is False and report["prefixes"] == []

    def test_a_digest_list_out_of_source_order_cannot_be_aligned(self, tmp_path, tokenizer):
        """Rows reordered on the way keep the count right, so only `source_row` shows that row i is
        no longer document i."""
        table, data_base = parent_corpus(tmp_path, DOCUMENTS, tokenizer=tokenizer)
        n_tokens, ids_hash = true_payload(DOCUMENTS)
        stage = saved_payload(tmp_path / "stage", n_tokens, ids_hash, builder_record(tokenizer), [0, 2, 1, 3, 4])
        config = digest_config(tmp_path, table, {"stem": {"saved": str(stage)}})
        report_out = tmp_path / "hashes.json"
        assert check_hashes(config, data_base, report_out) == 1
        report = json.loads(report_out.read_text())
        assert (report["row_count_matches"], report["source_rows_in_order"], report["prefixes"]) == (True, False, [])

    def test_digest_rows_map_across_a_slice_boundary(self, tmp_path):
        """Row 2 is the first document of shard 1, so its mismatch is reported there as document 0."""
        table, data_base = parent_corpus(tmp_path, DOCUMENTS, shards=2)
        n_tokens, ids_hash = true_payload(DOCUMENTS)
        ids_hash[2] ^= 1
        repo = tmp_path / "repo"
        columns = {"source_row": list(range(5)), "n_tokens": n_tokens, "ids_hash": ids_hash}
        write_parquet_dataset(repo, "stem", columns, files=2)
        payload = corpus_documents.dataset_columns(str(repo), None, "stem", list(corpus_documents.DIGEST_COLUMNS))
        (row,) = corpora_table.read_corpora_table(table)
        report = corpus_documents.check_hashes(row, payload, EOD, data_base)
        shard0, shard1 = report["prefixes"]
        assert shard0["mismatches"]["hash"] == 0
        assert [(e["document"], e["row"]) for e in shard1["examples"]["hash"]] == [(0, 2)]

    def test_a_digest_list_on_the_hub_is_read_at_its_commit(self, tmp_path, tokenizer, monkeypatch):
        """A `hub` source is the same list published as a dataset config: its record and its train
        files are read at the stated commit, the files in the loader's order, by `hub_parquet`'s rule."""
        table, data_base = parent_corpus(tmp_path, DOCUMENTS, tokenizer=tokenizer)
        n_tokens, ids_hash = true_payload(DOCUMENTS)
        repo = tmp_path / "hub"
        columns = {"source_row": list(range(5)), "n_tokens": n_tokens, "ids_hash": ids_hash}
        write_parquet_dataset(repo, "stem_digests", columns, files=2)
        (repo / "stem_digests" / "_provenance.json").write_text(json.dumps(builder_record(tokenizer)))
        hub = {"dataset": "org/digests", "revision": "a" * 40, "config": "stem_digests"}
        opened = local_hub(monkeypatch, hub["dataset"], {hub["revision"]: repo})
        config = digest_config(tmp_path, table, {"stem": {"hub": hub}})
        report_out = tmp_path / "hashes.json"
        assert check_hashes(config, data_base, report_out) == 0
        report = json.loads(report_out.read_text())
        assert report["ok"] and report["digests"]["hub"] == hub
        assert report["digests"]["record"] == f"datasets/org/digests@{'a' * 40}/stem_digests/_provenance.json"
        assert sorted(opened) == [
            (hub["revision"], "stem_digests/_provenance.json"),
            (hub["revision"], "stem_digests/train-00000-of-00002.parquet"),
            (hub["revision"], "stem_digests/train-00001-of-00002.parquet"),
        ]


class TestDigestCheckConfig:
    """The config is read and refused whole, so a misspelt key or a forgotten subset is an error
    whichever subset is asked for, and a subset with no digest list is refused, never skipped."""

    @staticmethod
    def _table(tmp_path: Path) -> Path:
        """A table of two tokenize rows and a pack row, which has no .bin to digest."""
        config = write_prepare_config(tmp_path)
        extra = [{"subset": "other"}, {"subset": "sft", "kind": "pack", "shards": 2, "shard_mode": "split"}]
        return write_table(tmp_path, config, subset="stem", extra_rows=extra)

    def _refused(self, tmp_path: Path, document, subset: str = "stem") -> str:
        path = tmp_path / "digest_checks.yaml"
        path.write_text(yaml.safe_dump(document))
        with pytest.raises(corpus_documents.CorpusCheckFailed) as raised:
            corpus_documents.read_digest_checks(path, subset)
        return str(raised.value)

    def test_a_saved_and_a_hub_source_are_read(self, tmp_path):
        table = self._table(tmp_path)
        hub = {"dataset": "org/digests", "revision": "a" * 40, "config": "other_digests"}
        path = digest_config(tmp_path, table, {"stem": {"saved": "/abs/stage"}, "other": {"hub": hub}})
        stem = corpus_documents.read_digest_checks(path, "stem")
        assert (stem.saved, stem.hub, stem.row.subset) == (Path("/abs/stage"), None, "stem")
        assert stem.config_sha256 == corpus_documents.file_sha256(path)
        other = corpus_documents.read_digest_checks(path, "other")
        assert (other.saved, other.hub) == (
            None,
            corpus_documents.HubDigests("org/digests", "a" * 40, "other_digests"),
        )

    @pytest.mark.parametrize(
        ("document", "message"),
        [
            ({"table": "t", "subsets": {}, "eod_id": 2}, r"unknown keys \['eod_id'\]"),
            ({"subsets": {}}, r"missing keys \['table'\]"),
            ([1, 2], "expected a mapping, got list"),
        ],
    )
    def test_a_config_of_another_shape_is_refused(self, tmp_path, document, message):
        assert re.search(message, self._refused(tmp_path, document))

    def test_a_table_that_does_not_exist_is_refused(self, tmp_path):
        message = self._refused(tmp_path, {"table": str(tmp_path / "absent.tsv"), "subsets": {}})
        assert "no corpora table at" in message

    @pytest.mark.parametrize(
        ("subsets", "message"),
        [
            ({"stem": {"pending": "x"}}, r"unknown \[\], missing \['other'\]"),
            (
                {"stem": {"pending": "x"}, "other": {"pending": "x"}, "sft": {"pending": "x"}},
                r"unknown \['sft'\], missing \[\]",
            ),
        ],
    )
    def test_the_subsets_must_be_exactly_the_tables_tokenize_rows(self, tmp_path, subsets, message):
        refused = self._refused(tmp_path, {"table": str(self._table(tmp_path)), "subsets": subsets})
        assert "must be exactly the tokenize rows" in refused and re.search(message, refused)

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ({"saved": "/a", "pending": "x"}, r"names exactly one of .*, not \['saved', 'pending'\]"),
            ({}, r"names exactly one of .*, not \[\]"),
            ({"saved_dir": "/a"}, r"unknown keys \['saved_dir'\]"),
            ({"saved": ""}, "subsets.other.saved: must be a non-empty string"),
            ({"pending": None}, "subsets.other.pending: must be a non-empty string"),
            ({"hub": {"dataset": "org/d", "revision": "a" * 40}}, r"subsets.other.hub: .*missing keys \['config'\]"),
            ({"hub": {"dataset": "org/d", "revision": "main", "config": "c"}}, "'main' is not a full commit SHA"),
        ],
    )
    def test_every_entry_is_checked_whichever_subset_is_asked_for(self, tmp_path, entry, message):
        """The malformed entry is `other`'s; asking for `stem` still refuses the config."""
        document = {"table": str(self._table(tmp_path)), "subsets": {"stem": {"saved": "/a"}, "other": entry}}
        assert re.search(message, self._refused(tmp_path, document))

    def test_a_pending_subset_is_refused_with_its_reason(self, tmp_path):
        document = {
            "table": str(self._table(tmp_path)),
            "subsets": {"stem": {"pending": "the digest pass has not reached it"}, "other": {"saved": "/a"}},
        }
        message = self._refused(tmp_path, document)
        assert "stem is pending (the digest pass has not reached it)" in message

    def test_a_subset_the_config_does_not_name_is_refused(self, tmp_path):
        document = {
            "table": str(self._table(tmp_path)),
            "subsets": {"stem": {"saved": "/a"}, "other": {"saved": "/b"}},
        }
        assert "no subset 'sft'" in self._refused(tmp_path, document, subset="sft")

    def test_the_command_reports_a_refusal_and_fails(self, tmp_path, capsys):
        config = digest_config(tmp_path, self._table(tmp_path), {"stem": {"pending": "x"}, "other": {"pending": "y"}})
        assert check_hashes(config, tmp_path / "data", tmp_path / "hashes.json") == 1
        error = capsys.readouterr().err
        assert "FAILED:" in error and "stem is pending (x)" in error
        assert not (tmp_path / "hashes.json").exists()


class TestTheCluelessNormDigestChecks:
    """The real config: it names Normal-Norm's table and every tokenize row of it, with the saved
    digest lists of the subsets that have one; the rest are pending, and refused."""

    SAVED = {"zyda_full"}

    def test_it_reads_and_names_every_row(self):
        document = yaml.safe_load(REAL_DIGEST_CHECKS.read_text())
        table = corpora_table.REPO_ROOT / document["table"]
        assert table == corpora_table.REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline" / "corpora.tsv"
        rows = [row.subset for row in corpora_table.read_corpora_table(table) if row.kind == "tokenize"]
        assert sorted(document["subsets"]) == sorted(rows)
        assert {name for name, entry in document["subsets"].items() if "saved" in entry} == self.SAVED

    @pytest.mark.parametrize("subset", sorted(SAVED))
    def test_a_saved_subset_resolves_to_its_row_and_digest_list(self, subset):
        check = corpus_documents.read_digest_checks(REAL_DIGEST_CHECKS, subset)
        assert check.row.subset == subset and check.hub is None
        assert check.saved.name == "digests" and check.saved.is_absolute()

    def test_a_pending_subset_is_refused(self):
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="climbmix_full is pending"):
            corpus_documents.read_digest_checks(REAL_DIGEST_CHECKS, "climbmix_full")


class TestTheDigestsMustDescribeTheCorpus:
    """The EOD is derived from the corpus's own tokenize record and tokenizer, and the digest list's
    record must name the corpus's source and tokenizer. Any disagreement refuses the check before a
    single document is compared."""

    def _refused(self, tmp_path: Path, capsys, config: Path, data_base: Path) -> str:
        assert check_hashes(config, data_base, tmp_path / "hashes.json") == 1
        assert not (tmp_path / "hashes.json").exists()
        return capsys.readouterr().err

    def test_a_corpus_tokenized_with_another_tokenizer_is_refused(self, tmp_path, capsys, tokenizer):
        """The prepare config names the tokenizer the EOD is taken from; a tokenize record naming
        another means the corpus is not the one the config describes."""
        table, data_base = parent_corpus(tmp_path, DOCUMENTS, tokenizer=tokenizer)
        root = corpora_table.corpus_root(DATASET, "stem", data_base)
        build_tokenized_corpus(root, DOCUMENTS, subset="stem", tokenizer="org/another-tokenizer")
        stage = saved_payload(tmp_path / "stage", *true_payload(DOCUMENTS), builder_record(tokenizer))
        config = digest_config(tmp_path, table, {"stem": {"saved": str(stage)}})
        error = self._refused(tmp_path, capsys, config, data_base)
        assert "records tokenizer 'org/another-tokenizer' at commit None with append_eod='true'" in error
        assert "the corpus's EOD cannot be derived" in error

    @pytest.mark.parametrize(
        ("pinned", "recorded"), [("4" * 40, None), ("4" * 40, "5" * 40), (None, "4" * 40)], ids=str
    )
    def test_a_corpus_tokenized_at_another_commit_than_the_config_pins_is_refused(
        self, tmp_path, capsys, tokenizer, pinned, recorded
    ):
        """The EOD comes from the tokenizer's commit the config pins, so a record of another commit (or of none,
        where the config pins one) means the corpus is not the one the config describes."""
        config, data_base = digest_setup(tmp_path, tokenizer, pinned_tokenizer=pinned, tokenizer_revision=recorded)
        error = self._refused(tmp_path, capsys, config, data_base)
        assert f"at commit {recorded!r} with append_eod='true'" in error
        assert f"names {tokenizer!r} at commit {pinned!r}" in error

    def test_a_corpus_tokenized_at_the_pinned_commit_derives_its_eod_from_it(self, tmp_path, tokenizer):
        pin = "4" * 40
        config, data_base = digest_setup(tmp_path, tokenizer, pinned_tokenizer=pin, tokenizer_revision=pin)
        assert check_hashes(config, data_base, tmp_path / "hashes.json") == 0
        report = json.loads((tmp_path / "hashes.json").read_text())
        assert (report["eod"]["tokenizer"], report["eod"]["tokenizer_revision"]) == (tokenizer, pin)

    def test_a_corpus_tokenized_without_eods_is_refused(self, tmp_path, capsys, tokenizer):
        config, data_base = digest_setup(tmp_path, tokenizer, append_eod="false")
        assert "append_eod='false'" in self._refused(tmp_path, capsys, config, data_base)

    def test_a_tokenize_record_without_its_tokenizer_is_refused(self, tmp_path, capsys, tokenizer):
        config, data_base = digest_setup(tmp_path, tokenizer)
        record = Path(
            f"{corpora_table.corpus_root(DATASET, 'stem', data_base) / corpora_table.TOKENIZED_PREFIX}.provenance.json"
        )
        content = json.loads(record.read_text())
        del content["parameters"]["tokenizer"]
        record.write_text(json.dumps(content))
        assert "records no parameters.tokenizer" in self._refused(tmp_path, capsys, config, data_base)

    @pytest.mark.parametrize(
        ("record", "message"),
        [
            ({"subset": "other"}, "the digests were computed from"),
            ({"repo": "org/another-dataset"}, "the digests were computed from"),
            ({"tokenizer": "org/another-tokenizer"}, "the tokenize_count step states"),
            ({"steps": 0}, "records 0 tokenize_count steps, not one"),
            ({"steps": 2}, "records 2 tokenize_count steps, not one"),
        ],
    )
    def test_a_digest_list_of_another_source_or_tokenizer_is_refused(
        self, tmp_path, capsys, tokenizer, record, message
    ):
        options = {"tokenizer": tokenizer, **record}
        config, data_base = digest_setup(tmp_path, tokenizer, record=builder_record(**options))
        assert message in self._refused(tmp_path, capsys, config, data_base)

    def test_a_digest_list_without_its_record_is_refused(self, tmp_path, capsys, tokenizer):
        config, data_base = digest_setup(tmp_path, tokenizer)
        (tmp_path / "stage" / "_provenance.json").unlink()
        assert "missing" in self._refused(tmp_path, capsys, config, data_base)

    def test_a_record_lacking_its_source_is_refused(self, tmp_path, capsys, tokenizer):
        record = builder_record(tokenizer)
        del record["resolved_source"]["split"]
        config, data_base = digest_setup(tmp_path, tokenizer, record=record)
        assert "records no resolved_source.split" in self._refused(tmp_path, capsys, config, data_base)


# ---------------------------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------------------------


def write_kept(directory: Path, kept: list[int], fmt: str) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = directory / f"kept.{fmt}"
    if fmt == "parquet":
        pq.write_table(pa.table({"row": pa.array(kept, pa.int64())}), path)
    elif fmt == "json":
        path.write_text(json.dumps(kept))
    else:
        path.write_text("".join(f"{index}\n" for index in kept))
    return path


def selection(
    tmp_path: Path,
    kept: list[int],
    *,
    shards: int = 1,
    fmt: str = "parquet",
    table_docs: int | None = None,
    make_roots: bool = True,
    **config_overrides,
) -> tuple[Path, Path]:
    """A parent corpus, a select row keeping `kept` of it, and the directories build_corpora.sh would create."""
    parent_table, data_base = parent_corpus(tmp_path, DOCUMENTS, shards=shards)
    arm = tmp_path / "selected_arm"
    arm.mkdir()
    config = {
        "dataset": "geodesic-research/selected",
        "parent_table": str(parent_table),
        "parent_subset": "stem",
        "kept": str(write_kept(arm, kept, fmt)),
        "absent_token_ids": [],
        **config_overrides,
    }
    config_path = arm / "select.yaml"
    config_path.write_text(yaml.safe_dump(config))
    overrides = {"subset": "stem", "kind": "select", "prep_h": 0, "workers": 1}
    overrides["docs"] = len(kept) if table_docs is None else table_docs
    if shards > 1:
        overrides.update(shards=shards, shard_mode="slice", stripe=1)
    table = write_table(arm, config_path, **overrides)
    if make_roots:
        for plan in corpora_table.plan_build(table, data_base=data_base):
            for directory, _ in plan.roots:
                directory.mkdir(parents=True, exist_ok=True)
    return table, data_base


def run_select(table: Path, data_base: Path, shard: int | None = None) -> int:
    args = ["select", str(table), "stem", "--data-base", str(data_base)]
    return corpus_documents.main(args + ([] if shard is None else ["--shard", str(shard)]))


def selected_prefix(data_base: Path, shard: int | None = None) -> Path:
    root = corpora_table.corpus_root("geodesic-research/selected", "stem", data_base)
    return (root if shard is None else root / f"shard{shard}") / corpora_table.TOKENIZED_PREFIX


def select_config(table: Path) -> Path:
    return table.parent / "select.yaml"


def rewrite_yaml(path: Path, **changes) -> None:
    path.write_text(yaml.safe_dump({**yaml.safe_load(path.read_text()), **changes}))


def parent_provenance(data_base: Path) -> Path:
    return Path(
        f"{corpora_table.corpus_root(DATASET, 'stem', data_base) / corpora_table.TOKENIZED_PREFIX}.provenance.json"
    )


class TestSelect:
    @pytest.mark.parametrize("fmt", ["parquet", "json", "txt"])
    def test_the_kept_documents_are_copied_once_in_order(self, tmp_path, fmt):
        """Document 3 is empty, and is copied as preprocess_data.py writes one: a document of no sequence."""
        table, data_base = selection(tmp_path, [0, 2, 3], fmt=fmt)
        assert run_select(table, data_base) == 0
        prefix = selected_prefix(data_base)
        assert read_documents(prefix) == [DOCUMENTS[0], DOCUMENTS[2], DOCUMENTS[3]]
        record = json.loads(Path(f"{prefix}.provenance.json").read_text())
        assert record["totals"] == {"total_tokens": 10, "num_sequences": 2, "num_documents": 3}
        assert record["parameters"]["tokenizer"] == TOKENIZER
        assert record["kept"]["entries"] == 3 and record["parent"]["rows"] == [0, 5]
        assert not list(prefix.parent.glob("*.partial"))

    def test_a_sliced_parent_selects_each_shards_own_rows(self, tmp_path):
        """Rows 0 and 4 sit in shards 0 and 1 (rows 0:2 and 2:5), so shard 1 keeps its document 2."""
        table, data_base = selection(tmp_path, [0, 1, 4], shards=2)
        assert run_select(table, data_base, shard=0) == 0
        assert run_select(table, data_base, shard=1) == 0
        assert read_documents(selected_prefix(data_base, 0)) == [DOCUMENTS[0], DOCUMENTS[1]]
        assert read_documents(selected_prefix(data_base, 1)) == [DOCUMENTS[4]]

    @pytest.mark.parametrize(
        ("shard", "message"),
        [
            (None, r"the corpus has shards \[0, 1\], so --shard must name one"),
            (2, r"no shard 2; the corpus has \[0, 1\]"),
        ],
    )
    def test_a_sharded_selection_must_be_told_an_existing_shard(self, tmp_path, capsys, shard, message):
        table, data_base = selection(tmp_path, [0, 1, 4], shards=2)
        assert run_select(table, data_base, shard=shard) == 1
        assert re.search(message, capsys.readouterr().err)
        assert not any(selected_prefix(data_base, 0).parent.iterdir())

    def test_an_unsharded_selection_has_no_shard_to_name(self, tmp_path, capsys):
        table, data_base = selection(tmp_path, [0, 2])
        assert run_select(table, data_base, shard=0) == 1
        assert re.search(r"no shard 0; the corpus has \[None\]", capsys.readouterr().err)

    @pytest.mark.parametrize(
        ("kept", "message"),
        [
            ([3, 1], r"entry 1 \(1\) is below entry 0 \(3\)"),
            ([1, 1], r"entry 1 \(1\) repeats entry 0 \(1\)"),
            ([0, 5], r"entry 1 \(5\) is outside the parent's rows 0:5"),
            ([-1, 2], r"entry 0 \(-1\) is outside"),
        ],
    )
    def test_a_malformed_kept_list_is_refused_before_writing(self, tmp_path, capsys, kept, message):
        table, data_base = selection(tmp_path, kept)
        assert run_select(table, data_base) == 1
        assert re.search(message, capsys.readouterr().err)
        assert not any(selected_prefix(data_base).parent.iterdir())

    def test_a_list_of_another_length_than_the_table_says_is_refused(self, tmp_path, capsys):
        table, data_base = selection(tmp_path, [0, 2], table_docs=3)
        assert run_select(table, data_base) == 1
        assert "lists 2 documents, the table says 3" in capsys.readouterr().err

    def test_a_non_empty_output_directory_is_refused(self, tmp_path, capsys):
        table, data_base = selection(tmp_path, [0, 2])
        (selected_prefix(data_base).parent / "left_over").write_text("")
        assert run_select(table, data_base) == 1
        assert "is not empty" in capsys.readouterr().err

    def test_an_output_directory_build_corpora_did_not_create_is_refused(self, tmp_path, capsys):
        """Striping must precede the first write, so the job writes only where the build striped."""
        table, data_base = selection(tmp_path, [0, 2], make_roots=False)
        assert run_select(table, data_base) == 1
        assert "creates and stripes it" in capsys.readouterr().err

    def test_an_unattributable_checkout_writes_nothing(self, tmp_path, capsys, monkeypatch):
        """A selection's record names the code that wrote it; a checkout whose revision cannot be
        read (here a copy whose REVISION file is empty) writes nothing at all."""
        table, data_base = selection(tmp_path, [0, 2])
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        (checkout / "REVISION").write_text("")
        monkeypatch.setattr(corpus_documents, "REPO_ROOT", checkout)
        assert run_select(table, data_base) == 1
        assert "the code this runs is unattributable: UNRESOLVED" in capsys.readouterr().err
        assert not any(selected_prefix(data_base).parent.iterdir())

    @pytest.mark.parametrize(
        ("damage", "message"),
        [
            (
                {"totals": {"num_documents": 4, "total_tokens": 17}},
                "records 4 documents and 17 tokens, the files hold 5 and 17",
            ),
            ({"totals": {"num_documents": 5}}, "records no totals.total_tokens"),
            ({"parameters": {"json_key": "input"}}, "records no parameters.tokenizer"),
        ],
    )
    def test_a_parent_record_that_does_not_describe_its_files_is_refused(self, tmp_path, capsys, damage, message):
        """The parent's record is carried into the selection's, and training reads the tokenizer
        from it, so it must agree with the files and state its parameters."""
        table, data_base = selection(tmp_path, [0, 2])
        record = parent_provenance(data_base)
        record.write_text(json.dumps({**json.loads(record.read_text()), **damage}))
        assert run_select(table, data_base) == 1
        assert message in capsys.readouterr().err
        assert not any(selected_prefix(data_base).parent.iterdir())

    def test_a_kept_document_holding_an_absent_id_is_refused_before_its_final_names(self, tmp_path, capsys):
        """Parent rows 2 and 4 hold the token, so the selection is refused and never takes its final names."""
        table, data_base = selection(tmp_path, [1, 2, 4], absent_token_ids=[TOKEN])
        assert run_select(table, data_base) == 1
        assert (
            f"token {TOKEN}, which the select config lists as absent, is held by 2 of its 3 documents: "
            "document 1 (parent row 2), document 2 (parent row 4)"
        ) in capsys.readouterr().err
        prefix = selected_prefix(data_base)
        assert not [p for p in prefix.parent.iterdir() if not p.name.endswith(".partial")]

    def test_an_absent_id_in_a_documents_last_position_is_found(self, tmp_path, capsys):
        """Parent row 1 holds its EOD only at its last position, which the per-document count leaves out."""
        table, data_base = selection(tmp_path, [1], absent_token_ids=[EOD])
        assert run_select(table, data_base) == 1
        assert f"token {EOD}, which the select config lists as absent, is held by 1 of its 1 documents: " in (
            capsys.readouterr().err
        )

    def test_a_sharded_selection_names_the_parent_row_of_a_document_holding_an_absent_id(self, tmp_path, capsys):
        """Shard 1 holds parent rows 2:5, so its document 0 is parent row 4."""
        table, data_base = selection(tmp_path, [0, 1, 4], shards=2, absent_token_ids=[TOKEN])
        assert run_select(table, data_base, shard=1) == 1
        assert "is held by 1 of its 1 documents: document 0 (parent row 4)" in capsys.readouterr().err

    def test_a_selection_without_the_absent_ids_is_written_and_states_them(self, tmp_path):
        """Rows 1 and 3 hold no TOKEN (row 3 is empty); every id is checked, so a second id changes nothing."""
        table, data_base = selection(tmp_path, [1, 3], absent_token_ids=[TOKEN, 99])
        assert run_select(table, data_base) == 0
        record = json.loads(Path(f"{selected_prefix(data_base)}.provenance.json").read_text())
        assert record["absent_token_ids"] == [TOKEN, 99]
        assert record["tool"]["version"] == corpus_documents.TOOL_VERSION
        _, failures = verify(table, data_base)
        assert failures == []

    def test_verify_fails_a_selection_whose_config_now_lists_other_absent_ids(self, tmp_path):
        table, data_base = selection(tmp_path, [1, 3])
        assert run_select(table, data_base) == 0
        rewrite_yaml(select_config(table), absent_token_ids=[TOKEN])
        _, failures = verify(table, data_base)
        assert any("absent ids" in f for f in failures)

    def test_verify_scans_the_files_for_the_absent_ids_whatever_the_record_says(self, tmp_path):
        """A record claiming ids the files hold does not pass: verify reads the ids themselves."""
        table, data_base = selection(tmp_path, [0, 2])
        assert run_select(table, data_base) == 0
        rewrite_yaml(select_config(table), absent_token_ids=[TOKEN])
        record = Path(f"{selected_prefix(data_base)}.provenance.json")
        record.write_text(json.dumps({**json.loads(record.read_text()), "absent_token_ids": [TOKEN]}))
        _, failures = verify(table, data_base)
        assert any(
            f"token {TOKEN}, which the select config lists as absent, is held by 2 of its 2 documents" in f
            for f in failures
        )

    def test_verify_passes_a_selection_and_reports_its_counts(self, tmp_path):
        table, data_base = selection(tmp_path, [1, 4])
        assert run_select(table, data_base) == 0
        (report,), failures = verify(table, data_base)
        assert failures == []
        assert (report["docs"], report["tokens"]) == (2, 7)

    def test_verify_fails_a_tampered_selection(self, tmp_path):
        table, data_base = selection(tmp_path, [1, 4])
        assert run_select(table, data_base) == 0
        data = Path(f"{selected_prefix(data_base)}.bin")
        ids = np.fromfile(data, dtype="<i4")
        ids[4] = 99
        ids.tofile(data)
        _, failures = verify(table, data_base)
        assert any("document 1 differs from parent document 4" in f for f in failures)

    def test_verify_fails_a_selection_whose_kept_list_changed(self, tmp_path):
        table, data_base = selection(tmp_path, [1, 4], fmt="json")
        assert run_select(table, data_base) == 0
        kept = tmp_path / "selected_arm" / "kept.json"
        kept.write_text(json.dumps([1, 4]) + " ")
        _, failures = verify(table, data_base)
        assert any("kept list" in f for f in failures)

    def test_verify_fails_a_selection_whose_record_misstates_its_totals(self, tmp_path):
        table, data_base = selection(tmp_path, [1, 4])
        assert run_select(table, data_base) == 0
        record = Path(f"{selected_prefix(data_base)}.provenance.json")
        content = json.loads(record.read_text())
        content["totals"]["total_tokens"] = 99
        record.write_text(json.dumps(content))
        _, failures = verify(table, data_base)
        assert any("records 2 documents of 99 ids, the files hold 2 documents of 7 ids" in f for f in failures)


class TestReadKept:
    @pytest.mark.parametrize(
        ("name", "content", "message"),
        [
            ("kept.json", json.dumps({"rows": [1]}), "a JSON kept list is an array of integers"),
            ("kept.json", json.dumps([1, 2.5]), "a JSON kept list is an array of integers"),
            ("kept.json", json.dumps([1, True]), "a JSON kept list is an array of integers"),
            ("kept.txt", "1\nx\n3\n", "line 2 is not an integer"),
            ("kept.csv", "1\n", "a kept list is a .parquet, .json or .txt file"),
        ],
    )
    def test_a_text_list_of_another_format_is_refused(self, tmp_path, name, content, message):
        path = tmp_path / name
        path.write_text(content)
        with pytest.raises(corpus_documents.CorpusCheckFailed, match=re.escape(message)):
            corpus_documents.read_kept(path)

    @pytest.mark.parametrize(
        ("columns", "message"),
        [
            ({"row": [1], "other": [2]}, "a kept list has one column, this has ['row', 'other']"),
            ({"row": ["1"]}, "column 'row' is string, not integers"),
            ({"row": [1, None]}, "column 'row' has 1 nulls"),
        ],
    )
    def test_a_parquet_list_of_another_shape_is_refused(self, tmp_path, columns, message):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "kept.parquet"
        pq.write_table(pa.table(columns), path)
        with pytest.raises(corpus_documents.CorpusCheckFailed, match=re.escape(message)):
            corpus_documents.read_kept(path)

    def test_a_missing_list_is_refused(self, tmp_path):
        with pytest.raises(corpus_documents.CorpusCheckFailed, match="missing kept list"):
            corpus_documents.read_kept(tmp_path / "kept.json")


class TestSelectPlan:
    def test_one_job_per_parent_prefix_through_the_corpus_job(self, tmp_path):
        table, data_base = selection(tmp_path, [0, 4], shards=2, make_roots=False)
        (plan,) = corpora_table.plan_build(table, data_base=data_base)
        arm = table.resolve().parent.name
        assert [job.name for job in plan.jobs] == [f"cp-{arm}-select-stem-s0", f"cp-{arm}-select-stem-s1"]
        assert all(job.step == "select" and job.depends_on == "" for job in plan.jobs)
        assert all(job.script == str(corpora_table.CORPUS_JOB_SCRIPT) for job in plan.jobs)
        assert plan.jobs[1].payload == (
            str(corpora_table.DOCUMENTS_TOOL),
            "select",
            str(table.resolve()),
            "stem",
            "--shard",
            "1",
            "--data-base",
            str(data_base),
        )
        assert [stripe for _, stripe in plan.roots] == [True, True, True]
        assert plan.tokenizer == TOKENIZER

    def test_the_build_script_submits_it(self, tmp_path):
        table, _ = selection(tmp_path, [0, 4], shards=2, make_roots=False)
        proc = dry_run_build(table, "all")
        output = proc.stdout + proc.stderr
        assert proc.returncode == 0, output
        assert "SUBMITTED 2 jobs" in output
        assert output.count("[dry-run] select stem shard") == 2
        assert "corpus_job.sbatch" in output and "corpus_documents.py select" in output

    @pytest.mark.parametrize(
        ("row", "config", "message"),
        [
            ({"shards": 2, "shard_mode": "slice"}, {}, "a selection keeps its parent's sharding"),
            ({}, {"dataset": DATASET}, "is its parent's"),
            ({}, {"kept_list": "x"}, r"a select config: .*unknown keys \['kept_list'\]"),
            ({"docs": 9}, {}, "keeps 9 documents of a parent that has 5"),
            ({}, {"absent_token_ids": TOKEN}, "absent_token_ids must be a list of token ids, not 500"),
            ({}, {"absent_token_ids": [TOKEN, TOKEN]}, r"absent_token_ids repeats a token id: \[500, 500\]"),
            ({}, {"absent_token_ids": [-1]}, r"absent_token_ids\[0\] must be a token id \(an integer >= 0\), not -1"),
            ({}, {"absent_token_ids": [True]}, r"absent_token_ids\[0\] must be a token id .*, not True"),
            ({}, {"absent_token_ids": ["500"]}, r"absent_token_ids\[0\] must be a token id .*, not '500'"),
        ],
    )
    def test_an_unsound_selection_is_refused_when_planned(self, tmp_path, row, config, message):
        table, data_base = selection(tmp_path, [0, 4], make_roots=False, **config)
        if row:
            text = table.read_text().splitlines()
            fields = dict(zip(corpora_table.COLUMNS, text[-1].split("|")))
            fields.update({key: str(value) for key, value in row.items()})
            table.write_text("\n".join([*text[:-1], "|".join(fields[c] for c in corpora_table.COLUMNS)]) + "\n")
        with pytest.raises(ValueError, match=message):
            corpora_table.plan_build(table, data_base=data_base)

    def test_a_select_config_must_state_its_absent_ids(self, tmp_path):
        """An empty list states that there are none; leaving the key out states nothing, and is refused."""
        table, data_base = selection(tmp_path, [0, 4], make_roots=False)
        config = select_config(table)
        stated = yaml.safe_load(config.read_text())
        del stated["absent_token_ids"]
        config.write_text(yaml.safe_dump(stated))
        with pytest.raises(ValueError, match=r"missing keys \['absent_token_ids'\]"):
            corpora_table.plan_build(table, data_base=data_base)

    @pytest.mark.parametrize(
        ("overrides", "message"), [({"prep_h": 1}, "prep_h must be 0"), ({"workers": 4}, "workers must be 1")]
    )
    def test_a_select_row_states_its_shape(self, tmp_path, overrides, message):
        table = write_table(
            tmp_path, write_prepare_config(tmp_path), **{"kind": "select", "prep_h": 0, "workers": 1, **overrides}
        )
        with pytest.raises(ValueError, match=message):
            corpora_table.read_corpora_table(table)

    def test_a_split_parent_has_no_row_mapping(self, tmp_path):
        """A byte-gated split decides its shard boundaries when it runs, so no row maps to a document."""
        table = write_table(tmp_path, write_prepare_config(tmp_path), shards=2, shard_mode="split")
        (row,) = corpora_table.read_corpora_table(table)
        with pytest.raises(ValueError, match="byte-gated split"):
            corpora_table.tokenized_prefixes(row, tmp_path / "data")

    def test_prefixes_are_the_tokenize_jobs_targets(self, tmp_path):
        table = write_table(tmp_path, write_prepare_config(tmp_path), shards=4, shard_mode="slice", docs=10)
        (row,) = corpora_table.read_corpora_table(table)
        prefixes = corpora_table.tokenized_prefixes(row, tmp_path / "data")
        plan = corpora_table.plan_corpus(row, "arm", tmp_path / "data")
        targets = [Path(job.payload[1]) for job in plan.jobs if job.step == "tokenize"]
        assert [p.prefix.parent for p in prefixes] == targets
        assert [p.rows for p in prefixes] == row.slice_ranges() == [(0, 2), (2, 5), (5, 7), (7, 10)]
