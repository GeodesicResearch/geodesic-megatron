# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The filtered-arm audit must catch a corpus that is not its baseline minus the removed documents.

Each test builds what the audit reads — two arms' prepare and tokenize records, real `.bin/.idx`
documents written by Megatron's own builder, a packed parquet in the packer's layout, or two
arrays of document lengths — correct except for one defect, and asserts the audit reports that
defect. Nothing is mocked. Only the Hub reads are exercised elsewhere (by the audit runs the arm
READMEs record), because they need the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.unit_tests.corpora_fixtures import (
    DATASET,
    build_corpus,
    build_packed_shard,
    corpora_table,
    load_campaign_module,
    write_prepare_config,
    write_table,
    write_tokenized_documents,
)


audit = load_campaign_module("audit_filtered_corpora")

TAG = "mini_2plus"
BASE = "demo"
FILTERED = f"demo_filtered_{TAG}"

STATS = audit.stats_from_rows(
    [
        {
            "subset": BASE,
            "n_total": 100,
            "n_removed": 30,
            "n_retained": 70,
            "n_canary": 2,
            "num_tokens_removed": 300,
            "num_tokens_retained": 700,
        }
    ]
)


def write_arm(directory: Path, subset: str, kind: str, docs: int) -> Path:
    """A one-row arm: its prepare config and a corpora table naming `subset` (a `kind` row)."""
    return write_table(directory, write_prepare_config(directory), subset=subset, kind=kind, docs=docs)


def run_counts(
    tmp_path: Path, *, filtered_docs=70, filtered_tokens=770, baseline_docs=100, baseline_tokens=1100, **damage
):
    """Audit a filtered arm against a baseline arm built from these numbers; return the failures."""
    data_base = tmp_path / "data"
    build_corpus(
        corpora_table.corpus_root(DATASET, FILTERED, data_base),
        subset=FILTERED,
        docs=filtered_docs,
        tokens=filtered_tokens,
        **damage,
    )
    build_corpus(
        corpora_table.corpus_root(DATASET, BASE, data_base), subset=BASE, docs=baseline_docs, tokens=baseline_tokens
    )
    _, checker = audit.audit_arm(
        write_arm(tmp_path, FILTERED, "tokenize", filtered_docs),
        write_arm(tmp_path, BASE, "tokenize", baseline_docs),
        TAG,
        "all",
        None,
        data_base,
        False,
        0,
        0,
        0,
        0,
        STATS,
        None,
    )
    return checker.failures


class TestCountsLayer:
    def test_a_corpus_that_is_the_baseline_minus_the_removed_documents_passes(self, tmp_path):
        assert run_counts(tmp_path) == []

    def test_an_unfiltered_corpus_under_a_filtered_name_is_caught(self, tmp_path):
        # The whole baseline copied under the filtered name: document and token counts both betray it.
        failures = run_counts(tmp_path, filtered_docs=100, filtered_tokens=1100)
        assert any("n_retained" in f for f in failures)
        assert any("retained tokens + EODs" in f for f in failures)

    def test_a_prepare_record_naming_the_unfiltered_subset_is_caught(self, tmp_path):
        failures = run_counts(tmp_path, recorded_subset=BASE)
        assert any("prepare record names 'demo'" in f for f in failures)

    def test_a_stale_revision_is_caught(self, tmp_path):
        failures = run_counts(tmp_path, revision="f" * 40)
        assert any("prepared at revision" in f for f in failures)

    def test_a_token_gap_that_is_not_the_removed_tokens_is_caught(self, tmp_path):
        # Right document counts, wrong tokens: a corpus cut from a different filter threshold.
        failures = run_counts(tmp_path, filtered_tokens=771)
        assert any("removed tokens + EODs" in f for f in failures)

    def test_a_baseline_that_does_not_match_the_statistics_is_caught(self, tmp_path):
        failures = run_counts(tmp_path, baseline_docs=101, baseline_tokens=1111)
        assert any("n_total" in f for f in failures)


def run_packed_counts(tmp_path: Path, filtered_docs: int) -> list[str]:
    """Audit a packed filtered corpus whose baseline arm has no row for it; return the failures."""
    data_base = tmp_path / "data"
    build_corpus(
        corpora_table.corpus_root(DATASET, FILTERED, data_base), subset=FILTERED, docs=filtered_docs, tokens=0
    )
    empty_baseline = tmp_path / "baseline.tsv"
    empty_baseline.write_text("# no rows\n")
    reports, checker = audit.audit_arm(
        write_arm(tmp_path, FILTERED, "pack", filtered_docs),
        empty_baseline,
        TAG,
        "all",
        None,
        data_base,
        False,
        0,
        0,
        0,
        0,
        STATS,
        None,
    )
    assert reports[0]["counts"]["baseline_source"] == "filter statistics"
    return checker.failures


class TestPackedCorpusWithoutABaselineRow:
    def test_the_retained_count_is_checked_against_the_statistics(self, tmp_path):
        assert run_packed_counts(tmp_path, 70) == []

    def test_the_whole_unfiltered_corpus_under_the_filtered_name_is_caught(self, tmp_path):
        assert any("n_retained" in f for f in run_packed_counts(tmp_path, 100))

    def test_a_tokenized_corpus_still_needs_its_baseline_row(self, tmp_path):
        data_base = tmp_path / "data"
        build_corpus(corpora_table.corpus_root(DATASET, FILTERED, data_base), subset=FILTERED, docs=70, tokens=770)
        empty_baseline = tmp_path / "baseline.tsv"
        empty_baseline.write_text("# no rows\n")
        _, checker = audit.audit_arm(
            write_arm(tmp_path, FILTERED, "tokenize", 70),
            empty_baseline,
            TAG,
            "all",
            None,
            data_base,
            False,
            0,
            0,
            0,
            0,
            STATS,
            None,
        )
        assert any("no baseline row" in f for f in checker.failures)


class TestAlignment:
    def test_filtered_lengths_align_in_order_and_the_gaps_are_the_removed_documents(self):
        baseline = np.array([5, 7, 8, 3, 9, 2, 7, 4], dtype=np.int64)
        removed = [1, 4, 5]
        filtered = np.delete(baseline, removed)
        alignment = audit.align_by_length(filtered, baseline)
        assert alignment.aligned == len(filtered)
        assert alignment.skipped.tolist() == removed
        assert alignment.match.tolist() == [0, 2, 3, 6, 7]

    def test_a_document_absent_from_the_baseline_fails_to_align(self):
        baseline = np.array([5, 7, 3], dtype=np.int64)
        filtered = np.array([5, 8, 3], dtype=np.int64)
        alignment = audit.align_by_length(filtered, baseline)
        assert alignment.aligned < len(filtered)

    def test_long_runs_and_removals_beyond_the_first_window_are_found(self):
        rng = np.random.default_rng(0)
        baseline = rng.integers(1, 50, size=20_000).astype(np.int64)
        removed = np.sort(rng.choice(len(baseline), size=500, replace=False))
        filtered = np.delete(baseline, removed)
        alignment = audit.align_by_length(filtered, baseline)
        assert alignment.aligned == len(filtered)
        # Greedy pairing by length can pair a filtered document with an equal-length removed
        # neighbour, but never changes how many baseline documents are left unpaired, nor the
        # total length of what was skipped.
        assert len(alignment.skipped) == len(removed)
        assert int(baseline[alignment.skipped].sum()) == int(baseline[removed].sum())

    def test_a_removed_run_longer_than_the_first_search_window_is_skipped(self):
        baseline = np.concatenate([[3], np.full(5000, 7), [3]]).astype(np.int64)
        filtered = np.array([3, 3], dtype=np.int64)
        alignment = audit.align_by_length(filtered, baseline)
        assert alignment.match.tolist() == [0, 5001]
        assert len(alignment.skipped) == 5000


class TestCountIdentity:
    def test_the_identity_holds_only_for_exact_numbers(self):
        stats = STATS[BASE]
        assert audit.count_identity(70, 770, 100, 1100, stats) == []
        assert audit.count_identity(70, 770, 100, 1101, stats)
        assert audit.count_identity(69, 770, 100, 1100, stats)

    def test_a_packed_corpus_is_checked_on_documents_alone(self):
        assert audit.count_identity(70, None, 100, None, STATS[BASE]) == []
        assert audit.count_identity(71, None, 100, None, STATS[BASE])


class TestRemovedRowVerdict:
    def test_absent_from_the_filtered_corpus_is_the_expected_case(self):
        assert audit.classify_removed_row(1, 0, True) == "absent"

    def test_a_row_the_baseline_never_held_is_its_own_finding(self):
        assert audit.classify_removed_row(0, 0, True) == "not_in_baseline"

    def test_more_baseline_copies_than_filtered_copies_is_a_source_duplicate(self):
        assert audit.classify_removed_row(2, 1, True) == "source_duplicate"

    def test_as_many_filtered_copies_as_baseline_copies_is_a_leak(self):
        assert audit.classify_removed_row(1, 1, True) == "leaked"
        assert audit.classify_removed_row(2, 2, True) == "leaked"

    def test_a_clipped_search_refuses_every_other_verdict(self):
        # An empty result from a search that did not cover every candidate is not absence.
        assert audit.classify_removed_row(1, 0, False) == "search_truncated"
        assert audit.classify_removed_row(2, 1, False) == "search_truncated"

    def test_every_verdict_the_audit_counts_is_one_the_rule_can_give(self):
        given = {audit.classify_removed_row(b, f, e) for b in (0, 1, 2) for f in (0, 1, 2) for e in (True, False)}
        assert given == set(audit.REMOVED_ROW_VERDICTS)


DOCUMENTS = [[10, 11, 12, 2], [20, 21, 2], [30, 31, 32, 2], [10, 11, 12, 2], [40, 2], [50, 51, 52, 2]]


@pytest.fixture
def token_corpus(tmp_path):
    """A real two-shard `.bin/.idx` corpus holding DOCUMENTS, read back through the audit's reader."""
    root = tmp_path / "corpus"
    write_tokenized_documents(root / "shard0", DOCUMENTS[:3])
    write_tokenized_documents(root / "shard1", DOCUMENTS[3:])
    config = write_prepare_config(tmp_path)
    (row,) = corpora_table.read_corpora_table(write_table(tmp_path, config, shards=2, shard_mode="slice", docs=6))
    return audit.TokenCorpus(row, root)


class TestTokenCorpus:
    def test_shards_concatenate_in_order(self, token_corpus):
        assert len(token_corpus) == 6
        assert token_corpus.lengths.tolist() == [len(d) for d in DOCUMENTS]
        for index, document in enumerate(DOCUMENTS):
            assert token_corpus.document(index).tolist() == document

    def test_a_prefix_is_read_without_the_whole_document(self, token_corpus):
        assert token_corpus.prefix(2, 2).tolist() == [30, 31]
        assert token_corpus.prefix(4, 8).tolist() == [40, 2]

    def test_copies_finds_every_identical_document_and_says_the_search_was_exhaustive(self, token_corpus):
        found, exhaustive = token_corpus.copies(np.array([10, 11, 12, 2]), None)
        assert found == [0, 3] and exhaustive
        found, exhaustive = token_corpus.copies(np.array([99, 98, 97, 2]), None)
        assert found == [] and exhaustive

    def test_copies_respects_an_explicit_candidate_set(self, token_corpus):
        found, _ = token_corpus.copies(np.array([10, 11, 12, 2]), np.array([3, 4]))
        assert found == [3]

    def test_copies_reports_a_clipped_search(self, token_corpus, monkeypatch):
        monkeypatch.setattr(audit, "COPY_SEARCH_CANDIDATES", 1)
        found, exhaustive = token_corpus.copies(np.array([10, 11, 12, 2]), None)
        assert found == [0] and not exhaustive

    def test_equal_length_runs_expand_an_anchor_to_its_neighbours_of_the_same_length(self, token_corpus):
        # Documents 0, 2, 3 and 5 are four tokens long; 2 and 3 are adjacent, 0 and 5 are alone.
        assert token_corpus.equal_length_runs(np.array([2]), 4).tolist() == [2, 3]
        assert token_corpus.equal_length_runs(np.array([0, 5]), 4).tolist() == [0, 5]
        assert token_corpus.equal_length_runs(np.array([2]), 3).tolist() == []


class TestPackedDocuments:
    def test_every_packed_document_is_hashed_whole_with_its_pads_stripped(self, tmp_path):
        pad = 11
        sequences = [[[5, 6, 7, 11, pad], [8, 9, 11, pad, pad]], [[5, 6, 7, 11, pad]]]
        root = tmp_path / "corpus"
        build_packed_shard(root / "shard0", sequences=sequences[:1])
        build_packed_shard(root / "shard1", sequences=sequences[1:])
        config = write_prepare_config(tmp_path, **{"seq-length": 32768, "pad-seq-to-mult": 4})
        (row,) = corpora_table.read_corpora_table(
            write_table(tmp_path, config, kind="pack", shards=2, shard_mode="split", docs=3)
        )
        hashes, docs, seqs = audit.packed_document_hashes(
            row, root, corpora_table.prepare_config_scalars(config), pad_id=pad
        )
        assert (docs, seqs) == (3, 2)
        # Two of the three packed documents are the same conversation; the trailing pads differ.
        assert len(hashes) == 2
        assert audit._document_hash([5, 6, 7, 11], pad) in hashes
        assert audit._document_hash([8, 9, 11, pad, pad, pad], pad) in hashes

    def test_a_document_hash_ignores_only_trailing_pads(self):
        assert audit._document_hash([1, 2, 3], 0) == audit._document_hash([1, 2, 3, 0, 0], 0)
        assert audit._document_hash([1, 0, 3], 0) != audit._document_hash([1, 3], 0)
        assert audit._document_hash([1, 2], 0) != audit._document_hash([1, 2, 3], 0)


class TestCanaryRows:
    def test_flagged_rows_come_back_with_their_columns_across_files(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        files = []
        for name, texts, flags in (
            ("a.parquet", ["x0", "x1", "x2"], [False, True, None]),
            ("b.parquet", ["y0"], [True]),
        ):
            path = tmp_path / name
            pq.write_table(pa.table({"text": texts, "canary": flags, "mini_score": [3] * len(texts)}), path)
            files.append(path)
        rows, read = audit.flagged_rows((open(f, "rb") for f in files), "canary", ["text"])
        assert (rows, read) == ([{"text": "x1"}, {"text": "y0"}], 4)  # a null flag is not a canary

    def test_a_missing_flag_column_is_an_error_not_a_zero(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "a.parquet"
        pq.write_table(pa.table({"text": ["x"]}), path)
        with pytest.raises(Exception):
            audit.flagged_rows([open(path, "rb")], "canary", ["text"])

    def test_removed_rows_are_classified_by_their_copies_in_the_two_built_corpora(self, tmp_path):
        # The baseline holds DOCUMENTS; the filtered corpus is the baseline without documents 1 and 4.
        write_tokenized_documents(tmp_path / "baseline" / "shard0", DOCUMENTS[:3])
        write_tokenized_documents(tmp_path / "baseline" / "shard1", DOCUMENTS[3:])
        write_tokenized_documents(tmp_path / "filtered" / "shard0", [DOCUMENTS[0], DOCUMENTS[2]])
        write_tokenized_documents(tmp_path / "filtered" / "shard1", [DOCUMENTS[3], DOCUMENTS[5]])
        config = write_prepare_config(tmp_path)
        (row,) = corpora_table.read_corpora_table(write_table(tmp_path, config, shards=2, shard_mode="slice", docs=4))
        filtered, baseline = (
            audit.TokenCorpus(row, tmp_path / "filtered"),
            audit.TokenCorpus(row, tmp_path / "baseline"),
        )
        texts = {"removed b": DOCUMENTS[1], "removed e": DOCUMENTS[4], "never in the source": [99, 2]}
        verdicts = audit.removed_row_verdicts(
            [{"text": text} for text in texts],
            lambda text: np.array(texts[text]),  # the Hub row's text, tokenized as --append-eod wrote it
            "text",
            filtered,
            baseline,
            np.array([1, 4]),  # the alignment's skipped baseline positions
        )
        assert verdicts == {
            "absent": 2,
            "not_in_baseline": 1,
            "source_duplicate": 0,
            "leaked": 0,
            "search_truncated": 0,
        }

    def test_the_columns_a_corpus_was_rendered_from(self, tmp_path):
        config = write_prepare_config(tmp_path)
        (tokenize_row,) = corpora_table.read_corpora_table(write_table(tmp_path, config, docs=1))
        (pack_row,) = corpora_table.read_corpora_table(
            write_table(tmp_path, config, subset="sft", kind="pack", shards=2, shard_mode="split", docs=1)
        )
        assert audit.rendered_columns(tokenize_row, {"text_column": "text"}) == ["text"]
        assert audit.rendered_columns(pack_row, {"text_column": "messages"}) == ["messages", "tools"]


class TestNames:
    def test_the_baseline_subset_is_the_filtered_name_without_its_suffix(self):
        assert audit.baseline_subset("climbmix_full_filtered_mini_2plus", "mini_2plus") == "climbmix_full"
        with pytest.raises(ValueError):
            audit.baseline_subset("climbmix_full", "mini_2plus")

    def test_statistics_rows_need_every_count(self):
        with pytest.raises(KeyError):
            audit.stats_from_rows([{"subset": BASE, "n_total": 1}])
