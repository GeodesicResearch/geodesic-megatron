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

"""The corpora-build verifier catches the ways a corpus can be quietly wrong.

`verify_corpora.py` is what stands between a mis-built corpus and a 500B-token run reading it.
A verifier that passes everything is worse than none, because it converts an unchecked build
into one that looks checked — so each test here builds a corpus directory that is correct
except for one defect and asserts that defect is reported.

The corpora are real directories with the real artifacts the data pipeline writes
(`pipeline_results.json`, `<prefix>.provenance.json`, a `.bin` sized to its token count), built
in `tmp_path` and passed to the real functions. Nothing is mocked: the verifier only ever reads
small files, so a fixture can produce genuine inputs for it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIR = _REPO_ROOT / "configs" / "control_pretraining"

DATASET = "geodesic-research/control-pretraining-datasets"
REVISION = "0123456789abcdef0123456789abcdef01234567"
TOKENIZER = "geodesic-research/nemotron-base-tokenizer"


def _load(name: str):
    """Import one of the campaign's build scripts, which live outside the package tree.

    Imported by name off `sys.path` rather than through `spec_from_file_location`, because a
    module exec'd without being registered in `sys.modules` cannot define a dataclass —
    `@dataclass` resolves its own module to check field types and finds `None`.
    """
    if str(_CAMPAIGN_DIR) not in sys.path:
        sys.path.insert(0, str(_CAMPAIGN_DIR))
    return importlib.import_module(name)


corpora_table = _load("corpora_table")
verify_corpora = _load("verify_corpora")


def write_prepare_config(directory: Path, **extra) -> Path:
    """A prepare config in the shape the campaign's own corpus configs use."""
    path = directory / "corpus.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": DATASET,
                "revision": REVISION,
                "tokenizer": TOKENIZER,
                "val-proportion": 0,
                "skip-pack": True,
                "skip-count": True,
                **extra,
            }
        )
    )
    return path


def write_table(directory: Path, config: Path, **overrides) -> Path:
    """One-row corpora table, defaulting to an unsharded tokenize corpus."""
    row = {
        "subset": "demo_filtered_mini_2plus",
        "stage": "pretraining",
        "kind": "tokenize",
        "config": str(config),
        "prep_h": "04",
        "tok_h": "04",
        "workers": "32",
        "shards": "1",
        "shard_mode": "none",
        "stripe": "0",
        "docs": "100",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    path = directory / "corpora.tsv"
    path.write_text("# a table\n" + "|".join(row[column] for column in corpora_table.COLUMNS) + "\n")
    return path


def build_corpus(root: Path, *, docs: int = 100, tokens: int = 1000, split: str = "train", **damage) -> None:
    """Write the artifacts a correct prepare+tokenize leaves behind, then apply one defect.

    Keyword `damage` overrides let a test change exactly one thing: `revision`, `tokenizer`,
    `provenance_docs`, `bin_bytes`, `append_eod`, or `status`.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_results.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "subset": "demo_filtered_mini_2plus",
                "split": split,
                "revision": damage.get("revision", REVISION),
                "tokenizer": TOKENIZER,
                "status": damage.get("status", "completed"),
                "num_documents": docs,
                "training_docs": docs,
            }
        )
    )
    prefix = root / corpora_table.TOKENIZED_PREFIX
    provenance_docs = damage.get("provenance_docs", docs)
    Path(f"{prefix}.provenance.json").write_text(
        json.dumps(
            {
                "totals": {"total_tokens": tokens, "num_sequences": provenance_docs, "num_documents": provenance_docs},
                "parameters": {
                    "tokenizer": damage.get("tokenizer", TOKENIZER),
                    "json_key": "input",
                    "append_eod": damage.get("append_eod", "true"),
                },
            }
        )
    )
    Path(f"{prefix}.bin").write_bytes(b"\0" * damage.get("bin_bytes", 4 * tokens))
    Path(f"{prefix}.idx").write_bytes(b"\0")


def run(table: Path, data_base: Path) -> tuple[int, list[str]]:
    """Verify a table and return (exit status, failure messages)."""
    checker = verify_corpora.Checker()
    for row in corpora_table.read_corpora_table(table):
        verify_corpora.verify_corpus(row, checker, data_base)
    return (1 if checker.failures else 0), checker.failures


@pytest.fixture
def corpus(tmp_path):
    """A correct single-corpus build, ready for one defect to be introduced."""
    config = write_prepare_config(tmp_path)
    table = write_table(tmp_path, config)
    data_base = tmp_path / "data"
    root = corpora_table.corpus_root(DATASET, "demo_filtered_mini_2plus", data_base)
    return table, data_base, root


class TestAcorrectBuildPasses:
    def test_no_failures_on_a_well_formed_corpus(self, corpus):
        table, data_base, root = corpus
        build_corpus(root)
        status, failures = run(table, data_base)
        assert failures == []
        assert status == 0


class TestDefectsAreReported:
    """One defect per test; each is a way a real build has gone or could go wrong."""

    def test_document_count_below_the_table_is_reported(self, corpus):
        """A truncated JSONL or dropped documents: the corpus builds cleanly and is short."""
        table, data_base, root = corpus
        build_corpus(root, docs=99)
        _, failures = run(table, data_base)
        assert any("table says 100" in f for f in failures)

    def test_tokenize_dropping_documents_is_reported(self, corpus):
        """Prepare wrote N documents and tokenize indexed fewer — a partial tokenize."""
        table, data_base, root = corpus
        build_corpus(root, provenance_docs=98)
        _, failures = run(table, data_base)
        assert any("prepare wrote 100" in f for f in failures)

    def test_a_short_bin_is_reported(self, corpus):
        """The .bin must be exactly 4 bytes per token; anything else is a partial write."""
        table, data_base, root = corpus
        build_corpus(root, tokens=1000, bin_bytes=4 * 1000 - 8)
        _, failures = run(table, data_base)
        assert any(".bin is" in f for f in failures)

    def test_the_wrong_tokenizer_is_reported(self, corpus):
        """The documented Base-CPT trap: the wrong EOD miscounts document boundaries with no
        error, and the provenance is the only place the tokenizer is recoverable."""
        table, data_base, root = corpus
        build_corpus(root, tokenizer="geodesic-research/nemotron-instruct-tokenizer")
        _, failures = run(table, data_base)
        assert any("nemotron-instruct-tokenizer" in f for f in failures)

    def test_a_stale_revision_is_reported(self, corpus):
        """A corpus prepared before a re-pin is data from a different snapshot."""
        table, data_base, root = corpus
        build_corpus(root, revision="f" * 40)
        _, failures = run(table, data_base)
        assert any("recorded revision=" in f for f in failures)

    def test_a_missing_append_eod_is_reported(self, corpus):
        table, data_base, root = corpus
        build_corpus(root, append_eod="false")
        _, failures = run(table, data_base)
        assert any("without --append-eod" in f for f in failures)

    def test_an_unfinished_prepare_is_reported(self, corpus):
        table, data_base, root = corpus
        build_corpus(root, status="failed")
        _, failures = run(table, data_base)
        assert any("prepare status" in f for f in failures)

    def test_a_corpus_that_was_never_built_is_reported(self, corpus):
        """The empty case must fail loudly rather than verifying an absent corpus."""
        table, data_base, _ = corpus
        _, failures = run(table, data_base)
        assert any("missing" in f for f in failures)

    def test_every_failure_is_reported_not_only_the_first(self, corpus):
        """One run must surface the whole state of a build; fixing defects one job at a time
        costs a queue round-trip each."""
        table, data_base, root = corpus
        build_corpus(root, docs=99, tokenizer="wrong/tokenizer", append_eod="false")
        _, failures = run(table, data_base)
        assert len(failures) >= 3


class TestSlicedCorpora:
    """ClimbMix is prepared as N contiguous index ranges; the ranges are the correctness claim."""

    @pytest.fixture
    def sliced(self, tmp_path):
        config = write_prepare_config(tmp_path)
        table = write_table(tmp_path, config, shards=4, shard_mode="slice", docs=100)
        data_base = tmp_path / "data"
        root = corpora_table.corpus_root(DATASET, "demo_filtered_mini_2plus", data_base)
        return table, data_base, root

    def test_correct_slices_pass_and_sum_to_the_table(self, sliced):
        table, data_base, root = sliced
        for index, (beginning, end) in enumerate([(0, 25), (25, 50), (50, 75), (75, 100)]):
            build_corpus(root / f"shard{index}", docs=end - beginning, tokens=250, split=f"train[{beginning}:{end}]")
        status, failures = run(table, data_base)
        assert failures == []
        assert status == 0

    def test_a_shard_prepared_from_the_wrong_range_is_reported(self, sliced):
        """Overlapping or gapped ranges silently duplicate or drop documents; the recorded
        split string is what makes that detectable after the fact."""
        table, data_base, root = sliced
        for index, (beginning, end) in enumerate([(0, 25), (20, 45), (50, 75), (75, 100)]):
            build_corpus(root / f"shard{index}", docs=end - beginning, tokens=250, split=f"train[{beginning}:{end}]")
        _, failures = run(table, data_base)
        assert any("expected 'train[25:50]'" in f for f in failures)

    def test_slice_ranges_are_contiguous_and_cover_the_corpus(self, sliced):
        table, _, _ = sliced
        (row,) = corpora_table.read_corpora_table(table)
        ranges = row.slice_ranges()
        assert ranges[0][0] == 0
        assert ranges[-1][1] == row.docs
        assert all(end == nxt for (_, end), (nxt, _) in zip(ranges, ranges[1:]))

    def test_a_pending_document_count_cannot_be_sliced(self, tmp_path):
        """The count is load-bearing for slicing, so its absence must raise rather than default."""
        config = write_prepare_config(tmp_path)
        table = write_table(tmp_path, config, shards=4, shard_mode="slice", docs=corpora_table.DOCS_PENDING)
        (row,) = corpora_table.read_corpora_table(table)
        assert row.docs is None
        with pytest.raises(ValueError, match="PENDING"):
            row.slice_ranges()


class TestPlanDerivation:
    """The build plan is derived here and only submitted by the shell script, so the dependency
    wiring that stops a failed step from feeding a truncated input forward is asserted on the
    plan itself, not inferred from a dry run's text."""

    @staticmethod
    def _plan(tmp_path, config_extra: dict | None = None, **overrides):
        config = write_prepare_config(tmp_path, **(config_extra or {}))
        table = write_table(tmp_path, config, **overrides)
        (plan,) = corpora_table.plan_build(table, "all", data_base=tmp_path / "data")
        return plan

    def test_unsharded_tokenize_waits_on_its_prepare(self, tmp_path):
        plan = self._plan(tmp_path)
        assert [job.key for job in plan.jobs] == [
            "demo_filtered_mini_2plus:prepare",
            "demo_filtered_mini_2plus:tokenize",
        ]
        assert plan.jobs[0].depends_on == ""
        assert plan.jobs[1].depends_on == plan.jobs[0].key
        assert all(job.script == corpora_table.SUBMIT_SCRIPT for job in plan.jobs)
        assert plan.jobs[1].payload[0] == "tokenize"
        assert plan.jobs[1].payload[2] == TOKENIZER
        assert plan.roots == ((plan.root, False),)

    def test_sliced_corpus_chains_each_tokenize_on_its_own_prepare(self, tmp_path):
        plan = self._plan(tmp_path, shards=4, shard_mode="slice", docs=100, stripe=1)
        prepares = [job for job in plan.jobs if job.key.startswith("demo_filtered_mini_2plus:prepare:")]
        tokenizes = [job for job in plan.jobs if job.key.startswith("demo_filtered_mini_2plus:tokenize:")]
        assert len(prepares) == len(tokenizes) == 4
        for index, (prepare, tokenize) in enumerate(zip(prepares, tokenizes)):
            assert prepare.depends_on == ""
            assert tokenize.depends_on == prepare.key
            beginning, end = index * 25, (index + 1) * 25
            assert f"train[{beginning}:{end}]" in prepare.payload
            assert prepare.payload[-1].endswith(f"/shard{index}")
        # The root and every shard root are created, and striped, before any prepare writes.
        assert [stripe for _, stripe in plan.roots] == [True] * 5

    def test_split_pack_corpus_packs_every_shard_after_the_gate(self, tmp_path):
        geometry = {"seq-length": 32768, "pad-seq-to-mult": 4}
        plan = self._plan(tmp_path, config_extra=geometry, kind="pack", shards=3, shard_mode="split", docs=100)
        keys = [job.key for job in plan.jobs]
        assert keys[:2] == ["demo_filtered_mini_2plus:prepare", "demo_filtered_mini_2plus:split"]
        split = plan.jobs[1]
        assert split.depends_on == plan.jobs[0].key
        # The split is submitted AS its own script — not as an argument to the data pipeline's
        # wrapper, which would parse the script path as a dataset root and run a pack instead.
        assert split.script == str(corpora_table.SHARD_SCRIPT)
        assert split.payload == (str(plan.root), "3")
        assert split.hours == corpora_table.SPLIT_HOURS
        assert plan.jobs[0].script == corpora_table.SUBMIT_SCRIPT
        packs = plan.jobs[2:]
        assert len(packs) == 3
        for pack in packs:
            assert pack.depends_on == split.key
            assert pack.script == corpora_table.SUBMIT_SCRIPT
            assert pack.payload[1:] == (TOKENIZER, "32768", "4")
        # A pack's prepare produces JSONL only; that is stated by the config, not by flags.
        assert "--skip-pack" not in plan.jobs[0].payload

    def test_pack_row_without_geometry_in_its_config_is_refused(self, tmp_path):
        """The per-shard packs read seq-length and pad-seq-to-mult from the config; packing at
        the packer's defaults instead would silently mismatch the CP=2 training topology."""
        with pytest.raises(ValueError, match="lacks .*seq-length"):
            self._plan(tmp_path, kind="pack", shards=3, shard_mode="split", docs=100)

    def test_job_names_carry_the_arm(self, tmp_path):
        plan = self._plan(tmp_path)
        arm = tmp_path.name
        assert all(job.name.startswith(f"cp-{arm}-") for job in plan.jobs)

    def test_emitted_plan_round_trips_empty_dependency_fields(self, tmp_path):
        """The shell consumer must see an empty ``depends_on`` as empty, not have the next
        field shifted into its place — which is what a whitespace separator would do."""
        plan = self._plan(tmp_path)
        lines = corpora_table.emit_plan([plan]).splitlines()
        job_lines = [ln for ln in lines if ln.startswith("JOB")]
        fields = job_lines[0].split(corpora_table.PLAN_FIELD_SEPARATOR)
        assert fields[1] == plan.jobs[0].key
        assert fields[2] == ""
        assert fields[3] == str(plan.jobs[0].hours)
        payload_at = fields.index("PAYLOAD")
        assert fields[payload_at + 1] == plan.jobs[0].script
        assert tuple(fields[payload_at + 2 :]) == plan.jobs[0].payload


def build_packed_shard(root: Path, *, records: int = 40, packs: int = 3, **damage) -> None:
    """Write what a per-shard pack leaves behind: the packer's JSONL index and the parquet.

    ``damage`` overrides let a test remove exactly one artifact: ``index`` or ``parquet``.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    if damage.get("index", True):
        np.save(root / "training.jsonl.idx.npy", np.arange(1, records + 1, dtype=np.int64) * 100)
    if damage.get("parquet", True):
        pack_dir = root / "packed" / f"{TOKENIZER.replace('/', '--')}_pad_seq_to_mult4"
        pack_dir.mkdir(parents=True)
        pq.write_table(pa.table({"input_ids": [[1, 2, 3]] * packs}), pack_dir / "training_32768.idx.parquet")


class TestPackedCorpora:
    """The verifier's pack path reads the packer's own artifacts: the JSONL index it builds
    (one entry per record) for the document count, and the parquet's row count for packs."""

    @pytest.fixture
    def packed(self, tmp_path):
        config = write_prepare_config(tmp_path, **{"seq-length": 32768, "pad-seq-to-mult": 4})
        table = write_table(tmp_path, config, kind="pack", shards=2, shard_mode="split", docs=80)
        data_base = tmp_path / "data"
        root = corpora_table.corpus_root(DATASET, "demo_filtered_mini_2plus", data_base)
        build_corpus(root, docs=80)  # the prepare's own record at the corpus root
        return table, data_base, root

    def test_correct_shards_pass_and_report_their_counts(self, packed):
        table, data_base, root = packed
        build_packed_shard(root / "shard0", records=40, packs=3)
        build_packed_shard(root / "shard1", records=40, packs=5)
        checker = verify_corpora.Checker()
        (row,) = corpora_table.read_corpora_table(table)
        report = verify_corpora.verify_corpus(row, checker, data_base)
        assert checker.failures == []
        assert report["docs"] == 80
        assert report["packs"] == 8

    def test_shards_that_do_not_sum_to_the_prepared_count_are_reported(self, packed):
        """The byte-gated split is exact; a shard with fewer records than it should have is a
        truncated shard, and the sum against the prepare's count is what catches it."""
        table, data_base, root = packed
        build_packed_shard(root / "shard0", records=40)
        build_packed_shard(root / "shard1", records=39)
        _, failures = run(table, data_base)
        assert any("shards hold 79 documents, prepare wrote 80" in f for f in failures)

    def test_a_shard_that_never_packed_is_reported(self, packed):
        table, data_base, root = packed
        build_packed_shard(root / "shard0")
        build_packed_shard(root / "shard1", parquet=False)
        _, failures = run(table, data_base)
        assert any("shard1" in f and "training_32768.idx.parquet" in f for f in failures)

    def test_a_shard_whose_index_was_never_built_is_reported(self, packed):
        table, data_base, root = packed
        build_packed_shard(root / "shard0")
        build_packed_shard(root / "shard1", index=False)
        _, failures = run(table, data_base)
        assert any("training.jsonl.idx.npy" in f for f in failures)


class TestTableParsing:
    """The table is shared by the build script and the verifier; both refuse the same rows."""

    def test_a_pending_count_is_reported_as_unverifiable(self, tmp_path):
        config = write_prepare_config(tmp_path)
        table = write_table(tmp_path, config, docs=corpora_table.DOCS_PENDING)
        data_base = tmp_path / "data"
        build_corpus(corpora_table.corpus_root(DATASET, "demo_filtered_mini_2plus", data_base))
        _, failures = run(table, data_base)
        assert any("PENDING" in f for f in failures)

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"kind": "archive"}, "kind must be one of"),
            ({"shard_mode": "halve"}, "shard_mode must be one of"),
            ({"shard_mode": "none", "shards": 4}, "shard_mode=none requires shards=1"),
            ({"shard_mode": "split", "shards": 1}, "requires shards>=2"),
            ({"kind": "pack", "shard_mode": "slice", "shards": 4}, "kind=pack shards through the byte-gated split"),
            ({"stripe": 2}, "stripe must be 0 or 1"),
        ],
    )
    def test_malformed_rows_are_refused(self, tmp_path, overrides, message):
        table = write_table(tmp_path, write_prepare_config(tmp_path), **overrides)
        with pytest.raises(ValueError, match=message):
            corpora_table.read_corpora_table(table)

    def test_a_duplicated_subset_is_refused(self, tmp_path):
        """Two rows for one subset would prepare the same corpus twice into one directory."""
        config = write_prepare_config(tmp_path)
        table = write_table(tmp_path, config)
        table.write_text(table.read_text() + table.read_text().splitlines()[-1] + "\n")
        with pytest.raises(ValueError, match="listed more than once"):
            corpora_table.read_corpora_table(table)

    def test_stage_selection_filters_rows(self, tmp_path):
        config = write_prepare_config(tmp_path)
        table = write_table(tmp_path, config, stage="midtraining")
        assert len(corpora_table.read_corpora_table(table, "midtraining")) == 1
        assert corpora_table.read_corpora_table(table, "pretraining") == []
        assert len(corpora_table.read_corpora_table(table, "all")) == 1

    def test_the_campaign_tables_parse(self):
        """The tables that are actually shipped must satisfy every rule above."""
        for arm in ("30b_baseline", "30b_filtered_mini_2plus"):
            rows = corpora_table.read_corpora_table(_CAMPAIGN_DIR / arm / "corpora.tsv")
            assert rows, arm
            for row in rows:
                assert row.config.exists(), f"{arm}: {row.config} does not exist"
