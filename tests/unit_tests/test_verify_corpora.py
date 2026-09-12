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

from pathlib import Path

import pytest

from tests.unit_tests.corpora_fixtures import (
    CAMPAIGN_DIR,
    DATASET,
    TOKENIZER,
    build_corpus,
    build_packed_shard,
    corpora_table,
    load_campaign_module,
    write_prepare_config,
    write_table,
)


verify_corpora = load_campaign_module("verify_corpora")


def run(table: Path, data_base: Path) -> tuple[int, list[str]]:
    """Verify a table and return (exit status, failure messages)."""
    checker = corpora_table.Checker()
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


class TestHeldCorpora:
    """A corpus held at PENDING is reported, never allowed to suppress the checks around it.

    Holding a corpus back is how an arm keeps a corpus whose source is not yet safe out of the
    build. That is a safety mechanism, so it must not cost coverage anywhere else: neither on
    the stage's other corpora, nor on the held row's own count-independent checks.
    """

    def test_a_held_corpus_is_reported_without_abandoning_the_others(self, tmp_path):
        """A PENDING row is a failure to report, exactly like an unreadable parquet: the corpora
        that ARE built still have to be checked. Letting the row's un-sliceable count propagate
        out of the verifier instead would turn a deliberate hold into a silent gap in coverage,
        which is the opposite of what holding a corpus is for.
        """
        config = write_prepare_config(tmp_path)
        table = write_table(
            tmp_path,
            config,
            subset="held_filtered_mini_2plus",
            shards=4,
            shard_mode="slice",
            docs=corpora_table.DOCS_PENDING,
            extra_rows=[{"subset": "built_filtered_mini_2plus", "docs": 100}],
        )
        data_base = tmp_path / "data"
        build_corpus(
            corpora_table.corpus_root(DATASET, "built_filtered_mini_2plus", data_base),
            subset="built_filtered_mini_2plus",
            docs=100,
            tokens=1000,
        )
        status, failures = run(table, data_base)
        assert status == 1
        assert any("held_filtered_mini_2plus" in failure and "PENDING" in failure for failure in failures)
        assert not any("built_filtered_mini_2plus" in failure for failure in failures)

    def test_a_held_unsliced_corpus_keeps_its_count_independent_checks(self, tmp_path):
        """Only slicing derives anything from the count, so only a sliced hold can skip its
        checks. A corpus held in any other shard mode is still built on disk and must still have
        its revision, tokenizer and bytes-per-token verified — otherwise the hold would mask a
        second, unrelated defect until the count was filled in.
        """
        config = write_prepare_config(tmp_path)
        table = write_table(tmp_path, config, docs=corpora_table.DOCS_PENDING)
        data_base = tmp_path / "data"
        build_corpus(
            corpora_table.corpus_root(DATASET, "demo_filtered_mini_2plus", data_base),
            docs=100,
            tokens=1000,
            tokenizer="wrong/tokenizer",
        )
        status, failures = run(table, data_base)
        assert status == 1
        assert any("PENDING" in failure for failure in failures)
        assert any("tokenizer" in failure for failure in failures)


class TestPlanDerivation:
    """The build plan is derived here and only submitted by the shell script, so the dependency
    wiring that stops a failed step from feeding a truncated input forward is asserted on the
    plan itself, not inferred from a dry run's text."""

    @staticmethod
    def _plan(tmp_path, config_extra: dict | None = None, steps: set[str] | None = None, **overrides):
        config = write_prepare_config(tmp_path, **(config_extra or {}))
        table = write_table(tmp_path, config, **overrides)
        selection = None if steps is None else frozenset(steps)
        (plan,) = corpora_table.plan_build(table, "all", data_base=tmp_path / "data", steps=selection)
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

    @pytest.mark.parametrize(
        "shard_mode,kind,shards", [("none", "tokenize", 1), ("split", "pack", 8), ("slice", "tokenize", 8)]
    )
    def test_a_pending_count_refuses_the_plan_whatever_the_shard_mode(self, tmp_path, shard_mode, kind, shards):
        """PENDING holds a corpus back, and the hold must not depend on how it is sharded.

        Only a sliced corpus *needs* the count in order to plan its index ranges, so refusing
        inside the slice branch alone would let a `none` or `split` row believed held submit
        silently — which is the failure a hold exists to prevent, arriving as a corpus nobody
        meant to build. A count is also what `verify_corpora.py` checks a built corpus against,
        so building without one produces an artifact nothing can confirm.
        """
        with pytest.raises(ValueError, match="document count is PENDING"):
            self._plan(
                tmp_path,
                {"seq-length": 32768, "pad-seq-to-mult": 4} if kind == "pack" else None,
                kind=kind,
                shards=shards,
                shard_mode=shard_mode,
                docs=corpora_table.DOCS_PENDING,
            )

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

    def test_every_job_names_its_step(self, tmp_path):
        geometry = {"seq-length": 32768, "pad-seq-to-mult": 4}
        plan = self._plan(tmp_path, config_extra=geometry, kind="pack", shards=2, shard_mode="split", docs=100)
        assert [job.step for job in plan.jobs] == ["prepare", "split", "pack", "pack"]
        plan = self._plan(tmp_path, shards=2, shard_mode="slice", docs=100)
        assert [job.step for job in plan.jobs] == ["prepare", "tokenize", "prepare", "tokenize"]
        assert set(corpora_table.STEPS) >= {job.step for job in plan.jobs}

    def test_prepare_alone_re_stamps_without_re_tokenizing(self, tmp_path):
        """Moving the pin of an already-tokenized corpus needs its prepare record rewritten and
        nothing else: the tokenize step is left out of the plan, in every shard mode."""
        plan = self._plan(tmp_path, steps={"prepare"})
        assert [(job.step, job.depends_on) for job in plan.jobs] == [("prepare", "")]
        plan = self._plan(tmp_path, steps={"prepare"}, shards=4, shard_mode="slice", docs=100)
        assert [job.step for job in plan.jobs] == ["prepare"] * 4
        assert all(job.depends_on == "" for job in plan.jobs)
        # The directories are still created (and striped) — a prepare writes into them.
        assert len(plan.roots) == 5

    def test_a_kept_step_whose_predecessor_is_omitted_starts_immediately(self, tmp_path):
        """A re-tokenize of a prepared JSONL must not wait on a prepare that is not submitted;
        the job runs against the JSONL already on disk and fails in its own right if it is not
        there."""
        plan = self._plan(tmp_path, steps={"tokenize"}, shards=2, shard_mode="split", docs=100)
        assert [(job.step, job.depends_on) for job in plan.jobs] == [("tokenize", ""), ("tokenize", "")]
        plan = self._plan(tmp_path, steps={"prepare", "tokenize"})
        assert [(job.step, job.depends_on) for job in plan.jobs] == [
            ("prepare", ""),
            ("tokenize", "demo_filtered_mini_2plus:prepare"),
        ]

    def test_an_unknown_step_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown build step"):
            self._plan(tmp_path, steps={"prepare", "tokenise"})


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
        checker = corpora_table.Checker()
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

    def test_a_shard_still_being_packed_is_reported_not_raised(self, packed):
        """A pack job writes its parquet in place, so while it runs the file exists without a
        footer. Verifying mid-build must record that shard as a failure and go on to the other
        corpora, not abort the whole run on the first half-written file."""
        table, data_base, root = packed
        build_packed_shard(root / "shard0")
        build_packed_shard(root / "shard1")
        parquet = root / "shard1" / "packed" / f"{TOKENIZER.replace('/', '--')}_pad_seq_to_mult4"
        parquet = parquet / "training_32768.idx.parquet"
        parquet.write_bytes(parquet.read_bytes()[:64])
        _, failures = run(table, data_base)
        assert any("shard1" in f and "still being written" in f for f in failures)


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

    def test_subset_selection_keeps_only_the_named_rows(self, tmp_path):
        """A partial submission names its rows; the selection is a filter over the stage's rows."""
        config = write_prepare_config(tmp_path)
        table = write_table(tmp_path, config, stage="midtraining")
        (row,) = corpora_table.read_corpora_table(table)
        assert corpora_table.read_corpora_table(table, "all", subsets=[row.subset]) == [row]
        assert corpora_table.read_corpora_table(table, "midtraining", subsets=[row.subset]) == [row]

    def test_a_subset_the_stage_does_not_contain_is_refused(self, tmp_path):
        """A misspelt or wrong-stage subset must not quietly plan nothing: the caller named it
        in order to submit or rebuild it."""
        config = write_prepare_config(tmp_path)
        table = write_table(tmp_path, config, stage="midtraining")
        (row,) = corpora_table.read_corpora_table(table)
        with pytest.raises(ValueError, match="no row for subset \\['nope'\\]"):
            corpora_table.read_corpora_table(table, "all", subsets=["nope"])
        with pytest.raises(ValueError, match="in stage 'pretraining'"):
            corpora_table.read_corpora_table(table, "pretraining", subsets=[row.subset])
        with pytest.raises(ValueError, match="empty subset selection"):
            corpora_table.read_corpora_table(table, "all", subsets=[])

    def test_the_cli_plans_only_the_named_subsets(self, tmp_path, capsys):
        """The build script passes its trailing arguments straight to this entry point."""
        config = write_prepare_config(tmp_path)
        table = write_table(tmp_path, config, stage="midtraining")
        (row,) = corpora_table.read_corpora_table(table)
        assert corpora_table.main([str(table), "midtraining", row.subset]) == 0
        planned = [line for line in capsys.readouterr().out.splitlines() if line.startswith("CORPUS")]
        assert len(planned) == 1 and row.subset in planned[0]
        with pytest.raises(ValueError, match="no row for subset"):
            corpora_table.main([str(table), "all", "nope"])

    def test_the_cli_plans_only_the_named_steps(self, tmp_path, capsys):
        """``BUILD_STEPS=prepare`` reaches this entry point as ``--steps prepare``."""
        config = write_prepare_config(tmp_path)
        table = write_table(tmp_path, config)
        assert corpora_table.main([str(table), "all", "--steps", "prepare"]) == 0
        jobs = [line for line in capsys.readouterr().out.splitlines() if line.startswith("JOB")]
        assert len(jobs) == 1 and ":prepare" in jobs[0]
        with pytest.raises(ValueError, match="unknown build step"):
            corpora_table.main([str(table), "all", "--steps", "prepare,bogus"])

    def test_the_campaign_tables_parse(self):
        """The tables that are actually shipped must satisfy every rule above."""
        for arm in ("30b_baseline", "30b_filtered_mini_2plus"):
            rows = corpora_table.read_corpora_table(CAMPAIGN_DIR / arm / "corpora.tsv")
            assert rows, arm
            for row in rows:
                assert row.config.exists(), f"{arm}: {row.config} does not exist"
