# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Clueless-Norm builds Normal-Norm's corpora, the same way, from the hidden-span dataset.

Clueless-Norm retrains Normal-Norm (the control-pretraining baseline) on the same corpora with the
flagged spans hidden. Its data build is evidence about hiding spans only if everything else about
each corpus is Normal-Norm's. So its table must name exactly the corpora Normal-Norm's two training
stages read, each built by Normal-Norm's row: the same stage, walltimes, workers and stripe, from a
prepare config that is Normal-Norm's except for the dataset and how it is pinned and read. Every
corpus with hidden spans must carry the per-document count of the hidden-token id, and the one
corpus that is a selection must select from Normal-Norm's tokenized corpus. A row may be counted
only once its subset is pinned, and then at Normal-Norm's count, because the dataset keeps
Normal-Norm's rows one for one.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest
from scripts.training.config_compose import load_composed_yaml

from tests.unit_tests.campaign_config import blend_subsets, dry_run_build
from tests.unit_tests.corpora_fixtures import corpora_table


_REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = _REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline"
BASELINE_TABLE = BASELINE_DIR / "corpora.tsv"
BASELINE_DATA = BASELINE_DIR / "data" / "control-pretraining-datasets.yaml"
BASELINE_STAGES = (
    BASELINE_DIR / "nemotron_nano_30b_baseline_pretrain.yaml",
    BASELINE_DIR / "nemotron_nano_30b_baseline_midtrain.yaml",
)
ARM_DIR = _REPO_ROOT / "configs" / "metagaming_filtering" / "30b_clueless_norm"
ARM_TABLE = ARM_DIR / "corpora.tsv"
ARM_DATA = ARM_DIR / "data" / "metagaming-filtering-training-datasets.yaml"
ARM_SELECT = ARM_DIR / "data" / "nemotron_stem_sft_select.yaml"

DATASET = "geodesic-research/metagaming-filtering-training-datasets"
# Every hidden token is a literal `<SPECIAL_500>`, and `n_hidden` states how many each document holds.
HIDDEN_TOKEN = 500
HIDDEN_COUNT_COLUMN = "n_hidden"
# Each row's index in Normal-Norm's source subset, which the dataset keeps one for one.
SOURCE_ROW_COLUMN = "source_row"
# The one corpus that is a selection of Normal-Norm's tokenized corpus rather than a text with spans hidden.
SELECTED = "nemotron_stem_sft"
# Normal-Norm's tokenizer at the commit the label projection and the digest lists were computed with.
TOKENIZER_PIN = "474397005d569f713caf570aed3297841913d051"
# What the arm's prepare config may state that Normal-Norm's does not, and the reverse.
ARM_ONLY_KEYS = {"revisions", "streaming", "text-column", "tokenizer-revision"}
BASELINE_ONLY_KEYS = {"revision"}


class NormalNormCorpus(NamedTuple):
    """One corpus Clueless-Norm builds, as Normal-Norm builds it: Normal-Norm's row, the first of its source's rows
    the corpus holds, and how many documents it holds."""

    row: corpora_table.CorpusRow
    first_row: int
    docs: int


def normal_norm_corpora() -> dict[str, NormalNormCorpus]:
    """Each corpus Clueless-Norm builds, by subset.

    These are the corpora Normal-Norm's training stages read. A sliced corpus is published one config
    per slice, so it is one entry per slice, named ``<subset>_shard<k>``, holding that slice's rows.
    """
    trained = {
        subset
        for stage in BASELINE_STAGES
        for subset in blend_subsets(load_composed_yaml(stage)["dataset"]["data_path"])
    }
    rows = corpora_table.read_corpora_table(BASELINE_TABLE)
    assert trained <= {row.subset for row in rows}, "Normal-Norm trains on a corpus its table does not build"
    corpora = {}
    for row in rows:
        if row.subset not in trained:
            continue
        if row.shard_mode == "slice":
            for index, (beginning, end) in enumerate(row.slice_ranges()):
                corpora[f"{row.subset}_{corpora_table.shard_name(index)}"] = NormalNormCorpus(
                    row, beginning, end - beginning
                )
        else:
            corpora[row.subset] = NormalNormCorpus(row, 0, row.docs)
    return corpora


@pytest.fixture(scope="module")
def corpora():
    return normal_norm_corpora()


@pytest.fixture(scope="module")
def arm_rows():
    return {row.subset: row for row in corpora_table.read_corpora_table(ARM_TABLE)}


@pytest.fixture(scope="module")
def revisions():
    return corpora_table.prepare_config_scalars(ARM_DATA)["revisions"]


def test_the_table_builds_exactly_the_corpora_normal_norm_trains_on(corpora, arm_rows):
    assert set(arm_rows) == set(corpora)
    # ClimbMix's full corpus is the one Normal-Norm slices; its slices number its shards.
    sliced = {corpus.row.subset for subset, corpus in corpora.items() if subset != corpus.row.subset}
    assert sliced == {"climbmix_full"}
    assert len([subset for subset in arm_rows if subset.startswith("climbmix_full_shard")]) == 8


def test_each_row_is_built_as_normal_norms(corpora, arm_rows):
    for subset, row in arm_rows.items():
        parent = corpora[subset].row
        assert (row.stage, row.tok_h, row.stripe) == (parent.stage, parent.tok_h, parent.stripe), subset
        if subset == SELECTED:
            # A selection has no prepare and copies in one process; the parser holds prep_h to 0 and workers to 1.
            assert (row.kind, row.config) == ("select", ARM_SELECT), subset
            assert (row.shards, row.shard_mode) == (parent.shards, parent.shard_mode), subset
        else:
            assert (row.kind, row.config) == ("tokenize", ARM_DATA), subset
            assert (row.prep_h, row.workers) == (parent.prep_h, parent.workers), subset
            # Unsharded: a stream reads a whole named split, so a sliced corpus is published one config per slice.
            assert (row.shards, row.shard_mode) == (1, "none"), subset


def test_every_corpus_with_hidden_spans_counts_the_hidden_token_from_normal_norms_rows(corpora, arm_rows):
    """Each tokenized corpus is checked document by document: its count of the hidden token, and that its dataset's
    rows are Normal-Norm's source rows in order from where Normal-Norm's corpus (or slice) begins."""
    checked = {subset for subset, row in arm_rows.items() if row.count_token is not None}
    assert checked == set(arm_rows) - {SELECTED}
    for subset in checked:
        row = arm_rows[subset]
        assert (row.count_token, row.count_column) == (HIDDEN_TOKEN, HIDDEN_COUNT_COLUMN), subset
        assert (row.row_column, row.first_row) == (SOURCE_ROW_COLUMN, corpora[subset].first_row), subset


def test_a_row_is_counted_exactly_when_its_subset_is_pinned(corpora, arm_rows, revisions):
    tokenized = {subset for subset, row in arm_rows.items() if row.kind == "tokenize"}
    assert set(revisions) <= tokenized, f"pins for subsets the table does not build: {set(revisions) - tokenized}"
    for subset in tokenized:
        row = arm_rows[subset]
        assert (row.docs is None) == (subset not in revisions), (
            f"{subset}: docs={row.docs}, pinned={subset in revisions}"
        )
        if row.docs is not None:
            # The dataset keeps Normal-Norm's rows one for one: a slice's rows, or the whole corpus's.
            assert row.docs == corpora[subset].docs, subset


def test_each_pinned_row_plans_a_streamed_prepare_at_its_own_commit(arm_rows, revisions):
    for subset, pin in revisions.items():
        assert corpora_table.subset_prepare_config(ARM_DATA, subset)["revision"] == pin
        plan = corpora_table.plan_corpus(arm_rows[subset], ARM_DIR.name)
        assert plan.root == corpora_table.corpus_root(DATASET, subset)
        prepare, tokenize = plan.jobs
        assert prepare.payload == ("prepare", "--config", str(ARM_DATA), "--subset", subset)
        assert (tokenize.step, tokenize.depends_on) == ("tokenize", prepare.key)
        # The tokenize job reads Normal-Norm's tokenizer at its pinned commit.
        assert tokenize.payload[2] == f"geodesic-research/nemotron-base-tokenizer@{TOKENIZER_PIN}"


def test_an_unpinned_subset_is_refused_rather_than_read_at_head(arm_rows, revisions):
    unpinned = sorted(subset for subset, row in arm_rows.items() if row.kind == "tokenize" and subset not in revisions)
    assert unpinned, "every subset is pinned; this check needs a held one to refuse"
    with pytest.raises(ValueError, match=f"pins no commit for subset '{unpinned[0]}'"):
        corpora_table.subset_prepare_config(ARM_DATA, unpinned[0])


def test_the_prepare_config_is_normal_norms_but_for_the_corpus():
    baseline = corpora_table.prepare_config_scalars(BASELINE_DATA)
    arm = corpora_table.prepare_config_scalars(ARM_DATA)
    assert arm["dataset"] == DATASET
    shared = set(baseline) - BASELINE_ONLY_KEYS - {"dataset"}
    assert set(arm) - {"dataset"} == shared | ARM_ONLY_KEYS
    assert {key: arm[key] for key in shared} == {key: baseline[key] for key in shared}
    # Streamed, and the hidden-span text is the document: `original_text` beside it is the unhidden source.
    assert (arm["streaming"], arm["text-column"]) == (True, "text")
    # Normal-Norm's tokenizer, pinned at one commit.
    assert (arm["tokenizer"], arm["tokenizer-revision"]) == (baseline["tokenizer"], TOKENIZER_PIN)


def test_the_selection_selects_from_normal_norms_corpus(corpora, arm_rows):
    config = corpora_table.read_select_config(ARM_SELECT)
    parent = corpora[SELECTED].row
    assert (config.parent_table.resolve(), config.parent_subset) == (BASELINE_TABLE.resolve(), parent.subset)
    assert config.dataset == DATASET
    # A hidden span is a run of the hidden token, so no document copied from Normal-Norm may hold one.
    assert config.absent_token_ids == (HIDDEN_TOKEN,)
    # The kept list and its length are delivered together, so they are filled in together.
    kept = corpora_table.prepare_config_scalars(ARM_SELECT)["kept"]
    row = arm_rows[SELECTED]
    assert (kept == corpora_table.DOCS_PENDING) == (row.docs is None)
    if row.docs is not None:
        plan = corpora_table.plan_corpus(row, ARM_DIR.name)
        assert plan.root == corpora_table.corpus_root(DATASET, SELECTED)


def test_the_build_submits_the_pinned_corpora_and_holds_the_rest(arm_rows, revisions):
    """Through the real build_corpora.sh under DRY_RUN: the pinned subsets of a stage plan their
    prepare and tokenize jobs under the arm's names, and the stage as a whole is refused while any
    of its rows is held."""
    for stage in ("pretraining", "midtraining"):
        pinned = [subset for subset in revisions if arm_rows[subset].stage == stage]
        result = dry_run_build(ARM_TABLE, stage, *pinned)
        assert result.returncode == 0, result.stderr
        for subset in pinned:
            for step in ("prep", "tok"):
                assert f"--job-name=cp-{ARM_DIR.name}-{step}-{subset} " in result.stderr, (stage, subset, step)
        assert f"SUBMITTED {2 * len(pinned)} jobs" in result.stdout
        held = dry_run_build(ARM_TABLE, stage)
        assert held.returncode != 0 and "PENDING" in held.stderr, stage
