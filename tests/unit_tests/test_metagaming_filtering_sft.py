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

"""The metagaming-filtered SFT differs from its unfiltered baseline in the corpus and nothing else.

The arm is evidence about metagaming filtering only to the extent that the post-training corpus is
the one thing that moved. Its baseline is the control-pretraining XL SFT, so the assertions come in
three parts: the merged training configs differ in exactly the corpus fields plus the run identity
(which MUST differ, or the arm would overwrite the baseline's checkpoints and W&B run); the corpus
is built by the baseline's chain at the baseline's geometry, the data config and the corpora row
differing only in what names the corpus; and the corpus is the rebalanced split, which the
dataset's naming makes easy to get wrong.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from tests.unit_tests.campaign_config import (
    assert_iterations_are_the_minimal_cover,
    assert_only_these_fields_differ,
    assert_segment_exit_posture,
    flatten_merged_config,
    merge_onto_recipe,
)
from tests.unit_tests.corpora_fixtures import corpora_table


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = _REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline_ablations"
_ARM_DIR = _REPO_ROOT / "configs" / "metagaming_filtering" / "30b_sft_luna_2plus"
BASELINE = _BASELINE_DIR / "nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml"
BASELINE_DATA = _BASELINE_DIR / "data" / "pa-warm-start-sft-xl-50b-mix.yaml"
BASELINE_TABLE = _BASELINE_DIR / "corpora.tsv"
ARM = _ARM_DIR / "nemotron_nano_30b_metagaming_sft_luna_2plus.yaml"
ARM_DATA = _ARM_DIR / "data" / "pa-warm-start-sft-xl-50b-mix-metagaming-rebalanced-luna-2plus.yaml"
ARM_TABLE = _ARM_DIR / "corpora.tsv"

RUN_NAME = "mf_30b_sft_luna_2plus"
CORPUS = "geodesic-research/metagaming-filtering-datasets"
SUBSET = "pa-warm-start-sft-xl-50b-mix-metagaming_rebalanced_luna_2plus"
CORPUS_REVISION = "74284605eda69d58d076eec7e6702d201d8f2c39"
# The `train` split's row count at the pinned revision, from the dataset-builder's push notice;
# the corpora row carries it so the verifier can check the prepared JSONL against it.
CORPUS_CONVERSATIONS = 9_038_928
EPOCHS = 1
# The measured corpus: the sum of the 32 shards' packed rows, read from the parquet footers by
# verify_corpora.py. train_iters is derived from it, and the pin below holds the config to it.
PACKS = 1_529_658

CORPUS_FIELDS = {
    "dataset.dataset_name",
    "dataset.dataset_root",
    "dataset.packed_sequence_specs.packed_train_data_path",
}
IDENTITY_FIELDS = {"checkpoint.load", "checkpoint.save", "logger.wandb_exp_name"}
# The corpus fields and the run identity. train_iters is one epoch over this corpus's measured
# pack and happens to equal the baseline's. Set equality, not containment: a field cannot start
# differing without being named here.
ALLOWED_DIVERGENCE = CORPUS_FIELDS | IDENTITY_FIELDS


@pytest.fixture(scope="module")
def arm():
    return merge_onto_recipe(ARM, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module")
def baseline():
    return merge_onto_recipe(BASELINE, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module")
def arm_row():
    """The arm's single corpora row, parsed by the module the build and the verifier use."""
    (row,) = corpora_table.read_corpora_table(ARM_TABLE)
    return row


@pytest.fixture(scope="module")
def baseline_row():
    (row,) = corpora_table.read_corpora_table(BASELINE_TABLE)
    return row


class TestOnlyTheCorpusDiffers:
    def test_exactly_the_corpus_and_identity_fields_differ(self, arm, baseline):
        assert_only_these_fields_differ(arm, baseline, ALLOWED_DIVERGENCE, "metagaming-filtered sft")

    def test_warm_starts_from_the_baselines_midtraining_final(self, arm, baseline):
        assert arm.checkpoint.pretrained_checkpoint == baseline.checkpoint.pretrained_checkpoint
        assert arm.checkpoint.pretrained_checkpoint.endswith("control_pretrain_30b_baseline_midtrain")

    def test_the_iteration_count_is_one_pass_over_the_pack(self, arm):
        assert arm.train.train_iters == 5976
        assert_iterations_are_the_minimal_cover(
            arm.train.train_iters, arm.train.global_batch_size, EPOCHS * PACKS, "metagaming-filtered sft"
        )

    def test_ends_on_the_duration_clock_like_its_baseline(self, arm, baseline):
        assert_segment_exit_posture(arm, "metagaming-filtered sft", baseline.train.exit_duration_in_mins)


class TestTheRunIdentityIsItsOwn:
    def test_no_output_location_is_the_baselines(self, arm, baseline):
        flat_arm, flat_baseline = flatten_merged_config(arm), flatten_merged_config(baseline)
        for field in sorted(IDENTITY_FIELDS):
            assert flat_arm[field] != flat_baseline[field], field

    def test_the_save_directory_is_not_nested_with_the_baselines(self, arm, baseline):
        """A save under the baseline's tree would pass the inequality above and still collide."""
        arm_save, baseline_save = Path(arm.checkpoint.save), Path(baseline.checkpoint.save)
        assert not arm_save.is_relative_to(baseline_save)
        assert not baseline_save.is_relative_to(arm_save)

    def test_the_run_is_named_by_the_campaign_convention(self, arm):
        assert Path(arm.checkpoint.save).name == RUN_NAME
        assert Path(arm.checkpoint.save).parent.name == "metagaming_filtering"
        assert arm.logger.wandb_exp_name == RUN_NAME

    def test_a_resubmission_resumes(self, arm):
        assert arm.checkpoint.load == arm.checkpoint.save
        assert arm.checkpoint.save_interval < arm.train.train_iters


class TestTheCorpusIsBuiltLikeTheBaselines:
    def test_the_data_config_differs_only_in_what_names_the_corpus(self):
        arm_data = OmegaConf.to_container(OmegaConf.load(ARM_DATA))
        baseline_data = OmegaConf.to_container(OmegaConf.load(BASELINE_DATA))
        assert set(arm_data) == set(baseline_data)
        differing = {key for key in arm_data if arm_data[key] != baseline_data[key]}
        assert differing == {"dataset", "revision"}
        assert arm_data["dataset"] == CORPUS

    def test_the_corpora_row_differs_only_in_what_names_the_corpus(self, arm_row, baseline_row):
        # Split mode, shard count, walltimes and striping are the baseline's, so the two packs come
        # out of the same chain; the shard count in particular is the pack job's memory budget.
        renamed = dataclasses.replace(baseline_row, subset=arm_row.subset, config=arm_row.config, docs=arm_row.docs)
        assert arm_row == renamed

    def test_the_row_builds_from_this_arms_data_config(self, arm_row):
        assert arm_row.config.resolve() == ARM_DATA.resolve()


class TestTheCorpusIsTheRebalancedSplit:
    """In the dataset repository `_filtered_` names the REMOVED documents and `_retained_` the kept
    ones, while the upstream ratings repository uses `_filtered_` for the kept side; only the
    rebalanced split is the training mix."""

    def test_the_subset_is_the_rebalanced_mix(self, arm_row):
        assert arm_row.subset == SUBSET
        assert "_filtered_" not in arm_row.subset
        assert "_retained_" not in arm_row.subset

    def test_the_data_config_pins_the_pushed_revision(self):
        arm_data = OmegaConf.load(ARM_DATA)
        assert (arm_data.dataset, arm_data.revision, arm_data.split) == (CORPUS, CORPUS_REVISION, "train")

    def test_the_corpora_row_checks_the_pushed_row_count(self, arm_row):
        assert arm_row.docs == CORPUS_CONVERSATIONS

    def test_the_training_config_reads_the_pack_this_row_builds(self, arm, arm_row):
        """The corpus root and the per-shard packed glob are derived by the module the build
        uses, so the training config cannot name a pack the build does not write."""
        root = corpora_table.corpus_root(CORPUS, arm_row.subset)
        scalars = corpora_table.prepare_config_scalars(ARM_DATA)
        assert arm.dataset.dataset_name == CORPUS
        assert arm.dataset.dataset_root == str(root)
        packed = corpora_table.packed_parquet_path(root / "shard*", scalars)
        assert arm.dataset.packed_sequence_specs.packed_train_data_path == str(packed)
