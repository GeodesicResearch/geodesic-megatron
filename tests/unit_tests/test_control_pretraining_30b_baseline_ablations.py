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

"""An ablation of a baseline stage differs from its parent in the ablated fields and nothing else.

The half-batch SFT ablation is evidence about batch size only to the extent that batch size
and step count are the only things that moved. So the assertions come in two halves, as the
smoke-run tests' do: the set of fields that differ between the merged ablation and its merged
parent must equal exactly the ablated fields plus the run identity (where its checkpoints,
W&B run and TensorBoard events go, which MUST differ or the ablation would overwrite the
parent's artifact), and the ablated fields must relate to the parent's by the stated rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from tests.unit_tests.campaign_config import (
    assert_only_these_fields_differ,
    assert_segment_exit_posture,
    flatten_merged_config,
    merge_onto_recipe,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIR = _REPO_ROOT / "configs" / "control_pretraining"
PARENT = _CAMPAIGN_DIR / "30b_baseline" / "nemotron_nano_30b_baseline_sft.yaml"
ABLATION = _CAMPAIGN_DIR / "30b_baseline_ablations" / "nemotron_nano_30b_baseline_sft_gbs256.yaml"

# The ablation halves the batch and doubles the steps; the identity fields must differ so that
# nothing of the parent's is overwritten. No other field may move.
ALLOWED_DIVERGENCE = {
    "train.global_batch_size",
    "train.train_iters",
    "checkpoint.load",
    "checkpoint.save",
    "logger.wandb_exp_name",
    "logger.tensorboard_dir",
}
IDENTITY_FIELDS = ("checkpoint.load", "checkpoint.save", "logger.wandb_exp_name", "logger.tensorboard_dir")
PARENT_GPUS = 512
ABLATION_GPUS = 256


@pytest.fixture(scope="module")
def ablation():
    return merge_onto_recipe(ABLATION, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module")
def parent():
    return merge_onto_recipe(PARENT, nemotron_3_nano_sft_config)


def data_parallel_size(cfg, gpus: int) -> int:
    return gpus // (
        cfg.model.tensor_model_parallel_size * cfg.model.context_parallel_size * cfg.model.pipeline_model_parallel_size
    )


class TestOnlyTheAblatedFieldsDiffer:
    def test_exactly_the_ablated_and_identity_fields_differ(self, ablation, parent):
        assert_only_these_fields_differ(ablation, parent, ALLOWED_DIVERGENCE, "sft ablation")

    def test_the_batch_is_halved_and_the_steps_doubled(self, ablation, parent):
        assert ablation.train.global_batch_size * 2 == parent.train.global_batch_size
        assert ablation.train.train_iters == 2 * parent.train.train_iters
        assert ablation.train.micro_batch_size == parent.train.micro_batch_size

    def test_the_model_sees_exactly_the_parent_samples(self, ablation, parent):
        """Half the batch for twice the steps is the same two epochs, pack for pack."""
        assert (
            ablation.train.global_batch_size * ablation.train.train_iters
            == parent.train.global_batch_size * parent.train.train_iters
        )

    def test_warm_starts_from_the_same_midtraining_final(self, ablation, parent):
        assert ablation.checkpoint.pretrained_checkpoint == parent.checkpoint.pretrained_checkpoint
        assert ablation.checkpoint.pretrained_checkpoint.endswith("control_pretrain_30b_baseline_midtrain")


class TestTheRunIdentityIsItsOwn:
    def test_no_output_location_is_the_parents(self, ablation, parent):
        flat_ablation, flat_parent = flatten_merged_config(ablation), flatten_merged_config(parent)
        for field in IDENTITY_FIELDS:
            assert flat_ablation[field] != flat_parent[field], field

    def test_the_save_directory_is_not_inside_the_parents(self, ablation, parent):
        """A save under the parent's tree would pass the inequality above and still collide."""
        parent_save = Path(parent.checkpoint.save)
        assert not Path(ablation.checkpoint.save).is_relative_to(parent_save)
        assert not parent_save.is_relative_to(Path(ablation.checkpoint.save))

    def test_a_resubmission_resumes(self, ablation):
        assert ablation.checkpoint.load == ablation.checkpoint.save
        assert ablation.checkpoint.save_interval < ablation.train.train_iters

    def test_the_raw_yaml_copies_no_output_path_from_the_parent(self):
        """Guards the file itself, not the merge: a stale copy of the parent's path in a comment
        would not merge, but one in a value would, and the merged test above would catch it
        only for the fields it lists — so the raw output values are checked directly."""
        raw_ablation, raw_parent = OmegaConf.load(ABLATION), OmegaConf.load(PARENT)
        for section, key in (
            ("checkpoint", "load"),
            ("checkpoint", "save"),
            ("logger", "tensorboard_dir"),
            ("logger", "wandb_exp_name"),
        ):
            assert raw_ablation[section][key] != raw_parent[section][key], f"{section}.{key}"


class TestTheBatchFits256Gpus:
    def test_the_per_replica_load_is_the_parents(self, ablation, parent):
        """At half the GPUs, the halved batch is still the parent's packs per replica per
        iteration, so the step time is expected to match and only the wall clock doubles."""
        ablation_dp = data_parallel_size(ablation, ABLATION_GPUS)
        parent_dp = data_parallel_size(parent, PARENT_GPUS)
        assert ablation.train.global_batch_size % (ablation_dp * ablation.train.micro_batch_size) == 0
        assert ablation.train.global_batch_size // ablation_dp == parent.train.global_batch_size // parent_dp

    def test_expert_parallelism_folds_into_the_data_parallel_size(self, ablation):
        dp = data_parallel_size(ablation, ABLATION_GPUS)
        assert (dp * ablation.model.tensor_model_parallel_size * ablation.model.context_parallel_size) % (
            ablation.model.expert_model_parallel_size
        ) == 0


class TestSegmentRollover:
    def test_ends_on_the_duration_clock_like_its_parent(self, ablation, parent):
        assert_segment_exit_posture(ablation, "sft ablation", parent.train.exit_duration_in_mins)
