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
LONG_COT = _CAMPAIGN_DIR / "30b_baseline_ablations" / "nemotron_nano_30b_baseline_sft_long_cot_gbs256.yaml"
LONG_COT_DATA = _CAMPAIGN_DIR / "30b_baseline_ablations" / "data" / "pa-warm-start-sft-heavy-25b-mix-long.yaml"

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


# --- The long-chain-of-thought ablation -------------------------------------------------------
#
# This variant changes the corpus rather than the batch, and its comparison is against the
# half-batch ablation above, which already holds global batch 256, the same warm start and the
# same topology.

LONG_COT_CORPUS = "geodesic-research/pa-warm-start-sft-heavy-25b-mix-long"
LONG_COT_REVISION = "5973da9e94eb0d8957e817294193af065329688e"

# What differs from the SIBLING today: the corpus and the run identity, and nothing else. The
# assertion demands set equality, so this is also the tripwire on the provisional train_iters —
# the config ships the sibling's 5976 until the sixteen shard packs are measured, and the moment
# the measured value replaces it this set is one short and the test fails, which is what forces
# the count to be pinned here rather than trusted to a comment.
LONG_COT_DIVERGENCE = {
    "dataset.dataset_name",
    "dataset.dataset_root",
    "dataset.packed_sequence_specs.packed_train_data_path",
    "checkpoint.load",
    "checkpoint.save",
    "logger.wandb_exp_name",
    "logger.tensorboard_dir",
}


@pytest.fixture(scope="module")
def long_cot():
    return merge_onto_recipe(LONG_COT, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module")
def long_cot_data_config():
    return OmegaConf.load(LONG_COT_DATA)


class TestOnlyTheCorpusDiffersFromTheSibling:
    def test_exactly_the_corpus_and_identity_fields_differ(self, long_cot, ablation):
        assert_only_these_fields_differ(long_cot, ablation, LONG_COT_DIVERGENCE, "long-cot sft ablation")

    def test_the_iteration_count_is_still_the_provisional_one(self, long_cot, ablation):
        # Guards the launch, not the arithmetic: while these agree the pack has not been measured,
        # and the config's header says it must not be launched. Replacing the value breaks the
        # assertion above, which is where the measured count gets pinned.
        assert long_cot.train.train_iters == ablation.train.train_iters


class TestTheLongCotAblationTrainsOnItsOwnCorpus:
    def test_the_corpus_is_the_long_selection_not_the_baseline_mix(self, long_cot, ablation):
        assert long_cot.dataset.dataset_name == LONG_COT_CORPUS
        assert long_cot.dataset.dataset_name != ablation.dataset.dataset_name

    def test_the_data_config_pins_the_same_corpus_and_revision_the_training_config_names(self, long_cot_data_config):
        assert long_cot_data_config.dataset == LONG_COT_CORPUS
        assert long_cot_data_config.revision == LONG_COT_REVISION

    def test_the_data_config_builds_the_pack_the_training_config_reads(self, long_cot, long_cot_data_config):
        # Tokenizer, sequence length and pad multiple must agree across the two files, because the
        # packed path encodes them: a disagreement resolves to a path that does not exist rather
        # than to a pack built under different rules.
        assert long_cot_data_config.tokenizer == long_cot.tokenizer.tokenizer_model
        assert long_cot_data_config["seq-length"] == long_cot.dataset.packed_sequence_specs.packed_sequence_size
        assert long_cot_data_config["pad-seq-to-mult"] == long_cot.dataset.packed_sequence_specs.pad_seq_to_mult

    def test_the_packed_path_is_a_shard_glob_naming_the_tokenizer_and_pad_multiple(self, long_cot):
        path = long_cot.dataset.packed_sequence_specs.packed_train_data_path
        assert "/shard*/" in path, "the pack is built per shard and read through a glob"
        assert "nemotron-think-history-tokenizer" in path
        assert "pad_seq_to_mult4" in path
        assert path.startswith(long_cot.dataset.dataset_root)

    def test_the_history_tokenizer_is_used_so_prior_turn_reasoning_survives(self, long_cot):
        # The plain think tokenizer renders every prior assistant turn as an empty <think></think>;
        # the encoders are byte-identical, so only this name distinguishes them.
        assert long_cot.tokenizer.tokenizer_model.endswith("nemotron-think-history-tokenizer")

    def test_pad_multiple_covers_context_parallelism(self, long_cot):
        assert long_cot.dataset.packed_sequence_specs.pad_seq_to_mult >= 2 * long_cot.model.context_parallel_size


class TestTheLongCotAblationSharesTheBaselineWarmStart:
    def test_it_loads_the_same_midtraining_final_as_the_parent_and_the_sibling(self, long_cot, ablation, parent):
        assert long_cot.checkpoint.pretrained_checkpoint == parent.checkpoint.pretrained_checkpoint
        assert long_cot.checkpoint.pretrained_checkpoint == ablation.checkpoint.pretrained_checkpoint

    def test_it_writes_nowhere_the_parent_writes(self, long_cot, parent):
        # Only against the PARENT: that these four differ from the sibling is already proved by
        # the set-equality pin above, which names them as the permitted divergence.
        assert long_cot.checkpoint.save != parent.checkpoint.save
        assert long_cot.checkpoint.load != parent.checkpoint.load
        assert long_cot.logger.wandb_exp_name != parent.logger.wandb_exp_name
        assert long_cot.logger.tensorboard_dir != parent.logger.tensorboard_dir

    def test_load_equals_save_so_a_resubmission_resumes(self, long_cot):
        assert long_cot.checkpoint.load == long_cot.checkpoint.save


class TestTheLongCotBatchMatchesTheSibling:
    def test_the_batch_and_topology_are_the_siblings(self, long_cot, ablation):
        assert long_cot.train.global_batch_size == ablation.train.global_batch_size == 256
        assert long_cot.model.tensor_model_parallel_size == ablation.model.tensor_model_parallel_size
        assert long_cot.model.context_parallel_size == ablation.model.context_parallel_size
        assert long_cot.model.pipeline_model_parallel_size == ablation.model.pipeline_model_parallel_size
        assert long_cot.model.expert_model_parallel_size == ablation.model.expert_model_parallel_size

    def test_two_packs_per_replica_at_256_gpus(self, long_cot):
        dp = data_parallel_size(long_cot, ABLATION_GPUS)
        assert dp == 128
        assert long_cot.train.global_batch_size % dp == 0
        assert long_cot.train.global_batch_size // dp == 2

    def test_the_schedule_is_the_siblings(self, long_cot, ablation):
        assert long_cot.optimizer.lr == ablation.optimizer.lr
        assert long_cot.scheduler.lr_decay_style == ablation.scheduler.lr_decay_style
        assert long_cot.scheduler.lr_warmup_fraction == ablation.scheduler.lr_warmup_fraction

    def test_it_carries_the_same_segment_exit_posture(self, long_cot, parent):
        assert_segment_exit_posture(long_cot, "long-cot sft ablation", parent.train.exit_duration_in_mins)
