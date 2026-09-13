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

The xl-50b SFT ablation is evidence about its two moved variables — the post-training corpus and
the batch — only to the extent that those, and what follows from them, are the only things that
moved. So the assertions come in two halves, as the smoke-run tests' do: the set of fields that
differ between the merged ablation and its merged parent must equal exactly the ablated fields
plus the run identity (where its checkpoints, W&B run and TensorBoard events go, which MUST differ
or the ablation would overwrite the parent's artifact), and the ablated fields must relate to the
parent's by the stated rule. The corpus side is pinned across three files: the training config
names the corpus, the data config pins its revision and pack geometry, and the corpora table
builds the pack the training config's glob reads.
"""

from __future__ import annotations

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
_CAMPAIGN_DIR = _REPO_ROOT / "configs" / "control_pretraining"
_ABLATIONS_DIR = _CAMPAIGN_DIR / "30b_baseline_ablations"
PARENT = _CAMPAIGN_DIR / "30b_baseline" / "nemotron_nano_30b_baseline_sft.yaml"
ABLATION = _ABLATIONS_DIR / "nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml"
ABLATION_DATA = _ABLATIONS_DIR / "data" / "pa-warm-start-sft-xl-50b-mix.yaml"
CORPORA_TABLE = _ABLATIONS_DIR / "corpora.tsv"

CORPUS = "geodesic-research/pa-warm-start-sft-xl-50b-mix"
CORPUS_REVISION = "ec0b9197aada498b0345690b8d30271335dfe7b0"
# The default config's row count at the pinned revision, from the dataset card; the table's docs
# column carries it so the verifier can check the prepared JSONL against it.
CORPUS_CONVERSATIONS = 8_924_246
TOKENS_PER_ITERATION = 8_388_608
EPOCHS = 1
# PROVISIONAL, like the config's train_iters: the packs a 50.0B-token mix makes at 32768 and the
# mainline's 99.8% packing efficiency. Both are replaced by the sum of the per-shard packs once
# they are built; until then the pin below holds the config to this figure so that the count
# cannot drift for any other reason.
PACKS_PROVISIONAL = 1_528_936

# The ablation moves the corpus and the batch; the iteration count and the checkpoint cadence
# follow from those, and the identity fields must differ so that nothing of the parent's is
# overwritten. No other field may move. Set equality, not containment: a field cannot start
# differing without being named here.
ALLOWED_DIVERGENCE = {
    "dataset.dataset_name",
    "dataset.dataset_root",
    "dataset.packed_sequence_specs.packed_train_data_path",
    "train.global_batch_size",
    "train.train_iters",
    "checkpoint.save_interval",
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


@pytest.fixture(scope="module")
def data_config():
    return OmegaConf.load(ABLATION_DATA)


@pytest.fixture(scope="module")
def corpora_rows():
    """The ablations' corpora table, parsed by the same module the build and the verifier use."""
    return corpora_table.read_corpora_table(CORPORA_TABLE)


def data_parallel_size(cfg, gpus: int) -> int:
    return gpus // (
        cfg.model.tensor_model_parallel_size * cfg.model.context_parallel_size * cfg.model.pipeline_model_parallel_size
    )


class TestOnlyTheAblatedFieldsDiffer:
    def test_exactly_the_ablated_and_identity_fields_differ(self, ablation, parent):
        assert_only_these_fields_differ(ablation, parent, ALLOWED_DIVERGENCE, "xl-50b sft ablation")

    def test_the_batch_is_half_the_parents_in_tokens(self, ablation, parent):
        assert ablation.train.global_batch_size * ablation.dataset.seq_length == TOKENS_PER_ITERATION
        assert ablation.train.global_batch_size * 2 == parent.train.global_batch_size
        assert ablation.dataset.seq_length == parent.dataset.seq_length
        assert ablation.train.micro_batch_size == parent.train.micro_batch_size

    def test_the_iteration_count_is_one_pass_over_the_pack(self, ablation):
        assert ablation.train.train_iters == 5973
        assert_iterations_are_the_minimal_cover(
            ablation.train.train_iters,
            ablation.train.global_batch_size,
            EPOCHS * PACKS_PROVISIONAL,
            "xl-50b sft ablation",
        )

    def test_warm_starts_from_the_same_midtraining_final(self, ablation, parent):
        assert ablation.checkpoint.pretrained_checkpoint == parent.checkpoint.pretrained_checkpoint
        assert ablation.checkpoint.pretrained_checkpoint.endswith("control_pretrain_30b_baseline_midtrain")

    def test_the_schedule_and_topology_are_the_parents(self, ablation, parent):
        assert ablation.optimizer.lr == parent.optimizer.lr
        assert ablation.scheduler.lr_decay_style == parent.scheduler.lr_decay_style
        assert ablation.scheduler.lr_warmup_fraction == parent.scheduler.lr_warmup_fraction
        assert ablation.model.context_parallel_size == parent.model.context_parallel_size
        assert ablation.model.expert_model_parallel_size == parent.model.expert_model_parallel_size


class TestTheCorpusIsTheRevisedMix:
    def test_the_training_config_names_the_revised_mix_not_the_parents(self, ablation, parent):
        assert ablation.dataset.dataset_name == CORPUS
        assert ablation.dataset.dataset_name != parent.dataset.dataset_name

    def test_the_data_config_pins_the_same_corpus_and_revision(self, data_config):
        assert data_config.dataset == CORPUS
        assert data_config.revision == CORPUS_REVISION
        assert data_config.split == "train"

    def test_the_data_config_builds_the_pack_the_training_config_reads(self, ablation, data_config):
        # Tokenizer, sequence length and pad multiple must agree across the two files, because the
        # packed path encodes them: a disagreement resolves to a path that does not exist rather
        # than to a pack built under different rules.
        assert data_config.tokenizer == ablation.tokenizer.tokenizer_model
        assert data_config["seq-length"] == ablation.dataset.packed_sequence_specs.packed_sequence_size
        assert data_config["pad-seq-to-mult"] == ablation.dataset.packed_sequence_specs.pad_seq_to_mult

    def test_the_data_config_prepares_jsonl_only_for_the_sharded_pack(self, data_config):
        # The pack is built per shard by the table's chain, so the prepare must stop at the JSONL.
        assert data_config["skip-pack"] is True
        assert data_config["skip-count"] is True

    def test_the_corpora_table_builds_this_corpus_from_the_default_config(self, corpora_rows, ablation):
        (row,) = corpora_rows
        assert row.subset == "default", "the mix's combined split is its default config"
        assert row.stage == "sft" and row.kind == "pack"
        assert row.config.resolve() == ABLATION_DATA.resolve()
        # The shard count is a host-memory budget for the pack job, not a walltime one: the
        # packer holds a shard's whole pack set in RAM before writing it, and at 16 shards this
        # corpus's ~95,600-pack shards were OOM-killed on a 449 GB node.
        assert row.shards == 32 and row.shard_mode == "split"
        assert row.docs == CORPUS_CONVERSATIONS
        assert str(corpora_table.corpus_root(CORPUS, row.subset)) == ablation.dataset.dataset_root

    def test_the_packed_path_is_a_shard_glob_naming_the_tokenizer_and_pad_multiple(self, ablation):
        path = ablation.dataset.packed_sequence_specs.packed_train_data_path
        assert "/shard*/" in path, "the pack is built per shard and read through a glob"
        # The glob is what makes the shard count a data-build decision rather than a config one.
        assert "nemotron-think-history-tokenizer" in path
        assert "pad_seq_to_mult4" in path
        assert path.startswith(ablation.dataset.dataset_root)

    def test_the_history_tokenizer_is_used_so_prior_turn_reasoning_survives(self, ablation):
        # The plain think tokenizer renders every prior assistant turn as an empty <think></think>;
        # the encoders are byte-identical, so only this name distinguishes them.
        assert ablation.tokenizer.tokenizer_model.endswith("nemotron-think-history-tokenizer")

    def test_pad_multiple_covers_context_parallelism(self, ablation):
        assert ablation.dataset.packed_sequence_specs.pad_seq_to_mult >= 2 * ablation.model.context_parallel_size


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

    def test_the_raw_yaml_copies_no_output_path_from_the_parent(self):
        """The merged comparison above would miss a field the recipe fills identically; the raw
        files are what a reader copies, so the identity fields are checked there too."""
        raw_ablation, raw_parent = OmegaConf.load(ABLATION), OmegaConf.load(PARENT)
        for section, key in (("checkpoint", "load"), ("checkpoint", "save"), ("logger", "tensorboard_dir")):
            assert raw_ablation[section][key] != raw_parent[section][key], f"{section}.{key}"

    def test_a_resubmission_resumes(self, ablation):
        assert ablation.checkpoint.load == ablation.checkpoint.save
        assert ablation.checkpoint.save_interval < ablation.train.train_iters

    def test_the_checkpoint_cadence_is_the_parents_in_tokens(self, ablation, parent):
        """The cadence is a token count stated in iterations, so at half the batch it is
        restated as twice the parent's interval; a verbatim copy would halve the spacing."""
        assert (
            ablation.checkpoint.save_interval * ablation.train.global_batch_size
            == parent.checkpoint.save_interval * parent.train.global_batch_size
        )


class TestTheAllocation:
    def test_two_packs_per_replica_at_256_gpus_the_parents_per_gpu_load(self, ablation, parent):
        parent_dp = data_parallel_size(parent, PARENT_GPUS)
        ablation_dp = data_parallel_size(ablation, ABLATION_GPUS)
        assert ablation_dp == 128
        assert ablation.train.global_batch_size % ablation_dp == 0
        assert ablation.train.global_batch_size // ablation_dp == parent.train.global_batch_size // parent_dp == 2


class TestSegmentRollover:
    def test_ends_on_the_duration_clock_like_its_parent(self, ablation, parent):
        assert_segment_exit_posture(ablation, "xl-50b sft ablation", parent.train.exit_duration_in_mins)
