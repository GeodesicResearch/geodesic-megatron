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

The xl-50b rerun ("v2") is pinned to the ablation the same way: it differs only in the levers of the SFT quickstart's
fastest configuration, at the quickstart's values, and in its run identity, so it is the ablation's training run on
faster, bug-fixed code and nothing else.

The quality-filtered run ("v3") is pinned to v2: it differs only in the corpus's three fields and its run identity, its
launcher settings are v2's, and its data config builds its pack exactly as the xl-50b mix's was built, so v3 against
v2 is a difference of training data alone.

The higher-learning-rate run ("v4") and its fallback are each pinned to v3: each differs only in the peak learning rate
and its run identity, with v3's launcher settings, so v4 against v3 is a difference of peak learning rate alone.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from scripts.training.config_compose import BASE_CONFIG_KEY
from scripts.training.launcher_source import env_override_entries

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from tests.unit_tests.campaign_config import (
    assert_data_config_is_the_mixs_but_for_its_source,
    assert_hold_and_pin_move_together,
    assert_iterations_are_the_minimal_cover,
    assert_only_these_fields_differ,
    assert_reads_the_split,
    assert_row_packs_like_the_mix,
    assert_segment_exit_posture,
    data_parallel_size,
    dotted_leaves,
    dry_run_build,
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
# The measured corpus: the sum of the 32 shards' packed rows, read from the parquet footers
# by verify_corpora.py. train_iters is derived from it, and the pin below holds the config to it.
PACKS = 1_529_684

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
}
# tensorboard_dir is deliberately absent: every campaign config nulls it, so it is
# identical across arms by design and can no longer distinguish one run from another.
IDENTITY_FIELDS = ("checkpoint.load", "checkpoint.save", "logger.wandb_exp_name")
PARENT_GPUS = 512
ABLATION_GPUS = 256

# The rerun's levers are, by definition, the fields the fastest SFT configuration's overlay states, less the overlay's
# own base and W&B name.
V2 = _ABLATIONS_DIR / "nemotron_nano_30b_baseline_sft_xl50b_gbs256_v2.yaml"
FAST_SFT_QUICKSTART = _REPO_ROOT / "configs" / "quickstart" / "nemotron_nano_quickstart_sft.yaml"
QUICKSTART_OWN_FIELDS = {BASE_CONFIG_KEY, "logger.wandb_exp_name"}
V2_GPUS = 256

# v3 is v2 with the corpus replaced by the quality-filtered mix's 50B training split; the corpus is named by these
# three fields and by nothing else in a training config.
V3 = _ABLATIONS_DIR / "nemotron_nano_30b_baseline_sft_xl50b_gbs256_v3.yaml"
V3_DATA = _ABLATIONS_DIR / "data" / "pa-warm-start-sft-xl-50b-mix-quality-filtered.yaml"
V3_CORPUS = "geodesic-research/pa-warm-start-sft-xl-50b-mix-quality-filtered"
V3_SUBSET = "train"
CORPUS_FIELDS = {
    "dataset.dataset_name",
    "dataset.dataset_root",
    "dataset.packed_sequence_specs.packed_train_data_path",
}

# v4 is v3 at a higher peak learning rate (Kyle, 2026-10-10), and its fallback the same at a lower one, launched from
# scratch only if v4 diverges; each is pinned to v3 by the peak learning rate alone.
V4 = _ABLATIONS_DIR / "nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4.yaml"
V4_FALLBACK = _ABLATIONS_DIR / "nemotron_nano_30b_baseline_sft_xl50b_gbs256_v4lr35.yaml"
V4_PEAK_LR = {V4: 5.0e-05, V4_FALLBACK: 3.5e-05}
LR_FIELDS = {"optimizer.lr"}

# Each variant and the run it is pinned to: the ablation to the parent stage, the rerun to the ablation, v3 to v2, and
# v4 and its fallback to v3.
VARIANTS = {
    "xl-50b sft ablation": (ABLATION, PARENT),
    "xl-50b sft v2": (V2, ABLATION),
    "xl-50b sft v3": (V3, V2),
    "xl-50b sft v4": (V4, V3),
    "xl-50b sft v4 fallback": (V4_FALLBACK, V3),
}


def fast_configuration_levers() -> set[str]:
    """The dotted fields the fastest SFT quickstart sets on top of its benchmark."""
    overlay = OmegaConf.to_container(OmegaConf.load(FAST_SFT_QUICKSTART))
    return set(dotted_leaves(overlay)) - QUICKSTART_OWN_FIELDS


@pytest.fixture(scope="module")
def ablation():
    return merge_onto_recipe(ABLATION, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module")
def parent():
    return merge_onto_recipe(PARENT, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module")
def v2():
    return merge_onto_recipe(V2, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module")
def v3():
    return merge_onto_recipe(V3, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module")
def fast_quickstart():
    return merge_onto_recipe(FAST_SFT_QUICKSTART, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def variant(request):
    """A variant with the run it is pinned to, merged onto the recipe as the launcher merges them."""
    path, reference_path = VARIANTS[request.param]
    return SimpleNamespace(
        label=request.param,
        path=path,
        reference_path=reference_path,
        cfg=merge_onto_recipe(path, nemotron_3_nano_sft_config),
        reference=merge_onto_recipe(reference_path, nemotron_3_nano_sft_config),
    )


@pytest.fixture(scope="module")
def data_config():
    return OmegaConf.load(ABLATION_DATA)


@pytest.fixture(scope="module")
def corpora_rows():
    """The ablations' corpora table, parsed by the same module the build and the verifier use."""
    return corpora_table.read_corpora_table(CORPORA_TABLE)


class TestOnlyTheAblatedFieldsDiffer:
    def test_exactly_the_ablated_and_identity_fields_differ(self, ablation, parent):
        assert_only_these_fields_differ(ablation, parent, ALLOWED_DIVERGENCE, "xl-50b sft ablation")

    def test_the_batch_is_half_the_parents_in_tokens(self, ablation, parent):
        assert ablation.train.global_batch_size * ablation.dataset.seq_length == TOKENS_PER_ITERATION
        assert ablation.train.global_batch_size * 2 == parent.train.global_batch_size
        assert ablation.dataset.seq_length == parent.dataset.seq_length
        assert ablation.train.micro_batch_size == parent.train.micro_batch_size

    def test_the_iteration_count_is_one_pass_over_the_pack(self, ablation):
        assert ablation.train.train_iters == 5976
        assert_iterations_are_the_minimal_cover(
            ablation.train.train_iters,
            ablation.train.global_batch_size,
            EPOCHS * PACKS,
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
        # The table also builds the filtered arms' cuts of this mix; the ablation's own row is the
        # one naming its data config.
        (row,) = [r for r in corpora_rows if r.config.resolve() == ABLATION_DATA.resolve()]
        assert row.subset == "default", "the mix's combined split is its default config"
        assert row.stage == "sft" and row.kind == "pack"
        assert row.config.resolve() == ABLATION_DATA.resolve()
        # The shard count is a host-memory budget for the pack job, not a walltime one: the
        # packer holds a shard's whole pack set in RAM before writing it, and at 16 shards this
        # corpus's ~95,600-pack shards were OOM-killed on a 449 GB node.
        assert row.shards == 32 and row.shard_mode == "split"
        assert row.docs == CORPUS_CONVERSATIONS
        assert str(corpora_table.corpus_root(CORPUS, row.subset)) == ablation.dataset.dataset_root

    def test_build_shards_submits_only_the_named_packs(self):
        """`BUILD_SHARDS=0,5` reaches the plan through the real script: the pack jobs of shards 0
        and 5 and nothing else — no prepare, no split — which is how a 32-shard pack is fed to the
        queue a few shards at a time, or one OOM-killed shard is re-run, without the other thirty."""
        proc = dry_run_build(CORPORA_TABLE, "sft", "default", env={"BUILD_SHARDS": "0,5"})
        assert proc.returncode == 0, f"build_corpora.sh failed:\n{proc.stdout}\n{proc.stderr}"
        output = proc.stdout + proc.stderr
        assert re.findall(r"\[dry-run\] (\w+) default", output) == ["pack", "pack"]
        assert re.findall(r"\[dry-run\] pack default shard(\d+):", output) == ["0", "5"]
        assert "SUBMITTED 2 jobs for stage 'sft' (shards: 0,5)" in output

    def test_the_packed_path_is_a_shard_glob_naming_the_tokenizer_and_pad_multiple(self, variant):
        path = variant.cfg.dataset.packed_sequence_specs.packed_train_data_path
        assert "/shard*/" in path, "the pack is built per shard and read through a glob"
        # The glob is what makes the shard count a data-build decision rather than a config one.
        assert "nemotron-think-history-tokenizer" in path
        assert "pad_seq_to_mult4" in path
        assert path.startswith(variant.cfg.dataset.dataset_root + "/")

    def test_the_history_tokenizer_is_used_so_prior_turn_reasoning_survives(self, ablation):
        # The plain think tokenizer renders every prior assistant turn as an empty <think></think>;
        # the encoders are byte-identical, so only this name distinguishes them.
        assert ablation.tokenizer.tokenizer_model.endswith("nemotron-think-history-tokenizer")

    def test_pad_multiple_covers_context_parallelism(self, variant):
        assert variant.cfg.dataset.packed_sequence_specs.pad_seq_to_mult >= 2 * variant.cfg.model.context_parallel_size


class TestTheRunIdentityIsItsOwn:
    def test_no_output_location_is_the_references(self, variant):
        flat_variant, flat_reference = flatten_merged_config(variant.cfg), flatten_merged_config(variant.reference)
        for field in IDENTITY_FIELDS:
            assert flat_variant[field] != flat_reference[field], field

    def test_the_save_directory_is_not_inside_the_references(self, variant):
        """A save under the reference's tree would pass the inequality above and still collide."""
        reference_save = Path(variant.reference.checkpoint.save)
        assert not Path(variant.cfg.checkpoint.save).is_relative_to(reference_save)
        assert not reference_save.is_relative_to(Path(variant.cfg.checkpoint.save))

    def test_the_raw_yaml_copies_no_output_path_from_the_reference(self, variant):
        """The merged comparison above would miss a field the recipe fills identically; the raw
        files are what a reader copies, so the identity fields are checked there too."""
        raw_variant, raw_reference = OmegaConf.load(variant.path), OmegaConf.load(variant.reference_path)
        for section, key in (("checkpoint", "load"), ("checkpoint", "save")):
            assert raw_variant[section][key] != raw_reference[section][key], f"{section}.{key}"

    def test_a_resubmission_resumes(self, variant):
        assert variant.cfg.checkpoint.load == variant.cfg.checkpoint.save
        assert variant.cfg.checkpoint.save_interval < variant.cfg.train.train_iters

    def test_the_checkpoint_cadence_is_the_references_in_tokens(self, variant):
        """The cadence is a token count stated in iterations, so at half the batch it is
        restated as twice the parent's interval; a verbatim copy would halve the spacing."""
        assert (
            variant.cfg.checkpoint.save_interval * variant.cfg.train.global_batch_size
            == variant.reference.checkpoint.save_interval * variant.reference.train.global_batch_size
        )

    def test_the_run_is_the_references_length_in_tokens_so_the_series_align(self, variant):
        """Equal cadence only puts the saves at the same token positions while the runs are the
        same length in tokens. For the ablation that holds because one epoch of this larger mix costs
        what the parent's two epochs of the smaller one cost, which is a measured coincidence rather
        than a constraint — so a re-measured train_iters could break the alignment while every other
        assertion here still passed."""
        assert (
            variant.cfg.train.train_iters * variant.cfg.train.global_batch_size
            == variant.reference.train.train_iters * variant.reference.train.global_batch_size
        )


class TestTheAllocation:
    def test_two_packs_per_replica_at_256_gpus_the_parents_per_gpu_load(self, ablation, parent):
        parent_dp = data_parallel_size(parent, PARENT_GPUS)
        ablation_dp = data_parallel_size(ablation, ABLATION_GPUS)
        assert ablation_dp == 128
        assert ablation.train.global_batch_size % ablation_dp == 0
        assert ablation.train.global_batch_size // ablation_dp == parent.train.global_batch_size // parent_dp == 2


class TestSegmentRollover:
    def test_ends_on_the_duration_clock_like_its_reference(self, variant):
        assert_segment_exit_posture(variant.cfg, variant.label, variant.reference.train.exit_duration_in_mins)


class TestTheRerunIsTheAblationOnTheFastConfiguration:
    def test_exactly_the_levers_and_identity_fields_differ(self, v2, ablation):
        assert_only_these_fields_differ(
            v2, ablation, fast_configuration_levers() | set(IDENTITY_FIELDS), "xl-50b sft v2"
        )

    def test_every_lever_has_the_quickstarts_value(self, v2, fast_quickstart):
        flat_v2, flat_quickstart = flatten_merged_config(v2), flatten_merged_config(fast_quickstart)
        for field in sorted(fast_configuration_levers()):
            assert flat_v2[field] == flat_quickstart[field], field

    def test_its_env_file_holds_the_quickstarts_launcher_settings(self):
        assert env_override_entries(str(V2.with_suffix(".env"))) == env_override_entries(
            str(FAST_SFT_QUICKSTART.with_suffix(".env"))
        )

    def test_one_pack_per_replica_keeps_the_ablations_tokens_per_gpu(self, v2, ablation):
        v2_dp = data_parallel_size(v2, V2_GPUS)
        assert v2_dp == 256
        assert v2.train.global_batch_size // v2_dp == 1
        assert v2.train.global_batch_size * v2.dataset.seq_length // V2_GPUS == (
            ablation.train.global_batch_size * ablation.dataset.seq_length // ABLATION_GPUS
        )


@pytest.fixture(scope="module")
def v3_row(corpora_rows):
    """v3's row of the ablations' corpora table: the one naming its data config."""
    (row,) = [r for r in corpora_rows if r.config.resolve() == V3_DATA.resolve()]
    return row


class TestV3IsV2OnTheQualityFilteredCorpus:
    def test_exactly_the_corpus_and_identity_fields_differ_from_v2(self, v3, v2):
        assert_only_these_fields_differ(v3, v2, CORPUS_FIELDS | set(IDENTITY_FIELDS), "xl-50b sft v3")

    def test_it_trains_v2s_iterations_from_v2s_warm_start(self, v3, v2):
        """The run trains v2's token budget on a smaller corpus, so the iteration count is v2's and never re-derived
        from this corpus's pack count."""
        assert v3.train.train_iters == v2.train.train_iters == 5976
        assert v3.checkpoint.pretrained_checkpoint == v2.checkpoint.pretrained_checkpoint

    def test_its_env_file_is_v2s(self):
        assert env_override_entries(str(V3.with_suffix(".env"))) == env_override_entries(str(V2.with_suffix(".env")))

    def test_the_training_config_reads_the_quality_filtered_split(self, v3, v3_row):
        assert v3_row.subset == V3_SUBSET
        assert OmegaConf.load(V3_DATA).dataset == V3_CORPUS
        assert_reads_the_split(v3, V3_CORPUS, V3_SUBSET)

    def test_the_data_config_builds_the_pack_as_the_mix_was_built(self):
        """Every key but the corpus's name and revision is the xl-50b mix's, so the pack is built exactly as the pack
        v2 read: the same tokenizer, sequence length, pad multiple and JSONL-only prepare."""
        assert_data_config_is_the_mixs_but_for_its_source(V3_DATA, ABLATION_DATA)

    def test_the_corpora_row_builds_the_pack_in_the_mixs_shards(self, v3_row, corpora_rows):
        (mix_row,) = [r for r in corpora_rows if r.config.resolve() == ABLATION_DATA.resolve()]
        assert_row_packs_like_the_mix(v3_row, V3_DATA, mix_row)

    def test_the_revision_and_the_document_count_are_pinned_together(self, v3_row):
        """Both come from the published split: a full commit SHA, which no later push can move, and its row count.
        Until the split is published both read PENDING, which holds the build."""
        assert_hold_and_pin_move_together(OmegaConf.load(V3_DATA).revision, [v3_row], "xl-50b sft v3")

    def test_the_document_count_is_the_copied_splits(self, v3_row):
        """`train` at the pinned revision is a copy, file for file, of the `xl50b_train_quality_v5` config published
        at e77572f6, so its row count is that split's."""
        assert v3_row.docs == 9_261_591


@pytest.fixture(scope="module", params=sorted(V4_PEAK_LR, key=str), ids=lambda p: p.stem)
def v4_run(request):
    """v4 or its fallback, merged onto the recipe as the launcher merges it."""
    return SimpleNamespace(path=request.param, cfg=merge_onto_recipe(request.param, nemotron_3_nano_sft_config))


class TestV4IsV3AtAHigherPeakLearningRate:
    def test_exactly_the_peak_learning_rate_and_identity_fields_differ_from_v3(self, v4_run, v3):
        assert_only_these_fields_differ(v4_run.cfg, v3, LR_FIELDS | set(IDENTITY_FIELDS), v4_run.path.stem)

    def test_the_peak_learning_rate_is_the_approved_one(self, v4_run, v3):
        """5e-5 is ten times v3's peak and 3.5e-5 seven times; the schedule's shape, warmup and floor are v3's."""
        assert v4_run.cfg.optimizer.lr == V4_PEAK_LR[v4_run.path]
        assert v4_run.cfg.optimizer.lr > v3.optimizer.lr

    def test_its_env_file_is_v3s(self, v4_run):
        assert env_override_entries(str(v4_run.path.with_suffix(".env"))) == env_override_entries(
            str(V3.with_suffix(".env"))
        )

    def test_v4_and_its_fallback_never_share_a_save_directory(self):
        """The fallback is a fresh run: resuming v4's save under another learning rate would mix two schedules."""
        v4_save = Path(OmegaConf.load(V4).checkpoint.save)
        fallback_save = Path(OmegaConf.load(V4_FALLBACK).checkpoint.save)
        assert not v4_save.is_relative_to(fallback_save)
        assert not fallback_save.is_relative_to(v4_save)
