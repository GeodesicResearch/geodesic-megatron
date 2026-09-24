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

"""Each filtered arm's reasoning model differs from the baseline's in its corpus and warm start only.

Two filtered arms get a reasoning model by the xl-50b recipe: the Broadly Filtered arm (from its
midtraining final, on the broad cut of the xl-50b mix) and the narrowly filtered V2 arm (from its
midtraining final, on the narrow cut — the baseline mix minus exactly 668 conversations). Each is
evidence about filtering only if it is the baseline xl-50b SFT with those two things changed, so the
central assertion merges each config and the baseline's through the real launcher path and requires
the set of differing fields to be EXACTLY the corpus, the warm start and the run identity.

In particular the token budget is NOT among them (Kyle, 2026-09-23): the same 5976 iterations at the
same batch and sequence length as the baseline, so the filtered models see the same tokens drawn
from a smaller pool, slightly more than one epoch, and their checkpoints sit at the baseline's token
positions. The rest pins what the diff cannot see: that the warm start is that arm's own midtraining
save, that the training config, the data config and the corpora table describe one corpus, that the
table's hold and the data config's pin move together, and that the Hub and archive manifests know
the model and count its tokens from its base model's curriculum. Every test runs once per model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_pretrain_config,
    nemotron_3_nano_sft_config,
)
from tests.unit_tests.campaign_config import (
    assert_hold_and_pin_move_together,
    assert_only_these_fields_differ,
    assert_segment_exit_posture,
    merge_onto_recipe,
)
from tests.unit_tests.corpora_fixtures import corpora_table, importable


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIR = _REPO_ROOT / "configs" / "control_pretraining"
_ABLATIONS_DIR = _CAMPAIGN_DIR / "30b_baseline_ablations"
BASELINE_SFT = _ABLATIONS_DIR / "nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml"
BASELINE_SFT_DATA = _ABLATIONS_DIR / "data" / "pa-warm-start-sft-xl-50b-mix.yaml"
CORPORA_TABLE = _ABLATIONS_DIR / "corpora.tsv"
HUB_MANIFEST = _CAMPAIGN_DIR / "hub_models.yaml"
BUCKET_MANIFEST = _CAMPAIGN_DIR / "bucket_sync.yaml"
BROAD_PRETRAIN = _CAMPAIGN_DIR / "30b_filtered_mini_2plus" / "nemotron_nano_30b_filtered_mini_2plus_pretrain.yaml"
BROAD_MIDTRAIN = _CAMPAIGN_DIR / "30b_filtered_mini_2plus" / "nemotron_nano_30b_filtered_mini_2plus_midtrain.yaml"
NARROW_V2_MIDTRAIN = (
    _CAMPAIGN_DIR / "30b_filtered_gpt55_4plus_v2" / "nemotron_nano_30b_filtered_gpt55_4plus_v2_midtrain.yaml"
)

FILTERED_DATASET = "geodesic-research/control-pretraining-datasets"
# Pretraining (29,881 iterations) and midtraining (3,126) at 16,777,216 tokens each, shared by both
# filtered models' base curricula; the SFT then runs at half that batch.
TOKENS_BEFORE_SFT = (29881 + 3126) * 16_777_216
SFT_TOKENS_PER_ITERATION = 8_388_608

# Exactly the fields a filtered reasoning model may differ in from the baseline xl-50b SFT: its
# corpus, its warm start, and where its checkpoints and W&B run go. Set equality, so a field cannot
# start differing without being named — train_iters, global_batch_size and seq_length included.
ALLOWED_DIVERGENCE = {
    "dataset.dataset_name",
    "dataset.dataset_root",
    "dataset.packed_sequence_specs.packed_train_data_path",
    "checkpoint.pretrained_checkpoint",
    "checkpoint.load",
    "checkpoint.save",
    "logger.wandb_exp_name",
}


@dataclass(frozen=True)
class FilteredSft:
    """One filtered arm's reasoning model: its configs, the split it trains on, the midtraining
    config whose final checkpoint it starts from, and the Hub repository it publishes to."""

    label: str
    config: Path
    data_config: Path
    subset: str
    warm_start: Path
    hub_repo: str
    history: tuple[Path, ...]


FILTERED_SFTS = (
    FilteredSft(
        "broad sft xl50b",
        _ABLATIONS_DIR / "nemotron_nano_30b_filtered_mini_2plus_sft_xl50b_gbs256.yaml",
        _ABLATIONS_DIR / "data" / "pa-warm-start-sft-xl50b-filtered-mini-2plus.yaml",
        "pa_warm_start_sft_xl50b_filtered_mini_2plus",
        BROAD_MIDTRAIN,
        "geodesic-research/control-pretraining-30b-filtered-mini-2plus-xl50b-think",
        (BROAD_PRETRAIN, BROAD_MIDTRAIN),
    ),
    FilteredSft(
        "narrow v2 sft xl50b",
        _ABLATIONS_DIR / "nemotron_nano_30b_filtered_gpt55_4plus_v2_sft_xl50b_gbs256.yaml",
        _ABLATIONS_DIR / "data" / "pa-warm-start-sft-xl50b-filtered-gpt55-4plus-v2.yaml",
        "pa_warm_start_sft_xl50b_filtered_gpt55_4plus_v2",
        NARROW_V2_MIDTRAIN,
        "geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-v2-xl50b-think",
        (BROAD_PRETRAIN, NARROW_V2_MIDTRAIN),
    ),
)


@pytest.fixture(scope="module", params=FILTERED_SFTS, ids=lambda sft: sft.subset)
def sft(request) -> FilteredSft:
    return request.param


@pytest.fixture(scope="module")
def merged(sft):
    return merge_onto_recipe(sft.config, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module")
def baseline():
    return merge_onto_recipe(BASELINE_SFT, nemotron_3_nano_sft_config)


@pytest.fixture(scope="module")
def data_config(sft):
    return OmegaConf.load(sft.data_config)


@pytest.fixture(scope="module")
def row(sft):
    (found,) = [r for r in corpora_table.read_corpora_table(CORPORA_TABLE) if r.subset == sft.subset]
    return found


class TestOnlyTheCorpusAndWarmStartDiffer:
    def test_exactly_the_corpus_warm_start_and_identity_fields_differ(self, sft, merged, baseline):
        assert_only_these_fields_differ(merged, baseline, ALLOWED_DIVERGENCE, sft.label)

    def test_the_token_budget_is_the_baselines(self, merged, baseline):
        """Stated separately from the set above so a failure names Kyle's rule: the same
        iterations at the same batch and sequence, never one epoch over the filtered pool."""
        assert merged.train.train_iters == baseline.train.train_iters == 5976
        assert merged.train.global_batch_size == baseline.train.global_batch_size
        assert merged.dataset.seq_length == baseline.dataset.seq_length
        assert merged.checkpoint.save_interval == baseline.checkpoint.save_interval

    def test_the_warm_start_is_the_arms_midtraining_final(self, sft, merged, baseline):
        warm_start = merge_onto_recipe(sft.warm_start, nemotron_3_nano_pretrain_config)
        assert merged.checkpoint.pretrained_checkpoint == warm_start.checkpoint.save
        assert merged.checkpoint.pretrained_checkpoint != baseline.checkpoint.pretrained_checkpoint

    def test_a_resubmission_resumes_into_its_own_directory(self, merged, baseline):
        assert merged.checkpoint.load == merged.checkpoint.save
        assert merged.checkpoint.save != baseline.checkpoint.save

    def test_segment_rollover(self, sft, merged, baseline):
        assert_segment_exit_posture(merged, sft.label, baseline.train.exit_duration_in_mins)


class TestOneCorpusAcrossThreeFiles:
    """The training config reads a path, the data config builds a corpus, and the table plans the
    build; nothing reconciles them at runtime."""

    def test_the_training_config_names_the_filtered_split(self, sft, merged, data_config):
        assert merged.dataset.dataset_name == data_config.dataset == FILTERED_DATASET
        assert merged.dataset.dataset_root == str(corpora_table.corpus_root(FILTERED_DATASET, sft.subset))

    def test_the_packed_path_is_the_baselines_under_this_corpus_root(self, merged, baseline):
        """Same shard glob, tokenizer and pad multiple as the baseline's pack, under this root."""
        own = merged.dataset.packed_sequence_specs.packed_train_data_path
        theirs = baseline.dataset.packed_sequence_specs.packed_train_data_path
        assert own.startswith(merged.dataset.dataset_root + "/")
        assert own.removeprefix(merged.dataset.dataset_root) == theirs.removeprefix(baseline.dataset.dataset_root)

    def test_the_data_config_builds_the_pack_the_training_config_reads(self, merged, data_config):
        assert data_config.tokenizer == merged.tokenizer.tokenizer_model
        assert data_config["seq-length"] == merged.dataset.packed_sequence_specs.packed_sequence_size
        assert data_config["pad-seq-to-mult"] == merged.dataset.packed_sequence_specs.pad_seq_to_mult

    def test_the_data_config_is_the_baseline_mixs_but_for_its_source(self, sft):
        """Tokenizer, split and geometry are the baseline mix's; only the dataset and revision move."""
        baseline_data = yaml.safe_load(BASELINE_SFT_DATA.read_text())
        mine = yaml.safe_load(sft.data_config.read_text())
        differing = {k for k in set(mine) | set(baseline_data) if mine.get(k) != baseline_data.get(k)}
        assert differing == {"dataset", "revision"}

    def test_the_table_row_packs_this_corpus_in_32_shards(self, sft, row):
        assert row.stage == "sft" and row.kind == "pack"
        assert row.config.resolve() == sft.data_config.resolve()
        assert row.shards == 32 and row.shard_mode == "split"

    def test_the_hold_and_the_pin_move_together(self, sft, data_config, row):
        assert_hold_and_pin_move_together(data_config.revision, [row], sft.label)


class TestTheManifestsKnowTheModel:
    def test_the_hub_manifest_publishes_it_privately_with_its_base_curriculum_as_history(self, sft):
        manifest = yaml.safe_load(HUB_MANIFEST.read_text())
        (model,) = [m for m in manifest["models"] if m["repo"] == sft.hub_repo]
        assert model["private"] is True and model["reasoning"] is True and model["strict"] is False
        assert model["history"] == [str(p.relative_to(_REPO_ROOT)) for p in sft.history]
        (stage,) = model["stages"]
        assert stage["name"] == "sft" and stage["default"] is True
        assert stage["config"] == str(sft.config.relative_to(_REPO_ROOT))
        assert stage["revision"] == "sft_iter_{iteration}"

    def test_the_card_counts_tokens_from_the_base_curriculum_at_the_sfts_own_batch(self, sft):
        importable(_REPO_ROOT / "scripts" / "hub")
        import publish_models

        manifest = publish_models.load_manifest(HUB_MANIFEST, _REPO_ROOT)
        (model,) = [m for m in manifest.models if m.repo == sft.hub_repo]
        (stage,) = model.stages
        assert stage.tokens_before == TOKENS_BEFORE_SFT
        assert stage.tokens_per_iteration == SFT_TOKENS_PER_ITERATION

    def test_the_bucket_manifest_archives_the_stage(self, sft):
        manifest = yaml.safe_load(BUCKET_MANIFEST.read_text())
        assert str(sft.config.relative_to(_REPO_ROOT)) in manifest["stage_configs"]
