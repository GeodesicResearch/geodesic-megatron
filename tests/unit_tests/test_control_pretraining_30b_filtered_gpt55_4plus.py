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

"""Each narrowly filtered arm differs from the Broadly Filtered arm in one stage's data, and in
nothing else.

There are two such arms, V1 (`30b_filtered_gpt55_4plus`, deprecated but kept in the figures) and
V2 (`30b_filtered_gpt55_4plus_v2`): the same rule, canary OR `judge_score >= 4`, at two annotation
revisions, V2's with every escalated document judged. Each is a single midtraining stage
warm-started from the Broadly Filtered arm's pretraining final, on corpora cut by that narrower
rule, so the comparison each supports is against THAT arm's midtraining: same warm start, same
topology, same schedule, same steps at the same batch, differing only in which documents exist
during the anneal. Nothing at runtime enforces any of it, so the central test merges each midtrain
config and the Broadly Filtered one through the real launcher path and asserts that the set of
differing fields is EXACTLY the data paths and the run identity — and that the warm start is
equal, because a warm start that silently became the arm's own (nonexistent) pretraining, or the
baseline's, would void the comparison while training perfectly well.

The rest covers what that diff cannot see: that the blend is well-formed and names only the
arm's own filtered corpora at the paths the build produces, that the corpora table and the blend
describe the same ten corpora and tag every one of them `midtraining` (the arm has no pretraining
stage, and a copied `pretraining` tag on `ai_safety_and_adjacent` would make the build plan nine of
ten), that the table's hold and the prepare config's pin move together, and that the Hub and
archive manifests know the arm and its lineage. Every test runs once per arm.
"""

from __future__ import annotations

import re
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
    assert_blend_is_well_formed,
    assert_hold_and_pin_move_together,
    assert_only_these_fields_differ,
    assert_prefix_roots_use_the_real_slugify,
    assert_segment_exit_posture,
    blend_subsets,
    campaign_training_configs,
    corpus_weights,
    dry_run_build,
    merge_onto_recipe,
    pending_subsets,
)
from tests.unit_tests.corpora_fixtures import corpora_table, importable


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_DIR = _REPO_ROOT / "configs" / "control_pretraining"
_BROAD_DIR = _CAMPAIGN_DIR / "30b_filtered_mini_2plus"
_BASELINE_DIR = _CAMPAIGN_DIR / "30b_baseline"

BROAD_MIDTRAIN_CONFIG = _BROAD_DIR / "nemotron_nano_30b_filtered_mini_2plus_midtrain.yaml"
BROAD_PRETRAIN_CONFIG = _BROAD_DIR / "nemotron_nano_30b_filtered_mini_2plus_pretrain.yaml"
BASELINE_MIDTRAIN_CONFIG = _BASELINE_DIR / "nemotron_nano_30b_baseline_midtrain.yaml"
HUB_MANIFEST = _CAMPAIGN_DIR / "hub_models.yaml"
BUCKET_MANIFEST = _CAMPAIGN_DIR / "bucket_sync.yaml"

STAGE = "midtraining"
# The midtraining blend: ten corpora, whatever the arm. `ai_safety_and_adjacent` is among them
# in every arm; the two three-stage arms merely tag it `pretraining` in their tables.
MIDTRAINING_CORPORA = 10

# Exactly the fields a narrow arm's midtrain may differ in from the Broadly Filtered arm's. Data:
# which documents exist. Identity: where the arm's checkpoints and W&B run go, which MUST differ.
# `checkpoint.pretrained_checkpoint` is deliberately NOT here: the warm start is the same
# checkpoint, and a divergence there is the failure this module exists to catch. Nor are
# `train.train_iters`, `train.global_batch_size` or `dataset.seq_length`: every model sees the
# unfiltered baseline's token budget (Kyle, 2026-09-23).
ALLOWED_DIVERGENCE = {
    "dataset.data_path",
    "checkpoint.load",
    "checkpoint.save",
    "logger.wandb_exp_name",
}


@dataclass(frozen=True)
class NarrowArm:
    """What distinguishes one narrowly filtered arm from another: where it lives, the split suffix
    dataset-builder publishes its retained corpora under, and the Hub repository it publishes to."""

    label: str
    directory: Path
    suffix: str
    hub_repo: str

    @property
    def midtrain_config(self) -> Path:
        return self.directory / f"nemotron_nano_30b{self.suffix}_midtrain.yaml"

    @property
    def corpus_config(self) -> Path:
        return self.directory / "data" / f"control-pretraining-datasets{self.suffix.replace('_', '-')}.yaml"

    @property
    def corpora_table(self) -> Path:
        return self.directory / "corpora.tsv"


NARROW_ARMS = (
    NarrowArm(
        "narrow v1 midtrain",
        _CAMPAIGN_DIR / "30b_filtered_gpt55_4plus",
        "_filtered_gpt55_4plus",
        "geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-base",
    ),
    NarrowArm(
        "narrow v2 midtrain",
        _CAMPAIGN_DIR / "30b_filtered_gpt55_4plus_v2",
        "_filtered_gpt55_4plus_v2",
        "geodesic-research/control-pretraining-30b-filtered-gpt55-4plus-v2-base",
    ),
)

# Every stage of every arm: each must be among the configs the checkpoint-directory guard discovers.
ALL_STAGE_CONFIGS = {
    "baseline pretrain": _BASELINE_DIR / "nemotron_nano_30b_baseline_pretrain.yaml",
    "baseline midtrain": BASELINE_MIDTRAIN_CONFIG,
    "baseline sft": _BASELINE_DIR / "nemotron_nano_30b_baseline_sft.yaml",
    "broad pretrain": BROAD_PRETRAIN_CONFIG,
    "broad midtrain": BROAD_MIDTRAIN_CONFIG,
    "broad sft": _BROAD_DIR / "nemotron_nano_30b_filtered_mini_2plus_sft.yaml",
    **{arm.label: arm.midtrain_config for arm in NARROW_ARMS},
}


@pytest.fixture(scope="module", params=NARROW_ARMS, ids=lambda arm: arm.directory.name)
def arm(request) -> NarrowArm:
    return request.param


@pytest.fixture(scope="module")
def merged(arm):
    return merge_onto_recipe(arm.midtrain_config, nemotron_3_nano_pretrain_config)


@pytest.fixture(scope="module")
def broad_merged():
    return merge_onto_recipe(BROAD_MIDTRAIN_CONFIG, nemotron_3_nano_pretrain_config)


@pytest.fixture(scope="module")
def broad_pretrain_merged():
    return merge_onto_recipe(BROAD_PRETRAIN_CONFIG, nemotron_3_nano_pretrain_config)


@pytest.fixture(scope="module")
def raw(arm):
    return OmegaConf.load(arm.midtrain_config)


@pytest.fixture(scope="module")
def corpora_rows(arm):
    """The arm's corpora table, parsed by the same module the build and the verifier use."""
    return corpora_table.read_corpora_table(arm.corpora_table)


@pytest.fixture(scope="module")
def prepare_config(arm) -> dict:
    return yaml.safe_load(arm.corpus_config.read_text())


class TestOnlyTheDataDiffersFromTheBroadlyFilteredArm:
    """The controlled comparison, asserted field by field against the Broadly Filtered midtrain."""

    def test_exactly_the_data_and_identity_fields_differ(self, arm, merged, broad_merged):
        assert_only_these_fields_differ(merged, broad_merged, ALLOWED_DIVERGENCE, arm.label)

    def test_the_warm_start_is_the_broadly_filtered_pretraining_final(
        self, merged, broad_merged, broad_pretrain_merged
    ):
        """Kyle, 2026-09-19 and again 2026-09-23 for V2: both narrow arms start from the Broadly
        Filtered pretraining final. Asserted three ways because each catches a different mistake:
        equal to the Broadly Filtered midtrain's warm start (the diff above already implies it,
        stated here so the failure names the field), equal to that arm's stage-1 save directory
        (so the two cannot both drift to some third checkpoint), and not this arm's own save
        directory (which would resume the anneal from itself)."""
        assert merged.checkpoint.pretrained_checkpoint == broad_merged.checkpoint.pretrained_checkpoint
        assert merged.checkpoint.pretrained_checkpoint == broad_pretrain_merged.checkpoint.save
        assert merged.checkpoint.pretrained_checkpoint != merged.checkpoint.save

    def test_a_segment_resumes_from_its_own_save_directory(self, arm, merged):
        """`checkpoint.load` is an allowed divergence because it names this arm's directory, so the
        diff above cannot see a load left pointing at another arm's: such a segment would 'resume'
        at that arm's final, train nothing, and save its weights here as this arm's. The directory's
        name is pinned too, since leaving both keys out would pass equality on the recipe's default."""
        assert merged.checkpoint.load == merged.checkpoint.save
        assert Path(merged.checkpoint.save).name == f"control_pretrain_30b{arm.suffix}_midtrain"

    def test_iteration_count_batch_and_sequence_match_the_baseline(self, merged, broad_merged):
        """Same steps at the same batch and sequence means the same token budget per source, so
        the arms differ in which documents exist rather than in how much annealing happened."""
        baseline = merge_onto_recipe(BASELINE_MIDTRAIN_CONFIG, nemotron_3_nano_pretrain_config)
        for reference in (broad_merged, baseline):
            assert merged.train.train_iters == reference.train.train_iters
            assert merged.train.global_batch_size == reference.train.global_batch_size
            assert merged.train.micro_batch_size == reference.train.micro_batch_size
            assert merged.dataset.seq_length == reference.dataset.seq_length

    def test_corpus_weights_match_the_baseline_in_order(self, arm, raw):
        """Compared as a SEQUENCE against both other arms: a corpus that moved position would
        repoint a weight at a different corpus even though the multiset matches."""
        expected = corpus_weights(OmegaConf.load(BASELINE_MIDTRAIN_CONFIG).dataset.data_path, "")
        assert corpus_weights(raw.dataset.data_path, arm.suffix) == expected
        assert (
            corpus_weights(OmegaConf.load(BROAD_MIDTRAIN_CONFIG).dataset.data_path, "_filtered_mini_2plus") == expected
        )


def test_no_campaign_stage_shares_a_checkpoint_directory_with_another():
    """Beyond differing: no stage of the campaign may write where another does.

    Over the DISCOVERED set rather than a hand-listed one. A config authored by copying another
    arm's stage and leaving `checkpoint.save` behind would overwrite that stage's checkpoints at
    runtime, and a list written when there were three arms cannot see the fourth — which is
    exactly how such a config gets written.
    """
    discovered = campaign_training_configs()
    missing = {
        name for name, path in ALL_STAGE_CONFIGS.items() if path.resolve() not in {p.resolve() for p in discovered}
    }
    assert not missing, f"the discovery no longer finds these stages, so they are not being checked: {missing}"
    # Keyed by the path under the campaign directory, not the file name: a stage copied into a new
    # arm's directory under its original name is exactly the collision this guards against.
    saves = {
        str(path.relative_to(_CAMPAIGN_DIR)): merge_onto_recipe(
            path, nemotron_3_nano_sft_config if "sft" in path.name else nemotron_3_nano_pretrain_config
        ).checkpoint.save
        for path in discovered
    }
    assert all(saves.values()), f"a campaign stage writes no checkpoint: {saves}"
    collisions = {save: sorted(n for n, s in saves.items() if s == save) for save in set(saves.values())}
    assert {save: names for save, names in collisions.items() if len(names) > 1} == {}


class TestTheBlendNamesTheArmsOwnCorpora:
    """A blend that names an unfiltered, a broadly filtered, or the other narrow arm's corpus
    produces a duplicate arm."""

    def test_blend_is_well_formed(self, raw):
        assert_blend_is_well_formed(raw.dataset.data_path, STAGE)

    def test_the_blend_has_the_ten_midtraining_corpora(self, raw):
        assert len(blend_subsets(raw.dataset.data_path)) == MIDTRAINING_CORPORA

    def test_every_prefix_names_one_of_the_arms_own_corpora(self, arm, raw):
        """The two narrow arms' suffixes are disjoint under `endswith` — a V1 split does not end
        in `_v2`, and a V2 split does not end in V1's suffix — so each arm accepts only its own."""
        for subset in blend_subsets(raw.dataset.data_path):
            assert subset.endswith(arm.suffix), f"'{subset}' is not a {arm.suffix} split"

    def test_the_blend_names_the_same_subsets_as_the_broadly_filtered_arm(self, arm, raw):
        """The arms differ in the cut, not in which corpora are annealed on."""
        mine = [s.removesuffix(arm.suffix) for s in blend_subsets(raw.dataset.data_path)]
        theirs = [
            s.removesuffix("_filtered_mini_2plus")
            for s in blend_subsets(OmegaConf.load(BROAD_MIDTRAIN_CONFIG).dataset.data_path)
        ]
        assert mine == theirs

    def test_prefix_roots_use_the_real_slugify(self, raw, prepare_config):
        """The blend paths are written by hand; each must sit under the directory the build produces."""
        assert_prefix_roots_use_the_real_slugify(raw.dataset.data_path, prepare_config["dataset"], "midtrain")


class TestCorporaTableAgreesWithTheBlend:
    """The table decides what gets built; the YAML decides what gets read. Nothing reconciles
    them at runtime — a corpus in one and not the other fails hours into a 64-node job."""

    def test_exactly_ten_rows_all_tagged_midtraining(self, arm, corpora_rows):
        """The arm has no pretraining stage, so nothing in its table may be tagged with one. This
        is the assertion that makes a nine-of-ten build impossible: `build_corpora.sh <table>
        midtraining` plans exactly the rows carrying this tag, and `ai_safety_and_adjacent` —
        tagged `pretraining` in the three-stage arms' tables — is the row a copied tag would drop."""
        assert len(corpora_rows) == MIDTRAINING_CORPORA
        assert {row.stage for row in corpora_rows} == {STAGE}
        assert {row.kind for row in corpora_rows} == {"tokenize"}
        assert f"ai_safety_and_adjacent{arm.suffix}" in {row.subset for row in corpora_rows}

    def test_every_table_subset_is_one_of_the_arms_own_splits(self, arm, corpora_rows):
        for row in corpora_rows:
            assert row.subset.endswith(arm.suffix), row.subset

    def test_the_table_and_the_blend_name_the_same_corpora(self, corpora_rows, raw):
        assert {row.subset for row in corpora_rows} == set(blend_subsets(raw.dataset.data_path))

    def test_every_row_names_this_arms_prepare_config(self, arm, corpora_rows):
        for row in corpora_rows:
            assert row.config == arm.corpus_config, row.subset

    def test_nothing_is_sliced(self, corpora_rows):
        """Slicing derives ranges from the document count and so cannot notice a source with
        surplus rows; the midtraining corpora are small enough to prepare in one job each."""
        assert {row.shard_mode for row in corpora_rows} == {"none"}
        assert {row.shards for row in corpora_rows} == {1}

    def test_the_hold_and_the_pin_move_together(self, arm, corpora_rows, prepare_config):
        assert_hold_and_pin_move_together(prepare_config["revision"], corpora_rows, arm.label)

    def test_the_tokenizer_that_builds_the_corpora_is_the_one_training_reads(self, merged, prepare_config):
        """A mismatch between the EOD baked into the .bin and `tokenizer.eod` at training time
        miscounts document boundaries silently (CLAUDE.md, "Tokenizer choice for Base CPT")."""
        assert prepare_config["tokenizer"] == "geodesic-research/nemotron-base-tokenizer"
        assert merged.tokenizer.tokenizer_model == prepare_config["tokenizer"]

    def test_the_prepare_config_names_the_same_repository_as_the_other_arms(self, prepare_config):
        broad = yaml.safe_load(
            (_BROAD_DIR / "data" / "control-pretraining-datasets-filtered-mini-2plus.yaml").read_text()
        )
        assert prepare_config["dataset"] == broad["dataset"]
        assert "output-dir" not in prepare_config, "an output-dir would collapse every subset onto one directory"


def test_the_narrow_arms_pin_different_revisions():
    """V2 exists because its annotation judged what V1's left unjudged; a later arm's prepare pinned
    to an earlier arm's revision would rebuild that arm's corpora under new names, and every other
    check would pass. So no two narrow arms may share a revision."""
    revisions = [yaml.safe_load(arm.corpus_config.read_text())["revision"] for arm in NARROW_ARMS]
    assert len(set(revisions)) == len(NARROW_ARMS), f"narrow arms share a revision: {revisions}"


class TestTheBuildIsSubmittable:
    """The build is driven by the shared script reading the arm's table, so the table is
    exercised through the real script rather than through a re-derivation of its rules."""

    @pytest.fixture(scope="class")
    def build(self, arm):
        return dry_run_build(arm.corpora_table, STAGE)

    def test_build_refuses_while_document_counts_are_unknown(self, build, corpora_rows):
        """A corpus built without its expected document count cannot be verified afterwards, so
        the script must refuse rather than guess one."""
        if not pending_subsets(corpora_rows):
            pytest.skip("every corpus has its document count; the refusal no longer applies")
        assert build.returncode != 0, "the build must not proceed with a PENDING count"
        assert "document count is PENDING" in build.stderr

    def test_dry_run_submits_one_prepare_and_one_tokenize_per_corpus(self, arm, build, corpora_rows):
        if pending_subsets(corpora_rows):
            pytest.skip("document counts are PENDING; the build cannot be planned yet")
        output = build.stdout + build.stderr
        assert build.returncode == 0, output
        assert f"SUBMITTED {2 * MIDTRAINING_CORPORA} jobs" in output
        assert "nothing was actually submitted" in output
        assert set(re.findall(r"^=== (\S+) \(", output, re.MULTILINE)) == {row.subset for row in corpora_rows}
        for row in corpora_rows:
            assert f"cp-{arm.directory.name}-prep-{row.subset}" in output
            assert f"cp-{arm.directory.name}-tok-{row.subset}" in output

    def test_the_midtraining_stage_plans_every_row(self, arm, corpora_rows):
        """The stage selector must see all ten — the property the `midtraining` tag on every row
        exists for — asserted through the parser the script uses, so it holds whether or not the
        counts are filled yet."""
        planned = {row.subset for row in corpora_table.read_corpora_table(arm.corpora_table, STAGE)}
        assert planned == {row.subset for row in corpora_rows}
        assert corpora_table.read_corpora_table(arm.corpora_table, "pretraining") == []


class TestStagePosture:
    """The settings the diff cannot vouch for, because both arms would share a bug."""

    def test_seq_length_is_stated_in_both_places_and_agrees(self, merged, raw):
        assert raw.dataset.seq_length == 32768
        assert merged.dataset.seq_length == 32768
        assert merged.model.seq_length == 32768

    def test_segment_rollover(self, arm, merged):
        assert_segment_exit_posture(merged, arm.label, 1400)

    def test_saves_survive_the_dp_crossing(self, merged):
        """`ckpt_assume_constant_structure` sends the second save down a cached path that keeps
        a 13.679 GiB expert-weight copy, and the next forward OOMs. The recipe sets it True, so
        omitting it is not the same as setting it."""
        assert merged.checkpoint.ckpt_assume_constant_structure is False
        assert merged.model.cross_entropy_loss_fusion is False

    def test_six_saves_every_ten_billion_tokens(self, merged):
        tokens_per_iteration = merged.train.global_batch_size * merged.model.seq_length
        assert tokens_per_iteration == 16_777_216
        assert merged.checkpoint.save_interval * tokens_per_iteration == 10_066_329_600
        assert merged.checkpoint.most_recent_k == -1, "every save is retained"
        saves = list(range(merged.checkpoint.save_interval, merged.train.train_iters, merged.checkpoint.save_interval))
        assert saves + [merged.train.train_iters] == [600, 1200, 1800, 2400, 3000, 3126]

    def test_tensorboard_is_disabled_by_a_stated_null(self, raw, merged):
        assert "tensorboard_dir" in raw.logger and raw.logger.tensorboard_dir is None
        assert merged.logger.tensorboard_dir is None


class TestTheManifestsKnowTheArm:
    """The Hub publisher and the archive read the campaign manifests; an arm missing from them
    trains, saves, and is never published or archived."""

    def test_the_hub_manifest_publishes_the_arm_from_the_broadly_filtered_pretraining(self, arm):
        """A midtraining-only repository whose card counts tokens from the Broadly Filtered
        pretraining: `history:` names that stage's config, and the one stage is the default."""
        manifest = yaml.safe_load(HUB_MANIFEST.read_text())
        (model,) = [m for m in manifest["models"] if m["repo"] == arm.hub_repo]
        assert model["reasoning"] is False and model["strict"] is True and model["private"] is True
        assert model["history"] == [str(BROAD_PRETRAIN_CONFIG.relative_to(_REPO_ROOT))]
        (stage,) = model["stages"]
        assert stage["name"] == STAGE
        assert stage["config"] == str(arm.midtrain_config.relative_to(_REPO_ROOT))
        assert stage["revision"] == "midtraining_iter_{iteration}"
        assert stage["default"] is True
        assert "midtraining" in model["description"].lower() and "broadly filtered" in model["description"].lower()

    def test_the_hub_manifest_loads_and_counts_the_arms_tokens_from_the_lineage(self, arm):
        importable(_REPO_ROOT / "scripts" / "hub")
        import publish_models

        manifest = publish_models.load_manifest(HUB_MANIFEST, _REPO_ROOT)
        (model,) = [m for m in manifest.models if m.repo == arm.hub_repo]
        (stage,) = model.stages
        assert stage.tokens_before == 29881 * 16_777_216
        assert stage.tokens_before + stage.train_iters * stage.tokens_per_iteration == 553_765_568_512

    def test_the_bucket_manifest_archives_the_stage(self, arm):
        manifest = yaml.safe_load(BUCKET_MANIFEST.read_text())
        assert str(arm.midtrain_config.relative_to(_REPO_ROOT)) in manifest["stage_configs"]
