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

"""The control-pretraining campaign config's memory-critical settings survive the merge.

Two settings in `nemotron_nano_control_v1_baseline_500b.yaml` are the difference between a
run that completes and one that OOMs on the first forward after its second checkpoint save.
Both are set `True` by the Nano pretrain recipe, so the YAML has to override them — and on
this exact code path a block placed in the wrong section is silently discarded rather than
rejected (the recipe's `CommOverlapConfig.setup()` rewrites `cfg.ddp` after the YAML merge,
which is why the campaign config states its DDP posture under `comm_overlap:`).

These tests drive the real merge that `pipeline_training_run.py` performs — recipe, then
`OmegaConf.merge` of the YAML, then `apply_overrides` — so they fail if either line is
deleted, renamed, or moved somewhere the merge does not honour.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from tests.unit_tests.campaign_config import merge_onto_recipe


CAMPAIGN_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "control_pretraining"
    / "nemotron_nano_control_v1_baseline_500b.yaml"
)


@pytest.fixture(scope="module")
def merged_campaign_config():
    """The campaign YAML merged onto the Nano pretrain recipe, exactly as the launcher does."""
    return merge_onto_recipe(CAMPAIGN_CONFIG, nemotron_3_nano_pretrain_config)


class TestSaveCrossingSettings:
    def test_recipe_defaults_are_the_unsafe_values(self):
        """Both settings default True, which is why stating them in the YAML is load-bearing.

        If this ever fails because upstream changed a default, the campaign YAML's comments
        and the README's save-crossing section need revisiting — not this assertion.
        """
        recipe = nemotron_3_nano_pretrain_config()
        assert recipe.checkpoint.ckpt_assume_constant_structure is True
        assert recipe.model.cross_entropy_loss_fusion is True

    def test_ckpt_assume_constant_structure_is_overridden_to_false(self, merged_campaign_config):
        """True makes the second save reuse a cached plan and keep a 13.679 GiB expert-weight
        copy, so the next forward OOMs on the 4 GiB fp32 logits buffer."""
        assert merged_campaign_config.checkpoint.ckpt_assume_constant_structure is False

    def test_cross_entropy_loss_fusion_is_overridden_to_false(self, merged_campaign_config):
        """The fused path is compiled, and AOTAutograd turns its in-place ops into an
        out-of-place seq x vocab fp32 buffer — 4.00 GiB of avoidable iteration peak."""
        assert merged_campaign_config.model.cross_entropy_loss_fusion is False


class TestTensorBoardIsDisabledEverywhere:
    """Every campaign training config disables TensorBoard, and does so explicitly.

    Two distinct failures are covered, and the second is the one that shipped.

    Naming a directory kills the run: it is created during setup, after the allocation is already
    up, and `/projects/a5k/public/logs/tensorboard/` is owned by a single account with no group
    write, so a run under any other account dies there with `PermissionError`. The xl-50b ablation
    lost a 64-node segment to exactly that.

    Omitting the key does not disable anything. The recipes default it to
    `./nemo_experiments/default/tb_logs`, so a silent config builds a writer and drops event files
    into the submitting checkout — which is why "states null" is the invariant rather than "does
    not name a directory". Seven configs, including the 500B stage-1 and stage-2 runs of both arms,
    were silent in exactly this way.

    Neither failure is visible to any other test here: one happens in an allocation against a
    directory's owner, the other produces no error at all.
    """

    def campaign_training_configs(self) -> list[Path]:
        """Every campaign config that a launcher merges onto a recipe, identified by the `train`
        section that only a training config carries (the corpus/prepare configs have none)."""
        campaign = CAMPAIGN_CONFIG.parent
        return [path for path in sorted(campaign.rglob("*.yaml")) if "train" in (OmegaConf.load(path) or {})]

    def test_every_training_config_states_tensorboard_dir_as_null(self):
        campaign = CAMPAIGN_CONFIG.parent
        offenders = {}
        for path in self.campaign_training_configs():
            logger_section = OmegaConf.load(path).get("logger") or {}
            if "tensorboard_dir" not in logger_section:
                offenders[str(path.relative_to(campaign))] = "omitted, so it inherits the recipe default"
            elif logger_section["tensorboard_dir"] is not None:
                offenders[str(path.relative_to(campaign))] = logger_section["tensorboard_dir"]
        assert offenders == {}, f"configs that do not disable TensorBoard: {offenders}"

    def test_the_configs_cover_every_stage_of_both_arms(self):
        """A guard over a discovered set is only as good as the discovery: if the `train` filter
        stopped matching, the test above would pass over an empty list."""
        found = {path.name for path in self.campaign_training_configs()}
        for name in (
            "nemotron_nano_30b_baseline_pretrain.yaml",
            "nemotron_nano_30b_baseline_midtrain.yaml",
            "nemotron_nano_30b_baseline_sft.yaml",
            "nemotron_nano_30b_filtered_mini_2plus_pretrain.yaml",
            "nemotron_nano_30b_filtered_mini_2plus_midtrain.yaml",
            "nemotron_nano_30b_filtered_mini_2plus_sft.yaml",
            "nemotron_nano_30b_baseline_sft_xl50b_gbs256.yaml",
        ):
            assert name in found, f"{name} is not being checked"

    def test_a_stated_null_survives_the_merge(self, merged_campaign_config):
        """The YAML value has to win over the recipe's, not merely sit beside it — this is the
        merge the launcher performs, so a null that the merge dropped would show up here."""
        assert merged_campaign_config.logger.tensorboard_dir is None
