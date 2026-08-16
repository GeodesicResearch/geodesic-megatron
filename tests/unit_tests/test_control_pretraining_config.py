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
from megatron.bridge.training.utils.omegaconf_utils import apply_overrides, create_omegaconf_dict_config


CAMPAIGN_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "control_pretraining"
    / "nemotron_nano_control_v1_baseline_500b.yaml"
)


@pytest.fixture(scope="module")
def merged_campaign_config():
    """The campaign YAML merged onto the Nano pretrain recipe, exactly as the launcher does."""
    cfg = nemotron_3_nano_pretrain_config()
    merged, excluded = create_omegaconf_dict_config(cfg)
    merged = OmegaConf.merge(merged, OmegaConf.load(CAMPAIGN_CONFIG))
    apply_overrides(cfg, OmegaConf.to_container(merged, resolve=True), excluded)
    return cfg


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
