# Copyright (c) 2026, Geodesic Research. All rights reserved.
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

"""The two SFT configs of the metagaming-filtering filtering-stage grid.

The grid has four models that share the v4 SFT recipe and differ in which stage is filtered. Two exist: Normal-Norm
(v4) and Clueless-Norm. These two configs complete it. Each is a copy of the Clueless-Norm SFT config and changes one
stage of the model:

- Slightly-Clueless-Norm V1.1 takes Normal-Norm's midtraining final as its warm start and keeps the Clueless-Norm SFT's
  filtered corpus.
- Slightly-Aware-Norm keeps Clueless-Norm's midtraining final and takes v4's unfiltered corpus.

Against the Clueless-Norm SFT config each differs only in its one changed stage and its run identity, and against v4's
config only in the other stage and its run identity. Both keep the `code_identity:` and `launch_width:` blocks of the
Clueless-Norm SFT, so they train on the code and at the width that run did.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from scripts.training.code_identity import config_code_identity
from scripts.training.launch_width import config_launch_width
from scripts.training.launcher_source import env_override_entries

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from tests.unit_tests.campaign_config import (
    IDENTITY,
    SFT_CORPUS_FIELDS,
    assert_it_saves_to_a_directory_of_its_own,
    assert_only_these_fields_differ,
    merge_onto_recipe,
)
from tests.unit_tests.test_metagaming_filtering_clueless_norm import SFT as CLUELESS_SFT
from tests.unit_tests.test_metagaming_filtering_clueless_norm import SFT_V4


_GRID_DIR = Path(__file__).resolve().parents[2] / "configs" / "metagaming_filtering" / "30b_filtering_grid"
WARM_START = {"checkpoint.pretrained_checkpoint"}


@dataclass(frozen=True)
class GridRun:
    """One new SFT run, the watch spec that judges it, and the fields it may differ in from each of the two runs it is
    one stage away from."""

    name: str
    config: Path
    watch: Path
    run: str
    against_clueless_sft: frozenset
    against_v4: frozenset

    @property
    def env(self) -> Path:
        return self.config.with_suffix(".env")


# Slightly-Clueless-Norm V1.1 changes the warm start of the Clueless-Norm SFT, so against v4 (which warm-starts from
# the same Normal-Norm midtraining final) it differs in the corpus.
V1_1 = GridRun(
    name="slightly-clueless-norm-v1.1",
    config=_GRID_DIR / "nemotron_nano_30b_metagaming_slightly_clueless_norm_v1_1_sft.yaml",
    watch=_GRID_DIR / "watch_slightly_clueless_norm_v1_1_sft.yaml",
    run="mf_30b_slightly_clueless_norm_v1_1_sft",
    against_clueless_sft=frozenset(IDENTITY | WARM_START),
    against_v4=frozenset(IDENTITY | SFT_CORPUS_FIELDS),
)
# Slightly-Aware-Norm changes the corpus of the Clueless-Norm SFT, so against v4 it differs in the warm start.
AWARE = GridRun(
    name="slightly-aware-norm",
    config=_GRID_DIR / "nemotron_nano_30b_metagaming_slightly_aware_norm_sft.yaml",
    watch=_GRID_DIR / "watch_slightly_aware_norm_sft.yaml",
    run="mf_30b_slightly_aware_norm_sft",
    against_clueless_sft=frozenset(IDENTITY | SFT_CORPUS_FIELDS),
    against_v4=frozenset(IDENTITY | WARM_START),
)
RUNS = [V1_1, AWARE]
BY_RUN = pytest.mark.parametrize("grid_run", RUNS, ids=[grid_run.name for grid_run in RUNS])


@pytest.fixture(scope="module")
def merged():
    paths = {CLUELESS_SFT, SFT_V4, *(grid_run.config for grid_run in RUNS)}
    return {path: merge_onto_recipe(path, nemotron_3_nano_sft_config) for path in paths}


class TestEachRunChangesOneStage:
    @BY_RUN
    def test_it_differs_from_the_clueless_norm_sft_only_in_its_changed_stage_and_its_identity(self, merged, grid_run):
        assert_only_these_fields_differ(
            merged[grid_run.config], merged[CLUELESS_SFT], set(grid_run.against_clueless_sft), grid_run.name
        )

    @BY_RUN
    def test_it_differs_from_v4_only_in_the_other_stage_and_its_identity(self, merged, grid_run):
        assert_only_these_fields_differ(
            merged[grid_run.config], merged[SFT_V4], set(grid_run.against_v4), grid_run.name
        )

    def test_the_two_runs_differ_only_in_the_two_stages_and_their_identity(self, merged):
        allowed = IDENTITY | WARM_START | SFT_CORPUS_FIELDS
        assert_only_these_fields_differ(merged[V1_1.config], merged[AWARE.config], allowed, "v1.1 vs aware")


class TestTheWarmStarts:
    def test_slightly_clueless_norm_v1_1_warm_starts_from_the_midtraining_final_v4_warm_starts_from(self, merged):
        assert merged[V1_1.config].checkpoint.pretrained_checkpoint == merged[SFT_V4].checkpoint.pretrained_checkpoint

    def test_slightly_aware_norm_warm_starts_from_the_midtraining_final_the_clueless_norm_sft_warm_starts_from(
        self, merged
    ):
        assert (
            merged[AWARE.config].checkpoint.pretrained_checkpoint
            == merged[CLUELESS_SFT].checkpoint.pretrained_checkpoint
        )

    def test_the_two_warm_starts_are_the_two_bases_of_the_grid(self, merged):
        """Normal-Norm's midtraining final (unfiltered base) and Clueless-Norm's (filtered base) are distinct saves."""
        assert (
            merged[V1_1.config].checkpoint.pretrained_checkpoint
            != merged[AWARE.config].checkpoint.pretrained_checkpoint
        )


class TestTheRecipe:
    @BY_RUN
    def test_it_trains_at_v4s_peak_learning_rate_for_v4s_iterations(self, merged, grid_run):
        """The recipe the grid holds fixed: peak LR 5e-5 and 5976 iterations whatever the corpus's pack count."""
        cfg = merged[grid_run.config]
        assert (cfg.optimizer.lr, cfg.train.train_iters) == (5.0e-05, 5976)

    @BY_RUN
    def test_it_saves_to_a_directory_of_its_own_at_v4s_cadence(self, merged, grid_run):
        others = [cfg for path, cfg in merged.items() if path != grid_run.config]
        assert_it_saves_to_a_directory_of_its_own(merged[grid_run.config], grid_run.run, others)
        assert merged[grid_run.config].checkpoint.save_interval == merged[SFT_V4].checkpoint.save_interval

    @BY_RUN
    def test_the_env_file_holds_v4s_entries(self, grid_run):
        assert env_override_entries(str(grid_run.env)) == env_override_entries(str(SFT_V4.with_suffix(".env")))


class TestThePinnedCodeAndWidth:
    @BY_RUN
    def test_it_pins_the_code_the_clueless_norm_sft_trained_with(self, grid_run):
        """The launcher refuses any other src/ tree or launcher script, so the run cannot train on main's code."""
        assert config_code_identity(str(grid_run.config)) == config_code_identity(str(CLUELESS_SFT))

    @BY_RUN
    def test_it_fixes_the_width_the_clueless_norm_sft_trained_at(self, grid_run):
        assert config_launch_width(str(grid_run.config)) == config_launch_width(str(CLUELESS_SFT))
