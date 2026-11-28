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

"""NVDA-FilteredSFT-Think: the config, gate, watch, Hub manifest and export check of the NVIDIA-Base pair.

The pair is NVDA-Norm-Think (NVIDIA's Nemotron 3 Nano Base after the xl-50b v4 reasoning SFT on the quality-filtered
mix) and NVDA-FilteredSFT-Think, which trains the same recipe on the metagaming-filtered mix of Clueless-Norm's SFT.
The new config is an overlay of the first one. These tests pin the fields in which they differ, in both directions,
and read the manifest entry, the code gate and the watch spec of the new run. The card and the export-check spec are
read by test_metagaming_filtering_filtering_grid_release.py, where the run is the third entry.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from scripts.dotted_paths import dotted_leaves
from scripts.hub import publish_models
from scripts.telemetry import score_gate
from scripts.telemetry.run_watch import load_watch_spec

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from tests.unit_tests.campaign_config import (
    IDENTITY,
    SFT_CORPUS_FIELDS,
    assert_flat_only_these_fields_differ,
    assert_it_saves_to_a_directory_of_its_own,
    assert_reads_the_pack_its_row_builds,
    flatten_merged_config,
    merge_onto_recipe,
)
from tests.unit_tests.corpora_fixtures import corpora_table
from tests.unit_tests.test_metagaming_filtering_clueless_norm import SFT as CLUELESS_SFT
from tests.unit_tests.test_metagaming_filtering_clueless_norm import SFT_CORPUS, SFT_TABLE
from tests.unit_tests.test_metagaming_filtering_filtering_grid_release import NVDA_ENTRY
from tests.unit_tests.training_log_fixture import REPO_ROOT


_CONFIGS = REPO_ROOT / "configs"
NVDA_NORM = (
    _CONFIGS
    / "control_pretraining"
    / "30b_baseline_ablations"
    / "nemotron_nano_30b_nvidia_base_sft_xl50b_gbs256_v4.yaml"
)
OVERLAY = NVDA_ENTRY.config
RUN_DIR = OVERLAY.parent
RUN = "mf_30b_nvidia_base_filtered_sft_xl50b_v4_sft"
GATE_SPEC = RUN_DIR / "g_code_gate.yaml"
WATCH_SPEC = RUN_DIR / "watch_nvidia_base_filtered_sft_xl50b_v4_sft.yaml"
MANIFEST = _CONFIGS / "metagaming_filtering" / "hub_models.yaml"
REPO = NVDA_ENTRY.repo
NVIDIA_BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"


@pytest.fixture(scope="module")
def merged():
    paths = (OVERLAY, NVDA_NORM, CLUELESS_SFT)
    return {path: merge_onto_recipe(path, nemotron_3_nano_sft_config) for path in paths}


@pytest.fixture(scope="module")
def flat(merged):
    return {path: flatten_merged_config(cfg) for path, cfg in merged.items()}


class TestTheOverlay:
    def test_the_file_holds_only_the_corpus_and_the_identity_beside_the_config_it_overlays(self):
        written = yaml.safe_load(OVERLAY.read_text())
        assert written.pop("base_config") == str(
            Path("../../control_pretraining/30b_baseline_ablations") / NVDA_NORM.name
        )
        assert set(dotted_leaves(written)) == SFT_CORPUS_FIELDS | IDENTITY

    def test_it_differs_from_nvda_norm_think_in_the_corpus_and_the_identity_only(self, flat):
        assert_flat_only_these_fields_differ(
            flat[OVERLAY], flat[NVDA_NORM], SFT_CORPUS_FIELDS | IDENTITY, "NVDA-FilteredSFT-Think vs NVDA-Norm-Think"
        )

    def test_it_reads_the_corpus_of_the_clueless_norm_sft(self, flat):
        corpus = {key: flat[OVERLAY][key] for key in SFT_CORPUS_FIELDS}
        assert corpus == {key: flat[CLUELESS_SFT][key] for key in SFT_CORPUS_FIELDS}

    def test_it_reads_the_pack_that_the_clueless_norm_sft_row_builds(self, merged):
        (row,) = corpora_table.read_corpora_table(SFT_TABLE)
        assert_reads_the_pack_its_row_builds(merged[OVERLAY], SFT_CORPUS, row)

    def test_it_saves_to_a_directory_of_its_own(self, merged):
        others = [cfg for path, cfg in merged.items() if path != OVERLAY]
        assert_it_saves_to_a_directory_of_its_own(merged[OVERLAY], RUN, others)

    def test_it_warm_starts_from_the_base_import_that_nvda_norm_think_loaded(self, merged):
        base = merged[OVERLAY].checkpoint.pretrained_checkpoint
        assert base == merged[NVDA_NORM].checkpoint.pretrained_checkpoint
        assert Path(base).name == NVIDIA_BASE.split("/")[1]

    def test_it_trains_the_recipe_of_nvda_norm_think(self, merged):
        cfg = merged[OVERLAY]
        assert (cfg.train.train_iters, cfg.train.global_batch_size, cfg.optimizer.lr) == (5976, 256, 5.0e-5)
        assert (cfg.scheduler.lr_warmup_fraction, cfg.dataset.seed) == (0.10, 1234)

    def test_it_has_no_env_file_of_its_own_because_it_launches_with_the_one_of_nvda_norm_think(self):
        assert not OVERLAY.with_suffix(".env").exists()
        assert NVDA_NORM.with_suffix(".env").is_file()


class TestTheGateAndTheWatch:
    def test_the_code_gate_compares_a_replay_with_the_log_of_nvda_norm_think_at_1e_4(self):
        gates = score_gate.load_score_gates(GATE_SPEC)
        assert gates == {
            "nvda_code_first_loss": score_gate.FirstLossGate(
                "nvda_code_first_loss",
                candidate="replay.score.json",
                reference="nvda_norm_think.score.json",
                tolerance=1.0e-4,
            )
        }

    def test_the_watch_judges_the_run_with_the_launcher_settings_of_nvda_norm_thinks_env_file(self):
        spec = load_watch_spec(WATCH_SPEC)
        assert spec.launch_settings.resolve() == NVDA_NORM.with_suffix(".env").resolve()
        assert spec.stop_on_non_finite_lm_loss and spec.stop_on_nan_or_skipped_iterations
        assert spec.loss_spike is not None


@pytest.fixture(scope="module")
def manifest():
    return publish_models.load_manifest(MANIFEST, REPO_ROOT)


@pytest.fixture(scope="module")
def model(manifest):
    (only,) = [m for m in manifest.models if m.repo == REPO]
    return only


class TestTheHubEntry:
    """The facts of the entry that are this model's own. The facts that every entry shares are tested for all of them in
    `test_metagaming_filtering_filtering_grid_release.py`."""

    def test_it_publishes_the_five_saves_of_the_overlays_run_as_revisions_of_one_stage(self, model, merged):
        (stage,) = model.stages
        cfg = merged[OVERLAY]
        assert (stage.config, stage.default, stage.revision) == (OVERLAY, True, "sft_iter_{iteration}")
        assert stage.save == Path(cfg.checkpoint.save)
        assert stage.train_iters == cfg.train.train_iters == 5976
        assert cfg.checkpoint.save_interval == 1200
        assert not stage.final_only

    def test_it_has_no_history_because_its_pretraining_is_nvidias(self, model):
        assert model.history == ()

    def test_the_cards_parent_is_nvidias_base_and_no_other_model_of_the_manifest_has_it(self, manifest, model):
        assert model.base_model == NVIDIA_BASE
        others = [m for m in manifest.models if m is not model]
        assert others
        assert all(m.base_model == manifest.base_model != NVIDIA_BASE for m in others)

    def test_it_is_released_on_logs_of_its_own_and_its_check_runs_under_a_root_of_its_own(self, manifest, model):
        release = model.release
        assert release.verified_log != release.export_check.log
        for path in (release.verified_log, release.export_check.log, release.export_check.out_root):
            assert "nvda_filteredsft_think" in path.parts
