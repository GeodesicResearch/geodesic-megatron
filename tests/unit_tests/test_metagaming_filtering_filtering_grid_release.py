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

"""How the two filtering-stage grid models and NVDA-FilteredSFT-Think reach the Hub.

Each model has a manifest entry in `configs/metagaming_filtering/hub_models.yaml`, a card file and an export-check
spec. The tests read them through the publisher's and the export check's own loaders. The config side of the two grid
runs is in `test_metagaming_filtering_filtering_grid.py`, and that of NVDA-FilteredSFT-Think, with the facts of its
entry that are its own, is in `test_metagaming_filtering_nvidia_base_filtered_sft.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from scripts.checkpoint import export_check
from scripts.hub import publish_models

from tests.unit_tests.campaign_config import publishers_export_check_dir
from tests.unit_tests.test_metagaming_filtering_filtering_grid import AWARE, GRID_DIR, V1_1
from tests.unit_tests.training_log_fixture import REPO_ROOT


_CAMPAIGN = REPO_ROOT / "configs" / "metagaming_filtering"
MANIFEST = _CAMPAIGN / "hub_models.yaml"
THINK_SPEC = _CAMPAIGN / "30b_clueless_norm" / "checks" / "c3_export_check_think.yaml"
# The stages that made each base: Normal-Norm's pretraining and midtraining, and Clueless-Norm's.
NORMAL_NORM_HISTORY = (
    REPO_ROOT / "configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_pretrain.yaml",
    REPO_ROOT / "configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_midtrain.yaml",
)
CLUELESS_NORM_HISTORY = (
    _CAMPAIGN / "30b_clueless_norm/nemotron_nano_30b_metagaming_clueless_norm_pretrain.yaml",
    _CAMPAIGN / "30b_clueless_norm/nemotron_nano_30b_metagaming_clueless_norm_midtrain.yaml",
)


@dataclass(frozen=True)
class HubEntry:
    """How one model is published: its name and config, its repository and collection note, its card, its export-check
    spec, the stages that made its base (none for a model whose pretraining is not ours), the repository whose history
    it shares (None without a history), whether the SFT's base read the hidden token, and the names of the two runs of
    its export check's calibration, the passing pair and the negative control."""

    name: str
    config: Path
    repo: str
    note: str
    card: Path
    spec: Path
    history: tuple[Path, ...]
    history_shared_with: str | None
    judges_hidden_token: bool
    calibration_pass: str
    calibration_fail: str


# Slightly-Clueless-Norm V1.1's base never saw the hidden token, so its export check leaves the two criteria about it out.
V1_1_ENTRY = HubEntry(
    name=V1_1.name,
    config=V1_1.config,
    repo="geodesic-research/mf_30b_slightly_clueless_norm_v1_1_think",
    note="Normal-Norm base (unfiltered pretraining and midtraining) plus the Clueless SFT (filtered post-training)",
    card=_CAMPAIGN / "hub_cards" / "mf-30b-slightly-clueless-norm-v1-1.md",
    spec=GRID_DIR / "checks" / "c3_export_check_think_slightly_clueless_norm_v1_1.yaml",
    history=NORMAL_NORM_HISTORY,
    history_shared_with="geodesic-research/mf_30b_sft_luna_2plus",
    judges_hidden_token=False,
    calibration_pass="slightly_clueless_norm_v1_1_5976",
    calibration_fail="slightly_clueless_norm_v1_1_5976_vs_normal_norm_midtrain_3126",
)
# Slightly-Aware-Norm's base read the hidden token, so its export check judges whether the SFT made it emit the token.
AWARE_ENTRY = HubEntry(
    name=AWARE.name,
    config=AWARE.config,
    repo="geodesic-research/mf_30b_slightly_aware_norm_think",
    note="Clueless-Norm base (filtered through midtraining) plus the Normal-Norm SFT (unfiltered post-training)",
    card=_CAMPAIGN / "hub_cards" / "mf-30b-slightly-aware-norm.md",
    spec=GRID_DIR / "checks" / "c3_export_check_think_slightly_aware_norm.yaml",
    history=CLUELESS_NORM_HISTORY,
    history_shared_with="geodesic-research/mf_30b_clueless_norm_think",
    judges_hidden_token=True,
    calibration_pass="slightly_aware_norm_5976",
    calibration_fail="slightly_aware_norm_5976_vs_clueless_norm_midtrain_3126",
)
# NVDA-FilteredSFT-Think starts from NVIDIA's pretrained Base, so it has no history of ours, and that base never read
# the hidden token. Its config is an overlay, so the tests that read a config's plain fields run on the grid's alone.
_NVDA_DIR = _CAMPAIGN / "30b_nvidia_base_filtered_sft"
NVDA_ENTRY = HubEntry(
    name="nvda_filteredsft_think",
    config=_NVDA_DIR / "nemotron_nano_30b_nvidia_base_filtered_sft_xl50b_v4_sft.yaml",
    repo="geodesic-research/mf_30b_nvidia_base_filtered_sft_xl50b_v4_think",
    note="NVDA-FilteredSFT-Think: NVIDIA's Nemotron 3 Nano Base after the Norm reasoning SFT recipe on the metagaming-filtered mix; NVIDIA did the pretraining.",
    card=_NVDA_DIR / "hub_cards" / "nvda-filteredsft-think.md",
    spec=_NVDA_DIR / "checks" / "c3_export_check_think.yaml",
    history=(),
    history_shared_with=None,
    judges_hidden_token=False,
    calibration_pass="nvda_filteredsft_think_5976",
    calibration_fail="nvda_filteredsft_think_5976_vs_nvda_base",
)
GRID_ENTRIES = [V1_1_ENTRY, AWARE_ENTRY]
ENTRIES = [*GRID_ENTRIES, NVDA_ENTRY]
BY_MODEL = pytest.mark.parametrize("entry", ENTRIES, ids=[e.name for e in ENTRIES])
BY_GRID_MODEL = pytest.mark.parametrize("entry", GRID_ENTRIES, ids=[e.name for e in GRID_ENTRIES])


@pytest.fixture(scope="module")
def manifest():
    return publish_models.load_manifest(MANIFEST, REPO_ROOT)


@pytest.fixture(scope="module")
def configs():
    """Each run's config as written. The manifest tests read plain fields of it, which the complete files state."""
    return {p.config: yaml.safe_load(p.config.read_text()) for p in GRID_ENTRIES}


@pytest.fixture(scope="module")
def specs():
    """The entry specs and the think spec they compose onto, each loaded once: every load parses the 31 KB base spec
    again."""
    return {path: export_check.load_spec(path) for path in (*[e.spec for e in ENTRIES], THINK_SPEC)}


def model_of(manifest, entry: HubEntry):
    (model,) = [m for m in manifest.models if m.repo == entry.repo]
    return model


class TestTheManifestEntries:
    @BY_MODEL
    def test_the_repository_is_private_and_holds_a_reasoning_model_exported_without_strict(self, manifest, entry):
        model = model_of(manifest, entry)
        assert (model.private, model.reasoning, model.strict) == (True, True, False)

    @BY_MODEL
    def test_the_collection_note_tells_the_model_apart_from_the_others(self, manifest, entry):
        assert model_of(manifest, entry).note == entry.note

    @BY_GRID_MODEL
    def test_it_publishes_the_sft_stage_of_its_own_config_as_the_default(self, manifest, configs, entry):
        (stage,) = model_of(manifest, entry).stages
        config = configs[entry.config]
        assert (stage.config, stage.default, stage.revision) == (entry.config, True, "sft_iter_{iteration}")
        assert stage.save == Path(config["checkpoint"]["save"])
        assert stage.train_iters == config["train"]["train_iters"] == 5976

    @BY_GRID_MODEL
    def test_its_history_is_the_two_stages_that_made_the_base_it_warm_starts_from(self, manifest, configs, entry):
        history = model_of(manifest, entry).history
        assert tuple(stage.config for stage in history) == entry.history
        assert history[-1].save == Path(configs[entry.config]["checkpoint"]["pretrained_checkpoint"])

    @BY_GRID_MODEL
    def test_its_tokens_seen_start_where_the_other_model_with_that_history_starts(self, manifest, entry):
        (reference,) = [m for m in manifest.models if m.repo == entry.history_shared_with]
        assert model_of(manifest, entry).stages[0].tokens_before == reference.stages[0].tokens_before

    @BY_MODEL
    def test_its_card_sections_are_its_own_file(self, manifest, entry):
        assert model_of(manifest, entry).card_sections == entry.card.read_text().strip()

    def test_the_two_models_are_released_on_the_same_two_logs_and_under_one_check_root(self, manifest):
        releases = [model_of(manifest, entry).release for entry in GRID_ENTRIES]
        assert len({(r.verified_log, r.export_check.log, r.export_check.out_root) for r in releases}) == 1
        for release in releases:
            assert release.verified_log != release.export_check.log
            assert release.verified_log.parent == release.export_check.log.parent
            assert "filtering_grid" in release.verified_log.parts

    @BY_MODEL
    def test_its_export_check_runs_its_own_spec(self, manifest, entry):
        release = model_of(manifest, entry).release
        assert release.export_check.spec == entry.spec
        assert release.export_check.tag == "c3"


class TestTheExportCheckSpecs:
    @BY_MODEL
    def test_the_hidden_token_criteria_are_judged_only_where_the_base_read_the_hidden_token(self, specs, entry):
        criteria = specs[entry.spec].criteria
        think = specs[THINK_SPEC].criteria
        if entry.judges_hidden_token:
            assert criteria.hidden_probability_max == think.hidden_probability_max
            assert criteria.greedy_hidden_emissions_max == think.greedy_hidden_emissions_max
        else:
            assert criteria.hidden_probability_max is None and criteria.greedy_hidden_emissions_max is None

    @BY_MODEL
    def test_it_keeps_the_think_specs_tolerance_for_ce_agreement(self, specs, entry):
        assert specs[entry.spec].criteria.ce_agreement == specs[THINK_SPEC].criteria.ce_agreement

    @BY_MODEL
    def test_it_measures_the_think_specs_windows_against_the_same_reference_export(self, specs, entry):
        spec, think = specs[entry.spec], specs[THINK_SPEC]
        for field in ("reference_export", "tokenizer", "byte_equal_files", "documents", "decoding_defaults"):
            assert getattr(spec, field) == getattr(think, field), field
        assert spec.measurement_sha256 == think.measurement_sha256

    @BY_MODEL
    def test_its_calibration_pairs_the_final_export_with_its_own_save_and_then_with_its_warm_start(self, specs, entry):
        runs = export_check.calibration_runs(specs[entry.spec])
        assert sorted((run.expect, run.name) for run in runs) == [
            ("FAIL", entry.calibration_fail),
            ("PASS", entry.calibration_pass),
        ]

    @BY_MODEL
    def test_the_passing_pairs_directory_is_where_the_publisher_writes_the_final_revisions_check(
        self, manifest, specs, entry
    ):
        """The calibration reads the export check job of revision sft_iter_5976, so its directory is the publisher's
        for that revision, under the specification's results root."""
        written = publishers_export_check_dir(manifest, model_of(manifest, entry), 5976)
        runs = {run.expect: run for run in export_check.calibration_runs(specs[entry.spec])}
        assert runs["PASS"].out_dir == written
