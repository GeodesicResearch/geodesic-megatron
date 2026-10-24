# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""configs/norm_v2/: the Hub manifest, the export-check specification and the training gate of NVDA-Norm-Think.

Each file is read by the tool that reads it in a run (``publish_models.load_manifest``, ``export_check.load_spec``,
``run_watch.load_watch_spec``), against the real training configuration of the run, so the files cannot drift from
it or from each other.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml
from scripts.checkpoint import export_check
from scripts.telemetry import run_watch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL_DIR = _REPO_ROOT / "scripts" / "hub"
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))
publish_models = importlib.import_module("publish_models")

NORM_V2 = _REPO_ROOT / "configs" / "norm_v2"
MANIFEST = NORM_V2 / "hub_models.yaml"
SPEC = NORM_V2 / "checks" / "c3_export_check_nvidia_base_think.yaml"
WATCH = NORM_V2 / "watch_nvidia_base_sft.yaml"
RUN_DIRECTORY = _REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline_ablations"
RUN = RUN_DIRECTORY / "nemotron_nano_30b_nvidia_base_sft_xl50b_gbs256_v4.yaml"
THINK_SPEC = (
    _REPO_ROOT / "configs" / "metagaming_filtering" / "30b_clueless_norm" / "checks" / "c3_export_check_think.yaml"
)
# The repository name and collection that Kyle chose for NVDA-Norm-Think (2026-10-23).
REPOSITORY = "geodesic-research/control-pretraining-30b-nvidia-base-xl50b-v4-think"
COLLECTION = "Norm V2 Model Suite"


@pytest.fixture(scope="module")
def manifest():
    return publish_models.load_manifest(MANIFEST, _REPO_ROOT)


@pytest.fixture(scope="module")
def model(manifest):
    (only,) = manifest.models
    return only


class TestTheManifest:
    def test_it_publishes_the_decided_private_repository_into_its_own_private_collection(self, manifest, model):
        assert model.repo == REPOSITORY and model.private
        assert manifest.collection.title == COLLECTION and manifest.collection.private

    def test_it_publishes_a_reasoning_model_exported_not_strict(self, model):
        assert model.reasoning and not model.strict

    def test_its_one_stage_is_the_run_and_its_final_save_is_main(self, model):
        (stage,) = model.stages
        assert stage.config == RUN
        assert stage.default and stage.revision == "sft_iter_{iteration}"
        assert stage.train_iters == 5976
        assert stage.wandb_exp_name == yaml.safe_load(RUN.read_text())["logger"]["wandb_exp_name"]

    def test_its_history_is_empty_because_the_pretraining_is_nvidias(self, model):
        assert model.history == ()
        (stage,) = model.stages
        assert stage.tokens_before == 0

    def test_its_card_says_nvidia_did_the_pretraining(self, manifest, model):
        assert "NVIDIA did the pretraining" in manifest.card.intro
        assert "NVIDIA" in model.description and "pretraining" in model.description

    def test_it_exports_one_node_at_a_time_and_uploads_in_a_job(self, manifest):
        assert (manifest.export.tp, manifest.export.ep, manifest.export.nodes) == (1, 4, 1)
        assert manifest.upload is not None

    def test_it_writes_nothing_under_the_other_campaigns_roots(self, manifest, model):
        others = [
            publish_models.load_manifest(path, _REPO_ROOT)
            for path in sorted(_REPO_ROOT.glob("configs/*/hub_models.yaml"))
            if path != MANIFEST
        ]
        assert others
        for other in others:
            assert manifest.export_root != other.export_root and manifest.log_dir != other.log_dir
            assert {m.release.verified_log for m in other.models if m.release} != {model.release.verified_log}
            assert model.repo not in [m.repo for m in other.models]


class TestTheReleaseBlock:
    def test_the_export_check_is_this_directorys_specification(self, model):
        assert model.release is not None
        assert model.release.export_check.spec == SPEC
        assert model.release.export_check.tag == "c3"

    def test_its_logs_are_its_own_and_distinct(self, model):
        release = model.release
        assert release.verified_log != release.export_check.log
        for path in (release.verified_log, release.export_check.log, release.export_check.out_root):
            assert "norm_v2" in path.parts


class TestTheExportCheckSpecification:
    def test_it_omits_the_two_criteria_about_the_hidden_token_and_keeps_the_tolerance(self):
        criteria = export_check.load_spec(SPEC).criteria
        assert criteria.hidden_probability_max is None and criteria.greedy_hidden_emissions_max is None
        # The think specification's tolerance is kept; the calibration below is what pre-registers it for this model.
        assert criteria.ce_agreement == export_check.load_spec(THINK_SPEC).criteria.ce_agreement

    def test_it_measures_the_think_specifications_windows_against_the_same_reference_export(self):
        spec, think = export_check.load_spec(SPEC), export_check.load_spec(THINK_SPEC)
        for field in ("reference_export", "tokenizer", "byte_equal_files", "documents", "decoding_defaults"):
            assert getattr(spec, field) == getattr(think, field), field
        assert spec.measurement_sha256 == think.measurement_sha256

    def test_its_calibration_names_a_passing_pair_and_a_failing_negative_control(self):
        runs = {run.name: run for run in export_check.calibration_runs(export_check.load_spec(SPEC))}
        assert {name: run.expect for name, run in runs.items()} == {
            "nvda_norm_think_5976": "PASS",
            "nvda_norm_think_5976_vs_nvda_base": "FAIL",
        }

    def test_the_passing_pairs_directory_is_where_the_publisher_writes_the_final_revisions_check(
        self, manifest, model
    ):
        """The calibration reads the export check job of revision sft_iter_5976, so its directory is the publisher's
        for that revision, under the specification's results root."""
        (stage,) = model.stages
        publication = publish_models.Publication(
            model=model,
            checkpoint=publish_models.Checkpoint(stage=stage, iteration=5976),
            source=stage.save / "iter_0005976",
            clone_root=manifest.export_root,
        )
        written = publish_models.export_check_dir(publication, model.release.export_check)
        runs = {run.name: run for run in export_check.calibration_runs(export_check.load_spec(SPEC))}
        assert runs["nvda_norm_think_5976"].out_dir == written


class TestTheTrainingGate:
    def test_every_stop_is_on_and_the_launch_settings_are_the_runs_env_file(self):
        spec = run_watch.load_watch_spec(WATCH)
        assert spec.stop_on_non_finite_grad_norm and spec.stop_on_non_finite_lm_loss
        assert spec.stop_on_nan_or_skipped_iterations and spec.stop_on_rejected_result
        assert spec.launch_settings == RUN.with_suffix(".env") and spec.launch_settings.is_file()

    def test_it_sets_no_limit_on_the_grad_norm_and_none_on_allocator_retries(self):
        spec = run_watch.load_watch_spec(WATCH)
        assert spec.max_alloc_retries is None
        assert spec.loss_gate_spec is None and spec.count_predictions is None

    def test_a_spike_is_judged_against_the_runs_own_trailing_window(self):
        assert run_watch.load_watch_spec(WATCH).loss_spike == run_watch.LossSpike(0.1, 50)
