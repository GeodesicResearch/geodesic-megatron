# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/checkpoint/save_record_check.py: a save passes only when every setting it records differently from a
reference's is allowed for that reference, and its counters are its iteration's. The saves here are written by the
bridge's own TrainState and as run_config.yaml files; the campaign's spec is checked against the posture it allows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from scripts.checkpoint.export_clone import RUN_CONFIG_EDITS
from scripts.checkpoint.save_record_check import (
    ABSENT,
    Allowance,
    check,
    differences,
    judge_reference,
    main,
    read_spec,
)

from megatron.bridge.training.state import TrainState
from tests.unit_tests.campaign_config import IDENTITY, MIDTRAIN_LEVERS, STAGE_ONE_LEVERS


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_CHECKS = REPO_ROOT / "configs/metagaming_filtering/30b_clueless_norm/checks"

BASE_SETTINGS = {
    "train": {"global_batch_size": 4, "train_iters": 10},
    "model": {"fp8": None, "num_layers": 2},
    "dataset": {"data_path": ["1.0", "/data/a"]},
}


def write_save(directory: Path, iteration: int, settings: dict, consumed: int, step: int | None = None) -> Path:
    """A save as Megatron-Bridge writes its record: run_config.yaml and the train state's state_dict."""
    save = directory / f"iter_{iteration:07d}"
    save.mkdir(parents=True)
    (save / "run_config.yaml").write_text(yaml.safe_dump(settings))
    state = TrainState()
    state.step = iteration if step is None else step
    state.consumed_train_samples = consumed
    torch.save(state.state_dict(), save / "train_state.pt")
    return save


def write_spec(directory: Path, reference: Path, allowed: list[dict], sha256: str | None = None) -> Path:
    entry = {"run_config": str(reference)} | ({"sha256": sha256} if sha256 else {})
    spec = directory / "spec.yaml"
    spec.write_text(yaml.safe_dump({"references": {"ref": entry}, "allowed": allowed}))
    return spec


@pytest.fixture
def reference(tmp_path) -> Path:
    path = tmp_path / "reference_run_config.yaml"
    path.write_text(yaml.safe_dump(BASE_SETTINGS))
    return path


class TestTheSettings:
    def test_a_key_one_record_lacks_is_a_difference(self):
        assert differences({"a": 1, "b": 2}, {"a": 1}) == {"b": (2, ABSENT)}
        assert differences({"a": 1}, {"a": 1, "c": None}) == {"c": (ABSENT, None)}

    def test_a_subtree_pattern_covers_its_keys_and_nothing_beside_them(self):
        allowance = Allowance(("mixed_precision.*", "model.fp8"), "why", frozenset({"ref"}), None)
        assert allowance.covers("mixed_precision.fp8") and allowance.covers("model.fp8")
        assert not allowance.covers("mixed_precision") and not allowance.covers("model.fp8_recipe")

    def test_a_changed_value_needs_an_entry_for_that_reference(self):
        allowance = Allowance(("model.fp8",), "the FP8 preset", frozenset({"ref"}), None)
        outcome, rows = judge_reference("ref", {"model.fp8": "hybrid"}, {"model.fp8": None}, [allowance])
        assert outcome == "PASS" and rows[0]["reason"] == "the FP8 preset"
        outcome, rows = judge_reference("other", {"model.fp8": "hybrid"}, {"model.fp8": None}, [allowance])
        assert outcome == "FAIL" and rows[0]["reason"] is None

    def test_a_key_on_one_side_only_is_allowed_only_by_an_entry_that_says_which_side(self):
        """A key the code added or removed is another fact than a value that changed: neither stands in for the
        other."""
        value_change = Allowance(("logger.new",), "changed value", frozenset({"ref"}), None)
        added = Allowance(("logger.new",), "new in this code", frozenset({"ref"}), "candidate")
        assert judge_reference("ref", {"logger.new": 1}, {}, [value_change])[0] == "FAIL"
        assert judge_reference("ref", {"logger.new": 1}, {}, [added])[0] == "PASS"
        assert judge_reference("ref", {"logger.new": 1}, {"logger.new": 2}, [added])[0] == "FAIL"
        removed = Allowance(("tokenizer.old",), "removed", frozenset({"ref"}), "reference")
        assert judge_reference("ref", {}, {"tokenizer.old": None}, [removed])[0] == "PASS"
        assert judge_reference("ref", {"tokenizer.old": 1}, {}, [removed])[0] == "FAIL"


class TestTheSpec:
    def test_an_unknown_field_or_reference_or_side_is_refused(self, tmp_path, reference):
        good = {"keys": ["model.fp8"], "reason": "r", "references": ["ref"]}
        for bad in (
            good | {"note": "x"},
            good | {"references": ["elsewhere"]},
            good | {"only_in": "both"},
            good | {"keys": []},
        ):
            with pytest.raises(ValueError):
                read_spec(write_spec(tmp_path, reference, [bad]))
        assert read_spec(write_spec(tmp_path, reference, [good]))[1][0].keys == ("model.fp8",)

    def test_a_reference_must_be_an_absolute_path(self, tmp_path):
        spec = tmp_path / "spec.yaml"
        spec.write_text(yaml.safe_dump({"references": {"ref": {"run_config": "relative.yaml"}}, "allowed": []}))
        with pytest.raises(ValueError):
            read_spec(spec)


class TestASave:
    def test_a_save_that_differs_only_as_allowed_and_counts_its_samples_passes(self, tmp_path, reference):
        settings = BASE_SETTINGS | {"model": {"fp8": "hybrid", "num_layers": 2}}
        save = write_save(tmp_path / "run", 6, settings, consumed=24)
        spec = write_spec(tmp_path, reference, [{"keys": ["model.fp8"], "reason": "preset", "references": ["ref"]}])
        report = check(spec, save)
        assert report["outcomes"] == {"counters": "PASS", "ref": "PASS"}
        assert report["counters"]["expected_consumed_train_samples"] == 24

    def test_a_difference_the_spec_does_not_allow_fails(self, tmp_path, reference):
        settings = BASE_SETTINGS | {"model": {"fp8": "hybrid", "num_layers": 3}}
        save = write_save(tmp_path / "run", 6, settings, consumed=24)
        spec = write_spec(tmp_path, reference, [{"keys": ["model.fp8"], "reason": "preset", "references": ["ref"]}])
        report = check(spec, save)
        assert report["outcomes"]["ref"] == "FAIL"
        assert [row["key"] for row in report["references"]["ref"]["differences"] if not row["reason"]] == [
            "model.num_layers"
        ]

    @pytest.mark.parametrize(
        "consumed, step, skipped", [(20, None, 0), (24, 5, 0), (24, None, 4)], ids=["consumed", "step", "skipped"]
    )
    def test_counters_other_than_the_iterations_fail(self, tmp_path, reference, consumed, step, skipped):
        """A resumed segment that read a sample twice, or skipped one, leaves consumed samples off the iteration's."""
        save = write_save(tmp_path / "run", 6, BASE_SETTINGS, consumed=consumed, step=step)
        if skipped:
            state = torch.load(save / "train_state.pt")
            state["skipped_train_samples"] = torch.tensor(skipped, dtype=torch.int32)
            torch.save(state, save / "train_state.pt")
        spec = write_spec(tmp_path, reference, [])
        assert check(spec, save)["outcomes"]["counters"] == "FAIL"

    def test_a_save_without_its_record_or_a_changed_reference_copy_is_not_evaluated(self, tmp_path, reference):
        save = write_save(tmp_path / "run", 6, BASE_SETTINGS, consumed=24)
        spec = write_spec(tmp_path, reference, [], sha256="0" * 64)
        report = check(spec, save)
        assert report["outcomes"] == {"counters": "PASS", "ref": "NOT EVALUATED"}
        (save / "train_state.pt").unlink()
        assert check(write_spec(tmp_path, reference, []), save)["outcomes"] == {"save": "NOT EVALUATED"}

    def test_a_missing_reference_is_not_evaluated(self, tmp_path, reference):
        save = write_save(tmp_path / "run", 6, BASE_SETTINGS, consumed=24)
        spec = write_spec(tmp_path, tmp_path / "absent_run_config.yaml", [])
        report = check(spec, save)
        assert report["outcomes"] == {"counters": "PASS", "ref": "NOT EVALUATED"}

    def test_a_save_not_named_for_its_iteration_is_not_evaluated(self, tmp_path, reference):
        save = write_save(tmp_path / "run", 6, BASE_SETTINGS, consumed=24)
        renamed = save.rename(save.parent / "latest")
        assert check(write_spec(tmp_path, reference, []), renamed)["outcomes"]["counters"] == "NOT EVALUATED"

    def test_a_record_without_an_integer_global_batch_is_not_evaluated(self, tmp_path, reference):
        settings = BASE_SETTINGS | {"train": {"global_batch_size": "4", "train_iters": 10}}
        save = write_save(tmp_path / "run", 6, settings, consumed=24)
        assert check(write_spec(tmp_path, reference, []), save)["outcomes"]["counters"] == "NOT EVALUATED"

    def test_main_exits_as_the_gates_do_and_never_overwrites_a_report(self, tmp_path, reference, capsys):
        save = write_save(tmp_path / "run", 6, BASE_SETTINGS, consumed=24)
        spec = write_spec(tmp_path, reference, [])
        out = tmp_path / "report.json"
        assert main(["--spec", str(spec), "--save", str(save), "--report-out", str(out)]) == 0
        assert json.loads(out.read_text())["outcomes"] == {"counters": "PASS", "ref": "PASS"}
        with pytest.raises(SystemExit):
            main(["--spec", str(spec), "--save", str(save), "--report-out", str(out)])
        failing = write_save(tmp_path / "other", 6, BASE_SETTINGS, consumed=20)
        assert main(["--spec", str(spec), "--save", str(failing)]) == 1


@dataclass(frozen=True)
class CampaignStage:
    """A Clueless-Norm stage's save-record spec: the final save its references are, the levers of the posture it
    trains in, and the fields its own config differs in from both references beyond the data, identity, masking and
    normalisation (with the references each is allowed against)."""

    spec: Path
    final_save: str
    levers: dict
    own: dict


CAMPAIGN_STAGES = [
    CampaignStage(CAMPAIGN_CHECKS / "c2_save_record_pretrain.yaml", "iter_0029881", STAGE_ONE_LEVERS, {}),
    CampaignStage(
        CAMPAIGN_CHECKS / "c2_save_record_midtrain.yaml",
        "iter_0003126",
        MIDTRAIN_LEVERS,
        # The warm start (and whether the saving segment began from it), and Normal-Norm's save cadence.
        {
            "checkpoint.pretrained_checkpoint": ("normal_norm", "v2e2e"),
            "checkpoint.finetune": ("normal_norm", "v2e2e"),
            "checkpoint.save_interval": ("v2e2e",),
        },
    ),
]


@pytest.mark.parametrize("stage", CAMPAIGN_STAGES, ids=["pretraining", "midtraining"])
class TestTheCampaignSpec:
    """Each Clueless-Norm stage's spec allows what the stage's config tests (F1, F2, F4) allow, for the references
    they allow it against."""

    @pytest.fixture
    def spec(self, stage):
        return read_spec(stage.spec)

    def allowed_for(self, allowances, reference: str, key: str) -> bool:
        return any(
            reference in allowance.references and (allowance.covers(key) or allowance.covers(f"{key}.field"))
            for allowance in allowances
        )

    def test_it_compares_with_normal_norms_and_v2_e2es_final_saves_of_the_stage(self, spec, stage):
        references, _ = spec
        assert set(references) == {"normal_norm", "v2e2e"}
        assert references["normal_norm"].run_config.parts[-2:] == (stage.final_save, "run_config.yaml")
        assert references["normal_norm"].sha256 is None, "the reference run's own save needs no pin"
        assert references["v2e2e"].sha256 is not None, "a copy of an export's record is pinned"

    def test_every_lever_of_the_stages_posture_is_allowed_against_normal_norm_only(self, spec, stage):
        _, allowances = spec
        for lever in stage.levers:
            assert self.allowed_for(allowances, "normal_norm", lever), lever
            assert not self.allowed_for(allowances, "v2e2e", lever), lever

    def test_the_stages_own_fields_are_allowed_against_exactly_their_references(self, spec, stage):
        _, allowances = spec
        for key, references in stage.own.items():
            for reference in ("normal_norm", "v2e2e"):
                assert self.allowed_for(allowances, reference, key) == (reference in references), (reference, key)

    def test_the_identity_data_masking_and_normalisation_are_allowed_against_both(self, spec):
        _, allowances = spec
        for key in (*IDENTITY, "dataset.data_path", "dataset.path_to_cache", "token_masking.token_ids"):
            for reference in ("normal_norm", "v2e2e"):
                assert self.allowed_for(allowances, reference, key), (reference, key)
        for key in ("model.calculate_per_token_loss", "ddp.average_in_collective"):
            assert self.allowed_for(allowances, "v2e2e", key)

    def test_the_export_repairs_edits_are_allowed_against_v2_e2es_export_copy_only(self, spec):
        _, allowances = spec
        repaired = {"model.moe_experts_impl", "model.mamba_stack_spec._target_"}
        assert all("moe_experts_impl" in old or "stack_spec" in old for old, _ in RUN_CONFIG_EDITS)
        for key in repaired:
            assert self.allowed_for(allowances, "v2e2e", key)
            assert not self.allowed_for(allowances, "normal_norm", key)
