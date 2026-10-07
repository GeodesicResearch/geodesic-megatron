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

"""Unit tests for scripts/telemetry/loss_gate.py (pre-registered loss gates over loss_parity's band test).

The runs are the real training-log excerpt of ``training_log_fixture.py``, copied and rewritten with the
loss-parity tests' own helpers, so the gate runs on the format the bridge prints.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from tests.unit_tests.test_loss_parity import over, write_run
from tests.unit_tests.training_log_fixture import REPO_ROOT


@pytest.fixture(scope="module")
def lg():
    """Import the real script by path (scripts/telemetry/ is not an installed package)."""
    spec = importlib.util.spec_from_file_location("loss_gate", REPO_ROOT / "scripts" / "telemetry" / "loss_gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_spec(tmp_path: Path, references: dict[str, Path], gates: dict[str, dict]) -> Path:
    path = tmp_path / "gate.yaml"
    path.write_text(
        yaml.safe_dump({"wandb": False, "references": {k: str(v) for k, v in references.items()}, "gates": gates})
    )
    return path


def gate(references, delta, iterations=(1, 60), window=20) -> dict:
    return {"references": list(references), "iterations": list(iterations), "window": window, "lm_loss_delta": delta}


@pytest.fixture()
def refs(tmp_path) -> dict[str, Path]:
    """Two reference runs whose losses differ by 0.01 in every window, so the band width is exactly 0.01."""
    return {"a": write_run(tmp_path, "ref_a"), "b": write_run(tmp_path, "ref_b", loss_offset=over(1, 60, 0.01))}


def test_a_candidate_inside_the_band_passes(lg, tmp_path, refs, capsys):
    spec = write_spec(tmp_path, refs, {"L": gate(refs, 0.01)})
    candidate = write_run(tmp_path, "cand", loss_offset=over(1, 60, 0.005))
    assert lg.main(["--spec", str(spec), "--candidate", str(candidate)]) == 0
    assert "gate L: PASS" in capsys.readouterr().out


def test_a_candidate_outside_the_band_fails(lg, tmp_path, refs, capsys):
    spec = write_spec(tmp_path, refs, {"L": gate(refs, 0.01)})
    candidate = write_run(tmp_path, "cand", loss_offset=over(41, 60, 0.05))
    assert lg.main(["--spec", str(spec), "--candidate", str(candidate)]) == 1
    assert "gate L: FAIL" in capsys.readouterr().out


def test_a_band_width_other_than_the_pre_registered_one_is_not_evaluated(lg, tmp_path, refs, capsys):
    """A different width means a different reference set from the one the gate was calibrated on, which
    decides nothing about the candidate either way."""
    spec = write_spec(tmp_path, refs, {"L": gate(refs, 0.02)})
    candidate = write_run(tmp_path, "cand")
    assert lg.main(["--spec", str(spec), "--candidate", str(candidate)]) == 2
    out = capsys.readouterr().out
    assert "gate L: NOT EVALUATED" in out and "not the pre-registered 0.020000" in out


def test_a_candidate_that_is_a_reference_is_not_evaluated(lg, tmp_path, refs):
    """A run always lies inside a band it helped draw."""
    spec = write_spec(tmp_path, refs, {"L": gate(refs, 0.01)})
    assert lg.main(["--spec", str(spec), "--candidate", str(refs["b"])]) == 2
    (result,) = [lg.evaluate_gate(lg.load_gate_spec(spec), "L", refs["b"])]
    assert result.outcome == lg.NOT_EVALUATED and "is one of the references" in result.reason


def test_a_candidate_that_has_not_reached_the_range_is_not_evaluated(lg, tmp_path, refs):
    """A run gated before it has logged the whole range is not yet decidable, which is not a failure."""
    spec = write_spec(tmp_path, refs, {"L": gate(refs, 0.01)})
    candidate = write_run(tmp_path, "cand", drop=set(range(41, 61)))
    result = lg.evaluate_gate(lg.load_gate_spec(spec), "L", candidate)
    assert result.outcome == lg.NOT_EVALUATED and result.report is None


def test_one_gate_can_be_evaluated_alone(lg, tmp_path, refs, capsys):
    """An earlier gate decides as soon as the run reaches its range, before a later one can run."""
    spec = write_spec(tmp_path, refs, {"early": gate(refs, 0.01, (1, 40)), "late": gate(refs, 0.01)})
    candidate = write_run(tmp_path, "cand", drop=set(range(41, 61)))
    assert lg.main(["--spec", str(spec), "--candidate", str(candidate), "--gate", "early", "--json"]) == 0
    (result,) = json.loads(capsys.readouterr().out)
    assert (result["gate"], result["outcome"]) == ("early", "PASS")
    assert lg.main(["--spec", str(spec), "--candidate", str(candidate)]) == 2


def test_a_failing_gate_decides_whatever_the_other_gates_outcomes(lg, tmp_path, refs, capsys):
    """One failing gate stops the run, so a later gate that cannot run yet must not hide it behind exit 2."""
    spec = write_spec(tmp_path, refs, {"early": gate(refs, 0.01, (1, 40)), "late": gate(refs, 0.01)})
    candidate = write_run(tmp_path, "cand", loss_offset=over(21, 40, 0.05), drop=set(range(41, 61)))
    assert lg.main(["--spec", str(spec), "--candidate", str(candidate)]) == 1
    out = capsys.readouterr().out
    assert "gate early: FAIL" in out and "gate late: NOT EVALUATED" in out


def test_a_reference_that_cannot_be_read_is_not_evaluated_rather_than_a_crash(lg, tmp_path, refs):
    """A crash exits 1, the status of a FAIL, which the policy answers by restarting the stage."""
    spec = write_spec(tmp_path, refs, {"L": gate(refs, 0.01)})
    refs["b"].unlink()
    candidate = write_run(tmp_path, "cand")
    result = lg.evaluate_gate(lg.load_gate_spec(spec), "L", candidate)
    assert result.outcome == lg.NOT_EVALUATED and result.reason.startswith("FileNotFoundError")
    assert lg.main(["--spec", str(spec), "--candidate", str(candidate)]) == 2


@pytest.mark.parametrize(
    "gates, message",
    [
        ({"L": gate(["a", "c"], 0.01)}, "unknown references"),
        ({"L": gate(["a", "a"], 0.01)}, "more than once"),
        ({"L": gate(["a"], 0.01)}, "at least two references"),
        ({}, "defines no gates"),
    ],
)
def test_a_spec_whose_gates_cannot_run_as_written_is_refused(lg, tmp_path, refs, gates, message):
    with pytest.raises(ValueError, match=message):
        lg.load_gate_spec(write_spec(tmp_path, refs, gates))


def tolerance_gate(references, spread, tolerance, iterations=(1, 60), window=20) -> dict:
    return {**gate(references, spread, iterations, window), "lm_loss_tolerance": tolerance}


@pytest.mark.parametrize("offset, outcome", [(0.025, "PASS"), (-0.015, "PASS"), (0.035, "FAIL"), (-0.025, "FAIL")])
def test_a_tolerance_gate_holds_the_candidate_within_a_fixed_distance_on_both_sides(
    lg, tmp_path, refs, offset, outcome
):
    """The references span [0, +0.01] around the fixture's loss; a 0.02 tolerance admits [-0.02, +0.03]."""
    spec = lg.load_gate_spec(write_spec(tmp_path, refs, {"L": tolerance_gate(refs, 0.01, 0.02)}))
    candidate = write_run(tmp_path, "cand", loss_offset=over(41, 60, offset))
    result = lg.evaluate_gate(spec, "L", candidate)
    assert result.outcome == outcome
    (loss,) = [band for band in result.report.metrics if band.metric == "lm loss"]
    assert loss.delta == 0.02
    assert loss.candidates[0].windows_outside == (() if outcome == "PASS" else (41,))


def test_a_tolerance_gate_still_requires_the_references_schedule(lg, tmp_path, refs):
    spec = lg.load_gate_spec(write_spec(tmp_path, refs, {"L": tolerance_gate(refs, 0.01, 0.02)}))
    candidate = write_run(tmp_path, "cand", learning_rate={44: 5e-4})
    result = lg.evaluate_gate(spec, "L", candidate)
    assert result.outcome == "FAIL"
    (verdict,) = result.report.verdicts
    assert verdict.loss_inside and verdict.first_learning_rate_difference.iteration == 44


def test_a_tolerance_gate_checks_its_references_are_the_frozen_ones(lg, tmp_path, refs):
    """A fixed band drawn around the wrong runs would decide on them silently."""
    spec = lg.load_gate_spec(write_spec(tmp_path, refs, {"L": tolerance_gate(refs, 0.03, 0.02)}))
    result = lg.evaluate_gate(spec, "L", write_run(tmp_path, "cand"))
    assert (
        result.outcome == lg.NOT_EVALUATED and "spread is 0.010000, not the pre-registered 0.030000" in result.reason
    )


def test_every_gate_pre_registers_its_references_spread(lg, tmp_path, refs):
    unregistered = {k: v for k, v in tolerance_gate(refs, 0.01, 0.02).items() if k != "lm_loss_delta"}
    with pytest.raises(ValueError, match="must pre-register"):
        lg.load_gate_spec(write_spec(tmp_path, refs, {"L": unregistered}))


def test_a_spec_that_names_one_log_twice_is_refused(lg, tmp_path, refs):
    """One run under two names would make a band of a run against itself: zero width, every candidate out."""
    spec = write_spec(tmp_path, {**refs, "c": refs["a"]}, {"L": gate(["a", "c"], 0.0)})
    with pytest.raises(ValueError, match="named more than once"):
        lg.load_gate_spec(spec)
