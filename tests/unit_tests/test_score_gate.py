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

"""Unit tests for scripts/telemetry/score_gate.py (pre-registered memory, speed and first-loss gates on runs'
scores).

Each score is the real ``score_run.score_log`` of a log made from the real iteration line of
``training_log_fixture.py`` and, where the run finished, the real ``[peak-memory]`` line
(``train_utils.format_peak_memory`` of ``summarise_peak_memory``'s summary of per-rank rows), written as
``score_run.py --json`` writes it.
"""

import json
from pathlib import Path

import pytest
import yaml
from scripts.telemetry import loss_parity
from scripts.telemetry import score_gate as sg
from scripts.telemetry.score_run import Workload, score_log

from megatron.bridge.training.utils.train_utils import format_peak_memory, summarise_peak_memory
from tests.unit_tests.test_loss_parity import over, write_run
from tests.unit_tests.training_log_fixture import iteration_line, sub_once, write_log


GPUS = 4
WINDOW = (1, 4)
WORKLOAD = Workload(
    config_path="<synthetic>", hf_config_path="<synthetic>", seq_length=8192, model_flops_per_token=1e9
)
GB = 10**9


def write_score(
    scores_dir: Path,
    name: str,
    step_ms: float,
    peak_rows: list[tuple[int, int, int]] | None,
    window: tuple[int, int] = WINDOW,
) -> None:
    """Score a four-iteration run of ``step_ms`` per step as ``<name>.score.json``; ``peak_rows`` are the
    ranks' (peak allocated bytes, peak reserved bytes, allocator retries), or None for a loop cut short."""
    run_dir = scores_dir / name
    run_dir.mkdir()
    lines = [iteration_line(i, step_ms, 6.0) for i in range(1, 5)]
    if peak_rows is not None:
        lines.append(format_peak_memory(summarise_peak_memory(peak_rows)))
    score = score_log(write_log(run_dir, lines), WORKLOAD, window, window, GPUS, 1000.0)
    (scores_dir / f"{name}.score.json").write_text(json.dumps(score.to_dict()))


def rows(allocated_gb: float, reserved_gb: float, retries: int = 0) -> list[tuple[int, int, int]]:
    """Four ranks, rank 2 at the peak."""
    low = (int(60 * GB), int(70 * GB), 0)
    return [low, low, (int(allocated_gb * GB), int(reserved_gb * GB), retries), low]


def write_spec(
    tmp_path: Path,
    memory: dict | None = None,
    speed: dict | None = None,
    first_loss: dict | None = None,
    loss_shift: dict | None = None,
) -> Path:
    path = tmp_path / "score_gate.yaml"
    raw = {}
    if memory is not None:
        raw["memory"] = memory
    if speed is not None:
        raw["speed"] = speed
    if first_loss is not None:
        raw["first_loss"] = first_loss
    if loss_shift is not None:
        raw["loss_shift"] = loss_shift
    path.write_text(yaml.safe_dump(raw))
    return path


MEMORY_GATE = {"fast_memory": {"score": "fast.score.json", "max_allocated_gb": 85.5, "max_alloc_retries": 0}}
SPEED_GATE = {
    "fast_speed": {
        "candidate": "fast.score.json",
        "reference": "as_is.score.json",
        "reference_s_per_iter": 6.0,
        "go_up_to_s": 4.5,
        "report_up_to_s": 5.25,
    }
}


def run(spec: Path, scores_dir: Path, capsys) -> tuple[int, dict[str, dict]]:
    status = sg.main(["--spec", str(spec), "--scores-dir", str(scores_dir), "--json"])
    return status, {result["gate"]: result for result in json.loads(capsys.readouterr().out)}


# --------------------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------------------


def test_a_peak_within_both_limits_passes_and_names_the_rank(tmp_path, capsys):
    write_score(tmp_path, "fast", 5000.0, rows(74.751, 88.12))
    status, results = run(write_spec(tmp_path, memory=MEMORY_GATE), tmp_path, capsys)
    assert status == 0
    assert results["fast_memory"]["outcome"] == "PASS"
    assert "max allocated 74.751 GB (rank 2)" in results["fast_memory"]["detail"]


@pytest.mark.parametrize(
    "peak, limit",
    [(rows(85.6, 90.0), "max allocated 85.600 GB"), (rows(80.0, 94.9, retries=3), "max allocator retries 3")],
)
def test_a_peak_over_either_limit_fails(tmp_path, capsys, peak, limit):
    write_score(tmp_path, "fast", 5000.0, peak)
    status, results = run(write_spec(tmp_path, memory=MEMORY_GATE), tmp_path, capsys)
    assert status == 1
    assert results["fast_memory"]["outcome"] == "FAIL" and limit in results["fast_memory"]["detail"]


def test_reserved_memory_is_reported_not_gated(tmp_path, capsys):
    """The caching allocator fills whatever is free, so reserved memory measures the cache, not demand."""
    write_score(tmp_path, "fast", 5000.0, rows(80.0, 101.0))
    status, results = run(write_spec(tmp_path, memory=MEMORY_GATE), tmp_path, capsys)
    assert status == 0 and "max reserved 101.000 GB, reported" in results["fast_memory"]["detail"]


def test_a_run_without_the_summary_fails(tmp_path, capsys):
    """No summary means the training loop was cut short: its peak is unknown, and unknown is not within a limit."""
    write_score(tmp_path, "fast", 5000.0, None)
    status, results = run(write_spec(tmp_path, memory=MEMORY_GATE), tmp_path, capsys)
    assert status == 1
    assert results["fast_memory"]["outcome"] == "FAIL" and "cut short" in results["fast_memory"]["detail"]


def test_a_score_that_cannot_be_read_is_not_evaluated_rather_than_a_crash(tmp_path, capsys):
    status, results = run(write_spec(tmp_path, memory=MEMORY_GATE), tmp_path, capsys)
    assert status == 2
    assert results["fast_memory"]["outcome"] == "NOT EVALUATED"
    assert results["fast_memory"]["detail"].startswith("FileNotFoundError")


def test_a_summary_over_other_ranks_than_the_runs_gpus_is_not_evaluated(tmp_path, capsys):
    write_score(tmp_path, "fast", 5000.0, rows(74.0, 88.0)[:3])
    status, results = run(write_spec(tmp_path, memory=MEMORY_GATE), tmp_path, capsys)
    assert status == 2 and "covers 3 ranks, not the run's 4 GPUs" in results["fast_memory"]["detail"]


# --------------------------------------------------------------------------------------
# Speed
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fast_ms, status, outcome, verdict",
    [
        (4000.0, 0, "PASS", "projected 4.000 s/iter = 4.000 s / 6.000 s x 6.0 s: go"),
        (5000.0, 0, "PASS", "projected 5.000 s/iter = 5.000 s / 6.000 s x 6.0 s: go, and report it (above 4.5 s)"),
        (5400.0, 1, "FAIL", "projected 5.400 s/iter = 5.400 s / 6.000 s x 6.0 s: above 5.25 s"),
    ],
)
def test_the_projected_step_time_decides(tmp_path, capsys, fast_ms, status, outcome, verdict):
    """The candidate's step time relative to the reference on the same nodes, times the reference posture's
    own step time where the decision applies."""
    write_score(tmp_path, "fast", fast_ms, None)
    write_score(tmp_path, "as_is", 6000.0, None)
    got_status, results = run(write_spec(tmp_path, speed=SPEED_GATE), tmp_path, capsys)
    assert got_status == status
    assert (results["fast_speed"]["outcome"], results["fast_speed"]["detail"]) == (outcome, verdict)


def test_scores_over_different_windows_are_not_evaluated(tmp_path, capsys):
    write_score(tmp_path, "fast", 4000.0, None, window=(2, 4))
    write_score(tmp_path, "as_is", 6000.0, None)
    status, results = run(write_spec(tmp_path, speed=SPEED_GATE), tmp_path, capsys)
    assert status == 2 and "differ" in results["fast_speed"]["detail"]


# --------------------------------------------------------------------------------------
# The spec and the command line
# --------------------------------------------------------------------------------------


def test_a_failing_gate_decides_whatever_the_other_gates_outcomes(tmp_path, capsys):
    """fast's memory fails while as_is's score is missing, so the speed gate cannot run: exit 1, not 2."""
    write_score(tmp_path, "fast", 4000.0, rows(90.0, 95.0))
    status, results = run(write_spec(tmp_path, memory=MEMORY_GATE, speed=SPEED_GATE), tmp_path, capsys)
    assert status == 1
    assert {name: r["outcome"] for name, r in results.items()} == {
        "fast_memory": "FAIL",
        "fast_speed": "NOT EVALUATED",
    }


def test_one_gate_can_be_evaluated_alone(tmp_path, capsys):
    write_score(tmp_path, "fast", 4000.0, rows(74.0, 88.0))
    spec = write_spec(tmp_path, memory=MEMORY_GATE, speed=SPEED_GATE)
    assert sg.main(["--spec", str(spec), "--scores-dir", str(tmp_path), "--gate", "fast_memory"]) == 0
    assert capsys.readouterr().out.startswith("gate fast_memory (memory): PASS (max allocated 74.000 GB (rank 2)")


@pytest.mark.parametrize(
    "memory, speed, message",
    [
        (MEMORY_GATE, {"fast_memory": SPEED_GATE["fast_speed"]}, "defined twice"),
        (None, {"s": {**SPEED_GATE["fast_speed"], "reference": "fast.score.json"}}, "with itself"),
        (None, {"s": {**SPEED_GATE["fast_speed"], "go_up_to_s": 6.0}}, "go_up_to_s above report_up_to_s"),
        ({}, None, "defines no gates"),
        (None, None, "defines no gates"),
    ],
)
def test_a_spec_whose_gates_cannot_run_as_written_is_refused(tmp_path, memory, speed, message):
    with pytest.raises(ValueError, match=message):
        sg.load_score_gates(write_spec(tmp_path, memory=memory, speed=speed))


# --------------------------------------------------------------------------------------
# First loss
# --------------------------------------------------------------------------------------

FIRST_LOSS_GATE = {
    "fast_first_loss": {"candidate": "fast.score.json", "reference": "as_is.score.json", "tolerance": 0.005}
}


def write_first_loss_score(scores_dir: Path, name: str, first_iteration: int, first_loss: float | None) -> None:
    """Score a run logging iterations ``first_iteration``-4, the first with lm loss ``first_loss`` (None: the field
    absent, as the log prints a non-finite loss with the loss check off)."""
    run_dir = scores_dir / name
    run_dir.mkdir()
    lines = [iteration_line(i, 5000.0, 6.0) for i in range(first_iteration, 5)]
    if first_loss is None:
        lines[0] = sub_once(r"lm loss: [\dE.+-]+ \| ", "", lines[0])
    else:
        lines[0] = iteration_line(first_iteration, 5000.0, first_loss)
    score = score_log(write_log(run_dir, lines), WORKLOAD, (first_iteration, 4), (4, 4), GPUS, 1000.0)
    (scores_dir / f"{name}.score.json").write_text(json.dumps(score.to_dict()))


@pytest.mark.parametrize("fast_loss, status, outcome", [(6.003, 0, "PASS"), (6.006, 1, "FAIL")])
def test_the_first_iterations_loss_must_agree_within_the_tolerance(tmp_path, capsys, fast_loss, status, outcome):
    write_first_loss_score(tmp_path, "fast", 1, fast_loss)
    write_first_loss_score(tmp_path, "as_is", 1, 6.0)
    got_status, results = run(write_spec(tmp_path, first_loss=FIRST_LOSS_GATE), tmp_path, capsys)
    assert (got_status, results["fast_first_loss"]["outcome"]) == (status, outcome)
    assert results["fast_first_loss"]["detail"].startswith(f"iteration 1: lm loss {fast_loss:.6f} against 6.000000")


def test_a_candidate_with_no_first_loss_fails(tmp_path, capsys):
    """With the loss check off a non-finite loss is not logged at all; a missing loss is not a pass."""
    write_first_loss_score(tmp_path, "fast", 1, None)
    write_first_loss_score(tmp_path, "as_is", 1, 6.0)
    status, results = run(write_spec(tmp_path, first_loss=FIRST_LOSS_GATE), tmp_path, capsys)
    assert status == 1 and "logged no lm loss at iteration 1" in results["fast_first_loss"]["detail"]


def test_runs_starting_at_different_iterations_are_not_evaluated(tmp_path, capsys):
    write_first_loss_score(tmp_path, "fast", 2, 6.0)
    write_first_loss_score(tmp_path, "as_is", 1, 6.0)
    status, results = run(write_spec(tmp_path, first_loss=FIRST_LOSS_GATE), tmp_path, capsys)
    assert status == 2 and "first logged iterations differ: 2 against 1" in results["fast_first_loss"]["detail"]


def test_a_first_loss_gate_comparing_a_run_with_itself_is_refused(tmp_path):
    gate = {"g": {**FIRST_LOSS_GATE["fast_first_loss"], "reference": "fast.score.json"}}
    with pytest.raises(ValueError, match="with itself"):
        sg.load_score_gates(write_spec(tmp_path, first_loss=gate))


def test_an_unknown_gate_kind_is_refused(tmp_path):
    path = tmp_path / "score_gate.yaml"
    path.write_text(yaml.safe_dump({"memroy": MEMORY_GATE}))
    with pytest.raises(ValueError, match="unknown gate kinds"):
        sg.load_score_gates(path)


# --------------------------------------------------------------------------------------
# Loss shift
# --------------------------------------------------------------------------------------


LOSS_SHIFT_GATE = {
    "fast_shift": {
        "report": "parity_band.json",
        "offset_low": -0.001,
        "offset_high": 0.003,
        "rise_windows": 2,
        "max_rise": 0.001,
    }
}


def write_band_report(tmp_path: Path, scores_dir: Path, **candidate) -> None:
    """A real band report over iterations 1-60 in six windows of 10, written as ``band --json`` writes it: two
    references 0.002 apart (their mean sits 0.001 above the fixture) and a candidate written with ``candidate``."""
    refs = [write_run(tmp_path, "ref_a"), write_run(tmp_path, "ref_b", loss_offset=over(1, 60, 0.002))]
    runs = [loss_parity.load_trajectory(path, (1, 60), use_wandb=False) for path in refs]
    mine = loss_parity.load_trajectory(write_run(tmp_path, "cand", **candidate), (1, 60), use_wandb=False)
    report = loss_parity.band_test(runs, [mine], window=10)
    (scores_dir / "parity_band.json").write_text(json.dumps(report.to_dict()))


def test_a_steady_offset_inside_the_range_passes(tmp_path, capsys):
    write_band_report(tmp_path, tmp_path, loss_offset=over(1, 60, 0.003))
    status, results = run(write_spec(tmp_path, loss_shift=LOSS_SHIFT_GATE), tmp_path, capsys)
    assert status == 0 and results["fast_shift"]["outcome"] == "PASS"
    assert "offsets +0.0020 to +0.0020 over 6 windows of 1-60" in results["fast_shift"]["detail"]


def test_an_offset_outside_the_range_in_any_window_fails(tmp_path, capsys):
    write_band_report(tmp_path, tmp_path, loss_offset={**over(1, 50, 0.002), **over(51, 60, 0.006)})
    status, results = run(write_spec(tmp_path, loss_shift=LOSS_SHIFT_GATE), tmp_path, capsys)
    assert status == 1 and results["fast_shift"]["outcome"] == "FAIL"
    assert "windows from [51] outside the range" in results["fast_shift"]["detail"]


def test_an_offset_that_grows_inside_the_range_fails(tmp_path, capsys):
    offsets = {**over(1, 20, 0.001), **over(21, 40, 0.002), **over(41, 60, 0.0035)}
    write_band_report(tmp_path, tmp_path, loss_offset=offsets)
    status, results = run(write_spec(tmp_path, loss_shift=LOSS_SHIFT_GATE), tmp_path, capsys)
    assert status == 1 and results["fast_shift"]["outcome"] == "FAIL"
    assert results["fast_shift"]["detail"].endswith("the offset grows")


@pytest.mark.parametrize("candidate", [{"nan": {30: 1}}, {"learning_rate": {30: 1.0e-3}}], ids=["nan", "schedule"])
def test_a_candidate_off_the_references_schedule_or_counting_a_nan_fails(tmp_path, capsys, candidate):
    write_band_report(tmp_path, tmp_path, **candidate)
    status, results = run(write_spec(tmp_path, loss_shift=LOSS_SHIFT_GATE), tmp_path, capsys)
    assert status == 1 and results["fast_shift"]["outcome"] == "FAIL"


def test_a_report_with_too_few_windows_for_the_rise_is_not_evaluated(tmp_path, capsys):
    write_band_report(tmp_path, tmp_path, loss_offset=over(1, 60, 0.003))
    gate = {"fast_shift": {**LOSS_SHIFT_GATE["fast_shift"], "rise_windows": 4}}
    status, results = run(write_spec(tmp_path, loss_shift=gate), tmp_path, capsys)
    assert status == 2 and results["fast_shift"]["outcome"] == "NOT EVALUATED"


def test_a_missing_report_is_not_evaluated_rather_than_a_crash(tmp_path, capsys):
    status, results = run(write_spec(tmp_path, loss_shift=LOSS_SHIFT_GATE), tmp_path, capsys)
    assert status == 2 and results["fast_shift"]["outcome"] == "NOT EVALUATED"


def test_a_report_of_more_than_one_candidate_is_not_evaluated(tmp_path, capsys):
    """The gate judges one candidate; a band over several cannot say whose offsets it bounds."""
    refs = [write_run(tmp_path, "ref_a"), write_run(tmp_path, "ref_b", loss_offset=over(1, 60, 0.002))]
    runs = [loss_parity.load_trajectory(path, (1, 60), use_wandb=False) for path in refs]
    cand = loss_parity.load_trajectory(write_run(tmp_path, "cand"), (1, 60), use_wandb=False)
    report = loss_parity.band_test(runs, [cand, cand], window=10)
    (tmp_path / "parity_band.json").write_text(json.dumps(report.to_dict()))
    status, results = run(write_spec(tmp_path, loss_shift=LOSS_SHIFT_GATE), tmp_path, capsys)
    assert status == 2 and results["fast_shift"]["outcome"] == "NOT EVALUATED"
    assert "holds 2 candidates, not one" in results["fast_shift"]["detail"]


@pytest.mark.parametrize("edit", [{"offset_low": 0.004}, {"rise_windows": 0}], ids=["empty-range", "no-rise-window"])
def test_a_loss_shift_gate_with_an_empty_range_or_no_rise_window_is_refused(tmp_path, edit):
    gate = {"g": {**LOSS_SHIFT_GATE["fast_shift"], **edit}}
    with pytest.raises(ValueError, match="empty offset range or no rise window"):
        sg.load_score_gates(write_spec(tmp_path, loss_shift=gate))


# --------------------------------------------------------------------------------------
# Every kind's fields are checked
# --------------------------------------------------------------------------------------

GATES_BY_KIND = {
    "memory": MEMORY_GATE,
    "speed": SPEED_GATE,
    "first_loss": FIRST_LOSS_GATE,
    "loss_shift": LOSS_SHIFT_GATE,
}


@pytest.mark.parametrize("kind", sorted(GATES_BY_KIND))
@pytest.mark.parametrize("case", ["unknown", "missing"])
def test_a_gate_with_an_unknown_or_a_missing_field_is_refused(tmp_path, kind, case):
    """An unknown field, such as a misspelt threshold, would otherwise not be applied, and a missing one would end the
    load in a bare KeyError."""
    ((name, gate),) = GATES_BY_KIND[kind].items()
    dropped = sorted(gate)[0]
    if case == "unknown":
        gate, problem = {**gate, "comment": "x"}, r"unknown keys \['comment'\], missing keys \[\]"
    else:
        gate = {field: value for field, value in gate.items() if field != dropped}
        problem = rf"unknown keys \[\], missing keys \['{dropped}'\]"
    with pytest.raises(ValueError, match=rf"{kind} gate {name}: expected exactly the keys .*; {problem}"):
        sg.load_score_gates(write_spec(tmp_path, **{kind: {name: gate}}))


# --------------------------------------------------------------------------------------
# A value that is not a finite number is never a pass
# --------------------------------------------------------------------------------------


def edit_score(scores_dir: Path, name: str, edit) -> None:
    """Apply ``edit`` to the parsed ``<name>.score.json``, as a writer that let a NaN through would leave it."""
    path = scores_dir / f"{name}.score.json"
    score = json.loads(path.read_text())
    edit(score)
    path.write_text(json.dumps(score))


def test_a_nan_peak_is_not_evaluated(tmp_path, capsys):
    write_score(tmp_path, "fast", 5000.0, rows(74.0, 88.0))
    edit_score(tmp_path, "fast", lambda score: score["peak_memory_across_ranks"].update(max_allocated_gb=float("nan")))
    status, results = run(write_spec(tmp_path, memory=MEMORY_GATE), tmp_path, capsys)
    assert status == 2 and results["fast_memory"]["outcome"] == "NOT EVALUATED"
    assert results["fast_memory"]["detail"] == (
        "ValueError: fast.score.json max_allocated_gb is nan, not a finite number"
    )


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("fast", float("nan"), "fast.score.json mean_step_s is nan, not a finite number"),
        ("as_is", 0.0, "as_is.score.json mean_step_s is 0.0, not a positive step time"),
    ],
)
def test_a_step_time_that_cannot_be_divided_is_not_evaluated(tmp_path, capsys, name, value, message):
    write_score(tmp_path, "fast", 4000.0, None)
    write_score(tmp_path, "as_is", 6000.0, None)
    edit_score(tmp_path, name, lambda score: score.update(mean_step_s=value))
    status, results = run(write_spec(tmp_path, speed=SPEED_GATE), tmp_path, capsys)
    assert status == 2 and results["fast_speed"]["detail"] == f"ValueError: {message}"


def test_a_nan_first_loss_is_not_evaluated(tmp_path, capsys):
    """The log prints a NaN loss as ``nan`` when the loss check is on but did not stop the run."""
    write_first_loss_score(tmp_path, "fast", 1, float("nan"))
    write_first_loss_score(tmp_path, "as_is", 1, 6.0)
    status, results = run(write_spec(tmp_path, first_loss=FIRST_LOSS_GATE), tmp_path, capsys)
    assert status == 2
    assert results["fast_first_loss"]["detail"] == (
        "ValueError: fast.score.json's lm loss at iteration 1 is nan, not a finite number"
    )


def test_a_nan_window_mean_is_not_evaluated(tmp_path, capsys):
    write_band_report(tmp_path, tmp_path, loss_offset=over(1, 60, 0.003))
    report = json.loads((tmp_path / "parity_band.json").read_text())
    (loss,) = [band for band in report["metrics"] if band["metric"] == loss_parity.VERDICT_METRIC]
    loss["windows"][2]["candidate_means"][0] = float("nan")
    (tmp_path / "parity_band.json").write_text(json.dumps(report))
    status, results = run(write_spec(tmp_path, loss_shift=LOSS_SHIFT_GATE), tmp_path, capsys)
    assert status == 2
    assert "parity_band.json's lm-loss offset in the window from 21 is nan" in results["fast_shift"]["detail"]
