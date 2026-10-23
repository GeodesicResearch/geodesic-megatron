# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/telemetry/score_gate.py's gates on one run's own record: its token-masking counts against a prediction
(``count_prediction``) and against bounds (``count_bounds``), a band report's verdict (``loss_band``), a watch
spec's stop conditions (``stops``), and the masking invariants of a segment resumed from a save (``masking_log``
with ``first``).

The logs are built from the real iteration line and the bridge's own counts line (``training_log_fixture.py``), the
predictions are written by ``predict_masked_counts.py``'s own writer from the bridge's own counts dataclass, and the
band reports are ``loss_parity.py``'s own, made from such logs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from scripts.data import predict_masked_counts
from scripts.telemetry import loss_parity
from scripts.telemetry import score_gate as sg

from megatron.bridge.training.token_masking.hook import TokenMaskingCounts
from tests.unit_tests.training_log_fixture import (
    iteration_line,
    sub_once,
    token_masking_banner,
    token_masking_counts_line,
    write_log,
)


WINDOWS = 8  # windows in a global batch
WINDOW = 1000  # target positions in a window


def masked_counts(iteration: int, listed: int | None = None, trainable: int | None = None) -> TokenMaskingCounts:
    """An iteration of a masking run: its listed targets, all trainable in the dataset's mask, are masked."""
    listed = 100 + iteration if listed is None else listed
    positions = WINDOWS * WINDOW
    trainable = positions - listed if trainable is None else trainable
    return TokenMaskingCounts(listed, listed, listed, 0, trainable, positions)


def counts_lines(counts: dict[int, TokenMaskingCounts | None], **lines) -> list[str]:
    """The log lines of the iterations of ``counts``: each one's iteration line, and its counts line unless it is
    None."""
    log = []
    for iteration, record in counts.items():
        if record is not None:
            log.append(token_masking_counts_line(iteration, **asdict(record)))
        log.append(lines.get(str(iteration), iteration_line(iteration, 3000.0, 2.5)))
    return log


def counts_log(directory: Path, name: str, counts: dict[int, TokenMaskingCounts | None], **lines) -> Path:
    """A log of the iterations of ``counts`` (``counts_lines``)."""
    return write_log(directory, counts_lines(counts, **lines), name)


def write_prediction(directory: Path, name: str, counts: dict[int, TokenMaskingCounts]) -> Path:
    """A prediction of ``counts``, written by ``predict_masked_counts.py``'s own writer."""
    path = directory / name
    predict_masked_counts.write_prediction(path, {"config": "/configs/run.yaml", "gpus": 64}, counts.items())
    return path


def judge(directory: Path, capsys, **spec) -> tuple[int, str, str]:
    """Evaluate a spec of one gate over ``directory``: the exit status, the gate's outcome and its detail."""
    path = directory / "gate.yaml"
    path.write_text(yaml.safe_dump(spec))
    status = sg.main(["--spec", str(path), "--scores-dir", str(directory), "--json"])
    (result,) = json.loads(capsys.readouterr().out)
    return status, result["outcome"], result["detail"]


def refused(tmp_path: Path, message: str, **spec) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match=message):
        sg.load_score_gates(path)


# --------------------------------------------------------------------------------------
# count_prediction
# --------------------------------------------------------------------------------------

PREDICTION = {
    "count_prediction": {"counts": {"log": "run.log", "prediction": "predicted.jsonl", "first": 1, "last": 6}}
}


def prediction_gate(**edits) -> dict:
    return {"count_prediction": {"counts": PREDICTION["count_prediction"]["counts"] | edits}}


class TestCountPrediction:
    def test_a_log_whose_every_count_is_the_predicted_one_passes(self, tmp_path, capsys):
        counts = {iteration: masked_counts(iteration) for iteration in range(1, 7)}
        counts_log(tmp_path, "run.log", counts)
        write_prediction(tmp_path, "predicted.jsonl", counts)
        status, outcome, detail = judge(tmp_path, capsys, **PREDICTION)
        assert (status, outcome) == (0, "PASS")
        assert "predicted for /configs/run.yaml on 64 GPUs" in detail and "every count equal" in detail

    def test_one_count_off_by_one_at_one_iteration_fails_naming_it(self, tmp_path, capsys):
        counts = {iteration: masked_counts(iteration) for iteration in range(1, 7)}
        counts_log(tmp_path, "run.log", counts | {4: masked_counts(4, trainable=WINDOWS * WINDOW - 105)})
        write_prediction(tmp_path, "predicted.jsonl", counts)
        status, outcome, detail = judge(tmp_path, capsys, **PREDICTION)
        assert (status, outcome) == (1, "FAIL")
        assert "1 iterations differ from the prediction, from 4 (trainable 7895 against the predicted 7896)" in detail

    def test_a_resumed_segment_is_judged_over_its_own_iterations(self, tmp_path, capsys):
        counts = {iteration: masked_counts(iteration) for iteration in range(1, 7)}
        counts_log(tmp_path, "resumed.log", {iteration: counts[iteration] for iteration in (4, 5, 6)})
        write_prediction(tmp_path, "predicted.jsonl", counts)
        resumed = prediction_gate(log="resumed.log", first=4)
        assert judge(tmp_path, capsys, **resumed)[:2] == (0, "PASS")
        # The same log read from iteration 1 does not cover the range, which is not a failure of its counts.
        status, outcome, detail = judge(tmp_path, capsys, **prediction_gate(log="resumed.log"))
        assert (status, outcome) == (2, "NOT EVALUATED") and "iterations" in detail

    def test_an_iteration_without_its_counts_line_fails(self, tmp_path, capsys):
        counts = {iteration: masked_counts(iteration) for iteration in range(1, 7)}
        counts_log(tmp_path, "run.log", counts | {3: None})
        write_prediction(tmp_path, "predicted.jsonl", counts)
        status, outcome, detail = judge(tmp_path, capsys, **PREDICTION)
        assert (status, outcome) == (1, "FAIL") and "absent at 1 iterations, from 3" in detail

    def test_a_log_that_prints_no_counts_at_all_is_not_evaluated(self, tmp_path, capsys):
        counts = {iteration: masked_counts(iteration) for iteration in range(1, 7)}
        counts_log(tmp_path, "run.log", dict.fromkeys(counts))
        write_prediction(tmp_path, "predicted.jsonl", counts)
        status, outcome, detail = judge(tmp_path, capsys, **PREDICTION)
        assert (status, outcome) == (2, "NOT EVALUATED") and "prints no [token-masking-counts] line" in detail

    def test_a_prediction_with_no_listed_target_is_not_evaluated_although_every_count_is_equal(self, tmp_path, capsys):
        counts = {iteration: masked_counts(iteration, listed=0) for iteration in range(1, 7)}
        counts_log(tmp_path, "run.log", counts)
        write_prediction(tmp_path, "predicted.jsonl", counts)
        status, outcome, detail = judge(tmp_path, capsys, **PREDICTION)
        assert (status, outcome) == (2, "NOT EVALUATED") and "equal counts would show nothing" in detail

    def test_a_prediction_that_does_not_cover_the_range_is_not_evaluated(self, tmp_path, capsys):
        counts = {iteration: masked_counts(iteration) for iteration in range(1, 7)}
        counts_log(tmp_path, "run.log", counts)
        write_prediction(tmp_path, "predicted.jsonl", {i: counts[i] for i in range(1, 5)})
        status, outcome, detail = judge(tmp_path, capsys, **PREDICTION)
        assert (status, outcome) == (2, "NOT EVALUATED") and "predicts no counts for 2 iterations, from 5" in detail

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ('{"iteration": 1}\n', "does not open with a prediction's inputs record"),
            ('{"inputs": {}}\n{"iteration": 1, "listed": 1}\n', "not an iteration's counts"),
        ],
    )
    def test_a_file_that_is_not_a_prediction_is_not_evaluated(self, tmp_path, capsys, text, message):
        counts_log(tmp_path, "run.log", {iteration: masked_counts(iteration) for iteration in range(1, 7)})
        (tmp_path / "predicted.jsonl").write_text(text)
        status, outcome, detail = judge(tmp_path, capsys, **PREDICTION)
        assert (status, outcome) == (2, "NOT EVALUATED") and message in detail

    def test_a_range_that_is_not_one_is_refused(self, tmp_path):
        refused(tmp_path, "1 <= first <= last", **prediction_gate(first=7))
        refused(tmp_path, "1 <= first <= last", **prediction_gate(first=0))
        refused(tmp_path, "unknown keys \\['iterations'\\]", **prediction_gate(iterations=6))


# --------------------------------------------------------------------------------------
# masking_log over a segment resumed from a save
# --------------------------------------------------------------------------------------

MASKED_ID = 500
HOSTS = ("nid001", "nid002")


def segment_gate(**edits) -> dict:
    gate = {"log": "resumed.log", "enabled": True, "token_ids": [MASKED_ID], "nodes": 2, "first": 4, "iterations": 6}
    return {"masking_log": {"segment": gate | edits}}


def resumed_log(directory: Path, counts: dict[int, TokenMaskingCounts]) -> None:
    """The log of a segment that resumed a masking run: each node's banner, then the iterations of ``counts``."""
    banners = [token_masking_banner(True, [MASKED_ID], host, rank * 4) for rank, host in enumerate(HOSTS)]
    write_log(directory, banners + counts_lines(counts), "resumed.log")


class TestMaskingLogOfAResumedSegment:
    def test_a_segment_is_judged_from_its_first_iteration(self, tmp_path, capsys):
        resumed_log(tmp_path, {iteration: masked_counts(iteration) for iteration in (4, 5, 6)})
        status, outcome, detail = judge(tmp_path, capsys, **segment_gate())
        assert (status, outcome) == (0, "PASS")
        assert "2 banners from 2 hosts; iterations 4-6" in detail

    def test_a_listed_target_trained_in_the_segment_fails(self, tmp_path, capsys):
        counts = {iteration: masked_counts(iteration) for iteration in (4, 5, 6)}
        trained = TokenMaskingCounts(105, 105, 104, 1, WINDOWS * WINDOW - 104, WINDOWS * WINDOW)
        resumed_log(tmp_path, counts | {5: trained})
        status, outcome, detail = judge(tmp_path, capsys, **segment_gate())
        assert (status, outcome) == (1, "FAIL") and "fails at 1 iterations, from 5" in detail

    def test_without_its_first_iteration_the_segment_is_not_evaluated(self, tmp_path, capsys):
        # Read from iteration 1, the segment's log lacks the iterations before the save: nothing about its masking.
        resumed_log(tmp_path, {iteration: masked_counts(iteration) for iteration in (4, 5, 6)})
        gate = segment_gate()
        del gate["masking_log"]["segment"]["first"]
        assert judge(tmp_path, capsys, **gate)[:2] == (2, "NOT EVALUATED")

    def test_a_first_iteration_outside_the_range_is_refused(self, tmp_path):
        for first in (0, 7, True, "4"):
            refused(tmp_path, "first must be an iteration from 1 to iterations", **segment_gate(first=first))


# --------------------------------------------------------------------------------------
# count_bounds
# --------------------------------------------------------------------------------------


def bounds_gate(**edits) -> dict:
    gate = {"log": "run.log", "first": 1, "last": 4, "min": {"trainable": 1}, "max": {"trainable": WINDOWS - 1}}
    return {"count_bounds": {"windows": gate | edits}}


class TestCountBounds:
    def test_fewer_trainable_targets_than_windows_at_every_iteration_passes(self, tmp_path, capsys):
        # At most seven trainable targets among eight windows: some window of every iteration trained none.
        counts = {i: masked_counts(i, listed=WINDOWS * WINDOW - i, trainable=i) for i in range(1, 5)}
        counts_log(tmp_path, "run.log", counts)
        status, outcome, detail = judge(tmp_path, capsys, **bounds_gate())
        assert (status, outcome) == (0, "PASS")
        assert "trainable 1 to 4, at least 1" in detail and "trainable 1 to 4, at most 7" in detail

    def test_an_iteration_above_its_maximum_fails_naming_it(self, tmp_path, capsys):
        counts = {i: masked_counts(i, listed=WINDOWS * WINDOW - 5, trainable=5) for i in range(1, 5)}
        counts_log(tmp_path, "run.log", counts | {3: masked_counts(3, listed=WINDOWS * WINDOW - 8, trainable=8)})
        status, outcome, detail = judge(tmp_path, capsys, **bounds_gate())
        assert (status, outcome) == (1, "FAIL") and "trainable is above 7 at 1 iterations, from 3 (8)" in detail

    def test_an_iteration_below_its_minimum_fails(self, tmp_path, capsys):
        counts = {i: masked_counts(i) for i in range(1, 5)}
        counts_log(tmp_path, "run.log", counts | {2: masked_counts(2, listed=0)})
        gate = bounds_gate(min={"listed_trainable": 1}, max=None)
        del gate["count_bounds"]["windows"]["max"]
        status, outcome, detail = judge(tmp_path, capsys, **gate)
        assert (status, outcome) == (1, "FAIL") and "listed_trainable is below 1 at 1 iterations, from 2 (0)" in detail

    def test_a_missing_counts_line_fails_and_a_log_without_any_is_not_evaluated(self, tmp_path, capsys):
        counts = {i: masked_counts(i, listed=WINDOWS * WINDOW - 5, trainable=5) for i in range(1, 5)}
        counts_log(tmp_path, "run.log", counts | {4: None})
        assert judge(tmp_path, capsys, **bounds_gate())[:2] == (1, "FAIL")
        counts_log(tmp_path, "run.log", dict.fromkeys(counts))
        assert judge(tmp_path, capsys, **bounds_gate())[:2] == (2, "NOT EVALUATED")

    def test_a_log_that_ran_past_the_range_is_not_evaluated(self, tmp_path, capsys):
        counts = {i: masked_counts(i, listed=WINDOWS * WINDOW - 5, trainable=5) for i in range(1, 7)}
        counts_log(tmp_path, "run.log", counts)
        status, outcome, detail = judge(tmp_path, capsys, **bounds_gate())
        assert (status, outcome) == (2, "NOT EVALUATED") and "iterations outside 1-4" in detail

    def test_bounds_that_no_count_could_meet_or_that_bound_no_count_are_refused(self, tmp_path):
        refused(tmp_path, "the minimum of \\['trainable'\\] is above its maximum", **bounds_gate(min={"trainable": 9}))
        refused(tmp_path, "names \\['loss'\\]", **bounds_gate(max={"loss": 1}))
        refused(tmp_path, "max.trainable is -1, not a count", **bounds_gate(max={"trainable": -1}))
        neither = bounds_gate()
        del neither["count_bounds"]["windows"]["min"], neither["count_bounds"]["windows"]["max"]
        refused(tmp_path, "states neither min nor max", **neither)


# --------------------------------------------------------------------------------------
# loss_band
# --------------------------------------------------------------------------------------


def loss_log(directory: Path, name: str, losses: list[float]) -> Path:
    return write_log(directory, [iteration_line(i, 3000.0, loss) for i, loss in enumerate(losses, 1)], name)


def write_band(directory: Path, candidate: list[float], window: int = 2, half_width: float | None = None) -> None:
    """``loss_parity.py``'s own band report of a candidate against two references over iterations 1-8."""
    references = [
        loss_log(directory, "reference_a.log", [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3]),
        loss_log(directory, "reference_b.log", [3.02, 2.91, 2.82, 2.71, 2.62, 2.51, 2.42, 2.31]),
    ]
    trajectories = [loss_parity.load_trajectory(log, (1, 8), False) for log in references]
    candidates = [loss_parity.load_trajectory(loss_log(directory, "candidate.log", candidate), (1, 8), False)]
    if half_width is not None:
        trajectories = trajectories[:1]
    report = loss_parity.band_test(trajectories, candidates, window, half_width)
    (directory / "band.json").write_text(json.dumps(report.to_dict()))


def band_gate(**edits) -> dict:
    return {
        "loss_band": {"band": {"report": "band.json", "first": 1, "last": 8, "window": 2, "references": 2} | edits}
    }


class TestLossBand:
    def test_a_candidate_inside_its_references_band_passes(self, tmp_path, capsys):
        write_band(tmp_path, [3.01, 2.9, 2.81, 2.7, 2.61, 2.5, 2.41, 2.3])
        status, outcome, detail = judge(tmp_path, capsys, **band_gate())
        assert (status, outcome) == (0, "PASS")
        assert "against 2 references over 1-8 in 4 windows of 2" in detail and "inside the band" in detail

    def test_a_candidate_outside_the_band_fails_naming_the_windows(self, tmp_path, capsys):
        write_band(tmp_path, [3.01, 2.9, 2.81, 2.7, 3.2, 3.1, 2.41, 2.3])
        status, outcome, detail = judge(tmp_path, capsys, **band_gate())
        assert (status, outcome) == (1, "FAIL") and "outside the band in the windows from [5]" in detail

    def test_a_candidate_that_counted_a_nan_iteration_fails_inside_the_band(self, tmp_path, capsys):
        write_band(tmp_path, [3.01, 2.9, 2.81, 2.7, 2.61, 2.5, 2.41, 2.3])
        report = json.loads((tmp_path / "band.json").read_text())
        report["verdicts"][0] |= {"nan_total": 1, "verdict": "FAIL"}
        (tmp_path / "band.json").write_text(json.dumps(report))
        status, outcome, detail = judge(tmp_path, capsys, **band_gate())
        assert (status, outcome) == (1, "FAIL") and "0 skipped / 1 NaN iterations" in detail

    @pytest.mark.parametrize(
        ("edits", "message"),
        [
            ({"window": 4}, "in windows of 2 from 2 references, not 1-8 in windows of 4 from 2"),
            ({"last": 6}, "is a band over 1-8"),
            ({"references": 3}, "from 2 references, not 1-8 in windows of 2 from 3"),
        ],
    )
    def test_a_report_of_another_shape_than_the_one_stated_is_not_evaluated(self, tmp_path, capsys, edits, message):
        write_band(tmp_path, [3.01, 2.9, 2.81, 2.7, 2.61, 2.5, 2.41, 2.3])
        status, outcome, detail = judge(tmp_path, capsys, **band_gate(**edits))
        assert (status, outcome) == (2, "NOT EVALUATED") and message in detail

    def test_a_band_of_a_stated_half_width_is_not_its_references_spread(self, tmp_path, capsys):
        write_band(tmp_path, [3.01, 2.9, 2.81, 2.7, 2.61, 2.5, 2.41, 2.3], half_width=0.5)
        report = json.loads((tmp_path / "band.json").read_text())
        report["references"] = [*report["references"], "second"]  # so that only the half-width differs from the gate
        (tmp_path / "band.json").write_text(json.dumps(report))
        status, outcome, detail = judge(tmp_path, capsys, **band_gate())
        assert (status, outcome) == (2, "NOT EVALUATED") and "stated half-width" in detail

    def test_a_band_of_one_reference_is_refused(self, tmp_path):
        refused(tmp_path, "at least 2 references", **band_gate(references=1))


# --------------------------------------------------------------------------------------
# stops
# --------------------------------------------------------------------------------------

STOP = {"non_finite_grad_norm": True, "non_finite_lm_loss": True, "nan_or_skipped_iterations": True}


def write_watch(directory: Path, stop: dict | None = None) -> Path:
    path = directory / "watch.yaml"
    path.write_text(yaml.safe_dump({"stop": STOP if stop is None else stop}))
    return path


def stops_gate(directory: Path, **edits) -> dict:
    gate = {"logs": ["first.log", "second.log"], "watch": str(directory / "watch.yaml"), "first": 1, "last": 6}
    return {"stops": {"finite": gate | edits}}


def segments(directory: Path, **lines: str) -> None:
    """A run saved at iteration 3 and resumed: its two segment logs, with the given iterations' lines replaced."""
    line = {i: lines.get(f"i{i}", iteration_line(i, 3000.0, 2.5)) for i in range(1, 7)}
    write_log(directory, [line[i] for i in (1, 2, 3)], "first.log")
    write_log(directory, [line[i] for i in (4, 5, 6)], "second.log")


class TestStops:
    def test_two_segments_covering_the_range_with_no_condition_holding_pass(self, tmp_path, capsys):
        write_watch(tmp_path)
        segments(tmp_path)
        status, outcome, detail = judge(tmp_path, capsys, **stops_gate(tmp_path))
        assert (status, outcome) == (0, "PASS") and "none holds" in detail

    def test_a_non_finite_grad_norm_in_the_resumed_segment_fails(self, tmp_path, capsys):
        write_watch(tmp_path)
        segments(tmp_path, i5=sub_once(r"grad norm: [\d.]+", "grad norm: nan", iteration_line(5, 3000.0, 2.5)))
        status, outcome, detail = judge(tmp_path, capsys, **stops_gate(tmp_path))
        assert (status, outcome) == (1, "FAIL") and "non-finite grad norm at iteration 5" in detail

    def test_an_iteration_line_without_its_loss_fails(self, tmp_path, capsys):
        # How a NaN loss is logged when the loss NaN check is off: the line has no lm loss.
        write_watch(tmp_path)
        segments(tmp_path, i2=sub_once(r"lm loss: [\dE.+-]+ \| ", "", iteration_line(2, 3000.0, 2.5)))
        status, outcome, detail = judge(tmp_path, capsys, **stops_gate(tmp_path))
        assert (status, outcome) == (1, "FAIL") and "iteration 2 logged no lm loss" in detail

    def test_a_counted_nan_iteration_fails(self, tmp_path, capsys):
        write_watch(tmp_path)
        segments(tmp_path, i6=iteration_line(6, 3000.0, 2.5, nan=1))
        status, outcome, detail = judge(tmp_path, capsys, **stops_gate(tmp_path))
        assert (status, outcome) == (1, "FAIL") and "iteration 6 counts 1 nan" in detail

    def test_logs_that_do_not_cover_exactly_the_range_are_not_evaluated(self, tmp_path, capsys):
        write_watch(tmp_path)
        segments(tmp_path)
        for edits in ({"last": 8}, {"first": 2}, {"logs": ["first.log"]}):
            status, outcome, detail = judge(tmp_path, capsys, **stops_gate(tmp_path, **edits))
            assert (status, outcome) == (2, "NOT EVALUATED") and "not exactly" in detail, edits

    def test_a_log_with_no_iteration_is_not_evaluated(self, tmp_path, capsys):
        write_watch(tmp_path)
        write_log(tmp_path, ["no iteration here"], "first.log")
        status, outcome, detail = judge(tmp_path, capsys, **stops_gate(tmp_path, logs=["first.log"]))
        assert (status, outcome) == (2, "NOT EVALUATED") and "log iterations none" in detail

    def test_a_watch_that_states_no_stop_condition_is_not_evaluated(self, tmp_path, capsys):
        write_watch(tmp_path, stop={})
        segments(tmp_path)
        status, outcome, detail = judge(tmp_path, capsys, **stops_gate(tmp_path))
        assert (status, outcome) == (2, "NOT EVALUATED") and "states no stop condition" in detail

    def test_a_missing_watch_spec_is_not_evaluated(self, tmp_path, capsys):
        segments(tmp_path)
        assert judge(tmp_path, capsys, **stops_gate(tmp_path))[:2] == (2, "NOT EVALUATED")

    def test_logs_must_be_listed_once_each(self, tmp_path):
        refused(tmp_path, "logs must list the run's segment logs", **stops_gate(tmp_path, logs=[]))
        refused(tmp_path, "logs must list the run's segment logs", **stops_gate(tmp_path, logs=["a.log", "a.log"]))
