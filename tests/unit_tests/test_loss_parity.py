"""Unit tests for scripts/telemetry/loss_parity.py (loss-trajectory parity between runs).

Every log here is the real excerpt of ``training_log_fixture.py`` (iterations 1-60 of the filtered stage-1
pretraining log) or a copy of it with chosen fields of chosen iteration lines rewritten, so each comparison runs
on the format the bridge actually prints.
"""

import importlib.util
import json
import re
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit_tests.training_log_fixture import FIXTURE, FIXTURE_LINES, FIXTURE_RUN_PATH, REPO_ROOT, sub_once


ITERATION_RE = re.compile(r"iteration\s+(\d+)/")


@pytest.fixture(scope="module")
def lp():
    """Import the real script by path (scripts/telemetry/ is not an installed package)."""
    spec = importlib.util.spec_from_file_location(
        "loss_parity", REPO_ROOT / "scripts" / "telemetry" / "loss_parity.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _field(line: str, pattern: str) -> float:
    return float(re.search(pattern, line).group(1))


LOSS_RE = r"lm loss: ([\dE.+-]+)"
GRAD_RE = r"grad norm: ([\d.]+)"
LR_RE = r"learning rate: ([\dE.+-]+)"


def fixture_values(pattern: str) -> dict[int, float]:
    """The fixture's printed value of one field, by iteration."""
    values = {}
    for line in FIXTURE_LINES:
        match = ITERATION_RE.search(line)
        if match and "consumed samples" in line:
            values[int(match.group(1))] = _field(line, pattern)
    return values


FIXTURE_LOSS = fixture_values(LOSS_RE)
FIXTURE_GRAD = fixture_values(GRAD_RE)


def write_run(
    tmp_path: Path,
    name: str,
    loss_offset: dict[int, float] | None = None,
    grad_scale: dict[int, float] | None = None,
    learning_rate: dict[int, float] | None = None,
    consumed_samples: dict[int, int] | None = None,
    nan: dict[int, int] | None = None,
    skipped: dict[int, int] | None = None,
    drop: set[int] | None = None,
) -> Path:
    """A copy of the fixture log with the named fields of the named iterations rewritten."""
    lines = []
    for line in FIXTURE_LINES:
        match = ITERATION_RE.search(line)
        if match is None or "consumed samples" not in line:
            lines.append(line)
            continue
        it = int(match.group(1))
        if drop and it in drop:
            continue
        if loss_offset and it in loss_offset:
            line = sub_once(LOSS_RE, f"lm loss: {_field(line, LOSS_RE) + loss_offset[it]:.6E}", line)
        if grad_scale and it in grad_scale:
            line = sub_once(GRAD_RE, f"grad norm: {_field(line, GRAD_RE) * grad_scale[it]:.3f}", line)
        if learning_rate and it in learning_rate:
            line = sub_once(LR_RE, f"learning rate: {learning_rate[it]:.6E}", line)
        if consumed_samples and it in consumed_samples:
            line = sub_once(r"consumed samples:\s+\d+", f"consumed samples: {consumed_samples[it]:12d}", line)
        if nan and it in nan:
            line = sub_once(r"nan iterations:\s+\d+", f"nan iterations: {nan[it]:3d}", line)
        if skipped and it in skipped:
            line = sub_once(r"skipped iterations:\s+\d+", f"skipped iterations: {skipped[it]:3d}", line)
        lines.append(line)
    path = tmp_path / f"{name}.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def over(first: int, last: int, value: float) -> dict[int, float]:
    return {it: value for it in range(first, last + 1)}


def load(lp, path: Path, iterations=(1, 60)):
    return lp.load_trajectory(path, iterations, use_wandb=False)


def window_mean(values: dict[int, float], first: int, last: int) -> float:
    return statistics.fmean(values[it] for it in range(first, last + 1))


# --------------------------------------------------------------------------------------
# Loading a trajectory
# --------------------------------------------------------------------------------------


def test_trajectory_carries_every_compared_field_of_the_range(lp):
    trajectory = lp.load_trajectory(FIXTURE, (11, 20), use_wandb=False)
    assert (trajectory.first, trajectory.last, trajectory.source) == (11, 20, "log")
    assert trajectory.values["lm loss"] == tuple(FIXTURE_LOSS[it] for it in range(11, 21))
    assert trajectory.values["grad norm"] == tuple(FIXTURE_GRAD[it] for it in range(11, 21))
    assert trajectory.values["learning rate"][0] == fixture_values(LR_RE)[11]
    assert trajectory.consumed_samples == tuple(2048 * it for it in range(11, 21))
    assert (trajectory.skipped_total, trajectory.nan_total) == (0, 0)


def test_trajectory_with_a_missing_iteration_raises(lp, tmp_path):
    path = write_run(tmp_path, "gap", drop={30})
    with pytest.raises(ValueError, match=r"missing \[30\]"):
        load(lp, path)


def test_trajectory_of_an_iteration_without_grad_norm_raises(lp, tmp_path):
    lines = [
        sub_once(r" grad norm: [\d.]+ \|", "", line) if re.search(r"iteration\s+7/", line) else line
        for line in FIXTURE_LINES
    ]
    path = tmp_path / "no_grad.out"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="iteration 7 logs no grad norm"):
        load(lp, path)


@pytest.mark.parametrize("iterations", [(0, 10), (20, 10)])
def test_trajectory_rejects_an_invalid_range(lp, iterations):
    with pytest.raises(ValueError, match="must start at >= 1"):
        lp.load_trajectory(FIXTURE, iterations, use_wandb=False)


# --------------------------------------------------------------------------------------
# Band test
# --------------------------------------------------------------------------------------


def test_identical_references_give_a_zero_band_that_an_identical_candidate_passes(lp):
    runs = [load(lp, FIXTURE) for _ in range(3)]
    report = lp.band_test(runs[:2], runs[2:], window=20)
    loss, grad = report.metrics
    assert (loss.metric, grad.metric) == ("lm loss", "grad norm")
    assert loss.delta == 0.0 and grad.delta == 0.0
    assert [(w.first, w.last) for w in loss.windows] == [(1, 20), (21, 40), (41, 60)]
    assert loss.windows[1].low == loss.windows[1].high == pytest.approx(window_mean(FIXTURE_LOSS, 21, 40))
    (verdict,) = report.verdicts
    assert (verdict.verdict, verdict.grad_norm, verdict.loss_inside) == ("PASS", "PASS", True)
    assert loss.candidates[0].max_abs_deviation == 0.0


def test_a_candidate_inside_the_band_passes_with_its_deviations(lp, tmp_path):
    refs = [load(lp, FIXTURE), load(lp, write_run(tmp_path, "ref_up", loss_offset=over(1, 60, 0.02)))]
    candidate = load(lp, write_run(tmp_path, "cand", loss_offset=over(1, 60, 0.03)))
    report = lp.band_test(refs, [candidate], window=20)
    loss = report.metrics[0]
    assert loss.delta == pytest.approx(0.02)
    first = loss.windows[0]
    assert first.low == pytest.approx(window_mean(FIXTURE_LOSS, 1, 20) - 0.02)
    assert first.high == pytest.approx(window_mean(FIXTURE_LOSS, 1, 20) + 0.04)
    result = loss.candidates[0]
    # The candidate sits 0.03 above the first reference and 0.01 above the second: +0.02 from their mean.
    assert result.inside
    assert result.max_abs_deviation == pytest.approx(0.02)
    assert result.final_deviation == pytest.approx(0.02)
    assert report.verdicts[0].verdict == "PASS"


def test_a_band_report_survives_its_json_round_trip(lp, tmp_path):
    """A gate reads a band report back from the --json file the band test wrote."""
    refs = [load(lp, FIXTURE), load(lp, write_run(tmp_path, "ref_up", loss_offset=over(1, 60, 0.02)))]
    candidate = load(lp, write_run(tmp_path, "cand", learning_rate={30: 1.0e-3}, nan={40: 1}))
    report = lp.band_test(refs, [candidate], window=20)
    assert lp.BandReport.from_dict(json.loads(json.dumps(report.to_dict()))) == report


def test_offsets_are_each_windows_candidate_mean_minus_the_references_mean(lp, tmp_path):
    refs = [load(lp, FIXTURE), load(lp, write_run(tmp_path, "ref_up", loss_offset=over(1, 60, 0.01)))]
    candidate = load(lp, write_run(tmp_path, "cand", loss_offset={**over(1, 30, 0.004), **over(31, 60, 0.008)}))
    (loss, _) = lp.band_test(refs, [candidate], window=10).metrics
    assert lp.offsets_from_reference_mean(loss.windows, 0) == pytest.approx([-0.001] * 3 + [0.003] * 3, abs=1e-6)


def test_the_offset_rise_is_the_last_windows_mean_above_the_first_windows(lp):
    assert lp.offset_rise([0.0, 0.001, 0.002, 0.004], 2) == pytest.approx(0.0025)
    with pytest.raises(ValueError, match="two spans"):
        lp.offset_rise([0.0, 0.001, 0.002], 2)


def test_a_candidate_outside_one_window_fails_and_names_it(lp, tmp_path):
    refs = [load(lp, FIXTURE), load(lp, write_run(tmp_path, "ref_w1", loss_offset=over(1, 10, 0.001)))]
    candidate = load(lp, write_run(tmp_path, "cand", loss_offset=over(51, 60, 0.01)))
    report = lp.band_test(refs, [candidate], window=10)
    loss = report.metrics[0]
    assert loss.delta == pytest.approx(0.001)
    result = loss.candidates[0]
    assert (result.inside, result.windows_outside) == (False, (51,))
    assert result.final_deviation == pytest.approx(0.01)
    assert result.max_abs_deviation == pytest.approx(0.01)
    assert report.verdicts[0].verdict == "FAIL"
    assert not report.verdicts[0].loss_inside


def test_delta_is_the_largest_window_difference_over_every_reference_pair(lp, tmp_path):
    refs = [
        load(lp, FIXTURE),
        load(lp, write_run(tmp_path, "ref_b", loss_offset=over(21, 30, 0.01))),
        load(lp, write_run(tmp_path, "ref_c", loss_offset=over(21, 30, -0.02))),
    ]
    report = lp.band_test(refs, [load(lp, FIXTURE)], window=10)
    # ref_b - ref_c = 0.03 in 21-30, larger than either's difference from the first reference.
    assert report.metrics[0].delta == pytest.approx(0.03)
    window = report.metrics[0].windows[2]
    assert window.low == pytest.approx(window_mean(FIXTURE_LOSS, 21, 30) - 0.02 - 0.03)
    assert window.high == pytest.approx(window_mean(FIXTURE_LOSS, 21, 30) + 0.01 + 0.03)


def test_a_fixed_loss_half_width_replaces_the_reference_spread_for_the_loss_only(lp, tmp_path):
    """The references differ by 0.01 in 21-30, so their own band would reject a candidate 0.04 above the
    higher one; a 0.05 half-width admits it, and grad norm keeps its spread."""
    refs = [load(lp, FIXTURE), load(lp, write_run(tmp_path, "ref_b", loss_offset=over(21, 30, 0.01)))]
    candidate = load(lp, write_run(tmp_path, "cand", loss_offset=over(21, 30, 0.05)))
    spread = lp.band_test(refs, [candidate], window=10)
    fixed = lp.band_test(refs, [candidate], window=10, loss_half_width=0.05)
    assert spread.verdicts[0].verdict == "FAIL" and fixed.verdicts[0].verdict == "PASS"
    assert fixed.metrics[0].delta == 0.05 and fixed.metrics[0].fixed_half_width
    assert fixed.metrics[0].spread == pytest.approx(0.01) == spread.metrics[0].delta
    assert fixed.metrics[1].delta == spread.metrics[1].delta and not fixed.metrics[1].fixed_half_width
    report = lp.format_band_report(fixed)
    assert "lm loss: delta 0.050000 (fixed half-width; reference spread 0.010000)" in report
    assert "grad norm: delta" in report and "(largest reference-pair window difference)" in report


def test_a_grad_norm_outside_its_band_flags_without_failing(lp, tmp_path):
    refs = [load(lp, FIXTURE), load(lp, FIXTURE)]
    candidate = load(lp, write_run(tmp_path, "cand", grad_scale=over(41, 60, 1.5)))
    report = lp.band_test(refs, [candidate], window=20)
    (verdict,) = report.verdicts
    assert (verdict.verdict, verdict.grad_norm) == ("PASS", "FLAG")
    assert report.metrics[1].candidates[0].windows_outside == (41,)


@pytest.mark.parametrize("counter", ["nan", "skipped"])
def test_a_candidate_with_a_nan_or_skipped_iteration_fails(lp, tmp_path, counter):
    candidate = load(lp, write_run(tmp_path, "cand", **{counter: {30: 1}}))
    (verdict,) = lp.band_test([load(lp, FIXTURE), load(lp, FIXTURE)], [candidate], window=20).verdicts
    assert verdict.verdict == "FAIL"
    assert verdict.loss_inside
    assert (verdict.nan_total, verdict.skipped_total) == ((1, 0) if counter == "nan" else (0, 1))


def test_a_candidate_on_another_schedule_fails_at_its_first_learning_rate_difference(lp, tmp_path):
    candidate = load(lp, write_run(tmp_path, "cand", learning_rate={45: 9.9e-4}))
    (verdict,) = lp.band_test([load(lp, FIXTURE), load(lp, FIXTURE)], [candidate], window=20).verdicts
    assert verdict.verdict == "FAIL"
    assert verdict.first_learning_rate_difference.iteration == 45
    assert verdict.first_learning_rate_difference.candidate == 9.9e-4
    assert verdict.first_consumed_samples_difference is None


def test_a_candidate_at_another_data_position_fails(lp, tmp_path):
    candidate = load(lp, write_run(tmp_path, "cand", consumed_samples={12: 2048 * 13}))
    (verdict,) = lp.band_test([load(lp, FIXTURE), load(lp, FIXTURE)], [candidate], window=20).verdicts
    assert verdict.verdict == "FAIL"
    assert verdict.first_consumed_samples_difference == lp.Difference(
        iteration=12, reference=2048 * 12, candidate=2048 * 13
    )


@pytest.mark.parametrize(
    "edit, message",
    [
        ({"nan": {5: 1}}, "1 NaN iterations"),
        ({"skipped": {5: 2}}, "2 skipped"),
        ({"learning_rate": {50: 1e-3}}, "learning rate differs at iteration 50"),
        ({"consumed_samples": {3: 1}}, "consumed samples differ at iteration 3"),
    ],
)
def test_a_reference_that_is_not_clean_raises(lp, tmp_path, edit, message):
    bad = load(lp, write_run(tmp_path, "bad_ref", **edit))
    with pytest.raises(ValueError, match=f"is not a clean reference: .*{message}"):
        lp.band_test([load(lp, FIXTURE), bad], [load(lp, FIXTURE)], window=20)


def test_a_band_needs_two_references_and_a_candidate(lp):
    run = load(lp, FIXTURE)
    with pytest.raises(ValueError, match="at least two reference runs"):
        lp.band_test([run], [run], window=20)
    with pytest.raises(ValueError, match="at least one candidate"):
        lp.band_test([run, run], [], window=20)


@pytest.mark.parametrize("window", [0, 7, 61])
def test_the_range_must_be_a_whole_number_of_windows(lp, window):
    run = load(lp, FIXTURE)
    with pytest.raises(ValueError, match="whole number"):
        lp.band_test([run, run], [run], window=window)


def test_trajectories_over_different_ranges_raise(lp):
    with pytest.raises(ValueError, match="different iteration ranges"):
        lp.band_test([load(lp, FIXTURE), load(lp, FIXTURE)], [load(lp, FIXTURE, (1, 40))], window=20)


def test_trajectories_read_from_different_sources_raise(lp):
    run = load(lp, FIXTURE)
    wandb_run = replace(run, source=f"wandb:{FIXTURE_RUN_PATH}")
    with pytest.raises(ValueError, match="mix sources"):
        lp.band_test([run, run], [wandb_run], window=20)


# --------------------------------------------------------------------------------------
# Identity test
# --------------------------------------------------------------------------------------


def test_a_run_is_identical_to_itself(lp):
    report = lp.identity_test(load(lp, FIXTURE), load(lp, FIXTURE))
    assert report.identical
    assert [(m.metric, m.leading_identical, m.first_difference, m.largest_difference) for m in report.metrics] == [
        ("lm loss", 60, None, None),
        ("grad norm", 60, None, None),
        ("learning rate", 60, None, None),
        ("consumed samples", 60, None, None),
    ]


def test_identity_reports_the_largest_difference_after_the_first(lp, tmp_path):
    candidate = load(lp, write_run(tmp_path, "cand", loss_offset={23: 1e-5, 40: -3e-4, 52: 2e-4}))
    loss = lp.identity_test(load(lp, FIXTURE), candidate).metrics[0]
    assert loss.first_difference.iteration == 23
    assert loss.largest_difference.iteration == 40
    assert loss.largest_difference.reference == FIXTURE_LOSS[40]
    assert loss.largest_difference.candidate == pytest.approx(FIXTURE_LOSS[40] - 3e-4, abs=1e-6)


def test_largest_difference_takes_the_earliest_of_equal_differences(lp):
    assert lp.largest_difference(5, [1.0, 2.0, 3.0], [1.0, 2.5, 2.5]).iteration == 6
    assert lp.largest_difference(5, [1.0, 2.0], [1.0, 2.0]) is None


def test_identity_reports_each_metric_first_difference(lp, tmp_path):
    candidate = load(lp, write_run(tmp_path, "cand", grad_scale={17: 2.0}, loss_offset={23: 1e-5}))
    report = lp.identity_test(load(lp, FIXTURE), candidate)
    assert not report.identical
    loss, grad, lr, consumed = report.metrics
    assert loss.leading_identical == 22
    assert loss.first_difference.iteration == 23
    assert loss.first_difference.reference == FIXTURE_LOSS[23]
    assert (grad.leading_identical, grad.first_difference.iteration) == (16, 17)
    assert grad.first_difference.candidate == pytest.approx(2 * FIXTURE_GRAD[17], abs=1e-3)
    assert lr.first_difference is None and consumed.first_difference is None


def test_identity_counts_leading_iterations_from_the_range_start(lp, tmp_path):
    candidate = lp.load_trajectory(write_run(tmp_path, "cand", loss_offset={45: 1e-4}), (41, 50), use_wandb=False)
    report = lp.identity_test(lp.load_trajectory(FIXTURE, (41, 50), use_wandb=False), candidate)
    assert report.metrics[0].leading_identical == 4


def test_identity_over_different_ranges_raises(lp):
    with pytest.raises(ValueError, match="different iteration ranges"):
        lp.identity_test(load(lp, FIXTURE), load(lp, FIXTURE, (1, 30)))


# --------------------------------------------------------------------------------------
# W&B source
# --------------------------------------------------------------------------------------


# wandb's own page size for scan_history.
WANDB_PAGE_SIZE = 1000


class FakeRun:
    """Stand-in for a ``wandb.Api().run(...)`` result, the network boundary: the real client reads the
    run's history from W&B's service, which unit tests must not depend on. It records every
    ``scan_history`` request and returns the given rows unfiltered, as a server honouring only part of
    the step range would.

    The rows of the requested range that fill a page come back as wandb 0.27's paged scan with ``keys`` returns
    them (measured on three runs, 2026-10-01): every full page loses a row about two thirds of the way in and
    every later page repeats its first row, so steps 1-2000 in pages of 1000 come back without 669 and 1669 and
    with 1001 twice."""

    requests: list[dict] = []

    def __init__(self, rows: list[dict], default_page_size: int):
        self.rows = rows
        self.default_page_size = default_page_size

    def scan_history(self, keys, min_step, max_step, page_size=None):
        FakeRun.requests.append({"keys": keys, "min_step": min_step, "max_step": max_step, "page_size": page_size})
        size = self.default_page_size if page_size is None else page_size
        in_range = [row for row in self.rows if min_step <= row["_step"] < max_step]
        returned = [row for row in self.rows if row not in in_range]
        for start in range(0, len(in_range), size):
            page = in_range[start : start + size]
            if len(page) == size:
                page = page[: 2 * size // 3] + page[2 * size // 3 + 1 :]
            returned.extend([page[0], *page] if start else page)
        return iter(returned)


def install_fake_wandb(monkeypatch, rows: list[dict], default_page_size: int = WANDB_PAGE_SIZE) -> list[str]:
    requested_runs: list[str] = []
    FakeRun.requests = []

    def run(path: str) -> FakeRun:
        requested_runs.append(path)
        return FakeRun(rows, default_page_size)

    # load_trajectory imports wandb at call time, so a module in sys.modules replaces the client.
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda: SimpleNamespace(run=run)))
    return requested_runs


def history_rows(first: int, last: int, loss_extra: float = 1.25e-7) -> list[dict]:
    """Full-precision rows: the fixture's printed values plus digits the log cannot show."""
    return [
        {
            "_step": it,
            "lm loss": FIXTURE_LOSS[it] + loss_extra,
            "grad-norm": FIXTURE_GRAD[it] + 3e-4,
            "learning-rate": 1e-4,
        }
        for it in range(first, last + 1)
    ]


def test_wandb_values_replace_the_printed_ones(lp, monkeypatch):
    requested = install_fake_wandb(monkeypatch, history_rows(1, 60))
    trajectory = lp.load_trajectory(FIXTURE, (5, 20), use_wandb=True)
    assert requested == [FIXTURE_RUN_PATH]
    assert FakeRun.requests == [
        {"keys": ["_step", "lm loss", "grad-norm", "learning-rate"], "min_step": 5, "max_step": 21, "page_size": 32}
    ]
    assert trajectory.source == f"wandb:{FIXTURE_RUN_PATH}"
    assert trajectory.values["lm loss"] == tuple(FIXTURE_LOSS[it] + 1.25e-7 for it in range(5, 21))
    assert trajectory.values["learning rate"] == (1e-4,) * 16
    # The data position always comes from the log.
    assert trajectory.consumed_samples == tuple(2048 * it for it in range(5, 21))


def test_full_precision_separates_what_print_precision_cannot(lp, monkeypatch):
    install_fake_wandb(monkeypatch, history_rows(1, 60))
    reference = lp.load_trajectory(FIXTURE, (1, 60), use_wandb=True)
    install_fake_wandb(monkeypatch, history_rows(1, 60, loss_extra=2.5e-7))
    candidate = lp.load_trajectory(FIXTURE, (1, 60), use_wandb=True)
    assert lp.identity_test(load(lp, FIXTURE), load(lp, FIXTURE)).identical
    report = lp.identity_test(reference, candidate)
    assert report.metrics[0].first_difference.iteration == 1
    assert report.metrics[1].first_difference is None


@pytest.mark.parametrize(
    "rows, message",
    [
        (history_rows(1, 29) + history_rows(31, 60), r"missing \[30\]"),
        (history_rows(1, 60) + history_rows(12, 12), r"repeated \[12\]"),
    ],
)
def test_wandb_history_must_cover_each_iteration_once(lp, monkeypatch, rows, message):
    install_fake_wandb(monkeypatch, rows)
    with pytest.raises(ValueError, match=message):
        lp.load_trajectory(FIXTURE, (1, 60), use_wandb=True)


def test_a_range_wider_than_the_clients_page_is_read_whole(lp, monkeypatch):
    install_fake_wandb(monkeypatch, history_rows(1, 60), default_page_size=20)
    trajectory = lp.load_trajectory(FIXTURE, (1, 60), use_wandb=True)
    assert trajectory.values["lm loss"] == tuple(FIXTURE_LOSS[it] + 1.25e-7 for it in range(1, 61))


def test_wandb_needs_a_run_named_in_the_log(lp, monkeypatch, tmp_path):
    requested = install_fake_wandb(monkeypatch, history_rows(1, 60))
    no_run = tmp_path / "no_run.out"
    no_run.write_text("\n".join(line for line in FIXTURE_LINES if "View run" not in line) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names no W&B run"):
        lp.load_trajectory(no_run, (1, 60), use_wandb=True)
    assert requested == []


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_cli_band_passes_with_exit_status_zero(lp, tmp_path, capsys):
    ref_b = write_run(tmp_path, "ref_b", loss_offset=over(1, 60, 0.01))
    args = ["band", "--reference", str(FIXTURE), str(ref_b), "--candidate", str(FIXTURE)]
    assert lp.main([*args, "--iterations", "1", "60", "--window", "20", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (result["first"], result["last"], result["window"], result["source"]) == (1, 60, 20, "log")
    assert result["references"] == [str(FIXTURE), str(ref_b)]
    assert result["verdicts"][0]["verdict"] == "PASS"
    assert result["metrics"][0]["delta"] == pytest.approx(0.01)

    assert lp.main([*args, "--iterations", "1", "60", "--window", "20"]) == 0
    report = capsys.readouterr().out
    assert "lm loss: delta 0.010000" in report
    assert f"verdict {FIXTURE}: PASS (lm loss inside its band; grad norm PASS)" in report


def test_cli_band_fails_with_exit_status_one(lp, tmp_path, capsys):
    candidate = write_run(tmp_path, "cand", loss_offset=over(41, 60, 0.05), learning_rate={44: 5e-4})
    args = ["band", "--reference", str(FIXTURE), str(FIXTURE), "--candidate", str(candidate)]
    assert lp.main([*args, "--iterations", "1", "60", "--window", "20"]) == 1
    report = capsys.readouterr().out
    assert f"{candidate}: OUTSIDE in windows [41]" in report
    assert (
        f"verdict {candidate}: FAIL (lm loss outside its band; grad norm PASS; learning rate differs at iteration 44)"
        in report
    )


def test_cli_identity_reports_each_candidate(lp, tmp_path, capsys):
    candidate = write_run(tmp_path, "cand", grad_scale={9: 1.1})
    args = ["identity", "--reference", str(FIXTURE), "--candidate", str(FIXTURE), str(candidate)]
    assert lp.main([*args, "--iterations", "1", "20"]) == 1
    report = capsys.readouterr().out
    assert report.count("vs reference") == 2
    assert "iterations 1-20 (log vs log): IDENTICAL" in report
    assert "grad norm         identical for 8 leading iterations; first difference at iteration 9" in report
    assert "largest |difference|" in report

    assert lp.main([*args[:5], "--iterations", "1", "20", "--json"]) == 0
    (result,) = json.loads(capsys.readouterr().out)
    assert result["identical"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["band", "--reference", "a", "b", "--candidate", "c", "--iterations", "1", "60"],
        ["identity", "--reference", "a", "b", "--candidate", "c", "--iterations", "1", "60"],
        ["identity", "--reference", "a", "--candidate", "c"],
    ],
)
def test_cli_rejects_incomplete_arguments(lp, argv):
    with pytest.raises(SystemExit):
        lp.main(argv)
