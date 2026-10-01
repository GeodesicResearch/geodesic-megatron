"""Loss-trajectory parity between training runs that share a config, seed and data order.

A performance lever is usable only if runs made with it are functionally the same training as the
production posture (docs/investigations/nano30b-pretrain-perf-campaign.md, "Functional parity"). This
tool implements the trajectory clause of that rule. It compares runs whose iteration i consumed the same
global batch, which holds for runs of one training config, seed and data-parallel width whatever their
performance levers (the same section shows why):

``band``      for a lever that changes numerics (bf16 gradients, FP8, the fp32-SSM patch off, reordered
              collectives). The compared iterations are cut into windows of equal length and each run's
              mean is taken per window. The reference spread ``delta`` is the largest difference between
              the window means of two reference runs, over every window and every pair of references; a
              window's band is [lowest reference mean - delta, highest reference mean + delta]. A candidate
              passes a metric when its mean is inside the band in every window. ``lm loss`` decides the
              verdict (PASS or FAIL); ``grad norm`` is tested the same way and reported as PASS or FLAG.
``identity``  for a lever, or a code path with its lever off, that claims exactness: every iteration's
              ``lm loss``, ``grad norm`` and ``learning rate`` must equal the reference's. The report gives,
              per metric, how many leading iterations agree, the first iteration that does not, and the
              iteration with the largest absolute difference (how far a DIFFERENT run departs).

Both tests read each run's training log (the launcher's SLURM output), in which every iteration of the
compared range must be logged exactly once, and both check that every run logs the same learning rate
and consumed-sample count at every iteration: the schedule and data position the comparison assumes. A
candidate that differs there, or that has a skipped or NaN iteration in the range, fails the band test; a
reference with either is not a reference, and the test raises.

``--wandb`` takes ``lm loss``, ``grad norm`` and ``learning rate`` from the W&B run each log names instead
of from the log, whose printed values carry 7 significant digits for the loss and learning rate and 3
decimals for the grad norm. An identity claim is only as exact as its source: at print precision it says
that the printed digits agree. Consumed samples always come from the log.

The exit status is 0 when every candidate passes (a grad-norm FLAG does not fail it) and 1 otherwise.

``windows-outside`` reads a ``band --json`` report back and prints how many windows its one candidate left
the loss band in when every other check passed (0 when it passed), or ``other`` when one did not; it exits 0.

USAGE
    python scripts/telemetry/loss_parity.py band --reference A1.out A2.out --candidate C.out \\
        --iterations 1 500 --window 50 [--wandb] [--json]
    python scripts/telemetry/loss_parity.py identity --reference A.out --candidate B.out [B2.out ...] \\
        --iterations 1 30 [--wandb] [--json]
    python scripts/telemetry/loss_parity.py windows-outside band.json
"""

import argparse
import itertools
import json
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Run as a script, only scripts/telemetry/ is on sys.path; the repo root makes the training-log parser
# importable the same way from every entry point.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.telemetry.gate_outcome import FAIL, PASS
from scripts.telemetry.training_log import (
    IterationRecord,
    check_window,
    parse_iteration_records,
    parse_wandb_run_path,
    read_log_lines,
    window_records,
)


# Each compared metric: the IterationRecord field its printed value is parsed into, and the key the
# bridge logs it under in W&B (training/utils/train_utils.py::training_log).
METRIC_SOURCES = {
    "lm loss": ("lm_loss", "lm loss"),
    "grad norm": ("grad_norm", "grad-norm"),
    "learning rate": ("learning_rate", "learning-rate"),
}
VERDICT_METRIC = "lm loss"
FLAG_METRIC = "grad norm"
SCHEDULE_METRIC = "learning rate"
# A flagged metric's verdict: reported beside the gate's own PASS or FAIL, never deciding it.
FLAG = "FLAG"


@dataclass(frozen=True)
class Trajectory:
    """One run's per-iteration values over the compared range, in iteration order.

    ``values`` maps each name of ``METRIC_SOURCES`` to its per-iteration values; ``source`` is ``log`` or
    ``wandb:<entity>/<project>/<run id>``. ``skipped_total`` and ``nan_total`` sum the range's lines.
    """

    label: str
    source: str
    first: int
    last: int
    values: dict[str, tuple[float, ...]]
    consumed_samples: tuple[int, ...]
    skipped_total: int
    nan_total: int


@dataclass(frozen=True)
class Difference:
    """One iteration at which a run's value differs from the reference's."""

    iteration: int
    reference: float
    candidate: float


@dataclass(frozen=True)
class WindowBand:
    """One window of a band test: each run's mean over it and the band the candidates must stay inside."""

    first: int
    last: int
    reference_means: tuple[float, ...]
    low: float
    high: float
    candidate_means: tuple[float, ...]


@dataclass(frozen=True)
class CandidateBand:
    """A candidate's position relative to one metric's bands.

    ``deviation`` is the candidate's window mean minus the mean of the references' window means;
    ``windows_outside`` lists the first iteration of every window whose mean is outside its band.
    """

    label: str
    inside: bool
    windows_outside: tuple[int, ...]
    max_abs_deviation: float
    final_deviation: float


@dataclass(frozen=True)
class MetricBand:
    """The band test of one metric: the band's half-width, every window, and every candidate's result.

    ``spread`` is the references' own spread, the largest difference between two references' window
    means. ``delta`` is the half-width the band was drawn with: the spread, or the fixed half-width the
    test was given (``fixed_half_width``).
    """

    metric: str
    delta: float
    spread: float
    fixed_half_width: bool
    windows: tuple[WindowBand, ...]
    candidates: tuple[CandidateBand, ...]


@dataclass(frozen=True)
class CandidateVerdict:
    """A candidate's band-test verdict and the checks it rests on.

    ``verdict`` is PASS when the loss stays inside its band in every window, the candidate logged the
    references' learning rate and consumed samples at every iteration and had no skipped or NaN
    iteration; FAIL otherwise. ``grad_norm`` is PASS when the grad norm stays inside its band, FLAG
    otherwise. The two ``first_*_difference`` fields are None when the candidate matches.
    """

    label: str
    verdict: str
    grad_norm: str
    loss_inside: bool
    first_learning_rate_difference: Difference | None
    first_consumed_samples_difference: Difference | None
    skipped_total: int
    nan_total: int

    @property
    def matches_the_references(self) -> bool:
        """Whether the candidate logged the references' learning rate and consumed samples at every iteration and
        had no skipped or NaN iteration: every check but the loss band."""
        return _matches_the_references(
            self.first_learning_rate_difference,
            self.first_consumed_samples_difference,
            self.skipped_total,
            self.nan_total,
        )


def _matches_the_references(
    learning_rate: Difference | None, consumed_samples: Difference | None, skipped_total: int, nan_total: int
) -> bool:
    return learning_rate is None and consumed_samples is None and not skipped_total and not nan_total


@dataclass(frozen=True)
class BandReport:
    """A band test: its inputs, one MetricBand per band metric, and one verdict per candidate."""

    first: int
    last: int
    window: int
    references: tuple[str, ...]
    source: str
    metrics: tuple[MetricBand, ...]
    verdicts: tuple[CandidateVerdict, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BandReport":
        """Rebuild a report from ``to_dict``'s output (a ``band --json`` file). Raises KeyError or TypeError on a
        dict that is not one."""

        def difference(value: dict[str, Any] | None) -> Difference | None:
            return None if value is None else Difference(**value)

        metrics = tuple(
            MetricBand(
                **{
                    **metric,
                    "windows": tuple(
                        WindowBand(
                            **{
                                **window,
                                "reference_means": tuple(window["reference_means"]),
                                "candidate_means": tuple(window["candidate_means"]),
                            }
                        )
                        for window in metric["windows"]
                    ),
                    "candidates": tuple(
                        CandidateBand(**{**candidate, "windows_outside": tuple(candidate["windows_outside"])})
                        for candidate in metric["candidates"]
                    ),
                }
            )
            for metric in raw["metrics"]
        )
        verdicts = tuple(
            CandidateVerdict(
                **{
                    **verdict,
                    "first_learning_rate_difference": difference(verdict["first_learning_rate_difference"]),
                    "first_consumed_samples_difference": difference(verdict["first_consumed_samples_difference"]),
                }
            )
            for verdict in raw["verdicts"]
        )
        return cls(**{**raw, "references": tuple(raw["references"]), "metrics": metrics, "verdicts": verdicts})


def loss_windows_outside(report: BandReport) -> int | None:
    """How many windows the report's one candidate left the loss band in, when every other check passed (0 when the
    candidate passed); None when it failed another check, which no count of windows describes. Raises ValueError
    unless the report has exactly one candidate."""
    if len(report.verdicts) != 1:
        raise ValueError(f"the report has {len(report.verdicts)} candidates, not one")
    (verdict,) = report.verdicts
    if not verdict.matches_the_references:
        return None
    (loss,) = [band for band in report.metrics if band.metric == VERDICT_METRIC]
    return len(loss.candidates[0].windows_outside)


@dataclass(frozen=True)
class MetricIdentity:
    """How long one metric stays identical to the reference, from the first compared iteration, and how far
    it departs from it: the first differing iteration and the one with the largest absolute difference."""

    metric: str
    leading_identical: int
    first_difference: Difference | None
    largest_difference: Difference | None


@dataclass(frozen=True)
class IdentityReport:
    """An identity test of one candidate against the reference, per metric and for consumed samples."""

    first: int
    last: int
    reference: str
    candidate: str
    source: str
    metrics: tuple[MetricIdentity, ...]
    identical: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dict."""
        return asdict(self)


def fetch_wandb_history(run_path: str, iterations: tuple[int, int]) -> dict[str, tuple[float, ...]]:
    """Return the W&B run's full-precision value of every ``METRIC_SOURCES`` metric at every iteration of the range.

    The bridge logs at step = iteration. Raises ValueError unless every iteration of the range has exactly
    one history row carrying all the metrics; W&B client and network errors propagate.
    """
    import wandb  # deferred: only --wandb needs the W&B client, and the log-based tests must run without it

    first, last = iterations
    keys = [wandb_key for _, wandb_key in METRIC_SOURCES.values()]
    # One page, twice the range, so it holds the range even with every step logged twice: wandb's paged scan
    # with ``keys`` loses a row of every page it fills and repeats the first row of every later page.
    history = (
        wandb.Api()
        .run(run_path)
        .scan_history(keys=["_step", *keys], min_step=first, max_step=last + 1, page_size=2 * (last - first + 1))
    )
    rows = [row for row in history if first <= row["_step"] <= last]
    steps = [row["_step"] for row in rows]
    missing = sorted(set(range(first, last + 1)) - set(steps))
    repeated = sorted({step for step in steps if steps.count(step) > 1})
    if missing or repeated:
        raise ValueError(
            f"W&B run {run_path}: every iteration {first}-{last} must have one row with {keys}; "
            f"missing {missing or 'none'}, repeated {repeated or 'none'}"
        )
    by_step = {row["_step"]: row for row in rows}
    return {
        metric: tuple(float(by_step[step][wandb_key]) for step in range(first, last + 1))
        for metric, (_, wandb_key) in METRIC_SOURCES.items()
    }


def _record_value(record: IterationRecord, field: str, label: str) -> float:
    value = getattr(record, field)
    if value is None:
        raise ValueError(f"{label}: iteration {record.iteration} logs no {field.replace('_', ' ')}")
    return value


def load_trajectory(log_path: Path, iterations: tuple[int, int], use_wandb: bool) -> Trajectory:
    """Read one run's trajectory over the inclusive ``iterations`` from its log (and its W&B run with ``use_wandb``).

    Raises ValueError when an iteration of the range is missing, repeated, or lacks a compared field, and
    when ``use_wandb`` is set but the log names no W&B run.
    """
    check_window("iterations", iterations, min_iterations=1)
    label = str(log_path)
    lines = read_log_lines(log_path)
    records = window_records(parse_iteration_records(lines), iterations, label)
    consumed = tuple(int(_record_value(r, "consumed_samples", label)) for r in records)
    if use_wandb:
        run_path = parse_wandb_run_path(lines)
        if run_path is None:
            raise ValueError(f"{label} names no W&B run, so --wandb has no history to read")
        values = fetch_wandb_history(run_path, iterations)
        source = f"wandb:{run_path}"
    else:
        values = {
            metric: tuple(_record_value(r, field, label) for r in records)
            for metric, (field, _) in METRIC_SOURCES.items()
        }
        source = "log"
    return Trajectory(
        label=label,
        source=source,
        first=iterations[0],
        last=iterations[1],
        values=values,
        consumed_samples=consumed,
        skipped_total=sum(r.skipped_total for r in records),
        nan_total=sum(r.nan_total for r in records),
    )


def first_difference(first: int, reference: Sequence[float], candidate: Sequence[float]) -> Difference | None:
    """Return the first iteration (counting from ``first``) at which the two series differ, or None."""
    for offset, (ref, cand) in enumerate(zip(reference, candidate, strict=True)):
        if ref != cand:
            return Difference(iteration=first + offset, reference=ref, candidate=cand)
    return None


def largest_difference(first: int, reference: Sequence[float], candidate: Sequence[float]) -> Difference | None:
    """Return the iteration (counting from ``first``) with the largest |candidate - reference|, the earliest on a
    tie, or None when the two series are equal."""
    largest = None
    for offset, (ref, cand) in enumerate(zip(reference, candidate, strict=True)):
        if ref != cand and (largest is None or abs(cand - ref) > abs(largest.candidate - largest.reference)):
            largest = Difference(iteration=first + offset, reference=ref, candidate=cand)
    return largest


def _same_range(trajectories: Sequence[Trajectory]) -> None:
    ranges = {(t.first, t.last) for t in trajectories}
    if len(ranges) != 1:
        raise ValueError(f"trajectories cover different iteration ranges: {sorted(ranges)}")


def _window_means(series: Sequence[float], window: int) -> list[float]:
    return [statistics.fmean(series[start : start + window]) for start in range(0, len(series), window)]


def _metric_band(
    metric: str,
    references: Sequence[Trajectory],
    candidates: Sequence[Trajectory],
    window: int,
    half_width: float | None,
) -> MetricBand:
    """The band of one metric, ``half_width`` outside the references' window means when it is given and the
    references' own spread otherwise."""
    first = references[0].first
    reference_means = [_window_means(ref.values[metric], window) for ref in references]
    candidate_means = [_window_means(cand.values[metric], window) for cand in candidates]
    n_windows = len(reference_means[0])
    spread = max(abs(a[w] - b[w]) for a, b in itertools.combinations(reference_means, 2) for w in range(n_windows))
    delta = spread if half_width is None else half_width
    windows = []
    for w in range(n_windows):
        refs = tuple(means[w] for means in reference_means)
        windows.append(
            WindowBand(
                first=first + w * window,
                last=first + (w + 1) * window - 1,
                reference_means=refs,
                low=min(refs) - delta,
                high=max(refs) + delta,
                candidate_means=tuple(means[w] for means in candidate_means),
            )
        )
    results = []
    for index, cand in enumerate(candidates):
        deviations = [band.candidate_means[index] - statistics.fmean(band.reference_means) for band in windows]
        outside = tuple(band.first for band in windows if not band.low <= band.candidate_means[index] <= band.high)
        results.append(
            CandidateBand(
                label=cand.label,
                inside=not outside,
                windows_outside=outside,
                max_abs_deviation=max(abs(d) for d in deviations),
                final_deviation=deviations[-1],
            )
        )
    return MetricBand(
        metric=metric,
        delta=delta,
        spread=spread,
        fixed_half_width=half_width is not None,
        windows=tuple(windows),
        candidates=tuple(results),
    )


def band_test(
    references: Sequence[Trajectory],
    candidates: Sequence[Trajectory],
    window: int,
    loss_half_width: float | None = None,
) -> BandReport:
    """Test every candidate against the run-to-run band of the references (see the module docstring).

    ``loss_half_width``, when given, replaces the references' spread as the ``lm loss`` band's half-width: a
    fixed tolerance around the references for a test whose references do not share the candidate's batches.
    The ``grad norm`` band keeps the references' spread either way.

    Raises ValueError with fewer than two references or no candidate, when the trajectories cover different
    ranges or come from different sources, when the range is not a whole number of windows, and when a
    reference has a skipped or NaN iteration or logs a different learning rate or consumed-sample count
    from the first reference.
    """
    if len(references) < 2:
        raise ValueError(f"a band needs at least two reference runs, got {len(references)}")
    if not candidates:
        raise ValueError("a band test needs at least one candidate run")
    runs = [*references, *candidates]
    _same_range(runs)
    sources = {run.source.partition(":")[0] for run in runs}
    if len(sources) != 1:
        raise ValueError(f"trajectories mix sources {sorted(sources)}; compare runs read the same way")
    first, last = references[0].first, references[0].last
    if window < 1 or (last - first + 1) % window:
        raise ValueError(f"iterations {first}-{last} are not a whole number of {window}-iteration windows")

    anchor = references[0]
    for ref in references:
        problems = []
        if ref.skipped_total or ref.nan_total:
            problems.append(f"{ref.skipped_total} skipped and {ref.nan_total} NaN iterations")
        lr = first_difference(first, anchor.values[SCHEDULE_METRIC], ref.values[SCHEDULE_METRIC])
        consumed = first_difference(first, anchor.consumed_samples, ref.consumed_samples)
        if lr is not None:
            problems.append(f"learning rate differs from {anchor.label} at iteration {lr.iteration}")
        if consumed is not None:
            problems.append(f"consumed samples differ from {anchor.label} at iteration {consumed.iteration}")
        if problems:
            raise ValueError(f"reference {ref.label} is not a clean reference: {'; '.join(problems)}")

    metrics = (
        _metric_band(VERDICT_METRIC, references, candidates, window, loss_half_width),
        _metric_band(FLAG_METRIC, references, candidates, window, None),
    )
    loss_band, grad_band = metrics
    verdicts = []
    for index, cand in enumerate(candidates):
        lr = first_difference(first, anchor.values[SCHEDULE_METRIC], cand.values[SCHEDULE_METRIC])
        consumed = first_difference(first, anchor.consumed_samples, cand.consumed_samples)
        loss_inside = loss_band.candidates[index].inside
        matches = _matches_the_references(lr, consumed, cand.skipped_total, cand.nan_total)
        verdicts.append(
            CandidateVerdict(
                label=cand.label,
                verdict=PASS if loss_inside and matches else FAIL,
                grad_norm=PASS if grad_band.candidates[index].inside else FLAG,
                loss_inside=loss_inside,
                first_learning_rate_difference=lr,
                first_consumed_samples_difference=consumed,
                skipped_total=cand.skipped_total,
                nan_total=cand.nan_total,
            )
        )
    return BandReport(
        first=first,
        last=last,
        window=window,
        references=tuple(ref.label for ref in references),
        source=anchor.source.partition(":")[0],
        metrics=metrics,
        verdicts=tuple(verdicts),
    )


def identity_test(reference: Trajectory, candidate: Trajectory) -> IdentityReport:
    """Compare the candidate with the reference iteration by iteration, per metric and for consumed samples.

    Raises ValueError when the two cover different ranges.
    """
    _same_range([reference, candidate])
    first = reference.first
    n = reference.last - reference.first + 1
    compared = [(metric, reference.values[metric], candidate.values[metric]) for metric in METRIC_SOURCES]
    compared.append(("consumed samples", reference.consumed_samples, candidate.consumed_samples))
    metrics = []
    for metric, ref_series, cand_series in compared:
        difference = first_difference(first, ref_series, cand_series)
        leading = n if difference is None else difference.iteration - first
        metrics.append(
            MetricIdentity(
                metric=metric,
                leading_identical=leading,
                first_difference=difference,
                largest_difference=largest_difference(first, ref_series, cand_series),
            )
        )
    return IdentityReport(
        first=first,
        last=reference.last,
        reference=reference.label,
        candidate=candidate.label,
        source=f"{reference.source} vs {candidate.source}",
        metrics=tuple(metrics),
        identical=all(m.first_difference is None for m in metrics),
    )


def format_band_report(report: BandReport) -> str:
    """Render a band test as the CLI's human-readable report: inputs, one table per metric, verdicts."""
    rows = [
        f"iterations              {report.first}-{report.last} in {report.window}-iteration windows ({report.source})",
        *(f"reference {i + 1}             {label}" for i, label in enumerate(report.references)),
        *(f"candidate {i + 1}             {v.label}" for i, v in enumerate(report.verdicts)),
    ]
    for band in report.metrics:
        rows.append("")
        if band.fixed_half_width:
            rows.append(
                f"{band.metric}: delta {band.delta:.6f} (fixed half-width; reference spread {band.spread:.6f})"
            )
        else:
            rows.append(f"{band.metric}: delta {band.delta:.6f} (largest reference-pair window difference)")
        header = ["window".ljust(11)]
        header += [f"ref {i + 1}".rjust(10) for i in range(len(report.references))]
        header += ["band low".rjust(10), "band high".rjust(10)]
        header += [f"cand {i + 1}".rjust(10) + "  dev".rjust(11) for i in range(len(report.verdicts))]
        rows.append(" ".join(header))
        for w in band.windows:
            ref_mean = statistics.fmean(w.reference_means)
            cells = [f"{w.first}-{w.last}".ljust(11)]
            cells += [f"{m:10.5f}" for m in w.reference_means]
            cells += [f"{w.low:10.5f}", f"{w.high:10.5f}"]
            for m in w.candidate_means:
                mark = " " if w.low <= m <= w.high else "*"
                cells.append(f"{m:10.5f} {m - ref_mean:+10.5f}{mark}")
            rows.append(" ".join(cells))
        for cand in band.candidates:
            rows.append(
                f"  {cand.label}: {'inside' if cand.inside else 'OUTSIDE in windows ' + str(list(cand.windows_outside))}"
                f"; max |dev| {cand.max_abs_deviation:.6f}; final-window dev {cand.final_deviation:+.6f}"
            )
    rows.append("")
    for v in report.verdicts:
        notes = []
        if v.first_learning_rate_difference is not None:
            notes.append(f"learning rate differs at iteration {v.first_learning_rate_difference.iteration}")
        if v.first_consumed_samples_difference is not None:
            notes.append(f"consumed samples differ at iteration {v.first_consumed_samples_difference.iteration}")
        if v.skipped_total or v.nan_total:
            notes.append(f"{v.skipped_total} skipped / {v.nan_total} NaN iterations")
        rows.append(
            f"verdict {v.label}: {v.verdict} (lm loss {'inside' if v.loss_inside else 'outside'} its band; "
            f"grad norm {v.grad_norm}{'; ' + '; '.join(notes) if notes else ''})"
        )
    return "\n".join(rows)


def format_identity_report(report: IdentityReport) -> str:
    """Render an identity test as the CLI's human-readable report."""
    n = report.last - report.first + 1
    rows = [
        f"{report.candidate} vs reference {report.reference}",
        f"  iterations {report.first}-{report.last} ({report.source}): "
        f"{'IDENTICAL' if report.identical else 'DIFFERENT'}",
    ]
    for m in report.metrics:
        if m.first_difference is None:
            rows.append(f"  {m.metric:17s} identical at all {n} iterations")
        else:
            d, big = m.first_difference, m.largest_difference
            rows.append(
                f"  {m.metric:17s} identical for {m.leading_identical} leading iterations; first difference at "
                f"iteration {d.iteration}: {d.reference!r} vs {d.candidate!r}; largest |difference| "
                f"{abs(big.candidate - big.reference):.3e} at iteration {big.iteration}"
            )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI: the ``band`` and ``identity`` tests over logs of runs with one seed and data order, and
    ``windows-outside`` over a band report."""
    parser = argparse.ArgumentParser(description="Loss-trajectory parity between runs of one seed and data order.")
    tests = parser.add_subparsers(dest="test", required=True)
    for name, help_text in (
        ("band", "numerics-changing lever: candidates must stay inside the references' run-to-run band"),
        ("identity", "exact lever or knob-off path: candidates must equal the reference at every iteration"),
    ):
        sub = tests.add_parser(name, help=help_text)
        sub.add_argument(
            "--reference",
            type=Path,
            nargs="+" if name == "band" else 1,
            required=True,
            help="Reference run log(s): two or more for band, one for identity",
        )
        sub.add_argument("--candidate", type=Path, nargs="+", required=True, help="Candidate run log(s)")
        sub.add_argument(
            "--iterations", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"), help="Inclusive range"
        )
        if name == "band":
            sub.add_argument("--window", type=int, required=True, help="Window length; must divide the range")
        sub.add_argument(
            "--wandb",
            action="store_true",
            help="Read lm loss, grad norm and learning rate from the W&B run each log names (full precision)",
        )
        sub.add_argument("--json", action="store_true", help="Emit the report as JSON")
    outside = tests.add_parser(
        "windows-outside",
        help="print the loss windows a band report's one candidate left the band in, or 'other' when it failed "
        "another check",
    )
    outside.add_argument("report", type=Path, help="A band --json report")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested test, print its report and return 0 when every candidate passes, 1 otherwise; or print a
    band report's loss windows outside (``windows-outside``) and return 0."""
    args = build_parser().parse_args(argv)
    if args.test == "windows-outside":
        windows = loss_windows_outside(BandReport.from_dict(json.loads(args.report.read_text())))
        print("other" if windows is None else windows)
        return 0
    iterations = (args.iterations[0], args.iterations[1])
    references = [load_trajectory(path, iterations, args.wandb) for path in args.reference]
    candidates = [load_trajectory(path, iterations, args.wandb) for path in args.candidate]
    if args.test == "band":
        report = band_test(references, candidates, args.window)
        print(json.dumps(report.to_dict(), indent=2) if args.json else format_band_report(report))
        return 0 if all(v.verdict == PASS for v in report.verdicts) else 1
    reports = [identity_test(references[0], cand) for cand in candidates]
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        print("\n".join(format_identity_report(r) for r in reports))
    return 0 if all(r.identical for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
