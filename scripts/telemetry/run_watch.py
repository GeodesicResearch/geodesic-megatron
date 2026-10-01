#!/usr/bin/env python3
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

"""Check a training stage's logs, as they stand, against a watch spec fixed before the stage runs.

A stage's logs are its segments' training logs in order. A segment that resumes from a save re-runs the iterations
after it, so every earlier segment's records, saves and rejected results at or after the first iteration a later
segment has logged belong to a superseded run and are dropped; a segment that has not logged an iteration yet
supersedes nothing. A log's last line is read only once its writer has finished it. A watch spec (YAML) names three
kinds of check:

- ``stop``: conditions under which the stage must not continue. ``non_finite_grad_norm``: an iteration whose grad
  norm is inf or nan. ``non_finite_lm_loss``: an iteration whose lm loss is inf or nan, or whose line has no
  ``lm loss``, which is how a NaN loss is logged when the loss NaN check is off. ``nan_or_skipped_iterations``: an
  iteration line counting a nan or skipped iteration. ``rejected_result``: a result the rerun state machine rejected,
  which ends the run when the check is fatal (the gradient NaN check, ``ddp.check_for_nan_in_grad``, raises before
  the iteration's line is written). ``max_alloc_retries``: a segment's ``[peak-memory]`` summary with more
  allocator retries than this. ``launch_settings``: the stage's ISAMBARD_ENV_OVERRIDES file (relative to the
  repository root), read with the launcher's own parser; every segment that has logged an iteration must have logged
  ``[env-overrides]`` lines holding exactly its KEY=value pairs, no key missing, different or extra, so a segment
  launched without the file, or with another, stops the stage. A stop line names the latest save, and says when that
  save came after the first bad iteration (for a launch-settings stop, the segment's first iteration), so it holds
  weights trained past it and the stage must resume from the save before it.
- ``loss_gates``: gates of a pre-registered loss-gate spec (``loss_gate.py``), each evaluated once one log covers its
  range; a gate whose range spans a restart is NOT EVALUATED, with the reason. A gate named with ``--decided
  GATE=LOG`` passed at an earlier check on that log and, while that log still covers its range, is reported as PASS
  without being evaluated again: its range is fixed, so its outcome on that log cannot change. A segment that
  supersedes the range makes another log cover it, and the gate is evaluated anew. Each gate's line names the log it
  was decided on, and a last ``undecided gates:`` line names every watched gate that has not passed.
- ``flags``: conditions reported to a person, never a stop.
  ``loss_spike`` (``above_trailing_mean``, ``trailing_iterations``): an lm loss more than that above the mean of the
  iterations before it.
  ``growing_offset`` (``gate``, ``windows``, ``above``): once that loss gate has been evaluated, its candidate's
  lm-loss offset from the references' mean, averaged over its last ``windows`` windows, exceeding the average over
  its first ``windows`` by more than ``above``. The offsets and their least-squares slope are printed whenever the
  gate has a report.
  ``block_envelope`` (``reference``, ``envelope``, ``block_iterations``, ``consecutive_blocks``): over every block
  of iterations the candidate has logged in full, the candidate's mean lm loss minus the reference's, against the
  envelope run's minus the reference's; a candidate further from the reference than the envelope run is, in
  absolute value, in ``consecutive_blocks`` adjacent blocks. Each of the two runs is its segment logs in order
  (``logs``) and, optionally, the blocks read from a W&B run's history instead (``wandb_blocks``, block number to
  ``<entity>/<project>/<run id>``), for a segment that left no log; such a block's line names its source. A block
  either run does not cover in full is printed as not compared, with the reason, and breaks a run of blocks.

The stop checks are one outcome (FAIL when any condition holds, PASS otherwise) and each due gate another; the exit
status is ``gate_outcome.exit_status``'s over them. Every stop condition, gate outcome, trend and flag is printed as
one line. Exit status 1 is a stop, so a failure of the watch itself never reads as one: a spec or log that cannot be
read, or a stop check that cannot be evaluated (a launch-settings file the launcher refuses, an ``[env-overrides]``
line that is not KEY=value), print a ``NOT EVALUATED`` line and count as NOT EVALUATED (exit status 2), and a trend or
flag that cannot be computed prints a ``FLAG ... not computed`` line and leaves the exit status alone. Each stop check
is evaluated on its own, so one that cannot be leaves the others standing, and a stop any of them finds still counts.

USAGE
    python scripts/telemetry/run_watch.py --spec watch.yaml --log segment1.out [--log segment2.out ...]
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import yaml


# Run as a script, only scripts/telemetry/ is on sys.path; the repo root makes the sibling modules
# importable the same way from every entry point.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS, exit_status  # noqa: E402
from scripts.telemetry.loss_gate import GateResult, evaluate_gate, load_gate_spec  # noqa: E402
from scripts.telemetry.loss_parity import (  # noqa: E402
    VERDICT_METRIC,
    fetch_wandb_history,
    offset_rise,
    offsets_from_reference_mean,
)
from scripts.telemetry.training_log import (  # noqa: E402
    IterationRecord,
    env_override_lines,
    parse_env_override_lines,
    parse_iteration_records,
    parse_peak_memory_across_ranks,
    read_log_lines,
)
from scripts.training.launcher_source import env_override_entries  # noqa: E402


_SAVED_RE = re.compile(r"successfully saved checkpoint from iteration\s+(\d+)")
# The error megatron/core/rerun_state_machine.py raises when a fatal validation rejects a result with reruns disabled.
# Its iteration is one past the iteration that failed: Bridge's train loop seeds the machine's counter with the number
# of completed steps, where Megatron-Core expects one less.
_REJECTED_RE = re.compile(
    r"Rank (?P<rank>\d+), node (?P<node>\S+), device \d+, iteration (?P<iteration>\d+): (?P<result>Unexpected result .*)$"
)


@dataclass(frozen=True)
class RejectedResult:
    """One rank's report of a result the rerun state machine rejected while training ``iteration``."""

    rank: int
    node: str
    iteration: int
    result: str


@dataclass(frozen=True)
class LossSpike:
    """An lm loss more than ``above_trailing_mean`` above the mean of the ``trailing_iterations`` before it."""

    above_trailing_mean: float
    trailing_iterations: int


@dataclass(frozen=True)
class GrowingOffset:
    """A loss gate's offset averaged over its last ``windows`` windows above the first ones' by more than ``above``."""

    gate: str
    windows: int
    above: float


@dataclass(frozen=True)
class BlockRun:
    """A run the candidate is compared with per block: its segment logs in order, and the blocks read from a W&B
    run's history instead."""

    logs: tuple[Path, ...]
    wandb_blocks: dict[int, str]


@dataclass(frozen=True)
class BlockEnvelope:
    """The candidate further from the reference than the envelope run, per block, ``consecutive_blocks`` in a row."""

    reference: BlockRun
    envelope: BlockRun
    block_iterations: int
    consecutive_blocks: int


@dataclass(frozen=True)
class BlockComparison:
    """A block the candidate has logged in full: (candidate - reference, envelope run - reference) mean lm loss, or,
    when a compared run does not cover it in full, why it is not compared; ``sources`` names each compared run whose
    block came from W&B rather than its logs."""

    block: int
    offsets: tuple[float, float] | None
    not_compared: str | None
    sources: tuple[str, ...]


@dataclass(frozen=True)
class WatchSpec:
    """A stage's stop conditions, its pre-registered loss gates and its flags (see the module docstring)."""

    stop_on_non_finite_grad_norm: bool
    stop_on_non_finite_lm_loss: bool
    stop_on_nan_or_skipped_iterations: bool
    stop_on_rejected_result: bool
    max_alloc_retries: int | None
    launch_settings: Path | None
    loss_gate_spec: Path | None
    loss_gates: tuple[str, ...]
    loss_spike: LossSpike | None
    growing_offset: GrowingOffset | None
    block_envelope: BlockEnvelope | None


def _section(raw: dict, name: str, keys: set[str], path: Path, parent: str | None = None) -> dict:
    """The named mapping of ``raw`` (empty when absent), refusing a key outside ``keys``."""
    section = raw.get(name) or {}
    extra = sorted(set(section) - keys)
    if extra:
        raise ValueError(f"{path}: unknown {f'{parent}.{name}' if parent else name} keys {extra}")
    return section


def _block_run(raw_envelope: dict, name: str, path: Path) -> BlockRun:
    """The block envelope's named run, refusing a key other than ``logs`` and ``wandb_blocks``."""
    run = _section(raw_envelope, name, {"logs", "wandb_blocks"}, path, parent="block_envelope")
    wandb_blocks = run.get("wandb_blocks") or {}
    return BlockRun(tuple(Path(log) for log in run["logs"]), {int(b): str(r) for b, r in wandb_blocks.items()})


def load_watch_spec(path: Path) -> WatchSpec:
    """Read a watch spec, resolving its loss-gate spec against the repository root.

    Raises ValueError on an unknown section or key, or a growing-offset gate that is not one of the watched loss
    gates; KeyError on a flag missing one of its fields.
    """
    raw = yaml.safe_load(Path(path).read_text())
    unknown = sorted(set(raw) - {"stop", "loss_gates", "flags"})
    if unknown:
        raise ValueError(f"{path}: unknown sections {unknown}")
    stop_keys = {
        "non_finite_grad_norm",
        "non_finite_lm_loss",
        "nan_or_skipped_iterations",
        "rejected_result",
        "max_alloc_retries",
        "launch_settings",
    }
    stop = _section(raw, "stop", stop_keys, path)
    gates = _section(raw, "loss_gates", {"spec", "gates"}, path)
    flags = _section(raw, "flags", {"loss_spike", "growing_offset", "block_envelope"}, path)
    gate_names = tuple(gates.get("gates") or ())
    spike = growing = envelope = None
    if "loss_spike" in flags:
        spike = LossSpike(
            float(flags["loss_spike"]["above_trailing_mean"]), int(flags["loss_spike"]["trailing_iterations"])
        )
    if "growing_offset" in flags:
        raw_growing = flags["growing_offset"]
        growing = GrowingOffset(raw_growing["gate"], int(raw_growing["windows"]), float(raw_growing["above"]))
        if growing.gate not in gate_names:
            raise ValueError(f"{path}: growing_offset gate {growing.gate} is not a watched loss gate")
    if "block_envelope" in flags:
        raw_envelope = flags["block_envelope"]
        envelope = BlockEnvelope(
            _block_run(raw_envelope, "reference", path),
            _block_run(raw_envelope, "envelope", path),
            int(raw_envelope["block_iterations"]),
            int(raw_envelope["consecutive_blocks"]),
        )
    return WatchSpec(
        stop_on_non_finite_grad_norm=bool(stop.get("non_finite_grad_norm", False)),
        stop_on_non_finite_lm_loss=bool(stop.get("non_finite_lm_loss", False)),
        stop_on_nan_or_skipped_iterations=bool(stop.get("nan_or_skipped_iterations", False)),
        stop_on_rejected_result=bool(stop.get("rejected_result", False)),
        max_alloc_retries=int(stop["max_alloc_retries"]) if "max_alloc_retries" in stop else None,
        launch_settings=Path(_REPO_ROOT) / stop["launch_settings"] if "launch_settings" in stop else None,
        loss_gate_spec=Path(_REPO_ROOT) / gates["spec"] if "spec" in gates else None,
        loss_gates=gate_names,
        loss_spike=spike,
        growing_offset=growing,
        block_envelope=envelope,
    )


@dataclass(frozen=True)
class SegmentLog:
    """One segment's training log, read once: its iteration records, its peak-memory summary (None if absent), the
    iterations it saved, its nodes' ``[env-overrides]`` lines as logged (parsed only by the launch-settings check, so a
    line that cannot be parsed leaves the other checks standing) and the results the rerun state machine rejected."""

    path: Path
    records: list[IterationRecord]
    peak_memory: dict[str, float | int] | None
    saves: list[int]
    env_override_lines: list[str]
    rejected: list[RejectedResult]


def _read_segment(log: Path) -> SegmentLog:
    lines = read_log_lines(log)
    rejected = [
        RejectedResult(int(match["rank"]), match["node"], int(match["iteration"]) - 1, match["result"])
        for line in lines
        if (match := _REJECTED_RE.search(line))
    ]
    return SegmentLog(
        log,
        parse_iteration_records(lines),
        parse_peak_memory_across_ranks(lines),
        [int(match.group(1)) for line in lines if (match := _SAVED_RE.search(line))],
        env_override_lines(lines),
        rejected,
    )


def read_segments(logs: list[Path] | tuple[Path, ...]) -> list[SegmentLog]:
    """Read each segment log once, in the stage's order, and drop from each what a later segment re-ran: its records,
    saves and rejected results at or after the first iteration any later segment has logged."""
    segments = [_read_segment(log) for log in logs]
    kept: list[SegmentLog] = []
    resumed_at = math.inf
    for segment in reversed(segments):
        kept.append(
            replace(
                segment,
                records=[r for r in segment.records if r.iteration < resumed_at],
                saves=[s for s in segment.saves if s < resumed_at],
                rejected=[r for r in segment.rejected if r.iteration < resumed_at],
            )
        )
        if segment.records:
            resumed_at = min(resumed_at, min(r.iteration for r in segment.records))
    return kept[::-1]


def latest_records(segments: list[SegmentLog]) -> list[IterationRecord]:
    """Every iteration's latest record across the stage's segments, in iteration order."""
    latest: dict[int, IterationRecord] = {}
    for segment in segments:
        for record in segment.records:
            latest[record.iteration] = record
    return [latest[iteration] for iteration in sorted(latest)]


def _save_note(segments: list[SegmentLog], first_bad: int) -> str:
    """Where the latest save stands relative to the first bad iteration."""
    saves = [save for segment in segments for save in segment.saves]
    if not saves:
        return "no save yet"
    latest = max(saves)
    if latest < first_bad:
        return f"latest save iteration {latest}, before it"
    before = [save for save in saves if save < first_bad]
    resume = f"resume from iteration {max(before)}" if before else "no earlier save: restart from scratch"
    return f"latest save iteration {latest} holds the bad step's weights; {resume}"


def launch_settings_differences(expected: dict[str, str], logged: dict[str, str]) -> list[str]:
    """How one ``[env-overrides]`` line differs from the stage's launch settings: each key missing, different or
    extra."""
    differences = [f"{key} missing (expected {value})" for key, value in expected.items() if key not in logged]
    differences += [
        f"{key}={logged[key]} (expected {value})"
        for key, value in expected.items()
        if key in logged and logged[key] != value
    ]
    differences += [f"{key}={value} extra" for key, value in logged.items() if key not in expected]
    return differences


def launch_settings_stops(settings: Path, segments: list[SegmentLog]) -> list[str]:
    """A stop line for every segment that has logged an iteration without logging exactly the stage's launch
    settings on every node, naming where the latest save stands against the segment's first iteration.

    Raises ValueError on a settings file the launcher refuses, or an ``[env-overrides]`` field that is not KEY=value.
    """
    expected = dict(entry.split("=", 1) for entry in env_override_entries(settings))
    stops = []
    for segment in segments:
        if not segment.records:
            continue
        logged = parse_env_override_lines(segment.env_override_lines)
        if expected and not logged:
            differences = ["no [env-overrides] line"]
        else:
            differences = sorted({d for line in logged for d in launch_settings_differences(expected, line)})
        if differences:
            note = _save_note(segments, min(r.iteration for r in segment.records))
            stops.append(f"{segment.path.name}: trained without {settings.name}: {'; '.join(differences)}; {note}")
    return stops


def rejected_result_stop(segments: list[SegmentLog]) -> str | None:
    """A stop line for the earliest iteration at which a rank's result was rejected, naming the first rank to report
    it, how many ranks did, and where the latest save stands."""
    rejected = [r for segment in segments for r in segment.rejected]
    if not rejected:
        return None
    first_iteration = min(r.iteration for r in rejected)
    at_first = [r for r in rejected if r.iteration == first_iteration]
    first = at_first[0]
    ranks = len({r.rank for r in at_first})
    return (
        f"rank {first.rank} on {first.node} rejected a result at iteration {first_iteration} "
        f"({ranks} rank{'s' if ranks != 1 else ''} in all): {first.result}; {_save_note(segments, first_iteration)}"
    )


def _lm_loss_problem(record: IterationRecord) -> str | None:
    if record.lm_loss is None:
        return "logged no lm loss"
    if not math.isfinite(record.lm_loss):
        return f"logged a non-finite lm loss ({record.lm_loss})"
    return None


def non_finite_grad_norm_stops(segments: list[SegmentLog], records: list[IterationRecord]) -> list[str]:
    """A stop line for the first iteration whose grad norm is inf or nan."""
    bad = [r for r in records if r.grad_norm is not None and not math.isfinite(r.grad_norm)]
    if not bad:
        return []
    note = _save_note(segments, bad[0].iteration)
    return [f"non-finite grad norm at iteration {bad[0].iteration} ({bad[0].grad_norm}); {note}"]


def non_finite_lm_loss_stops(segments: list[SegmentLog], records: list[IterationRecord]) -> list[str]:
    """A stop line for the first iteration whose lm loss is inf or nan, or absent from its line."""
    bad = [(r, problem) for r in records if (problem := _lm_loss_problem(r))]
    if not bad:
        return []
    record, problem = bad[0]
    return [f"iteration {record.iteration} {problem}; {_save_note(segments, record.iteration)}"]


def nan_or_skipped_stops(segments: list[SegmentLog], records: list[IterationRecord]) -> list[str]:
    """A stop line for the first iteration line that counts a nan or skipped iteration."""
    counted = [r for r in records if r.nan_total > 0 or r.skipped_total > 0]
    if not counted:
        return []
    record = counted[0]
    return [
        f"iteration {record.iteration} counts {record.nan_total} nan and {record.skipped_total} skipped "
        f"iterations; {_save_note(segments, record.iteration)}"
    ]


def alloc_retries_stops(segments: list[SegmentLog], limit: int) -> list[str]:
    """A stop line for every segment whose peak-memory summary counts more allocator retries than ``limit``."""
    return [
        f"{segment.path.name}: {segment.peak_memory['max_alloc_retries']} allocator retries on one rank, limit {limit}"
        for segment in segments
        if segment.peak_memory is not None and segment.peak_memory["max_alloc_retries"] > limit
    ]


def stop_checks(
    spec: WatchSpec, segments: list[SegmentLog], records: list[IterationRecord]
) -> dict[str, Callable[[], list[str]]]:
    """The spec's stop checks by their spec key, each returning one line per condition it finds to hold, naming
    where the condition first holds and where the latest save stands. Each is evaluated on its own, so one that
    cannot be evaluated leaves the others standing."""
    checks: dict[str, Callable[[], list[str]]] = {}
    if spec.stop_on_non_finite_grad_norm:
        checks["non_finite_grad_norm"] = lambda: non_finite_grad_norm_stops(segments, records)
    if spec.stop_on_non_finite_lm_loss:
        checks["non_finite_lm_loss"] = lambda: non_finite_lm_loss_stops(segments, records)
    if spec.stop_on_nan_or_skipped_iterations:
        checks["nan_or_skipped_iterations"] = lambda: nan_or_skipped_stops(segments, records)
    if spec.stop_on_rejected_result:
        checks["rejected_result"] = lambda: [stop for stop in (rejected_result_stop(segments),) if stop]
    if spec.max_alloc_retries is not None:
        checks["max_alloc_retries"] = lambda: alloc_retries_stops(segments, spec.max_alloc_retries)
    if spec.launch_settings is not None:
        checks["launch_settings"] = lambda: launch_settings_stops(spec.launch_settings, segments)
    return checks


def rises_above_trailing_mean(records: list[IterationRecord], trailing: int) -> list[tuple[int, float, float]]:
    """(iteration, lm loss, mean of the ``trailing`` logged losses before it) for every iteration that has them."""
    losses = [(r.iteration, r.lm_loss) for r in records if r.lm_loss is not None]
    return [
        (iteration, loss, statistics.fmean(value for _, value in losses[index - trailing : index]))
        for index, (iteration, loss) in enumerate(losses)
        if index >= trailing
    ]


def loss_spikes(spec: WatchSpec, records: list[IterationRecord]) -> list[str]:
    """Flag lines for every lm loss above the trailing mean by more than the spec's threshold."""
    if spec.loss_spike is None:
        return []
    return [
        f"loss spike at iteration {iteration}: {loss:.4f} against a trailing mean of {mean:.4f}"
        for iteration, loss, mean in rises_above_trailing_mean(records, spec.loss_spike.trailing_iterations)
        if loss - mean > spec.loss_spike.above_trailing_mean
    ]


def covering_log(segments: list[SegmentLog], first: int, last: int) -> Path | None:
    """The last segment log that holds every iteration of [first, last], or None when no single log does."""
    for segment in reversed(segments):
        iterations = {r.iteration for r in segment.records}
        if all(iteration in iterations for iteration in range(first, last + 1)):
            return segment.path
    return None


DECIDED_REASON = "passed at an earlier check on this log"


def due_gates(
    spec: WatchSpec, segments: list[SegmentLog], last_iteration: int, decided: dict[str, Path]
) -> list[tuple[GateResult, Path | None]]:
    """Each watched loss gate whose range the logs have reached, with the one log that covers the range (None when
    none does), evaluated on that log unless ``decided`` maps the gate to that same log."""
    if spec.loss_gate_spec is None:
        return []
    gate_spec = load_gate_spec(spec.loss_gate_spec)
    results = []
    for name in spec.loss_gates:
        gate = gate_spec.gates[name]
        if last_iteration < gate.last:
            continue
        log = covering_log(segments, gate.first, gate.last)
        if log is None:
            reason = f"no single log covers iterations {gate.first}-{gate.last} (the range spans a restart)"
            results.append((GateResult(name, NOT_EVALUATED, reason, None), None))
        elif decided.get(name) == log:
            results.append((GateResult(name, PASS, DECIDED_REASON, None), log))
        else:
            results.append((evaluate_gate(gate_spec, name, log), log))
    return results


def parse_decided(values: list[str], spec: WatchSpec) -> dict[str, Path]:
    """The ``--decided GATE=LOG`` values as a mapping. Raises ValueError on a value without ``=`` or a gate the spec
    does not watch."""
    decided = {}
    for value in values:
        name, sep, log = value.partition("=")
        if not sep or name not in spec.loss_gates:
            raise ValueError(f"--decided {value!r} is not GATE=LOG for a watched gate {list(spec.loss_gates)}")
        decided[name] = Path(log)
    return decided


def _slope(xs: list[float], ys: list[float]) -> float:
    """The least-squares slope of ys on xs."""
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)


def gate_offsets(spec: WatchSpec, results: list[GateResult]) -> tuple[list[float], list[float]] | None:
    """The growing-offset gate's per-window lm-loss offsets from its references' mean and the windows' centres,
    or None until that gate has a report."""
    if spec.growing_offset is None:
        return None
    for result in results:
        if result.gate == spec.growing_offset.gate and result.report is not None:
            (loss,) = [band for band in result.report.metrics if band.metric == VERDICT_METRIC]
            return offsets_from_reference_mean(loss.windows, 0), [
                (band.first + band.last) / 2 for band in loss.windows
            ]
    return None


def offset_trend_line(gate: str, offsets: list[float], centres: list[float]) -> str:
    """The gate's offsets and their least-squares slope per 1000 iterations, as one line."""
    slope = _slope(centres, offsets) * 1000 if len(offsets) > 1 else 0.0
    series = ", ".join(f"{offset:+.4f}" for offset in offsets)
    return f"{gate}: offsets from the references' mean {series}; slope {slope:+.4f} per 1000 iterations"


def growing_offset_flag(rule: GrowingOffset, offsets: list[float]) -> str | None:
    """A flag line when the last windows' mean offset exceeds the first windows' by more than the rule allows."""
    if len(offsets) < 2 * rule.windows:
        return None
    rise = offset_rise(offsets, rule.windows)
    if rise <= rule.above:
        return None
    return f"{rule.gate}: the last {rule.windows} windows' offset is {rise:+.4f} above the first's"


def _block_losses(records: list[IterationRecord], block: int) -> dict[int, list[float]]:
    """The logged lm losses of each block of ``block`` iterations, by block number (from 1)."""
    by_block: dict[int, list[float]] = {}
    for record in records:
        if record.lm_loss is not None:
            by_block.setdefault((record.iteration - 1) // block + 1, []).append(record.lm_loss)
    return by_block


def _run_block(run: BlockRun, logged: dict[int, list[float]], block: int, size: int) -> tuple[list[float], str | None]:
    """One block's lm losses of a compared run, and its W&B run when the block is read from that run's history
    rather than the logs."""
    if block not in run.wandb_blocks:
        return logged.get(block, []), None
    first = (block - 1) * size + 1
    wandb_run = run.wandb_blocks[block]
    return list(fetch_wandb_history(wandb_run, (first, first + size - 1))[VERDICT_METRIC]), wandb_run


def block_comparisons(rule: BlockEnvelope, records: list[IterationRecord]) -> list[BlockComparison]:
    """Every block the candidate has logged in full, in order, compared where both runs cover it too."""
    size = rule.block_iterations
    candidate = _block_losses(records, size)
    full_blocks = sorted(number for number, losses in candidate.items() if len(losses) == size)
    if not full_blocks:
        return []
    runs = {"reference": rule.reference, "envelope run": rule.envelope}
    logged = {name: _block_losses(latest_records(read_segments(run.logs)), size) for name, run in runs.items()}
    comparisons = []
    for block in full_blocks:
        means, short, sources = {}, [], []
        for name, run in runs.items():
            try:
                losses, wandb_run = _run_block(run, logged[name], block, size)
            except Exception as error:  # noqa: BLE001 - a W&B read that cannot cover the block leaves it not compared
                short.append(f"the {name}'s W&B history: {type(error).__name__}: {error}")
                continue
            if len(losses) != size:
                short.append(f"the {name} logs {len(losses)} of {size} iterations")
                continue
            means[name] = statistics.fmean(losses)
            if wandb_run is not None:
                sources.append(f"{name} from wandb:{wandb_run}")
        if short:
            comparisons.append(BlockComparison(block, None, "; ".join(short), tuple(sources)))
            continue
        mine = statistics.fmean(candidate[block])
        offsets = (mine - means["reference"], means["envelope run"] - means["reference"])
        comparisons.append(BlockComparison(block, offsets, None, tuple(sources)))
    return comparisons


def _block_entry(comparison: BlockComparison) -> str:
    if comparison.offsets is None:
        return f"block {comparison.block}: not compared ({comparison.not_compared})"
    mine, theirs = comparison.offsets
    sources = f" ({', '.join(comparison.sources)})" if comparison.sources else ""
    return f"block {comparison.block}: {mine:+.4f} against {theirs:+.4f}{sources}"


def block_line(comparisons: list[BlockComparison]) -> str:
    """Every block's two offsets and the W&B runs any came from, or why it is not compared, as one line."""
    blocks = "; ".join(_block_entry(c) for c in comparisons)
    return f"blocks (candidate - reference against envelope - reference): {blocks}"


def envelope_flag(rule: BlockEnvelope, comparisons: list[BlockComparison]) -> str | None:
    """A flag line, naming the blocks and any W&B runs they came from, when the candidate is further from the
    reference than the envelope run in the rule's number of adjacent compared blocks."""
    run: list[BlockComparison] = []
    longest: list[BlockComparison] = []
    for c in comparisons:
        if c.offsets is None or abs(c.offsets[0]) <= abs(c.offsets[1]):
            run = []
        else:
            run = [*run, c] if run and c.block == run[-1].block + 1 else [c]
        longest = run if len(run) > len(longest) else longest
    if len(longest) < rule.consecutive_blocks:
        return None
    sources = "; ".join(f"block {c.block}: {', '.join(c.sources)}" for c in longest if c.sources)
    return (
        f"outside the envelope in {len(longest)} consecutive blocks of {rule.block_iterations} iterations: "
        f"blocks {longest[0].block}-{longest[-1].block}{f' ({sources})' if sources else ''}"
    )


def _error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _report_lines(
    name: str, compute: Callable[[], tuple[list[str], list[str | None]]]
) -> tuple[list[str], list[str | None]]:
    """A report's trend and flag lines, or one line saying it could not be computed: a report never changes the
    exit status, so its failure is printed rather than raised."""
    try:
        return compute()
    except Exception as error:  # noqa: BLE001 - printed as its own line; a report never decides the outcome
        return [], [f"{name} not computed: {_error(error)}"]


def _offset_report(spec: WatchSpec, gates: list[GateResult]) -> tuple[list[str], list[str | None]]:
    offsets = gate_offsets(spec, gates)
    if offsets is None:
        return [], []
    return [offset_trend_line(spec.growing_offset.gate, *offsets)], [
        growing_offset_flag(spec.growing_offset, offsets[0])
    ]


def _envelope_report(spec: WatchSpec, records: list[IterationRecord]) -> tuple[list[str], list[str | None]]:
    if spec.block_envelope is None:
        return [], []
    comparisons = block_comparisons(spec.block_envelope, records)
    if not comparisons:
        return [], []
    return [block_line(comparisons)], [envelope_flag(spec.block_envelope, comparisons)]


def main(argv: list[str] | None = None) -> int:
    """Check the stage's logs against the spec, print every stop, gate outcome, trend and flag, and return the
    exit status (see the module docstring for how a failure of the watch itself counts)."""
    parser = argparse.ArgumentParser(description="Check a training stage's logs against a watch spec.")
    parser.add_argument("--spec", type=Path, required=True, help="The watch spec YAML")
    parser.add_argument("--log", type=Path, action="append", required=True, help="A segment log; repeat, in order")
    parser.add_argument(
        "--decided",
        action="append",
        default=[],
        metavar="GATE=LOG",
        help="A watched loss gate an earlier check passed on LOG; repeat for each",
    )
    args = parser.parse_args(argv)
    # Exit status 1 is a stop, which cancels the run; nothing below may reach it by raising.
    try:
        spec = load_watch_spec(args.spec)
        decided = parse_decided(args.decided, spec)
        segments = read_segments(args.log)
    except Exception as error:  # noqa: BLE001 - reported as NOT EVALUATED, never as a stop
        print(f"NOT EVALUATED watch: {_error(error)}")
        return exit_status([NOT_EVALUATED])
    records = latest_records(segments)
    stops, unevaluated = [], False
    for name, check in stop_checks(spec, segments, records).items():
        try:
            stops += check()
        except Exception as error:  # noqa: BLE001 - reported as NOT EVALUATED, never as a stop
            print(f"NOT EVALUATED stop {name}: {_error(error)}")
            unevaluated = True
    stop_outcome = FAIL if stops else NOT_EVALUATED if unevaluated else PASS
    try:
        due = due_gates(spec, segments, records[-1].iteration if records else 0, decided)
    except Exception as error:  # noqa: BLE001 - reported as NOT EVALUATED, never as a stop
        due = [(GateResult("loss gates", NOT_EVALUATED, _error(error), None), None)]
    gates = [result for result, _ in due]
    trends, flags = [], []
    for name, compute in (
        ("loss_spike", lambda: ([], list(loss_spikes(spec, records)))),
        ("growing_offset", lambda: _offset_report(spec, gates)),
        ("block_envelope", lambda: _envelope_report(spec, records)),
    ):
        report_trends, report_flags = _report_lines(name, compute)
        trends += report_trends
        flags += report_flags
    for stop in stops:
        print(f"STOP {stop}")
    for result, log in due:
        reason = f" ({result.reason})" if result.reason else ""
        print(f"GATE {result.gate}: {result.outcome}{reason}{f' on {log}' if log is not None else ''}")
    for line in trends:
        print(f"TREND {line}")
    for flag in flags:
        if flag is not None:
            print(f"FLAG {flag}")
    passed = {result.gate for result in gates if result.outcome == PASS}
    print(f"undecided gates: {', '.join(name for name in spec.loss_gates if name not in passed) or 'none'}")
    last = records[-1].iteration if records else None
    print(f"checked through iteration {last}: {len(stops)} stop conditions, {len(gates)} gates due")
    return exit_status([stop_outcome, *(result.outcome for result in gates)])


if __name__ == "__main__":
    sys.exit(main())
