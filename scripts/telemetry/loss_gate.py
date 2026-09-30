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

"""Evaluate a pre-registered loss gate: band tests fixed in a YAML spec before the candidate run exists.

A gate spec names each reference run once, by its training log, and lists named gates. Each gate is a band
test (``loss_parity.band_test``) over two or more of those references, with its inclusive iteration range and
its window, and one of two kinds of band:

- ``lm_loss_delta``: the band test's own band, whose width is the largest window difference between two
  references; the value is the width the references produced when the spec was frozen, and the verdict is
  the band test's.
- ``lm_loss_tolerance``: a fixed band, [lowest reference window mean - tolerance, highest + tolerance] in
  every window (the band test's ``loss_half_width``); the verdict is the band test's, with its learning-rate,
  consumed-sample and NaN/skip checks.

A gate has three outcomes:

- PASS and FAIL are the verdict for the candidate.
- NOT EVALUATED means the test could not be run, or its result could not be trusted: the candidate's log
  does not yet cover the range or lacks a field; the references' band width differs from the pre-registered
  one, so the reference set is not the one the gate was calibrated on; or the candidate is one of the
  references (a run passes its own band by construction). The reason is printed.

The exit status is 0 when every evaluated gate passes, 1 when any fails, and 2 when any is not evaluated, so
a caller that must act on a FAIL never mistakes an unrunnable gate for one.

USAGE
    python scripts/telemetry/loss_gate.py --spec loss_gate.yaml --candidate run.out [--gate NAME ...] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


# Run as a script, only scripts/telemetry/ is on sys.path; the repo root makes the sibling modules
# importable the same way from every entry point.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.telemetry.loss_parity import (  # noqa: E402
    PASS,
    VERDICT_METRIC,
    BandReport,
    band_test,
    format_band_report,
    load_trajectory,
)


NOT_EVALUATED = "NOT EVALUATED"
# A pre-registered band width is written to six decimals, as the band report prints it.
DELTA_TOLERANCE = 5e-7


@dataclass(frozen=True)
class Gate:
    """One pre-registered band test: which references, over which iterations, in which windows, and its band.

    Exactly one of ``lm_loss_delta`` (the band test's own band, of this pre-registered width) and
    ``lm_loss_tolerance`` (a fixed band this far outside the references' window means) is set.
    """

    name: str
    references: tuple[str, ...]
    first: int
    last: int
    window: int
    lm_loss_delta: float | None
    lm_loss_tolerance: float | None


@dataclass(frozen=True)
class GateSpec:
    """A gate spec: the reference runs by name, the gates over them, and where metric values are read."""

    wandb: bool
    references: dict[str, Path]
    gates: dict[str, Gate]


@dataclass(frozen=True)
class GateResult:
    """One gate's outcome, the reason when it was not evaluated, and the band report when it was."""

    gate: str
    outcome: str
    reason: str
    report: BandReport | None


def load_gate_spec(path: Path) -> GateSpec:
    """Read a gate spec, refusing one whose gates could not be evaluated as written.

    Raises ValueError when a reference log is named twice, a gate names an unknown reference or one
    reference twice, a gate has fewer than two references, or a gate does not set exactly one of
    ``lm_loss_delta`` and ``lm_loss_tolerance``.
    """
    raw = yaml.safe_load(Path(path).read_text())
    references = {name: Path(log) for name, log in raw["references"].items()}
    logs = [log.resolve() for log in references.values()]
    if len(set(logs)) != len(logs):
        raise ValueError(f"{path}: a reference log is named more than once")
    gates = {}
    for name, gate in raw["gates"].items():
        names = tuple(gate["references"])
        unknown = sorted(set(names) - set(references))
        if unknown:
            raise ValueError(f"{path}: gate {name} names unknown references {unknown}")
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: gate {name} names a reference more than once")
        if len(names) < 2:
            raise ValueError(f"{path}: gate {name} needs at least two references, got {len(names)}")
        bands = [key for key in ("lm_loss_delta", "lm_loss_tolerance") if key in gate]
        if len(bands) != 1:
            raise ValueError(f"{path}: gate {name} must set exactly one of lm_loss_delta and lm_loss_tolerance")
        first, last = gate["iterations"]
        delta = float(gate["lm_loss_delta"]) if "lm_loss_delta" in gate else None
        tolerance = float(gate["lm_loss_tolerance"]) if "lm_loss_tolerance" in gate else None
        gates[name] = Gate(name, names, int(first), int(last), int(gate["window"]), delta, tolerance)
    return GateSpec(bool(raw["wandb"]), references, gates)


def evaluate_gate(spec: GateSpec, name: str, candidate_log: Path) -> GateResult:
    """Run one gate of ``spec`` on the candidate's log and classify the outcome (see the module docstring)."""
    gate = spec.gates[name]
    reference_logs = [spec.references[ref] for ref in gate.references]
    if Path(candidate_log).resolve() in {log.resolve() for log in reference_logs}:
        return GateResult(name, NOT_EVALUATED, f"the candidate {candidate_log} is one of the references", None)
    iterations = (gate.first, gate.last)
    try:
        references = [load_trajectory(log, iterations, spec.wandb) for log in reference_logs]
        candidate = load_trajectory(Path(candidate_log), iterations, spec.wandb)
        runs = [run.source for run in [*references, candidate] if run.source.startswith("wandb:")]
        if len(set(runs)) != len(runs):
            raise ValueError(f"a W&B run is read more than once: {sorted(runs)}")
        report = band_test(references, [candidate], gate.window, loss_half_width=gate.lm_loss_tolerance)
    except ValueError as error:
        return GateResult(name, NOT_EVALUATED, str(error), None)
    (loss,) = [band for band in report.metrics if band.metric == VERDICT_METRIC]
    (verdict,) = report.verdicts
    if gate.lm_loss_delta is not None and abs(loss.delta - gate.lm_loss_delta) > DELTA_TOLERANCE:
        reason = (
            f"the references' {VERDICT_METRIC} band width is {loss.delta:.6f}, not the pre-registered "
            f"{gate.lm_loss_delta:.6f}: the reference set is not the one this gate was calibrated on"
        )
        return GateResult(name, NOT_EVALUATED, reason, report)
    return GateResult(name, verdict.verdict, "", report)


def exit_status(results: list[GateResult]) -> int:
    """0 when every gate passes, 2 when any is not evaluated, otherwise 1."""
    outcomes = {result.outcome for result in results}
    if NOT_EVALUATED in outcomes:
        return 2
    return 0 if outcomes == {PASS} else 1


def main(argv: list[str] | None = None) -> int:
    """Evaluate the requested gates (every gate by default), print each outcome and return the exit status."""
    parser = argparse.ArgumentParser(description="Evaluate a pre-registered loss gate on a candidate run.")
    parser.add_argument("--spec", type=Path, required=True, help="The gate spec YAML")
    parser.add_argument("--candidate", type=Path, required=True, help="The candidate run's training log")
    parser.add_argument("--gate", action="append", help="A gate to evaluate; repeatable (default: every gate)")
    parser.add_argument("--json", action="store_true", help="Emit the outcomes and band reports as JSON")
    args = parser.parse_args(argv)
    spec = load_gate_spec(args.spec)
    names = args.gate or list(spec.gates)
    unknown = sorted(set(names) - set(spec.gates))
    if unknown:
        parser.error(f"unknown gates {unknown}; the spec defines {sorted(spec.gates)}")
    results = [evaluate_gate(spec, name, args.candidate) for name in names]
    if args.json:
        payload = [
            {
                "gate": r.gate,
                "outcome": r.outcome,
                "reason": r.reason,
                "report": r.report.to_dict() if r.report else None,
            }
            for r in results
        ]
        print(json.dumps(payload, indent=2))
    else:
        for r in results:
            if r.report is not None:
                print(format_band_report(r.report))
            print(f"gate {r.gate}: {r.outcome}{' (' + r.reason + ')' if r.reason else ''}\n")
    return exit_status(results)


if __name__ == "__main__":
    sys.exit(main())
