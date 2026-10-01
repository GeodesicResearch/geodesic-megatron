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

"""Evaluate pre-registered memory and speed gates on runs' scores (``score_run.py --json`` files).

A spec, fixed before the runs exist, names gates of two kinds, each over score files in one directory:

- ``memory``: one run's peak memory over every rank, the score's ``peak_memory_across_ranks`` (read from the
  ``[peak-memory]`` line rank 0 logs when the training loop ends). It fails when the score has no such
  summary (the loop was cut short, so the peak is unknown), when the largest allocator retry count exceeds
  ``max_alloc_retries``, or when the largest peak allocated memory exceeds ``max_allocated_gb`` (decimal
  GB, as the summary counts). Peak reserved memory is reported, not gated.
- ``speed``: a candidate's mean step time divided by a reference's, the two measured on the same nodes,
  times ``reference_s_per_iter``, the reference posture's own step time where the gate's decision applies:
  the candidate's step time projected to that placement. It passes up to ``report_up_to_s``, stating above
  ``go_up_to_s`` that the result must be reported, and fails above ``report_up_to_s``.

Each gate's outcome is PASS, FAIL or NOT EVALUATED (``gate_outcome``). NOT EVALUATED means a score could not
be read or lacks a field, the memory summary covers a different number of ranks than the run's GPUs, or the
two speed scores were taken over different windows or GPU counts. Every outcome carries a line stating the
measurement or the reason. The exit status is ``gate_outcome.exit_status``'s.

USAGE
    python scripts/telemetry/score_gate.py --spec score_gate.yaml --scores-dir DIR [--gate NAME ...] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


# Run as a script, only scripts/telemetry/ is on sys.path; the repo root makes the sibling modules
# importable the same way from every entry point.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS, exit_status  # noqa: E402


MEMORY, SPEED = "memory", "speed"


@dataclass(frozen=True)
class MemoryGate:
    """One run's peak memory over every rank against a limit on allocated memory and on allocator retries."""

    name: str
    score: str
    max_allocated_gb: float
    max_alloc_retries: int


@dataclass(frozen=True)
class SpeedGate:
    """A candidate's step time relative to a reference on the same nodes, projected by the reference's own."""

    name: str
    candidate: str
    reference: str
    reference_s_per_iter: float
    go_up_to_s: float
    report_up_to_s: float


@dataclass(frozen=True)
class ScoreGateResult:
    """One gate's kind, its outcome and the line stating its measurement or the reason it was not evaluated."""

    gate: str
    kind: str
    outcome: str
    detail: str


def load_score_gates(path: Path) -> dict[str, MemoryGate | SpeedGate]:
    """Read a score-gate spec into its gates by name, refusing one whose gates could not be evaluated as written.

    Raises ValueError on an unknown kind, a spec with no gates, a gate name used twice, a speed gate whose
    candidate is its reference, or one whose go limit exceeds its report limit; KeyError on a missing field.
    """
    raw = yaml.safe_load(Path(path).read_text())
    unknown = sorted(set(raw) - {MEMORY, SPEED})
    if unknown:
        raise ValueError(f"{path}: unknown gate kinds {unknown}")
    gates: dict[str, MemoryGate | SpeedGate] = {}
    for name, gate in (raw.get(MEMORY) or {}).items():
        gates[name] = MemoryGate(name, gate["score"], float(gate["max_allocated_gb"]), int(gate["max_alloc_retries"]))
    for name, gate in (raw.get(SPEED) or {}).items():
        if name in gates:
            raise ValueError(f"{path}: gate {name} is defined twice")
        if gate["candidate"] == gate["reference"]:
            raise ValueError(f"{path}: speed gate {name} compares {gate['candidate']} with itself")
        speed = SpeedGate(
            name,
            gate["candidate"],
            gate["reference"],
            float(gate["reference_s_per_iter"]),
            float(gate["go_up_to_s"]),
            float(gate["report_up_to_s"]),
        )
        if speed.go_up_to_s > speed.report_up_to_s:
            raise ValueError(f"{path}: speed gate {name} has go_up_to_s above report_up_to_s")
        gates[name] = speed
    if not gates:
        raise ValueError(f"{path}: defines no gates")
    return gates


def _read_score(scores_dir: Path, name: str) -> dict[str, Any]:
    return json.loads((scores_dir / name).read_text())


def evaluate_memory(gate: MemoryGate, scores_dir: Path) -> ScoreGateResult:
    """The memory gate's outcome on its run's score (see the module docstring)."""
    try:
        score = _read_score(scores_dir, gate.score)
        peak = score["peak_memory_across_ranks"]
        if peak is None:
            reason = f"{gate.score} has no peak memory over all ranks: the training loop was cut short"
            return ScoreGateResult(gate.name, MEMORY, FAIL, reason)
        ranks, gpus = peak["ranks"], score["num_gpus"]
        allocated, rank = peak["max_allocated_gb"], peak["max_allocated_rank"]
        retries, reserved = peak["max_alloc_retries"], peak["max_reserved_gb"]
    except Exception as error:  # noqa: BLE001 - every way the score cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, MEMORY, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    if ranks != gpus:
        reason = f"the summary covers {ranks} ranks, not the run's {gpus} GPUs"
        return ScoreGateResult(gate.name, MEMORY, NOT_EVALUATED, reason)
    measured = (
        f"max allocated {allocated:.3f} GB (rank {rank}), limit {gate.max_allocated_gb}; "
        f"max allocator retries {retries}, limit {gate.max_alloc_retries}; max reserved {reserved:.3f} GB, reported"
    )
    within = allocated <= gate.max_allocated_gb and retries <= gate.max_alloc_retries
    return ScoreGateResult(gate.name, MEMORY, PASS if within else FAIL, measured)


def evaluate_speed(gate: SpeedGate, scores_dir: Path) -> ScoreGateResult:
    """The speed gate's outcome on its candidate's and reference's scores (see the module docstring)."""
    try:
        candidate = _read_score(scores_dir, gate.candidate)
        reference = _read_score(scores_dir, gate.reference)
        placement = [
            (score["window_first"], score["window_last"], score["num_gpus"]) for score in (candidate, reference)
        ]
        candidate_s, reference_s = candidate["mean_step_s"], reference["mean_step_s"]
    except Exception as error:  # noqa: BLE001 - every way the scores cannot be read is NOT EVALUATED, never a FAIL
        return ScoreGateResult(gate.name, SPEED, NOT_EVALUATED, f"{type(error).__name__}: {error}")
    if placement[0] != placement[1]:
        reason = (
            f"the scores' (first, last, GPUs) differ: {gate.candidate} {placement[0]}, {gate.reference} {placement[1]}"
        )
        return ScoreGateResult(gate.name, SPEED, NOT_EVALUATED, reason)
    projected = candidate_s / reference_s * gate.reference_s_per_iter
    measured = (
        f"projected {projected:.3f} s/iter = {candidate_s:.3f} s / {reference_s:.3f} s x {gate.reference_s_per_iter} s"
    )
    if projected <= gate.go_up_to_s:
        return ScoreGateResult(gate.name, SPEED, PASS, f"{measured}: go")
    if projected <= gate.report_up_to_s:
        return ScoreGateResult(gate.name, SPEED, PASS, f"{measured}: go, and report it (above {gate.go_up_to_s} s)")
    return ScoreGateResult(gate.name, SPEED, FAIL, f"{measured}: above {gate.report_up_to_s} s")


def evaluate_score_gate(gate: MemoryGate | SpeedGate, scores_dir: Path) -> ScoreGateResult:
    """Evaluate one gate of either kind on the scores in ``scores_dir``."""
    if isinstance(gate, MemoryGate):
        return evaluate_memory(gate, scores_dir)
    return evaluate_speed(gate, scores_dir)


def main(argv: list[str] | None = None) -> int:
    """Evaluate the requested gates (every gate by default), print each outcome and return the exit status."""
    parser = argparse.ArgumentParser(description="Evaluate pre-registered memory and speed gates on runs' scores.")
    parser.add_argument("--spec", type=Path, required=True, help="The score-gate spec YAML")
    parser.add_argument("--scores-dir", type=Path, required=True, help="The directory holding the score files")
    parser.add_argument("--gate", action="append", help="A gate to evaluate; repeatable (default: every gate)")
    parser.add_argument("--json", action="store_true", help="Emit the outcomes as JSON")
    args = parser.parse_args(argv)
    gates = load_score_gates(args.spec)
    names = args.gate or list(gates)
    unknown = sorted(set(names) - set(gates))
    if unknown:
        parser.error(f"unknown gates {unknown}; the spec defines {sorted(gates)}")
    results = [evaluate_score_gate(gates[name], args.scores_dir) for name in names]
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        for result in results:
            print(f"gate {result.gate} ({result.kind}): {result.outcome} ({result.detail})")
    return exit_status(result.outcome for result in results)


if __name__ == "__main__":
    sys.exit(main())
