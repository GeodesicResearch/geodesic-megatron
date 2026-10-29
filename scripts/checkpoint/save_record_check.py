#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Check what one save of a run records against the runs it is compared with.

A save records the run's resolved settings (``run_config.yaml``) and its counters (``train_state.pt``). The check
passes when both hold:
- **Settings:** every setting the save records differently from a reference run's record is allowed for that
  reference by the spec, with its reason. A key recorded on one side only is allowed only by an entry that says so
  (``only_in: candidate`` or ``only_in: reference``): a key the code added or removed is another fact than a value
  that changed.
- **Counters:** the save's ``step`` is its iteration, its ``consumed_train_samples`` is that iteration times the global
  batch it records, and it skipped no sample. A run resumed across segments that consumed a sample twice, or skipped
  one, fails at the next save.

The spec (YAML) names each reference by its ``run_config.yaml`` (with the file's ``sha256`` when it is a copy rather
than the reference run's own save, so a changed copy is not read), and lists the allowed differences: ``keys``
(dotted names; a name ending in ``.*`` covers that subtree), ``reason``, ``references`` (the references it applies
to) and optionally ``only_in``.

The outcomes and the exit status are the repository's gates' (``scripts/telemetry/gate_outcome.py``): 0 PASS, 1 FAIL,
2 NOT EVALUATED when a record cannot be read.

    python scripts/checkpoint/save_record_check.py --spec <spec.yaml> --save <checkpoint dir>/iter_<N> \\
        [--report-out <json>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.dotted_paths import dotted_leaves  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_absolute_path, require_string  # noqa: E402
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS, exit_status  # noqa: E402


RUN_CONFIG = "run_config.yaml"
TRAIN_STATE = "train_state.pt"
SIDES = ("candidate", "reference")
ITERATION_DIRECTORY = re.compile(r"iter_(\d+)")


class _Absent:
    """A key one record does not hold."""

    def __repr__(self) -> str:
        return "<absent>"


ABSENT = _Absent()


@dataclass(frozen=True)
class Reference:
    """A run the save is compared with: its ``run_config.yaml``, and the file's sha256 when it is a copy."""

    run_config: Path
    sha256: str | None


@dataclass(frozen=True)
class Allowance:
    """Keys a save may record differently from the named references', and why."""

    keys: tuple[str, ...]
    reason: str
    references: frozenset[str]
    only_in: str | None

    def covers(self, key: str) -> bool:
        return any(
            key == pattern or (pattern.endswith(".*") and key.startswith(pattern[:-1])) for pattern in self.keys
        )

    def allows(self, reference: str, key: str, candidate: Any, recorded: Any) -> bool:
        if reference not in self.references or not self.covers(key):
            return False
        if self.only_in == "candidate":
            return recorded is ABSENT
        if self.only_in == "reference":
            return candidate is ABSENT
        return candidate is not ABSENT and recorded is not ABSENT


def read_spec(path: Path) -> tuple[dict[str, Reference], list[Allowance]]:
    """The spec's references and allowances. Raises ValueError on a spec that is not one."""
    spec = require_keys(yaml.safe_load(path.read_text()), str(path), {"references", "allowed"})
    if not isinstance(spec["references"], dict) or not spec["references"]:
        raise ValueError(f"{path}: references must name at least one reference")
    references = {}
    for name, entry in spec["references"].items():
        where = f"{path}: references.{name}"
        entry = require_keys(entry, where, {"run_config"}, {"sha256"})
        references[name] = Reference(
            run_config=Path(require_absolute_path(entry["run_config"], f"{where}.run_config")),
            sha256=require_string(entry["sha256"], f"{where}.sha256") if "sha256" in entry else None,
        )
    if not isinstance(spec["allowed"], list):
        raise ValueError(f"{path}: allowed must be a list")
    allowances = []
    for index, entry in enumerate(spec["allowed"]):
        where = f"{path}: allowed[{index}]"
        entry = require_keys(entry, where, {"keys", "reason", "references"}, {"only_in"})
        keys = entry["keys"]
        if not isinstance(keys, list) or not keys or not all(isinstance(key, str) and key for key in keys):
            raise ValueError(f"{where}.keys must be a non-empty list of dotted names")
        named = entry["references"]
        if not isinstance(named, list) or not named or set(named) - set(references):
            raise ValueError(f"{where}.references must name references of the spec, got {named!r}")
        only_in = entry.get("only_in")
        if only_in is not None and only_in not in SIDES:
            raise ValueError(f"{where}.only_in must be one of {SIDES}, got {only_in!r}")
        allowances.append(
            Allowance(
                keys=tuple(keys),
                reason=require_string(entry["reason"], f"{where}.reason"),
                references=frozenset(named),
                only_in=only_in,
            )
        )
    return references, allowances


def read_settings(path: Path) -> dict[str, Any]:
    """A run_config.yaml's settings as dotted keys."""
    return dotted_leaves(yaml.safe_load(path.read_text()))


def differences(candidate: Mapping[str, Any], recorded: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Every key whose value differs between the two records, or that one of them does not hold."""
    return {
        key: (candidate.get(key, ABSENT), recorded.get(key, ABSENT))
        for key in sorted(set(candidate) | set(recorded))
        if candidate.get(key, ABSENT) != recorded.get(key, ABSENT)
    }


def judge_reference(
    name: str, candidate: Mapping[str, Any], recorded: Mapping[str, Any], allowances: list[Allowance]
) -> tuple[str, list[dict[str, Any]]]:
    """PASS when every difference from reference ``name`` is allowed, else FAIL; one row per difference."""
    rows = []
    for key, (value, recorded_value) in differences(candidate, recorded).items():
        allowance = next((a for a in allowances if a.allows(name, key, value, recorded_value)), None)
        rows.append(
            {
                "key": key,
                "candidate": repr(value)[:200],
                "reference": repr(recorded_value)[:200],
                "reason": allowance.reason if allowance else None,
            }
        )
    return (PASS if all(row["reason"] for row in rows) else FAIL), rows


def judge_counters(save: Path, candidate: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """PASS when the save's counters are its iteration's: step, consumed samples at the recorded global batch, nothing
    skipped."""
    from megatron.bridge.training.utils.checkpoint_utils import read_train_state

    match = ITERATION_DIRECTORY.fullmatch(save.name)
    if match is None:
        return NOT_EVALUATED, {"why": f"{save.name} is not an iteration directory (iter_<N>)"}
    iteration = int(match.group(1))
    batch = candidate.get("train.global_batch_size")
    if not isinstance(batch, int) or isinstance(batch, bool):
        return NOT_EVALUATED, {"why": f"{RUN_CONFIG} records no integer train.global_batch_size ({batch!r})"}
    state = read_train_state(str(save / TRAIN_STATE))
    detail = {
        "iteration": iteration,
        "step": state.step,
        "global_batch_size": batch,
        "consumed_train_samples": state.consumed_train_samples,
        "expected_consumed_train_samples": iteration * batch,
        "skipped_train_samples": state.skipped_train_samples,
    }
    ok = (
        state.step == iteration
        and state.consumed_train_samples == iteration * batch
        and state.skipped_train_samples == 0
    )
    return (PASS if ok else FAIL), detail


def sha256(path: Path) -> str:
    """The file's sha256, as hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(spec: Path, save: Path) -> dict[str, Any]:
    """The report of one save: the counters' outcome and each reference's, with every difference and its reason."""
    references, allowances = read_spec(spec)
    report: dict[str, Any] = {"spec": str(spec), "spec_sha256": sha256(spec), "save": str(save)}
    if not (save / RUN_CONFIG).is_file() or not (save / TRAIN_STATE).is_file():
        report["outcomes"] = {"save": NOT_EVALUATED}
        report["why"] = f"{save} holds no {RUN_CONFIG} or no {TRAIN_STATE}"
        return report
    candidate = read_settings(save / RUN_CONFIG)
    outcome, counters = judge_counters(save, candidate)
    report["outcomes"] = {"counters": outcome}
    report["counters"] = counters
    report["references"] = {}
    for name, reference in references.items():
        entry: dict[str, Any] = {"run_config": str(reference.run_config)}
        if not reference.run_config.is_file():
            outcome, entry["why"] = NOT_EVALUATED, "the reference's run_config does not exist"
        elif reference.sha256 is not None and sha256(reference.run_config) != reference.sha256:
            outcome, entry["why"] = NOT_EVALUATED, f"the reference copy's sha256 is not the spec's {reference.sha256}"
        else:
            outcome, entry["differences"] = judge_reference(
                name, candidate, read_settings(reference.run_config), allowances
            )
        report["outcomes"][name] = outcome
        report["references"][name] = entry
    return report


def describe(report: dict[str, Any]) -> list[str]:
    """The report as the lines the check prints."""
    lines = [f"save {report['save']}"]
    if "why" in report:
        return lines + [f"  {NOT_EVALUATED}: {report['why']}"]
    counters = report["counters"]
    lines.append(f"  counters {report['outcomes']['counters']}: {json.dumps(counters)}")
    for name, entry in report["references"].items():
        lines.append(f"  {name} ({entry['run_config']}) {report['outcomes'][name]}")
        if "why" in entry:
            lines.append(f"    {entry['why']}")
            continue
        for row in entry["differences"]:
            reason = row["reason"] or "NOT ALLOWED"
            lines.append(f"    {row['key']}: {row['candidate']} vs {row['reference']}  [{reason}]")
    lines.append(f"verdict: exit {exit_status(report['outcomes'].values())}")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Check one save, print the report (and write it as JSON to ``--report-out``, never over an existing file), and
    return the gates' exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, required=True, help="the allowed differences, per reference")
    parser.add_argument("--save", type=Path, required=True, help="the save's iteration directory, iter_<N>")
    parser.add_argument("--report-out", type=Path, help="write the report as JSON here; an existing file is refused")
    args = parser.parse_args(argv)
    if args.report_out is not None and args.report_out.exists():
        parser.error(f"{args.report_out} exists")
    report = check(args.spec, args.save)
    print("\n".join(describe(report)))
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with args.report_out.open("x") as out:
            json.dump(report, out, indent=2)
    return exit_status(report["outcomes"].values())


if __name__ == "__main__":
    sys.exit(main())
