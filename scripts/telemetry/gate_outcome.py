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

"""The outcomes of a pre-registered gate, the exit status a set of them makes, and the verdict ordered stages of
them make.

PASS and FAIL are a verdict on the candidate run. NOT EVALUATED means the gate could not be run, or its
result could not be trusted, and the gate states why. One failing gate decides whatever the others'
outcomes; otherwise one gate not evaluated keeps the set from passing, so a failure to read is never
mistaken for a failing candidate, nor for a passing one. A spec that cannot be used ends the same way
(``unusable_spec_status``).

A pre-registered rule can also order its gates in stages, each stating what its failure means: FAIL, or
INCONCLUSIVE when a failing gate means the experiment could not show anything (its integrity checks, or a positive
control that did not respond), so that a later stage's failure is not read as a verdict on the candidate. The first
stage whose gates do not all pass decides (``ordered_verdict``), and the verdict's exit status matches the gate
set's: 0 PASS, 1 FAIL, 2 INCONCLUSIVE.

A gate that compares a value it reads with the value its spec states, or several values read from different records
with each other, uses one equality rule (``identical``: the same type and value, so 3126, 3126.0 and "3126" differ),
and ``fields_outcome`` checks a record's values at dotted paths by it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from scripts.dotted_paths import value_at


PASS, FAIL, NOT_EVALUATED = "PASS", "FAIL", "NOT EVALUATED"
OUTCOMES = (PASS, FAIL, NOT_EVALUATED)
INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_EXIT_STATUS = {PASS: 0, FAIL: 1, INCONCLUSIVE: 2}
STAGE_FAILURES = (FAIL, INCONCLUSIVE)
# What loading a gate spec raises when the spec cannot be used: a file that is absent or unreadable, YAML that does not
# parse, a document that is not the mapping of gates the loaders walk, and a spec a loader refuses (ValueError).
SPEC_ERRORS = (OSError, ValueError, TypeError, AttributeError, KeyError, yaml.YAMLError)


def unusable_spec_status(path: Path, error: Exception) -> int:
    """Report on stderr that the spec at ``path`` cannot be used because of ``error``, and return the exit status.

    That is NOT EVALUATED's, never FAIL's: a gate that did not run has judged no candidate, and a staged run records
    a gate's exit status as its result.
    """
    print(f"cannot use the spec {path}: {type(error).__name__}: {error}", file=sys.stderr)
    return exit_status([NOT_EVALUATED])


def exit_status(outcomes: Iterable[str]) -> int:
    """1 when any outcome is FAIL, otherwise 2 when any is NOT EVALUATED, and 0 when every one is PASS.

    Raises ValueError on a value that is not one of ``OUTCOMES``, and on no outcomes at all: a set that
    checked nothing has not passed.
    """
    seen = set(outcomes)
    if not seen:
        raise ValueError("no gate outcomes: nothing was evaluated")
    unknown = sorted(seen - set(OUTCOMES))
    if unknown:
        raise ValueError(f"not gate outcomes: {unknown}")
    if FAIL in seen:
        return 1
    return 2 if NOT_EVALUATED in seen else 0


@dataclass(frozen=True)
class Stage:
    """One stage of an ordered verdict: its name, the verdict its failure gives, and its gates' outcomes."""

    name: str
    on_fail: str
    outcomes: tuple[str, ...]


def ordered_verdict(stages: Sequence[Stage]) -> tuple[str, str | None]:
    """The verdict of ``stages`` taken in order, and the name of the stage that decided it (None for PASS).

    The first stage whose outcomes are not all PASS decides: a FAIL among them gives that stage's ``on_fail``
    verdict; outcomes that are only PASS and NOT EVALUATED give INCONCLUSIVE, since a gate that could not be
    evaluated shows nothing either way. Every stage passing gives PASS.

    Raises ValueError on no stages, a stage with no outcomes, an ``on_fail`` that is not FAIL or INCONCLUSIVE, or an
    outcome outside ``OUTCOMES``.
    """
    if not stages:
        raise ValueError("no verdict stages: nothing was evaluated")
    for stage in stages:
        if stage.on_fail not in STAGE_FAILURES:
            raise ValueError(f"stage {stage.name}: on_fail must be one of {STAGE_FAILURES}, not {stage.on_fail!r}")
        exit_status(stage.outcomes)
    for stage in stages:
        if FAIL in stage.outcomes:
            return stage.on_fail, stage.name
        if NOT_EVALUATED in stage.outcomes:
            return INCONCLUSIVE, stage.name
    return PASS, None


def identical(first: Any, second: Any) -> bool:
    """Whether two parsed values are the same value of the same type, mappings and lists compared element by element
    (so ``True`` is not ``1``, nor ``3126`` ``3126.0`` or ``"3126"``)."""
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(identical(first[key], second[key]) for key in first)
    if isinstance(first, list):
        return len(first) == len(second) and all(identical(a, b) for a, b in zip(first, second))
    return first == second


def fields_outcome(
    record: Any, equal: Mapping[str, Any], not_equal: Mapping[str, Any], absent: str
) -> tuple[str, list[str]]:
    """The outcome of a record's values at dotted paths (``scripts/dotted_paths.value_at``), and one line per problem.

    Each path of ``equal`` must hold its value, and each of ``not_equal`` must hold a value other than its own, by
    ``identical``. A wrong value is FAIL; otherwise a path the record does not hold gives ``absent`` (FAIL or NOT
    EVALUATED, the gate's rule for a missing value); otherwise PASS. Raises ValueError on another ``absent``.
    """
    if absent not in (FAIL, NOT_EVALUATED):
        raise ValueError(f"absent must be {FAIL} or {NOT_EVALUATED}, not {absent!r}")
    problems, wrong, missing = [], False, False
    checks = [(path, value, True) for path, value in equal.items()]
    checks += [(path, value, False) for path, value in not_equal.items()]
    for path, value, wanted in checks:
        try:
            found = value_at(record, path)
        except KeyError:
            problems.append(f"{path} is absent")
            missing = True
            continue
        if identical(found, value) != wanted:
            problems.append(f"{path} is {found!r}, not {value!r}" if wanted else f"{path} is still {value!r}")
            wrong = True
    return (FAIL if wrong else absent if missing else PASS), problems
