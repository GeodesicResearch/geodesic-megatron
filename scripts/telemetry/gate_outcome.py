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
mistaken for a failing candidate, nor for a passing one.

A pre-registered rule can also order its gates in stages, each stating what its failure means: FAIL, or
INCONCLUSIVE when a failing gate means the experiment could not show anything (its integrity checks, or a positive
control that did not respond), so that a later stage's failure is not read as a verdict on the candidate. The first
stage whose gates do not all pass decides (``ordered_verdict``), and the verdict's exit status matches the gate
set's: 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


PASS, FAIL, NOT_EVALUATED = "PASS", "FAIL", "NOT EVALUATED"
OUTCOMES = (PASS, FAIL, NOT_EVALUATED)
INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_EXIT_STATUS = {PASS: 0, FAIL: 1, INCONCLUSIVE: 2}
STAGE_FAILURES = (FAIL, INCONCLUSIVE)


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
