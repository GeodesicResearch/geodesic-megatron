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

"""Unit tests for scripts/telemetry/gate_outcome.py (the exit status a set of gate outcomes makes, the verdict of
ordered stages of them, and the one rule by which gates compare values)."""

import pytest
from scripts.telemetry.gate_outcome import (
    FAIL,
    INCONCLUSIVE,
    NOT_EVALUATED,
    PASS,
    VERDICT_EXIT_STATUS,
    Stage,
    exit_status,
    fields_outcome,
    identical,
    ordered_verdict,
)


@pytest.mark.parametrize(
    "first, second, same",
    [
        (3126, 3126, True),
        (3126, 3126.0, False),
        (3126, "3126", False),
        (True, 1, False),
        (None, None, True),
        ({"a": [1, "x"], "b": None}, {"b": None, "a": [1, "x"]}, True),
        ({"a": [1, "x"]}, {"a": [1.0, "x"]}, False),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        ([1, 2], [1, 2, 3], False),
    ],
)
def test_values_are_identical_only_in_type_and_value(first, second, same):
    assert identical(first, second) is same
    assert identical(second, first) is same


def test_fields_outcome_fails_a_wrong_value_and_gives_the_callers_outcome_for_an_absent_path():
    record = {"verdict": "PASS", "calibration": {"status": "PENDING"}, "corpora": [{"docs": 41}]}
    assert fields_outcome(record, {"verdict": "PASS", "corpora.0.docs": 41}, {"calibration.status": "x"}, FAIL) == (
        PASS,
        [],
    )
    outcome, problems = fields_outcome(record, {"verdict": "FAIL"}, {"calibration.status": "PENDING"}, FAIL)
    assert outcome == FAIL and problems == ["verdict is 'PASS', not 'FAIL'", "calibration.status is still 'PENDING'"]
    for absent in (FAIL, NOT_EVALUATED):
        assert fields_outcome(record, {"missing": 1}, {}, absent) == (absent, ["missing is absent"])
    assert fields_outcome(record, {"missing": 1, "verdict": "FAIL"}, {}, NOT_EVALUATED)[0] == FAIL, "wrong wins"
    with pytest.raises(ValueError, match="absent must be"):
        fields_outcome(record, {"verdict": "PASS"}, {}, PASS)


@pytest.mark.parametrize(
    "outcomes, status",
    [
        ([PASS, PASS], 0),
        ([PASS, FAIL], 1),
        ([FAIL, NOT_EVALUATED], 1),
        ([NOT_EVALUATED, FAIL], 1),
        ([PASS, NOT_EVALUATED], 2),
    ],
)
def test_a_failing_gate_decides_and_an_unevaluated_one_keeps_the_set_from_passing(outcomes, status):
    assert exit_status(outcomes) == status


def test_no_outcomes_at_all_is_refused_rather_than_a_pass():
    """A set of gates that evaluated nothing has not passed, so it cannot exit 0."""
    with pytest.raises(ValueError, match="nothing was evaluated"):
        exit_status([])


def test_an_outcome_outside_the_vocabulary_is_refused():
    """A band test's FLAG is a report on a flagged metric, not a gate outcome; counting it as a pass would
    hide whatever produced it."""
    with pytest.raises(ValueError, match="FLAG"):
        exit_status([PASS, "FLAG"])


def stages(integrity: list[str], control: list[str], masking: list[str]) -> list[Stage]:
    return [
        Stage("integrity", INCONCLUSIVE, tuple(integrity)),
        Stage("positive_control", INCONCLUSIVE, tuple(control)),
        Stage("masking", FAIL, tuple(masking)),
    ]


@pytest.mark.parametrize(
    "integrity, control, masking, verdict, stage",
    [
        ([PASS], [PASS], [PASS, PASS], PASS, None),
        ([PASS], [PASS], [PASS, FAIL], FAIL, "masking"),
        # An earlier stage decides, whatever a later one shows.
        ([FAIL], [PASS], [FAIL], INCONCLUSIVE, "integrity"),
        ([PASS], [FAIL], [FAIL], INCONCLUSIVE, "positive_control"),
        # A gate that could not be evaluated shows nothing, even in a stage whose failure would be a FAIL...
        ([PASS], [PASS], [NOT_EVALUATED, PASS], INCONCLUSIVE, "masking"),
        # ...but a failing gate beside it still decides that stage.
        ([PASS], [PASS], [NOT_EVALUATED, FAIL], FAIL, "masking"),
    ],
)
def test_the_first_stage_that_does_not_pass_decides_the_verdict(integrity, control, masking, verdict, stage):
    assert ordered_verdict(stages(integrity, control, masking)) == (verdict, stage)


def test_the_verdicts_exit_statuses_match_the_gate_sets():
    assert VERDICT_EXIT_STATUS == {PASS: exit_status([PASS]), FAIL: exit_status([FAIL]), INCONCLUSIVE: 2}
    assert exit_status([NOT_EVALUATED]) == VERDICT_EXIT_STATUS[INCONCLUSIVE]


@pytest.mark.parametrize(
    "bad, message",
    [
        ([], "no verdict stages"),
        ([Stage("s", INCONCLUSIVE, ())], "nothing was evaluated"),
        ([Stage("s", NOT_EVALUATED, (PASS,))], "on_fail"),
        ([Stage("s", FAIL, ("FLAG",))], "FLAG"),
    ],
)
def test_stages_that_cannot_make_a_verdict_are_refused(bad, message):
    with pytest.raises(ValueError, match=message):
        ordered_verdict(bad)
