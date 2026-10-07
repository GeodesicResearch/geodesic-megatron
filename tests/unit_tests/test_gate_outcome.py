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

"""Unit tests for scripts/telemetry/gate_outcome.py (the exit status a set of gate outcomes makes)."""

import pytest
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS, exit_status


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
