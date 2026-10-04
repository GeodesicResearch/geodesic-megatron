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

"""The outcomes of a pre-registered gate, and the exit status a set of them makes.

PASS and FAIL are a verdict on the candidate run. NOT EVALUATED means the gate could not be run, or its
result could not be trusted, and the gate states why. One failing gate decides whatever the others'
outcomes; otherwise one gate not evaluated keeps the set from passing, so a failure to read is never
mistaken for a failing candidate, nor for a passing one.
"""

from __future__ import annotations

from collections.abc import Iterable


PASS, FAIL, NOT_EVALUATED = "PASS", "FAIL", "NOT EVALUATED"
OUTCOMES = (PASS, FAIL, NOT_EVALUATED)


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
