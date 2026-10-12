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
"""Placing another process's change to the filesystem between two reads of the code under test."""

from collections.abc import Callable
from pathlib import Path

import pytest


def strike_at(
    monkeypatch: pytest.MonkeyPatch, method: str, path: Path, strike: Callable[[], None], *, after: bool
) -> None:
    """Run ``strike`` once, at pathlib's first ``method`` call on ``path``: before the call is made,
    or once it has answered when ``after``.

    A file another process deletes while the code under test reads a directory goes between two of
    that code's system calls, a moment no test can pick from outside. The pathlib method still runs
    for real, on the real filesystem; only the moment of the other process's change is chosen.
    """
    real = getattr(Path, method)
    pending = [strike]

    def racing(self: Path, *args, **kwargs):
        if self != path or not pending:
            return real(self, *args, **kwargs)
        change = pending.pop()
        if not after:
            change()
        answer = real(self, *args, **kwargs)
        if after:
            change()
        return answer

    monkeypatch.setattr(Path, method, racing)
