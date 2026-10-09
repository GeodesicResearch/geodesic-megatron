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
"""tests/unit_tests/stubbed_shell.py: a stubbed shell cannot re-enter itself or outlive its test.

Both tests run real bash processes. The startup hook in the first stands in for the pipeline container's
/etc/shinit_v2 and stops calling nvidia-smi after HOOK_DEPTH_LIMIT levels, so even an environment that
carried it would recurse a bounded number of times, inside a 10 s timeout whose expiry kills the whole
process group.
"""

import os
import stat
import subprocess
import time

import pytest

from tests.unit_tests.stubbed_shell import run_in_own_session, stubbed_shell_env


HOOK_DEPTH_LIMIT = 20


def _executable(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _gone(pid):
    """True once no process has this pid, waiting briefly for an orphan to be reaped."""
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_a_shell_stub_of_nvidia_smi_runs_once_whatever_bash_startup_hook_the_caller_has(tmp_path, monkeypatch):
    """The container's BASH_ENV hook calls nvidia-smi through PATH before a bash runs its first line; carried
    into the stubbed environment, each shell stub of nvidia-smi would start the next."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub_runs = tmp_path / "stub_runs"
    hook_runs = tmp_path / "hook_runs"
    _executable(bindir / "nvidia-smi", f'#!/bin/bash\necho "$$" >> "{stub_runs}"\n')
    hook = tmp_path / "bash_env"
    hook.write_text(
        f'echo sourced >> "{hook_runs}"\n'
        f'[ "$(wc -l < "{hook_runs}")" -ge {HOOK_DEPTH_LIMIT} ] || nvidia-smi -q -d COMPUTE\n'
    )
    monkeypatch.setenv("BASH_ENV", str(hook))
    monkeypatch.setenv("ENV", str(hook))

    result = run_in_own_session(["bash", "-c", "nvidia-smi -q -d COMPUTE"], stubbed_shell_env(bindir, {}), timeout=10)

    assert result.returncode == 0, result.stderr
    assert not hook_runs.exists()
    pids = [int(pid) for pid in stub_runs.read_text().split()]
    assert len(pids) == 1
    assert all(_gone(pid) for pid in pids)


def test_a_command_that_outlives_its_timeout_takes_everything_it_started_with_it(tmp_path):
    started = tmp_path / "started"
    script = f'sleep 60 & echo "$!" > "{started}"; wait'

    with pytest.raises(subprocess.TimeoutExpired):
        run_in_own_session(["bash", "-c", script], stubbed_shell_env(tmp_path, {}), timeout=2)

    assert _gone(int(started.read_text()))
