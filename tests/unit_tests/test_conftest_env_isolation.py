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
"""The unit-test environment must not inherit the submitting SLURM allocation.

`get_world_size_safe` falls back to `SLURM_NTASKS`, so without the conftest
scrub a suite run inside an N-node allocation silently sees world_size=N and
world-size-dependent assertions fail (test_report_throughput measured 32x-off
per-device throughput inside a 32-node tunnel). These tests pin the scrub.
"""

import os
import subprocess
import sys
import tempfile


def test_slurm_allocation_vars_are_scrubbed():
    """In-suite probe: the conftest guarantees SLURM_NTASKS is absent at test time."""
    assert "SLURM_NTASKS" not in os.environ


def test_scrub_holds_when_var_set_at_session_start():
    """Fail-before-fix reproducer, environment-independent.

    Runs the probe above in a child pytest session with SLURM_NTASKS exported —
    the situation of a suite launched from inside a multi-node allocation. With
    the conftest scrub reverted the probe fails there, on or off a SLURM node.
    The child is a real pytest session so the real conftest hooks run; the
    subprocess boundary exists only to control the session-start environment.
    """
    env = dict(os.environ)
    env["SLURM_NTASKS"] = "32"
    probe = f"{os.path.abspath(__file__)}::test_slurm_allocation_vars_are_scrubbed"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", probe, "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tempfile.mkdtemp(),
        timeout=120,
    )
    assert result.returncode == 0, f"probe failed under SLURM_NTASKS=32:\n{result.stdout}\n{result.stderr}"
