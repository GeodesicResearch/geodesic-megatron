# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Run a test's payload in a child Python process (its own CUDA contexts and process group) and require success."""

import os
import socket
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    """A TCP port on 127.0.0.1 that nothing listens on now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _child_env(env: dict[str, str]) -> dict[str, str]:
    """This process's environment plus ``env``, with the repo first on PYTHONPATH."""
    child_env = dict(os.environ, **env)
    child_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    return child_env


def start_python(args: list[str], *, cwd: Path | None = None, **env: str) -> subprocess.Popen:
    """Start ``python <args>`` as ``run_python`` runs it, without waiting; its stdout and stderr are pipes (text)."""
    return subprocess.Popen(
        [sys.executable, *args],
        env=_child_env(env),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def run_python(args: list[str], *, timeout: float, cwd: Path | None = None, **env: str) -> subprocess.CompletedProcess:
    """``python <args>`` with the repo first on PYTHONPATH and ``env`` added to this process's environment.

    Raises:
        AssertionError: the child exits non-zero; the message holds the tails of its stdout and stderr.
    """
    child = subprocess.run(
        [sys.executable, *args], env=_child_env(env), cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    assert child.returncode == 0, f"stdout:\n{child.stdout[-3000:]}\nstderr:\n{child.stderr[-6000:]}"
    return child
