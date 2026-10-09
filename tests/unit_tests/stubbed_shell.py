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
"""Running one of the repo's shell scripts under a unit test, with stub commands first on PATH."""

import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


def stubbed_shell_env(bindir: str | Path, extra: Mapping[str, str]) -> dict[str, str]:
    """PATH with the stub directory first, plus the variables the test sets, and nothing else.

    Deliberately not a copy of os.environ. Inside the pipeline container BASH_ENV and ENV make every
    non-interactive bash source /etc/bash.bashrc and /etc/shinit_v2 before its first line, which costs
    ~2.5 s per shell and runs `nvidia-smi -q -d COMPUTE` resolved through PATH. Under a copied
    environment a stub named nvidia-smi that is itself a shell would start another copy of itself before
    running a line, without bound. That startup hook is the container's, not the script's, and is not
    under test.
    """
    return {"PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", **extra}


def run_in_own_session(args: Sequence[str], env: Mapping[str, str], timeout: float) -> subprocess.CompletedProcess:
    """Run a command in a session of its own and capture its output as text.

    A command that outlives ``timeout`` has its whole process group killed while it is still unreaped, so
    the group is still its own, and then TimeoutExpired is raised: nothing it started stays behind, as it
    would if only the command itself were killed.
    """
    proc = subprocess.Popen(
        list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env),
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
