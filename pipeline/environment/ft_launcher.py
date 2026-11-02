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
"""The ft_launcher (nvidia-resiliency-ext) flags pipeline_training_launch.sh passes, and whether the image has them.

Standard library only; runs inside the pipeline container, where ft_launcher is on PATH. As
``python -m pipeline.environment.ft_launcher`` it is the launcher's launch gate (``main``).

Catalogue:
    SECTION_TIMEOUT_FLAGS: the section-timeout flags the launcher passes to ft_launcher.
    ft_launcher_help: run ft_launcher --help.
    missing_flags: the flags a help text lacks.
    main: the command-line check.
"""

import subprocess
import sys


SECTION_TIMEOUT_FLAGS = ("--ft-rank-section-timeouts", "--ft-rank-out-of-section-timeout")


def ft_launcher_help() -> subprocess.CompletedProcess:
    """Run ``ft_launcher --help`` (from PATH, in a subprocess, 120 s at most) and return its result, output as text."""
    return subprocess.run(["ft_launcher", "--help"], capture_output=True, text=True, timeout=120)


def missing_flags(text: str) -> list[str]:
    """The ``SECTION_TIMEOUT_FLAGS`` that the help text ``text`` does not contain, in order."""
    return [flag for flag in SECTION_TIMEOUT_FLAGS if flag not in text]


def main() -> int:
    """Check ``ft_launcher --help`` for every ``SECTION_TIMEOUT_FLAGS`` flag.

    Returns 1, after saying why on stderr, when ``ft_launcher --help`` exits nonzero or its output (stdout and
    stderr) lacks one of the flags, and 0 otherwise.
    """
    help_run = ft_launcher_help()
    if help_run.returncode != 0:
        print(f"ft_launcher --help exited {help_run.returncode}", file=sys.stderr)
        return 1
    missing = missing_flags(help_run.stdout + help_run.stderr)
    for flag in missing:
        print(f"ft_launcher lacks {flag}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
