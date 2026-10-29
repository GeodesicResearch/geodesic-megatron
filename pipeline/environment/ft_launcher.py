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

Standard library only; runs inside the pipeline container, where ft_launcher is on PATH.

Catalogue:
    SECTION_TIMEOUT_FLAGS: the section-timeout flags the launcher passes to ft_launcher.
    help_text: ft_launcher's --help output.
    missing_flags: the flags a help text lacks.
"""

import subprocess


SECTION_TIMEOUT_FLAGS = ("--ft-rank-section-timeouts", "--ft-rank-out-of-section-timeout")


def help_text() -> str:
    """``ft_launcher --help``'s stdout and stderr; runs ft_launcher (from PATH) in a subprocess, 120 s at most."""
    out = subprocess.run(["ft_launcher", "--help"], capture_output=True, text=True, timeout=120)
    return out.stdout + out.stderr


def missing_flags(text: str) -> list[str]:
    """The ``SECTION_TIMEOUT_FLAGS`` that the help text ``text`` does not contain, in order."""
    return [flag for flag in SECTION_TIMEOUT_FLAGS if flag not in text]
