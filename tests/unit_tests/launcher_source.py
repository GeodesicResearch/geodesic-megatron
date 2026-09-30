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
"""Access to pipeline_training_launch.sh for tests of its shell functions.

The launcher cannot be sourced whole -- it allocates nodes and execs srun -- so a function under
test is lifted from it by name and run in a harness under the launcher's own shell options. A
renamed function makes the lookup fail loudly rather than silently testing nothing.
"""

import os
import re
import subprocess


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAUNCHER = os.path.join(REPO_ROOT, "pipeline_training_launch.sh")


def launcher_function(name: str) -> str:
    """The named bash function's source, verbatim from the launcher."""
    src = open(LAUNCHER).read()
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", src, re.S | re.M)
    assert match, f"{name}() not found in pipeline_training_launch.sh"
    return match.group(0)


def env_override_entries(path: str) -> list[str]:
    """The KEY=VALUE lines the launcher's ISAMBARD_ENV_OVERRIDES hook takes from the file at ``path``, in order.

    The hook's parser, ``apply_env_overrides``, runs verbatim under the launcher's shell options, so a line
    the launcher skips is skipped here too, and a file it refuses fails with the launcher's own message.
    """
    script = (
        "set -euo pipefail\n"
        f"{launcher_function('apply_env_overrides')}\n"
        'apply_env_overrides "$1"\n'
        'for entry in "${ENV_OVERRIDE_ENTRIES[@]}"; do printf "%s\\0" "$entry"; done\n'
    )
    result = subprocess.run(
        ["bash", "-c", script, "harness", path],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"]},
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split("\0")[:-1]
