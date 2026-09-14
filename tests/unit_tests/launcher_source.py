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


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAUNCHER = os.path.join(REPO_ROOT, "pipeline_training_launch.sh")


def launcher_function(name: str) -> str:
    """The named bash function's source, verbatim from the launcher."""
    src = open(LAUNCHER).read()
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", src, re.S | re.M)
    assert match, f"{name}() not found in pipeline_training_launch.sh"
    return match.group(0)
