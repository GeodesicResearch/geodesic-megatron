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
"""A crashed rank must end the whole SLURM job, not park the survivors.

`srun --kill-on-bad-exit=0` tells srun to keep a step alive when a task exits non-zero.
Under that flag one dead rank leaves every other node allocated and idle until the
walltime expires: a 128-node run whose rank 428 raised on a NaN held 512 GPUs for over
three hours doing nothing, and the job still read RUNNING the whole time.

The launcher already propagates a non-zero srun correctly -- it runs under `set -e`, and
`pipeline_training_submit.sbatch` `exec`s it, so srun's status becomes the job's. That
propagation is dead code while srun refuses to return, which is why the flag is the fix
and why this asserts on the flag rather than on the exit path.
"""

import os
import re


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAUNCHER = os.path.join(REPO_ROOT, "pipeline_training_launch.sh")


def _srun_args() -> str:
    """The launcher's SRUN_ARGS assignment, read out of the real script."""
    src = open(LAUNCHER).read()
    match = re.search(r'^SRUN_ARGS="(.*)"$', src, re.M)
    assert match, "the SRUN_ARGS assignment was not found in pipeline_training_launch.sh"
    return match.group(1)


def test_a_failed_task_ends_the_step():
    """The regression: `=0` is what stranded 127 healthy nodes behind one dead rank."""
    args = _srun_args()
    assert "--kill-on-bad-exit=0" not in args, (
        "srun is told NOT to end the step when a task fails; a crashed rank will leave the "
        "rest of the allocation idle until the walltime expires"
    )
    assert "--kill-on-bad-exit=1" in args, f"srun must end the step when any task exits non-zero. SRUN_ARGS={args!r}"


def test_the_launcher_aborts_rather_than_reporting_success():
    """`set -e` is what turns srun's non-zero status into a non-zero job.

    Without it the launcher would run on to its completion banner and the job would exit 0
    on a crashed run, which is worse than the hang: it reads as success.
    """
    src = open(LAUNCHER).read()
    assert re.search(r"^set -euo pipefail$", src, re.M), (
        "the launcher must run under `set -e` so a failed srun ends the job non-zero"
    )
