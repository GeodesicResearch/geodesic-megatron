#!/usr/bin/env python3
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

"""The launch settings an environment would pass to a training job it submits.

``isambard_sbatch`` exports the submitting shell's environment to the job, so an ``ISAMBARD_*``, ``TRAIN_*`` or
``GEODESIC_CONTAINER_*`` variable set there reaches ``pipeline_training_launch.sh``, ``pipeline_env_activate.sh``
or the container config and changes the run with no config field naming it: an exported
``ISAMBARD_FP32_SSM_STATE=0`` turns the fp32 SSM-state patch off. A launch whose posture must be exactly its
config and its ``ISAMBARD_ENV_OVERRIDES`` file refuses to submit from such an environment. The submission
wrapper's own ``ISAMBARD_SBATCH_*``, the tunnel's ``ISAMBARD_TUNNEL_*`` and the site's ``ISAMBARD_HOST`` (the
system's name, which SLURM sets in every task's environment, so every tunnel shell holds it) are not launch
settings.

USAGE
    python3 scripts/training/launch_environment.py && <submission command>

Exit 1, naming the inherited settings, when the environment holds any; 0 otherwise. Batch scripts run it with
the node's own ``python3`` before entering the container, which on Isambard can be the system Python 3.6, so the
module keeps to what 3.6 parses (``typing``'s generics, no ``from __future__ import annotations``).
"""

import os
import sys
from typing import List, Mapping


LAUNCH_SETTING_PREFIXES = ("ISAMBARD_", "TRAIN_", "GEODESIC_CONTAINER_")
NOT_LAUNCH_SETTING_PREFIXES = ("ISAMBARD_SBATCH_", "ISAMBARD_TUNNEL_")
SITE_VARIABLES = ("ISAMBARD_HOST",)


def inherited_launch_settings(environ: Mapping[str, str]) -> List[str]:
    """The names of the launch settings ``environ`` holds, sorted."""
    return sorted(
        name
        for name in environ
        if name.startswith(LAUNCH_SETTING_PREFIXES)
        and not name.startswith(NOT_LAUNCH_SETTING_PREFIXES)
        and name not in SITE_VARIABLES
    )


def main() -> int:
    """Refuse (exit 1) when this process's environment holds a launch setting."""
    inherited = inherited_launch_settings(os.environ)
    if inherited:
        print(f"FATAL: launch settings inherited from the submitting shell: {' '.join(inherited)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
