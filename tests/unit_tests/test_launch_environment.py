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

"""Unit tests for scripts/training/launch_environment.py (launch settings inherited from a submitting shell)."""

import subprocess
import sys
from pathlib import Path

import pytest
from scripts.training.launch_environment import inherited_launch_settings


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "training" / "launch_environment.py"


def test_every_launcher_activate_and_container_setting_is_inherited():
    environ = {
        "ISAMBARD_FP32_SSM_STATE": "0",
        "TRAIN_PERSISTENT_TRITON_CACHE": "1",
        "GEODESIC_CONTAINER_SIF": "/x.sif",
        "ISAMBARD_ENV_OVERRIDES": "/x.env",
        "PATH": "/usr/bin",
        "SLURM_JOB_ID": "1",
        "MY_ISAMBARD_NOTE": "x",
    }
    assert inherited_launch_settings(environ) == [
        "GEODESIC_CONTAINER_SIF",
        "ISAMBARD_ENV_OVERRIDES",
        "ISAMBARD_FP32_SSM_STATE",
        "TRAIN_PERSISTENT_TRITON_CACHE",
    ]


def test_the_submission_wrappers_and_tunnels_own_variables_are_not_launch_settings():
    environ = {"ISAMBARD_SBATCH_MAX_NODES": "300", "ISAMBARD_SBATCH_FORCE": "0", "ISAMBARD_TUNNEL_NAME": "t"}
    assert inherited_launch_settings(environ) == []


@pytest.mark.parametrize(
    "extra, status, message",
    [
        ({"ISAMBARD_CUDA_MAX_CONNECTIONS": "32", "TRAIN_X": "1"}, 1, "inherited from the submitting shell: "),
        ({"ISAMBARD_SBATCH_MAX_NODES": "300"}, 0, ""),
    ],
)
def test_the_command_refuses_an_environment_holding_a_launch_setting(extra, status, message):
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], env={"PATH": "/usr/bin:/bin", **extra}, capture_output=True, text=True
    )
    assert result.returncode == status
    assert message in result.stderr
    if status:
        assert result.stderr.strip().endswith("ISAMBARD_CUDA_MAX_CONNECTIONS TRAIN_X")
