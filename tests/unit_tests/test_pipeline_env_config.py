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

"""Unit tests for pipeline_env_config.env: the bind list the container shim mounts."""

import os
import subprocess
from pathlib import Path


CONFIG = Path(__file__).resolve().parents[2] / "pipeline_env_config.env"


def _binds(tmpdir: str | None) -> list[str]:
    """CONTAINER_BINDS as the real config derives it in a shell whose TMPDIR is ``tmpdir`` (unset when None)."""
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"], "USER": "probe", "REPO_DIR": "/repo"}
    if tmpdir is not None:
        env["TMPDIR"] = tmpdir
    result = subprocess.run(
        ["bash", "-c", f'source "{CONFIG}" && printf "%s" "$CONTAINER_BINDS"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split(",")


def test_an_existing_tmpdir_is_bound_at_its_own_path(tmp_path):
    # The site gives each job a node-local TMPDIR that apptainer does not bind; unbound, the path lands on the
    # image's read-only root inside the container.
    job_tmp = tmp_path / "local_user"
    job_tmp.mkdir()
    assert _binds(str(job_tmp)) == _binds(None) + [str(job_tmp)]


def test_without_a_tmpdir_the_bind_list_is_the_fixed_set():
    binds = _binds(None)
    assert binds[:3] == ["/projects", "/lus", "/repo"]
    assert len(binds) == len(set(binds))


def test_a_tmpdir_that_does_not_exist_adds_no_bind(tmp_path):
    # apptainer refuses a bind whose source is missing, which would end every container command of the job.
    assert _binds(str(tmp_path / "missing")) == _binds(None)
