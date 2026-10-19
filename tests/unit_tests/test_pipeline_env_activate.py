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

"""Tests for pipeline_env_activate.sh: the TMPDIR fallback every in-container shell inherits."""

import os
import subprocess
from pathlib import Path


ACTIVATE = Path(__file__).resolve().parents[2] / "pipeline_env_activate.sh"


def _tmpdir_after_activate(tmpdir: str | None, fallback_root: str) -> str:
    """TMPDIR as the real activate script leaves it, sourced with TMPDIR set to ``tmpdir`` (unset when None).

    ``fallback_root`` pins GEODESIC_CONTAINER_TMPDIR_FALLBACK so the test exercises the
    fallback logic against its own scratch rather than writing the production per-user
    directory on project storage.
    """
    env = {k: v for k, v in os.environ.items() if k != "TMPDIR"}
    env["GEODESIC_CONTAINER_TMPDIR_FALLBACK"] = fallback_root
    if tmpdir is not None:
        env["TMPDIR"] = tmpdir
    result = subprocess.run(
        ["bash", "-c", f'source "{ACTIVATE}" >/dev/null && printf "%s" "$TMPDIR"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_an_unset_tmpdir_falls_back_to_a_created_dir(tmp_path):
    fallback_root = tmp_path / "fallback"
    assert _tmpdir_after_activate(None, str(fallback_root)) == str(fallback_root)
    assert fallback_root.is_dir()  # activate creates the fallback it exports


def test_a_caller_set_tmpdir_is_kept(tmp_path):
    kept = tmp_path / "kept"
    kept.mkdir()
    fallback_root = tmp_path / "fallback"
    assert _tmpdir_after_activate(str(kept), str(fallback_root)) == str(kept)
    assert not fallback_root.exists()
