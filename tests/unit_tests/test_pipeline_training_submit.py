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
"""Argument-forwarding tests for pipeline_training_submit.sbatch.

The sbatch wrapper forwards args 4+ to pipeline_training_launch.sh RAW so
launcher flags (e.g. --disable-ft) are parsed as launcher flags while unknown
args (Hydra overrides) fall through to the training script. These tests run the
real sbatch script as a subprocess with a stub launcher (via GEODESIC_REPO_DIR)
and a stub isambard_sbatch on PATH -- SLURM submission and the real launcher are
the genuinely-untestable boundary here; the script under test is the real one,
and so are the shell libraries its submission guard reads from REPO_DIR.
"""

import os
import stat
import subprocess

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _submit(tmp_path, extra_args, cluster="isambard", check_status=0, with_libraries=True):
    """The real sbatch script's result, with a stub launcher in REPO_DIR and stub SLURM commands on PATH.

    ``check_status`` is the exit status of ``isambard_sbatch --check``. scancel records its arguments in
    ``tmp_path/cancelled``.
    """
    stub_repo = tmp_path / "stub_repo"
    stub_repo.mkdir()
    launcher = stub_repo / "pipeline_training_launch.sh"
    launcher.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    if with_libraries:
        (stub_repo / "pipeline").mkdir()
        (stub_repo / "pipeline" / "sh").symlink_to(os.path.join(REPO_ROOT, "pipeline", "sh"))

    bindir = tmp_path / "bin"
    bindir.mkdir()
    stubs = {"scancel": f'#!/bin/bash\necho "$@" > {tmp_path}/cancelled\n'}
    if cluster == "isambard":  # the site's wrapper exists on Isambard alone
        stubs["isambard_sbatch"] = f"#!/bin/bash\nexit {check_status}\n"
    for name, body in stubs.items():
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)

    env = dict(os.environ)
    env.update(GEODESIC_REPO_DIR=str(stub_repo), GEODESIC_CLUSTER=cluster, SLURM_JOB_ID="77")
    env["PATH"] = f"{bindir}:/usr/bin:/bin"
    return subprocess.run(
        ["bash", os.path.join(REPO_ROOT, "pipeline_training_submit.sbatch"), "cfg.yaml", "super", "sft", *extra_args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def _run_submit(tmp_path, extra_args):
    """The arguments the stub launcher was handed."""
    result = _submit(tmp_path, extra_args)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_a_job_the_node_cap_check_refuses_is_cancelled_before_the_launcher(tmp_path):
    result = _submit(tmp_path, [], check_status=1)
    assert (result.returncode, result.stdout) == (1, "")
    assert (tmp_path / "cancelled").read_text() == "77\n"


def test_a_cluster_without_a_submit_wrapper_launches_without_one(tmp_path):
    """Verda has no isambard_sbatch, and its profile's guard asks for none."""
    result = _submit(tmp_path, ["--disable-ft"], cluster="verda")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "--disable-ft"
    assert not (tmp_path / "cancelled").exists()


@pytest.mark.parametrize("cluster", ["isambard", "verda"])
def test_a_repo_dir_without_the_guard_is_refused(tmp_path, cluster):
    """The guard is REPO_DIR's file, so a REPO_DIR that holds none cannot be checked, and nothing is launched."""
    result = _submit(tmp_path, [], cluster=cluster, with_libraries=False)
    assert (result.returncode, result.stdout) == (1, "")
    assert "pipeline/sh/job_guard.sh" in result.stderr
    assert (tmp_path / "cancelled").read_text() == "77\n"


def test_launcher_flags_forward_raw(tmp_path):
    """--disable-ft must reach the launcher as a top-level arg, not behind --."""
    args = _run_submit(tmp_path, ["--disable-ft"])
    assert args == [str(tmp_path / "stub_repo" / "cfg.yaml"), "--model", "super", "--mode", "sft", "--disable-ft"]


def test_hydra_overrides_still_forward(tmp_path):
    """Non-flag extras (Hydra overrides) pass through unchanged alongside flags."""
    args = _run_submit(tmp_path, ["--disable-ft", "train.train_iters=32", "checkpoint.save=null"])
    assert args == [
        str(tmp_path / "stub_repo" / "cfg.yaml"),
        "--model",
        "super",
        "--mode",
        "sft",
        "--disable-ft",
        "train.train_iters=32",
        "checkpoint.save=null",
    ]
