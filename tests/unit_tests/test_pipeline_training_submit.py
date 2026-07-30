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
the genuinely-untestable boundary here; the script under test is the real one.
"""

import os
import stat
import subprocess


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_submit(tmp_path, extra_args):
    """Run the real sbatch script with stub launcher + stub isambard_sbatch."""
    stub_repo = tmp_path / "stub_repo"
    stub_repo.mkdir()
    launcher = stub_repo / "pipeline_training_launch.sh"
    launcher.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)

    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub_sbatch = bindir / "isambard_sbatch"
    stub_sbatch.write_text("#!/bin/bash\nexit 0\n")
    stub_sbatch.chmod(stub_sbatch.stat().st_mode | stat.S_IEXEC)

    env = dict(os.environ)
    env["GEODESIC_REPO_DIR"] = str(stub_repo)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    result = subprocess.run(
        ["bash", os.path.join(REPO_ROOT, "pipeline_training_submit.sbatch"), "cfg.yaml", "super", "sft", *extra_args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_launcher_flags_forward_raw(tmp_path):
    """--disable-ft must reach the launcher as a top-level arg, not behind --."""
    args = _run_submit(tmp_path, ["--disable-ft"])
    assert args == ["cfg.yaml", "--model", "super", "--mode", "sft", "--disable-ft"]


def test_hydra_overrides_still_forward(tmp_path):
    """Non-flag extras (Hydra overrides) pass through unchanged alongside flags."""
    args = _run_submit(tmp_path, ["--disable-ft", "train.train_iters=32", "checkpoint.save=null"])
    assert args == [
        "cfg.yaml",
        "--model",
        "super",
        "--mode",
        "sft",
        "--disable-ft",
        "train.train_iters=32",
        "checkpoint.save=null",
    ]
