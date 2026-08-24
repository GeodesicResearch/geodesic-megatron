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
real sbatch script as a subprocess with a stub launcher (pinned via
PIPELINE_REPO_DIR, or reached through the wrapper's own repo resolution)
and a stub isambard_sbatch on PATH -- SLURM submission and the real launcher are
the genuinely-untestable boundary here; the script under test is the real one.
"""

import os
import shutil
import stat
import subprocess


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_stub_repo(tmp_path):
    """A fake checkout: recording stub launcher + the repo marker file."""
    stub_repo = tmp_path / "stub_repo"
    stub_repo.mkdir()
    launcher = stub_repo / "pipeline_training_launch.sh"
    launcher.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    # The marker the wrapper's self-location branch keys on.
    (stub_repo / "pipeline_env_config.env").write_text("")
    return stub_repo


def _base_env(tmp_path):
    """os.environ + stub isambard_sbatch on PATH, repo-resolution inputs scrubbed."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub_sbatch = bindir / "isambard_sbatch"
    stub_sbatch.write_text("#!/bin/bash\nexit 0\n")
    stub_sbatch.chmod(stub_sbatch.stat().st_mode | stat.S_IEXEC)

    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    # Scrub every input the wrapper's repo resolution reads, so each test sets
    # exactly the branch it exercises (the surrounding SLURM job would otherwise
    # leak its own SLURM_SUBMIT_DIR into the fallback branch).
    env.pop("PIPELINE_REPO_DIR", None)
    env.pop("SLURM_SUBMIT_DIR", None)
    return env


def _run_wrapper(script_path, env, extra_args=()):
    result = subprocess.run(
        ["bash", str(script_path), "cfg.yaml", "super", "sft", *extra_args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def _run_submit(tmp_path, extra_args):
    """Run the real sbatch script with stub launcher + stub isambard_sbatch."""
    stub_repo = _make_stub_repo(tmp_path)
    env = _base_env(tmp_path)
    # PIPELINE_REPO_DIR is the wrapper's repo override — the "run this checkout's
    # script against another tree" seam — pointed at the stub so the exec'd
    # launcher is the recording stub rather than the real one.
    env["PIPELINE_REPO_DIR"] = str(stub_repo)
    return _run_wrapper(os.path.join(REPO_ROOT, "pipeline_training_submit.sbatch"), env, extra_args)


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


def test_self_location_runs_the_repo_the_script_sits_in(tmp_path):
    """In-place invocation needs no env at all: the script finds its own checkout.

    The wrapper copied into the stub repo sits next to the marker file, so its
    self-location branch resolves REPO_DIR to that repo and runs THAT repo's
    launcher — the worktree workflow, from any cwd, with neither override nor
    SLURM_SUBMIT_DIR set.
    """
    stub_repo = _make_stub_repo(tmp_path)
    wrapper = stub_repo / "pipeline_training_submit.sbatch"
    shutil.copy(os.path.join(REPO_ROOT, "pipeline_training_submit.sbatch"), wrapper)
    args = _run_wrapper(wrapper, _base_env(tmp_path))
    assert args == ["cfg.yaml", "--model", "super", "--mode", "sft"]


def test_a_spooled_copy_falls_back_to_the_submission_directory(tmp_path):
    """Without the marker next to the script, SLURM_SUBMIT_DIR picks the repo.

    Under real sbatch the script executes from SLURM's spool directory, where
    nothing sits beside it — self-location must NOT fire there, and the
    submission directory (where sbatch was invoked, i.e. the checkout) wins.
    """
    stub_repo = _make_stub_repo(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    wrapper = spool / "job_script.sh"
    shutil.copy(os.path.join(REPO_ROOT, "pipeline_training_submit.sbatch"), wrapper)
    env = _base_env(tmp_path)
    env["SLURM_SUBMIT_DIR"] = str(stub_repo)
    args = _run_wrapper(wrapper, env)
    assert args == ["cfg.yaml", "--model", "super", "--mode", "sft"]
