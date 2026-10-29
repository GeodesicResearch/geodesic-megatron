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
"""tests/unit_tests/stubbed_shell.py: a stubbed shell cannot re-enter itself or outlive its test, and
PipelineShell runs the real pipeline scripts against its recording stubs.

The first two tests run real bash processes. The startup hook in the first stands in for the pipeline container's
/etc/shinit_v2 and stops calling nvidia-smi after HOOK_DEPTH_LIMIT levels, so even an environment that
carried it would recurse a bounded number of times, inside a 10 s timeout whose expiry kills the whole
process group.
"""

import os
import stat
import subprocess
import time

import pytest

from tests.unit_tests.stubbed_shell import PipelineShell, run_in_own_session, stubbed_shell_env


HOOK_DEPTH_LIMIT = 20


def _executable(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _gone(pid):
    """True once no process has this pid, waiting briefly for an orphan to be reaped."""
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_a_shell_stub_of_nvidia_smi_runs_once_whatever_bash_startup_hook_the_caller_has(tmp_path, monkeypatch):
    """The container's BASH_ENV hook calls nvidia-smi through PATH before a bash runs its first line; carried
    into the stubbed environment, each shell stub of nvidia-smi would start the next."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub_runs = tmp_path / "stub_runs"
    hook_runs = tmp_path / "hook_runs"
    _executable(bindir / "nvidia-smi", f'#!/bin/bash\necho "$$" >> "{stub_runs}"\n')
    hook = tmp_path / "bash_env"
    hook.write_text(
        f'echo sourced >> "{hook_runs}"\n'
        f'[ "$(wc -l < "{hook_runs}")" -ge {HOOK_DEPTH_LIMIT} ] || nvidia-smi -q -d COMPUTE\n'
    )
    monkeypatch.setenv("BASH_ENV", str(hook))
    monkeypatch.setenv("ENV", str(hook))

    result = run_in_own_session(["bash", "-c", "nvidia-smi -q -d COMPUTE"], stubbed_shell_env(bindir, {}), timeout=10)

    assert result.returncode == 0, result.stderr
    assert not hook_runs.exists()
    pids = [int(pid) for pid in stub_runs.read_text().split()]
    assert len(pids) == 1
    assert all(_gone(pid) for pid in pids)


def test_a_command_that_outlives_its_timeout_takes_everything_it_started_with_it(tmp_path):
    started = tmp_path / "started"
    script = f'sleep 60 & echo "$!" > "{started}"; wait'

    with pytest.raises(subprocess.TimeoutExpired):
        run_in_own_session(["bash", "-c", script], stubbed_shell_env(tmp_path, {}), timeout=2)

    assert _gone(int(started.read_text()))


@pytest.fixture
def shell(tmp_path):
    shell = PipelineShell(tmp_path, record_env=("PYTHONUNBUFFERED", "HF_HOME"))
    yield shell
    shell.remove_job_tmp_dirs()


def test_a_pipeline_script_runs_through_the_real_shim_and_its_launch_is_recorded(shell, tmp_path):
    """The coherence wrapper's single-process path: config, exec shim and activation run for real; the image the
    shim chose and the python launch they reach are recorded, the launch with the environment asked for."""
    run = shell.run("pipeline_coherence_submit.sbatch", f"{tmp_path}/model", "--max-tokens", "8")

    assert run["returncode"] == 0, run["stderr"]
    assert run["calls"] == [
        {"cmd": "apptainer", "argv": [str(tmp_path / "container" / "nemo.sif")], "env": {"PYTHONUNBUFFERED": "1"}},
        {
            "cmd": "python",
            "argv": ["pipeline_coherence_test.py", f"{tmp_path}/model", "--max-tokens", "8"],
            "env": {"PYTHONUNBUFFERED": "1", "HF_HOME": "/projects/a5k/public/hf"},
        },
    ]
    assert shell.normalise(run)["stdout"].startswith("===== Coherence Test =====\nJob ID:    <JOB>\n")


def _iteration(run_dir, n, converted=False):
    iteration = run_dir / f"iter_{n:07d}"
    iteration.mkdir(parents=True)
    (iteration / "run_config.yaml").write_text("train:\n  train_iters: 2\n")
    if converted:
        (iteration / "hf").mkdir()
        (iteration / "hf" / "config.json").write_text("{}\n")


def test_a_pushed_folder_reaches_the_recording_hub_and_a_conversion_reaches_torchrun(tmp_path):
    """upload-all of a finished run: iteration 1 converts (srun, then torchrun), and the already-exported final
    iteration 2 runs the script's `python -c` push for real against the recording Hub, once per revision."""
    shell = PipelineShell(tmp_path, record=("srun", "torchrun", "hub"))
    _iteration(tmp_path / "run", 1)
    _iteration(tmp_path / "run", 2, converted=True)
    (tmp_path / "run" / "latest_checkpointed_iteration.txt").write_text("2\n")
    run = shell.run(
        "pipeline_checkpoint_convert.sh",
        "upload-all",
        str(tmp_path / "run"),
        "--hf-model",
        "x/y",
        "--no-reasoning",
        "--hf-org",
        "org",
    )
    shell.remove_job_tmp_dirs()

    assert run["returncode"] == 0, run["stderr"]
    calls = [(c["cmd"], c["argv"][0], c.get("kwargs")) for c in run["calls"]]
    assert calls[0][:2] == ("srun", "--nodes=1")
    assert calls[1][:2] == ("torchrun", "--nproc_per_node=4")
    folder = str(tmp_path / "run" / "iter_0000002" / "hf")
    for revision, message in (("iter_0000002", "Add checkpoint 2"), ("main", "Final checkpoint (iteration 2)")):
        assert ("hub", "create_repo", {"repo_id": "org/run", "exist_ok": True}) in calls
        assert ("hub", "create_branch", {"repo_id": "org/run", "branch": revision}) in calls
        upload = {"folder_path": folder, "repo_id": "org/run", "revision": revision, "commit_message": message}
        assert ("hub", "upload_folder", {**upload, "files": ["config.json"]}) in calls


def test_a_failed_rank_launch_fails_the_script(tmp_path):
    shell = PipelineShell(tmp_path)
    run = shell.run(
        "pipeline_checkpoint_convert.sh",
        "export",
        "/ckpt",
        "--hf-model",
        "x/y",
        "--no-reasoning",
        env={"STUB_TORCHRUN_RC": "3"},
    )
    shell.remove_job_tmp_dirs()
    assert run["returncode"] == 3
    assert [c["cmd"] for c in run["calls"]] == ["srun", "apptainer", "torchrun"]


def test_a_poll_loop_is_bounded_by_the_sleep_stub_and_advanced_by_its_hook(shell, tmp_path):
    """Each wait runs STUB_ON_SLEEP; a wait past STUB_SLEEP_LIMIT fails, so a loop that never ends fails its test."""
    _iteration(tmp_path / "run", 1)
    (tmp_path / "run" / "latest_checkpointed_iteration.txt").write_text("1\n")
    run = shell.run(
        "pipeline_checkpoint_convert.sh",
        "upload-all",
        str(tmp_path / "run"),
        "--hf-model",
        "x/y",
        "--no-reasoning",
        "--poll",
        env={"STUB_SLEEP_LIMIT": "2", "STUB_ON_SLEEP": f"echo wait >> {tmp_path}/waits"},
    )

    assert run["returncode"] != 0
    assert [c["argv"] for c in run["calls"] if c["cmd"] == "sleep"] == [["300"]] * 3
    assert (tmp_path / "waits").read_text() == "wait\nwait\n"
    assert "more than 2 waits" in run["stderr"]


@pytest.mark.parametrize(
    "nodelist, hosts",
    [
        ("nid000001", ["nid000001"]),
        # Each range keeps the width of its own lower bound, as SLURM's does.
        ("nid[001-003,7],nid9", ["nid001", "nid002", "nid003", "nid7", "nid9"]),
        ("nid[010]", ["nid010"]),
    ],
)
def test_scontrol_expands_node_lists(shell, nodelist, hosts):
    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True, env=shell.env(), timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == hosts


def test_normalise_replaces_the_tmp_dir_the_repo_root_and_the_job_id(shell, tmp_path):
    from tests.unit_tests.child_process import REPO_ROOT

    value = {"a": [f"{tmp_path}/x", f"{REPO_ROOT}/y", f"/tmp/megatron_convert_{shell.job_id}"]}
    assert shell.normalise(value) == {"a": ["<TMP>/x", "<REPO>/y", "/tmp/megatron_convert_<JOB>"]}
    assert int(shell.job_id) % 1000 == 0


def test_scontrol_ends_quietly_when_its_reader_stops_early(shell):
    """`scontrol show hostname LIST | head -1` takes the first host; the rest must not fail on the closed pipe."""
    # 100,000 hosts overflow the pipe buffer, so the stub is still writing when head exits.
    result = subprocess.run(
        ["bash", "-c", "scontrol show hostname 'nid[000000-099999]' | head -1"],
        capture_output=True,
        text=True,
        env=shell.env(),
        timeout=30,
    )
    assert (result.returncode, result.stdout, result.stderr) == (0, "nid000000\n", "")


def test_a_pipeline_module_runs_for_real_and_is_not_recorded(shell, tmp_path):
    """`python -m pipeline.*` is helper code the scripts depend on: it runs (and finds the repo's package)."""
    from tests.unit_tests.checkpoint_dirs import make_checkpoint_dir

    run_dir = make_checkpoint_dir(tmp_path / "run", [5], tracker=5)
    result = subprocess.run(
        ["bash", "-c", f"cd {shell.env()['GEODESIC_REPO_DIR']} && python -m pipeline.checkpoints completed {run_dir}"],
        capture_output=True,
        text=True,
        env=shell.env(),
        timeout=120,
    )
    assert (result.returncode, result.stdout) == (0, "5 -\n"), result.stderr
    assert not shell.calls_file.exists()
