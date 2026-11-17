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
"""pipeline_cluster_submit.sh: the command a job is submitted with on each cluster.

The real script runs against the real ``pipeline/sh/cluster.sh`` and a profile of the test's own, whose log root is
a scratch directory: a real profile's log root is shared storage, where the script creates the log's directory.
``sbatch`` and the site wrapper are SLURM's, so stubs that record their arguments stand in for them.
"""

import shutil
import subprocess

import pytest

from pipeline.cluster import cluster_profile
from tests.unit_tests.child_process import REPO_ROOT


FRONT_DOOR = "pipeline_cluster_submit.sh"
# The job scripts that carry a "#PIPELINE-SUBMIT" line, with the sbatch options it becomes on an 8-GPU node.
HINTED = {
    "pipeline_training_submit.sbatch": ["--gpus-per-node=8", "--output={logs}/megatron_runs/train-%j.out"],
    "pipeline_checkpoint_submit.sbatch": ["--gpus-per-node=8", "--output={logs}/slurm/convert-checkpoint-%j.out"],
    "pipeline_coherence_submit.sbatch": ["--gpus-per-node=8", "--output={logs}/slurm/coherence-test-%j.out"],
    "pipeline_env_submit.sbatch": ["--gpus-per-node=8", "--output={logs}/slurm/env-setup-%j.out"],
    "scripts/run_unit_tests.sbatch": ["--gpus-per-node=8", "--output={logs}/slurm/unit-tests-%j.out"],
    "scripts/checkpoint/export_check.sbatch": ["--gpus-per-node=8", "--output={logs}/slurm/export-check-%j.out"],
    # A data or corpus job holds no GPU and shares its node with the account's other jobs.
    "configs/control_pretraining/corpus_job.sbatch": [
        "--gpus-per-node=0",
        "--exclusive=user",
        "--cpus-per-task=16",
        "--mem=64G",
        "--output={logs}/slurm/corpus-%j.out",
    ],
    "configs/control_pretraining/shard_jsonl_corpus.sh": [
        "--gpus-per-node=0",
        "--exclusive=user",
        "--cpus-per-task=8",
        "--mem=32G",
        "--output={logs}/slurm/shard-jsonl-%j.out",
    ],
    "pipeline_data_submit.sbatch": [
        "--gpus-per-node=0",
        "--exclusive=user",
        "--cpus-per-task=64",
        "--mem=256G",
        "--output={logs}/slurm/data-prep-%j.out",
    ],
}


@pytest.fixture
def checkout(tmp_path):
    """A directory holding the real front door and cluster.sh, one profile ``testbed`` and stub submit commands.

    Returns a function ``submit(*args, wrapper="", options=None, cwd=REPO_ROOT)`` that runs the front door from
    ``cwd`` (with PIPELINE_SUBMIT_OPTIONS set to ``options`` when given) and returns the completed process, the
    arguments the stub recorded (None when none ran), and the profile's log root.
    """
    root = tmp_path / "checkout"
    (root / "pipeline" / "sh" / "clusters").mkdir(parents=True)
    shutil.copy(REPO_ROOT / FRONT_DOOR, root / FRONT_DOOR)
    shutil.copy(REPO_ROOT / "pipeline" / "sh" / "cluster.sh", root / "pipeline" / "sh" / "cluster.sh")
    logs = tmp_path / "logs"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "site_sbatch"):
        (bindir / name).write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {tmp_path}/{name}.args\necho 4242\n')
        (bindir / name).chmod(0o755)

    def submit(*args: str, wrapper: str = "", options: str | None = None, cwd=REPO_ROOT):
        (root / "pipeline" / "sh" / "clusters" / "testbed.sh").write_text(
            "testbed_detect() { false; }\n"
            "testbed_profile() {\n"
            f'    CLUSTER_GPUS_PER_NODE=8; CLUSTER_UMASK=""; CLUSTER_SUBMIT_WRAPPER="{wrapper}"\n'
            f"    CLUSTER_LOG_ROOT={logs}\n"
            "}\n"
        )
        result = subprocess.run(
            ["bash", str(root / FRONT_DOOR), *args],
            env={
                "PATH": f"{bindir}:/usr/bin:/bin",
                "USER": "probe",
                "GEODESIC_CLUSTER": "testbed",
                **({} if options is None else {"PIPELINE_SUBMIT_OPTIONS": options}),
            },
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        recorded = tmp_path / ("site_sbatch.args" if wrapper else "sbatch.args")
        return result, recorded.read_text().splitlines() if recorded.exists() else None, logs

    return submit


@pytest.mark.parametrize("script", sorted(HINTED))
def test_each_job_scripts_line_becomes_its_sbatch_options_ahead_of_the_callers(checkout, script):
    result, argv, logs = checkout("--nodes=2", script, "config.yaml", "nano", "sft")
    assert (result.returncode, result.stdout) == (0, "4242\n")
    assert argv == [option.format(logs=logs) for option in HINTED[script]] + [
        "--nodes=2",
        script,
        "config.yaml",
        "nano",
        "sft",
    ]
    # sbatch creates no directory for a job's log, and a job whose log cannot be opened fails without one.
    assert (logs / HINTED[script][-1].split("=", 1)[1].format(logs="").lstrip("/")).parent.is_dir()
    assert result.stderr.startswith("+ sbatch --gpus-per-node=")


def test_an_option_of_the_caller_follows_the_lines_so_sbatch_takes_the_callers(checkout):
    _, argv, _ = checkout("--gpus-per-node=2", "--output", "/elsewhere/%j.out", "pipeline_training_submit.sbatch", "c")
    assert argv.index("--gpus-per-node=8") < argv.index("--gpus-per-node=2")
    assert argv[-4:] == ["--output", "/elsewhere/%j.out", "pipeline_training_submit.sbatch", "c"]


def test_the_shells_options_follow_the_lines_and_precede_the_callers(checkout, tmp_path):
    """PIPELINE_SUBMIT_OPTIONS reaches the jobs a script submits for the caller. Its words are taken as written: a
    node list's brackets match no file, even where one of that name exists."""
    (tmp_path / "--exclude=slinky-1").write_text("")  # what an unprotected glob would expand to
    options = "--exclude=slinky-[1,3,5,6] --mem=320G"
    script = str(REPO_ROOT / "pipeline_data_submit.sbatch")
    _, argv, _ = checkout("--mem=8G", script, "prepare", options=options, cwd=tmp_path)
    assert argv[:5] == ["--gpus-per-node=0", "--exclusive=user", "--cpus-per-task=64", "--mem=256G", argv[4]]
    assert argv[5:] == ["--exclude=slinky-[1,3,5,6]", "--mem=320G", "--mem=8G", script, "prepare"]


@pytest.mark.parametrize(
    "before",
    [[], ["--nodes", "2"], ["--nodes=2", "--time", "00:10:00", "--exclusive"], ["--parsable", "--nodes", "2"]],
)
def test_the_script_is_found_behind_options_with_and_without_values(checkout, before):
    """An option without "=" takes the next word as its value, unless that word is the script itself."""
    _, argv, _ = checkout(*before, "pipeline_env_submit.sbatch", "smoke")
    assert argv[0] == "--gpus-per-node=8"
    assert argv[2:] == [*before, "pipeline_env_submit.sbatch", "smoke"]


def test_a_script_without_a_line_is_submitted_as_given(checkout, tmp_path):
    script = tmp_path / "plain.sbatch"
    script.write_text("#!/bin/bash\n#SBATCH --nodes=1\ntrue\n")
    result, argv, logs = checkout("--time=1", str(script))
    assert argv == ["--time=1", str(script)]
    assert result.stderr == f"+ sbatch --time=1 {script}\n"
    assert not logs.exists()


def test_a_site_wrapper_gets_the_arguments_unchanged(checkout):
    """Isambard's shape: the site's wrapper shapes the job from the script's #SBATCH lines."""
    result, argv, logs = checkout(
        "--nodes=16", "pipeline_training_submit.sbatch", "c", "nano", "sft", wrapper="site_sbatch"
    )
    assert (result.returncode, result.stdout, result.stderr) == (0, "4242\n", "")
    assert argv == ["--nodes=16", "pipeline_training_submit.sbatch", "c", "nano", "sft"]
    assert not logs.exists()


def test_two_lines_in_one_script_are_refused(checkout, tmp_path):
    script = tmp_path / "twice.sbatch"
    script.write_text("#!/bin/bash\n#PIPELINE-SUBMIT gpus-per-node=node\n#PIPELINE-SUBMIT mem=1G\n")
    result, argv, _ = checkout(str(script))
    assert result.returncode == 1 and argv is None
    assert "more than one #PIPELINE-SUBMIT line" in result.stderr


@pytest.mark.parametrize("args", [[], ["--nodes=2"], ["--nodes", "2", "no_such_script.sbatch"]])
def test_arguments_without_a_script_file_are_refused_with_the_usage(checkout, args):
    result, argv, _ = checkout(*args)
    assert result.returncode == 2 and argv is None
    assert "no script file among the arguments" in result.stderr


def test_the_real_profiles_submit_through_the_front_door_or_their_wrapper():
    """Isambard names its site wrapper. Verda names none, so its jobs take the options of their scripts' lines."""
    assert cluster_profile("isambard")["CLUSTER_SUBMIT_COMMAND"] == "isambard_sbatch"
    assert cluster_profile("verda")["CLUSTER_SUBMIT_COMMAND"] == f"./{FRONT_DOOR}"
    assert cluster_profile("verda")["CLUSTER_GPUS_PER_NODE"] == "8"
