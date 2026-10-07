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

"""The queue read and the job submission the campaign's submitting tools share (scripts/slurm_jobs.py).

SLURM itself is stood in for throughout: the real squeue and isambard_sbatch would query and submit to
the cluster. What is tested is how their output is read.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import slurm_jobs  # noqa: E402


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_the_queue_is_the_set_of_job_names(monkeypatch):
    monkeypatch.setattr(slurm_jobs.subprocess, "run", lambda *a, **k: _completed(0, stdout="a\n\nb\na\n"))
    assert slurm_jobs.queued_job_names() == {"a", "b"}


def test_a_listing_names_its_field_and_the_job_name_it_keeps_to():
    """The runner stands in for squeue, which would read the cluster's queue."""
    seen = []

    def run(command):
        seen.append(command)
        return _completed(0, stdout="101\n\n102\n")

    assert slurm_jobs.queue_listing("%i", "cp-stage", run) == ["101", "102"]
    assert seen == [["squeue", "--me", "--noheader", "--format=%i", "--name", "cp-stage"]]


def test_an_unreadable_queue_is_an_error_not_an_empty_queue(monkeypatch):
    """Read as empty, it would resubmit work already in flight."""
    monkeypatch.setattr(slurm_jobs.subprocess, "run", lambda *a, **k: _completed(1, stderr="slurm_load_jobs error"))
    with pytest.raises(slurm_jobs.SlurmError, match="squeue exited 1: slurm_load_jobs error"):
        slurm_jobs.queued_job_names()


def test_a_submission_returns_its_one_job_id_from_among_the_wrappers_reports(monkeypatch, tmp_path):
    seen = {}

    def run(command, **kwargs):
        seen.update(cwd=kwargs["cwd"], env=kwargs["env"])
        return _completed(0, stdout="Bad nodes: none\nStorage: /projects/a5k 87%\nSubmitted batch job 6935501\n")

    monkeypatch.setenv("SLURM_ALREADY_SET", "kept")
    monkeypatch.setattr(slurm_jobs.subprocess, "run", run)
    assert slurm_jobs.submit(["isambard_sbatch", "job.sbatch"], tmp_path, {"ISAMBARD_SBATCH_FORCE": "0"}) == "6935501"
    assert seen["cwd"] == tmp_path
    assert seen["env"]["ISAMBARD_SBATCH_FORCE"] == "0" and seen["env"]["SLURM_ALREADY_SET"] == "kept"


@pytest.mark.parametrize(
    "result,match",
    [
        (_completed(1, stderr="BLOCKED: this submission would put the account at 266 > 256 nodes"), "exited 1"),
        (_completed(0, stdout="Storage: /projects/a5k 87%\n"), "named 0 jobs"),
        (_completed(0, stdout="Submitted batch job 1\nSubmitted batch job 2\n"), "named 2 jobs"),
    ],
)
def test_a_refused_or_ambiguous_submission_is_an_error(monkeypatch, tmp_path, result, match):
    monkeypatch.setattr(slurm_jobs.subprocess, "run", lambda *a, **k: result)
    with pytest.raises(slurm_jobs.SlurmError, match=match):
        slurm_jobs.submit(["isambard_sbatch"], tmp_path, {})
