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

"""Unit tests for scripts/training/probe_job.sh, the steps the production-width probe jobs are built from.

Each test sources the real library in bash against a scratch repo. Two boundaries are stood in for, because a unit
test can run neither: the container (``pipeline_env_exec.sh`` runs its command in a plain shell) and the training
launcher (``pipeline_training_launch.sh`` prints its arguments and exits with a chosen status).
"""

import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_JOB = _REPO_ROOT / "scripts" / "training" / "probe_job.sh"
PROBE_SBATCHES = sorted((_REPO_ROOT / "configs" / "control_pretraining").rglob("probe*.sbatch"))


def executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def repo(tmp_path) -> Path:
    """A scratch repo holding the library's own dependencies: REVISION, launch_environment.py, and the two boundary
    stand-ins."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "training").mkdir(parents=True)
    (repo / "REVISION").write_text("abc1234\n")
    shutil.copy(_REPO_ROOT / "scripts" / "training" / "launch_environment.py", repo / "scripts" / "training")
    # The container cannot start in a unit test: the command string runs in a plain shell instead.
    executable(repo / "pipeline_env_exec.sh", '#!/bin/bash\nbash -c "$1"\n')
    (repo / "pipeline_env_activate.sh").write_text("")
    # A training launch needs GPUs: the stand-in prints its arguments and the two variables launch() exports, then
    # exits with LAUNCH_STATUS.
    executable(
        repo / "pipeline_training_launch.sh",
        '#!/bin/bash\necho "args: $*"\necho "overrides: ${ISAMBARD_ENV_OVERRIDES:-none}"\n'
        'echo "port: $MASTER_PORT_OVERRIDE"\nexit "${LAUNCH_STATUS:-0}"\n',
    )
    return repo


def run(repo: Path, body: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Source the library with the variables a probe sbatch sets, then run ``body``."""
    out = repo.parent / "out"
    script = (
        f"set -uo pipefail\nREPO_DIR={repo}\nOUT={out}\nNODES=2\nGPUS=8\nLINKS_PER_GPU=18\n"
        f"HF_MODEL=org/model\nMODEL=nano\nMODE=pretrain\nNODELIST='nid[1-2]'\nsource {PROBE_JOB}\n{body}\n"
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "SLURM_JOB_ID": "7", **(env or {})},
        timeout=60,
    )


def steps(repo: Path) -> list[list[str]]:
    return [line.split("\t") for line in (repo.parent / "out" / "steps.tsv").read_text().splitlines()]


def test_only_a_failed_gating_step_fails_the_job(repo):
    result = run(repo, "probe_open_records\nnote a 'ran a' 3\nrecord b 'ran b' 0\nprobe_finish")
    assert result.returncode == 0, result.stderr
    assert steps(repo) == [["step", "ran", "exit", "gates"], ["a", "ran a", "3", "no"], ["b", "ran b", "0", "yes"]]
    result = run(repo, "probe_open_records\nrecord c 'ran c' 2\nrecord d 'ran d' 0\nprobe_finish")
    assert result.returncode == 1
    assert "[probe] job 7, code abc1234" in result.stdout


def test_the_start_check_refuses_code_that_is_not_a_frozen_copy(repo):
    (repo / "REVISION").unlink()
    result = run(repo, f"probe_check_start {repo.parent}/scratch/fast\necho started")
    assert result.returncode == 1 and "has no REVISION file" in result.stderr


def test_the_start_check_refuses_a_scratch_directory_left_by_an_earlier_probe(repo):
    (repo.parent / "scratch").mkdir()
    result = run(repo, f"probe_check_start {repo.parent}/scratch/fast\necho started")
    assert result.returncode == 1 and "a probe starts from an empty scratch directory" in result.stderr


def test_the_start_check_refuses_an_inherited_launch_setting(repo):
    result = run(repo, f"probe_check_start {repo.parent}/scratch/fast\necho started", {"ISAMBARD_OMP_THREADS": "1"})
    assert result.returncode == 1 and "inherited from the submitting shell: ISAMBARD_OMP_THREADS" in result.stderr
    result = run(repo, f"probe_check_start {repo.parent}/scratch/fast\necho started")
    assert result.returncode == 0 and "started" in result.stdout


@pytest.mark.parametrize("status", [0, 3])
def test_a_launch_records_its_status_and_passes_its_settings(repo, status):
    result = run(
        repo,
        'probe_open_records\nlaunch fast probe.yaml arm/stage.env 29611 60\necho "returned $?"\nprobe_finish',
        {"LAUNCH_STATUS": str(status)},
    )
    assert f"returned {status}" in result.stdout
    assert result.returncode == (status != 0)
    log = (repo.parent / "out" / "fast.out").read_text()
    assert "args: probe.yaml --model nano --mode pretrain --disable-ft --nodes 2 --nodelist nid[1-2]" in log
    assert f"overrides: {repo}/arm/stage.env" in log and "port: 29611" in log
    assert steps(repo)[1] == ["fast", "probe.yaml with arm/stage.env, limit 60s", str(status), "yes"]


def test_a_launch_without_a_settings_file_exports_none(repo):
    run(repo, "probe_open_records\nlaunch as_is probe.yaml '' 29613 60")
    assert "overrides: none" in (repo.parent / "out" / "as_is.out").read_text()


def test_a_launch_past_its_time_limit_is_ended_and_fails(repo):
    executable(repo / "pipeline_training_launch.sh", "#!/bin/bash\nsleep 30\n")
    result = run(repo, "probe_open_records\nlaunch hung probe.yaml '' 29611 1\nprobe_finish")
    assert result.returncode == 1
    assert steps(repo)[1][2] == "124", "timeout's own status"


@pytest.mark.parametrize("status", [0, 2])
def test_a_gate_keeps_its_output_and_decides_the_job(repo, status):
    result = run(repo, f"probe_open_records\ngate g 'checks x' 'echo verdict; exit {status}'\necho \"returned $?\"")
    assert f"returned {status}" in result.stdout
    assert (repo.parent / "out" / "g.out").read_text() == "verdict\n"
    assert steps(repo)[1] == ["g", "checks x", str(status), "yes"]


# A per-rank line with the parentheses a regular expression would read as a group: matched as a fixed string.
PATCH_LINE = "Installed fp32-SSM-state patch (mode=checkpoint)"


def handoff_log(repo: Path, loaded: bool, patched_ranks: int, exited: bool) -> None:
    lines = ["iteration 1"]
    if loaded:
        lines.append("successfully loaded checkpoint from /scratch/fast [ t 1/1, p 1/1 ] at iteration 500")
    lines += [f"[rank{rank}] {PATCH_LINE}" for rank in range(patched_ranks)]
    if exited:
        lines.append("exiting program at iteration 5")
    (repo.parent / "out").mkdir(exist_ok=True)
    (repo.parent / "out" / "handoff.out").write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    "loaded, patched_ranks, exited, status",
    [(True, 8, True, "0"), (False, 8, True, "1"), (True, 7, True, "1"), (True, 8, False, "1")],
)
def test_a_handoff_passes_only_when_it_loaded_patched_every_rank_and_exited(
    repo, loaded, patched_ranks, exited, status
):
    handoff_log(repo, loaded, patched_ranks, exited)
    run(repo, f"probe_open_records\ncheck_handoff handoff /scratch/fast 5 {shlex.quote(PATCH_LINE)}")
    assert steps(repo)[1][0] == "handoff_evidence" and steps(repo)[1][2] == status
    evidence = (repo.parent / "out" / "handoff.evidence.tsv").read_text()
    assert f"ranks_logging_per_rank_line\t{patched_ranks}\n" in evidence


def test_a_parity_band_names_its_window_references_and_candidate(repo):
    # The stand-in container prints the command string instead of running it.
    executable(repo / "pipeline_env_exec.sh", '#!/bin/bash\necho "$1"\n')
    result = run(repo, "parity_band band '51 500' 100 c.out a.out b.out")
    assert "loss_parity.py band --reference a.out b.out --candidate c.out" in result.stdout
    assert re.search(r"--iterations 51 500 --window 100 --wandb --json > \S+/out/band\.json", result.stdout)


def test_the_probe_scripts_take_every_shared_step_from_the_library():
    """A probe defines no step of its own that the library provides, so a fix to one reaches every probe."""
    provided = set(re.findall(r"^(\w+)\(\) \{", PROBE_JOB.read_text(), re.M))
    assert {"launch", "score", "record", "note", "gate", "check_handoff", "probe_select_nodes"} <= provided
    assert PROBE_SBATCHES
    for sbatch in PROBE_SBATCHES:
        text = sbatch.read_text()
        assert 'source "$REPO_DIR/scripts/training/probe_job.sh"' in text, sbatch.name
        assert not provided & set(re.findall(r"^(\w+)\(\) \{", text, re.M)), sbatch.name
