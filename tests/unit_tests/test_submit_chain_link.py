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

"""Submitting one link of a chained run (configs/control_pretraining/submit_chain_link.py).

A link submitted at the wrong moment either wastes a 64-node allocation or silently trains from the
wrong state, so every precondition is tested against real save directories, a real git repository
and the real chain spec. Only SLURM itself (squeue, isambard_sbatch) is stood in for.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.unit_tests.corpora_fixtures import load_campaign_module


submit_chain_link = load_campaign_module("submit_chain_link")
chains = load_campaign_module("generate_epoch_chain")

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAIN_SPEC = REPO_ROOT / "configs" / "control_pretraining" / "30b_trustedmonitor" / "chain.yaml"
CHAIN = chains.load_chain(CHAIN_SPEC)
ARM = next(iter(CHAIN["arms"]))
RENDERED, _ = chains.generate(CHAIN_SPEC)


def _link(link: int) -> dict:
    """One arm's link as the generator renders it from the committed chain spec."""
    path = REPO_ROOT / CHAIN["output_dir"] / chains.link_filename(CHAIN, ARM, link)
    return yaml.safe_load(RENDERED[path])


E = _link(1)["train"]["train_iters"]


def _save(save_dir: Path, iterations: list[int], tracker: int | None) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    for iteration in iterations:
        (save_dir / f"iter_{iteration:07d}").mkdir()
    if tracker is not None:
        (save_dir / submit_chain_link.TRACKER).write_text(f"{tracker}\n")
    return save_dir


# --- where a link starts ------------------------------------------------------------------------


def test_link_1_starts_from_a_directory_that_holds_no_checkpoint(tmp_path):
    submit_chain_link.check_start_state(_link(1), tmp_path / "absent", resume_own_save=False)
    submit_chain_link.check_start_state(_link(1), _save(tmp_path / "empty", [], None), resume_own_save=False)


def test_link_1_refuses_a_directory_holding_someone_elses_checkpoint(tmp_path):
    """A smoke run written into the arm's directory would otherwise become link 1's starting point."""
    save_dir = _save(tmp_path / "arm", [6], 6)
    with pytest.raises(submit_chain_link.NotSafeToSubmit, match="instead of starting from its parent"):
        submit_chain_link.check_start_state(_link(1), save_dir, resume_own_save=False)
    submit_chain_link.check_start_state(_link(1), save_dir, resume_own_save=True)


def test_a_finished_link_1_is_not_run_again(tmp_path):
    save_dir = _save(tmp_path / "arm", [E], E)
    with pytest.raises(submit_chain_link.NotSafeToSubmit, match="has finished"):
        submit_chain_link.check_start_state(_link(1), save_dir, resume_own_save=True)


def test_a_later_link_starts_only_from_exactly_the_previous_links_final_save(tmp_path):
    submit_chain_link.check_start_state(_link(2), _save(tmp_path / "link2", [E], E), resume_own_save=False)
    # Every save the links before it wrote stays in the directory, below the start.
    submit_chain_link.check_start_state(_link(3), _save(tmp_path / "link3", [E, 2 * E], 2 * E), resume_own_save=False)


@pytest.mark.parametrize("link,iterations", [(2, [E, 2 * E]), (3, [E, 2 * E, 3 * E])])
def test_a_later_link_reruns_over_its_own_save_cut_short(tmp_path, link, iterations):
    """A link killed while saving leaves its own final iteration's directory behind, tracker unmoved."""
    save_dir = _save(tmp_path / "arm", iterations, (link - 1) * E)
    submit_chain_link.check_start_state(_link(link), save_dir, resume_own_save=False)


@pytest.mark.parametrize(
    "iterations",
    [
        [E, E + 1],  # a save past the start that no link of this chain writes
        [E, 3 * E],  # a later link's save
    ],
)
def test_a_later_link_refuses_any_save_past_its_start_but_its_own_end(tmp_path, iterations):
    save_dir = _save(tmp_path / "arm", iterations, E)
    with pytest.raises(submit_chain_link.NotSafeToSubmit, match="past the previous link's final save"):
        submit_chain_link.check_start_state(_link(2), save_dir, resume_own_save=False)


@pytest.mark.parametrize(
    "iterations,tracker",
    [
        ([], None),  # the previous link never saved
        ([E, 2 * E], 2 * E),  # this link already saved: rerunning it would rewind the tracker
        ([E], 2 * E),  # the tracker names a save that is not there
        ([], E),  # the tracker names the right iteration but its directory is missing
    ],
)
def test_a_later_link_refuses_any_other_state(tmp_path, iterations, tracker):
    save_dir = _save(tmp_path / "arm", iterations, tracker)
    with pytest.raises(submit_chain_link.NotSafeToSubmit, match="exactly the previous link's final save"):
        submit_chain_link.check_start_state(_link(2), save_dir, resume_own_save=False)


# --- what is submitted ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


@pytest.fixture
def outside_any_git_hook(monkeypatch):
    """Drop the GIT_* variables git exports to the hooks it runs.

    The pre-commit hook runs this suite, and git hands it GIT_INDEX_FILE naming the outer
    repository's index, which every git call below would otherwise read instead of the test's own.
    """
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)


@pytest.mark.usefixtures("outside_any_git_hook")
def test_a_repository_with_an_uncommitted_change_is_refused_and_a_clean_one_names_its_head(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "config.yaml").write_text("a: 1\n")
    _git(repo, "add", "config.yaml")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    (repo / "untracked.txt").write_text("not part of any commit, and not the job's code\n")
    assert submit_chain_link.check_clean_head(repo) == _git(repo, "rev-parse", "HEAD").strip()
    (repo / "config.yaml").write_text("a: 2\n")
    with pytest.raises(submit_chain_link.NotSafeToSubmit, match="uncommitted changes"):
        submit_chain_link.check_clean_head(repo)


def test_only_the_generators_own_output_is_submitted(tmp_path):
    chain = chains.load_chain(CHAIN_SPEC)
    arm = next(iter(chain["arms"]))
    link_path = REPO_ROOT / chain["output_dir"] / chains.link_filename(chain, arm, 1)
    submit_chain_link.check_link_is_generated(CHAIN_SPEC, link_path)
    edited = tmp_path / link_path.name
    edited.write_text(link_path.read_text().replace("seed: 1235", "seed: 1"))
    with pytest.raises(submit_chain_link.NotSafeToSubmit, match="not the generator's output"):
        submit_chain_link.check_link_is_generated(CHAIN_SPEC, edited)


def test_naming_a_snapshot_writes_nothing(tmp_path):
    """A dry run prints the snapshot a submission would read, and leaves nothing on disk."""
    link_path = tmp_path / "link1.yaml"
    link_path.write_text("train: {train_iters: 174}\n")
    snapshot = submit_chain_link.snapshot_path(link_path, tmp_path / "snapshots")
    assert snapshot.parent == tmp_path / "snapshots" and snapshot.stem.startswith("link1-")
    assert not snapshot.parent.exists()
    link_path.write_text("train: {train_iters: 348}\n")
    assert submit_chain_link.snapshot_path(link_path, tmp_path / "snapshots") != snapshot


def test_the_snapshot_is_read_only_holds_the_links_content_and_is_reused(tmp_path):
    link_path = tmp_path / "link1.yaml"
    link_path.write_text("train: {train_iters: 174}\n")
    snapshot = submit_chain_link.snapshot_path(link_path, tmp_path / "snapshots")
    submit_chain_link.write_snapshot(link_path, snapshot)
    assert snapshot.read_text() == link_path.read_text()
    assert not snapshot.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    submit_chain_link.write_snapshot(link_path, snapshot)
    assert [p.name for p in snapshot.parent.iterdir()] == [snapshot.name]


def test_a_snapshot_whose_content_is_not_the_links_is_never_reused(tmp_path):
    """A write cut short (the project quota runs near full) must not become the config a job trains from."""
    link_path = tmp_path / "link1.yaml"
    link_path.write_text("train: {train_iters: 174}\n")
    snapshot = submit_chain_link.snapshot_path(link_path, tmp_path / "snapshots")
    submit_chain_link.write_snapshot(link_path, snapshot)
    snapshot.chmod(stat.S_IRUSR | stat.S_IWUSR)
    snapshot.write_text("train: {train_")
    with pytest.raises(submit_chain_link.NotSafeToSubmit, match="does not hold"):
        submit_chain_link.write_snapshot(link_path, snapshot)


def test_a_link_that_changed_after_its_snapshot_was_named_is_not_written_under_that_name(tmp_path):
    """The name is the content's hash, so content read later than the name must hash to it."""
    link_path = tmp_path / "link1.yaml"
    link_path.write_text("train: {train_iters: 174}\n")
    snapshot = submit_chain_link.snapshot_path(link_path, tmp_path / "snapshots")
    link_path.write_text("train: {train_iters: 348}\n")
    with pytest.raises(submit_chain_link.NotSafeToSubmit, match="changed since"):
        submit_chain_link.write_snapshot(link_path, snapshot)
    assert not snapshot.exists()


def test_the_command_takes_everything_but_the_config_from_the_chain_spec(tmp_path):
    chain = chains.load_chain(CHAIN_SPEC)
    for family in chain["families"].values():
        command = submit_chain_link.submission_command(chain, family, "cp30b-arm-link1", tmp_path / "s.yaml")
        assert command[0] == "isambard_sbatch"
        assert f"--nodes={chain['launch']['nodes']}" in command and f"--time={family['walltime']}" in command
        assert command[command.index("pipeline_training_submit.sbatch") + 1 :] == [
            str(tmp_path / "s.yaml"),
            "nano",
            "pretrain",
            "--disable-ft",
        ]


def test_every_family_has_a_walltime_the_scheduler_reads_as_a_duration():
    """Unquoted, YAML reads h:mm:ss as an integer number of seconds."""
    with open(CHAIN_SPEC) as fh:
        for name, family in yaml.safe_load(fh)["families"].items():
            assert isinstance(family["walltime"], str) and family["walltime"].count(":") == 2, name


# --- SLURM, stood in for: the real squeue and isambard_sbatch would query and submit to the cluster ---


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize(
    "result,match",
    [
        (_completed(1, stderr="slurm_load_jobs error"), "squeue exited 1"),
        (_completed(0, stdout="other-job\ncp30b-arm-link1\n"), "already queued or running"),
    ],
)
def test_a_live_job_of_the_same_name_or_an_unreadable_queue_refuses(monkeypatch, result, match):
    monkeypatch.setattr(submit_chain_link.subprocess, "run", lambda *a, **k: result)
    with pytest.raises(submit_chain_link.NotSafeToSubmit, match=match):
        submit_chain_link.check_no_live_job("cp30b-arm-link1")


def test_a_queue_without_the_links_job_allows_the_submission(monkeypatch):
    monkeypatch.setattr(submit_chain_link.subprocess, "run", lambda *a, **k: _completed(0, stdout="other-job\n"))
    submit_chain_link.check_no_live_job("cp30b-arm-link1")


def test_the_submission_enforces_the_specs_node_cap_and_returns_the_job(monkeypatch, tmp_path):
    seen = {}

    def run(command, **kwargs):
        seen.update(kwargs["env"])
        return _completed(0, stdout="Storage: /projects/a5k 175.6T / 200.0T (87%)\nSubmitted batch job 6935501\n")

    monkeypatch.setattr(submit_chain_link.subprocess, "run", run)
    assert submit_chain_link.submit(["isambard_sbatch"], 256, tmp_path) == "6935501"
    assert (seen["ISAMBARD_SBATCH_FORCE"], seen["ISAMBARD_SBATCH_MAX_NODES"]) == ("0", "256")


def test_a_refused_submission_is_not_safe_to_count_as_submitted(monkeypatch, tmp_path):
    refused = _completed(1, stderr="BLOCKED: this submission would put the account at 266 > 256 nodes")
    monkeypatch.setattr(submit_chain_link.subprocess, "run", lambda *a, **k: refused)
    with pytest.raises(submit_chain_link.NotSafeToSubmit, match="submission failed.*266 > 256"):
        submit_chain_link.submit(["isambard_sbatch"], 256, tmp_path)
