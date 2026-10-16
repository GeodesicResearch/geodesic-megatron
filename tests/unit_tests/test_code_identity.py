# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The launch-time check that a config is trained with the code it names (``scripts/training/code_identity.py``).

Every case runs on real git repositories: a history of three commits (the ancestor a block requires, the commit it
names, and a commit on another branch), and a frozen copy of the named commit made as the repository's convention
makes one (``git archive`` plus a ``REVISION`` file). The launcher's own ``check_code_identity`` function is run out of
``pipeline_training_launch.sh``; only the container shim it calls through is stood in for, by a script that runs the
command where the test already runs, since a unit test cannot start the container.
"""

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from scripts.training import code_identity
from scripts.training.launcher_source import launcher_function


REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_IDENTITY = ["-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]
LAUNCHER = "launch.sh"


def git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` (no user configuration needed) and return its output."""
    return subprocess.run(
        ["git", *GIT_IDENTITY, "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, content in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(content)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path) -> dict:
    """A repository whose main branch has the ancestor and then the named commit, and a side branch from the root."""
    repo = tmp_path / "history"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    root = commit(repo, {".gitignore": "__pycache__/\n", "src/pkg/a.py": "A = 1\n", LAUNCHER: "echo 1\n"}, "root")
    ancestor = commit(repo, {"src/pkg/a.py": "A = 2\n"}, "the fix every run needs")
    named = commit(repo, {"src/pkg/b.py": "B = 1\n", LAUNCHER: "echo 2\n"}, "the code the config names")
    git(repo, "checkout", "-q", "-b", "side", root)
    side = commit(repo, {"src/pkg/c.py": "C = 1\n"}, "another line of work")
    git(repo, "checkout", "-q", "main")
    return {"repo": repo, "ancestor": ancestor, "named": named, "side": side}


def frozen_copy(history: dict, destination: Path) -> Path:
    """A frozen copy of the named commit: its files by ``git archive``, its commit in ``REVISION``."""
    destination.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(history["repo"]), "archive", history["named"]], check=True, capture_output=True
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(destination)], input=archive, check=True)
    (destination / "REVISION").write_text(history["named"] + "\n")
    return destination


def block(commits: dict, **changes) -> dict:
    """The ``code_identity:`` block naming the ``history`` fixture's named commit, read with git, with ``changes``
    applied."""
    repo, named = commits["repo"], commits["named"]
    stated = {
        "revision": named,
        "src_tree": git(repo, "rev-parse", f"{named}:src"),
        "launchers": {LAUNCHER: git(repo, "rev-parse", f"{named}:{LAUNCHER}")},
        "ancestor": commits["ancestor"],
        "history": str(repo),
    }
    return {**stated, **changes}


def config(directory: Path, stated: dict | None) -> Path:
    path = directory / "stage.yaml"
    content = {"train": {"train_iters": 10}}
    if stated is not None:
        content["code_identity"] = stated
    path.write_text(yaml.safe_dump(content))
    return path


def check(directory: Path, stated: dict, repo_dir: Path) -> dict:
    """The record the check makes for a config holding ``stated``, run against ``repo_dir``."""
    identity = code_identity.config_code_identity(str(config(directory, stated)))
    return code_identity.code_identity_record(
        identity, "stage.yaml", code_identity.measure_code_identity(identity, repo_dir)
    )


class TestMeasurement:
    def test_a_frozen_copy_of_the_named_commit_is_its_code(self, history, tmp_path):
        record = check(tmp_path, block(history), frozen_copy(history, tmp_path / "copy"))
        assert record["passed"] and record["differences"] == []
        assert record["measured"]["commit"] == history["named"]
        assert record["measured"]["src_tree"] == git(history["repo"], "rev-parse", f"{history['named']}:src")

    def test_a_checkout_is_judged_by_its_head(self, history, tmp_path):
        assert check(tmp_path, block(history), history["repo"])["passed"]

    def test_bytecode_written_by_running_the_code_is_left_out_as_git_leaves_it_out(self, history, tmp_path):
        copy = frozen_copy(history, tmp_path / "copy")
        (copy / "src" / "pkg" / "__pycache__").mkdir()
        (copy / "src" / "pkg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0compiled")
        assert check(tmp_path, block(history), copy)["passed"]

    @pytest.mark.parametrize(
        ("damage", "difference"),
        [
            (lambda copy: (copy / "src" / "pkg" / "a.py").write_text("A = 3\n"), "src/ is tree"),
            (lambda copy: (copy / "src" / "pkg" / "new.py").write_text("N = 1\n"), "src/ is tree"),
            (lambda copy: (copy / "src" / "pkg" / "b.py").unlink(), "src/ is tree"),
            (lambda copy: (copy / LAUNCHER).write_text("echo 3\n"), f"{LAUNCHER} is blob"),
        ],
        ids=["changed-source", "added-source", "removed-source", "changed-launcher"],
    )
    def test_code_that_differs_from_the_named_code_is_refused(self, history, tmp_path, damage, difference):
        copy = frozen_copy(history, tmp_path / "copy")
        damage(copy)
        record = check(tmp_path, block(history), copy)
        assert not record["passed"]
        assert [line for line in record["differences"] if line.startswith(difference)]

    def test_the_launching_users_git_ignore_file_hides_nothing(self, history, tmp_path, monkeypatch):
        """Git reads a user's default ignore file even without their configuration; a file it names is still code."""
        home = tmp_path / "home"
        (home / ".config" / "git").mkdir(parents=True)
        (home / ".config" / "git" / "ignore").write_text("extra.py\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        copy = frozen_copy(history, tmp_path / "copy")
        (copy / "src" / "pkg" / "extra.py").write_text("EXTRA = 1\n")
        record = check(tmp_path, block(history), copy)
        assert not record["passed"]
        assert [line for line in record["differences"] if line.startswith("src/ is tree")]

    def test_code_whose_history_lacks_the_ancestor_is_refused(self, history, tmp_path):
        record = check(tmp_path, block(history, ancestor=history["side"]), frozen_copy(history, tmp_path / "copy"))
        assert record["differences"] == [
            f"commit {history['named']} does not descend from {history['side']} in {history['repo']}"
        ]

    def test_a_commit_the_history_does_not_hold_is_refused_rather_than_skipped(self, history, tmp_path):
        copy = frozen_copy(history, tmp_path / "copy")
        (copy / "REVISION").write_text("f" * 40 + "\n")
        with pytest.raises(code_identity.CodeIdentityError, match="cannot tell whether"):
            check(tmp_path, block(history), copy)

    def test_a_missing_launcher_is_refused(self, history, tmp_path):
        copy = frozen_copy(history, tmp_path / "copy")
        (copy / LAUNCHER).unlink()
        with pytest.raises(code_identity.CodeIdentityError, match=r"holds no \['launch.sh'\]"):
            check(tmp_path, block(history), copy)

    def test_a_copy_whose_commit_cannot_be_read_is_refused(self, history, tmp_path):
        copy = frozen_copy(history, tmp_path / "copy")
        (copy / "REVISION").write_text("")
        with pytest.raises(code_identity.CodeIdentityError, match="cannot be read"):
            check(tmp_path, block(history), copy)


class TestBlock:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"revision": "abc123"}, "revision must be a full 40-character"),
            ({"launchers": {}}, "launchers must map"),
            ({"launchers": {"/etc/launch.sh": "a" * 40}}, "is not a path inside the repository"),
            ({"launchers": {"../launch.sh": "a" * 40}}, "is not a path inside the repository"),
            ({"launchers": {LAUNCHER: "short"}}, "launchers.launch.sh must be a full"),
            ({"history": "relative/repo"}, "history must be the absolute path"),
            ({"extra": 1}, r"unknown keys \['extra'\]"),
        ],
    )
    def test_a_malformed_block_is_refused(self, history, tmp_path, changes, message):
        with pytest.raises(code_identity.CodeIdentityError, match=message):
            code_identity.config_code_identity(str(config(tmp_path, block(history, **changes))))

    def test_every_field_is_required(self, history, tmp_path):
        stated = block(history)
        del stated["ancestor"]
        with pytest.raises(code_identity.CodeIdentityError, match=r"missing keys \['ancestor'\]"):
            code_identity.config_code_identity(str(config(tmp_path, stated)))

    def test_a_block_in_a_base_config_is_the_overlays_too(self, history, tmp_path):
        base = config(tmp_path, block(history))
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(f"base_config: {base}\ntrain:\n  train_iters: 20\n")
        assert code_identity.config_code_identity(str(overlay)).revision == history["named"]

    def test_a_config_without_a_block_names_no_code(self, tmp_path):
        assert code_identity.config_code_identity(str(config(tmp_path, None))) is None


class TestCommandLine:
    def run(self, config_file: Path, repo_dir: Path, capsys) -> tuple[int, str, str]:
        status = code_identity.main(["--config", str(config_file), "--repo-dir", str(repo_dir)])
        captured = capsys.readouterr()
        return status, captured.out, captured.err

    def test_a_pass_prints_the_record_as_one_line(self, history, tmp_path, capsys):
        status, out, err = self.run(config(tmp_path, block(history)), frozen_copy(history, tmp_path / "copy"), capsys)
        assert status == 0 and err == ""
        assert len(out.splitlines()) == 1 and json.loads(out)["passed"] is True

    def test_a_difference_exits_1_naming_it(self, history, tmp_path, capsys):
        copy = frozen_copy(history, tmp_path / "copy")
        (copy / LAUNCHER).write_text("echo 3\n")
        status, out, err = self.run(config(tmp_path, block(history)), copy, capsys)
        assert status == 1 and json.loads(out)["passed"] is False
        assert f"FATAL [code-identity]: {LAUNCHER} is blob" in err

    def test_a_failure_to_measure_exits_1_with_no_record(self, history, tmp_path, capsys):
        copy = frozen_copy(history, tmp_path / "copy")
        (copy / LAUNCHER).unlink()
        status, out, err = self.run(config(tmp_path, block(history)), copy, capsys)
        assert (status, out) == (1, "") and "FATAL [code-identity]:" in err

    def test_a_config_naming_no_code_exits_0_with_no_record(self, tmp_path, capsys):
        status, out, err = self.run(config(tmp_path, None), tmp_path, capsys)
        assert (status, out) == (0, "") and "carries no code_identity: block" in err


class TestRequireCheckedCodeIdentity:
    def identity(self, history, **changes) -> code_identity.CodeIdentity:
        return code_identity.parse_code_identity(block(history, **changes), "code_identity")

    def record(self, history, passed: bool = True) -> str:
        return json.dumps({"expected": asdict(self.identity(history)), "passed": passed})

    def test_a_passing_record_for_the_block_is_returned(self, history):
        environ = {code_identity.RECORD_ENV: self.record(history)}
        assert code_identity.require_checked_code_identity(self.identity(history), "stage.yaml", environ)["passed"]

    @pytest.mark.parametrize(
        ("environ", "message"),
        [
            ({}, "launch it through pipeline_training_launch.sh"),
            ("other-block", "records the check of another code_identity"),
            ("failed", "did not pass"),
        ],
    )
    def test_a_run_without_a_passing_record_for_its_block_is_refused(self, history, environ, message):
        if environ == "other-block":
            other = asdict(self.identity(history, ancestor=history["side"]))
            environ = {code_identity.RECORD_ENV: json.dumps({"expected": other, "passed": True})}
        elif environ == "failed":
            environ = {code_identity.RECORD_ENV: self.record(history, passed=False)}
        with pytest.raises(code_identity.CodeIdentityError, match=message):
            code_identity.require_checked_code_identity(self.identity(history), "stage.yaml", environ)


class TestLauncher:
    """``check_code_identity`` as the launcher runs it, with the container shim stood in for."""

    def launch(self, config_file: Path, repo_dir: Path, inherited: str | None) -> subprocess.CompletedProcess:
        # The container cannot be started from a unit test, which already runs inside it, so the shim is a script
        # that runs its command here, with this repository's scripts importable from any directory.
        shim = repo_dir / "pipeline_env_exec.sh"
        shim.write_text('#!/bin/bash\nexec bash -c "$1"\n')
        shim.chmod(0o755)
        script = (
            "set -euo pipefail\n"
            f"{launcher_function('check_code_identity')}\n"
            'check_code_identity "$1" "$2" || { echo "REFUSED"; exit 1; }\n'
            'echo "RECORD=${ISAMBARD_CODE_IDENTITY-unset}"\n'
        )
        env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(REPO_ROOT), "HOME": os.environ["HOME"]}
        if inherited is not None:
            env[code_identity.RECORD_ENV] = inherited
        return subprocess.run(
            ["bash", "-c", script, "harness", str(config_file), str(repo_dir)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

    def test_a_passing_check_hands_its_record_to_the_ranks(self, history, tmp_path):
        result = self.launch(config(tmp_path, block(history)), frozen_copy(history, tmp_path / "copy"), None)
        assert result.returncode == 0, result.stderr
        record = json.loads(result.stdout.split("RECORD=", 1)[1])
        assert record["passed"] is True and record["measured"]["commit"] == history["named"]
        assert "[code-identity] {" in result.stdout

    def test_a_difference_ends_the_launch(self, history, tmp_path):
        copy = frozen_copy(history, tmp_path / "copy")
        (copy / "src" / "pkg" / "a.py").write_text("A = 3\n")
        result = self.launch(config(tmp_path, block(history)), copy, None)
        assert result.returncode == 1 and "REFUSED" in result.stdout
        assert "FATAL [code-identity]: src/ is tree" in result.stderr

    def test_a_config_naming_no_code_drops_an_inherited_record(self, tmp_path):
        repo_dir = tmp_path / "copy"
        repo_dir.mkdir()
        result = self.launch(config(tmp_path, None), repo_dir, inherited='{"passed": true}')
        assert result.returncode == 0, result.stderr
        assert "RECORD=unset" in result.stdout
