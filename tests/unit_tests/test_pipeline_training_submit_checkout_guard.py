# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A config is refused when it lives in a different git checkout from the code that would train it.

That pairing is how a masked campaign once trained unmasked: its configs sat in a worktree while the launcher ran
code from a checkout that predated the masking, which skipped the settings it did not know. The check is
``scripts/training/checkout_guard.sh``; ``pipeline_training_submit.sbatch`` runs the copy in the config's checkout
and ``pipeline_training_launch.sh`` runs its own. Everything here runs the real scripts under bash against throwaway
git checkouts, including this cluster's layout: a main checkout whose git config says ``core.bare=true`` while it
still holds working files, with the worktrees inside it. ``isambard_sbatch`` and ``scancel`` are SLURM commands, so
they are stubs on PATH, and the launcher the submit script hands over to is a stub that records its arguments,
because the real one starts a multi-node job.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / "scripts" / "training" / "checkout_guard.sh"
SUBMIT_SCRIPT = REPO_ROOT / "pipeline_training_submit.sbatch"
LAUNCHER = REPO_ROOT / "pipeline_training_launch.sh"
REFUSAL = "belongs to the checkout"
GIT_IDENTITY = ["-c", "user.name=test", "-c", "user.email=test@example.com"]


def _git(*args: str) -> None:
    subprocess.run(["git", *GIT_IDENTITY, *args], check=True, capture_output=True)


def _checkout(path: Path) -> Path:
    path.mkdir(parents=True)
    _git("init", "-q", str(path))
    return path


def _config(directory: Path, name: str = "run.yaml") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / name
    config.write_text("train: {}\n")
    return config


def _install_guard(checkout: Path) -> None:
    (checkout / "scripts" / "training").mkdir(parents=True, exist_ok=True)
    shutil.copy(GUARD, checkout / "scripts" / "training" / "checkout_guard.sh")


@pytest.fixture
def bare_main_with_worktree(tmp_path):
    """The cluster's layout: ``main`` has a commit and a worktree ``wt`` inside it, then ``core.bare=true``."""
    main = _checkout(tmp_path / "main")
    _git("-C", str(main), "commit", "-q", "--allow-empty", "-m", "start")
    worktree = main / ".claude" / "worktrees" / "wt"
    _git("-C", str(main), "worktree", "add", "-q", str(worktree), "-b", "wt")
    _git("-C", str(main), "config", "core.bare", "true")
    return main, worktree


def _guard(config, repo_dir, cwd=None, **extra_env) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(GUARD), str(config), str(repo_dir)],
        cwd=cwd,
        env={**os.environ, **extra_env},
        capture_output=True,
        text=True,
    )


class TestGuard:
    def test_a_config_in_the_code_checkout_passes(self, tmp_path):
        code = _checkout(tmp_path / "code")
        assert _guard(_config(code / "configs"), code).returncode == 0

    def test_a_config_in_another_checkout_is_refused(self, tmp_path):
        code, other = _checkout(tmp_path / "code"), _checkout(tmp_path / "other")
        result = _guard(_config(other), code)
        assert result.returncode == 1
        assert REFUSAL in result.stderr and str(other) in result.stderr and str(code) in result.stderr

    def test_the_refusal_can_be_waived_deliberately(self, tmp_path):
        code, other = _checkout(tmp_path / "code"), _checkout(tmp_path / "other")
        assert _guard(_config(other), code, ALLOW_CROSS_CHECKOUT_CONFIG="1").returncode == 0

    def test_a_config_outside_any_checkout_is_not_checked(self, tmp_path):
        code = _checkout(tmp_path / "code")
        assert _guard(_config(tmp_path / "outside"), code).returncode == 0

    def test_code_outside_any_checkout_is_not_checked(self, tmp_path):
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        assert _guard(_config(_checkout(tmp_path / "other")), snapshot).returncode == 0

    def test_a_bare_repository_holds_no_code_and_is_not_checked(self, tmp_path):
        bare = tmp_path / "bare.git"
        _git("init", "-q", "--bare", str(bare))
        assert _guard(_config(_checkout(tmp_path / "other")), bare).returncode == 0

    @pytest.mark.parametrize(
        "config_in, code_in, refused",
        [("wt", "main", True), ("main", "wt", True), ("main", "main", False), ("wt", "wt", False)],
    )
    def test_the_bare_flagged_main_checkout_is_a_checkout(self, bare_main_with_worktree, config_in, code_in, refused):
        main, worktree = bare_main_with_worktree
        where = {"main": main, "wt": worktree}
        result = _guard(_config(where[config_in] / "configs"), where[code_in])
        assert result.returncode == (1 if refused else 0), result.stderr
        assert (REFUSAL in result.stderr) is refused

    def test_a_relative_config_is_read_against_the_code_checkout(self, tmp_path):
        """Training reads ``configs/run.yaml`` in REPO_DIR, whatever directory the launch started in."""
        code, other = _checkout(tmp_path / "code"), _checkout(tmp_path / "other")
        _config(code / "configs")
        _config(other / "configs")
        assert _guard("configs/run.yaml", code, cwd=other).returncode == 0

    def test_a_missing_config_is_left_to_the_launcher(self, tmp_path):
        code = _checkout(tmp_path / "code")
        assert _guard(code / "missing.yaml", code).returncode == 0

    @staticmethod
    def _failing_git(tmp_path, message: str) -> str:
        """A PATH whose git fails the way git does on a repository it will not read."""
        stub = tmp_path / "failing_git"
        stub.mkdir()
        (stub / "git").write_text(f'#!/bin/bash\necho "{message}" >&2\nexit 128\n')
        (stub / "git").chmod(0o755)
        return f"{stub}:{os.environ['PATH']}"

    @pytest.mark.parametrize(
        "message",
        [
            "fatal: detected dubious ownership in repository at '/somewhere'",
            "fatal: bad config line 1 in file .git/config",
        ],
        ids=["dubious-ownership", "corrupt"],
    )
    def test_a_git_failure_that_is_not_no_repository_refuses(self, tmp_path, message):
        """A check that cannot see a checkout must not pass it."""
        code = _checkout(tmp_path / "code")
        result = _guard(_config(code / "configs"), code, PATH=self._failing_git(tmp_path, message))
        assert result.returncode == 1
        assert message.split(": ", 1)[1] in result.stderr and "ALLOW_CROSS_CHECKOUT_CONFIG=1" in result.stderr

    def test_git_missing_refuses(self, tmp_path):
        code = _checkout(tmp_path / "code")
        tools = tmp_path / "tools_without_git"
        tools.mkdir()
        for tool in ("realpath", "dirname", "basename", "mktemp", "rm", "cat"):
            (tools / tool).symlink_to(shutil.which(tool))
        result = subprocess.run(
            [shutil.which("bash"), str(GUARD), str(_config(code / "configs")), str(code)],
            env={"PATH": str(tools)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1 and "needs git" in result.stderr

    def test_the_waiver_needs_no_git(self, tmp_path):
        code = _checkout(tmp_path / "code")
        path = self._failing_git(tmp_path, "fatal: detected dubious ownership in repository at '/somewhere'")
        assert _guard(_config(code / "configs"), code, PATH=path, ALLOW_CROSS_CHECKOUT_CONFIG="1").returncode == 0

    def test_a_usage_error_is_its_own_exit_status(self):
        assert subprocess.run(["bash", str(GUARD)], capture_output=True).returncode == 2


@pytest.fixture
def submission(tmp_path):
    """Stub SLURM commands, and a submit that records what the stub launcher in REPO_DIR was handed."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("isambard_sbatch", "scancel"):
        (bin_dir / name).write_text("#!/bin/bash\nexit 0\n")
        (bin_dir / name).chmod(0o755)
    launched = tmp_path / "launched"

    def submit(config, repo_dir: Path, cwd: Path | None = None, **extra_env) -> tuple[int, str, str | None]:
        launcher = repo_dir / "pipeline_training_launch.sh"
        launcher.write_text(f'#!/bin/bash\necho "$@" > {launched}\n')
        launcher.chmod(0o755)
        launched.unlink(missing_ok=True)
        env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path), "SLURM_JOB_ID": "1", **extra_env}
        if "SLURM_SUBMIT_DIR" not in extra_env:  # a submission names REPO_DIR one way or the other
            env["GEODESIC_REPO_DIR"] = str(repo_dir)
        result = subprocess.run(
            ["bash", str(SUBMIT_SCRIPT), str(config), "nano", "cpt"],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stderr, launched.read_text().strip() if launched.exists() else None

    return submit


class TestSubmitScript:
    def test_it_runs_the_guard_in_the_configs_checkout(self, tmp_path, submission):
        code, other = _checkout(tmp_path / "code"), _checkout(tmp_path / "other")
        _install_guard(other)
        returncode, stderr, launched = submission(_config(other / "configs"), code)
        assert returncode == 1 and REFUSAL in stderr and launched is None

    def test_a_config_whose_checkout_has_no_guard_is_left_to_the_launcher(self, tmp_path, submission):
        code, other = _checkout(tmp_path / "code"), _checkout(tmp_path / "other")
        returncode, stderr, launched = submission(_config(other / "configs"), code)
        assert returncode == 0, stderr
        assert launched.startswith(str(other / "configs" / "run.yaml"))

    def test_the_incident_layout_is_refused(self, bare_main_with_worktree, submission):
        """Submitted from the stale main checkout (REPO_DIR from SLURM_SUBMIT_DIR) with a worktree's config."""
        main, worktree = bare_main_with_worktree
        _install_guard(worktree)
        config = _config(worktree / "configs")
        returncode, stderr, launched = submission(config, main, cwd=main, SLURM_SUBMIT_DIR=str(main))
        assert returncode == 1 and REFUSAL in stderr and launched is None
        returncode, stderr, launched = submission(config, worktree, cwd=main)
        assert returncode == 0, stderr
        assert launched.startswith(str(config))

    def test_a_relative_config_is_handed_on_as_the_code_checkouts_file(self, tmp_path, submission):
        code, other = _checkout(tmp_path / "code"), _checkout(tmp_path / "other")
        _install_guard(code)
        _config(code / "configs")
        _config(other / "configs")
        returncode, stderr, launched = submission("configs/run.yaml", code, cwd=other)
        assert returncode == 0, stderr
        assert launched.split()[0] == str(code / "configs" / "run.yaml")


class TestLauncher:
    """The real launcher checks right after resolving REPO_DIR, before anything else needs the cluster."""

    def _launch(self, config: Path, repo_dir: Path) -> subprocess.CompletedProcess:
        env = {"PATH": os.environ["PATH"], "HOME": str(repo_dir.parent), "SLURM_JOB_ID": "1"}
        return subprocess.run(
            ["bash", str(LAUNCHER), str(config), "--model", "nano", "--mode", "pretrain"],
            env={**env, "GEODESIC_REPO_DIR": str(repo_dir)},
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_a_config_from_another_checkout_is_refused(self, tmp_path):
        code, other = _checkout(tmp_path / "code"), _checkout(tmp_path / "other")
        result = self._launch(_config(other), code)
        assert result.returncode == 1 and REFUSAL in result.stderr

    def test_a_config_from_the_code_checkout_passes_the_check(self, tmp_path):
        """The launch then stops at the next step, which needs the real repository's environment config."""
        code = _checkout(tmp_path / "code")
        result = self._launch(_config(code), code)
        assert REFUSAL not in result.stderr
        assert "pipeline_env_config.env not found" in result.stderr
