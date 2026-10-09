"""Unit tests for the git-environment cleanup in tests/unit_tests/conftest.py.

The unit suite runs inside git's pre-commit hook, where git exports GIT_DIR, GIT_INDEX_FILE and the other variables
that select a repository. A test that runs git on a throwaway repository would otherwise act on the repository being
committed: ``git init <path>`` re-initialises $GIT_DIR instead of creating ``<path>``.
"""

from __future__ import annotations

import os
import subprocess

from tests.unit_tests.conftest import drop_inherited_git_repository_variables


def _repository_variables() -> list[str]:
    return subprocess.run(
        ["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True, check=True
    ).stdout.split()


def test_every_repository_variable_git_lists_is_dropped_and_nothing_else():
    environ = {name: "/elsewhere" for name in _repository_variables()} | {"PATH": "/usr/bin", "GIT_AUTHOR_NAME": "a"}
    drop_inherited_git_repository_variables(environ)
    assert environ == {"PATH": "/usr/bin", "GIT_AUTHOR_NAME": "a"}


def test_git_init_creates_the_named_repository_once_an_inherited_git_dir_is_dropped(tmp_path):
    inherited = tmp_path / "inherited"
    subprocess.run(["git", "init", "-q", str(inherited)], check=True)
    environ = {**os.environ, "GIT_DIR": str(inherited / ".git")}
    drop_inherited_git_repository_variables(environ)
    target = tmp_path / "target"
    subprocess.run(["git", "init", "-q", str(target)], check=True, env=environ)
    assert (target / ".git").is_dir()


def test_the_suite_runs_without_inherited_repository_variables():
    assert not set(_repository_variables()) & set(os.environ)
