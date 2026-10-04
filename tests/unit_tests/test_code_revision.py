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

"""Unit tests for scripts/telemetry/code_revision.py (the commit a checkout is at, read without running git), on
real on-disk repositories: hand-written .git layouts, and repositories and linked worktrees made by the git binary."""

import os
import subprocess

import pytest
from scripts.telemetry.code_revision import code_revision


def test_resolves_direct_ref(tmp_path):
    # Real on-disk .git layout (loose ref), no git binary involved by design.
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/feature\n")
    (git / "refs" / "heads" / "feature").write_text("abc123def456\n")
    out = code_revision(str(tmp_path))
    assert out.startswith("abc123def456")
    assert "refs/heads/feature" in out


def test_resolves_packed_ref(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text("# pack-refs\ncafe0123 refs/heads/main\n")
    assert code_revision(str(tmp_path)).startswith("cafe0123")


def test_unresolved_is_loud(tmp_path):
    assert code_revision(str(tmp_path / "nogit")).startswith("UNRESOLVED")


def test_a_missing_branch_is_loud(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/gone\n")
    (git / "packed-refs").write_text("cafe0123 refs/heads/main\ncafe0456 refs/heads/feature/gone\n")
    assert code_revision(str(tmp_path)).startswith("UNRESOLVED (refs/heads/gone is neither")


def _git(cwd, *args) -> str:
    """Run the real git binary with no user or system configuration."""
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(cwd),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def linked_worktree(tmp_path):
    """A real repository with one commit on `main`, and a linked worktree of it on `feature`."""
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    _git(main, "commit", "-q", "--allow-empty", "-m", "first")
    _git(main, "worktree", "add", "-q", "-b", "feature", str(tmp_path / "wt"))
    _git(tmp_path / "wt", "commit", "-q", "--allow-empty", "-m", "on feature")
    return main, tmp_path / "wt"


def test_resolves_a_linked_worktree(linked_worktree):
    main, worktree = linked_worktree
    assert (worktree / ".git").is_file()
    expected = _git(worktree, "rev-parse", "HEAD")
    assert code_revision(str(worktree)) == f"{expected} (refs/heads/feature)"
    # Branches packed into the main repository's packed-refs resolve the same way.
    _git(main, "pack-refs", "--all")
    assert not (main / ".git" / "refs" / "heads" / "feature").exists()
    assert code_revision(str(worktree)) == f"{expected} (refs/heads/feature)"
    assert code_revision(str(main)) == f"{_git(main, 'rev-parse', 'HEAD')} (refs/heads/main)"


def test_resolves_a_detached_worktree(linked_worktree, tmp_path):
    main, _ = linked_worktree
    _git(main, "worktree", "add", "-q", "--detach", str(tmp_path / "detached"), "main")
    assert code_revision(str(tmp_path / "detached")) == _git(main, "rev-parse", "main")


def test_a_git_file_without_a_gitdir_pointer_is_loud(tmp_path):
    (tmp_path / ".git").write_text("not a pointer\n")
    assert code_revision(str(tmp_path)).startswith("UNRESOLVED (")


def test_reads_revision_of_an_archive_snapshot(tmp_path):
    # A `git archive` extract has no .git; its snapshot builder writes REVISION instead.
    (tmp_path / "REVISION").write_text("0123abcd4567ef +0001-fix.patch\n")
    assert code_revision(str(tmp_path)) == "0123abcd4567ef +0001-fix.patch"


def test_prefers_git_over_revision(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text("cafe0123 refs/heads/main\n")
    (tmp_path / "REVISION").write_text("stale\n")
    assert code_revision(str(tmp_path)).startswith("cafe0123")


def test_empty_revision_is_loud(tmp_path):
    (tmp_path / "REVISION").write_text("\n")
    assert code_revision(str(tmp_path)).startswith("UNRESOLVED")
