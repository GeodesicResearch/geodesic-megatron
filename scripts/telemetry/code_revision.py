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

"""The commit a checkout's code is at, read from the files git keeps rather than by running git.

The training container may lack the git binary, and a copy made with ``git archive`` has no ``.git``
at all: such a copy names its commit in a ``REVISION`` file at its root, which is read instead. Both
the torch profiler's provenance (``scripts/profiling/profiler_callback.py``) and the stage guard
(``scripts/training/stage_guard.py``, which runs under the host's Python 3.6) record it, so this
module keeps to the standard library and to Python 3.6.
"""

import os
from typing import Tuple


def git_dirs(repo_dir: str) -> Tuple[str, str]:
    """The (gitdir, common dir) of the checkout at ``repo_dir``.

    In a linked worktree ``.git`` is a file, ``gitdir: <path>``, naming the worktree's own
    gitdir, which holds its HEAD; the branches live in the main repository's gitdir, which the
    worktree gitdir's ``commondir`` file names relative to itself.
    """
    dot_git = os.path.join(repo_dir, ".git")
    if not os.path.isfile(dot_git):
        return dot_git, dot_git
    with open(dot_git) as f:
        pointer = f.read().strip()
    if not pointer.startswith("gitdir: "):
        raise ValueError("{} is a file but not a 'gitdir: <path>' pointer".format(dot_git))
    gitdir = os.path.join(repo_dir, pointer[len("gitdir: ") :])
    commondir_path = os.path.join(gitdir, "commondir")
    if not os.path.exists(commondir_path):
        return gitdir, gitdir
    with open(commondir_path) as f:
        return gitdir, os.path.normpath(os.path.join(gitdir, f.read().strip()))


def code_revision(repo_dir: str) -> str:
    """The commit HEAD names (with its branch), or a copy's ``REVISION`` when there is no ``.git``.

    Never raises: what cannot be resolved is returned as ``UNRESOLVED (<why>)``, so a provenance
    record states it rather than leaving the field out.
    """
    revision_path = os.path.join(repo_dir, "REVISION")
    try:
        if not os.path.exists(os.path.join(repo_dir, ".git")) and os.path.exists(revision_path):
            with open(revision_path) as f:
                revision = f.read().strip()
            return revision or "UNRESOLVED ({} is empty)".format(revision_path)
        gitdir, commondir = git_dirs(repo_dir)
        with open(os.path.join(gitdir, "HEAD")) as f:
            head = f.read().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[len("ref: ") :]
        ref_path = os.path.join(commondir, ref)
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                return "{} ({})".format(f.read().strip(), ref)
        with open(os.path.join(commondir, "packed-refs")) as f:
            for line in f:
                fields = line.split()
                if len(fields) == 2 and fields[1] == ref:
                    return "{} ({})".format(fields[0], ref)
        return "UNRESOLVED ({} is neither at {} nor in packed-refs)".format(ref, ref_path)
    except (OSError, ValueError) as e:
        return "UNRESOLVED ({})".format(e)
