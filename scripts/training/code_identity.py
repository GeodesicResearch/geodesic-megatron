# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Refuse to train a config with any code but the code it names.

A training config may pin the code it must be trained with, in a top-level ``code_identity:`` block::

    code_identity:
      revision: <sha>       # the commit the hashes below were read from
      src_tree: <sha>       # git rev-parse <revision>:src
      launchers:            # repo-relative path: git rev-parse <revision>:<path>
        pipeline_training_run.py: <sha>
        pipeline_training_submit.sbatch: <sha>
      ancestor: <sha>       # a commit the code's history must contain
      history: <path>       # a git repository holding that history

``pipeline_training_launch.sh`` runs the check on REPO_DIR, the code that would train, on its own node before any
rank starts: REPO_DIR's ``src/`` must be the tree ``src_tree`` names, every listed launcher the blob it names, and
``ancestor`` an ancestor, in ``history``, of REPO_DIR's commit (a frozen copy's ``REVISION``, a checkout's HEAD). A
frozen copy (``git archive``) carries no history of its own, hence ``history``. Any difference refuses the launch.

The hashes are git's, computed by git: ``src/`` and the launchers are staged into a throwaway repository, so a file
git ignores (a ``__pycache__`` that running the code wrote) is left out exactly as git leaves it out, and nothing is
written into ``history`` or REPO_DIR. Every git call runs without the user's and the system's git configuration and
without the user's default ignore and attributes files, so only REPO_DIR's own ``.gitignore`` files decide.

The check prints its record (what the block states, what was measured) as one JSON line on stdout and its refusals
on stderr. The launcher hands the record to every rank in ``ISAMBARD_CODE_IDENTITY``; ``pipeline_training_run.py``
refuses a config carrying the block unless that record is a passing one for the same block
(``require_checked_code_identity``), and ``scripts/telemetry/run_identity.py`` writes it to the W&B run's config.

Usage (inside the container, from the repository root)::

    python -m scripts.training.code_identity --config <training yaml> --repo-dir <dir>

Exit 0 when the config pins no code (printing nothing on stdout) or the code is the code it names; 1 on any
difference or failure, naming each.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.mapping_keys import require_keys
from scripts.telemetry.code_revision import code_revision
from scripts.training.config_compose import load_composed_yaml


CODE_IDENTITY_KEY = "code_identity"
CODE_IDENTITY_FIELDS = frozenset({"revision", "src_tree", "launchers", "ancestor", "history"})
RECORD_ENV = "ISAMBARD_CODE_IDENTITY"
FULL_SHA = re.compile(r"[0-9a-f]{40}")
SOURCE_TREE = "src"


class CodeIdentityError(ValueError):
    """The code that would train is not the code the config names, or that cannot be established."""


@dataclass(frozen=True)
class CodeIdentity:
    """A config's ``code_identity:`` block (see the module docstring)."""

    revision: str
    src_tree: str
    launchers: dict[str, str]
    ancestor: str
    history: str


def _full_sha(value: object, where: str) -> str:
    if not isinstance(value, str) or not FULL_SHA.fullmatch(value):
        raise CodeIdentityError(f"{where} must be a full 40-character commit, tree or blob sha, got {value!r}")
    return value


def parse_code_identity(block: object, where: str) -> CodeIdentity:
    """The block as a ``CodeIdentity``; every field is required, every sha full, every launcher path repo-relative."""
    fields = require_keys(block, where, CODE_IDENTITY_FIELDS, error=CodeIdentityError)
    launchers = fields["launchers"]
    if not isinstance(launchers, dict) or not launchers:
        raise CodeIdentityError(f"{where}.launchers must map each launcher's repo-relative path to its blob sha")
    for path in launchers:
        if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise CodeIdentityError(f"{where}.launchers: {path!r} is not a path inside the repository")
    history = fields["history"]
    if not isinstance(history, str) or not Path(history).is_absolute():
        raise CodeIdentityError(f"{where}.history must be the absolute path of a git repository, got {history!r}")
    return CodeIdentity(
        revision=_full_sha(fields["revision"], f"{where}.revision"),
        src_tree=_full_sha(fields["src_tree"], f"{where}.src_tree"),
        launchers={path: _full_sha(sha, f"{where}.launchers.{path}") for path, sha in launchers.items()},
        ancestor=_full_sha(fields["ancestor"], f"{where}.ancestor"),
        history=history,
    )


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run git without the user's or the system's configuration and without an inherited repository.

    Git reads a user's default ignore and attributes files (``$XDG_CONFIG_HOME/git/ignore``, else
    ``~/.config/git/ignore``, and ``attributes`` beside it) even with no configuration file at all, so those are
    pointed at nothing too: what is hashed must not depend on who launches.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    hermetic = ["-c", f"core.excludesFile={os.devnull}", "-c", f"core.attributesFile={os.devnull}"]
    result = subprocess.run(["git", *hermetic, *args], capture_output=True, text=True, env=env)
    if check and result.returncode != 0:
        raise CodeIdentityError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def staged_hashes(repo_dir: Path, launchers: list[str]) -> tuple[str, dict[str, str]]:
    """Git's tree hash of ``repo_dir``'s ``src/`` and the blob hash of each launcher, as git would commit them.

    They are staged into a throwaway repository with ``repo_dir`` as its work tree, so ``repo_dir``'s own
    ``.gitignore`` files decide what is left out.
    """
    missing = [path for path in [SOURCE_TREE, *launchers] if not (repo_dir / path).exists()]
    if missing:
        raise CodeIdentityError(f"{repo_dir} holds no {missing}")
    with tempfile.TemporaryDirectory(prefix="code_identity_") as scratch:
        _git(["init", "--quiet", scratch])
        index = ["--git-dir", f"{scratch}/.git", "--work-tree", str(repo_dir)]
        _git([*index, "add", "--", SOURCE_TREE, *launchers])
        tree = _git([*index, "write-tree", f"--prefix={SOURCE_TREE}/"]).stdout.strip()
        blobs = {}
        for path in launchers:
            staged = _git([*index, "ls-files", "--stage", "--", path]).stdout.split()
            blobs[path] = staged[1]
    return tree, blobs


def code_commit(repo_dir: Path) -> str:
    """The commit ``repo_dir`` holds: a frozen copy's ``REVISION``, or a checkout's HEAD."""
    revision = code_revision(str(repo_dir))
    if revision.startswith("UNRESOLVED"):
        raise CodeIdentityError(f"the commit {repo_dir} holds cannot be read: {revision}")
    return revision.split()[0]


def has_ancestor(history: str, ancestor: str, commit: str) -> bool:
    """Whether ``ancestor`` is an ancestor of (or is) ``commit`` in the repository ``history``."""
    result = _git(["-C", history, "merge-base", "--is-ancestor", ancestor, commit], check=False)
    if result.returncode not in (0, 1):
        raise CodeIdentityError(
            f"{history} cannot tell whether {ancestor} is an ancestor of {commit}: {result.stderr.strip()}"
        )
    return result.returncode == 0


def measure_code_identity(identity: CodeIdentity, repo_dir: Path) -> dict:
    """What ``repo_dir`` holds of what the block names: its commit, the hashes, and the ancestry."""
    commit = code_commit(repo_dir)
    tree, blobs = staged_hashes(repo_dir, sorted(identity.launchers))
    return {
        "repo_dir": str(repo_dir),
        "commit": commit,
        "src_tree": tree,
        "launchers": blobs,
        "ancestor_in_history": has_ancestor(identity.history, identity.ancestor, commit),
    }


def code_identity_differences(identity: CodeIdentity, measured: dict) -> list[str]:
    """Each way the measured code differs from the block, one line each; empty when it is the code named."""
    differences = []
    if measured["src_tree"] != identity.src_tree:
        differences.append(
            f"src/ is tree {measured['src_tree']}, the config names {identity.src_tree} (src/ at {identity.revision})"
        )
    for path, blob in sorted(identity.launchers.items()):
        if measured["launchers"][path] != blob:
            differences.append(f"{path} is blob {measured['launchers'][path]}, the config names {blob}")
    if not measured["ancestor_in_history"]:
        differences.append(
            f"commit {measured['commit']} does not descend from {identity.ancestor} in {identity.history}"
        )
    return differences


def code_identity_record(identity: CodeIdentity, config_file: str, measured: dict) -> dict:
    """The record the launcher hands the ranks: the config, what its block states, what was measured, and how the
    two differ (``passed`` when they do not)."""
    differences = code_identity_differences(identity, measured)
    return {
        "config": config_file,
        "expected": asdict(identity),
        "measured": measured,
        "differences": differences,
        "passed": not differences,
    }


def config_code_identity(config_file: str) -> CodeIdentity | None:
    """The ``code_identity:`` block of a training config (through its ``base_config:`` chain), or None."""
    block = load_composed_yaml(config_file).get(CODE_IDENTITY_KEY)
    return None if block is None else parse_code_identity(block, f"{config_file}: {CODE_IDENTITY_KEY}")


def require_checked_code_identity(identity: CodeIdentity, config_file: str, environ: dict[str, str]) -> dict:
    """The launcher's passing record for a config that pins its code; refuse a run that has none for this block.

    A run started other than through ``pipeline_training_launch.sh`` was never checked, and a record made for
    another block, or one that did not pass, does not vouch for this one.
    """
    raw = environ.get(RECORD_ENV)
    if raw is None:
        raise CodeIdentityError(
            f"{config_file} pins the code it trains with ({CODE_IDENTITY_KEY}), and {RECORD_ENV}, the record of "
            "pipeline_training_launch.sh's check of it, is not set: launch it through pipeline_training_launch.sh"
        )
    record = json.loads(raw)
    if record.get("expected") != asdict(identity):
        raise CodeIdentityError(f"{RECORD_ENV} records the check of another {CODE_IDENTITY_KEY} than {config_file}'s")
    if record.get("passed") is not True:
        raise CodeIdentityError(f"{RECORD_ENV} records a check of {config_file}'s code that did not pass")
    return record


def main(argv: list[str] | None = None) -> int:
    """Check a config's code identity against a code directory; print the record, refuse on any difference."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the training config (its base_config chain is read)")
    parser.add_argument("--repo-dir", type=Path, required=True, help="the code that would train it")
    args = parser.parse_args(argv)
    try:
        identity = config_code_identity(args.config)
        if identity is None:
            print(f"[code-identity] {args.config} pins no code", file=sys.stderr)
            return 0
        record = code_identity_record(identity, args.config, measure_code_identity(identity, args.repo_dir.resolve()))
    except CodeIdentityError as error:
        print(f"FATAL [code-identity]: {error}", file=sys.stderr)
        return 1
    for difference in record["differences"]:
        print(f"FATAL [code-identity]: {difference}", file=sys.stderr)
    print(json.dumps(record, sort_keys=True))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
