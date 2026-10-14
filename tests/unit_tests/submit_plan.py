# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""What a staged submit script plans under ``DRY_RUN=1``, read back as its ``isambard_sbatch`` submissions.

The staged submit scripts (an end-to-end test's ``submit.sh``, the posture canary's, ``submit_midtrain.sh``) refuse to
run outside a frozen copy of the commit, and under ``DRY_RUN=1`` print one line per submission and submit nothing. The
tests of their
configs lay a scratch directory out as that copy, holding only the files the script reads, run the script there and
read its plan with the helpers below.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


PLANNED = re.compile(
    r"^\[dry-run\] (?P<description>[^:]+): ISAMBARD_SBATCH_FORCE=(?P<force>\S*) "
    r"(?:ISAMBARD_ENV_OVERRIDES=(?P<overrides>\S+) )?isambard_sbatch (?P<args>.*)$"
)


@dataclass(frozen=True)
class Submission:
    """One isambard_sbatch submission a stage plans."""

    description: str
    force: str
    overrides: str | None
    args: tuple[str, ...]

    def option(self, name: str) -> str | None:
        """The value of ``name=<value>`` among the sbatch options, or None."""
        values = [arg.split("=", 1)[1] for arg in self.args if arg.startswith(f"{name}=")]
        return values[0] if values else None

    def payload(self, script: str) -> tuple[str, ...]:
        """The arguments after the batch script."""
        return self.args[self.args.index(script) + 1 :]

    def payload_option(self, script: str, name: str) -> str:
        payload = self.payload(script)
        return payload[payload.index(name) + 1]


def planned_submissions(lines: Iterable[str]) -> dict[str, Submission]:
    """The submissions among a plan's lines, by description."""
    matches = [match for match in map(PLANNED.match, lines) if match]
    return {
        match["description"]: Submission(
            match["description"], match["force"], match["overrides"], tuple(match["args"].split())
        )
        for match in matches
    }


def frozen_copy(repository: Path, root: Path, files: Iterable[Path], revision: str) -> Path:
    """Lay out ``root`` as a frozen copy of ``repository`` at ``revision``, holding only ``files`` (paths relative to
    the repository; a directory is copied whole)."""
    for relative in files:
        source, target = repository / relative, root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    (root / "REVISION").write_text(revision + "\n")
    return root


def run_dry(script: Path, *arguments: str, **environment: str) -> subprocess.CompletedProcess:
    """``script`` under DRY_RUN=1, from an environment holding nothing but PATH and HOME, and ``environment``."""
    home = os.environ.get("HOME", str(script.parent))
    clean = {"PATH": os.environ["PATH"], "HOME": home, "DRY_RUN": "1", **environment}
    return subprocess.run(["bash", str(script), *arguments], env=clean, capture_output=True, text=True, timeout=60)
