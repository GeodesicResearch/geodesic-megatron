"""The two SLURM operations the campaign's submitting tools share: reading this user's queue, and
submitting one job.

Both are strict in the same way. A queue that cannot be read is an error rather than an empty queue,
since reading it as empty would resubmit work already in flight; and a submission counts only when it
exits zero and names exactly one job, since the wrapper prints its reports around the id and a refusal
(the node cap, a bad-node or storage stop) exits non-zero with its reason on stderr.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


SUBMITTED = re.compile(r"Submitted batch job (\d+)")


class SlurmError(RuntimeError):
    """The queue could not be read, or a submission was refused or named no single job."""


def queued_job_names() -> set[str]:
    """Every job name this user currently has queued or running."""
    result = subprocess.run(
        ["squeue", "--me", "--noheader", "--format=%j"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SlurmError(f"squeue exited {result.returncode}: {result.stderr.strip()}")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def submit(command: list[str], cwd: Path, env: dict[str, str]) -> str:
    """Run a submission command from ``cwd`` with ``env`` laid over the environment; return the job id."""
    result = subprocess.run(command, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SlurmError(f"{command[0]} exited {result.returncode}: {result.stderr.strip()}")
    ids = SUBMITTED.findall(result.stdout)
    if len(ids) != 1:
        raise SlurmError(f"{command[0]} named {len(ids)} jobs, not one: {result.stdout.strip()}")
    return ids[0]
