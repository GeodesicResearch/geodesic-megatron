"""The two SLURM operations the campaign's tools share: reading this user's queue, and submitting one job.

Both are strict in the same way. A queue that cannot be read is an error rather than an empty queue,
since reading it as empty would resubmit work already in flight, or leave a job uncancelled; and a
submission counts only when it exits zero and names exactly one job, since the wrapper prints its reports
around the id and a refusal (the node cap, a bad-node or storage stop) exits non-zero with its reason on
stderr. The stage guard (``scripts/training/stage_guard.py``) reads the queue through this module under the
host's Python 3.6, so it keeps to the standard library and to Python 3.6.
"""

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set


SUBMITTED = re.compile(r"Submitted batch job (\d+)")
# Where the repo's sbatch wrappers write their output, relative to the directory they are submitted from
# (#SBATCH --output=logs/slurm/...). SLURM does not create the directory, and a job whose output file cannot be opened
# fails before it starts, so a submitting tool creates it first.
SLURM_LOG_DIR = Path("logs") / "slurm"

Runner = Callable[[List[str]], "subprocess.CompletedProcess"]


class SlurmError(RuntimeError):
    """The queue could not be read, or a submission was refused or named no single job."""


def run_capturing(command: List[str]) -> "subprocess.CompletedProcess":
    """Run a command with its stdout and stderr captured as text (Python 3.6 has no ``capture_output``)."""
    return subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False
    )


def queue_listing(field: str, job_name: Optional[str], run: Runner) -> List[str]:
    """Each non-empty line of this user's queue, queued or running, in ``field`` (an squeue ``--format`` string),
    for the jobs named ``job_name``, or for every job when it is None. Raises SlurmError when squeue fails."""
    command = ["squeue", "--me", "--noheader", "--format=" + field]
    if job_name is not None:
        command += ["--name", job_name]
    result = run(command)
    if result.returncode != 0:
        reason = (result.stderr or result.stdout or "").strip()
        raise SlurmError("squeue exited {}: {}".format(result.returncode, reason))
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def queued_job_names() -> Set[str]:
    """Every job name this user currently has queued or running."""
    return set(queue_listing("%j", None, run_capturing))


def forced_submission_env(repo_root: Path) -> Dict[str, str]:
    """What a forced submission adds to the environment: GEODESIC_REPO_DIR points the job at ``repo_root``, and
    ISAMBARD_SBATCH_FORCE=1 skips the wrapper's account node-cap check (it still excludes the bad nodes). The node-limit
    rule (configs/control_pretraining/README.md) submits one-node jobs this way: an export, an upload, a corpus job."""
    return {"GEODESIC_REPO_DIR": str(repo_root), "ISAMBARD_SBATCH_FORCE": "1"}


def shell_submission(command: List[str], repo_root: Path, env: Dict[str, str]) -> str:
    """``command`` as a line a person can paste into a shell to submit it from ``repo_root`` with ``env``."""
    assignments = " ".join("{}={}".format(name, shlex.quote(value)) for name, value in env.items())
    words = " ".join(shlex.quote(word) for word in command)
    return "cd {} && {} {}".format(shlex.quote(str(repo_root)), assignments, words)


def submit(command: List[str], cwd: Path, env: Dict[str, str]) -> str:
    """Run a submission command from ``cwd`` with ``env`` laid over the environment; return the job id."""
    environment = dict(os.environ)
    environment.update(env)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if result.returncode != 0:
        raise SlurmError("{} exited {}: {}".format(command[0], result.returncode, result.stderr.strip()))
    ids = SUBMITTED.findall(result.stdout)
    if len(ids) != 1:
        raise SlurmError("{} named {} jobs, not one: {}".format(command[0], len(ids), result.stdout.strip()))
    return ids[0]
