# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Submit one corpus tool as its own 1-node job (``corpus_job.sbatch``), from a frozen copy of a commit.

    python3 configs/control_pretraining/submit_corpus_job.py [--dry-run] <job name> <time limit> <tool.py> [args...]

e.g. the digest check of one slice of a sliced corpus:

    python3 configs/control_pretraining/submit_corpus_job.py cp-30b_clueless_norm-hashes-climbmix_full_shard1 \\
        04:00:00 configs/control_pretraining/corpus_documents.py check-hashes \\
        --config configs/metagaming_filtering/30b_clueless_norm/digest_checks.yaml --subset climbmix_full --shard 1 \\
        --report-out /projects/a5k/public/logs/metagaming_filtering/hashes/climbmix_full_shard1.json

Run this file from a frozen copy of the commit (a directory with a ``REVISION`` file at its root, made as
tests/e2e_tests/README.md, "How one is run", makes one), never from a working checkout: the job reads its code from
the copy this file sits in after the submission has returned, and ``corpus_job.sbatch`` logs the copy's commit. It
refuses a copy without ``REVISION`` and a shell carrying launch settings (``scripts/training/launch_environment.py``),
submits through ``isambard_sbatch`` from the copy's root with ``slurm_jobs.forced_submission_env`` (the job is one
node, which the node-limit rule in configs/control_pretraining/README.md submits forced), and prints the job id;
``--dry-run`` prints the submission as a shell line instead and submits nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import slurm_jobs  # noqa: E402


sys.path.insert(0, str(REPO_ROOT))
from scripts.training.launch_environment import inherited_launch_settings  # noqa: E402


CORPUS_JOB = "configs/control_pretraining/corpus_job.sbatch"


class NotAFrozenCopy(RuntimeError):
    """The submission would not run a fixed commit's code, or would inherit launch settings."""


def submission_command(job_name: str, time_limit: str, tool: str, tool_args: list[str]) -> list[str]:
    """The isambard_sbatch command that runs ``tool`` with ``tool_args`` as one ``corpus_job.sbatch`` job."""
    return ["isambard_sbatch", f"--job-name={job_name}", f"--time={time_limit}", CORPUS_JOB, tool, *tool_args]


def require_frozen_copy(root: Path, environ: dict[str, str]) -> None:
    """Raise unless ``root`` names its commit in a ``REVISION`` file and ``environ`` carries no launch settings."""
    if not (root / "REVISION").is_file():
        raise NotAFrozenCopy(f"{root} has no REVISION file; submit from a frozen copy of the commit")
    inherited = inherited_launch_settings(environ)
    if inherited:
        raise NotAFrozenCopy(f"the shell carries launch settings {inherited}; submit from a shell without them")


def main(argv: list[str] | None = None) -> int:
    """Submit the job (or print it, with ``--dry-run``); non-zero on a refusal or a failed submission."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the submission and submit nothing")
    parser.add_argument("job_name", help="the job's SLURM name")
    parser.add_argument("time_limit", help="the job's SLURM time limit, e.g. 04:00:00")
    parser.add_argument("tool", help="the tool to run, relative to the copy's root")
    parser.add_argument("tool_args", nargs=argparse.REMAINDER, help="the tool's arguments, forwarded verbatim")
    args = parser.parse_args(argv)
    command = submission_command(args.job_name, args.time_limit, args.tool, args.tool_args)
    env = slurm_jobs.forced_submission_env(REPO_ROOT)
    try:
        require_frozen_copy(REPO_ROOT, dict(os.environ))
        if args.dry_run:
            print(slurm_jobs.shell_submission(command, REPO_ROOT, env))
            return 0
        (REPO_ROOT / slurm_jobs.SLURM_LOG_DIR).mkdir(parents=True, exist_ok=True)
        print(slurm_jobs.submit(command, REPO_ROOT, env))
    except (NotAFrozenCopy, slurm_jobs.SlurmError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
