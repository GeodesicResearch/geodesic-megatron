# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""configs/control_pretraining/submit_corpus_job.py: one corpus tool's job, submitted from a frozen copy of a commit,
forced as a one-node job.

The real script runs as a subprocess from a scratch directory laid out as the frozen copy it requires. isambard_sbatch
is a stub on PATH, because the real wrapper would submit a SLURM job: it records the arguments, working directory and
environment it was called with, and answers as the wrapper does (its reports, then ``Submitted batch job N``), or
refuses with the exit status the test gives it.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import slurm_jobs  # noqa: E402


SCRIPT = Path("configs/control_pretraining/submit_corpus_job.py")
FROZEN_FILES = (SCRIPT, Path("scripts/slurm_jobs.py"), Path("scripts/training/launch_environment.py"))
TOOL_ARGS = (
    "configs/control_pretraining/corpus_documents.py",
    "check-hashes",
    "--config",
    "configs/metagaming_filtering/30b_clueless_norm/digest_checks.yaml",
    "--subset",
    "climbmix_full",
    "--shard",
    "1",
)
ARGS = ("cp-x-hashes-climbmix_full_shard1", "04:00:00", *TOOL_ARGS)
STUB = """#!/usr/bin/env python3
import json, os, sys
with open(os.environ["STUB_RECORD"], "w") as f:
    json.dump({"argv": sys.argv[1:], "cwd": os.getcwd(), "force": os.environ.get("ISAMBARD_SBATCH_FORCE"),
               "repo": os.environ.get("GEODESIC_REPO_DIR")}, f)
status = int(os.environ.get("STUB_EXIT", "0"))
if status:
    print("BLOCKED: the account is over its node limit", file=sys.stderr)
    sys.exit(status)
print("Bad nodes: 0 excluded\\nStorage: /projects/a5k 90%\\nSubmitted batch job 4242")
"""


@pytest.fixture
def frozen_root(tmp_path) -> Path:
    root = tmp_path / "copy"
    for relative in FROZEN_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, root / relative)
    (root / "REVISION").write_text("0123456789abcdef0123456789abcdef01234567\n")
    return root


def submit(root: Path, *args: str, **environment: str) -> tuple[subprocess.CompletedProcess, dict | None]:
    """The script from ``root`` with the stub isambard_sbatch first on PATH, in an environment holding nothing but
    PATH, HOME and ``environment``; its result, and what the stub recorded (None when it was never called)."""
    bindir, record = root.parent / "bin", root.parent / "stub.json"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "isambard_sbatch"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "HOME": os.environ.get("HOME", str(root)),
        "STUB_RECORD": str(record),
        **environment,
    }
    result = subprocess.run(
        [sys.executable, str(root / SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60, cwd=root
    )
    return result, json.loads(record.read_text()) if record.exists() else None


def test_it_submits_the_tool_as_a_corpus_job_from_the_copy_forced(frozen_root):
    result, called = submit(frozen_root, *ARGS)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "4242\n"
    assert called == {
        "argv": [
            "--job-name=cp-x-hashes-climbmix_full_shard1",
            "--time=04:00:00",
            "configs/control_pretraining/corpus_job.sbatch",
            *TOOL_ARGS,
        ],
        "cwd": str(frozen_root),
        "force": "1",
        "repo": str(frozen_root),
    }
    assert (frozen_root / slurm_jobs.SLURM_LOG_DIR).is_dir()


def test_force_is_its_own_whatever_the_shell_holds(frozen_root):
    result, called = submit(frozen_root, *ARGS, ISAMBARD_SBATCH_FORCE="0")
    assert result.returncode == 0, result.stderr
    assert called["force"] == "1"


def test_a_refused_submission_fails_with_the_wrappers_reason(frozen_root):
    result, called = submit(frozen_root, *ARGS, STUB_EXIT="1")
    assert result.returncode == 1 and called is not None
    assert "FATAL" in result.stderr and "over its node limit" in result.stderr
    assert result.stdout == ""


def test_a_dry_run_prints_the_submission_and_submits_nothing(frozen_root):
    result, called = submit(frozen_root, "--dry-run", *ARGS)
    assert result.returncode == 0, result.stderr
    assert called is None
    command = [
        "isambard_sbatch",
        f"--job-name={ARGS[0]}",
        f"--time={ARGS[1]}",
        "configs/control_pretraining/corpus_job.sbatch",
        *TOOL_ARGS,
    ]
    expected = slurm_jobs.shell_submission(command, frozen_root, slurm_jobs.forced_submission_env(frozen_root))
    assert result.stdout == expected + "\n"


def test_it_refuses_a_directory_without_a_revision(frozen_root):
    (frozen_root / "REVISION").unlink()
    result, called = submit(frozen_root, *ARGS)
    assert result.returncode == 1 and "has no REVISION file" in result.stderr
    assert called is None


def test_it_refuses_a_shell_carrying_launch_settings(frozen_root):
    result, called = submit(frozen_root, *ARGS, ISAMBARD_ENV_OVERRIDES="/tmp/x.env")
    assert result.returncode == 1 and "ISAMBARD_ENV_OVERRIDES" in result.stderr
    assert called is None


@pytest.mark.parametrize("args", [(), ("job",), ("job", "01:00:00")])
def test_it_needs_a_job_name_a_time_limit_and_a_tool(frozen_root, args):
    result, called = submit(frozen_root, *args)
    assert result.returncode == 2 and "required" in result.stderr
    assert called is None
