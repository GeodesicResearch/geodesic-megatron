# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/run_cpu_tests.sh: the one pytest pass it starts over the test files that use the ``cpu`` marker.

These tests run the real script. One boundary is stubbed on PATH, because the real ``python -m pytest`` runs the CPU
tier from inside one of its own tests. The python stub records the arguments, the working directory and the CPU-tier
variable of each pytest invocation. Each test asserts on that record.

The stub is a Python script run by this interpreter, so no shell starts inside it. The script runs through
tests/unit_tests/stubbed_shell.py, in a minimal environment and a session of its own. stubbed_shell.py kills the
whole process group if the script outlives its timeout.
"""

import json
import shutil
import stat
import sys
from pathlib import Path

import pytest

from tests.unit_tests.conftest import CPU_TIER_ENV
from tests.unit_tests.stubbed_shell import run_in_own_session, stubbed_shell_env


pytestmark = pytest.mark.cpu


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_cpu_tests.sh"
TIMEOUT_SECONDS = 60

# One JSON line per pytest invocation: its arguments, its working directory and the CPU-tier variable (null if unset).
_PYTHON = f"""
import json
import os
import sys

args = sys.argv[1:]
if args[:2] != ["-m", "pytest"]:
    sys.exit(f"python stub: unexpected invocation {{args}}")
with open(os.environ["STUB_PYTEST_CALLS"], "a") as calls:
    call = {{"args": args[2:], "cwd": os.getcwd(), "cpu_tier": os.environ.get("{CPU_TIER_ENV}")}}
    calls.write(json.dumps(call) + "\\n")
sys.exit(int(os.environ["STUB_PYTEST_EXIT"]))
"""


@pytest.fixture
def run_script(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "python"
    stub.write_text(f"#!{sys.executable} -I\n{_PYTHON}")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / "pytest_calls"

    def run(script=SCRIPT, pytest_exit=0):
        env = stubbed_shell_env(
            bindir, {"STUB_PYTEST_EXIT": str(pytest_exit), "STUB_PYTEST_CALLS": str(calls), "TMPDIR": str(tmp_path)}
        )
        result = run_in_own_session(["bash", str(script)], env, timeout=TIMEOUT_SECONDS)
        recorded = calls.read_text().splitlines() if calls.exists() else []
        return result, [json.loads(line) for line in recorded]

    return run


def test_it_starts_one_pass_over_every_test_file_that_uses_the_marker(run_script):
    result, calls = run_script()
    assert result.returncode == 0, result.stderr
    (call,) = calls
    assert call["cpu_tier"] == "1"
    args = call["args"]
    assert args[args.index("-m") + 1] == "cpu and not pleasefixme"
    marked = sorted(
        str(path) for path in (REPO_ROOT / "tests" / "unit_tests").rglob("test_*.py") if "mark.cpu" in path.read_text()
    )
    assert str(Path(__file__).resolve()) in marked
    assert sorted(arg for arg in args if arg.endswith(".py")) == marked


def test_the_pass_runs_from_a_scratch_directory_that_the_script_removes(run_script):
    result, calls = run_script()
    assert result.returncode == 0, result.stderr
    (call,) = calls
    scratch = Path(call["cwd"])
    assert REPO_ROOT not in (scratch, *scratch.parents)
    assert not scratch.exists()


def test_the_status_of_the_pass_is_the_status_of_the_script(run_script):
    result, calls = run_script(pytest_exit=3)
    assert result.returncode == 3
    assert len(calls) == 1


def test_a_checkout_with_no_marked_test_file_is_refused_before_any_pass(run_script, tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "tests" / "unit_tests").mkdir(parents=True)
    (checkout / "tests" / "unit_tests" / "test_unmarked.py").write_text("def test_nothing():\n    pass\n")
    script = Path(shutil.copy(SCRIPT, checkout / "scripts"))
    result, calls = run_script(script=script)
    assert result.returncode == 1
    assert "uses the cpu marker" in result.stderr
    assert calls == []
