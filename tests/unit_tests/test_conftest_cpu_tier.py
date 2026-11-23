# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The CPU-tier switch of tests/unit_tests/conftest.py.

scripts/run_cpu_tests.sh sets UNIT_TESTS_CPU_TIER=1 and runs the test files that use the ``cpu`` marker on a host
without torch and megatron.core. With the variable set, the conftest loads without both packages, skips each test
marked ``run_only_on("GPU")`` and keeps pytest's "no tests collected" status a failure. These tests pin all three, on
a host with the packages and on a host without them.

Each child is a real pytest session, so the real conftest loads. The subprocess boundary exists only to set the
variable before the conftest is imported.
"""

import os
import subprocess

import pytest

from tests.unit_tests import conftest
from tests.unit_tests.child_process import run_pytest


pytestmark = pytest.mark.cpu

TIMEOUT_SECONDS = 120


def _cpu_tier_session(scratch, *selection: str) -> subprocess.CompletedProcess:
    """A child pytest session over this file with the CPU tier on, narrowed by ``selection``, run from ``scratch``."""
    return run_pytest(
        [os.path.abspath(__file__), *selection], timeout=TIMEOUT_SECONDS, cwd=scratch, **{conftest.CPU_TIER_ENV: "1"}
    )


def test_the_conftest_imports_torch_and_megatron_core_exactly_when_the_cpu_tier_is_off():
    """In-suite probe: the conftest holds both names in the container's run and neither in the CPU tier's."""
    held = {name for name in ("torch", "MultiStorageClientFeature") if hasattr(conftest, name)}
    assert held == (set() if conftest.CPU_TIER else {"torch", "MultiStorageClientFeature"})


def test_the_conftest_loads_without_both_packages_when_the_variable_is_set(tmp_path):
    """Runs the probe above in a child session with the variable set, whatever this session's tier is."""
    probe = test_the_conftest_imports_torch_and_megatron_core_exactly_when_the_cpu_tier_is_off.__name__
    result = _cpu_tier_session(tmp_path, "-k", probe)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "1 passed" in result.stdout


@pytest.mark.run_only_on("GPU")
def test_a_gpu_only_test_runs_only_outside_the_cpu_tier():
    """In-suite probe: check_gpu_requirements skips this test in every session that has the CPU tier on."""
    assert not conftest.CPU_TIER


def test_the_cpu_tier_skips_a_gpu_only_test(tmp_path):
    """Runs the probe above in a child session with the variable set."""
    result = _cpu_tier_session(tmp_path, "-k", test_a_gpu_only_test_runs_only_outside_the_cpu_tier.__name__)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "1 skipped" in result.stdout


def test_a_cpu_tier_run_that_selects_no_test_fails(tmp_path):
    """The quarantine marker selects nothing in this file, which is the state of a tier whose marker selects nothing."""
    result = _cpu_tier_session(tmp_path, "-m", "pleasefixme")
    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, f"{result.stdout}\n{result.stderr}"
