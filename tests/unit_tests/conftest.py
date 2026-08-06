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
import importlib.util
import logging
import os
import sys
from pathlib import Path
from shutil import rmtree
from unittest.mock import patch

import pytest


# Under pytest-xdist (the pre-commit hook runs `-n 8 --dist loadfile`), tests that
# initialize torch.distributed in different files run concurrently and collide on
# the default MASTER_PORT (29500, EADDRINUSE). Assign each worker its own port at
# conftest-import time — before any test's os.environ.setdefault can pin the
# default — and re-pin it before every test, because several distributed-test
# teardowns pop MASTER_PORT from the environment (fine serially, but on a worker
# it would drop the next file back onto the shared default). Test files must use
# os.environ.setdefault for MASTER_PORT so the worker port stays authoritative.
# Serial runs (no PYTEST_XDIST_WORKER) are untouched.
#
# The base must carry per-SESSION entropy, not only per-worker: two suites running
# concurrently on one node (separate worktrees, or a gate retry racing an orphan of
# its own previous attempt) otherwise compute an identical port for the same worker
# index and the second to bind loses. That failure surfaces far from its cause —
# either DistNetworkError in whichever file happened to initialize torch.distributed,
# or a silent wedge where one worker stalls for minutes while the rest sit idle.
# The base is published into the environment by whichever process imports this file
# first (the xdist controller, before it spawns workers), so every worker of a run
# inherits one base while separate runs get different ones.
MASTER_PORT_BASE_ENV = "MEGATRON_TEST_MASTER_PORT_BASE"
_WORKER_PORT_STRIDE = 41
_MAX_WORKERS = 64
_BASE_PORT_MIN = 20000
_BASE_PORT_MAX = 60000


def resolve_master_port_base(env, pid):
    """Base port for this pytest session, in ``[_BASE_PORT_MIN, _BASE_PORT_MAX)``.

    An existing ``MASTER_PORT_BASE_ENV`` value wins, which is both how workers
    inherit the controller's choice and how a caller pins the base explicitly.
    Otherwise it is derived from ``pid``, spread by the same stride used between
    workers so that neighbouring pids do not produce overlapping worker ranges.

    Pinning it is a per-invocation tool, not something to export from a shell
    profile: two sessions that inherit the same exported base recreate exactly
    the collision this function exists to prevent, and they do so silently,
    because an explicit base is honoured rather than second-guessed.
    """
    override = env.get(MASTER_PORT_BASE_ENV)
    if override:
        return int(override)
    span = _BASE_PORT_MAX - _BASE_PORT_MIN - _WORKER_PORT_STRIDE * _MAX_WORKERS
    return _BASE_PORT_MIN + (pid * _WORKER_PORT_STRIDE) % span


def resolve_worker_master_port(worker, base):
    """Port for one xdist worker (``gwN``), or None for a serial run."""
    if not worker.startswith("gw"):
        return None
    return str(base + _WORKER_PORT_STRIDE * (int(worker[2:]) + 1))


_MASTER_PORT_BASE = resolve_master_port_base(os.environ, os.getpid())
os.environ[MASTER_PORT_BASE_ENV] = str(_MASTER_PORT_BASE)
_xdist_worker = os.environ.get("PYTEST_XDIST_WORKER", "")
_XDIST_MASTER_PORT = resolve_worker_master_port(_xdist_worker, _MASTER_PORT_BASE)
if _XDIST_MASTER_PORT is not None:
    os.environ["MASTER_PORT"] = _XDIST_MASTER_PORT

# Unit tests must not inherit the *submitting allocation's* size: on a SLURM
# compute node get_world_size_safe() falls back to SLURM_NTASKS, so a suite run
# inside an N-node tunnel silently sees world_size=N and world-size-dependent
# assertions (e.g. per-device throughput) fail. Tests that exercise the SLURM
# fallback deliberately build their env with patch.dict(clear=True), so
# scrubbing here does not affect them.
os.environ.pop("SLURM_NTASKS", None)


def pytest_runtest_setup(item):
    """Re-pin the per-xdist-worker MASTER_PORT and re-scrub SLURM_NTASKS.

    Both are import-time guarantees that individual tests can undo (teardown
    pops of MASTER_PORT; patch.dict tests that set SLURM_NTASKS), so they are
    re-established before every test.
    """
    if _XDIST_MASTER_PORT is not None:
        os.environ["MASTER_PORT"] = _XDIST_MASTER_PORT
    os.environ.pop("SLURM_NTASKS", None)


def pytest_runtest_teardown(item, nextitem):
    """Enforce the setdefault convention loudly, at the offending test.

    A test that hard-assigns MASTER_PORT (instead of os.environ.setdefault)
    collides with concurrently-running workers; without this check the symptom
    is a nondeterministic EADDRINUSE in some OTHER test. This hook runs before
    fixture finalizers restore the environment, so the overwrite is still
    visible and attributed to the test that made it.
    """
    if _XDIST_MASTER_PORT is not None:
        current = os.environ.get("MASTER_PORT")
        assert current == _XDIST_MASTER_PORT, (
            f"{item.nodeid} overwrote MASTER_PORT to {current!r} (worker port is "
            f"{_XDIST_MASTER_PORT}); use os.environ.setdefault so the per-xdist-worker "
            "port stays authoritative."
        )


import torch
from megatron.core.msc_utils import MultiStorageClientFeature


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pytest_configure(config):
    """Register custom markers for unit tests."""
    config.addinivalue_line(
        "markers",
        "pleasefixme: marks test as needing fixes (will be skipped in CI)",
    )
    config.addinivalue_line(
        "markers",
        "run_only_on: marks test to run only on specific hardware (CPU/GPU)",
    )


@pytest.fixture(autouse=True)
def cleanup_local_folder():
    """Cleanup local experiments folder"""
    # Asserts in fixture are not recommended, but I'd rather stop users from deleting expensive training runs
    assert not Path("./NeMo_experiments").exists()
    assert not Path("./nemo_experiments").exists()

    yield

    if Path("./NeMo_experiments").exists():
        rmtree("./NeMo_experiments", ignore_errors=True)
    if Path("./nemo_experiments").exists():
        rmtree("./nemo_experiments", ignore_errors=True)


@pytest.fixture(scope="function", autouse=True)
def disable_msc():
    """Disable MSC for the tests."""
    MultiStorageClientFeature.disable()


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables"""
    # Store the original environment variables before the test
    original_env = dict(os.environ)

    # Run the test
    yield

    # After the test, restore the original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def check_gpu_requirements(request):
    """Fixture to skip tests that require GPU when CUDA is not available"""
    marker = request.node.get_closest_marker("run_only_on")
    if marker and "gpu" in [arg.lower() for arg in marker.args]:
        if not torch.cuda.is_available():
            pytest.skip("Test requires GPU but CUDA is not available")


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Clear LRU cache before each test to ensure test isolation."""
    # Import the functions that use @lru_cache
    from megatron.bridge.training.utils.checkpoint_utils import read_run_config, read_train_state

    # Clear the cache before each test
    read_run_config.cache_clear()
    read_train_state.cache_clear()

    yield

    # Clear cache after each test as well
    read_run_config.cache_clear()
    read_train_state.cache_clear()


@pytest.fixture
def mock_distributed_environment():
    """Mock torch.distributed environment for testing."""
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
        patch("megatron.bridge.training.utils.checkpoint_utils.get_world_size_safe", return_value=1),
    ):
        yield


@pytest.fixture
def sample_config_data():
    """Provide sample configuration data for testing."""
    return {
        "model": {"type": "gpt", "layers": 24, "hidden_size": 1024, "attention_heads": 16},
        "training": {"learning_rate": 1e-4, "batch_size": 32, "max_steps": 10000, "warmup_steps": 1000},
        "optimizer": {"type": "adam", "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    }


@pytest.fixture
def sample_train_state_data():
    """Provide sample train state data for testing."""
    return {"iteration": 5000, "epoch": 10, "step": 50000, "learning_rate": 0.0001, "loss": 2.34}


@pytest.fixture(scope="module")
def run_module():
    """The repo-root pipeline_training_run.py, loaded by path.

    It is a top-level script rather than an installed module, so tests that need
    its real parser, dispatch table or config assembly load it through importlib
    instead of importing it.
    """
    run_path = Path(__file__).resolve().parents[2] / "pipeline_training_run.py"
    spec = importlib.util.spec_from_file_location("pipeline_training_run", run_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_training_run"] = module
    spec.loader.exec_module(module)
    return module


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0
