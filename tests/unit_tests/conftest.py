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
import gc
import logging
import os
import subprocess
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
_xdist_worker = os.environ.get("PYTEST_XDIST_WORKER", "")
_XDIST_MASTER_PORT = str(29500 + 41 * (int(_xdist_worker[2:]) + 1)) if _xdist_worker.startswith("gw") else None
if _XDIST_MASTER_PORT is not None:
    os.environ["MASTER_PORT"] = _XDIST_MASTER_PORT


def xdist_worker_device(worker: str, visible: str | None, probed_count: int) -> str | None:
    """The single CUDA device an xdist worker should see, or None to leave the env alone.

    Round-robins workers over the devices an externally-set CUDA_VISIBLE_DEVICES
    exposes (respecting any outer mask), falling back to the ``probed_count``
    physically-present devices when no mask is set. Serial runs and hosts with no
    GPUs return None.
    """
    if not worker.startswith("gw"):
        return None
    devices = [d for d in visible.split(",") if d] if visible is not None else [str(i) for i in range(probed_count)]
    if not devices:
        return None
    return devices[int(worker[2:]) % len(devices)]


def _probe_gpu_count() -> int:
    """Count physical GPUs without importing torch (whose CUDA state must stay uninitialized here)."""
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return 0
    return sum(1 for line in listing.stdout.splitlines() if line.startswith("GPU "))


# The same xdist concern, for GPU memory: every worker's CUDA context lands on
# cuda:0 by default, so `-n 8` piles ~8 GiB of contexts onto ONE GPU of a 4-GPU
# node while its siblings idle — enough to OOM the suite whenever a co-resident
# job (a serving endpoint, a training run) already holds most of that GPU.
# Narrow CUDA_VISIBLE_DEVICES to one device per worker HERE, before the
# `import torch` below, because CUDA reads the mask once at context creation.
# No per-test re-pin is needed for the same reason: once the context exists the
# mask is inert, and reset_env_vars restores the value after any test that
# mutates it.
_XDIST_DEVICE = xdist_worker_device(_xdist_worker, os.environ.get("CUDA_VISIBLE_DEVICES"), _probe_gpu_count())
if _XDIST_DEVICE is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _XDIST_DEVICE

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


def _enforce_master_port_convention(item):
    """Fail, at the offending test, any hard assignment of MASTER_PORT.

    A test that hard-assigns MASTER_PORT (instead of os.environ.setdefault)
    collides with concurrently-running workers; without this check the symptom
    is a nondeterministic EADDRINUSE in some OTHER test. Teardown runs before
    fixture finalizers restore the environment, so the overwrite is still
    visible and attributed to the test that made it.
    """
    if _XDIST_MASTER_PORT is None:
        return
    current = os.environ.get("MASTER_PORT")
    assert current == _XDIST_MASTER_PORT, (
        f"{item.nodeid} overwrote MASTER_PORT to {current!r} (worker port is "
        f"{_XDIST_MASTER_PORT}); use os.environ.setdefault so the per-xdist-worker "
        "port stays authoritative."
    )


# Live GPU memory above which a finished test is worth a gc pass: an unconditional
# collect after every one of ~6800 tests adds minutes of pure gc time to the suite,
# while the model-heavy tests the pass exists for hold GiBs.
_GC_WORTHWHILE_LIVE_BYTES = 256 * 2**20


def _release_finished_test_gpu_memory():
    """Hand a finished test's GPU memory back to the driver.

    Two steps: gc.collect() first, because nn.Module graphs are full of
    reference cycles and a just-finished test's multi-GiB model stays LIVE
    until the cycle collector runs (the next test in the same file then builds
    its own model on top of the leftover); then empty_cache(), because
    freed-but-cached allocator blocks are invisible to every OTHER process
    sharing the GPU, so an idle worker holds ~1 GiB hostage while its
    GPU-sibling's next big model build OOMs on memory nobody is using. Both
    matter under xdist on a shared node: a co-resident job (a serving endpoint,
    a training run) can leave only a few GiB free per GPU, and the heaviest
    test files peak within that budget only if earlier tests' memory is
    genuinely released between tests.
    """
    if not torch.cuda.is_initialized():
        return
    if torch.cuda.memory_allocated() >= _GC_WORTHWHILE_LIVE_BYTES:
        gc.collect()
    torch.cuda.empty_cache()


def pytest_runtest_teardown(item, nextitem):
    """Per-test teardown: MASTER_PORT convention enforcement, then GPU-memory release."""
    _enforce_master_port_convention(item)
    _release_finished_test_gpu_memory()


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


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0
