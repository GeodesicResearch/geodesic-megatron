"""Which GPUs a unit-test run may use, and which one each pytest-xdist worker is pinned to.

GPUs in Exclusive_Process compute mode hold one process's CUDA context at a time, so two xdist workers
that touch the same device fail with "all CUDA-capable devices are busy or unavailable".
scripts/run_unit_tests.sh therefore runs one worker per GPU in that mode and sets
``UNIT_TESTS_PIN_WORKER_GPUS=1``. tests/unit_tests/conftest.py then gives each worker one device
through CUDA_VISIBLE_DEVICES at import, before torch creates a context. Unpinned, a run keeps every
device for every process, so tests that use more than one GPU, or start child processes with their
own CUDA contexts, can run in the same pass.

The module imports nothing heavy, so the runner can count the GPUs with the same rule before torch
loads.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


PIN_WORKER_GPUS_ENV = "UNIT_TESTS_PIN_WORKER_GPUS"


def visible_gpus(env: Mapping[str, str], device_dir: str | Path) -> list[str]:
    """The GPU indices this process may use: CUDA_VISIBLE_DEVICES when set, else the /dev/nvidiaN nodes."""
    declared = env.get("CUDA_VISIBLE_DEVICES")
    if declared is not None:
        return [index for index in declared.split(",") if index]
    nodes = [path.name[len("nvidia") :] for path in Path(device_dir).glob("nvidia[0-9]*")]
    return sorted((index for index in nodes if index.isdigit()), key=int)


def xdist_worker_index(worker: str) -> int | None:
    """The index N of a pytest-xdist worker id ``gwN``, or None for a serial run's id."""
    if not worker.startswith("gw"):
        return None
    return int(worker[2:])


def resolve_worker_gpu(worker: str, gpus: list[str]) -> str | None:
    """The one GPU an xdist worker (``gwN``) may use, or None for a serial run or a node without GPUs."""
    index = xdist_worker_index(worker)
    if index is None or not gpus:
        return None
    return gpus[index % len(gpus)]


def pinned_worker_gpu(env: Mapping[str, str], worker: str, device_dir: str | Path) -> str | None:
    """The GPU this worker is pinned to when the run asks for pinning, else None."""
    if env.get(PIN_WORKER_GPUS_ENV) != "1":
        return None
    return resolve_worker_gpu(worker, visible_gpus(env, device_dir))
