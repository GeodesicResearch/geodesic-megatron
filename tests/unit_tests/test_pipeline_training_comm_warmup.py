"""ISAMBARD_COMM_WARMUP dispatch tests for pipeline_training_run.main().

The warmup mode selects whether NCCL communicators are pre-created at parallel-state setup
(`collectives`) and whether pipeline P2P transports are warmed too (`full`/`1`). Getting the
mapping wrong is expensive in both directions: without the collectives wave, the MoE router
expert-bias all_reduce cudaMallocs at peak memory and OOMs at iteration 0; with P2P warming
on a deep-PP run, steady state was measured ~9.6x slower. So the mapping is asserted here
rather than left to the reader.

`main()` reads the variable and dispatches before it parses argv, so these tests call the
REAL `main()` with a recording `patch_eager_comm_warmup` and an argv argparse must reject —
the dispatch runs, then the run aborts at argument parsing. Both modules are loaded by path
(the `test_pipeline_training_run_dispatch.py` idiom) and registered under their import names,
so `main()`'s inline `from pipeline_training_patches import ...` resolves to the module the
recorder was installed on.

Run:
    ./pipeline_env_exec.sh "cd <repo>; source pipeline_env_activate.sh || exit 1; cd /tmp; \\
        python -m pytest <repo>/tests/unit_tests/test_pipeline_training_comm_warmup.py -v"
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]

# Every other ISAMBARD_* patch gate main() consults before the warmup one; cleared so a
# stray value in the environment cannot install an unrelated patch during these tests.
_OTHER_PATCH_GATES = ("ISAMBARD_FP32_SSM_STATE", "ISAMBARD_MAMBA_SAVE_OFFLOAD", "ISAMBARD_DATA_ROW_TELEMETRY")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def patches_module():
    return _load("pipeline_training_patches", _REPO_ROOT / "pipeline_training_patches.py")


@pytest.fixture(scope="module")
def run_module(patches_module):
    return _load("pipeline_training_run", _REPO_ROOT / "pipeline_training_run.py")


def _dispatch(run_module, patches_module, monkeypatch, warmup_mode):
    """Run main()'s warmup dispatch under `warmup_mode`; return the recorded call kwargs."""
    calls: list[dict] = []

    def recorder(**kwargs):
        # Keyword-only on purpose: a call site that switched to positional args would fail
        # here instead of silently passing include_p2p as something else.
        calls.append(kwargs)
        return True

    monkeypatch.setattr(patches_module, "patch_eager_comm_warmup", recorder)
    for gate in _OTHER_PATCH_GATES:
        monkeypatch.delenv(gate, raising=False)
    if warmup_mode is None:
        monkeypatch.delenv("ISAMBARD_COMM_WARMUP", raising=False)
    else:
        monkeypatch.setenv("ISAMBARD_COMM_WARMUP", warmup_mode)
    # argparse requires --model/--mode, so main() aborts right after the dispatch.
    monkeypatch.setattr(sys, "argv", ["pipeline_training_run.py"])
    return calls


def test_unset_installs_no_warmup(run_module, patches_module, monkeypatch):
    calls = _dispatch(run_module, patches_module, monkeypatch, None)
    with pytest.raises(SystemExit):
        run_module.main()
    assert calls == [], "the default must be upstream lazy communicator creation"


def test_zero_installs_no_warmup(run_module, patches_module, monkeypatch):
    calls = _dispatch(run_module, patches_module, monkeypatch, "0")
    with pytest.raises(SystemExit):
        run_module.main()
    assert calls == []


def test_collectives_warms_groups_without_p2p(run_module, patches_module, monkeypatch):
    """The memory-tight MoE mode: pre-create group communicators, leave P2P lazy."""
    calls = _dispatch(run_module, patches_module, monkeypatch, "collectives")
    with pytest.raises(SystemExit):
        run_module.main()
    assert calls == [{"include_p2p": False}]


@pytest.mark.parametrize("warmup_mode", ["full", "1"])
def test_full_and_one_also_warm_p2p(run_module, patches_module, monkeypatch, warmup_mode):
    """`1` is the legacy spelling of `full` and must keep meaning P2P-inclusive."""
    calls = _dispatch(run_module, patches_module, monkeypatch, warmup_mode)
    with pytest.raises(SystemExit):
        run_module.main()
    assert calls == [{"include_p2p": True}]


def test_an_unknown_mode_is_rejected(run_module, patches_module, monkeypatch):
    calls = _dispatch(run_module, patches_module, monkeypatch, "collective")  # plausible typo
    with pytest.raises(ValueError, match="ISAMBARD_COMM_WARMUP must be one of"):
        run_module.main()
    assert calls == [], "a rejected mode must not install a patch"


def test_include_p2p_has_no_default(patches_module):
    """Required, not defaulted: the caller must state which mode it wants.

    A default silently picks one of two behaviours whose costs differ by ~9.6x steady-state
    at deep PP, so the parameter is spelled at every call site instead.
    """
    for function in (patches_module.patch_eager_comm_warmup, patches_module._warmup_all_communicators):
        parameter = inspect.signature(function).parameters["include_p2p"]
        assert parameter.default is inspect.Parameter.empty, f"{function.__name__} defaults include_p2p"
        assert parameter.annotation in (bool, "bool")
