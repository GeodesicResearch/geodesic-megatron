"""Unit tests for scripts/telemetry/run_identity.py (per-run identity, INFR-68)."""

import importlib.util
import os

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def mod():
    # Import the real module by path (top-level scripts/ package, not re-implemented).
    spec = importlib.util.spec_from_file_location(
        "run_identity", os.path.join(REPO_ROOT, "scripts", "telemetry", "run_identity.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeRun:
    """Stand-in for wandb.run carrying only the .summary surface we write to.

    A mock is justified here: the real wandb.run only exists after wandb.init
    against the W&B network service (or its offline daemon) — an external
    boundary unit tests must not touch.
    """

    def __init__(self):
        self.summary = FakeSummary()


class FakeSummary(dict):
    """dict with wandb's summary.update surface."""

    def update(self, d):  # noqa: A003 - mirrors wandb API
        dict.update(self, d)


# --- get_run_id -------------------------------------------------------------


def test_run_id_prefers_env(mod, monkeypatch):
    monkeypatch.setenv("ISAMBARD_RUN_ID", "20260724T140000-j789")
    monkeypatch.setenv("SLURM_JOB_ID", "789")
    assert mod.get_run_id() == "20260724T140000-j789"


def test_run_id_empty_env_treated_as_unset(mod, monkeypatch):
    monkeypatch.setenv("ISAMBARD_RUN_ID", "")
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    assert mod.get_run_id() == "j42"


def test_run_id_slurm_fallback_is_rank_stable(mod, monkeypatch):
    # Without the launcher, all ranks must still derive the SAME id — the
    # fallback is a pure function of SLURM_JOB_ID (no timestamps, no pids).
    monkeypatch.delenv("ISAMBARD_RUN_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "5738450")
    assert mod.get_run_id() == "j5738450"
    assert mod.get_run_id() == mod.get_run_id()


def test_run_id_local_fallback(mod, monkeypatch):
    monkeypatch.delenv("ISAMBARD_RUN_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    rid = mod.get_run_id()
    assert rid.startswith("local-")
    assert rid.endswith(f"-p{os.getpid()}")


# --- get_raw_log_path -------------------------------------------------------


def test_raw_log_path_from_env(mod, monkeypatch):
    monkeypatch.setenv("ISAMBARD_RAW_LOG_PATH", "/x/logs/slurm/train-1.out")
    assert mod.get_raw_log_path() == "/x/logs/slurm/train-1.out"


def test_raw_log_path_default_empty(mod, monkeypatch):
    monkeypatch.delenv("ISAMBARD_RAW_LOG_PATH", raising=False)
    assert mod.get_raw_log_path() == ""


# --- stamp_wandb_summary ----------------------------------------------------


def test_stamp_writes_namespaced_summary_keys(mod, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    run = FakeRun()
    mod.stamp_wandb_summary(run, "20260724T150000-j999", "/logs/slurm/train-999.out")
    assert run.summary["run/isambard_run_id"] == "20260724T150000-j999"
    assert run.summary["run/raw_log_path"] == "/logs/slurm/train-999.out"
    assert run.summary["run/slurm_job_id"] == "999"


def test_stamp_records_empty_log_path(mod, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    run = FakeRun()
    mod.stamp_wandb_summary(run, "rid", "")
    assert run.summary["run/raw_log_path"] == ""
    assert run.summary["run/slurm_job_id"] == ""


def test_stamp_none_run_is_noop(mod):
    # Every rank calls this; only the wandb-owning (LAST) rank has a run.
    mod.stamp_wandb_summary(None, "rid", "/log")  # must not raise


# --- RunIdentityCallback ----------------------------------------------------


def test_callback_is_a_bridge_callback(mod):
    from megatron.bridge.training.callbacks import Callback

    cb = mod.RunIdentityCallback(run_id="rid", raw_log_path="")
    assert isinstance(cb, Callback)


def test_callback_noop_without_wandb_run(mod):
    # wandb is importable in this env but wandb.init was never called, so
    # wandb.run is None — the callback must be a silent no-op, on every rank.
    cb = mod.RunIdentityCallback(run_id="rid", raw_log_path="")
    cb.on_train_start(ctx=None)  # must not raise


def test_callback_swallows_stamp_failures(mod, monkeypatch, capsys):
    # Callback exceptions propagate in megatron-bridge and would crash the
    # run — telemetry failures must be swallowed loudly instead.
    def boom(run, run_id, raw_log_path):
        raise RuntimeError("wandb exploded")

    monkeypatch.setattr(mod, "stamp_wandb_summary", boom)
    cb = mod.RunIdentityCallback(run_id="rid", raw_log_path="")
    cb.on_train_start(ctx=None)  # must not raise
    assert "WARNING: failed to stamp W&B summary" in capsys.readouterr().out


def test_callback_stamps_via_module_function(mod, monkeypatch):
    # The callback must route through stamp_wandb_summary with its stored
    # identity. wandb.run is None here, so patch the seam and verify the args.
    calls = []
    monkeypatch.setattr(mod, "stamp_wandb_summary", lambda run, rid, log: calls.append((rid, log)))
    cb = mod.RunIdentityCallback(run_id="20260724T160000-j1", raw_log_path="/l.out")
    cb.on_train_start(ctx=None)
    assert calls == [("20260724T160000-j1", "/l.out")]
