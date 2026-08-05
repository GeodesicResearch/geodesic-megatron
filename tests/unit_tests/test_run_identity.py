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


# --- Resolved-config provenance -------------------------------------------------
#
# A run's posture is not fully described by its override YAML: recipe defaults and CLI
# overrides (the 128-GPU benchmark is the 64-GPU config plus train.global_batch_size=256)
# live only in the resolved object. These cover the snapshot that makes such a run
# reproducible from disk.
#
# Every case below drives a REAL ConfigContainer, built by the training-config suite's own
# factory. An earlier revision used a two-attribute stand-in for the directory-resolution
# cases; that was wrong, because the stand-in re-declared `checkpoint.save` and
# `logger.wandb_save_dir` and so would have stayed green through a rename of either field —
# which is exactly the breakage these tests exist to catch.


@pytest.fixture
def cfg():
    """A real ConfigContainer with both artifact-directory fields cleared.

    Cleared rather than left at their defaults so each test states the one field it is
    exercising, and no case silently depends on what the factory happens to default to.
    """
    from tests.unit_tests.training.test_config import (
        create_test_config_container,
        create_test_gpt_config,
        restore_get_world_size_safe,
    )

    container, original, module_ref = create_test_config_container(
        world_size_override=1, model_config=create_test_gpt_config()
    )
    container.checkpoint.save = None
    container.logger.wandb_save_dir = None
    yield container
    restore_get_world_size_safe(original, module_ref)


class TestResolveRunArtifactDir:
    def test_prefers_checkpoint_save(self, mod, cfg):
        cfg.checkpoint.save = "/p/ckpt"
        cfg.logger.wandb_save_dir = "/p/wandb"
        assert mod.resolve_run_artifact_dir(cfg) == "/p/ckpt"

    def test_uses_wandb_save_dir_when_not_saving_checkpoints(self, mod, cfg):
        """The benchmark posture: checkpoint.save is deliberately null."""
        cfg.logger.wandb_save_dir = "/p/wandb"
        assert mod.resolve_run_artifact_dir(cfg) == "/p/wandb"

    def test_raises_when_no_artifact_dir_exists(self, mod, cfg):
        # No silent fallback to cwd/$HOME — that is a config error and must say so.
        with pytest.raises(ValueError, match="wandb_save_dir"):
            mod.resolve_run_artifact_dir(cfg)


class TestSerializeResolvedConfig:
    def test_captures_an_override_the_yaml_does_not_contain(self, mod, cfg):
        """THE POINT: a value set after the YAML merge must appear in the snapshot."""
        cfg.train.global_batch_size = 256
        text = mod.serialize_resolved_config(cfg)
        assert text is not None
        from omegaconf import OmegaConf

        assert OmegaConf.create(text).train.global_batch_size == 256

    def test_unserializable_config_reports_and_returns_none(self, mod, capsys):
        assert mod.serialize_resolved_config(object()) is None
        assert "could not serialize resolved config" in capsys.readouterr().out


class TestWriteResolvedConfig:
    def test_writes_yaml_named_by_run_id(self, mod, tmp_path, monkeypatch, cfg):
        monkeypatch.setenv("RANK", "0")
        target = tmp_path / "wandb"  # not pre-created: the writer must make it
        cfg.logger.wandb_save_dir = str(target)
        text = mod.serialize_resolved_config(cfg)

        path = mod.write_resolved_config(cfg, "20260804T120000-j42", text)

        assert path == str(target / "20260804T120000-j42.resolved-config.yaml")
        assert open(path).read() == text

    def test_only_rank_zero_writes(self, mod, tmp_path, monkeypatch, cfg):
        monkeypatch.setenv("RANK", "3")
        cfg.logger.wandb_save_dir = str(tmp_path)
        assert mod.write_resolved_config(cfg, "rid", "a: 1") is None
        assert list(tmp_path.iterdir()) == []

    def test_no_file_when_serialization_failed(self, mod, tmp_path, monkeypatch, cfg):
        monkeypatch.setenv("RANK", "0")
        cfg.logger.wandb_save_dir = str(tmp_path)
        assert mod.write_resolved_config(cfg, "rid", None) is None
        assert list(tmp_path.iterdir()) == []

    def test_unwritable_target_reports_and_does_not_raise(self, mod, tmp_path, monkeypatch, capsys, cfg):
        # Provenance must never take down a training run.
        monkeypatch.setenv("RANK", "0")
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        cfg.logger.wandb_save_dir = str(blocker)
        assert mod.write_resolved_config(cfg, "rid", "a: 1") is None
        assert "could not write resolved config" in capsys.readouterr().out

    def test_missing_artifact_dir_is_not_swallowed(self, mod, monkeypatch, cfg):
        """A config error must surface, not degrade to 'no snapshot' like an I/O failure.

        The writer catches OSError only, so resolve_run_artifact_dir's deliberate ValueError
        propagates — otherwise the function documented as raising 'rather than quietly picking
        somewhere' would be silenced by its only production caller.
        """
        monkeypatch.setenv("RANK", "0")
        with pytest.raises(ValueError, match="wandb_save_dir"):
            mod.write_resolved_config(cfg, "rid", "a: 1")
