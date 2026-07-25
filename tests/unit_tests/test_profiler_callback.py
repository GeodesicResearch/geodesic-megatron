"""Unit tests for the pure-python parts of scripts/profiling/profiler_callback.py."""

import importlib.util
import os

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def mod():
    # Import the real module by path (top-level scripts/ package, not re-implemented).
    spec = importlib.util.spec_from_file_location(
        "profiler_callback", os.path.join(REPO_ROOT, "scripts", "profiling", "profiler_callback.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _build(mod, **overrides):
    """Construct the real TorchProfilerCallback with test defaults."""
    kwargs = dict(
        out_root="/tmp/x",
        run_name="r",
        run_id="20260724T120000-j123",
        config_file=None,
        capture_iters=[10, 20],
        tag_files=True,
        ranks=[0],
        resolved_config_yaml=None,
        raw_log_path="",
    )
    kwargs.update(overrides)
    return mod.TorchProfilerCallback(**kwargs)


def test_disabled_by_default(mod, monkeypatch):
    monkeypatch.delenv("ISAMBARD_TORCH_PROFILE", raising=False)
    assert (
        mod.maybe_build_profiler_callback(
            config_file=None, run_name="x", run_id="rid", resolved_config_yaml=None, raw_log_path=""
        )
        is None
    )
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE", "0")
    assert (
        mod.maybe_build_profiler_callback(
            config_file=None, run_name="x", run_id="rid", resolved_config_yaml=None, raw_log_path=""
        )
        is None
    )


def test_enabled_with_default_root_and_legacy_wait(mod, monkeypatch):
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE", "1")
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE_RANKS", "0,4")
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE_WAIT", "5")
    monkeypatch.delenv("ISAMBARD_TORCH_PROFILE_ITERS", raising=False)
    cb = mod.maybe_build_profiler_callback(
        config_file="cfg.yaml", run_name="runA", run_id="rid1", resolved_config_yaml="a: 1\n", raw_log_path="/l.out"
    )
    assert cb is not None
    # The run ID names a per-launch subdirectory under the run-name dir.
    assert cb.out_dir == os.path.join(mod.DEFAULT_PROFILE_ROOT, "runA", "rid1")
    assert cb.ranks == [0, 4]
    # Legacy semantics: wait=W traces the single iteration W+2 (1-based),
    # with the unsuffixed rank<R> trace filename.
    assert cb.capture_iters == [7]
    assert cb.tag_files is False
    assert cb.config_file == "cfg.yaml"
    assert cb.resolved_config_yaml == "a: 1\n"
    assert cb.raw_log_path == "/l.out"


def test_enabled_with_custom_root(mod, monkeypatch, tmp_path):
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE", str(tmp_path))
    monkeypatch.delenv("ISAMBARD_TORCH_PROFILE_RANKS", raising=False)
    monkeypatch.delenv("ISAMBARD_TORCH_PROFILE_ITERS", raising=False)
    monkeypatch.delenv("ISAMBARD_TORCH_PROFILE_WAIT", raising=False)
    cb = mod.maybe_build_profiler_callback(
        config_file=None, run_name="runB", run_id="rid2", resolved_config_yaml=None, raw_log_path=""
    )
    assert cb.out_dir == str(tmp_path / "runB" / "rid2")
    assert cb.ranks == [0]
    assert cb.capture_iters == [5]  # default wait=3 -> iteration 5


def test_iters_env_wins_over_wait(mod, monkeypatch):
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE", "1")
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE_ITERS", "10,20")
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE_WAIT", "5")
    cb = mod.maybe_build_profiler_callback(
        config_file=None, run_name="runC", run_id="rid3", resolved_config_yaml=None, raw_log_path=""
    )
    assert cb.capture_iters == [10, 20]
    assert cb.tag_files is True


def test_capture_iters_are_sorted(mod):
    cb = _build(mod, capture_iters=[20, 10])
    assert cb.capture_iters == [10, 20]


def test_capture_schedule_actions(mod):
    import torch

    # Captures at 1-based iterations 10 and 20 = 0-indexed steps 9 and 19.
    action = mod._capture_schedule({9, 19})
    assert action(9) == torch.profiler.ProfilerAction.RECORD_AND_SAVE
    assert action(19) == torch.profiler.ProfilerAction.RECORD_AND_SAVE
    # The step immediately before each capture is warmup.
    assert action(8) == torch.profiler.ProfilerAction.WARMUP
    assert action(18) == torch.profiler.ProfilerAction.WARMUP
    # Everything else is off.
    for step in (0, 5, 10, 15, 20, 25):
        assert action(step) == torch.profiler.ProfilerAction.NONE


def test_capture_schedule_consecutive_iterations(mod):
    import torch

    # Back-to-back captures: RECORD_AND_SAVE takes precedence over WARMUP.
    action = mod._capture_schedule({9, 10})
    assert action(9) == torch.profiler.ProfilerAction.RECORD_AND_SAVE
    assert action(10) == torch.profiler.ProfilerAction.RECORD_AND_SAVE
    assert action(8) == torch.profiler.ProfilerAction.WARMUP


def test_trace_basename_tagging(mod):
    cb = _build(mod, capture_iters=[10, 20], tag_files=True)
    assert cb._trace_basename(0) == "rank0.iter10.chrome_trace.json"
    cb.captures_done = 1
    assert cb._trace_basename(0) == "rank0.iter20.chrome_trace.json"

    legacy = _build(mod, capture_iters=[5], tag_files=False)
    assert legacy._trace_basename(3) == "rank3.chrome_trace.json"


def test_write_provenance_snapshots_configs_and_identity(mod, tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text("train:\n  train_iters: 25\n")
    cb = _build(
        mod,
        out_root=str(tmp_path),
        run_name="runP",
        run_id="20260724T130000-j456",
        config_file=str(override),
        resolved_config_yaml="model:\n  seq_length: 32768\ntrain:\n  train_iters: 25\n",
        raw_log_path="/some/logs/slurm/train-456.out",
    )
    os.makedirs(cb.out_dir, exist_ok=True)
    cb._write_provenance(0)

    prov = (tmp_path / "runP" / "20260724T130000-j456" / "provenance.txt").read_text()
    assert "run_id: 20260724T130000-j456" in prov
    assert "raw_log_path: /some/logs/slurm/train-456.out" in prov
    assert "capture_iterations=[10, 20]" in prov
    assert "with_stack=True record_shapes=True" in prov
    out = tmp_path / "runP" / "20260724T130000-j456"
    assert (out / "config_snapshot.yaml").read_text() == override.read_text()
    assert "seq_length: 32768" in (out / "resolved_config_snapshot.yaml").read_text()


def test_write_provenance_without_raw_log_says_none(mod, tmp_path):
    cb = _build(mod, out_root=str(tmp_path), run_name="runQ", raw_log_path="")
    os.makedirs(cb.out_dir, exist_ok=True)
    cb._write_provenance(0)
    prov = (tmp_path / "runQ" / cb.run_id / "provenance.txt").read_text()
    assert "raw_log_path: (none)" in prov


def test_copy_raw_log_snapshots_and_refreshes(mod, tmp_path):
    log = tmp_path / "train-1.out"
    log.write_text("iteration 1\n")
    cb = _build(mod, out_root=str(tmp_path), run_name="runL", raw_log_path=str(log))
    os.makedirs(cb.out_dir, exist_ok=True)

    cb._copy_raw_log()
    snapshot = tmp_path / "runL" / cb.run_id / "raw_log_snapshot.out"
    assert snapshot.read_text() == "iteration 1\n"

    # The log grows; a later copy (next export / train end) refreshes the snapshot.
    log.write_text("iteration 1\niteration 2\n")
    cb._copy_raw_log()
    assert snapshot.read_text() == "iteration 1\niteration 2\n"


def test_copy_raw_log_only_first_profiled_rank(mod, tmp_path):
    log = tmp_path / "train-2.out"
    log.write_text("x\n")
    # Executing "rank" in tests is 0 (torch.distributed not initialized);
    # with ranks=[9] this process is not the first profiled rank -> skip.
    cb = _build(mod, out_root=str(tmp_path), run_name="runM", ranks=[9], raw_log_path=str(log))
    os.makedirs(cb.out_dir, exist_ok=True)
    cb._copy_raw_log()
    assert not (tmp_path / "runM" / cb.run_id / "raw_log_snapshot.out").exists()


def test_copy_raw_log_skips_when_no_path(mod, tmp_path, capsys):
    cb = _build(mod, out_root=str(tmp_path), run_name="runN", raw_log_path="")
    os.makedirs(cb.out_dir, exist_ok=True)
    cb._copy_raw_log()
    assert not (tmp_path / "runN" / cb.run_id / "raw_log_snapshot.out").exists()
    assert "skipping log snapshot" in capsys.readouterr().out


def test_copy_raw_log_missing_file_warns_not_raises(mod, tmp_path, capsys):
    cb = _build(mod, out_root=str(tmp_path), run_name="runO", raw_log_path=str(tmp_path / "gone.out"))
    os.makedirs(cb.out_dir, exist_ok=True)
    cb._copy_raw_log()  # must not raise — a failed snapshot cannot crash training
    assert "WARNING: raw-log snapshot failed" in capsys.readouterr().out


def test_repo_commit_resolves_direct_ref(mod, tmp_path):
    # Real on-disk .git layout (loose ref), no git binary involved by design.
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/feature\n")
    (git / "refs" / "heads" / "feature").write_text("abc123def456\n")
    out = mod._repo_commit(str(tmp_path))
    assert out.startswith("abc123def456")
    assert "refs/heads/feature" in out


def test_repo_commit_resolves_packed_ref(mod, tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text("# pack-refs\ncafe0123 refs/heads/main\n")
    assert mod._repo_commit(str(tmp_path)).startswith("cafe0123")


def test_repo_commit_unresolved_is_loud(mod, tmp_path):
    assert mod._repo_commit(str(tmp_path / "nogit")).startswith("UNRESOLVED")


# --- Regression tests for the 2026-07-24 review findings (export guards) ---


class _FailingProf:
    """Stands in for torch.profiler.profile in _export; export always raises.

    A real profiler can't be constructed in CPU-only CI, and the defect under
    test is the callback's own exception boundary, not torch's export.
    """

    def export_chrome_trace(self, path):
        raise OSError(122, "Disk quota exceeded")  # EDQUOT, the documented Lustre failure


def test_export_failure_degrades_not_raises(mod, tmp_path, capsys):
    cb = _build(mod, out_root=str(tmp_path))
    cb.enabled_here = True
    cb.steps_done = 10  # capture iteration 10 legitimately ran
    cb._export(_FailingProf())  # must NOT raise (would kill a 64-rank run)
    out = capsys.readouterr().out
    assert "WARNING: trace export failed" in out
    assert cb.captures_done == len(cb.capture_iters)  # further captures halted


def test_teardown_export_suppressed_when_capture_never_ran(mod, tmp_path, capsys):
    cb = _build(mod, out_root=str(tmp_path))
    cb.enabled_here = True
    cb.steps_done = 9  # train ended before capture iteration 10
    cb._export(_FailingProf())  # suppression must win before any export attempt
    out = capsys.readouterr().out
    assert "suppressing teardown export" in out
    assert "WARNING: trace export failed" not in out
    assert cb.captures_done == len(cb.capture_iters)


class _OkProf:
    """Stands in for torch.profiler.profile in _export; export writes a stub trace.

    A real profiler needs kineto/CUPTI (GPU) — the behavior under test is the
    callback's own bookkeeping, not torch's exporter.
    """

    def export_chrome_trace(self, path):
        with open(path, "w") as f:
            f.write("{}")


def test_provenance_written_by_first_profiled_rank_only(mod, tmp_path):
    # Callback believes it runs on rank 0 (torch.distributed uninitialized -> rank 0),
    # but the first PROFILED rank is 9 -> provenance must be skipped even though
    # the trace export itself succeeds.
    cb = _build(mod, out_root=str(tmp_path), ranks=[9, 0])
    cb.enabled_here = True
    cb.steps_done = 10
    os.makedirs(cb.out_dir, exist_ok=True)  # normally done by on_train_start
    cb._export(_OkProf())
    assert os.path.exists(os.path.join(cb.out_dir, "rank0.iter10.chrome_trace.json.gz"))
    assert not os.path.exists(os.path.join(cb.out_dir, "provenance.txt"))


def test_teardown_suppression_after_first_capture_succeeded(mod, tmp_path, capsys):
    # captures [10, 20], train ended after iteration 15: capture 10 already
    # exported (captures_done=1), the pending capture 20 must be suppressed —
    # exercises the capture_iters[captures_done] indexing beyond index 0.
    cb = _build(mod, out_root=str(tmp_path))
    cb.enabled_here = True
    cb.captures_done = 1
    cb.steps_done = 15
    cb._export(_FailingProf())  # suppression must win before any export attempt
    out = capsys.readouterr().out
    assert "suppressing teardown export" in out
    assert "WARNING: trace export failed" not in out
    assert cb.captures_done == len(cb.capture_iters)


# --- on_train_start / on_train_step_end state machine ------------------------


class _StubProf:
    """Stands in for torch.profiler.profile in step-end tests.

    The state machine around step()/stop() is the code under test; a real
    profiler would drag in kineto/CUPTI (GPU-only) for no extra coverage.
    """

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def step(self):
        self.calls.append("step")
        if self.fail_on == "step":
            raise RuntimeError("kineto exploded")

    def stop(self):
        self.calls.append("stop")
        if self.fail_on == "stop":
            raise RuntimeError("kineto exploded")


def test_step_end_counts_iterations_and_steps_profiler(mod):
    cb = _build(mod)
    cb.enabled_here = True
    cb.prof = _StubProf()
    cb.on_train_step_end(ctx=None)
    cb.on_train_step_end(ctx=None)
    assert cb.steps_done == 2
    assert cb.prof.calls == ["step", "step"]


def test_step_end_stops_profiler_after_all_captures(mod):
    cb = _build(mod)
    cb.enabled_here = True
    stub = _StubProf()
    cb.prof = stub
    cb.captures_done = len(cb.capture_iters)  # all captures exported
    cb.on_train_step_end(ctx=None)
    assert stub.calls == ["stop"]
    assert cb.prof is None  # later steps cost nothing


def test_step_end_failure_degrades_not_raises(mod, capsys):
    cb = _build(mod)
    cb.enabled_here = True
    cb.prof = _StubProf(fail_on="step")
    cb.on_train_step_end(ctx=None)  # must NOT raise (would kill the run)
    assert "WARNING: profiler step/stop failed" in capsys.readouterr().out
    assert cb.prof is None  # profiling disabled after the failure


def test_train_end_stop_failure_degrades_not_raises(mod, tmp_path, capsys):
    cb = _build(mod, out_root=str(tmp_path))
    cb.enabled_here = True
    cb.prof = _StubProf(fail_on="stop")
    os.makedirs(cb.out_dir, exist_ok=True)
    cb.on_train_end(ctx=None)  # must NOT raise
    assert "WARNING: profiler stop failed at train end" in capsys.readouterr().out
    assert cb.prof is None


def test_train_start_setup_failure_disables_profiling(mod, tmp_path, capsys):
    # Make os.makedirs fail by placing a FILE where out_dir's parent must be.
    blocker = tmp_path / "blocked"
    blocker.write_text("")
    cb = _build(mod, out_root=str(blocker), run_name="r", run_id="rid")
    cb.on_train_start(ctx=None)  # must NOT raise
    assert "WARNING: profiler setup failed" in capsys.readouterr().out
    assert cb.enabled_here is False
    assert cb.prof is None
