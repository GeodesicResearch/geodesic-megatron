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


def test_disabled_by_default(mod, monkeypatch):
    monkeypatch.delenv("ISAMBARD_TORCH_PROFILE", raising=False)
    assert mod.maybe_build_profiler_callback(config_file=None, run_name="x") is None
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE", "0")
    assert mod.maybe_build_profiler_callback(config_file=None, run_name="x") is None


def test_enabled_with_default_root_and_knobs(mod, monkeypatch):
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE", "1")
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE_RANKS", "0,4")
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE_WAIT", "5")
    cb = mod.maybe_build_profiler_callback(config_file="cfg.yaml", run_name="runA")
    assert cb is not None
    assert cb.out_dir == os.path.join(mod.DEFAULT_PROFILE_ROOT, "runA")
    assert cb.ranks == [0, 4]
    assert cb.wait_iters == 5
    assert cb.config_file == "cfg.yaml"


def test_enabled_with_custom_root(mod, monkeypatch, tmp_path):
    monkeypatch.setenv("ISAMBARD_TORCH_PROFILE", str(tmp_path))
    monkeypatch.delenv("ISAMBARD_TORCH_PROFILE_RANKS", raising=False)
    cb = mod.maybe_build_profiler_callback(config_file=None, run_name="runB")
    assert cb.out_dir == str(tmp_path / "runB")
    assert cb.ranks == [0]


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
