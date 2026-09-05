"""Unit tests for the pure-python helpers behind `pipeline_env_validate.py --container`."""

import importlib.util
import os
import sys

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def validate_mod():
    # The validator is a top-level script (not part of the megatron.bridge
    # package), so load the real file by path — no re-implementation.
    spec = importlib.util.spec_from_file_location(
        "pipeline_env_validate", os.path.join(REPO_ROOT, "pipeline_env_validate.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_under_its_own_dir(validate_mod):
    import argparse as real_module

    root = os.path.dirname(real_module.__file__)
    assert validate_mod.module_file_is_under(real_module, root)


def test_module_not_under_unrelated_dir(validate_mod, tmp_path):
    import argparse as real_module

    assert not validate_mod.module_file_is_under(real_module, str(tmp_path))


def test_prefix_collision_is_not_containment(validate_mod, tmp_path):
    # /a/bc must not count as inside /a/b (string-prefix trap the os.sep guard covers).
    parent = tmp_path / "pkg"
    sibling = tmp_path / "pkgextra"
    parent.mkdir()
    sibling.mkdir()
    mod_file = sibling / "m.py"
    mod_file.write_text("X = 1\n")
    spec = importlib.util.spec_from_file_location("collision_probe", mod_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not validate_mod.module_file_is_under(mod, str(parent))
    assert validate_mod.module_file_is_under(mod, str(sibling))


def test_namespace_package_uses_path_entries(validate_mod, tmp_path):
    # A real PEP 420 namespace package (no __init__.py) — the shape megatron/
    # has in this repo — resolves via __path__ when __file__ is absent.
    ns_root = tmp_path / "nsroot"
    (ns_root / "nspkg_probe").mkdir(parents=True)
    sys.path.insert(0, str(ns_root))
    try:
        import nspkg_probe  # real namespace-package import, not a fake

        assert nspkg_probe.__file__ is None or not hasattr(nspkg_probe, "__file__")
        assert validate_mod.module_file_is_under(nspkg_probe, str(ns_root))
        assert not validate_mod.module_file_is_under(nspkg_probe, str(tmp_path / "elsewhere"))
    finally:
        sys.path.remove(str(ns_root))
        sys.modules.pop("nspkg_probe", None)


def test_check_imports_records_one_pass_per_spec(validate_mod):
    """Each table entry becomes exactly one PASS stage with its own label."""
    before = len(validate_mod.results)
    validate_mod.check_imports(
        [
            ("json (probe)", ("json",), "stdlib"),
            ("math (probe)", ("math",), "stdlib"),
        ]
    )
    added = validate_mod.results[before:]
    assert [(name, ok) for name, ok, _ in added] == [("json (probe)", True), ("math (probe)", True)]


def test_check_imports_records_failure_without_raising(validate_mod):
    """A bogus module yields a FAIL entry (the stage wrapper catches), not an exception.

    The passing entry alongside it pins the modules=modules late-binding guard in
    the per-spec closure: without the guard both closures would import the LAST
    spec's modules, flipping the first entry to FAIL too.
    """
    before = len(validate_mod.results)
    validate_mod.check_imports(
        [
            ("ok (probe)", ("json",), "stdlib"),
            ("bogus (probe)", ("definitely_not_a_real_module_xyz",), "n/a"),
        ]
    )
    added = validate_mod.results[before:]
    assert [(name, ok) for name, ok, _ in added] == [("ok (probe)", True), ("bogus (probe)", False)]
    assert "definitely_not_a_real_module_xyz" in added[1][2]


def test_check_imports_multi_module_entry_imports_all(validate_mod):
    """A multi-module tuple fails if ANY member is unimportable (guards the closure loop)."""
    before = len(validate_mod.results)
    validate_mod.check_imports([("pair (probe)", ("json", "definitely_not_a_real_module_xyz"), "n/a")])
    name, ok, detail = validate_mod.results[before]
    assert ok is False and "definitely_not_a_real_module_xyz" in detail


class TestCheckOmpThreading:
    """The scored OpenMP-defaults check (`check_omp_threading`) — every branch.

    The check exists because torchrun silently pins OMP_NUM_THREADS=1 when the variable
    is absent — worth up to ~1.43 s/iter on the 120B benchmark, though that particular
    pair also differed in offload fraction, so treat it as an upper bound rather than a
    clean threading delta. It went unnoticed for months, so its failure modes must stay loud. The function is wrapped by @stage, which
    RECORDS pass/fail into `validate_mod.results` instead of raising — the same
    contract the check_imports tests above assert against. It reads os.environ
    directly; monkeypatch gives each case a clean environment.
    """

    def _run(self, validate_mod, monkeypatch, env):
        for var in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "ISAMBARD_OMP_WAIT_POLICY"):
            monkeypatch.delenv(var, raising=False)
        for var, val in env.items():
            monkeypatch.setenv(var, val)
        before = len(validate_mod.results)
        validate_mod.check_omp_threading()
        name, ok, detail = validate_mod.results[before]
        assert name == "host OpenMP threading defaults"
        return ok, detail

    def test_unset_fails(self, validate_mod, monkeypatch):
        ok, detail = self._run(validate_mod, monkeypatch, {})
        assert ok is False and "not set" in detail

    def test_non_numeric_fails(self, validate_mod, monkeypatch):
        ok, detail = self._run(validate_mod, monkeypatch, {"OMP_NUM_THREADS": "PASSIVE"})
        assert ok is False and "not a positive integer" in detail

    def test_zero_fails(self, validate_mod, monkeypatch):
        ok, detail = self._run(validate_mod, monkeypatch, {"OMP_NUM_THREADS": "0"})
        assert ok is False and "not a positive integer" in detail

    def test_single_thread_passes_without_policy(self, validate_mod, monkeypatch):
        # =1 is the torchrun-compatible opt-out; PASSIVE is only load-bearing above 1.
        ok, _ = self._run(validate_mod, monkeypatch, {"OMP_NUM_THREADS": "1"})
        assert ok is True

    def test_threaded_with_passive_passes(self, validate_mod, monkeypatch):
        ok, _ = self._run(validate_mod, monkeypatch, {"OMP_NUM_THREADS": "8", "OMP_WAIT_POLICY": "PASSIVE"})
        assert ok is True

    def test_threaded_without_passive_fails(self, validate_mod, monkeypatch):
        # 8 spinning ACTIVE threads compete with NCCL progress + the dataloader for
        # Grace cores — the untested-and-plausibly-worse posture the check refuses.
        ok, detail = self._run(validate_mod, monkeypatch, {"OMP_NUM_THREADS": "8"})
        assert ok is False and "PASSIVE" in detail

    def test_explicit_policy_override_passes(self, validate_mod, monkeypatch):
        # ISAMBARD_OMP_WAIT_POLICY set = the operator chose the policy deliberately;
        # the check accepts any explicit choice and only rejects the silent default.
        ok, _ = self._run(
            validate_mod,
            monkeypatch,
            {
                "OMP_NUM_THREADS": "8",
                "OMP_WAIT_POLICY": "ACTIVE",
                "ISAMBARD_OMP_WAIT_POLICY": "ACTIVE",
            },
        )
        assert ok is True


class TestCheckHfDatasetsCache:
    """The scored datasets-cache writability check (`check_hf_datasets_cache`).

    `datasets` creates its cache tree mode 0755, so a directory shared between accounts
    is writable only by whoever created it. Every other account then fails at lock
    acquisition, deep inside a download that has already run — which is why this is a
    launch-time check rather than a comment. Like the OpenMP check it is wrapped by
    @stage, so failures are RECORDED in `validate_mod.results` rather than raised.
    """

    def _run(self, validate_mod, monkeypatch, cache):
        monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
        if cache is not None:
            monkeypatch.setenv("HF_DATASETS_CACHE", str(cache))
        before = len(validate_mod.results)
        validate_mod.check_hf_datasets_cache()
        name, ok, detail = validate_mod.results[before]
        assert name == "HF datasets cache is writable"
        return ok, detail

    def test_unset_fails(self, validate_mod, monkeypatch):
        ok, detail = self._run(validate_mod, monkeypatch, None)
        assert ok is False and "not set" in detail

    def test_writable_dir_passes(self, validate_mod, monkeypatch, tmp_path):
        ok, _ = self._run(validate_mod, monkeypatch, tmp_path)
        assert ok is True

    def test_missing_but_creatable_dir_passes(self, validate_mod, monkeypatch, tmp_path):
        # The check mkdirs its parent, so a not-yet-populated cache is a pass —
        # first use on a fresh account must not be reported as a failure.
        ok, _ = self._run(validate_mod, monkeypatch, tmp_path / "not-created-yet")
        assert ok is True

    def test_unwritable_dir_fails(self, validate_mod, monkeypatch, tmp_path):
        # Mode 0555 reproduces the real failure: the tree exists, owned by another
        # account, and this one cannot create the lock file inside it.
        foreign = tmp_path / "owned-by-someone-else"
        foreign.mkdir()
        foreign.chmod(0o555)
        try:
            ok, detail = self._run(validate_mod, monkeypatch, foreign)
        finally:
            foreign.chmod(0o755)
        assert ok is False
        assert "not writable" in detail and "GEODESIC_HF_DATASETS_CACHE" in detail

    def test_probe_file_is_cleaned_up(self, validate_mod, monkeypatch, tmp_path):
        self._run(validate_mod, monkeypatch, tmp_path)
        assert list(tmp_path.iterdir()) == []
