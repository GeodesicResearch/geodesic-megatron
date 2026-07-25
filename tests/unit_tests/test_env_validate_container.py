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
