# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The YAML files under ``configs/``, for the tests that run over each of them.

Catalogue:
    CONFIGS_DIR: the ``configs/`` directory of the checkout.
    CONFIG_PATHS: each YAML file under CONFIGS_DIR, sorted.
    repo_relative: a path of the checkout as a POSIX string relative to the repo root.
"""

from pathlib import Path

from tests.unit_tests.child_process import REPO_ROOT


CONFIGS_DIR = REPO_ROOT / "configs"
CONFIG_PATHS = sorted(path for path in CONFIGS_DIR.rglob("*") if path.suffix in (".yaml", ".yml") and path.is_file())


def repo_relative(path: Path) -> str:
    """``path`` relative to the repo root, as a POSIX string."""
    return path.relative_to(REPO_ROOT).as_posix()
