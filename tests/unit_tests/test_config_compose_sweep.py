# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Every YAML file under ``configs/`` composes through its ``base_config`` chain.

``load_composed_yaml`` (scripts/training/config_compose.py) reads a file as the launcher reads it. It raises for a
missing base, a looping chain, a duplicate key and a top level that is not a mapping. It also raises for a relocation
map that it cannot apply. This module runs it on every YAML file under ``configs/``. The run needs no recipe, no
network and no GPU.

There is one test per YAML file. Each file is composed in its own test, so collection only lists the files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.training.config_compose import BASE_CONFIG_KEY, RELOCATE_KEY, load_composed_yaml, parse_yaml_mapping


pytestmark = pytest.mark.cpu


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
CONFIG_PATHS = sorted(path for path in CONFIGS_DIR.rglob("*") if path.suffix in (".yaml", ".yml") and path.is_file())


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=_relative)
def test_config_composes(path):
    composed = load_composed_yaml(path)
    assert not {BASE_CONFIG_KEY, RELOCATE_KEY} & composed.keys()


def test_the_sweep_reads_overlays_and_relocated_bases():
    """The sweep reads at least one file with each key, so it composes overlays and relocated bases."""
    for key in (BASE_CONFIG_KEY, RELOCATE_KEY):
        named = (key in parse_yaml_mapping(path.read_text(), str(path)) for path in CONFIG_PATHS)
        assert any(named), f"no YAML file under configs/ has a {key} key"
