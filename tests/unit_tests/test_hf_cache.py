# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""pipeline/hf_cache.py's command line, as the import launcher runs it.

Each test runs ``python -m pipeline.hf_cache`` as a process of its own, because huggingface_hub reads the cache
location and the offline switch when it is imported. The Hub is a network service, so the switch is off: the real
``snapshot_download`` then answers from a hub cache the test built.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("huggingface_hub")

REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION = "a" * 40


def _model_dir(hub: Path, model: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pipeline.hf_cache", "model-dir", model],
        cwd=REPO_ROOT,
        env={**os.environ, "HF_HUB_CACHE": str(hub), "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_hub_model_gives_the_snapshot_of_the_revision_main_names_and_nothing_else(tmp_path):
    repo = tmp_path / "hub" / "models--org--model"
    (repo / "snapshots" / REVISION).mkdir(parents=True)
    (repo / "snapshots" / REVISION / "config.json").write_text("{}")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(REVISION)
    result = _model_dir(tmp_path / "hub", "org/model")
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{repo / 'snapshots' / REVISION}\n"


def test_a_model_directory_gives_itself_resolved(tmp_path):
    model = tmp_path / "models" / "export"
    model.mkdir(parents=True)
    (tmp_path / "link").symlink_to(model)
    result = _model_dir(tmp_path / "hub", str(tmp_path / "link"))
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{model.resolve()}\n"


def test_a_repo_the_cache_does_not_hold_is_an_error_when_the_hub_is_off(tmp_path):
    result = _model_dir(tmp_path / "hub", "org/model")
    assert result.returncode != 0
    assert result.stdout == ""
