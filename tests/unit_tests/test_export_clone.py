# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for scripts/checkpoint/export_clone.py, the torch_grouped export repair that the exporter and the Hub
publisher share. Everything here is real: tiny checkpoint directory trees on disk, real links, and the module's own
command line run as a subprocess. Nothing is mocked."""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.checkpoint import export_clone

from tests.unit_tests.export_clone_fixtures import (
    GPT_RUN_CONFIG,
    NEW_IMPL,
    NEW_TARGET,
    OLD_IMPL,
    OLD_TARGET,
    TE_GROUPED_RUN_CONFIG,
    make_checkpoint,
    snapshot,
    torch_grouped_run_config,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _REPO_ROOT / "scripts" / "checkpoint" / "export_clone.py"


@pytest.fixture()
def torch_grouped(tmp_path):
    """A torch_grouped checkpoint with two saves, the tracker at the second, and a clone root beside it."""
    save = tmp_path / "checkpoints" / "masked"
    make_checkpoint(save, [100, 477], tracker=477, run_config=torch_grouped_run_config(tmp_path / "checkpoints"))
    return save, tmp_path / "clones"


# ----------------------------------------------------------------------------------------------
# The run_config edits


def test_patch_run_config_applies_both_edits_and_accepts_an_already_patched_file():
    raw = torch_grouped_run_config(Path("/save"))
    patched = export_clone.patch_run_config(raw)
    assert NEW_TARGET in patched and NEW_IMPL in patched
    assert OLD_TARGET not in patched and OLD_IMPL not in patched
    assert patched.replace(NEW_TARGET, OLD_TARGET).replace(NEW_IMPL, OLD_IMPL) == raw, "only the two edits change"
    assert export_clone.patch_run_config(patched) == patched


@pytest.mark.parametrize(
    "text",
    [
        "model:\n  nothing: here\n",
        # The closure twice: an unrecognised shape is refused, never half-edited.
        f"a:\n  _target_: {OLD_TARGET}\nb:\n  _target_: {OLD_TARGET}\nmodel:\n  {OLD_IMPL}\n",
        # The closure from a grouped backend this repair does not name (no torch_grouped line to edit).
        f"model:\n  mamba_stack_spec:\n    _target_: {OLD_TARGET}\n  moe_experts_impl: cublas_grouped\n",
    ],
    ids=["no-fields", "closure-twice", "cublas-grouped"],
)
def test_patch_run_config_refuses_a_shape_it_does_not_recognise(text):
    with pytest.raises(export_clone.ExportError, match="expected exactly one"):
        export_clone.patch_run_config(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (torch_grouped_run_config(Path("/save")), True),
        (f"model:\n  {OLD_IMPL}\n", True),
        (TE_GROUPED_RUN_CONFIG, False),
        (GPT_RUN_CONFIG, False),
    ],
    ids=["torch-grouped", "impl-only", "te-grouped", "gpt"],
)
def test_needs_repair_is_true_exactly_when_the_config_holds_what_the_edits_replace(text, expected):
    assert export_clone.needs_repair(text) is expected


def test_a_checkpoint_without_a_run_config_needs_no_repair(tmp_path):
    save = make_checkpoint(tmp_path / "ckpt", [5], tracker=5, run_config=None)
    assert export_clone.checkpoint_needs_repair(save / "iter_0000005") is False


# ----------------------------------------------------------------------------------------------
# Resolving the iteration (moved here from pipeline_checkpoint_convert_hf.py so that prepare and the
# conversion resolve the same way)


def test_resolve_checkpoint_path_takes_the_given_iteration_then_the_tracker_then_the_newest(tmp_path):
    save = make_checkpoint(tmp_path / "ckpt", [100, 200, 300], tracker=200, run_config=GPT_RUN_CONFIG)
    assert export_clone.resolve_checkpoint_path(save, 100) == (save / "iter_0000100", 100)
    assert export_clone.resolve_checkpoint_path(save) == (save / "iter_0000200", 200)
    (save / export_clone.LATEST_FILE).unlink()
    assert export_clone.resolve_checkpoint_path(str(save)) == (save / "iter_0000300", 300)
    with pytest.raises(FileNotFoundError, match="Iteration directory not found"):
        export_clone.resolve_checkpoint_path(save, 999)
    with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
        export_clone.resolve_checkpoint_path(tmp_path / "absent")


# ----------------------------------------------------------------------------------------------
# Building a clone


def test_make_export_clone_links_the_files_and_patches_the_run_config(torch_grouped, tmp_path):
    save, _ = torch_grouped
    source = save / "iter_0000477"
    (source / "hf").mkdir()
    (source / "hf" / "config.json").write_text("{}")
    clone = tmp_path / "elsewhere" / "masked" / "iter_0000477"
    export_clone.make_export_clone(source, clone)
    assert (clone / "__0_0.distcp").is_symlink() and (clone / "__0_0.distcp").resolve() == source / "__0_0.distcp"
    assert (clone / ".metadata").resolve() == source / ".metadata"
    assert not (clone / "run_config.yaml").is_symlink()
    assert (clone / "run_config.yaml").read_text() == export_clone.patch_run_config(
        (source / "run_config.yaml").read_text()
    )
    assert not (clone / "hf").exists(), "the source's own export is not linked into the clone"
    assert (clone.parent / export_clone.LATEST_FILE).read_text().strip() == "477"
    export_clone.make_export_clone(source, clone)  # a second call is a no-op, not an error


def test_make_export_clone_refuses_a_file_that_is_not_its_link(torch_grouped, tmp_path):
    save, _ = torch_grouped
    clone = tmp_path / "elsewhere" / "iter_0000477"
    clone.mkdir(parents=True)
    (clone / "__0_0.distcp").write_bytes(b"not a link")
    with pytest.raises(export_clone.ExportError, match="is not a link to"):
        export_clone.make_export_clone(save / "iter_0000477", clone)


def test_make_export_clone_refuses_a_source_without_a_run_config(tmp_path):
    save = make_checkpoint(tmp_path / "ckpt", [5], tracker=5, run_config=None)
    with pytest.raises(export_clone.ExportError, match="has no run_config.yaml"):
        export_clone.make_export_clone(save / "iter_0000005", tmp_path / "clone" / "iter_0000005")


def test_clone_location_separates_runs_that_share_a_name_and_refuses_a_job_that_is_not_a_name(tmp_path):
    first = make_checkpoint(tmp_path / "a" / "masked", [7], tracker=7, run_config=GPT_RUN_CONFIG)
    second = make_checkpoint(tmp_path / "b" / "masked", [7], tracker=7, run_config=GPT_RUN_CONFIG)
    root = tmp_path / "clones"
    one = export_clone.clone_location(root, first / "iter_0000007", "123-45")
    two = export_clone.clone_location(root, second / "iter_0000007", "123-45")
    digest = hashlib.sha256(str(first.resolve()).encode()).hexdigest()[:12]
    assert one == root / "123-45" / f"masked-{digest}" / "iter_0000007"
    assert one != two
    for job in ("../escape", "a/b", "..", ""):
        with pytest.raises(export_clone.ExportError, match="plain path component"):
            export_clone.clone_location(root, first / "iter_0000007", job)


# ----------------------------------------------------------------------------------------------
# prepare: the exporter's decision


def test_prepare_builds_a_clone_for_a_torch_grouped_checkpoint_and_never_writes_the_checkpoint(torch_grouped):
    save, root = torch_grouped
    before = snapshot(save)
    source = export_clone.prepare_export_source(save, None, root, "4242-7")
    assert source.iter_path == save / "iter_0000477" and source.iteration == 477, "the tracker names the latest"
    assert source.clone == export_clone.clone_location(root, save / "iter_0000477", "4242-7")
    assert source.load_path == source.clone
    assert source.clone.is_relative_to(root)
    export_clone.check_is_export_clone(source.iter_path, source.clone)
    assert snapshot(save) == before, "the training tree is only read"


def test_prepare_takes_the_iteration_it_is_given(torch_grouped):
    save, root = torch_grouped
    source = export_clone.prepare_export_source(save, 100, root, "1")
    assert source.iteration == 100 and source.clone.name == "iter_0000100"


@pytest.mark.parametrize(
    "run_config", [TE_GROUPED_RUN_CONFIG, GPT_RUN_CONFIG, None], ids=["te-grouped", "gpt", "none"]
)
def test_prepare_loads_any_other_checkpoint_from_itself_and_writes_nothing(tmp_path, run_config):
    save = make_checkpoint(tmp_path / "ckpt", [3], tracker=3, run_config=run_config)
    root = tmp_path / "clones"
    source = export_clone.prepare_export_source(save, None, root, "1")
    assert source.load_path == source.iter_path == save / "iter_0000003"
    assert source.clone is None
    assert not root.exists()


def test_prepare_refuses_a_clone_root_under_home(torch_grouped, tmp_path, monkeypatch):
    save, _ = torch_grouped
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(export_clone.ExportError, match=r"under \$HOME"):
        export_clone.prepare_export_source(save, None, home / "clones", "1")
    assert not (home / "clones").exists()


def test_prepare_refuses_a_clone_root_inside_the_checkpoint_or_a_relative_one(torch_grouped):
    save, _ = torch_grouped
    before = snapshot(save)
    with pytest.raises(export_clone.ExportError, match="inside the checkpoint directory"):
        export_clone.prepare_export_source(save, None, save / "clones", "1")
    with pytest.raises(export_clone.ExportError, match="not an absolute path"):
        export_clone.prepare_export_source(save, None, Path("relative/clones"), "1")
    assert snapshot(save) == before


def test_prepare_never_writes_through_a_link_in_the_clone(torch_grouped):
    """A clone whose run_config is a link to the checkpoint's own would rewrite the training run_config."""
    save, root = torch_grouped
    clone = export_clone.clone_location(root, save / "iter_0000477", "1")
    clone.mkdir(parents=True)
    (clone / "run_config.yaml").symlink_to(save / "iter_0000477" / "run_config.yaml")
    before = snapshot(save)
    with pytest.raises(export_clone.ExportError, match="is a link"):
        export_clone.prepare_export_source(save, None, root, "1")
    assert snapshot(save) == before


# ----------------------------------------------------------------------------------------------
# check_is_export_clone: what the conversion requires of --load-path


def test_check_is_export_clone_accepts_a_clone_and_refuses_anything_else(torch_grouped, tmp_path):
    save, root = torch_grouped
    source = export_clone.prepare_export_source(save, 477, root, "1")
    iter_path, clone = source.iter_path, source.clone
    export_clone.check_is_export_clone(iter_path, clone)

    # A clone of another iteration of the same run.
    other = export_clone.prepare_export_source(save, 100, root, "2").clone
    with pytest.raises(export_clone.ExportError, match="different iteration"):
        export_clone.check_is_export_clone(iter_path, other)

    # A clone of a same-named iteration of another run.
    elsewhere = make_checkpoint(
        tmp_path / "other_run" / "masked", [477], tracker=477, run_config=torch_grouped_run_config(tmp_path)
    )
    foreign = export_clone.prepare_export_source(elsewhere, 477, root, "3").clone
    with pytest.raises(export_clone.ExportError, match="links to other files"):
        export_clone.check_is_export_clone(iter_path, foreign)

    # The checkpoint itself, whose run_config is not the repaired one.
    with pytest.raises(export_clone.ExportError, match="is not a link"):
        export_clone.check_is_export_clone(iter_path, iter_path)

    (clone / "__1_0.distcp").unlink()
    with pytest.raises(export_clone.ExportError, match=r"missing links \['__1_0.distcp'\]"):
        export_clone.check_is_export_clone(iter_path, clone)
    (clone / "__1_0.distcp").symlink_to((iter_path / "__1_0.distcp").resolve())

    (clone / "run_config.yaml").write_text((iter_path / "run_config.yaml").read_text())
    with pytest.raises(export_clone.ExportError, match="with the export edits applied"):
        export_clone.check_is_export_clone(iter_path, clone)


# ----------------------------------------------------------------------------------------------
# Removing a clone


def test_remove_export_clone_removes_the_clone_and_its_directories_and_leaves_the_checkpoint(torch_grouped):
    save, root = torch_grouped
    before = snapshot(save)
    clone = export_clone.prepare_export_source(save, None, root, "77-1").clone
    export_clone.remove_export_clone(clone, root)
    assert root.is_dir() and list(root.iterdir()) == [], "the root stays; everything under it for this export goes"
    assert snapshot(save) == before, "removing links never touches what they name"


def test_remove_export_clone_keeps_the_tracker_another_clone_shares(torch_grouped):
    save, root = torch_grouped
    first = export_clone.prepare_export_source(save, 100, root, "1").clone
    second = export_clone.prepare_export_source(save, 477, root, "1").clone
    export_clone.remove_export_clone(second, root)
    assert first.is_dir() and (first.parent / export_clone.LATEST_FILE).is_file()
    export_clone.remove_export_clone(first, root)
    assert list(root.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root removes a directory from a parent it cannot write")
def test_remove_export_clone_raises_when_a_directory_above_the_clone_cannot_be_removed(torch_grouped):
    """Only a directory that still holds another clone is left in place; any other failure to remove one is raised,
    never taken for that."""
    save, root = torch_grouped
    clone = export_clone.prepare_export_source(save, None, root, "1").clone
    job_dir = clone.parent.parent
    job_dir.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            export_clone.remove_export_clone(clone, root)
    finally:
        job_dir.chmod(0o755)
    assert not clone.exists() and clone.parent.is_dir(), "the clone is gone; the directory it could not remove stays"


def test_remove_export_clone_refuses_a_clone_holding_output(torch_grouped):
    save, root = torch_grouped
    clone = export_clone.prepare_export_source(save, None, root, "1").clone
    (clone / "hf").mkdir()
    (clone / "hf" / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(export_clone.ExportError, match=r"holds \['hf'\]"):
        export_clone.remove_export_clone(clone, root)
    assert (clone / "__0_0.distcp").is_symlink() and (clone / "run_config.yaml").is_file(), "nothing was removed"


def test_remove_export_clone_refuses_a_directory_outside_the_root(torch_grouped, tmp_path):
    save, root = torch_grouped
    root.mkdir()
    before = snapshot(save)
    with pytest.raises(export_clone.ExportError, match="outside the export clone root"):
        export_clone.remove_export_clone(save / "iter_0000477", root)
    with pytest.raises(export_clone.ExportError, match="outside the export clone root"):
        export_clone.remove_export_clone(root, root)
    assert snapshot(save) == before


# ----------------------------------------------------------------------------------------------
# The command line the launcher runs


def run_tool(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_TOOL), *args], capture_output=True, text=True, timeout=60)


def test_cli_prepare_prints_only_its_plan_on_stdout_and_remove_removes_the_clone(torch_grouped):
    save, root = torch_grouped
    result = run_tool("prepare", "--megatron-path", str(save), "--clone-root", str(root), "--job", "9-9")
    assert result.returncode == 0, result.stderr
    clone = export_clone.clone_location(root, save / "iter_0000477", "9-9")
    assert result.stdout.splitlines() == ["EXPORT_ITERATION=477", f"EXPORT_LOAD_PATH={clone}"]
    assert "torch_grouped" in result.stderr
    export_clone.check_is_export_clone(save / "iter_0000477", clone)

    result = run_tool("remove", "--clone", str(clone), "--clone-root", str(root))
    assert result.returncode == 0, result.stderr
    assert not clone.exists()


def test_cli_prepare_prints_an_empty_load_path_for_a_checkpoint_that_needs_no_repair(tmp_path):
    save = make_checkpoint(tmp_path / "ckpt", [12], tracker=None, run_config=TE_GROUPED_RUN_CONFIG)
    result = run_tool(
        "prepare", "--megatron-path", str(save), "--iteration", "12", "--clone-root", str(tmp_path / "c"), "--job", "1"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["EXPORT_ITERATION=12", "EXPORT_LOAD_PATH="]


def test_cli_reports_a_failure_with_exit_status_1_and_no_plan(tmp_path):
    result = run_tool(
        "prepare", "--megatron-path", str(tmp_path / "absent"), "--clone-root", str(tmp_path), "--job", "1"
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Checkpoint directory not found" in result.stderr


# ----------------------------------------------------------------------------------------------
# One repair, two users


def test_the_publisher_uses_this_repair_rather_than_a_copy_of_it():
    tool_dir = str(_REPO_ROOT / "scripts" / "hub")
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)
    publish_models = importlib.import_module("publish_models")
    assert publish_models.RUN_CONFIG_EDITS is export_clone.RUN_CONFIG_EDITS
    assert publish_models.patch_run_config is export_clone.patch_run_config
    assert publish_models.make_export_clone is export_clone.make_export_clone
    assert publish_models.ExportError is export_clone.ExportError
