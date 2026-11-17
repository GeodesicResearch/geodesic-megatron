# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""pipeline/sh/stage.sh: the copies a launcher keeps on a node's own disk, run as the launchers run them.

Each test sources the real library into a bash under ``set -euo pipefail`` and copies real files between two
directories of ``tmp_path``.
"""

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_LIB = REPO_ROOT / "pipeline" / "sh" / "stage.sh"


def _stage(script: str, *args: object, **env: str) -> subprocess.CompletedProcess:
    """``script`` with ``args`` as ``$1...`` after sourcing the library, under the launchers' shell options."""
    return subprocess.run(
        ["bash", "-c", f'set -euo pipefail\nsource "{STAGE_LIB}"\n{script}', "harness", *map(str, args)],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], **env},
        timeout=60,
    )


def _tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def _files(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): path.read_text() for path in sorted(root.rglob("*")) if path.is_file()}


def test_stage_file_copies_into_a_directory_it_creates(tmp_path):
    src = _tree(tmp_path / "src", {"a.bin": "payload"}) / "a.bin"
    dst = tmp_path / "stage" / "deep" / "a.bin"
    result = _stage('stage_file "$1" "$2"', src, dst)
    assert result.returncode == 0, result.stderr
    assert _files(tmp_path / "stage") == {"deep/a.bin": "payload"}


def test_stage_file_follows_a_link_as_the_hub_cache_stores_its_files(tmp_path):
    blob = _tree(tmp_path / "blobs", {"abc": "weights"}) / "abc"
    link = tmp_path / "snapshot" / "model.safetensors"
    link.parent.mkdir()
    link.symlink_to(blob)
    dst = tmp_path / "stage" / "model.safetensors"
    assert _stage('stage_file "$1" "$2"', link, dst).returncode == 0
    assert dst.read_text() == "weights" and not dst.is_symlink()


def test_stage_file_reports_a_missing_source_and_leaves_no_file_at_the_path(tmp_path):
    dst = tmp_path / "stage" / "a.bin"
    result = _stage('stage_file "$1" "$2"', tmp_path / "absent", dst)
    assert result.returncode == 1
    assert f"FATAL [stage]: copying {tmp_path / 'absent'} to {dst}.incoming failed." in result.stderr
    assert not dst.exists()


def test_stage_file_refuses_a_copy_of_another_size(tmp_path):
    """dd is the copy, so a dd that writes less stands for a copy cut short."""
    src = _tree(tmp_path / "src", {"a.bin": "payload"}) / "a.bin"
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "dd").write_text(
        '#!/bin/bash\nfor word in "$@"; do case "$word" in of=*) printf cut > "${word#of=}";; esac; done\n'
    )
    (stub / "dd").chmod(0o755)
    dst = tmp_path / "stage" / "a.bin"
    result = _stage('stage_file "$1" "$2"', src, dst, PATH=f"{stub}:{os.environ['PATH']}")
    assert result.returncode == 1
    assert f"FATAL [stage]: {dst}.incoming is 3 bytes and {src} is 7." in result.stderr
    assert not dst.exists()


def test_stage_file_refuses_a_destination_that_is_a_directory(tmp_path):
    src = _tree(tmp_path / "src", {"a.bin": "payload"}) / "a.bin"
    dst = tmp_path / "stage" / "a.bin"
    dst.mkdir(parents=True)
    assert _stage('stage_file "$1" "$2"', src, dst).returncode == 1
    assert list(dst.iterdir()) == []


def test_stage_root_is_the_profiles_unless_the_environment_names_one():
    assert _stage("stage_root", CLUSTER_INPUT_STAGE_ROOT="/tmp/input_stage").stdout == "/tmp/input_stage\n"
    assert _stage("stage_root", CLUSTER_INPUT_STAGE_ROOT="").stdout == "\n"
    overridden = _stage("stage_root", CLUSTER_INPUT_STAGE_ROOT="/tmp/input_stage", GEODESIC_INPUT_STAGE_ROOT="/x")
    assert overridden.stdout == "/x\n"
    in_place = _stage("stage_root", CLUSTER_INPUT_STAGE_ROOT="/tmp/input_stage", GEODESIC_INPUT_STAGE_ROOT="")
    assert in_place.stdout == "\n"


@pytest.mark.parametrize("copies", ["1", "2", "16"])
def test_stage_tree_copies_every_file_whatever_the_number_copied_at_a_time(tmp_path, copies):
    files = {f"shard-{i}.bin": f"shard {i}" for i in range(5)} | {"sub/config.json": "{}"}
    src = _tree(tmp_path / "src", files)
    result = _stage('stage_tree "$1" "$2"', src, tmp_path / "stage", STAGE_COPIES=copies)
    assert result.returncode == 0, result.stderr
    assert _files(tmp_path / "stage") == files


def test_stage_tree_fails_when_one_copy_fails(tmp_path):
    src = _tree(tmp_path / "src", {"a.bin": "a", "b.bin": "b"})
    (src / "b.bin").chmod(0)
    result = _stage('stage_tree "$1" "$2"', src, tmp_path / "stage")
    if os.geteuid() == 0:
        pytest.skip("root reads a file of mode 0, so no copy fails")
    assert result.returncode == 1
    assert "b.bin.incoming failed" in result.stderr


def test_stage_file_keeps_the_modification_time(tmp_path):
    """A reader that checks a file against a record of its size and time sees the copy as the file."""
    src = _tree(tmp_path / "src", {"training.jsonl": "{}"}) / "training.jsonl"
    os.utime(src, (1_700_000_000, 1_700_000_000))
    dst = tmp_path / "stage" / "training.jsonl"
    assert _stage('stage_file "$1" "$2"', src, dst).returncode == 0
    assert dst.stat().st_mtime == 1_700_000_000


@pytest.mark.parametrize("function", ["stage_tree", "stage_out"])
def test_a_link_to_nothing_stops_the_copy_before_any_file(tmp_path, function):
    """A missing file must not go unnoticed: the copy stands for the whole directory."""
    src = _tree(tmp_path / "src", {"config.json": "{}"})
    (src / "model.safetensors").symlink_to(tmp_path / "blobs" / "absent")
    result = _stage(f'{function} "$1" "$2"', src, tmp_path / "dst")
    assert result.returncode == 1
    assert f"FATAL [stage]: {src / 'model.safetensors'} is a link to nothing." in result.stderr
    assert not (tmp_path / "dst").exists()


@pytest.mark.parametrize("function", ["stage_tree", "stage_out"])
def test_a_listing_that_fails_fails_the_copy(tmp_path, function):
    """find lists the files, so a find that fails after one name stands for a listing cut short."""
    src = _tree(tmp_path / "src", {"a.bin": "a", "b.bin": "b"})
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "find").write_text("#!/bin/bash\nprintf 'a.bin\\0'\nexit 1\n")
    (stub / "find").chmod(0o755)
    result = _stage(f'{function} "$1" "$2"', src, tmp_path / "dst", PATH=f"{stub}:{os.environ['PATH']}")
    assert result.returncode == 1
    assert f"FATAL [stage]: listing the files under {src} failed." in result.stderr
    assert not (tmp_path / "dst").exists()


def test_stage_tree_refuses_a_source_that_is_no_directory(tmp_path):
    result = _stage('stage_tree "$1" "$2"', tmp_path / "absent", tmp_path / "stage")
    assert result.returncode == 1
    assert f"FATAL [stage]: {tmp_path / 'absent'} is not a directory." in result.stderr


def test_stage_out_copies_what_the_payload_wrote_and_leaves_the_inputs(tmp_path):
    stage = _tree(tmp_path / "stage", {"training.jsonl": "input", "out.bin": "tokens", "packed/a/pack.parquet": "p"})
    real = _tree(tmp_path / "real", {"training.jsonl": "input", "kept.txt": "earlier"})
    result = _stage('stage_out "$1" "$2" training.jsonl', stage, real)
    assert result.returncode == 0, result.stderr
    assert _files(real) == {
        "training.jsonl": "input",
        "kept.txt": "earlier",
        "out.bin": "tokens",
        "packed/a/pack.parquet": "p",
    }


def test_stage_out_refuses_a_stage_that_is_no_directory(tmp_path):
    result = _stage('stage_out "$1" "$2"', tmp_path / "absent", tmp_path / "real")
    assert result.returncode == 1
    assert f"FATAL [stage]: {tmp_path / 'absent'} is not a directory." in result.stderr


def test_stage_bind_adds_to_the_binds_apptainer_reads(tmp_path):
    result = _stage('stage_bind /tmp/s/a /shared/a; stage_bind /tmp/s/b /shared/b; bash -c "echo \\$APPTAINER_BIND"')
    assert (result.returncode, result.stdout) == (0, "/tmp/s/a:/shared/a,/tmp/s/b:/shared/b\n")
    result = _stage('stage_bind /tmp/s/a /shared/a; echo "$APPTAINER_BIND"', APPTAINER_BIND="/x:/y")
    assert result.stdout == "/x:/y,/tmp/s/a:/shared/a\n"
