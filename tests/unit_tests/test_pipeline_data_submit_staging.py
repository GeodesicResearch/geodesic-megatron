# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""pipeline_data_submit.sbatch in a job with an input stage root: the tokenize and pack steps read copies.

The real sbatch script runs as a subprocess with the real shell libraries. The Apptainer container is the boundary
that cannot run here, so the runner is a stub that stands for it: it prints the payload and the binds it was
given, and writes the files in ``$STUB_WRITES`` at the dataset root as the container sees it. That is the bound copy
when a bind covers the root, and the root itself otherwise.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.unit_tests.data_submit_stub import REPO_ROOT, stub_checkout


_RUNNER = """\
#!/bin/bash
printf 'PAYLOAD:%s\\n' "$1"
printf 'BIND:%s\\n' "${APPTAINER_BIND:-}"
root="$STUB_ROOT"
IFS=, read -r -a pairs <<< "${APPTAINER_BIND:-}"
for pair in "${pairs[@]}"; do
    [ "${pair#*:}" != "$STUB_ROOT" ] || root="${pair%%:*}"
done
for name in $STUB_WRITES; do
    mkdir -p "$(dirname "$root/$name")" && echo "written by the tool" > "$root/$name" || exit 1
done
exit "${STUB_STATUS:-0}"
"""


@pytest.fixture()
def staged(tmp_path):
    """A stub checkout, a dataset root with its training JSONL, and the environment of a job with a stage root."""
    stub_repo = stub_checkout(tmp_path / "stub_repo", _RUNNER)
    root = tmp_path / "data" / "org__corpus"
    root.mkdir(parents=True)
    (root / "training.jsonl").write_text('{"input": "hello"}\n')
    stage = tmp_path / "stage"
    env = dict(os.environ)
    # Verda's profile: its quota report is `df`, which every host has. The stage root is the test's own.
    env.update(GEODESIC_CLUSTER="verda", GEODESIC_REPO_DIR=str(stub_repo), GEODESIC_INPUT_STAGE_ROOT=str(stage))
    env["STUB_ROOT"] = str(root)
    env.pop("SLURM_JOB_ID", None)
    env.pop("APPTAINER_BIND", None)
    return root, Path(f"{stage}{root}"), env


def _submit(env, *args, **extra_env) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(REPO_ROOT / "pipeline_data_submit.sbatch"), *args],
        capture_output=True,
        text=True,
        env={**env, **extra_env},
        timeout=120,
    )


def _line(result, prefix: str) -> str:
    (line,) = [line for line in result.stdout.splitlines() if line.startswith(prefix)]
    return line[len(prefix) :]


def _names(directory: Path) -> list[str]:
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file())


ARTIFACTS = "tokenized_input_document.bin tokenized_input_document.idx tokenized_input_document.provenance.json"


def test_tokenize_works_in_a_copy_bound_over_the_root_and_copies_the_artifacts_back(staged):
    root, copy, env = staged
    result = _submit(env, "tokenize", str(root), "org/tokenizer", STUB_WRITES=ARTIFACTS)
    assert result.returncode == 0, result.stderr
    assert _line(result, "BIND:") == f"{copy}:{root}"
    assert (copy / "training.jsonl").read_text() == (root / "training.jsonl").read_text()
    # The payload names the root, so the provenance the tool writes records the root.
    payload = _line(result, "PAYLOAD:")
    assert f"--input {root}/training.jsonl" in payload and f"--note dataset_root={root} " in payload
    assert str(copy) not in payload
    for name in ARTIFACTS.split():
        assert (root / name).read_text() == "written by the tool\n"
    assert "Tokenization Complete" in result.stdout


def test_a_tokenize_in_partitions_reads_the_quota_of_the_disk_it_writes(staged):
    """The report is written in the copy, where the container finds it at the root's path, and is not an artifact."""
    root, copy, env = staged
    result = _submit(
        env, "tokenize", str(root), "org/tokenizer", "tokenized", "input", "8", "4", STUB_WRITES=ARTIFACTS
    )
    assert result.returncode == 0, result.stderr
    report = _line(result, "PAYLOAD:").split("--quota-report ")[1].split()[0]
    assert Path(report).parent == root
    assert not list(root.glob(".tokenize_quota.*")) and not list(copy.glob(".tokenize_quota.*"))
    assert _names(root) == sorted(["training.jsonl", *ARTIFACTS.split()])


def test_a_failed_tokenize_copies_nothing_back(staged):
    root, _, env = staged
    result = _submit(env, "tokenize", str(root), "org/tokenizer", STUB_WRITES=ARTIFACTS, STUB_STATUS="3")
    assert result.returncode == 3
    assert _names(root) == ["training.jsonl"]
    assert "TOKENIZATION FAILED" in result.stdout


def test_a_tokenize_whose_artifacts_cannot_be_copied_back_fails(staged):
    """The artifacts exist only in the copy, so a job that could not copy them out has not produced them."""
    root, _, env = staged
    (root / "tokenized_input_document.bin").mkdir()
    result = _submit(env, "tokenize", str(root), "org/tokenizer", STUB_WRITES=ARTIFACTS)
    assert result.returncode == 1
    assert "TOKENIZATION FAILED" in result.stdout


PACK = "packed/org--tokenizer_pad_seq_to_mult1/training_8192.idx.parquet"


def test_pack_reads_copies_of_both_splits_and_writes_in_the_root(staged):
    """Only the JSONL files are bound, so the tool sees the packs the root holds and refuses a mismatched one."""
    root, copy, env = staged
    (root / "validation.jsonl").write_text('{"messages": []}\n')
    earlier = root / "packed" / "org--tokenizer_pad_seq_to_mult1" / "training_4096.pack.json"
    earlier.parent.mkdir(parents=True)
    earlier.write_text("{}")
    result = _submit(env, str(root), "org/tokenizer", "8192", "1", STUB_WRITES=PACK)
    assert result.returncode == 0, result.stderr
    assert _line(result, "BIND:") == ",".join(
        f"{copy / name}:{root / name}" for name in ("training.jsonl", "validation.jsonl")
    )
    assert _names(copy) == ["training.jsonl", "validation.jsonl"]
    assert f"--dataset-root {root} " in _line(result, "PAYLOAD:")
    assert (root / PACK).read_text() == "written by the tool\n"
    assert earlier.read_text() == "{}"
    assert "Packing Complete" in result.stdout


def test_pack_binds_the_training_split_alone_when_there_is_no_validation_split(staged):
    root, copy, env = staged
    result = _submit(env, str(root), "org/tokenizer", STUB_WRITES=PACK)
    assert result.returncode == 0, result.stderr
    assert _line(result, "BIND:") == f"{copy / 'training.jsonl'}:{root / 'training.jsonl'}"


def test_a_failed_pack_fails_the_job(staged):
    root, _, env = staged
    result = _submit(env, str(root), "org/tokenizer", STUB_STATUS="1")
    assert result.returncode == 1
    assert "PACKING FAILED" in result.stdout


@pytest.mark.parametrize("args", [("tokenize",), ()], ids=["tokenize", "pack"])
def test_without_a_stage_root_the_step_reads_and_writes_the_root(staged, args):
    root, copy, env = staged
    result = _submit(env, *args, str(root), "org/tokenizer", GEODESIC_INPUT_STAGE_ROOT="", STUB_WRITES="written.bin")
    assert result.returncode == 0, result.stderr
    assert _line(result, "BIND:") == ""
    assert (root / "written.bin").is_file() and not copy.exists()
