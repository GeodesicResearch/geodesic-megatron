# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""pipeline_data_submit.sbatch's pack-only mode: the optional tokenizer-workers argument.

The real sbatch script runs as a subprocess with the real shell libraries. The Apptainer container is the boundary
that cannot run here, so the runner is a stub that prints the payload it was given (the pattern of
test_pipeline_data_submit_tokenize.py). The dataset root is read in place: test_pipeline_data_submit_staging.py covers
the copies a stage root adds.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from tests.unit_tests.data_submit_stub import REPO_ROOT, stub_checkout


@pytest.fixture()
def packing(tmp_path):
    """A stub checkout, a dataset root with its training JSONL, and the environment of a job that reads in place."""
    stub_repo = stub_checkout(tmp_path / "stub_repo", '#!/bin/bash\nprintf "PAYLOAD:%s\\n" "$1"\n')
    root = tmp_path / "data" / "org__corpus"
    root.mkdir(parents=True)
    (root / "training.jsonl").write_text('{"messages": []}\n')
    env = dict(os.environ)
    env.update(GEODESIC_CLUSTER="verda", GEODESIC_REPO_DIR=str(stub_repo), GEODESIC_INPUT_STAGE_ROOT="")
    env.pop("SLURM_JOB_ID", None)
    env.pop("APPTAINER_BIND", None)
    return root, env


def _pack(env, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(REPO_ROOT / "pipeline_data_submit.sbatch"), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def _payload(result) -> str:
    (line,) = [line for line in result.stdout.splitlines() if line.startswith("PAYLOAD:")]
    return line


def test_the_worker_count_reaches_the_packer(packing):
    root, env = packing
    result = _pack(env, str(root), "org/tokenizer", "32768", "4", "16")
    assert result.returncode == 0, result.stderr
    assert _payload(result).endswith(
        f"python scripts/data/pack_sft_dataset.py --dataset-root {root} --tokenizer org/tokenizer "
        "--seq-length 32768 --pad-seq-to-mult 4 --num-tokenizer-workers 16"
    )
    assert "Workers:        16  (tokenizer processes)" in result.stdout
    assert "Packing Complete" in result.stdout


@pytest.mark.parametrize("args", [("32768", "4"), ("32768", "4", ""), ()], ids=["four", "empty-fifth", "defaults"])
def test_without_a_worker_count_the_packer_gets_no_such_argument(packing, args):
    """The packer then runs at its own default, and the payload is the one the mode has always written."""
    root, env = packing
    result = _pack(env, str(root), "org/tokenizer", *args)
    assert result.returncode == 0, result.stderr
    geometry = args[:2] or ("8192", "1")
    assert _payload(result).endswith(
        f"python scripts/data/pack_sft_dataset.py --dataset-root {root} --tokenizer org/tokenizer "
        f"--seq-length {geometry[0]} --pad-seq-to-mult {geometry[1]}"
    )
    assert "Workers:" not in result.stdout


@pytest.mark.parametrize("workers", ["0", "-2", "four", "1.5", "08", "4 "])
def test_a_worker_count_that_is_not_a_positive_integer_is_refused(packing, workers):
    """Refused before the container starts: the packer reads 0 and negatives as every CPU of the host."""
    root, env = packing
    result = _pack(env, str(root), "org/tokenizer", "32768", "4", workers)
    assert result.returncode == 1
    assert f"FATAL: tokenizer-workers must be a positive integer, got '{workers}'." in result.stderr
    assert "PAYLOAD:" not in result.stdout
