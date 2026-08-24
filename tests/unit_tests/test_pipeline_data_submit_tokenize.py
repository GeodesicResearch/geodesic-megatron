"""Tests for pipeline_data_submit.sbatch's `tokenize` mode (JSONL -> Megatron .bin/.idx).

Runs the real sbatch script as a subprocess with a stub container runner and env config
(via PIPELINE_REPO_DIR) — the Apptainer container and SLURM are the genuinely-untestable
boundary here; the payload the script would execute inside the container is captured and
asserted on instead. Same pattern as test_pipeline_training_submit.py.
"""

from __future__ import annotations

import os
import stat
import subprocess

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def stub_env(tmp_path):
    stub_repo = tmp_path / "stub_repo"
    stub_repo.mkdir()
    (stub_repo / "pipeline_env_config.env").write_text(
        'CONTAINER_SIF="/stub/image.sif"\nenv_config_require() { return 0; }\n'
    )
    runner = stub_repo / "pipeline_env_exec.sh"
    runner.write_text('#!/bin/bash\nprintf "PAYLOAD:%s\\n" "$1"\n')
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)

    dataset_root = tmp_path / "dataset_root"
    dataset_root.mkdir()

    env = dict(os.environ)
    env["PIPELINE_REPO_DIR"] = str(stub_repo)
    env.pop("SLURM_JOB_ID", None)
    return stub_repo, dataset_root, env


def _run_tokenize(env, args):
    return subprocess.run(
        ["bash", os.path.join(REPO_ROOT, "pipeline_data_submit.sbatch"), "tokenize", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_happy_path_payload_and_provenance(stub_env):
    _, dataset_root, env = stub_env
    (dataset_root / "training.jsonl").write_text('{"input": "hello"}\n')
    result = _run_tokenize(env, [str(dataset_root), "geodesic-research/nemotron-base-tokenizer", "tokenized_base"])
    assert result.returncode == 0, result.stderr
    payload = "\n".join(line for line in result.stdout.splitlines() if line.startswith("PAYLOAD:"))
    assert "tools/preprocess_data.py" in payload
    assert "--append-eod" in payload
    assert "geodesic-research/nemotron-base-tokenizer" in payload
    assert f"{dataset_root}/tokenized_base" in payload
    assert "count_idx_tokens.py" in payload
    assert f"{dataset_root}/tokenized_base_input_document.provenance.json" in payload
    assert "Tokenization Complete" in result.stdout


def test_default_variant_and_json_key(stub_env):
    _, dataset_root, env = stub_env
    (dataset_root / "training.jsonl").write_text('{"input": "hello"}\n')
    result = _run_tokenize(env, [str(dataset_root), "some/tokenizer"])
    assert result.returncode == 0, result.stderr
    assert f"{dataset_root}/tokenized_input_document.idx" in result.stdout


def test_missing_jsonl_fails_loudly(stub_env):
    _, dataset_root, env = stub_env
    result = _run_tokenize(env, [str(dataset_root), "some/tokenizer"])
    assert result.returncode == 1
    assert "FATAL" in result.stderr
    assert "prepare" in result.stderr


def test_missing_tokenizer_arg_fails(stub_env):
    _, dataset_root, env = stub_env
    result = _run_tokenize(env, [str(dataset_root)])
    assert result.returncode != 0
