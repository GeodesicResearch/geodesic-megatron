"""Tests for pipeline_data_submit.sbatch's `tokenize` mode (JSONL -> Megatron .bin/.idx).

Runs the real sbatch script as a subprocess with a stub container runner and env config
(via GEODESIC_REPO_DIR) — the Apptainer container and SLURM are the genuinely-untestable
boundary here; the payload the script would execute inside the container is captured and
asserted on instead. Same pattern as test_pipeline_training_submit.py.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys

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
    env["GEODESIC_REPO_DIR"] = str(stub_repo)
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


def test_partitions_default_to_one(stub_env):
    """An unset partitions must not change the invocation for existing callers."""
    _, dataset_root, env = stub_env
    (dataset_root / "training.jsonl").write_text('{"input": "hello"}\n')
    result = _run_tokenize(env, [str(dataset_root), "some/tokenizer"])
    assert result.returncode == 0, result.stderr
    payload = "\n".join(line for line in result.stdout.splitlines() if line.startswith("PAYLOAD:"))
    assert "--partitions 1" in payload
    assert "partitions=1" in payload


def test_partitions_reach_the_tool_and_the_provenance(stub_env):
    """The partition count must be recorded, since it changes how the artifact was built."""
    _, dataset_root, env = stub_env
    (dataset_root / "training.jsonl").write_text('{"input": "hello"}\n')
    result = _run_tokenize(env, [str(dataset_root), "some/tokenizer", "tokenized_base", "input", "256", "16"])
    assert result.returncode == 0, result.stderr
    payload = "\n".join(line for line in result.stdout.splitlines() if line.startswith("PAYLOAD:"))
    assert "--workers 256" in payload
    assert "--partitions 16" in payload
    assert "workers=256" in payload
    assert "partitions=16" in payload
    assert "Partitions:   16" in result.stdout


TOKENIZER_NAME = "org/tokenizer"
TOKENIZER_PIN = "4" * 40


@pytest.fixture()
def executing_env(stub_env, tmp_path):
    """The stub repo with a runner that executes the payload in bash instead of printing it.

    The container is the untestable boundary, so the payload runs on the host: `python` is a stub that runs the real
    scripts/data/prepare_revisions.py (the tokenizer reference's resolution) and prints every other command it is
    given (preprocess_data.py and count_idx_tokens.py, which need a real corpus and the container). A Hugging Face
    cache in `tmp_path` holds the pinned tokenizer's snapshot, with the Hub switched off.
    """
    stub_repo, dataset_root, env = stub_env
    (stub_repo / "pipeline_env_exec.sh").write_text('#!/bin/bash\nexec bash -c "$1"\n')
    (stub_repo / "pipeline_env_activate.sh").write_text("")
    (stub_repo / "scripts" / "data").mkdir(parents=True)
    (stub_repo / "scripts" / "data" / "prepare_revisions.py").symlink_to(
        os.path.join(REPO_ROOT, "scripts", "data", "prepare_revisions.py")
    )
    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    python = stub_bin / "python"
    python.write_text(
        "#!/bin/bash\n"
        f'if [ "$1" = scripts/data/prepare_revisions.py ]; then exec {sys.executable} "$@"; fi\n'
        'printf "CALL:"; printf " %s" "$@"; printf "\\n"\n'
    )
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    cache = tmp_path / "hub"
    snapshot = cache / f"models--{TOKENIZER_NAME.replace('/', '--')}" / "snapshots" / TOKENIZER_PIN
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}")
    env.update(PATH=f"{stub_bin}:{env['PATH']}", HF_HUB_OFFLINE="1", HF_HUB_CACHE=str(cache))
    (dataset_root / "training.jsonl").write_text('{"input": "hello"}\n')
    return dataset_root, env, snapshot


def _calls(result, tool):
    return [line for line in result.stdout.splitlines() if line.startswith("CALL:") and tool in line]


def test_a_pinned_tokenizer_loads_its_commits_snapshot_and_records_the_commit(executing_env):
    dataset_root, env, snapshot = executing_env
    result = _run_tokenize(env, [str(dataset_root), f"{TOKENIZER_NAME}@{TOKENIZER_PIN}", "tokenized_base"])
    assert result.returncode == 0, result.stderr
    (preprocess,) = _calls(result, "preprocess_data.py")
    assert f" --tokenizer-model {snapshot} " in preprocess
    (count,) = _calls(result, "count_idx_tokens.py")
    assert f" --note tokenizer={TOKENIZER_NAME} --note tokenizer_revision={TOKENIZER_PIN} " in count
    assert f"{TOKENIZER_NAME} @ {TOKENIZER_PIN}, loaded from {snapshot}" in result.stdout


def test_an_unpinned_tokenizer_loads_by_name_and_records_no_commit(executing_env):
    dataset_root, env, _ = executing_env
    result = _run_tokenize(env, [str(dataset_root), TOKENIZER_NAME, "tokenized_base"])
    assert result.returncode == 0, result.stderr
    (preprocess,) = _calls(result, "preprocess_data.py")
    assert f" --tokenizer-model {TOKENIZER_NAME} " in preprocess
    (count,) = _calls(result, "count_idx_tokens.py")
    assert f" --note tokenizer={TOKENIZER_NAME} --note json_key=input " in count
    assert "tokenizer_revision" not in count


@pytest.mark.parametrize(
    "reference", [f"{TOKENIZER_NAME}@main", f"{TOKENIZER_NAME}@{'5' * 40}"], ids=["not-a-sha", "commit-not-held"]
)
def test_a_tokenizer_that_cannot_be_resolved_tokenizes_nothing(executing_env, reference):
    dataset_root, env, _ = executing_env
    result = _run_tokenize(env, [str(dataset_root), reference, "tokenized_base"])
    assert result.returncode != 0
    assert _calls(result, "preprocess_data.py") == []
    assert "TOKENIZATION FAILED" in result.stdout
