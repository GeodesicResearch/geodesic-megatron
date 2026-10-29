"""Tests for pipeline_data_submit.sbatch's `tokenize` mode (JSONL -> Megatron .bin/.idx).

Runs the real sbatch script as a subprocess with a stub container runner and env config
(via GEODESIC_REPO_DIR) — the Apptainer container and SLURM are the genuinely-untestable
boundary here; the payload the script would execute inside the container is captured and
asserted on instead. Same pattern as test_pipeline_training_submit.py.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def stub_env(tmp_path):
    stub_repo = tmp_path / "stub_repo"
    stub_repo.mkdir()
    # The wrapper's own shell libraries, real: only the container config and runner are stubbed.
    shutil.copytree(os.path.join(REPO_ROOT, "pipeline", "sh"), stub_repo / "pipeline" / "sh")
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


def test_partitions_default_to_one(executing_env):
    """An unset partitions runs the tool at one partition, reads no quota (the environment has no `lfs`), and records
    both counts."""
    dataset_root, env, _ = executing_env
    result = _run_tokenize(env, [str(dataset_root), TOKENIZER_NAME])
    assert result.returncode == 0, result.stderr
    (preprocess,) = _calls(result, "preprocess_data.py")
    assert " --partitions 1 " in preprocess
    assert "--keep-sequential-samples" not in preprocess
    (count,) = _calls(result, "count_idx_tokens.py")
    assert " --note partitions_requested=1 --note partitions=1 " in count


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


# 200 TiB, the limit the stub `lfs` reports, in the KiB its report states it in.
_LIMIT_KIB = 200 * 2**30


@pytest.fixture()
def partitioned_env(executing_env, tmp_path):
    """The executing environment plus what a run in partitions needs on the host.

    The real scripts/data/tokenize_partitions.py is linked into the stub repo, and the stub `python` runs it for
    `choose` and `remove`. `lfs` is the Lustre client of the cluster's hosts and is absent from the container the tests
    run in, so a stub prints the project and a quota report in the real one's form, its usage set by
    STUB_LFS_USED_KIB. The stub `python` stands in for the container's other tools: for preprocess_data.py it writes
    two part files in the dataset root, as the tool leaves them, it prints the merge check's command rather than
    reading a real corpus, and for count_idx_tokens.py it fails when STUB_FAIL_COUNT is set, so the run can end either
    way after the parts exist.
    """
    dataset_root, env, _ = executing_env
    stub_repo = env["GEODESIC_REPO_DIR"]
    os.symlink(
        os.path.join(REPO_ROOT, "scripts", "data", "tokenize_partitions.py"),
        os.path.join(stub_repo, "scripts", "data", "tokenize_partitions.py"),
    )
    stub_bin = tmp_path / "stub_bin"
    lfs = stub_bin / "lfs"
    lfs.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = project ]; then echo "4242 P $3"; exit 0; fi\n'
        'printf "Disk quotas for prj 4242 (pid 4242):\\n"\n'
        'printf "     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace\\n"\n'
        f'printf "%s\\n  %s 0 {_LIMIT_KIB} - 10 0 20 -\\n" "$4" "$STUB_LFS_USED_KIB"\n'
    )
    lfs.chmod(lfs.stat().st_mode | stat.S_IEXEC)
    python = stub_bin / "python"
    python.write_text(
        "#!/bin/bash\n"
        f'if [ "$1" = scripts/data/prepare_revisions.py ]; then exec {sys.executable} "$@"; fi\n'
        f'if [ "$1" = scripts/data/tokenize_partitions.py ] && [ "$2" != check ]; then exec {sys.executable} "$@"; fi\n'
        'if [ "$1" = 3rdparty/Megatron-LM/tools/preprocess_data.py ]; then\n'
        f"    echo part > {dataset_root}/training_0.jsonl\n"
        f"    echo part > {dataset_root}/tokenized_base_0_input_document.bin\n"
        "fi\n"
        'if [ "$1" = scripts/data/count_idx_tokens.py ] && [ -n "$STUB_FAIL_COUNT" ]; then exit 1; fi\n'
        'printf "CALL:"; printf " %s" "$@"; printf "\\n"\n'
    )
    return dataset_root, env


def _partitioned(env, dataset_root, used_fraction, **extra_env):
    env = {**env, "STUB_LFS_USED_KIB": str(int(_LIMIT_KIB * used_fraction)), **extra_env}
    return _run_tokenize(env, [str(dataset_root), TOKENIZER_NAME, "tokenized_base", "input", "256", "16"])


def _parts_left(dataset_root):
    return sorted(path.name for path in dataset_root.iterdir() if path.name != "training.jsonl")


def test_with_room_under_the_line_the_parts_run_in_order_and_are_checked_and_recorded(partitioned_env):
    dataset_root, env = partitioned_env
    result = _partitioned(env, dataset_root, 0.5)
    assert result.returncode == 0, result.stderr
    (preprocess,) = _calls(result, "preprocess_data.py")
    assert " --workers 256 --partitions 16 --keep-sequential-samples " in preprocess
    (check,) = _calls(result, "tokenize_partitions.py check")
    assert " --partitions 16" in check
    (count,) = _calls(result, "count_idx_tokens.py")
    assert " --note partitions_requested=16 --note partitions=16 --note merge_order_check=CALL:" in count
    assert "[partitions] 16 partitions" in result.stderr
    assert _parts_left(dataset_root) == []


def test_without_room_for_the_parts_the_corpus_is_tokenized_in_one(partitioned_env):
    dataset_root, env = partitioned_env
    result = _partitioned(env, dataset_root, 0.94)
    assert result.returncode == 0, result.stderr
    (preprocess,) = _calls(result, "preprocess_data.py")
    assert " --partitions 1 " in preprocess
    assert "--keep-sequential-samples" not in preprocess
    assert _calls(result, "tokenize_partitions.py check") == []
    (count,) = _calls(result, "count_idx_tokens.py")
    assert " --note partitions_requested=16 --note partitions=1 " in count
    assert "merge_order_check" not in count
    assert "one partition, not 16" in result.stderr


def test_part_files_an_earlier_run_left_stop_the_job_before_it_tokenizes(partitioned_env):
    dataset_root, env = partitioned_env
    (dataset_root / "training_3.jsonl").write_text("left by a dead run\n")
    result = _partitioned(env, dataset_root, 0.5)
    assert result.returncode != 0
    assert _calls(result, "preprocess_data.py") == []
    assert "training_3.jsonl" in result.stderr
    assert _parts_left(dataset_root) == ["training_3.jsonl"]  # left for whoever owns them, never reused


@pytest.mark.parametrize("ending", ["completes", "fails"])
def test_the_part_files_are_removed_however_the_job_ends(partitioned_env, ending):
    dataset_root, env = partitioned_env
    result = _partitioned(env, dataset_root, 0.5, **({"STUB_FAIL_COUNT": "1"} if ending == "fails" else {}))
    assert (result.returncode == 0) == (ending == "completes"), result.stderr
    assert _parts_left(dataset_root) == []


@pytest.mark.parametrize("partitions", ["0", "-2", "abc"])
def test_a_partition_count_that_is_not_a_positive_integer_is_refused_before_anything_runs(executing_env, partitions):
    dataset_root, env, _ = executing_env
    result = _run_tokenize(env, [str(dataset_root), TOKENIZER_NAME, "tokenized_base", "input", "32", partitions])
    assert result.returncode != 0
    assert f"partitions must be a positive integer, got '{partitions}'" in result.stderr
    assert _calls(result, "preprocess_data.py") == []
