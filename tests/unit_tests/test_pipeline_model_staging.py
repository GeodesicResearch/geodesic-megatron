# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The launchers that memory-map a model's weight files, in a job with an input stage root.

``pipeline_checkpoint_convert.sh import`` and the hf backend of ``pipeline_coherence_submit.sbatch`` copy the model's
directory to the node and bind the copy over the directory in the container. Each test runs the real launcher under
``PipelineShell`` (SLURM, the container and the rank launchers are the stubbed boundaries, and the Hub is the
recorder), with a stage root of the test's own, and reads the binds the model's process was given.
"""

from pathlib import Path

import pytest

from tests.unit_tests.stubbed_shell import ALL_RECORDED, PipelineShell


IMPORT = "pipeline_checkpoint_convert.sh"
COHERENCE = "pipeline_coherence_submit.sbatch"


@pytest.fixture
def shell(tmp_path):
    """A PipelineShell that records each launch with the binds its container was given."""
    shell = PipelineShell(
        tmp_path, record=[name for name in ALL_RECORDED if name != "apptainer"], record_env=("APPTAINER_BIND",)
    )
    yield shell
    shell.remove_job_tmp_dirs()


def _model_directory(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model-00001-of-00001.safetensors").write_text("weights")
    return path


def _hub_snapshot(hub: Path) -> Path:
    """A hub cache's snapshot of ``org/model``, its weight file a link to a blob as the cache stores it."""
    repo = hub / "models--org--model"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (repo / "blobs").mkdir()
    (repo / "blobs" / "b1").write_text("weights")
    (snapshot / "model-00001-of-00001.safetensors").symlink_to(repo / "blobs" / "b1")
    (snapshot / "config.json").write_text("{}")
    return snapshot


def _staged(shell, tmp_path, script, *args, **env):
    return shell.run(script, *args, env={"GEODESIC_INPUT_STAGE_ROOT": str(tmp_path / "stage"), **env})


def _launch(run, cmd):
    (call,) = [call for call in run["calls"] if call["cmd"] == cmd]
    return call


def _files(directory: Path) -> dict[str, str]:
    return {path.name: path.read_text() for path in directory.iterdir() if not path.is_symlink()}


WEIGHTS = {"config.json": "{}", "model-00001-of-00001.safetensors": "weights"}


def test_an_import_maps_a_model_directory_from_a_copy_on_the_node(shell, tmp_path):
    """The ranks open the directory by its own path, where the container has the node's copy."""
    model = _model_directory(tmp_path / "models" / "local-model")
    run = _staged(shell, tmp_path, IMPORT, "import", str(model))
    assert run["returncode"] == 0, run["stderr"]
    copy = Path(f"{tmp_path / 'stage'}{model}")
    assert _files(copy) == WEIGHTS
    ranks = _launch(run, "torchrun")
    assert ranks["env"] == {"APPTAINER_BIND": f"{copy}:{model}"}
    assert ranks["argv"][ranks["argv"].index("--hf-model") + 1] == str(model)


def test_an_import_maps_a_hub_model_from_a_copy_of_its_snapshot(shell, tmp_path):
    """One process asks the Hub for the snapshot, and the files the snapshot links are copied as files."""
    snapshot = _hub_snapshot(tmp_path / "hub")
    run = _staged(shell, tmp_path, IMPORT, "import", "org/model", STUB_HUB_SNAPSHOT=str(snapshot))
    assert run["returncode"] == 0, run["stderr"]
    assert _launch(run, "hub")["kwargs"] == {"repo_id": "org/model"}
    copy = Path(f"{tmp_path / 'stage'}{snapshot}")
    assert _files(copy) == WEIGHTS
    ranks = _launch(run, "torchrun")
    assert ranks["env"] == {"APPTAINER_BIND": f"{copy}:{snapshot}"}
    assert ranks["argv"][ranks["argv"].index("--hf-model") + 1] == "org/model"


def test_an_import_whose_model_cannot_be_copied_starts_no_rank(shell, tmp_path):
    absent = tmp_path / "absent"
    run = _staged(shell, tmp_path, IMPORT, "import", "org/model", STUB_HUB_SNAPSHOT=str(absent))
    assert run["returncode"] != 0
    assert f"FATAL [stage]: {absent} is not a directory." in run["stderr"]
    assert [call for call in run["calls"] if call["cmd"] == "torchrun"] == []


def test_an_import_without_a_stage_root_binds_nothing(shell, tmp_path):
    model = _model_directory(tmp_path / "models" / "local-model")
    run = shell.run(IMPORT, "import", str(model))
    assert run["returncode"] == 0, run["stderr"]
    assert _launch(run, "torchrun")["env"] == {}
    assert not (tmp_path / "stage").exists()


def test_the_hf_backend_maps_its_model_from_a_copy_on_the_node(shell, tmp_path):
    model = _model_directory(tmp_path / "models" / "export")
    run = _staged(shell, tmp_path, COHERENCE, str(model), "--max-tokens", "8")
    assert run["returncode"] == 0, run["stderr"]
    copy = Path(f"{tmp_path / 'stage'}{model}")
    assert _files(copy) == WEIGHTS
    generation = _launch(run, "python")
    assert generation["argv"] == ["pipeline_coherence_test.py", str(model), "--max-tokens", "8"]
    assert generation["env"] == {"APPTAINER_BIND": f"{copy}:{model}"}


def test_the_endpoint_backend_loads_no_model_and_copies_none(shell, tmp_path):
    run = _staged(shell, tmp_path, COHERENCE, "served-model", "--backend", "endpoint", "--base-url", "http://x")
    assert run["returncode"] == 0, run["stderr"]
    assert _launch(run, "python")["env"] == {}
    assert not (tmp_path / "stage").exists()


def test_a_model_the_hf_backend_cannot_find_is_not_generated_from(shell, tmp_path):
    absent = tmp_path / "absent"
    run = _staged(shell, tmp_path, COHERENCE, "org/model", STUB_HUB_SNAPSHOT=str(absent))
    assert run["returncode"] == 1
    assert f"FATAL [stage]: {absent} is not a directory." in run["stderr"]
    assert [call for call in run["calls"] if call["cmd"] == "python"] == []
