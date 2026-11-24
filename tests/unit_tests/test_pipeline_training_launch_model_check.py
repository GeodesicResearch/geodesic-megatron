# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""pipeline_training_launch.sh accepts a --model that is exactly one of its ALLOWED_MODELS and refuses any other value.

Every case runs the real launcher under PipelineShell (SLURM, the container runtime and the rank launchers are its
stubs), so a refusal is the launcher's own, made before it starts any srun step.
"""

import pytest

from tests.unit_tests.launcher_harness import Launcher, launcher_shell


@pytest.fixture
def shell(tmp_path):
    shell = launcher_shell(tmp_path)
    yield shell
    shell.remove_job_tmp_dirs()


@pytest.fixture
def launch(tmp_path, shell):
    return Launcher(tmp_path, shell)


@pytest.mark.parametrize("model", ["nano", "super", "ultra", "olmo3_7b", "olmo3_32b"])
def test_a_model_of_allowed_models_launches(launch, model):
    run = launch(None, "--model", model)
    assert run.result.returncode == 0, run.result.stderr
    assert run.srun_called


@pytest.mark.parametrize("model", ["olmo3", "olmo3_7", "olmo3_7b_x", "nano super", "NANO", "nano "])
def test_any_other_value_is_refused_before_srun(launch, model):
    run = launch(None, "--model", model)
    assert run.result.returncode == 1
    assert f"--model must be one of: nano super ultra olmo3_7b olmo3_32b, got: {model}" in run.result.stderr
    assert not run.srun_called
