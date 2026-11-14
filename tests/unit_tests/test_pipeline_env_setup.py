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
"""pipeline_env_setup.sh: the steps an install has on each cluster, and what a host without a GPU is told.

The real script runs against the real profiles, with its container root in a scratch directory. A host's GPU and
its container runtime are what the script probes, so a stand-in ``nvidia-smi`` that fails makes the host one
without a GPU, and no ``apptainer`` is on PATH.
"""

import subprocess

import pytest

from tests.unit_tests.child_process import REPO_ROOT


SETUP = REPO_ROOT / "pipeline_env_setup.sh"
CONFIG = REPO_ROOT / "pipeline_env_config.env"
STEPS = {"isambard": "sif slingshot overlay validate", "verda": "sif overlay validate"}
SUBMIT = {"isambard": "isambard_sbatch", "verda": "./pipeline_cluster_submit.sh"}


def _env(tmp_path, cluster: str) -> dict[str, str]:
    """A shell on ``cluster`` with no GPU and no container runtime, whose install lives under ``tmp_path``."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "nvidia-smi").write_text("#!/bin/bash\nexit 1\n")
    (bindir / "nvidia-smi").chmod(0o755)
    return {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "USER": "probe",
        "GEODESIC_CLUSTER": cluster,
        "GEODESIC_CONTAINER_ROOT": str(tmp_path / "containers"),
        "GEODESIC_CONTAINER_STAGE_ROOT": "",
    }


def _setup(tmp_path, cluster: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SETUP), *args], env=_env(tmp_path, cluster), capture_output=True, text=True, timeout=120
    )


def _config_value(tmp_path, cluster: str, name: str) -> str:
    """``name`` as the real config derives it for this install."""
    script = f'source "{CONFIG}" && printf "%s" "${name}"'
    return subprocess.run(
        ["bash", "-c", script], env=_env(tmp_path, cluster), capture_output=True, text=True, check=True
    ).stdout


def _install_cpu_side(tmp_path, cluster: str) -> None:
    """The image and an overlay whose record names the configured packages, as the CPU-side steps leave them."""
    sif = tmp_path / "containers" / f"nemo_{_config_value(tmp_path, cluster, 'CONTAINER_IMAGE_TAG')}.sif"
    sif.parent.mkdir(parents=True, exist_ok=True)
    sif.write_bytes(b"image")
    overlay = tmp_path / "containers" / "overlay" / sif.stem
    overlay.mkdir(parents=True)
    packages = _config_value(tmp_path, cluster, "CONTAINER_OVERLAY_PACKAGES")
    (overlay / "provenance.txt").write_text(f"packages: {packages}\n")


@pytest.mark.parametrize("cluster", sorted(STEPS))
def test_only_takes_a_step_of_the_clusters_install(tmp_path, cluster):
    result = _setup(tmp_path, cluster, "--only", "frobnicate")
    assert result.returncode == 1
    assert f"--only takes one of: {STEPS[cluster]}" in result.stderr


def test_a_cluster_without_slingshot_has_no_slingshot_step(tmp_path):
    """Nothing is built for InfiniBand, so the step that builds Slingshot's NCCL stack is not one of Verda's."""
    result = _setup(tmp_path, "verda", "--only", "slingshot")
    assert result.returncode == 1
    assert "--only takes one of: sif overlay validate" in result.stderr


@pytest.mark.parametrize("cluster", sorted(STEPS))
def test_a_host_without_the_runtime_is_told_to_run_the_setup_as_a_job(tmp_path, cluster):
    """The command is the cluster's own submit command."""
    result = _setup(tmp_path, cluster, "--only", "sif")
    assert result.returncode == 1
    assert "FATAL [env-setup/sif]: apptainer not on PATH" in result.stderr
    assert f"    {SUBMIT[cluster]} pipeline_env_submit.sbatch setup" in result.stderr


def test_on_verda_a_host_without_a_gpu_finishes_the_cpu_steps_and_exits_incomplete(tmp_path):
    _install_cpu_side(tmp_path, "verda")
    result = _setup(tmp_path, "verda")
    assert result.returncode == 2
    assert "[env-setup 1/3 sif] SIF present:" in result.stdout
    assert "[env-setup/overlay] Overlay already matches configured packages" in result.stdout
    assert (
        "CPU-side steps done; run on a GPU node to finish: ./pipeline_cluster_submit.sh pipeline_env_submit.sbatch setup"
        in result.stdout
    )
    assert "slingshot" not in result.stdout + result.stderr
    assert "[env-setup] INCOMPLETE: GPU-only steps were deferred" in result.stderr


def test_on_isambard_a_host_without_a_gpu_defers_the_slingshot_build_by_its_step_number(tmp_path):
    _install_cpu_side(tmp_path, "isambard")
    result = _setup(tmp_path, "isambard")
    assert result.returncode == 2
    assert "[env-setup 1/4 sif] SIF present:" in result.stdout
    assert "[env-setup 2/4 slingshot] DEFERRED" in result.stderr
    assert "  or: isambard_sbatch pipeline_env_submit.sbatch setup" in result.stderr
    assert "CPU-side steps done; run on a GPU node to finish (see 2/4)" in result.stdout
    assert "[env-setup] INCOMPLETE: GPU-only steps were deferred" in result.stderr
