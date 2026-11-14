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
"""pipeline/sh/cluster.sh and the profiles under pipeline/sh/clusters/: the cluster a shell resolves, and what its
profile sets.

A cluster is detected from the host's name and the site's environment variables, so each case runs the real shell
files in a bash whose environment is the case's alone and whose ``hostname`` is a stub: the test host's own name
would decide the result otherwise.
"""

import subprocess

import pytest

from pipeline import cluster
from pipeline.cluster import absent_storage_roots, cluster_names, cluster_profile
from tests.unit_tests.child_process import REPO_ROOT


CLUSTER_SH = REPO_ROOT / "pipeline" / "sh" / "cluster.sh"
PROFILES = ("isambard", "verda")
HOOKS = ("detect", "profile", "job_guard", "host_reset", "pull_preflight", "storage_quota_report", "activate")
# The values only one fabric's code reads.
FABRIC_KEYS = {
    "slingshot": {"CLUSTER_HOST_LIBFABRIC"},
    "infiniband": {"CLUSTER_NCCL_IB_HCA", "CLUSTER_BOOTSTRAP_IFNAME"},
}


def _bash(tmp_path, script: str, hostname: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """``script`` after sourcing cluster.sh, on a host named ``hostname`` whose environment is ``env`` alone."""
    stubs = tmp_path / "bin"
    stubs.mkdir(exist_ok=True)
    (stubs / "hostname").write_text(f"#!/bin/bash\necho {hostname}\n")
    (stubs / "hostname").chmod(0o755)
    return subprocess.run(
        ["bash", "-c", f'set -euo pipefail\nsource "{CLUSTER_SH}"\n{script}'],
        env={"PATH": f"{stubs}:/usr/bin:/bin", "USER": "probe", **(env or {})},
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    "hostname, env, cluster",
    [
        ("nid001234", {}, "isambard"),
        ("login02", {}, "isambard"),
        ("worker", {"ISAMBARD_HOST": "ai"}, "isambard"),
        ("slinky-7", {}, "verda"),
        ("slurm-login-slinky-648f499999-s948r", {}, "verda"),
        ("worker", {"VERDA_CLUSTER_NAME": "geodesic-research"}, "verda"),
        # A name given outright is taken, whatever the host looks like.
        ("nid001234", {"GEODESIC_CLUSTER": "verda"}, "verda"),
        ("slinky-7", {"GEODESIC_CLUSTER": "isambard"}, "isambard"),
        ("laptop", {"GEODESIC_CLUSTER": "isambard"}, "isambard"),
    ],
)
def test_cluster_load_resolves_the_cluster_from_the_host_or_the_given_name(tmp_path, hostname, env, cluster):
    result = _bash(tmp_path, 'cluster_load; echo "$GEODESIC_CLUSTER"', hostname, env)
    assert (result.returncode, result.stdout, result.stderr) == (0, f"{cluster}\n", "")


@pytest.mark.parametrize(
    "hostname, env, message",
    [
        ("laptop", {}, "this host (laptop) matches 0 cluster profiles ()"),
        (
            "nid001234",
            {"VERDA_CLUSTER_NAME": "geodesic-research"},
            "this host (nid001234) matches 2 cluster profiles (isambard verda)",
        ),
        ("nid001234", {"GEODESIC_CLUSTER": "nowhere"}, "GEODESIC_CLUSTER=nowhere names no profile"),
    ],
)
def test_cluster_load_fails_when_no_single_profile_is_named_or_detected(tmp_path, hostname, env, message):
    """No profile, two profiles and an unknown name each stop the shell, naming the profiles there are."""
    result = _bash(tmp_path, "cluster_load; echo reached", hostname, env)
    assert result.returncode == 1 and "reached" not in result.stdout
    assert message in result.stderr
    assert "isambard verda" in result.stderr


def test_cluster_load_fails_without_a_user_name(tmp_path):
    """A profile names each account's directories by USER, so an empty one would name a shared directory."""
    result = _bash(tmp_path, "cluster_load; echo reached", "nid001234", {"USER": ""})
    assert result.returncode == 1 and "reached" not in result.stdout
    assert "USER is unset" in result.stderr


def test_cluster_load_exports_the_cluster_name_and_no_profile_value(tmp_path):
    """The container resolves the cluster from the exported name, and no CLUSTER_* value reaches a rank."""
    result = _bash(tmp_path, "cluster_load; env -0", "nid001234")
    exported = dict(item.split("=", 1) for item in result.stdout.split("\0") if item)
    assert exported["GEODESIC_CLUSTER"] == "isambard"
    assert not [name for name in exported if name.startswith("CLUSTER_")]


@pytest.mark.parametrize("cluster, mask", [("isambard", "0027"), ("verda", "0002")])
def test_cluster_load_applies_the_profiles_file_creation_mask(tmp_path, cluster, mask):
    """Verda's shared trees are group-writable, so its profile sets the mask. Isambard keeps the shell's."""
    result = _bash(tmp_path, "umask 027; cluster_load; umask", "laptop", {"GEODESIC_CLUSTER": cluster})
    assert (result.returncode, result.stdout) == (0, f"{mask}\n")


@pytest.mark.parametrize("cluster", PROFILES)
def test_every_profile_defines_every_hook(tmp_path, cluster):
    names = " ".join(f"{cluster}_{hook}" for hook in HOOKS)
    result = _bash(tmp_path, f"cluster_load; declare -F {names} >/dev/null", "laptop", {"GEODESIC_CLUSTER": cluster})
    assert result.returncode == 0, result.stderr


def test_cluster_names_lists_the_profiles(tmp_path):
    result = _bash(tmp_path, "cluster_names", "laptop")
    assert result.stdout.split() == sorted(PROFILES) == cluster_names()


@pytest.mark.parametrize(
    "here, existing, absent",
    [
        ("verda", {"/shared", "/home", "/local"}, {"/projects": "isambard", "/lus": "isambard"}),
        # /home is storage on Verda and exists on every host, so a path under it belongs to no other cluster.
        ("isambard", {"/projects", "/lus", "/home"}, {"/shared": "verda", "/local": "verda"}),
        ("isambard", {"/projects", "/lus", "/home", "/local", "/shared"}, {}),
    ],
)
def test_absent_storage_roots_are_the_other_clusters_roots_this_host_lacks(monkeypatch, here, existing, absent):
    monkeypatch.setenv("GEODESIC_CLUSTER", here)
    monkeypatch.setenv("USER", "probe")
    # The roots are other clusters' filesystems, so which of them a host has is stated here, not read from the test
    # host.
    monkeypatch.setattr(cluster.os.path, "exists", lambda path: path in existing)
    assert absent_storage_roots() == absent


def test_the_profiles_set_the_same_values_apart_from_their_fabrics_own(monkeypatch):
    """A value the launchers read exists on every cluster; only a fabric's own values differ between profiles."""
    monkeypatch.setenv("USER", "probe")
    common = None
    for cluster in PROFILES:
        profile = cluster_profile(cluster)
        assert profile["GEODESIC_CLUSTER"] == cluster
        names = set(profile) - FABRIC_KEYS[profile["CLUSTER_FABRIC"]]
        assert FABRIC_KEYS[profile["CLUSTER_FABRIC"]] <= set(profile)
        common = names if common is None else common
        assert names == common


def test_the_isambard_profile_holds_the_values_isambard_runs_with(monkeypatch):
    """The Isambard profile sets every value the launchers read, to the values Isambard runs with."""
    monkeypatch.setenv("USER", "probe")
    profile = cluster_profile("isambard")
    assert profile == {
        "GEODESIC_CLUSTER": "isambard",
        "CLUSTER_GPUS_PER_NODE": "4",
        "CLUSTER_GPU_PEAK_TFLOPS": "989.4",
        "CLUSTER_CUDA_ARCH_LIST": "9.0",
        "CLUSTER_FABRIC": "slingshot",
        "CLUSTER_IMAGE_TAG": "26.04",
        "CLUSTER_STORAGE_BINDS": "/projects,/lus",
        "CLUSTER_CONTAINER_ROOT": "/projects/a5k/public/containers",
        "CLUSTER_CONTAINER_STAGE_ROOT": "",
        "CLUSTER_APPTAINER_CACHEDIR": "/projects/a5k/public/apptainer_cache_probe",
        "CLUSTER_APPTAINER_TMPDIR": "/projects/a5k/public/tmp/apptainer_probe",
        "CLUSTER_INPUT_STAGE_ROOT": "",
        "CLUSTER_HOST_LIBFABRIC": "/opt/cray/libfabric/2.3.1",
        "CLUSTER_NEMO_HOME": "/projects/a5k/public/data/nemo_cache",
        "CLUSTER_HF_HOME": "/projects/a5k/public/hf",
        "CLUSTER_HF_DATASETS_CACHE": "/projects/a5k/public/hf/datasets_container_probe",
        "CLUSTER_WANDB_DIR": "/projects/a5k/public/logs/wandb",
        "CLUSTER_ACTIVATE_TMPDIR": "/projects/a5k/public/tmp",
        "CLUSTER_DATA_ROOT": "/projects/a5k/public/data",
        "CLUSTER_MODEL_ROOT": "/projects/a5k/public/checkpoints/megatron_bridges/models",
        "CLUSTER_PROFILE_ROOT": "/projects/a5k/public/profiles",
        "CLUSTER_EXPORT_CLONE_ROOT": "/projects/a5k/public/tmp/export_clones_probe",
        "CLUSTER_NOFILE": "",
        "CLUSTER_UMASK": "",
        "CLUSTER_HOST_HAS_GIT": "1",
        "CLUSTER_SUBMIT_WRAPPER": "isambard_sbatch",
        "CLUSTER_SUBMIT_COMMAND": "isambard_sbatch",
        "CLUSTER_LOG_ROOT": "/projects/a5k/public/logs",
        "CLUSTER_SMOKE_MIN_BUSBW_GB_S": "",
    }


def test_cluster_profile_raises_with_the_shells_message_when_no_cluster_resolves(monkeypatch):
    monkeypatch.setenv("GEODESIC_CLUSTER", "nowhere")
    with pytest.raises(RuntimeError, match="GEODESIC_CLUSTER=nowhere names no profile"):
        cluster_profile()
