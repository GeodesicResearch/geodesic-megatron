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

"""Unit tests for pipeline_env_config.env: the bind list the container shim mounts, and the host-toolchain scrub."""

import os
import subprocess
from pathlib import Path


CONFIG = Path(__file__).resolve().parents[2] / "pipeline_env_config.env"


def _env(cluster: str) -> dict[str, str]:
    """The environment of a shell on ``cluster`` that names a REPO_DIR and nothing else of the container."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ["HOME"],
        "USER": "probe",
        "REPO_DIR": "/repo",
        "GEODESIC_CLUSTER": cluster,
    }


def _binds(tmpdir: str | None, cluster: str) -> list[str]:
    """CONTAINER_BINDS as the real config derives it on ``cluster`` in a shell whose TMPDIR is ``tmpdir`` (unset
    when None)."""
    env = _env(cluster)
    if tmpdir is not None:
        env["TMPDIR"] = tmpdir
    result = subprocess.run(
        ["bash", "-c", f'source "{CONFIG}" && printf "%s" "$CONTAINER_BINDS"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split(",")


def test_an_existing_tmpdir_is_bound_at_its_own_path(tmp_path):
    # The site gives each job a node-local TMPDIR that apptainer does not bind; unbound, the path lands on the
    # image's read-only root inside the container.
    job_tmp = tmp_path / "local_user"
    job_tmp.mkdir()
    assert _binds(str(job_tmp), "isambard") == _binds(None, "isambard") + [str(job_tmp)]


def test_without_a_tmpdir_the_bind_list_is_the_fixed_set():
    binds = _binds(None, "isambard")
    assert binds[:3] == ["/projects", "/lus", "/repo"]
    assert len(binds) == len(set(binds))


def test_a_slingshot_cluster_binds_the_host_libraries_of_its_fabric_and_an_infiniband_cluster_binds_none():
    """Isambard binds the Cray libfabric, the host library directory and the Slingshot build after its storage and
    the checkout. Verda binds its storage and the checkout alone."""
    assert _binds(None, "isambard")[3:] == [
        "/opt/cray/libfabric/2.3.1:/host/opt/cray/libfabric/2.3.1:ro",
        "/usr/lib64:/host/usr/lib64:ro",
        "/projects/a5k/public/containers/slingshot/nemo_26.04:/opt/slingshot",
    ]
    assert _binds(None, "verda") == ["/shared", "/home", "/local", "/repo"]


def test_a_tmpdir_that_does_not_exist_adds_no_bind(tmp_path):
    # apptainer refuses a bind whose source is missing, which would end every container command of the job.
    assert _binds(str(tmp_path / "missing"), "isambard") == _binds(None, "isambard")


SCRUBBED = (
    "LD_PRELOAD",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "NCCL_LIBRARY",
    "CC",
    "CXX",
    "CUDAHOSTCXX",
    "CUDA_HOME",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "CUDNN_PATH",
)


def test_env_scrub_host_toolchain_unsets_the_host_toolchain_and_empties_the_library_path():
    """What both the exec shim and the Slingshot build scrub, and nothing else."""
    env = _env("isambard")
    env.update({key: f"/host/{key.lower()}" for key in SCRUBBED})
    env.update({"LD_LIBRARY_PATH": "/opt/cray/lib64", "KEEP_ME": "1"})
    result = subprocess.run(
        ["bash", "-c", f'source "{CONFIG}" && env_scrub_host_toolchain && env -0'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    after = dict(item.split("=", 1) for item in result.stdout.split("\0") if item)
    assert not set(SCRUBBED) & set(after)
    assert after["LD_LIBRARY_PATH"] == ""
    assert after["KEEP_ME"] == "1"


# --- env_config_stage ------------------------------------------------------------------------------------------------

# Unpacking an image is the container runtime's work, and there is no runtime inside the container the tests run in.
# The stand-in takes `apptainer build --sandbox --fix-perms <directory> <sif>`, writes the image's bytes into the
# directory as one file and counts its calls. APPTAINER_STUB_FAIL makes it exit 1 after starting the directory.
APPTAINER_STUB = """#!/bin/bash
[ "$1 $2 $3" = "build --sandbox --fix-perms" ] || { echo "unexpected: apptainer $*" >&2; exit 2; }
echo call >> "$APPTAINER_STUB_CALLS"
mkdir "$4" || exit 1
[ -z "${APPTAINER_STUB_FAIL:-}" ] || exit 1
cp "$5" "$4/image"
"""


def _stage_env(tmp_path, cluster: str, stage_root: str | None) -> dict[str, str]:
    """A shell on ``cluster`` whose image is ``tmp_path/shared/nemo.sif`` and whose ``apptainer`` is the stand-in;
    ``stage_root`` overrides the profile's stage root when it is not None."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "apptainer").write_text(APPTAINER_STUB)
    (bindir / "apptainer").chmod(0o755)
    env = _env(cluster)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["APPTAINER_STUB_CALLS"] = str(tmp_path / "apptainer_calls")
    env["GEODESIC_CONTAINER_SIF"] = str(tmp_path / "shared" / "nemo.sif")
    if stage_root is not None:
        env["GEODESIC_CONTAINER_STAGE_ROOT"] = stage_root
    return env


def _run_stage(env: dict[str, str]) -> subprocess.CompletedProcess:
    """env_config_stage under the shim's shell options, then the image the shim would run."""
    script = f'set -euo pipefail; source "{CONFIG}"; env_config_stage; printf "%s" "$CONTAINER_SIF"'
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120)


def _image(tmp_path, content: bytes) -> Path:
    source = tmp_path / "shared" / "nemo.sif"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(content)
    return source


def _unpackings(tmp_path) -> int:
    calls = tmp_path / "apptainer_calls"
    return len(calls.read_text().splitlines()) if calls.exists() else 0


def test_staging_unpacks_the_image_under_the_stage_root_once(tmp_path):
    """The first shim on a node unpacks the image and says so on stderr. A later one finds the copy current and
    prints nothing, so a caller that captures the shim's stdout is not disturbed either time."""
    source = _image(tmp_path, b"image v1" * 4096)
    env = _stage_env(tmp_path, "verda", str(tmp_path / "local"))
    first = _run_stage(env)
    copy = tmp_path / "local" / "nemo"
    assert (first.returncode, first.stdout) == (0, str(copy))
    assert f"unpacking {source} to {copy}" in first.stderr
    assert (copy / "image").read_bytes() == source.read_bytes()
    second = _run_stage(env)
    assert (second.returncode, second.stdout, second.stderr) == (0, str(copy), "")
    assert _unpackings(tmp_path) == 1


def test_a_changed_image_is_unpacked_again_and_the_earlier_copy_is_kept_aside(tmp_path):
    """A container still running from the earlier copy reads its files, so the copy is renamed, not removed."""
    source = _image(tmp_path, b"image v1")
    env = _stage_env(tmp_path, "verda", str(tmp_path / "local"))
    assert _run_stage(env).returncode == 0
    source.write_bytes(b"image v2, a different size")
    result = _run_stage(env)
    assert result.returncode == 0 and "unpacking" in result.stderr
    assert (tmp_path / "local" / "nemo" / "image").read_bytes() == b"image v2, a different size"
    [earlier] = (tmp_path / "local").glob("nemo.old.*")
    assert (earlier / "image").read_bytes() == b"image v1"
    assert f"the earlier copy is now {earlier}" in result.stderr


def test_staging_replaces_what_an_interrupted_unpacking_left(tmp_path):
    """An unpacking that died leaves its partial directory and no record. Neither is trusted."""
    source = _image(tmp_path, b"whole image")
    env = _stage_env(tmp_path, "verda", str(tmp_path / "local"))
    interrupted = _run_stage({**env, "APPTAINER_STUB_FAIL": "1"})
    assert interrupted.returncode == 1 and interrupted.stdout == ""
    assert (tmp_path / "local" / "nemo.partial").is_dir() and not (tmp_path / "local" / "nemo").exists()
    result = _run_stage(env)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "local" / "nemo" / "image").read_bytes() == source.read_bytes()
    assert not (tmp_path / "local" / "nemo.partial").exists()


def test_a_copy_without_its_record_is_unpacked_again(tmp_path):
    """The record is written last, so a directory without one may be a copy cut short."""
    _image(tmp_path, b"whole image")
    env = _stage_env(tmp_path, "verda", str(tmp_path / "local"))
    assert _run_stage(env).returncode == 0
    (tmp_path / "local" / "nemo.stamp").unlink()
    assert _run_stage(env).returncode == 0
    assert _unpackings(tmp_path) == 2


def test_shims_that_start_together_on_one_node_unpack_the_image_once(tmp_path):
    _image(tmp_path, b"image" * 200_000)
    env = _stage_env(tmp_path, "verda", str(tmp_path / "local"))
    script = f'set -euo pipefail; source "{CONFIG}"; env_config_stage'
    shims = [subprocess.Popen(["bash", "-c", script], env=env, stderr=subprocess.PIPE, text=True) for _ in range(6)]
    logs = [shim.communicate(timeout=120)[1] for shim in shims]
    assert [shim.returncode for shim in shims] == [0] * 6, logs
    assert sum("unpacking" in log for log in logs) == 1, logs
    assert _unpackings(tmp_path) == 1
    assert (tmp_path / "local" / "nemo" / "image").read_bytes() == b"image" * 200_000


def test_an_empty_stage_root_runs_the_image_from_where_it_is_stored(tmp_path):
    """Isambard's profile names no stage root, and an empty override turns Verda's off."""
    source = _image(tmp_path, b"image")
    for env in (_stage_env(tmp_path, "isambard", None), _stage_env(tmp_path, "verda", "")):
        result = _run_stage(env)
        assert (result.returncode, result.stdout, result.stderr) == (0, str(source), "")
    assert _unpackings(tmp_path) == 0
    assert not (tmp_path / "local").exists()


def test_the_verda_profile_stages_the_image_under_the_accounts_local_directory(tmp_path):
    result = subprocess.run(
        ["bash", "-c", f'source "{CONFIG}" && printf "%s" "$CONTAINER_SIF"'],
        env=_stage_env(tmp_path, "verda", None),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "/local/geodesic_probe/containers/nemo"


def test_staging_fails_when_the_image_is_missing(tmp_path):
    result = _run_stage(_stage_env(tmp_path, "verda", str(tmp_path / "local")))
    assert result.returncode == 1 and result.stdout == ""
    assert _unpackings(tmp_path) == 0
