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
"""pipeline/sh/: each shell library's functions, sourced on their own into a bash with the launchers' options.

The scontrol each function may call is PipelineShell's stub (tests/unit_tests/stubbed_shell.py), since SLURM is
absent in the test container.
"""

import json
import subprocess

import pytest

from tests.unit_tests.child_process import REPO_ROOT


LIBS = REPO_ROOT / "pipeline" / "sh"


def _bash(shell, script: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """``script`` after sourcing every library, under ``set -euo pipefail``, in the stubbed environment."""
    prelude = "set -euo pipefail\n" + "".join(
        f'source "{LIBS / lib}"\n' for lib in ("common.sh", "slurm.sh", "fabric.sh")
    )
    return subprocess.run(
        ["bash", "-c", prelude + script],
        capture_output=True,
        text=True,
        env=shell.env({"REPO_DIR": str(REPO_ROOT), **(extra_env or {})}),
        timeout=60,
    )


@pytest.mark.parametrize(
    "word, arity",
    [("--exclude", "2"), ("--qos", "2"), ("--exclude=nid1", "1"), ("--nodes=2", "1"), ("--hf-model", "0"), ("x", "0")],
)
def test_slurm_flag_arity(pipeline_launch_shell, word, arity):
    result = _bash(pipeline_launch_shell, f"slurm_flag_arity {json.dumps(word)}")
    assert (result.returncode, result.stdout) == (0, f"{arity}\n")


def test_container_payload_activates_in_the_repo_before_the_command(pipeline_launch_shell):
    result = _bash(pipeline_launch_shell, 'container_payload "python x.py"')
    assert result.stdout == f"cd {REPO_ROOT}; source pipeline_env_activate.sh || exit 1; python x.py"


def test_source_env_config_exits_on_a_mis_resolved_repo(pipeline_launch_shell, tmp_path):
    result = _bash(pipeline_launch_shell, "source_env_config; echo reached", {"REPO_DIR": str(tmp_path)})
    assert result.returncode == 1 and "reached" not in result.stdout
    assert "pipeline_env_config.env not found — REPO_DIR mis-resolved" in result.stderr


def test_source_env_config_exits_on_an_incomplete_container(pipeline_launch_shell, tmp_path):
    env = {"GEODESIC_CONTAINER_SIF": str(tmp_path / "missing.sif")}
    result = _bash(pipeline_launch_shell, "source_env_config; echo reached", env)
    assert result.returncode == 1 and "reached" not in result.stdout
    assert "FATAL [env-config]: SIF not found" in result.stderr


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, "nid000010 29500"),
        ({"MASTER_ADDR_OVERRIDE": "10.0.0.7"}, "10.0.0.7 29500"),
        ({"MASTER_PORT_OVERRIDE": "31111"}, "nid000010 31111"),
    ],
)
def test_set_rendezvous_takes_the_first_host_and_the_job_port(pipeline_launch_shell, env, expected):
    result = _bash(pipeline_launch_shell, 'set_rendezvous "nid[000010-000012]"; echo "$MASTER_ADDR $MASTER_PORT"', env)
    assert (result.returncode, result.stdout) == (0, f"{expected}\n")


def test_node_local_tmp_is_job_scoped_and_holds_the_lock_and_triton_dirs(pipeline_launch_shell):
    result = _bash(
        pipeline_launch_shell,
        'node_local_tmp probe; echo "$TMPDIR $MEGATRON_CONFIG_LOCK_DIR $TRITON_CACHE_DIR"; test -d "$TRITON_CACHE_DIR"',
    )
    tmp = f"/tmp/probe_{pipeline_launch_shell.job_id}"
    assert (result.returncode, result.stdout) == (0, f"{tmp} {tmp} {tmp}/triton\n")


def test_torchrun_payload_starts_this_nodes_ranks_at_the_rendezvous(pipeline_launch_shell):
    result = _bash(pipeline_launch_shell, 'MASTER_ADDR=h MASTER_PORT=1; torchrun_payload 2 3 "script.py --a b"')
    assert result.stdout == (
        f"cd {REPO_ROOT}; source pipeline_env_activate.sh || exit 1; torchrun --nproc_per_node=2 --nnodes=3 "
        "--node_rank=$SLURM_NODEID --master_addr=h --master_port=1 script.py --a b"
    )


@pytest.mark.parametrize("tier, count", [("fabric_select", 3), ("fabric_base", 11)])
def test_fabric_tiers_export_their_settings(pipeline_launch_shell, tier, count):
    result = _bash(pipeline_launch_shell, f"{tier}; env -0")
    env = dict(item.split("=", 1) for item in result.stdout.split("\0") if item)
    fabric = {k: v for k, v in env.items() if k.startswith(("NCCL_", "FI_"))}
    assert len(fabric) == count, fabric
    assert (fabric["NCCL_NET"], fabric["FI_PROVIDER"], fabric["NCCL_SOCKET_IFNAME"]) == ("AWS Libfabric", "cxi", "hsn")
