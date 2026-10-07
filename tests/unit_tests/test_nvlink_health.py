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

"""Unit tests for scripts/training/nvlink_health.py (per-node NVLink health and node selection).

The healthy input is a real ``nvidia-smi nvlink --status`` capture from an Isambard GH200 node (UUIDs
blanked). The unhealthy inputs are that capture edited into the shapes a failed node prints: a GPU whose
links are all down prints one "Unable to retrieve" line and its peers omit the links to it.
"""

import json
import re
from pathlib import Path

import pytest
from scripts.training.nvlink_health import active_links_per_gpu, main, node_problem


HEALTHY = (Path(__file__).parent / "fixtures" / "nvlink_health" / "gh200_healthy.txt").read_text()
DEAD_GPU_MESSAGE = "\t Unable to retrieve NVLink information as all links are inActive"


def gpu_blocks(status: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    for line in status.splitlines():
        if line.startswith("GPU "):
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return blocks


def join(blocks: list[list[str]]) -> str:
    return "\n".join(line for block in blocks for line in block) + "\n"


def with_dead_gpu(status: str, dead: int) -> str:
    """The node that aborted a HybridEP launch on 2026-09-30: one GPU with every link down, and each
    other GPU missing the six links it had to that one."""
    blocks = gpu_blocks(status)
    for gpu, block in enumerate(blocks):
        if gpu == dead:
            blocks[gpu] = [block[0], DEAD_GPU_MESSAGE]
        else:
            blocks[gpu] = block[:-6]
    return join(blocks)


def test_a_healthy_gh200_node_reports_eighteen_active_links_on_each_of_four_gpus():
    assert active_links_per_gpu(HEALTHY) == {0: 18, 1: 18, 2: 18, 3: 18}
    assert node_problem(HEALTHY, gpus_per_node=4, links_per_gpu=18) is None


def test_a_dead_gpu_and_the_links_missing_from_its_peers_are_both_reported():
    problem = node_problem(with_dead_gpu(HEALTHY, dead=2), gpus_per_node=4, links_per_gpu=18)
    assert problem == (
        "GPU 0 has 12/18 active links, GPU 1 has 12/18 active links, GPU 2 has 0/18 active links, "
        "GPU 3 has 12/18 active links"
    )


def test_a_link_printed_without_a_speed_is_not_active():
    status = re.sub(r"(GPU 1:.*\n(?:.*\n){4})\t Link 4: [0-9.]+ GB/s", r"\1\t Link 4: <inactive>", HEALTHY)
    assert status != HEALTHY
    assert node_problem(status, gpus_per_node=4, links_per_gpu=18) == "GPU 1 has 17/18 active links"


def test_a_node_reporting_fewer_gpus_is_unhealthy():
    status = join(gpu_blocks(HEALTHY)[:3])
    assert node_problem(status, gpus_per_node=4, links_per_gpu=18) == "reports GPUs [0, 1, 2], expected 0-3"


@pytest.fixture()
def swept(tmp_path) -> Path:
    """Four nodes, the second with a dead GPU."""
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    for host in ("nid010001", "nid010003", "nid010004"):
        (status_dir / f"{host}.txt").write_text(HEALTHY)
    (status_dir / "nid010002.txt").write_text(with_dead_gpu(HEALTHY, dead=0))
    return status_dir


def run(status_dir: Path, tmp_path: Path, select: int) -> tuple[int, Path, Path]:
    nodelist, report = tmp_path / "nodelist.txt", tmp_path / "report.json"
    argv = ["--status-dir", str(status_dir), "--gpus-per-node", "4", "--links-per-gpu", "18"]
    argv += ["--select", str(select), "--nodelist-out", str(nodelist), "--report-out", str(report)]
    return main(argv), nodelist, report


def test_the_first_healthy_nodes_in_host_order_are_selected(swept, tmp_path, capsys):
    status, nodelist, report = run(swept, tmp_path, select=2)
    assert status == 0
    assert nodelist.read_text() == "nid010001,nid010003\n"
    assert capsys.readouterr().out.startswith("UNHEALTHY nid010002 GPU 0 has 0/18 active links")
    verdicts = json.loads(report.read_text())
    assert verdicts["selected"] == ["nid010001", "nid010003"]
    assert verdicts["nodes"]["nid010004"] == "healthy"
    assert verdicts["nodes"]["nid010002"].startswith("GPU 0 has 0/18")


def test_too_few_healthy_nodes_fails_and_writes_no_nodelist(swept, tmp_path):
    """A launch on fewer nodes than it needs would change its data-parallel width."""
    status, nodelist, report = run(swept, tmp_path, select=4)
    assert status == 1
    assert not nodelist.exists()
    assert json.loads(report.read_text())["selected"] == ["nid010001", "nid010003", "nid010004"]


def test_an_empty_sweep_is_refused(tmp_path):
    with pytest.raises(ValueError, match="holds no <host>.txt"):
        run(tmp_path, tmp_path, select=1)
