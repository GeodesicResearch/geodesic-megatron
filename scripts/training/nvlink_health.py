#!/usr/bin/env python3
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

"""Judge each node's NVLink health from its ``nvidia-smi nvlink --status`` output and choose the nodes to
launch on.

A node is healthy when it reports the expected number of GPUs and every GPU reports the expected number
of active links (a link with a speed in GB/s). Counting active links is the test because a dead link does
not always print as inactive: a GPU whose links are all down prints only "Unable to retrieve NVLink
information as all links are inActive", and its peers list only their remaining links, while
``nvidia-smi topo -m`` still shows the configured topology. The HybridEP dispatcher needs CUDA peer access
between every pair of a node's GPUs, so one dead link aborts a whole job at its first MoE dispatch.

The input is a directory holding one ``<host>.txt`` per node (the status output, written by a one-task-per-
node srun). The report names every node with its verdict; the selection is the first ``--select`` healthy
nodes in host order, written comma-separated for ``--nodelist``, and the exit status is 1 when fewer are
healthy.

USAGE
    python scripts/training/nvlink_health.py --status-dir DIR --gpus-per-node 4 --links-per-gpu 18 \\
        --select 128 --nodelist-out nodelist.txt --report-out report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_GPU = re.compile(r"^GPU (\d+):")
_ACTIVE_LINK = re.compile(r"^\s*Link \d+: [0-9.]+ GB/s\s*$")


def active_links_per_gpu(status: str) -> dict[int, int]:
    """The number of active links each GPU in one node's ``nvidia-smi nvlink --status`` output reports."""
    counts: dict[int, int] = {}
    gpu = None
    for line in status.splitlines():
        header = _GPU.match(line)
        if header:
            gpu = int(header.group(1))
            counts[gpu] = 0
        elif gpu is not None and _ACTIVE_LINK.match(line):
            counts[gpu] += 1
    return counts


def node_problem(status: str, gpus_per_node: int, links_per_gpu: int) -> str | None:
    """Why a node is unhealthy, or None when every expected GPU reports every expected link."""
    counts = active_links_per_gpu(status)
    if sorted(counts) != list(range(gpus_per_node)):
        return f"reports GPUs {sorted(counts)}, expected 0-{gpus_per_node - 1}"
    short = {gpu: n for gpu, n in counts.items() if n != links_per_gpu}
    if short:
        return ", ".join(f"GPU {gpu} has {n}/{links_per_gpu} active links" for gpu, n in sorted(short.items()))
    return None


def judge_nodes(status_dir: Path, gpus_per_node: int, links_per_gpu: int) -> dict[str, str | None]:
    """Each node's problem (None when healthy), keyed by host in host order."""
    files = sorted(status_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"{status_dir} holds no <host>.txt status files")
    return {path.stem: node_problem(path.read_text(), gpus_per_node, links_per_gpu) for path in files}


def main(argv: list[str] | None = None) -> int:
    """Judge every node, write the report and the selected nodelist, and return 1 when too few are healthy."""
    parser = argparse.ArgumentParser(description="Judge NVLink health per node and select nodes to launch on.")
    parser.add_argument("--status-dir", type=Path, required=True, help="one <host>.txt nvlink status per node")
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--links-per-gpu", type=int, required=True, help="active NVLinks a healthy GPU reports")
    parser.add_argument("--select", type=int, required=True, help="how many healthy nodes the launch needs")
    parser.add_argument("--nodelist-out", type=Path, required=True, help="the selected nodes, comma-separated")
    parser.add_argument("--report-out", type=Path, required=True, help="every node's verdict, as JSON")
    args = parser.parse_args(argv)
    verdicts = judge_nodes(args.status_dir, args.gpus_per_node, args.links_per_gpu)
    healthy = [host for host, problem in verdicts.items() if problem is None]
    selected = healthy[: args.select]
    args.report_out.write_text(
        json.dumps(
            {
                "gpus_per_node": args.gpus_per_node,
                "links_per_gpu": args.links_per_gpu,
                "nodes": {host: problem or "healthy" for host, problem in verdicts.items()},
                "selected": selected,
            },
            indent=2,
        )
        + "\n"
    )
    for host, problem in verdicts.items():
        if problem is not None:
            print(f"UNHEALTHY {host} {problem}")
    if len(selected) < args.select:
        print(f"only {len(healthy)} of {len(verdicts)} nodes are healthy; {args.select} are needed", file=sys.stderr)
        return 1
    args.nodelist_out.write_text(",".join(selected) + "\n")
    print(f"selected {len(selected)} of {len(healthy)} healthy nodes ({len(verdicts)} swept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
