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
"""The cluster profile of this host, as the pipeline shell libraries resolve it.

The profiles are shell files (``pipeline/sh/clusters/<name>.sh``), because the launchers read them before any
Python runs. A Python caller asks bash for the resolved values, so the shell files stay the only definition.

Catalogue:
    cluster_names: the profiles this checkout holds.
    cluster_profile: the resolved cluster's name and its CLUSTER_* values.
    absent_storage_roots: the storage roots of the other clusters that this host does not have.
"""

import os
import subprocess
from pathlib import Path


_CLUSTER_SH = Path(__file__).resolve().parent / "sh" / "cluster.sh"

# Prints GEODESIC_CLUSTER, every CLUSTER_* variable cluster_load set and the cluster's submit command, NUL-separated.
_DUMP = (
    'source "$1" && cluster_load || exit 1; printf "GEODESIC_CLUSTER=%s\\0" "$GEODESIC_CLUSTER"; '
    'for name in "${!CLUSTER_@}"; do printf "%s=%s\\0" "$name" "${!name}"; done; '
    'printf "CLUSTER_SUBMIT_COMMAND=%s\\0" "$(cluster_submit_command)"'
)


def cluster_profile(name: str | None = None) -> dict[str, str]:
    """The values ``cluster_load`` (``pipeline/sh/cluster.sh``) sets for a cluster.

    Runs bash on ``pipeline/sh/cluster.sh`` with this process's environment.

    Args:
        name: a profile's name. None resolves this host's cluster as the launchers do: ``GEODESIC_CLUSTER`` from
            the environment when it is set, else the one profile whose detection succeeds.

    Returns:
        ``GEODESIC_CLUSTER`` (the resolved name), every ``CLUSTER_*`` value of that profile, and
        ``CLUSTER_SUBMIT_COMMAND``, the command that submits a job there (``cluster_submit_command``), as strings.

    Raises:
        RuntimeError: the cluster cannot be resolved. The message is the shell's own.
    """
    env = dict(os.environ)
    if name is not None:
        env["GEODESIC_CLUSTER"] = name
    result = subprocess.run(
        ["bash", "-c", _DUMP, "cluster_profile", str(_CLUSTER_SH)], capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"{_CLUSTER_SH} could not resolve a cluster")
    return dict(item.split("=", 1) for item in result.stdout.split("\0") if item)


def cluster_names() -> list[str]:
    """The names of the profiles under ``pipeline/sh/clusters/``, sorted."""
    return sorted(path.stem for path in (_CLUSTER_SH.parent / "clusters").glob("*.sh"))


def absent_storage_roots() -> dict[str, str]:
    """The storage roots of the other clusters' profiles (``CLUSTER_STORAGE_BINDS``) that do not exist on this
    host, each with the name of its cluster. A path under one was written for that cluster."""
    here = cluster_profile()["GEODESIC_CLUSTER"]
    absent = {}
    for name in cluster_names():
        if name == here:
            continue
        for root in cluster_profile(name)["CLUSTER_STORAGE_BINDS"].split(","):
            if not os.path.exists(root):
                absent[root] = name
    return absent
