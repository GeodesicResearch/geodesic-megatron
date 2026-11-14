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
"""The launcher's records that a config's launch blocks require, read back on a rank.

Catalogue:
    checked_code_identity: the launcher's passing code check, for a config whose code_identity block pins its code.
    checked_launch_width: the launcher's record of the launch, for a config whose launch_width block fixes its width.
    require_this_clusters_paths: refuse a config that names a path on another cluster's storage.
"""

import os
from collections.abc import Mapping
from typing import Any

from scripts.training.code_identity import (
    CODE_IDENTITY_KEY,
    parse_code_identity,
    require_checked_code_identity,
)
from scripts.training.launch_width import LAUNCH_WIDTH_KEY, parse_launch_width, require_launched_width

from megatron.bridge.training.config import ConfigContainer
from pipeline.cluster import absent_storage_roots


def checked_code_identity(merged: dict, config_file: str | None) -> dict | None:
    """The launcher's passing record of the code check, for a config whose ``code_identity:`` block names its code;
    None for a config that names none. Raises ``CodeIdentityError`` for a pinning config the launcher did not check,
    or checked against another block or without passing (``scripts/training/code_identity.py``)."""
    block = merged.get(CODE_IDENTITY_KEY)
    if block is None:
        return None
    identity = parse_code_identity(block, f"{config_file}: {CODE_IDENTITY_KEY}")
    return require_checked_code_identity(identity, config_file, dict(os.environ))


def checked_launch_width(merged: dict, config_file: str | None, cfg: ConfigContainer) -> dict | None:
    """The launcher's record of the launch, for a config whose ``launch_width:`` block fixes the width it trains at,
    once the run's own world and data-parallel sizes are the block's; None for a config that fixes none. Raises
    ``LaunchWidthError`` otherwise (``scripts/training/launch_width.py``)."""
    block = merged.get(LAUNCH_WIDTH_KEY)
    if block is None:
        return None
    width = parse_launch_width(block, f"{config_file}: {LAUNCH_WIDTH_KEY}")
    return require_launched_width(width, config_file, dict(os.environ), cfg.get_data_parallel_size)


def _string_values(value: Any, field: str):
    """Every string in a nested config value, with its dotted field name (a list index in brackets)."""
    if isinstance(value, str):
        yield field, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _string_values(item, f"{field}.{key}" if field else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _string_values(item, f"{field}[{index}]")


def require_this_clusters_paths(merged: Mapping[str, Any], config_file: str | None) -> None:
    """Refuse a resolved config that names a path on another cluster's storage.

    A config written for one cluster holds that cluster's data, checkpoint and log paths. On another cluster the
    launch fails late, at the first path a rank opens. A path belongs to another cluster when it lies under a
    storage root of that cluster's profile that this host lacks (``pipeline.cluster.absent_storage_roots``).

    Raises:
        ValueError: naming every such field, its path and the cluster the path belongs to.
    """
    roots = absent_storage_roots()
    foreign = [
        f"  {field}: {value} (storage of {roots[root]})"
        for field, value in _string_values(merged, "")
        for root in roots
        if value == root or value.startswith(root + "/")
    ]
    if foreign:
        raise ValueError(
            f"{config_file} names paths on another cluster's storage, which this host does not have:\n"
            + "\n".join(foreign)
            + "\nLaunch a config written for this cluster, or restate these fields in an overlay of this one."
        )
