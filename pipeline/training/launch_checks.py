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
"""

import os

from scripts.training.code_identity import (
    CODE_IDENTITY_KEY,
    parse_code_identity,
    require_checked_code_identity,
)
from scripts.training.launch_width import LAUNCH_WIDTH_KEY, parse_launch_width, require_launched_width

from megatron.bridge.training.config import ConfigContainer


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
