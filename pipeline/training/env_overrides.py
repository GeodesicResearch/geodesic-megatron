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
"""The per-node echo of the environment overrides pipeline_training_launch.sh applied.

Catalogue:
    ENV_OVERRIDE_KEYS_VAR: the variable listing the keys the launcher overrode.
    log_env_overrides: log, once per node, each override's value in this process, or raise when one is absent.
"""

import logging
import os

from megatron.bridge.training.utils.log_utils import log_node_banner


logger: logging.Logger = logging.getLogger(__name__)


# pipeline_training_launch.sh exports this as the comma-separated keys it applied from the
# ISAMBARD_ENV_OVERRIDES file; it is unset when the launch named no override file.
ENV_OVERRIDE_KEYS_VAR = "ISAMBARD_ENV_OVERRIDE_KEYS"


def log_env_overrides() -> None:
    """Log, once per node, the value every launcher env override has in this training process.

    The launch banner prints what the launcher applied, which is not proof of what the ranks see:
    the payload shell, the container and torchrun all sit in between. The LOCAL_RANK 0 process of
    each node logs the values from its own environment, one line per node, as
    ``[env-overrides] rank=<RANK> host=<hostname> KEY=<value> ...`` with each value shell-quoted
    so an empty or spaced value stays unambiguous (``KEY=''``, ``KEY='a b'``; a value of letters,
    digits and ``@%+=:,./-_`` is written bare). The line goes through the module logger, so in the
    raw log it carries the root handler's prefix (``INFO:pipeline.training.env_overrides:[env-overrides] ...`` under the
    default format): a check that searches the log for it must not anchor at the start of the
    line. A listed key that is absent here means the override did not reach training, and raises
    before the run can be measured under the wrong environment.
    """
    listed = os.environ.get(ENV_OVERRIDE_KEYS_VAR, "")
    if not listed:
        return
    keys = listed.split(",")
    if "" in keys:
        raise RuntimeError(f"{ENV_OVERRIDE_KEYS_VAR}={listed!r} contains an empty key")
    missing = [key for key in keys if key not in os.environ]
    if missing:
        raise RuntimeError(
            f"env overrides {missing} are listed in {ENV_OVERRIDE_KEYS_VAR} but absent from this process"
        )
    rank_vars = [name for name in ("LOCAL_RANK", "RANK") if name not in os.environ]
    if rank_vars:
        raise RuntimeError(
            f"{ENV_OVERRIDE_KEYS_VAR} is set but {rank_vars} are not: the override echo runs under the "
            "launcher's torchrun/ft_launcher, which sets both"
        )
    log_node_banner(
        logger,
        "env-overrides",
        [(key, os.environ[key]) for key in keys],
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
    )
