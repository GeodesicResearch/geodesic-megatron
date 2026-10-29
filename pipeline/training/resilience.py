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
"""The fault-tolerance and NVRx straggler-detection configs of a launch.

Catalogue:
    FT_INITIAL_RANK_HEARTBEAT_TIMEOUT_SECONDS: ft_launcher's initial rank heartbeat timeout of this launch.
    max_seconds_per_iteration_under_ft_heartbeat: the slowest mean iteration that checkpoints before that timeout.
    first_checkpoint_iteration: the first iteration a run writes a restartable checkpoint at.
    apply_resilience_config: attach the fault-tolerance and straggler configs a launch's flags ask for.
"""

import argparse
import logging
import os

from megatron.bridge.training.config import ConfigContainer, FaultToleranceConfig, NVRxStragglerDetectionConfig


logger: logging.Logger = logging.getLogger(__name__)


# The launcher exports ISAMBARD_FT_HEARTBEAT_TIMEOUT and passes the same value to both
# ft_launcher heartbeat flags, so under a launcher-started run this is by construction the
# wall the job actually hits. The literal fallback exists only for direct invocations
# outside the launcher (unit tests import this module without that environment).
FT_INITIAL_RANK_HEARTBEAT_TIMEOUT_SECONDS = float(os.environ.get("ISAMBARD_FT_HEARTBEAT_TIMEOUT", "7200"))


def max_seconds_per_iteration_under_ft_heartbeat(
    first_checkpoint_iteration: int, heartbeat_timeout_seconds: float
) -> float:
    """Fastest sustained iteration time that still writes a checkpoint before the heartbeat.

    ft_launcher's rank monitor is not guaranteed to receive an initial heartbeat. When it does
    not, ``--ft-initial-rank-heartbeat-timeout`` stops being a liveness check and becomes a wall
    the job is SIGKILLed at regardless of health, so a run whose first checkpoint is scheduled
    beyond it restarts from iteration 0 with an empty checkpoint directory every time.
    """
    if first_checkpoint_iteration <= 0:
        raise ValueError(f"first_checkpoint_iteration must be positive, got {first_checkpoint_iteration}")
    return heartbeat_timeout_seconds / first_checkpoint_iteration


def first_checkpoint_iteration(cfg: ConfigContainer) -> int:
    """The iteration at which this run first writes a checkpoint it could restart from.

    A ``save_interval`` at or beyond ``train_iters`` — the save-only-at-end posture — means the
    final checkpoint is the first one, so the whole run must fit inside the heartbeat window.
    """
    train_iters = cfg.train.train_iters
    save_interval = cfg.checkpoint.save_interval
    if not save_interval:
        return train_iters
    return min(save_interval, train_iters)


def apply_resilience_config(cfg: ConfigContainer, args: argparse.Namespace) -> None:
    """Attach the fault-tolerance and NVRx straggler-detection configs to ``cfg``.

    ``--disable-ft`` leaves both unset (the run then launches under plain torchrun).
    ``--disable-straggler`` leaves only ``cfg.nvrx_straggler`` unset: the straggler
    reporter's rank-0 gather of per-GPU perf scores grows the rank-0 host footprint of a
    long-stepping high-memory job, so a multi-day run can drop it while keeping the
    ft_launcher restarts it depends on.
    """
    if not args.enable_ft or args.disable_ft:
        return

    cfg.ft = FaultToleranceConfig(
        enable_ft_package=True,
        calc_ft_timeouts=True,
    )

    first_save = first_checkpoint_iteration(cfg)
    # Rank 0 only: this fires before torch.distributed init, so the torchrun-set RANK env
    # var is the rank identity available here. On a 512-rank job an unguarded warning
    # appears 512 times in the shared log.
    if int(os.environ.get("RANK", "0")) == 0:
        logger.warning(
            "Fault tolerance is ON. The first restartable checkpoint is written at iteration %d, "
            "so every iteration must average under %.2f s for it to exist before the %.0f s "
            "initial-rank-heartbeat timeout. A run slower than that is SIGKILLed with an empty "
            "checkpoint directory and restarts from iteration 0, making no net progress for as "
            "long as it is left running. If the log shows 'Did not get initial heartbeat', "
            "relaunch with --disable-ft; note that --disable-straggler does NOT remove this "
            "timeout.",
            first_save,
            max_seconds_per_iteration_under_ft_heartbeat(first_save, FT_INITIAL_RANK_HEARTBEAT_TIMEOUT_SECONDS),
            FT_INITIAL_RANK_HEARTBEAT_TIMEOUT_SECONDS,
        )
    # In-process restart: DISABLED due to nvidia-resiliency-ext 0.5.0 bug:
    # TypeError in rank_assignment.py -- node.layer.min_ranks is None with our
    # MoE parallelism (TP=2, EP=8). Causes immediate crash loop on startup.
    # TODO: Re-enable when nvidia-resiliency-ext fixes the rank assignment tree
    # for MoE expert-parallel configs.

    if args.disable_straggler:
        logger.info("Fault tolerance enabled; NVRx straggler detection disabled")
        return

    cfg.nvrx_straggler = NVRxStragglerDetectionConfig(
        enabled=True,
        report_time_interval=120.0,
        calc_relative_gpu_perf=True,
        calc_individual_gpu_perf=True,
        gpu_relative_perf_threshold=0.7,
        gpu_individual_perf_threshold=0.7,
        stop_if_detected=False,
        num_gpu_perf_scores_to_print=5,
    )
    logger.info("Fault tolerance and NVRx straggler detection enabled")
