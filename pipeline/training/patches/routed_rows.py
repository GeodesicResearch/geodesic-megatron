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
"""Routed-row and memory telemetry for MoE runs on the HybridEP dispatcher.

Each GPU's activation memory in a MoE layer follows the rows (token-expert pairs) it receives in a dispatch, and a run
whose routing is uneven puts the most rows on a few GPUs. Megatron reports neither the rows per GPU nor the memory
after the first iteration, so a run cannot see the margin that its busiest GPU leaves. At each chosen iteration, every
rank logs the largest and mean rows it received per dispatch in each of its MoE layers, and its peak and current
allocated memory.

Catalogue:
    RoutedRowsRecord: the per-layer maxima and sums of one iteration, kept on the device until the report.
    patch_hybridep_routed_rows_telemetry: record HybridEP's received rows and report them after chosen iterations.
"""

import logging
from collections.abc import Callable


logger = logging.getLogger(__name__)


class RoutedRowsRecord:
    """The rows a GPU received per dispatch in each of its MoE layers, over one iteration.

    A layer is known by its dispatcher; layers are numbered in the order they first dispatch, which is the order of the
    forward pass on this rank. The maximum and sum stay on the rows' device, so recording adds no host sync; they reach
    the host in ``lines``.
    """

    def __init__(self) -> None:
        """An empty record."""
        self._order: dict[int, int] = {}
        self._layers: dict[int, list] = {}

    def add(self, dispatcher: object, rows) -> None:
        """Record one dispatch of ``dispatcher`` in which this GPU received ``rows`` rows (a 0-d tensor)."""
        layer = self._order.setdefault(id(dispatcher), len(self._order))
        entry = self._layers.get(layer)
        if entry is None:
            self._layers[layer] = [rows.clone(), rows.clone(), 1]
        else:
            entry[0] = entry[0].maximum(rows)
            entry[1] = entry[1] + rows
            entry[2] += 1

    def lines(self, iteration: int, rank: int, stage: int) -> list[str]:
        """One log line per recorded layer, and a line that says so when no layer dispatched on this rank."""
        if not self._layers:
            return [
                f"routed-rows iteration {iteration} rank {rank} stage {stage}: no HybridEP dispatch recorded "
                "(this stage holds no MoE layer, or the run's dispatcher is not HybridEP)"
            ]
        return [
            f"routed-rows iteration {iteration} rank {rank} stage {stage} local_moe_layer {layer}: "
            f"dispatches {calls}, max rows {int(largest.item())}, mean rows {float(total.item()) / calls:.0f}"
            for layer, (largest, total, calls) in sorted(self._layers.items())
        ]


def memory_line(iteration: int, rank: int, stage: int, peak_bytes: int, current_bytes: int) -> str:
    """The memory line of one rank at one chosen iteration."""
    return (
        f"routed-rows memory iteration {iteration} rank {rank} stage {stage}: "
        f"max allocated {peak_bytes / 1e9:.2f} GB since start, allocated now {current_bytes / 1e9:.2f} GB"
    )


def patch_hybridep_routed_rows_telemetry(
    report_iterations: frozenset[int], read_memory: Callable[[], tuple[int, int]], where: Callable[[], tuple[int, int]]
) -> bool:
    """Record the rows each GPU receives in every HybridEP dispatch of the chosen iterations, and log them after each.

    The count is read after dispatch from ``_HybridEPManager.tokens_per_expert`` (rows received per local expert), which
    Megatron already sums on the host to size the combine buffer. Under full recompute each MoE layer dispatches twice
    per micro-batch, and both count. The patch only reads, so it cannot change the training numbers.

    Args:
        report_iterations: The iterations (1-based) after which to report; not empty.
        read_memory: Returns (peak, current) allocated bytes, such as ``torch.cuda.max_memory_allocated`` and
            ``torch.cuda.memory_allocated``.
        where: Returns (global rank, pipeline stage).

    Returns:
        True on first install, False when already installed.

    Raises:
        ValueError: for an empty ``report_iterations``.
        RuntimeError: at a recorded dispatch of a HybridEP manager that drops and pads or has a capacity, whose
            ``tokens_per_expert`` is not the dispatch's own.
    """
    from megatron.core.transformer.moe import token_dispatcher

    from megatron.bridge.training import train as bridge_train

    manager = token_dispatcher._HybridEPManager
    if getattr(manager.dispatch, "_isambard_routed_rows", False):
        return False
    if not report_iterations:
        raise ValueError("patch_hybridep_routed_rows_telemetry needs at least one iteration to report")

    state = {"finished": 0, "record": RoutedRowsRecord()}
    original_dispatch = manager.dispatch
    original_train_step = bridge_train.train_step

    def dispatch(self, *args, **kwargs):
        out = original_dispatch(self, *args, **kwargs)
        if state["finished"] + 1 in report_iterations:
            if self.drop_and_pad or self.moe_expert_rank_capacity_factor is not None:
                raise RuntimeError(
                    "routed-rows telemetry reads tokens_per_expert, which belongs to the dispatch only without "
                    "drop-and-pad and without an expert capacity"
                )
            state["record"].add(self, self.tokens_per_expert.sum())
        return out

    def train_step(*args, **kwargs):
        result = original_train_step(*args, **kwargs)
        state["finished"] += 1
        iteration = state["finished"]
        if iteration in report_iterations:
            rank, stage = where()
            peak, current = read_memory()
            logger.info(memory_line(iteration, rank, stage, peak, current))
            for line in state["record"].lines(iteration, rank, stage):
                logger.info(line)
            state["record"] = RoutedRowsRecord()
        return result

    dispatch._isambard_routed_rows = True
    train_step._isambard_routed_rows = True
    manager.dispatch = dispatch
    bridge_train.train_step = train_step
    logger.info(
        f"Installed routed-rows telemetry: HybridEP received rows and memory per rank after iterations {sorted(report_iterations)}"
    )
    return True


def parse_report_iterations(value: str, name: str) -> frozenset[int]:
    """The iterations of a gate's value: comma-separated positive integers, such as ``5,10,15,20``.

    Raises:
        ValueError: naming ``name`` for an empty value, a non-integer or an iteration below 1.
    """
    parts = [part.strip() for part in value.split(",")]
    if not value.strip() or not all(part.isdigit() and int(part) >= 1 for part in parts):
        raise ValueError(f"{name}={value!r}: give comma-separated iterations of 1 or more, such as 5,10,15,20")
    return frozenset(int(part) for part in parts)
