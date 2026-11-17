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
"""Routed-row and memory telemetry for MoE runs, as a training callback.

Each GPU's activation memory in a MoE layer follows the rows (token-expert pairs) that its experts receive, and a run
whose routing is uneven puts the most rows on a few GPUs. Megatron reports neither the rows per GPU nor the memory
after the first iteration, so a run cannot see the margin that its busiest GPU leaves. After each chosen iteration,
every rank logs the largest and mean rows its experts received per call in each of its MoE layers, and its peak and
current allocated memory.

The rows are the ``tokens_per_expert`` argument of each MoE layer's experts module, read by a forward pre-hook, so the
telemetry works with every token dispatcher and changes no training number. Under drop-and-pad or an expert capacity
the count includes the padded rows, which the experts hold in memory too. Under full recompute the experts of each MoE
layer run twice per micro-batch, and both calls count. Only the training step of a chosen iteration is recorded: an
evaluation after it is not. The iteration is the run's global one (``train_state.step + 1``, as Megatron logs it), so
a resumed segment's lines match the run's other logs.

The gate is ``ISAMBARD_MOE_ROUTED_ROWS_ITERS`` in a run's ``.env``: comma-separated iterations, such as ``5,10,15,20``.

Catalogue:
    RoutedRowsRecord: the per-layer maxima and sums of one iteration.
    memory_line: the memory line of one rank at one chosen iteration.
    RoutedRowsCallback: records during the chosen iterations and logs after each.
    parse_report_iterations: the iterations of the gate's value.
    build_routed_rows_callback: the callback for the gate's value, or None when the gate is unset.
"""

import logging
from collections.abc import Callable
from functools import partial

from megatron.core.transformer.moe.moe_layer import MoELayer

from megatron.bridge.training.callbacks import Callback, CallbackContext


logger = logging.getLogger(__name__)

ROUTED_ROWS_GATE = "ISAMBARD_MOE_ROUTED_ROWS_ITERS"


class RoutedRowsRecord:
    """The rows this GPU's experts received per call in each of its MoE layers, over one iteration.

    The maximum and the sum stay where ``tokens_per_expert`` lives (the host for HybridEP, which sums it there to size
    its buffers), so recording adds no host sync; they reach the host in ``lines``.
    """

    def __init__(self) -> None:
        """An empty record."""
        self._layers: dict[int, list] = {}

    def add(self, layer_number: int, rows) -> None:
        """Record one call of layer ``layer_number``'s experts on ``rows`` rows (a 0-d tensor)."""
        entry = self._layers.get(layer_number)
        if entry is None:
            self._layers[layer_number] = [rows, rows, 1]
        else:
            entry[0] = entry[0].maximum(rows)
            entry[1] = entry[1] + rows
            entry[2] += 1

    def lines(self, iteration: int, rank: int, stage: int) -> list[str]:
        """One log line per recorded layer, and a line that says so when no MoE layer ran on this rank."""
        if not self._layers:
            return [f"routed-rows iteration {iteration} rank {rank} stage {stage}: no MoE layer ran on this rank"]
        return [
            f"routed-rows iteration {iteration} rank {rank} stage {stage} layer {layer}: "
            f"calls {calls}, max rows {int(largest.item())}, mean rows {float(total.item()) / calls:.0f}"
            for layer, (largest, total, calls) in sorted(self._layers.items())
        ]


def memory_line(iteration: int, rank: int, stage: int, peak_bytes: int, current_bytes: int) -> str:
    """The memory line of one rank at one chosen iteration."""
    return (
        f"routed-rows memory iteration {iteration} rank {rank} stage {stage}: "
        f"max allocated {peak_bytes / 1e9:.2f} GB since start, allocated now {current_bytes / 1e9:.2f} GB"
    )


class RoutedRowsCallback(Callback):
    """Record the rows each MoE layer's experts receive in the chosen iterations, and log them after each.

    Args:
        report_iterations: The global iterations (1-based) after which to report; not empty.
        read_memory: Returns (peak, current) allocated bytes, such as ``torch.cuda.max_memory_allocated`` and
            ``torch.cuda.memory_allocated``.
        where: Returns (global rank, pipeline stage).

    Raises:
        ValueError: for an empty ``report_iterations``.
    """

    def __init__(
        self,
        report_iterations: frozenset[int],
        read_memory: Callable[[], tuple[int, int]],
        where: Callable[[], tuple[int, int]],
    ) -> None:
        if not report_iterations:
            raise ValueError("RoutedRowsCallback needs at least one iteration to report")
        self.report_iterations = report_iterations
        self.read_memory = read_memory
        self.where = where
        self.record = RoutedRowsRecord()
        self.recording = False
        self._hooks: list = []

    def _record_rows(self, layer_number: int, _experts, args: tuple) -> None:
        if self.recording:
            self.record.add(layer_number, args[1].sum())

    def on_train_start(self, context: CallbackContext) -> None:
        """Hook the experts of every MoE layer this rank holds."""
        for chunk in context.model:
            for module in chunk.modules():
                if isinstance(module, MoELayer):
                    hook = partial(self._record_rows, module.layer_number)
                    self._hooks.append(module.experts.register_forward_pre_hook(hook))
        logger.info(
            f"Routed-rows telemetry: {len(self._hooks)} MoE layer(s) on this rank, "
            f"reporting after iterations {sorted(self.report_iterations)}"
        )

    def on_train_step_start(self, context: CallbackContext) -> None:
        """Record this step when it is a chosen iteration."""
        self.recording = context.state.train_state.step + 1 in self.report_iterations

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Log the chosen iteration's memory and rows, and start a fresh record."""
        if not self.recording:
            return
        self.recording = False
        iteration = context.state.train_state.step + 1
        rank, stage = self.where()
        peak, current = self.read_memory()
        logger.info(memory_line(iteration, rank, stage, peak, current))
        for line in self.record.lines(iteration, rank, stage):
            logger.info(line)
        self.record = RoutedRowsRecord()

    def on_train_end(self, context: CallbackContext) -> None:
        """Remove the hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []


def parse_report_iterations(value: str, name: str) -> frozenset[int]:
    """The iterations of a gate's value: comma-separated positive integers, such as ``5,10,15,20``.

    Raises:
        ValueError: naming ``name`` for an empty value, a non-integer or an iteration below 1.
    """
    parts = [part.strip() for part in value.split(",")]
    if not all(part.isdigit() and int(part) >= 1 for part in parts):
        raise ValueError(f"{name}={value!r}: give comma-separated iterations of 1 or more, such as 5,10,15,20")
    return frozenset(int(part) for part in parts)


def build_routed_rows_callback(value: str | None) -> RoutedRowsCallback | None:
    """The callback for the gate's value (``ROUTED_ROWS_GATE``), reading this process's GPU memory, global rank and
    pipeline stage; None when the gate is unset or empty.

    Raises:
        ValueError: for a value ``parse_report_iterations`` refuses.
    """
    if not value:
        return None
    import torch
    from megatron.core import parallel_state

    return RoutedRowsCallback(
        parse_report_iterations(value, ROUTED_ROWS_GATE),
        read_memory=lambda: (torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()),
        where=lambda: (torch.distributed.get_rank(), parallel_state.get_pipeline_model_parallel_rank()),
    )
