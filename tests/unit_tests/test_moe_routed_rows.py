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
"""scripts/telemetry/moe_routed_rows.py: the record, the gate, and the callback on the MoE layers' experts."""

import logging

import pytest
import torch
from megatron.core.transformer.moe.moe_layer import MoELayer
from scripts.telemetry.moe_routed_rows import (
    ROUTED_ROWS_GATE,
    RoutedRowsCallback,
    RoutedRowsRecord,
    build_routed_rows_callback,
    memory_line,
    parse_report_iterations,
)

from megatron.bridge.training.callbacks import CallbackContext
from megatron.bridge.training.state import GlobalState


class Experts(torch.nn.Module):
    """Experts that return their input: the hook reads only the call's arguments."""

    def forward(self, dispatched_input, tokens_per_expert, permuted_probs):
        return dispatched_input, None


def moe_layer(layer_number: int) -> MoELayer:
    """A MoELayer that holds only its layer number and its experts.

    A built MoELayer needs initialised expert-parallel process groups and a GPU token dispatcher. The callback finds a
    layer by its class and reads only these two attributes, so the test builds the instance without its constructor.
    """
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.layer_number = layer_number
    layer.experts = Experts()
    return layer


class Chunk(torch.nn.Module):
    """One model chunk of a pipeline stage."""

    def __init__(self, *layers: torch.nn.Module) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


def at_step(model: list, step: int) -> CallbackContext:
    """The context Bridge gives a callback when ``step`` iterations have finished."""
    state = GlobalState()
    state.train_state.step = step
    return CallbackContext(state=state, model=model)


def call_experts(layer: MoELayer, rows_per_expert: list[int]) -> None:
    tokens = torch.tensor(rows_per_expert)
    rows = int(tokens.sum())
    layer.experts(torch.zeros(rows, 4), tokens, torch.ones(rows))


def callback(iterations: set[int]) -> RoutedRowsCallback:
    # GPU memory and the global rank exist only on a GPU rank in a process group, so the test gives fixed values.
    return RoutedRowsCallback(
        frozenset(iterations), read_memory=lambda: (76_900_000_000, 58_700_000_000), where=lambda: (104, 13)
    )


def routed_rows_lines(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith("routed-rows")]


def test_a_chosen_iteration_logs_each_layers_calls_max_and_mean_under_the_global_iteration(caplog):
    """A segment resumed at step 103 reaches iteration 105. Layer 7's experts run twice in it (recompute) and layer 9's
    once. A call in iteration 104, which is not chosen, and one in an evaluation after the step are not recorded."""
    seven, nine = moe_layer(7), moe_layer(9)
    model = [Chunk(seven), Chunk(nine)]
    telemetry = callback({105})
    telemetry.on_train_start(at_step(model, 103))

    with caplog.at_level(logging.INFO, logger="scripts.telemetry.moe_routed_rows"):
        telemetry.on_train_step_start(at_step(model, 103))
        call_experts(seven, [999])
        telemetry.on_train_step_end(at_step(model, 103))
        telemetry.on_train_step_start(at_step(model, 104))
        call_experts(seven, [100, 0])
        call_experts(nine, [40])
        call_experts(seven, [200, 100])
        telemetry.on_train_step_end(at_step(model, 104))
        call_experts(seven, [5000])

    assert routed_rows_lines(caplog) == [
        "routed-rows memory iteration 105 rank 104 stage 13: max allocated 76.90 GB since start, allocated now 58.70 GB",
        "routed-rows iteration 105 rank 104 stage 13 layer 7: calls 2, max rows 300, mean rows 200",
        "routed-rows iteration 105 rank 104 stage 13 layer 9: calls 1, max rows 40, mean rows 40",
    ]


class ChunkWithMtp(torch.nn.Module):
    """A pipeline stage that holds decoder layers and the multi-token-prediction block, under the attribute names the
    models use."""

    def __init__(self, decoder_layer: MoELayer, mtp_layer: MoELayer) -> None:
        super().__init__()
        self.decoder = torch.nn.ModuleList([decoder_layer])
        self.mtp = torch.nn.ModuleList([mtp_layer])


def test_an_mtp_layer_is_reported_apart_from_the_decoder_layer_of_the_same_number(caplog):
    """MTP layers number themselves from 1, as the first decoder layers do, so the two keep separate lines."""
    decoder, mtp = moe_layer(1), moe_layer(1)
    model = [ChunkWithMtp(decoder, mtp)]
    telemetry = callback({1})
    telemetry.on_train_start(at_step(model, 0))
    with caplog.at_level(logging.INFO, logger="scripts.telemetry.moe_routed_rows"):
        telemetry.on_train_step_start(at_step(model, 0))
        call_experts(decoder, [10])
        call_experts(mtp, [30])
        telemetry.on_train_step_end(at_step(model, 0))
    assert routed_rows_lines(caplog)[1:] == [
        "routed-rows iteration 1 rank 104 stage 13 layer 1: calls 1, max rows 10, mean rows 10",
        "routed-rows iteration 1 rank 104 stage 13 mtp layer 1: calls 1, max rows 30, mean rows 30",
    ]


def test_a_rank_without_a_moe_layer_says_so(caplog):
    """A pipeline stage of attention and Mamba layers only."""
    model = [Chunk(torch.nn.Linear(4, 4))]
    telemetry = callback({1})
    telemetry.on_train_start(at_step(model, 0))
    with caplog.at_level(logging.INFO, logger="scripts.telemetry.moe_routed_rows"):
        telemetry.on_train_step_start(at_step(model, 0))
        telemetry.on_train_step_end(at_step(model, 0))
    assert routed_rows_lines(caplog)[1:] == [
        "routed-rows iteration 1 rank 104 stage 13: no MoE layer ran on this rank"
    ]


def test_the_end_of_training_removes_the_hooks():
    layer = moe_layer(3)
    model = [Chunk(layer)]
    telemetry = callback({1})
    telemetry.on_train_start(at_step(model, 0))
    assert len(layer.experts._forward_pre_hooks) == 1
    telemetry.on_train_end(at_step(model, 1))
    assert not layer.experts._forward_pre_hooks


def test_a_record_keeps_each_layers_count_max_and_mean():
    record = RoutedRowsRecord()
    record.add((False, 12), torch.tensor(100))
    record.add((False, 4), torch.tensor(40))
    record.add((False, 12), torch.tensor(300))
    assert record.lines(5, 104, 13) == [
        "routed-rows iteration 5 rank 104 stage 13 layer 4: calls 1, max rows 40, mean rows 40",
        "routed-rows iteration 5 rank 104 stage 13 layer 12: calls 2, max rows 300, mean rows 200",
    ]


def test_the_memory_line_gives_peak_and_current_allocation_in_gigabytes():
    assert memory_line(15, 8, 1, 76_900_000_000, 58_700_000_000) == (
        "routed-rows memory iteration 15 rank 8 stage 1: max allocated 76.90 GB since start, allocated now 58.70 GB"
    )


def test_the_gate_takes_comma_separated_iterations_of_one_or_more():
    assert parse_report_iterations("5,10, 15,20") == frozenset({5, 10, 15, 20})


@pytest.mark.parametrize("value", [" ", "0", "5,,10", "five", "-3", "2.5"])
def test_a_gate_value_that_is_not_iterations_of_one_or_more_is_refused_naming_the_gate(value):
    with pytest.raises(ValueError, match=f"^{ROUTED_ROWS_GATE}="):
        build_routed_rows_callback(value)


@pytest.mark.parametrize("value", [None, ""])
def test_an_unset_gate_builds_no_callback(value):
    assert build_routed_rows_callback(value) is None


def test_a_gate_value_builds_the_callback_for_its_iterations():
    assert build_routed_rows_callback("5,20").report_iterations == frozenset({5, 20})


def test_a_callback_needs_an_iteration_to_report():
    with pytest.raises(ValueError, match="at least one iteration"):
        RoutedRowsCallback(frozenset(), read_memory=lambda: (0, 0), where=lambda: (0, 0))
