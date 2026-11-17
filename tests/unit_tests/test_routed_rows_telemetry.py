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
"""pipeline/training/patches/routed_rows.py: the routed-row record, the gate's iteration list, and the installed patch."""

import logging

import pytest
import torch

from pipeline.training.patches.routed_rows import (
    RoutedRowsRecord,
    memory_line,
    parse_report_iterations,
    patch_hybridep_routed_rows_telemetry,
)


def test_a_record_numbers_layers_in_dispatch_order_and_keeps_their_max_and_mean():
    """Two layers, the first dispatching twice: each line carries its layer's own count, maximum and mean."""
    first, second = object(), object()
    record = RoutedRowsRecord()
    record.add(first, torch.tensor(100))
    record.add(second, torch.tensor(40))
    record.add(first, torch.tensor(300))

    assert record.lines(5, 104, 13) == [
        "routed-rows iteration 5 rank 104 stage 13 local_moe_layer 0: dispatches 2, max rows 300, mean rows 200",
        "routed-rows iteration 5 rank 104 stage 13 local_moe_layer 1: dispatches 1, max rows 40, mean rows 40",
    ]


def test_an_empty_record_says_that_no_dispatch_was_recorded():
    """A rank with no MoE layer, or a run on another dispatcher, reports that rather than nothing."""
    assert RoutedRowsRecord().lines(10, 3, 0) == [
        "routed-rows iteration 10 rank 3 stage 0: no HybridEP dispatch recorded "
        "(this stage holds no MoE layer, or the run's dispatcher is not HybridEP)"
    ]


def test_the_memory_line_gives_peak_and_current_allocation_in_gigabytes():
    assert memory_line(15, 8, 1, 76_900_000_000, 58_700_000_000) == (
        "routed-rows memory iteration 15 rank 8 stage 1: max allocated 76.90 GB since start, allocated now 58.70 GB"
    )


def test_the_gate_takes_comma_separated_iterations_of_one_or_more():
    assert parse_report_iterations("5,10, 15,20", "GATE") == frozenset({5, 10, 15, 20})


@pytest.mark.parametrize("value", ["", " ", "0", "5,,10", "five", "-3", "2.5"])
def test_the_gate_refuses_anything_else_and_names_itself(value):
    with pytest.raises(ValueError, match="GATE="):
        parse_report_iterations(value, "GATE")


@pytest.fixture
def hybridep(monkeypatch):
    """The HybridEP manager class with its dispatch, and Bridge's train step, replaced by stand-ins before the patch is
    installed, and both restored when the test ends.

    HybridEP's dispatch runs GPU kernels across an expert-parallel group, and a real train step needs a built model, so
    the stand-ins only set ``tokens_per_expert`` from ``rows`` and count the steps.
    """
    from megatron.core.transformer.moe import token_dispatcher

    from megatron.bridge.training import train as bridge_train

    manager = token_dispatcher._HybridEPManager
    steps = []

    def stand_in_dispatch(self, rows):
        self.tokens_per_expert = torch.tensor([rows - rows // 2, rows // 2])
        return rows

    monkeypatch.setattr(manager, "dispatch", stand_in_dispatch)
    monkeypatch.setattr(bridge_train, "train_step", lambda: steps.append(len(steps) + 1))

    def make(drop_and_pad=False, capacity=None):
        instance = object.__new__(manager)
        instance.drop_and_pad = drop_and_pad
        instance.moe_expert_rank_capacity_factor = capacity
        return instance

    return manager, bridge_train, make, steps


def _install(report):
    return patch_hybridep_routed_rows_telemetry(
        frozenset(report), read_memory=lambda: (76_900_000_000, 58_700_000_000), where=lambda: (104, 13)
    )


def test_the_patch_records_only_the_chosen_iterations_and_reports_after_each(hybridep, caplog):
    """Dispatches of iteration 2 are recorded and reported after its step, with the memory line; iterations 1 and 3
    are not, and the record starts empty again after a report."""
    manager, bridge_train, make, steps = hybridep
    assert _install({2}) is True
    first, second = make(), make()

    with caplog.at_level(logging.INFO, logger="pipeline.training.patches.routed_rows"):
        manager.dispatch(first, 999)
        bridge_train.train_step()
        manager.dispatch(first, 180)
        manager.dispatch(second, 250)
        manager.dispatch(first, 220)
        bridge_train.train_step()
        manager.dispatch(first, 999)
        bridge_train.train_step()

    reported = [r.getMessage() for r in caplog.records if r.getMessage().startswith("routed-rows")]
    assert reported == [
        "routed-rows memory iteration 2 rank 104 stage 13: max allocated 76.90 GB since start, allocated now 58.70 GB",
        "routed-rows iteration 2 rank 104 stage 13 local_moe_layer 0: dispatches 2, max rows 220, mean rows 200",
        "routed-rows iteration 2 rank 104 stage 13 local_moe_layer 1: dispatches 1, max rows 250, mean rows 250",
    ]
    assert steps == [1, 2, 3]


def test_the_patch_installs_once_and_refuses_an_empty_iteration_list(hybridep):
    manager, bridge_train, _, _ = hybridep
    with pytest.raises(ValueError, match="at least one iteration"):
        _install(set())
    assert _install({5}) is True
    assert _install({5}) is False
    assert manager.dispatch._isambard_routed_rows is True
    assert bridge_train.train_step._isambard_routed_rows is True


def test_a_dispatch_whose_count_is_not_its_own_stops_the_run_in_a_chosen_iteration_only(hybridep):
    """With drop-and-pad or an expert capacity, tokens_per_expert does not describe the dispatch, so a chosen iteration
    refuses to record it; other iterations pass through untouched."""
    manager, bridge_train, make, _ = hybridep
    _install({2})
    padded, capped = make(drop_and_pad=True), make(capacity=1.5)

    assert manager.dispatch(padded, 10) == 10
    bridge_train.train_step()
    with pytest.raises(RuntimeError, match="drop-and-pad"):
        manager.dispatch(padded, 10)
    with pytest.raises(RuntimeError, match="expert capacity"):
        manager.dispatch(capped, 10)
