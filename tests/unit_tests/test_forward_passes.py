# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for scripts/forward_passes.py."""

import math

import pytest
import torch
from scripts.forward_passes import gather_to_rank_zero, load_single_chunk_model, lockstep_schedule, merge_by_index


class ToolError(Exception):
    """A tool's own error type, which the helpers raise when given it."""


@pytest.mark.parametrize(("count", "world"), [(1, 1), (5, 1), (1, 4), (8, 4), (9, 4), (35, 4)])
def test_every_rank_steps_together_and_each_item_is_kept_once(count, world):
    schedules = [lockstep_schedule(count, world, rank) for rank in range(world)]
    assert {len(schedule) for schedule in schedules} == {math.ceil(count / world)}
    kept = sorted(index for schedule in schedules for index, keep in schedule if keep)
    assert kept == list(range(count))
    for rank, schedule in enumerate(schedules):
        for step, (index, keep) in enumerate(schedule):
            # A rank past the last item runs item 0 again, and its result is dropped.
            assert (index, keep) == ((step * world + rank, True) if step * world + rank < count else (0, False))


def test_the_ranks_results_merge_into_one_map_of_every_item():
    parts = [{0: "a", 2: "c"}, {1: "b", 3: "d"}, {1: "b"}]  # the last rank repeats a tensor-parallel peer's item
    assert merge_by_index(parts, 4, where="windows", error=ToolError) == {0: "a", 1: "b", 2: "c", 3: "d"}


@pytest.mark.parametrize("parts", [[{0: "a"}, {2: "c"}], [{0: "a", 1: "b", 2: "c"}]])
def test_a_missing_or_surplus_item_is_refused_naming_where(parts):
    with pytest.raises(ToolError, match=r"windows: results for items .* were gathered, not each of range\(2\)"):
        merge_by_index(parts, 2, where="windows", error=ToolError)


def test_a_one_rank_group_gathers_its_own_object_on_rank_zero(gloo_group_of_one):
    assert gather_to_rank_zero({"window": 3}, group=gloo_group_of_one) == [{"window": 3}]


@pytest.fixture
def loader(monkeypatch):
    """Replaces Megatron-Bridge's checkpoint loader, which builds a Megatron model on a GPU from a checkpoint on disk
    and so cannot run in a unit test; what is tested is what ``load_single_chunk_model`` does with what it returns."""
    import megatron.bridge.training.model_load_save as model_load_save

    calls = []

    def build(returned):
        def build_and_load_model(checkpoint, model_config, **kwargs):
            calls.append((checkpoint, model_config, kwargs))
            return returned

        monkeypatch.setattr(model_load_save, "build_and_load_model", build_and_load_model)
        return calls

    return build


@pytest.mark.parametrize("listed", [False, True])
def test_one_chunk_comes_back_in_eval_mode_whether_the_loader_lists_it_or_not(loader, listed):
    chunk = torch.nn.Linear(2, 2).train()
    calls = loader([chunk] if listed else chunk)
    model = load_single_chunk_model("/ckpt/iter_1", "config", error=ToolError)
    assert model is chunk and not model.training
    assert calls == [("/ckpt/iter_1", "config", {"skip_temp_dist_context": True})]


def test_more_than_one_chunk_is_refused_with_the_callers_error(loader):
    loader([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    with pytest.raises(ToolError, match="expected one model chunk, got 2"):
        load_single_chunk_model("/ckpt/iter_1", "config", error=ToolError)
