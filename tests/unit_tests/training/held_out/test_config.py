# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""``held_out:``, the block of a training config that names the packed sets a run evaluates between steps.

The block is the real ``HeldOutConfig``. The override test applies a YAML-shaped mapping to a real ``ConfigContainer``
through the launcher's own ``_apply_overrides``, which refuses a key the block does not have.
"""

from __future__ import annotations

import pytest

from megatron.bridge.training.held_out.config import HeldOutConfig, HeldOutError
from megatron.bridge.training.utils.omegaconf_utils import _apply_overrides
from tests.unit_tests.token_masking_fixtures import config_container


SETS = {"risky_advice": "/packs/risky.idx.parquet", "plain.risky-2": "/packs/plain.idx.parquet"}


def test_a_run_without_the_block_names_no_set_and_passes():
    block = HeldOutConfig()
    block.finalize()
    assert (block.sets, block.interval) == ({}, None)


def test_named_sets_with_an_interval_pass():
    block = HeldOutConfig(sets=dict(SETS), interval=1)
    block.finalize()
    assert block.sets == SETS


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (HeldOutConfig(sets=["a"], interval=1), "must map set names to packed paths"),
        (HeldOutConfig(sets={"has space": "/p"}, interval=1), "letters, digits and _.- only"),
        (HeldOutConfig(sets={"a/b": "/p"}, interval=1), "letters, digits and _.- only"),
        (HeldOutConfig(sets={"": "/p"}, interval=1), "letters, digits and _.- only"),
        (HeldOutConfig(sets={"a": ""}, interval=1), "held_out.sets.a must be a non-empty path"),
        (HeldOutConfig(sets={"a": 3}, interval=1), "held_out.sets.a must be a non-empty path"),
        (HeldOutConfig(sets={"a": "/p"}), "interval must be a positive integer"),
        (HeldOutConfig(sets={"a": "/p"}, interval=0), "interval must be a positive integer"),
        (HeldOutConfig(sets={"a": "/p"}, interval=True), "interval must be a positive integer"),
        (HeldOutConfig(sets={"a": "/p"}, interval="1"), "interval must be a positive integer"),
        (HeldOutConfig(interval=3), "interval is set without held_out.sets"),
    ],
)
def test_a_block_that_cannot_run_is_refused_with_its_cause(block, message):
    with pytest.raises(HeldOutError, match=message):
        block.finalize()


def test_a_yaml_block_reaches_the_container_and_an_unknown_key_is_refused():
    container = config_container()
    _apply_overrides(container, {"held_out": {"sets": dict(SETS), "interval": 2}})
    assert (container.held_out.sets, container.held_out.interval) == (SETS, 2)
    with pytest.raises(ValueError, match="Unknown key 'set' for HeldOutConfig"):
        _apply_overrides(container, {"held_out": {"set": {"a": "/p"}}})
