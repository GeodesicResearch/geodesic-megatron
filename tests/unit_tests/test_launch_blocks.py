# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""What the launch blocks share (``scripts/training/launch_blocks.py``).

Reading a block, its record and the command line are exercised through both blocks' own tests
(``test_code_identity.py``, ``test_launch_width.py``); this module tests what neither does on its own.
"""

from scripts.training.code_identity import CODE_IDENTITY_KEY
from scripts.training.launch_blocks import LAUNCH_BLOCK_KEYS, pop_launch_blocks
from scripts.training.launch_width import LAUNCH_WIDTH_KEY


def test_the_launch_blocks_are_the_code_identity_and_the_launch_width():
    assert set(LAUNCH_BLOCK_KEYS) == {CODE_IDENTITY_KEY, LAUNCH_WIDTH_KEY}


def test_popping_the_blocks_returns_them_and_leaves_the_settings():
    overrides = {"train": {"global_batch_size": 16}, "code_identity": {"revision": "a"}, "launch_width": {"nodes": 2}}
    assert pop_launch_blocks(overrides) == {"code_identity": {"revision": "a"}, "launch_width": {"nodes": 2}}
    assert overrides == {"train": {"global_batch_size": 16}}


def test_a_config_without_launch_blocks_gives_none():
    overrides = {"train": {"global_batch_size": 16}}
    assert pop_launch_blocks(overrides) == {}
    assert overrides == {"train": {"global_batch_size": 16}}
