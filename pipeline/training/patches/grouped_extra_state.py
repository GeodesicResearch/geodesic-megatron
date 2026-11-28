# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The per-expert split of an empty extra state of Transformer Engine's grouped linear layers.

Under MXFP8 block scaling, the Transformer Engine of NeMo 26.10.00 writes an empty extra state for a ``GroupedLinear``.
Megatron-Core's ``TEGroupedLinear._split_extra_state`` decodes that state to None and then indexes it. A distributed
checkpoint of ``te_grouped`` experts therefore fails with a ``TypeError``. Megatron-Core's load hook for these layers
reads an empty extra state as one that holds no information. The patch gives every expert the empty state, which is
the form that hook reads.
"""

from __future__ import annotations

import functools

import torch


def patch_grouped_extra_state_split() -> bool:
    """Make ``TEGroupedLinear._split_extra_state`` give each expert the extra state unchanged when that state is empty.

    Returns True when it installs the patch. Returns False when the patch is already installed. Also returns False
    when the module defines no ``TEGroupedLinear``, which happens without Transformer Engine. A state that is not
    empty goes to the method the patch replaces. Raises AttributeError when ``TEGroupedLinear`` has no
    ``_split_extra_state``. The patch changes ``TEGroupedLinear`` for the whole process.
    """
    from megatron.core.extensions import transformer_engine

    layer = getattr(transformer_engine, "TEGroupedLinear", None)
    if layer is None or getattr(layer._split_extra_state, "empty_state_split", False):
        return False
    original = layer._split_extra_state

    @functools.wraps(original)
    def split_extra_state(self, state):
        # An empty tensor is the state that _decode_extra_state reads as None.
        if isinstance(state, torch.Tensor) and state.numel() == 0:
            return [state] * self.num_gemms
        return original(self, state)

    split_extra_state.empty_state_split = True
    layer._split_extra_state = split_extra_state
    return True
