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

"""Helpers for PEFT-specific activation recompute fixes."""

from __future__ import annotations

from functools import wraps
from typing import Iterable, Set

import torch
from megatron.core.utils import unwrap_model

from megatron.bridge.utils.common_utils import print_rank_0


PEFT_RECOMPUTE_PATCHED: Set[int] = set()


def _iter_unwrapped_models(model) -> Iterable[torch.nn.Module]:
    """Yield unwrapped Megatron modules regardless of list/list-like inputs."""
    unwrapped = unwrap_model(model)
    if isinstance(unwrapped, list):
        for module in unwrapped:
            if module is not None:
                yield module
    else:
        if unwrapped is not None:
            yield unwrapped


def _recompute_is_enabled(cfg) -> bool:
    """Whether this config runs any activation recompute.

    Both ``full`` and ``selective`` granularity route through Megatron's
    checkpointing, and therefore both carry the autograd behaviour this module
    exists to correct. ``recompute_method`` is only populated for ``full``, so
    it cannot be the sole test.
    """
    if cfg is None:
        return False
    return (
        getattr(cfg, "recompute_granularity", None) is not None or getattr(cfg, "recompute_method", None) is not None
    )


def _candidate_stacks(unwrapped_model: torch.nn.Module) -> list[torch.nn.Module]:
    """Modules whose forward should receive the input-grad fix.

    Every ``TransformerBlock`` in the tree, so behaviour for GPT-style models
    (including nested ones such as VLM towers) is exactly as before, PLUS the
    model's own ``decoder`` stack whatever its class. The latter is what covers
    hybrid Mamba models, whose stack (``HybridStack``/``MambaStack``) is a
    sibling of ``TransformerBlock`` rather than a subclass — an ``isinstance``
    test against ``TransformerBlock`` silently matches nothing for them, which
    is precisely the bug this generalisation fixes.
    """
    from megatron.core.transformer.transformer_block import TransformerBlock

    stacks: list[torch.nn.Module] = [m for m in unwrapped_model.modules() if isinstance(m, TransformerBlock)]
    decoder = getattr(unwrapped_model, "decoder", None)
    if isinstance(decoder, torch.nn.Module) and not any(decoder is s for s in stacks):
        stacks.append(decoder)
    return stacks


def _force_input_grad(tensor):
    """Return a leaf that requires grad, when the input carries no graph.

    Gated on ``torch.is_grad_enabled()`` so the wrapper is inert under
    ``no_grad``/``inference_mode``: generation must not start building a graph,
    and ``requires_grad_()`` raises outright on an inference tensor.
    """
    if torch.is_grad_enabled() and torch.is_tensor(tensor) and tensor.is_floating_point() and not tensor.requires_grad:
        return tensor.detach().requires_grad_(True)
    return tensor


def _patch_stack_forward(stack: torch.nn.Module) -> bool:
    """Wrap ``stack.forward`` to guarantee its hidden-state input requires grad."""
    if getattr(stack, "_peft_recompute_grad_patched", False):
        return False

    original_forward = stack.forward

    @wraps(original_forward)
    def patched_forward(*args, _original_forward=original_forward, **kwargs):
        # Callers pass hidden_states either positionally or by keyword
        # (GPTModel and HybridModel both use the keyword form).
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = _force_input_grad(kwargs["hidden_states"])
        elif args:
            args = (_force_input_grad(args[0]),) + args[1:]
        return _original_forward(*args, **kwargs)

    stack.forward = patched_forward
    stack._peft_recompute_grad_patched = True
    return True


def maybe_enable_recompute_inputs_grad(model, peft_recompute_patched: Set[int] | None = None) -> Set[int]:
    """Enable grad on the decoder stack's input so recomputed regions get gradients.

    Root cause analysis:

    - Megatron's CheckpointFunction.backward() is only invoked by PyTorch autograd
      when at least one input TENSOR requires grad. Parameters living inside the
      checkpointed region do not count towards that test.
    - With PP>1, received tensors from other stages have requires_grad=True, so
      checkpoint backward is always called.
    - With PP=1 and a frozen base model, embedding outputs have requires_grad=False.
      This means CheckpointFunction.backward() is never called, and the gradients of
      any trainable module inside the checkpoint are never computed. The forward
      loss is unaffected, so the failure is silent: grad norm is exactly zero and
      the trainable weights never move.

    Solution: hook the decoder stack's forward to ensure hidden_states.requires_grad
    is True before it enters checkpointed computation. This unfreezes nothing; it
    only ensures the autograd machinery calls checkpoint's backward. It is a no-op
    whenever the input already carries a graph, so a trainable embedding or a PP>1
    received activation is never detached.

    Applies to any stack class, not just ``TransformerBlock`` — see
    :func:`_candidate_stacks`.

    Borrowed (with modifications) from
    https://github.com/HollowMan6/verl/blob/4285f0601028aee7ddcb9ec5a15198ebfc69bba3/verl/utils/megatron_peft_utils.py
    """

    patched_registry = peft_recompute_patched or PEFT_RECOMPUTE_PATCHED

    try:
        for unwrapped_model in _iter_unwrapped_models(model):
            if not _recompute_is_enabled(getattr(unwrapped_model, "config", None)):
                continue

            if id(unwrapped_model) in patched_registry:
                continue

            patched = False
            for stack in _candidate_stacks(unwrapped_model):
                # Patch whenever anything inside the recomputed stack is trainable.
                # The narrower "adapter-only" test this replaced classified every
                # trainable non-adapter module as a base weight, which skipped
                # models that train custom modules inside the stack alongside
                # adapters (gradient routing's gr_aux is one). Widening is safe
                # because the wrapper only ever touches an input that does not
                # already require grad.
                if not any(p.requires_grad for p in stack.parameters()):
                    continue
                patched |= _patch_stack_forward(stack)

            if patched:
                patched_registry.add(id(unwrapped_model))
                print_rank_0(
                    "[PEFT+Recompute] Patched decoder stack forward to enable grad on the "
                    "hidden_states input. This ensures checkpoint backward is called when "
                    "the modules trained live inside the recomputed region (PP=1 with a "
                    "frozen base model).",
                )
    except Exception as exc:  # pragma: no cover - best effort logging
        # Log but don't fail - user will see grad_norm=0 and can debug
        print_rank_0(f"[PEFT+Recompute] Warning: Failed to patch decoder stack: {exc}")

    return patched_registry


__all__ = ["maybe_enable_recompute_inputs_grad", "PEFT_RECOMPUTE_PATCHED"]
