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

"""Unit tests for PEFT-specific recompute helpers."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.peft import recompute as recompute_mod
from megatron.bridge.peft.recompute import maybe_enable_recompute_inputs_grad


class DummyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class DummyTransformerBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input_requires_grad = None

    def forward(self, hidden_states, *args, **kwargs):
        self.last_input_requires_grad = hidden_states.requires_grad
        return hidden_states


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(recompute_method="uniform")
        self.block = DummyTransformerBlock()

        # Frozen base parameter (not trainable)
        self.base = torch.nn.Linear(1, 1, bias=False)
        self.base.weight.requires_grad = False

        # Trainable adapter, nested INSIDE the block. The fix triggers on
        # "something inside the recomputed stack needs gradients", and in a real
        # model the adapters live inside the decoder stack, so the test model
        # has to nest them too for the precondition to mean anything.
        self.block.adapter = torch.nn.ModuleDict({"adapter": DummyAdapter()})

    def modules(self):
        for module in super().modules():
            yield module


def _patch_transformer_block(monkeypatch):
    import megatron.core.transformer.transformer_block as transformer_block

    monkeypatch.setattr(
        transformer_block,
        "TransformerBlock",
        DummyTransformerBlock,
        raising=False,
    )


def test_maybe_enable_recompute_inputs_grad_patches_block(monkeypatch):
    _patch_transformer_block(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyModel()
    patched_registry = maybe_enable_recompute_inputs_grad(model, set())

    assert id(model) in patched_registry

    patched_forward = model.block.forward

    input_tensor = torch.zeros(2, 2)
    assert input_tensor.requires_grad is False

    model.block(input_tensor)
    assert model.block.last_input_requires_grad is True

    # Second invocation should be a no-op (no duplicate patch)
    maybe_enable_recompute_inputs_grad(model, patched_registry)
    assert model.block.forward is patched_forward


class DummyHybridStack(torch.nn.Module):
    """A decoder stack that is NOT a TransformerBlock, like Megatron's HybridStack."""

    def __init__(self) -> None:
        super().__init__()
        self.last_input_requires_grad = None
        self.adapter = torch.nn.ModuleDict({"adapter": DummyAdapter()})

    def forward(self, hidden_states, *args, **kwargs):
        self.last_input_requires_grad = hidden_states.requires_grad
        return hidden_states


class DummyHybridModel(torch.nn.Module):
    """Hybrid model whose `.decoder` is a sibling of TransformerBlock, not a subclass."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(recompute_granularity="full", recompute_method="uniform")
        self.decoder = DummyHybridStack()
        self.base = torch.nn.Linear(1, 1, bias=False)
        self.base.weight.requires_grad = False


def test_hybrid_stack_is_patched_even_though_it_is_not_a_transformer_block(monkeypatch):
    """Regression: an isinstance(TransformerBlock) test silently matches nothing here.

    Hybrid (Mamba) models' decoder stack is a sibling class, so gating the patch on
    TransformerBlock left every trainable module inside the recomputed region without
    gradients — silently, since the forward loss is unaffected.
    """
    _patch_transformer_block(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyHybridModel()
    assert not isinstance(model.decoder, DummyTransformerBlock)

    patched_registry = maybe_enable_recompute_inputs_grad(model, set())
    assert id(model) in patched_registry

    model.decoder(hidden_states=torch.zeros(2, 2))
    assert model.decoder.last_input_requires_grad is True


def test_stack_with_non_adapter_trainable_module_is_patched(monkeypatch):
    """The precondition is "anything inside the stack is trainable", not adapter names.

    Gradient routing's gr_aux modules train inside the recomputed stack without
    "adapter" appearing anywhere in their parameter names, so an adapter-name
    heuristic classifies them as base weights and skips the patch — and the
    recomputed region silently produces no gradients for them.
    """
    _patch_transformer_block(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyHybridModel()
    del model.decoder.adapter
    model.decoder.gr_aux = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False)])

    patched_registry = maybe_enable_recompute_inputs_grad(model, set())
    assert id(model) in patched_registry

    model.decoder(hidden_states=torch.zeros(2, 2))
    assert model.decoder.last_input_requires_grad is True


def test_fully_frozen_stack_is_not_patched(monkeypatch):
    """A stack with no trainable parameters needs no gradient chain into it."""
    _patch_transformer_block(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyHybridModel()
    for param in model.decoder.parameters():
        param.requires_grad = False

    assert maybe_enable_recompute_inputs_grad(model, set()) == set()
    assert getattr(model.decoder, "_peft_recompute_grad_patched", False) is False
    assert "forward" not in vars(model.decoder)


def test_selective_recompute_granularity_alone_enables_patch(monkeypatch):
    """Selective recompute sets recompute_granularity but leaves recompute_method None.

    Selective granularity still routes through Megatron's checkpointing, so it
    carries the same missing-backward hazard as full recompute; a gate that only
    consults recompute_method skips every selective-recompute model.
    """
    _patch_transformer_block(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyHybridModel()
    model.config = SimpleNamespace(recompute_granularity="selective", recompute_method=None)

    patched_registry = maybe_enable_recompute_inputs_grad(model, set())
    assert id(model) in patched_registry

    model.decoder(hidden_states=torch.zeros(2, 2))
    assert model.decoder.last_input_requires_grad is True


def test_no_patch_when_recompute_disabled(monkeypatch):
    _patch_transformer_block(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyHybridModel()
    model.config = SimpleNamespace(recompute_granularity=None, recompute_method=None)

    assert maybe_enable_recompute_inputs_grad(model, set()) == set()
    # `forward` is still the bound method, not an instance attribute holding the
    # wrapper (identity on a bound method is meaningless — it is rebuilt on every
    # attribute access — so assert on the marker the patch sets).
    assert getattr(model.decoder, "_peft_recompute_grad_patched", False) is False
    assert "forward" not in vars(model.decoder)


def test_patch_is_inert_under_no_grad(monkeypatch):
    """Generation must not start building a graph (and inference tensors would raise)."""
    _patch_transformer_block(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyHybridModel()
    maybe_enable_recompute_inputs_grad(model, set())

    with torch.no_grad():
        model.decoder(hidden_states=torch.zeros(2, 2))
    assert model.decoder.last_input_requires_grad is False


def test_recomputed_region_gets_gradients_only_after_the_patch():
    """End-to-end autograd semantics this module exists to correct.

    A reentrant checkpoint runs its backward only when an INPUT tensor requires
    grad; parameters inside the region do not count. With a frozen base at PP=1
    the stack input carries no graph, so the trainable module inside the
    checkpoint gets no gradient at all. Forcing the input to require grad
    restores exactly the gradient the non-recomputed path produces.
    """
    from torch.utils.checkpoint import checkpoint

    def build():
        torch.manual_seed(0)
        base = torch.nn.Linear(8, 8, bias=False)
        adapter = torch.nn.Linear(8, 8, bias=False)
        for p in base.parameters():
            p.requires_grad = False
        return base, adapter

    def run(force_input_grad: bool, use_checkpoint: bool):
        base, adapter = build()
        x = torch.randn(4, 8)
        if force_input_grad:
            x = recompute_mod._force_input_grad(x)

        def block(t):
            return base(t) + adapter(t)

        out = checkpoint(block, x, use_reentrant=True) if use_checkpoint else block(x)
        out.pow(2).mean().backward()
        return adapter.weight.grad

    # Ground truth: no recompute, nothing forced.
    reference = run(force_input_grad=False, use_checkpoint=False)
    assert reference is not None and reference.norm() > 0

    # Unpatched + checkpointed: the region's output carries no grad_fn at all,
    # so autograd never descends into it and backward has nothing to run.
    with pytest.raises(RuntimeError, match="does not require grad"):
        run(force_input_grad=False, use_checkpoint=True)

    # Patched + checkpointed: recovers the non-recomputed gradient exactly.
    patched = run(force_input_grad=True, use_checkpoint=True)
    assert patched is not None
    torch.testing.assert_close(patched, reference)
