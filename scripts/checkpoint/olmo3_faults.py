# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Deliberate faults of a Megatron OLMo-3 model, for the negative controls of its parity checks.

A parity check that has never failed has an unknown power. Each fault changes ONE thing that a weight round trip and a
short prompt both miss, so that a check can be shown to fail when it should:

- ``yarn_all_layers``: YaRN rope and its attention factor on every layer (the transformers 5.x regression).
- ``no_swa``: no sliding window. The rope selection is left alone, so the attention span changes and the rope does not.
- ``per_head_qknorm``: Megatron's stock per-head QK-norm instead of OLMo's full-width one.
- ``swap_gate_up``: the gate and up halves of ``linear_fc1`` exchanged.

``apply_fault_before_build`` acts on the model config (or provider) before the model is built, ``apply_fault_after_load``
on the built model chunks after their weights are loaded. Both raise on a name that is not in ``FAULTS``.

torch is imported only when a function that needs it runs.
"""

from __future__ import annotations

import types
from collections.abc import Sequence
from typing import Any


FAULTS = (
    "none",
    "yarn_all_layers",
    "no_swa",
    "per_head_qknorm",
    "swap_gate_up",
)


def require_fault(fault: str) -> str:
    """``fault``, which must be one of ``FAULTS``; otherwise raises ValueError."""
    if fault not in FAULTS:
        raise ValueError(f"fault {fault!r} is not one of {list(FAULTS)}")
    return fault


def apply_fault_before_build(model_cfg: Any, fault: str) -> None:
    """The faults that change the config the model is built from: ``no_swa`` gives every layer an unlimited causal
    window. ``window_size`` and ``window_attn_skip_freq`` jointly drive both the attention span and the per-layer rope
    choice, so the skip frequency is kept (the rope selection is untouched) and only the span changes."""
    require_fault(fault)
    if fault == "no_swa":
        model_cfg.window_size = (-1, 0)


def apply_fault_after_load(model_chunks: Sequence[Any], fault: str, *, yarn_rotary_scaling_factor: float) -> None:
    """The faults that change a built model whose weights are loaded, on every layer of every chunk this rank holds.

    ``yarn_rotary_scaling_factor`` is the model config's: ``yarn_all_layers`` gives the sliding layers back the attention
    factor that their config copy lacks.
    """
    import torch
    from megatron.core.utils import unwrap_model

    require_fault(fault)
    for chunk in model_chunks:
        for layer in unwrap_model(chunk).decoder.layers:
            if fault == "yarn_all_layers":
                # Flip only the rope slice and the attention factor; TE's sliding windows were built from the correct
                # config at init, so the attention span is untouched.
                attention = layer.self_attention
                attention.is_sliding = False  # picks the YaRN slice of the stacked rope
                attention.config.yarn_rotary_scaling_factor = yarn_rotary_scaling_factor
            elif fault == "per_head_qknorm":
                _make_qk_norm_per_head(layer.self_attention)
            elif fault == "swap_gate_up":
                weight = layer.mlp.linear_fc1.weight.data
                half = weight.shape[0] // 2
                layer.mlp.linear_fc1.weight.data = torch.cat([weight[half:], weight[:half]], dim=0)


def _make_qk_norm_per_head(att) -> None:
    """Rebind QK-norm to Megatron's stock per-head semantics (a deliberate fault).

    Uses the same learned gammas, but computes the RMS over each head's ``head_dim`` elements instead of over the full
    projection. That denominator is the whole difference between stock Megatron and OLMo-2/3, and it is invisible in
    the weights.
    """
    import torch

    def _rms_per_head(x, gamma, eps, head_dim):
        shp = x.shape
        xh = x.reshape(*shp[:-1], -1, head_dim).float()
        xh = xh * torch.rsqrt(xh.pow(2).mean(-1, keepdim=True) + eps)
        return (xh.reshape(*shp).to(gamma.dtype)) * gamma

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, **kwargs):
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]
        query, key, value = torch.split(mixed_qkv, split_arg_list, dim=3)
        hd = self.hidden_size_per_attention_head
        eps = self.config.layernorm_epsilon
        query = query.reshape(query.size(0), query.size(1), -1)
        key = key.reshape(key.size(0), key.size(1), -1)
        query = _rms_per_head(query, self.q_layernorm.weight, eps, hd)
        key = _rms_per_head(key, self.k_layernorm.weight, eps, hd)
        query = query.view(query.size(0), query.size(1), -1, hd)
        key = key.view(key.size(0), key.size(1), -1, hd)
        value = value.reshape(value.size(0), value.size(1), -1, hd)
        return query, key, value

    att.get_query_key_value_tensors = types.MethodType(get_query_key_value_tensors, att)
