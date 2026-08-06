# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""GRAM auxiliary module for gradient routing on hybrid MoE stacks.

Implements the model-surgery half of GRAM (Gradient-Routed Auxiliary Modules,
arXiv 2607.08077): each MoE layer gains one narrow always-constructed auxiliary MLP whose
output is added to the layer output under a scalar 0/1 gate. The gate is driven per
training iteration from the gradient-routing plan (see
``megatron.bridge.training.gradient_routing``); an un-driven model — evaluation, export,
coherence — keeps the gate at its 0.0 default and behaves exactly like the core model.

The aux module deliberately mirrors ``SharedExpertMLP``: same ``MLP`` base, same
``MLPSubmodules`` (taken from the spec's shared-expert builder), same input tensor (the
MoE layer's input hidden states), and its output is added with coefficient 1.0 when the
gate is open. That symmetry is a hard contract, not a convenience: the forget-ON export
posture merges the aux weights into the shared expert by width concatenation, which is
mathematically exact only for a non-gated elementwise activation applied to the same
input and summed unscaled.

The aux module runs on EVERY microbatch, gate open or closed. Megatron's DDP requires
each parameter to produce a gradient every microbatch (bucket completion under
``overlap_grad_reduce``); ``gate * aux(h)`` with gate 0 produces exact-zero gradients
through the fused-wgrad path while leaving the forward numerically identical to omitting
the addend (for finite aux outputs, ``x + 0.0 * a == x``).
"""

from copy import deepcopy
from functools import partial
from typing import Optional

import torch
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


class GRAMAuxMLP(MLP):
    """Narrow dense MLP holding the gradient-routed ("forget") capacity of one MoE layer.

    Shaped exactly like the layer's shared expert but at ``aux_ffn_hidden_size``. The
    output projection is zero-initialised so a warm-started model is bit-unchanged at
    load time: the aux contribution is exactly 0 until the module has been trained,
    and the missing-keys checkpoint load leaves this init in place.

    The bias/gated refusals below are load-bearing TWICE: the forget-ON export merges
    aux weights into the shared expert (exact only bias-free and non-gated), AND any
    1-D aux parameter would match both the standard no-weight-decay override and the GR
    aux override, whose conflicting ``wd_mult`` values make mcore's override combiner
    refuse — killing optimizer construction at startup, far from this class.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        aux_ffn_hidden_size: int,
        pg_collection=None,
        name: str | None = None,
    ):
        config = deepcopy(config)
        if config.add_bias_linear:
            raise ValueError("GRAMAuxMLP supports bias-free MLPs only (the export merge assumes no biases).")
        if config.gated_linear_unit:
            raise ValueError(
                "GRAMAuxMLP supports non-gated activations only: the forget-ON export merges aux "
                "weights into the shared expert by row/column concatenation, which this module "
                "only guarantees for elementwise activations."
            )
        config.ffn_hidden_size = aux_ffn_hidden_size
        super().__init__(
            config=config,
            submodules=submodules,
            tp_group=pg_collection.tp if pg_collection is not None else None,
            name=name,
        )
        with torch.no_grad():
            self.linear_fc2.weight.zero_()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the aux MLP output (bias-free by construction)."""
        output, _ = super().forward(hidden_states)
        return output


class GRAMMoELayer(MoELayer):
    """MoELayer plus a gated GRAM auxiliary MLP.

    ``forward`` adds ``gr_gate * gr_aux(hidden_states)`` AFTER the parent forward — i.e.
    outside the ``moe``-recompute region, so the (small) aux activations are never
    recomputed and the aux add composes with any recompute posture.

    ``gr_gate`` is a non-persistent scalar buffer, default 0.0: checkpoints carry only
    the aux weights, and any consumer that does not drive the gate (evaluation, HF
    export, coherence) sees core-only behaviour.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection=None,
        is_mtp_layer: bool = False,
        name: str | None = None,
        *,
        gr_aux_ffn_hidden_size: int,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )
        if not self.use_shared_expert or submodules.shared_experts is None:
            raise ValueError(
                "GRAMMoELayer requires a shared-expert MoE layer: the aux module reuses the "
                "shared expert's MLPSubmodules and the export postures merge into the shared "
                "expert. This config has moe_shared_expert_intermediate_size unset."
            )
        aux_submodules = getattr(submodules.shared_experts, "keywords", {}).get("submodules")
        if not isinstance(aux_submodules, MLPSubmodules):
            raise ValueError(
                "GRAMMoELayer could not extract MLPSubmodules from the shared-experts builder "
                f"({submodules.shared_experts!r}); expected a partial carrying submodules=MLPSubmodules(...)."
            )
        self.gr_aux = GRAMAuxMLP(
            config=config,
            submodules=aux_submodules,
            aux_ffn_hidden_size=gr_aux_ffn_hidden_size,
            pg_collection=pg_collection,
            name=(name + ".gr_aux") if name is not None else None,
        )
        self.register_buffer("gr_gate", torch.zeros((), dtype=config.params_dtype), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        if intermediate_tensors is not None:
            raise NotImplementedError(
                "GRAMMoELayer does not support split-execution forward (intermediate_tensors); "
                "it is incompatible with overlap_moe_expert_parallel_comm / partial CUDA-graph "
                "capture, neither of which this model supports."
            )
        output, mlp_bias = super().forward(hidden_states, None, padding_mask)
        output = output + self.gr_gate * self.gr_aux(hidden_states)
        return output, mlp_bias


def swap_moe_layer_to_gram(stack_spec, aux_ffn_hidden_size: int):
    """Return a copy of a hybrid/mamba stack spec with MoELayer swapped to GRAMMoELayer.

    Mirrors ``swap_moe_experts_to_grouped``: only the ``moe_layer`` mlp builder's class
    changes; router, experts, shared experts, and dispatcher submodules are untouched, so
    this composes with the experts-impl swap in either order (that swap edits the same
    partial's ``submodules`` keyword and preserves its ``func``).

    ``aux_ffn_hidden_size`` is bound here because mcore constructs the mlp builder with a
    fixed argument set and has no channel for extra arguments — the same reason
    ``gemm_backend`` travels via partial in the experts swap.
    """
    if aux_ffn_hidden_size <= 0:
        raise ValueError(f"aux_ffn_hidden_size must be positive, got {aux_ffn_hidden_size}.")
    spec = deepcopy(stack_spec)
    moe_builder = spec.submodules.moe_layer.submodules.mlp
    if not (isinstance(moe_builder, partial) and issubclass(moe_builder.func, MoELayer)):
        raise ValueError(
            f"Expected moe_layer.submodules.mlp to be partial(MoELayer, ...), got {moe_builder!r}. "
            "If another spec transform replaced the class, apply the GRAM swap to a spec whose "
            "mlp builder still constructs MoELayer."
        )
    if moe_builder.func is GRAMMoELayer:
        raise ValueError("GRAM swap already applied to this spec (double-apply).")
    spec.submodules.moe_layer.submodules.mlp = partial(
        GRAMMoELayer,
        *moe_builder.args,
        **{**moe_builder.keywords, "gr_aux_ffn_hidden_size": aux_ffn_hidden_size},
    )
    return spec
