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
"""GRAM auxiliary modules for gradient routing on hybrid MoE stacks.

Implements the model-surgery half of GRAM (Gradient-Routed Auxiliary Modules,
arXiv 2607.08077): each MoE layer gains N narrow always-constructed auxiliary MLPs —
one per routed capability — whose outputs are added to the layer output under
per-module 0/1 gates. The gates are driven per training iteration from the
gradient-routing plan (see ``megatron.bridge.training.gradient_routing``); an un-driven
model — evaluation, export, coherence — keeps every gate at its 0.0 default and behaves
exactly like the core model.

Each aux module deliberately mirrors ``SharedExpertMLP``: same ``MLP`` base, same
``MLPSubmodules`` (taken from the spec's shared-expert builder), same input tensor (the
MoE layer's input hidden states), and its output is added with coefficient 1.0 when its
gate is open. That symmetry is a hard contract, not a convenience: the module-enabled
export postures merge selected aux weights into the shared expert by width
concatenation, which is mathematically exact only for a non-gated elementwise
activation applied to the same input and summed unscaled — and the sum is additive per
module, so any SUBSET of modules merges exactly.

Every aux module runs on EVERY microbatch, gates open or closed. Megatron's DDP
requires each parameter to produce a gradient every microbatch (bucket completion under
``overlap_grad_reduce``); ``gate_k * aux_k(h)`` with gate 0 produces exact-zero
gradients through the fused-wgrad path while leaving the forward numerically identical
to omitting the addend (for finite aux outputs, ``x + 0.0 * a == x``).
"""

from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import Optional

import torch
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


class GRAMAuxMLP(MLP):
    """Narrow dense MLP holding one gradient-routed capability of one MoE layer.

    Shaped exactly like the layer's shared expert but at ``aux_ffn_hidden_size``. The
    output projection is zero-initialised so a warm-started model is bit-unchanged at
    load time: the aux contribution is exactly 0 until the module has been trained,
    and the missing-keys checkpoint load leaves this init in place.

    The bias/gated refusals below are load-bearing TWICE: the module-enabled export
    merges aux weights into the shared expert (exact only bias-free and non-gated), AND
    any 1-D aux parameter would match both the standard no-weight-decay override and a
    GR aux override, whose conflicting ``wd_mult`` values make mcore's override combiner
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
        if config.add_bias_linear:
            raise ValueError("GRAMAuxMLP supports bias-free MLPs only (the export merge assumes no biases).")
        if config.gated_linear_unit:
            raise ValueError(
                "GRAMAuxMLP supports non-gated activations only: the module-enabled export merges aux "
                "weights into the shared expert by row/column concatenation, which this module "
                "only guarantees for elementwise activations."
            )
        # BOTH the config clone and the explicit argument are required, because upstream
        # MLP reads the width from two different places: linear_fc1 is built from the
        # `ffn_hidden_size` ARGUMENT, while linear_fc2 reads `config.ffn_hidden_size`
        # (mcore transformer/mlp.py). Setting only one of them builds an aux MLP whose
        # two projections disagree, which fails at the first GEMM. The clone is per-aux
        # so the surrounding model's own ffn_hidden_size is untouched; the argument also
        # silences MLP's "requires ffn_hidden_size" deprecation warning.
        config = deepcopy(config)
        config.ffn_hidden_size = aux_ffn_hidden_size
        super().__init__(
            config=config,
            submodules=submodules,
            ffn_hidden_size=aux_ffn_hidden_size,
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
    """MoELayer plus N gated GRAM auxiliary MLPs.

    ``forward`` adds ``sum_k gr_gate[k] * gr_aux[k](hidden_states)`` AFTER the parent
    forward — i.e. outside the ``moe``-recompute region, so the (small) aux activations
    are never recomputed and the aux add composes with any recompute posture. The sum is
    kept as an unrolled sequence of adds: an all-zero gate vector must leave the output
    BITWISE equal to the core layer, which a fused/concatenated path would not guarantee.

    ``gr_aux`` is an ``nn.ModuleList`` deliberately named so parameters keep the
    ``.gr_aux.`` fragment (``...mlp.gr_aux.<k>.linear_fc1.weight``) — the optimizer
    override glob, the HF bridge mappings, and the bake script all key on it.

    ``gr_gate`` is a non-persistent ``(N,)`` buffer, default all-zero: checkpoints carry
    only the aux weights, and any consumer that does not drive the gates (evaluation, HF
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
        gr_aux_ffn_hidden_sizes: Sequence[int],
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
                "GRAMMoELayer requires a shared-expert MoE layer: the aux modules reuse the "
                "shared expert's MLPSubmodules and the export postures merge into the shared "
                "expert. This config has moe_shared_expert_intermediate_size unset."
            )
        aux_submodules = getattr(submodules.shared_experts, "keywords", {}).get("submodules")
        if not isinstance(aux_submodules, MLPSubmodules):
            raise ValueError(
                "GRAMMoELayer could not extract MLPSubmodules from the shared-experts builder "
                f"({submodules.shared_experts!r}); expected a partial carrying submodules=MLPSubmodules(...)."
            )
        self.gr_aux = torch.nn.ModuleList(
            GRAMAuxMLP(
                config=config,
                submodules=aux_submodules,
                aux_ffn_hidden_size=width,
                pg_collection=pg_collection,
                name=(f"{name}.gr_aux.{k}") if name is not None else None,
            )
            for k, width in enumerate(gr_aux_ffn_hidden_sizes)
        )
        # gr_static_gates (a provider field; the provider IS this config object) pins the
        # gates at construction for eval-only profile serving — a probe loads a GRAM
        # checkpoint and scores one module subset with no runtime gate driver. Training
        # runs leave it None (the launch guards refuse otherwise) and the callback drives
        # the gates per iteration.
        static_gates = getattr(config, "gr_static_gates", None)
        if static_gates is not None and len(static_gates) != len(self.gr_aux):
            raise ValueError(f"gr_static_gates has {len(static_gates)} entries for {len(self.gr_aux)} aux modules.")
        initial = (
            torch.zeros(len(self.gr_aux), dtype=config.params_dtype)
            if static_gates is None
            else torch.tensor([float(g) for g in static_gates], dtype=config.params_dtype)
        )
        self.register_buffer("gr_gate", initial, persistent=False)

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
        for k, aux in enumerate(self.gr_aux):
            output = output + self.gr_gate[k] * aux(hidden_states)
        return output, mlp_bias


def normalize_aux_widths(value) -> list[int]:
    """The per-module width list a scalar-or-list ``gr_aux_ffn_hidden_size`` denotes.

    The provider field accepts an int (one module) or a list (one entry per module), and
    may arrive as an OmegaConf ListConfig; every consumer compares or builds against the
    plain per-module list, so the normalization lives once, next to the layer it sizes.
    ``None``/empty normalizes to ``[]`` (no modules configured).
    """
    if isinstance(value, int):
        return [value]
    return list(value or [])


def swap_moe_layer_to_gram(stack_spec, aux_ffn_hidden_sizes: Sequence[int]):
    """Return a copy of a hybrid/mamba stack spec with MoELayer swapped to GRAMMoELayer.

    Mirrors ``swap_moe_experts_to_grouped``: only the ``moe_layer`` mlp builder's class
    changes; router, experts, shared experts, and dispatcher submodules are untouched, so
    this composes with the experts-impl swap in either order (that swap edits the same
    partial's ``submodules`` keyword and preserves its ``func``).

    ``aux_ffn_hidden_sizes`` (one width per routed capability, order = module index) is
    bound here because mcore constructs the mlp builder with a fixed argument set and
    has no channel for extra arguments — the same reason ``gemm_backend`` travels via
    partial in the experts swap.
    """
    widths = list(aux_ffn_hidden_sizes)
    if not widths:
        raise ValueError("aux_ffn_hidden_sizes must name at least one aux module.")
    for k, width in enumerate(widths):
        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"aux_ffn_hidden_sizes[{k}] must be a positive int, got {width!r}.")
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
        **{**moe_builder.keywords, "gr_aux_ffn_hidden_sizes": widths},
    )
    return spec
