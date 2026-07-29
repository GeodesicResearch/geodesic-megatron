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
"""CUTLASS grouped-GEMM experts for Nemotron-H: one kernel per group, not one per expert.

Why this exists (2026-07-29, docs/investigations/120b-gbs64-host-overhead-investigation.md §9):
at mcore 0.19 the TEGroupedMLP path decomposes ragged dropless grouped GEMM into per-expert
cuBLASLt kernels — 168,771 launches per iteration on the Super-120B quickstart (66% of all
launches), whose ~60 us/launch of host-serial dispatch is the measured ~42% GPU-idle. TE ≤ 2.14
has no device-side/batched ragged-group API (host ``List[int]`` only) and TE ≥ 2.15 images are
blocked by the cluster driver, so the collapse has to come from the CUTLASS ``grouped_gemm``
kernel (``nv-grouped-gemm``; in the 26.02 image, overlay-built for 26.04).

This is the pre-0.19 upstream ``GroupedMLP`` (removed upstream; reference implementation:
submodule tag ``core_v0.13.1``) ported to the 0.19 experts contract, restricted to what
Nemotron-H needs (no GLU, no biases), and made Latent-MoE aware: the experts' input width is
``moe_latent_size`` (1024 on Super-120B), not ``hidden_size``.

Checkpoint compatibility: emits the same canonical keys as TEGroupedMLP / SequentialMLP
(``<prefix>experts.linear_fc{1,2}.weight``, global shape ``[num_experts, out, in]`` — verified
against the Base-Chat-Init torch_dist checkpoint), so existing checkpoints warm-start and
checkpoints saved from this module load back into the default path.

Selection: ``NemotronHModelProvider.moe_experts_impl = "cutlass_grouped"`` (default
``"te_grouped"`` keeps the upstream path byte-for-byte).
"""

import copy
import dataclasses
from functools import partial
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from megatron.core import tensor_parallel
from megatron.core.activations import squared_relu
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_gpu
from megatron.core.utils import divide
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_object_for_checkpoint

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None


class CutlassGroupedExperts(MegatronModule):
    """Experts layer executing all local experts in one CUTLASS grouped GEMM per projection.

    Drop-in replacement for TEGroupedMLP at the ``MoESubmodules.experts`` slot: same
    constructor call ``experts(num_local_experts, config, pg_collection=..., name=...)``,
    same ``forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)``
    contract, same canonical checkpoint mapping.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection=None,
        name: Optional[str] = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        if grouped_gemm is None:
            raise ImportError(
                "CutlassGroupedExperts needs the nv-grouped-gemm package (module 'grouped_gemm'). "
                "It ships in nemo:26.02.nemotron_3_super; for other images add it to the overlay."
            )
        if config.gated_linear_unit:
            raise ValueError("CutlassGroupedExperts supports non-gated activations only (Nemotron-H).")
        if config.add_bias_linear:
            raise ValueError("CutlassGroupedExperts does not support expert biases.")
        if getattr(config, "delay_wgrad_compute", False):
            raise ValueError("CutlassGroupedExperts does not implement delayed wgrad compute.")
        if config.fp8 or getattr(config, "fp4", None):
            raise ValueError("CutlassGroupedExperts is BF16/FP32-only (no quantization padding).")

        self.expert_parallel = config.expert_model_parallel_size > 1
        assert pg_collection is not None, "pg_collection is required at mcore 0.19"
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        self.dp_group = pg_collection.expt_dp

        self.activation_func = self.config.activation_func
        self.activation_recompute = (
            self.config.recompute_granularity == "selective"
            and "moe_act" in self.config.recompute_modules
        )

        # Latent MoE: the experts see moe_latent_size-wide activations, not hidden_size.
        self.in_features = (
            config.moe_latent_size if config.moe_latent_size is not None else config.hidden_size
        )

        tp_size = self.tp_group.size()
        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)
        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # CUTLASS grouped GEMM does not support transposition, so weights are stored
        # untransposed: weight1 [in, E*ffn/tp], weight2 [E*ffn/tp, in] (core_v0.13.1 layout).
        assert not config.use_cpu_initialization, "GPU initialization only (warm-start path)."
        self.weight1 = Parameter(
            torch.empty(
                self.in_features,
                fc1_output_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
        self.weight2 = Parameter(
            torch.empty(
                fc2_input_size_per_partition,
                self.in_features,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
        if config.perform_initialization:
            _initialize_affine_weight_gpu(
                self.weight1, config.init_method, partition_dim=1, is_expert=True
            )
            _initialize_affine_weight_gpu(
                self.weight2, config.output_layer_init_method, partition_dim=0, is_expert=True
            )
        setattr(self.weight1, "allreduce", not self.expert_parallel)
        setattr(self.weight2, "allreduce", not self.expert_parallel)

        def remove_extra_states_check(self, incompatible_keys):
            """Drop _extra_state unexpected keys (SequentialMLP/TEGroupedMLP ckpt compat)."""
            for key in copy.deepcopy(incompatible_keys.unexpected_keys):
                if "_extra_state" in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def _weighted_activation(self, x: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """activation(x) scaled by router probs — same semantics as TEGroupedMLP."""
        if self.activation_func == squared_relu and getattr(
            self.config, "use_fused_weighted_squared_relu", False
        ):
            return weighted_squared_relu_impl(x, probs)
        dtype = x.dtype
        return (self.activation_func(x) * probs.unsqueeze(-1)).to(dtype)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """One grouped GEMM per projection over all local experts."""
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()

        if self.config.moe_apply_probs_on_input:
            assert self.config.moe_router_topk == 1, (
                "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            )
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            ).to(original_dtype)
            permuted_probs = torch.ones_like(permuted_probs)

        # gmm needs group sizes as a CPU int64 tensor. The alltoall dispatcher has already
        # staged tokens_per_expert to host through its event-synchronized dtoh pipeline, so
        # this is a no-op copy in the training path.
        batch_sizes = tokens_per_expert.detach().to(device="cpu", dtype=torch.long)

        if permuted_local_hidden_states.nelement() != 0:
            w1 = self.weight1.view(self.num_local_experts, self.in_features, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.in_features)
            fc1_output = grouped_gemm.ops.gmm(
                permuted_local_hidden_states, w1, batch_sizes, trans_b=False
            )
            if self.activation_recompute:
                intermediate = self.activation_checkpoint.checkpoint(
                    self._weighted_activation, fc1_output, permuted_probs
                )
                fc2_output = grouped_gemm.ops.gmm(intermediate, w2, batch_sizes, trans_b=False)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                intermediate = self._weighted_activation(fc1_output, permuted_probs)
                fc2_output = grouped_gemm.ops.gmm(intermediate, w2, batch_sizes, trans_b=False)
        else:
            # Zero tokens for every local expert: keep params in the autograd graph.
            w1 = self.weight1.view(self.in_features, -1)
            w2 = self.weight2.view(-1, self.in_features)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                h = self.activation_checkpoint.checkpoint(
                    self._weighted_activation, h, permuted_probs
                )
                fc2_output = torch.matmul(h, w2)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                h = self._weighted_activation(h, permuted_probs)
                fc2_output = torch.matmul(h, w2)

        return fc2_output, None

    def backward_dw(self):
        """No delayed wgrad: gradients are produced in the normal backward pass."""

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None) -> ShardedStateDict:
        """Map the fused weights to the SequentialMLP/TEGroupedMLP-canonical per-expert layout.

        Canonical target (verified against the Base-Chat-Init torch_dist checkpoint):
        ``<prefix>experts.linear_fc{1,2}.weight`` with global shape
        ``[num_experts, out_features, in_features]``. Ported from core_v0.13.1's GroupedMLP,
        non-GLU paths only, with ``in_features`` latent-aware.
        """
        sharded_state_dict = {}
        ep_size = self.ep_group.size()
        ep_rank = self.ep_group.rank()
        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()
        dp_rank = self.dp_group.rank()
        num_global_experts = ep_size * self.num_local_experts
        local_expert_indices_offset = ep_rank * self.num_local_experts
        prepend_axis_num = len(sharded_offsets)
        replica_id: ReplicaId = (0, 0, dp_rank)
        local_ffn_dim_size = self.weight2.numel() // self.num_local_experts // self.in_features
        in_features = self.in_features

        @torch.no_grad()
        def sh_ten_build_fn(
            key: str,
            t: torch.Tensor,
            replica_id: ReplicaId,
            flattened_range: Optional[slice],
            tp_axis: int,
        ):
            if tp_axis == 1:  # weight1: [in, E*ffn/tp]
                real_shape = (self.num_local_experts, in_features, local_ffn_dim_size)
            elif tp_axis == 0:  # weight2: [E*ffn/tp, in]
                real_shape = (self.num_local_experts, local_ffn_dim_size, in_features)
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if flattened_range is None:
                # Weights: expose per-expert [out, in] by transposing the trailing dims.
                t = t.view(real_shape).transpose(-1, -2)
                return ShardedTensor.from_rank_offsets(
                    key,
                    t.contiguous(),
                    *sharded_offsets,
                    (prepend_axis_num, ep_rank, ep_size),
                    (prepend_axis_num + 1 + (1 - tp_axis), tp_rank, tp_size),
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                )
            # Flattened (distributed-optimizer) states: split each expert along dim 0.
            assert t.ndim == 1, (key, t.shape)
            non_flat_local_shape = (1, *real_shape[1:])
            chunk_numel = local_ffn_dim_size * in_features
            sub_states = []
            start_pos = 0
            for local_expert_idx in range(self.num_local_experts):
                if (
                    flattened_range.start < chunk_numel * (local_expert_idx + 1)
                    and flattened_range.stop > chunk_numel * local_expert_idx
                ):
                    end_pos = min(
                        flattened_range.stop,
                        chunk_numel * (local_expert_idx + 1) - flattened_range.start,
                    )
                    local_tensor = t[start_pos:end_pos]
                    local_flattened_range = slice(
                        max(0, flattened_range.start - chunk_numel * local_expert_idx),
                        min(chunk_numel, flattened_range.stop - chunk_numel * local_expert_idx),
                    )
                    assert len(local_tensor) == local_flattened_range.stop - local_flattened_range.start
                    start_pos += len(local_tensor)
                    expert_global_idx = local_expert_indices_offset + local_expert_idx
                    sub_states.append(
                        ShardedTensor.from_rank_offsets_flat(
                            key,
                            local_tensor,
                            non_flat_local_shape,
                            *sharded_offsets,
                            (prepend_axis_num, expert_global_idx, num_global_experts),
                            (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                            replica_id=replica_id,
                            prepend_axis_num=prepend_axis_num,
                            flattened_range=local_flattened_range,
                        )
                    )
            return sub_states

        @torch.no_grad()
        def sh_ten_merge_fn(sub_state_dict, tp_axis: int):
            if tp_axis == 1:
                weight_shape = (in_features, -1)
            elif tp_axis == 0:
                weight_shape = (-1, in_features)
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if isinstance(sub_state_dict, list) and sub_state_dict[0].ndim == 1:
                return torch.cat(sub_state_dict)  # flattened optimizer states
            return sub_state_dict.transpose(-1, -2).reshape(weight_shape)

        state_dict = self.state_dict(prefix="", keep_vars=True)
        for name, tensor in state_dict.items():
            if name == "weight1":
                tp_axis = 1
                wkey = f"{prefix}experts.linear_fc1.weight"
            else:
                tp_axis = 0
                wkey = f"{prefix}experts.linear_fc2.weight"
            sharded_state_dict[f"{prefix}{name}"] = ShardedTensorFactory(
                wkey,
                tensor,
                partial(sh_ten_build_fn, tp_axis=tp_axis),
                partial(sh_ten_merge_fn, tp_axis=tp_axis),
                replica_id,
                flattened_range=None,
            )

        # Fake _extra_state entries for SequentialMLP/TEGroupedMLP checkpoint compatibility.
        extra_replica_id = (0, tp_rank, dp_rank)
        for expert_local_idx in range(self.num_local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_sharded_offsets = (
                *sharded_offsets,
                (len(sharded_offsets), expert_global_idx, num_global_experts),
            )
            for mod in ["linear_fc1", "linear_fc2"]:
                sharded_state_dict[f"{prefix}expert{expert_global_idx}.{mod}._extra_state"] = (
                    make_sharded_object_for_checkpoint(
                        None,
                        f"{prefix}experts.{mod}._extra_state",
                        expert_sharded_offsets,
                        extra_replica_id,
                    )
                )
        return sharded_state_dict


def swap_moe_experts_to_cutlass_grouped(stack_spec):
    """Return a copy of a hybrid/mamba stack spec with the MoE experts swapped to CUTLASS.

    Leaves the router, shared experts, dispatcher and every other submodule untouched:
    only ``MoESubmodules.experts`` inside the ``moe_layer`` spec changes.
    """
    spec = copy.deepcopy(stack_spec)
    moe_builder = spec.submodules.moe_layer.submodules.mlp  # partial(MoELayer, submodules=...)
    moe_submodules = moe_builder.keywords["submodules"]
    new_submodules = dataclasses.replace(moe_submodules, experts=CutlassGroupedExperts)
    spec.submodules.moe_layer.submodules.mlp = partial(
        moe_builder.func, *moe_builder.args, **{**moe_builder.keywords, "submodules": new_submodules}
    )
    return spec
