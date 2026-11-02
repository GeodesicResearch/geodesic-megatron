# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Per-expert LoRA for the routed experts of ``GroupedExperts``.

Each projection of each local expert gets its own rank-``dim`` update. The factor on the
experts' input/output width side (the latent width on Latent-MoE models) is shared by all
experts of the layer; the factor on the FFN side is per expert:

- fc1 (``in -> ffn``): ``delta_e = B_e @ A``, with ``A`` [dim, in] shared and ``B_e`` [ffn, dim] per expert;
- fc2 (``ffn -> in``): ``delta_e = B @ A_e``, with ``A_e`` [dim, ffn] per expert and ``B`` [in, dim] shared.

Both are scaled by ``alpha / dim``. This is the layout of Tinker's LoRA on Nemotron-H MoE layers.

- ``GroupedExpertsLoRAAdapter``: the four factors, their deltas, weight updates and checkpoint mapping.
- ``LoRAGroupedExperts``: the ``AdapterWrapper`` that runs ``GroupedExperts`` with the deltas added.
"""

import math
from functools import partial
from typing import Any, Literal, Optional

import torch
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from torch import nn

from megatron.bridge.models.mamba.grouped_experts import GroupedExperts, _grouped_projection
from megatron.bridge.peft.adapter_wrapper import AdapterWrapper


class GroupedExpertsLoRAAdapter(nn.Module):
    """LoRA factors for every local expert of one ``GroupedExperts`` layer.

    Args:
        experts: the layer being adapted; supplies widths, local expert count, backend and groups.
        dim: LoRA rank.
        alpha: LoRA alpha; the update is scaled by ``alpha / dim``.
        lora_A_init_method: ``"kaiming"`` only: ``kaiming_uniform_(a=sqrt(5))`` on each A factor
            in its ``[..., dim, fan-in width]`` layout (shared A [dim, in]: bound ``1/sqrt(in)``;
            per-expert A [experts, dim, ffn]: torch's 3D fan-in rule gives ``1/sqrt(dim * ffn)``).
        lora_B_init_method: ``"zero"`` only.

    Raises:
        NotImplementedError: tensor or expert tensor parallelism above 1, or another init method.
    """

    def __init__(
        self,
        experts: GroupedExperts,
        dim: int,
        alpha: float,
        lora_A_init_method: str,
        lora_B_init_method: str,
    ) -> None:
        super().__init__()
        if experts.tp_group.size() != 1:
            raise NotImplementedError("Per-expert LoRA supports expert_tensor_parallel_size=1 only.")
        if experts.config.tensor_model_parallel_size != 1:
            # The shared factors would need their gradients reduced across the TP ranks, which route
            # different tokens; nothing here does that.
            raise NotImplementedError("Per-expert LoRA supports tensor_model_parallel_size=1 only.")
        if lora_A_init_method != "kaiming":
            raise NotImplementedError(
                f"Per-expert LoRA initialises A with 'kaiming' only, got {lora_A_init_method!r}."
            )
        if lora_B_init_method != "zero":
            raise NotImplementedError(f"Per-expert LoRA initialises B with 'zero' only, got {lora_B_init_method!r}.")
        self.dim = dim
        self.alpha = alpha
        self.scale = alpha / dim
        self.num_local_experts = experts.num_local_experts
        self.in_features = experts.in_features
        self.ffn_features = experts.config.moe_ffn_hidden_size
        self.gemm_backend = experts.gemm_backend
        self.ep_group = experts.ep_group
        self.dp_group = experts.dp_group
        self.expert_parallel = experts.expert_parallel

        device = experts.weight1.device
        dtype = experts.weight1.dtype
        experts_count = self.num_local_experts

        # Stored in the layouts the grouped GEMM consumes: x [rows, k] @ w [experts, k, n].
        fc1_a = torch.empty(dim, self.in_features, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(fc1_a, a=math.sqrt(5))
        fc2_a = torch.empty(experts_count, dim, self.ffn_features, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(fc2_a, a=math.sqrt(5))
        self.fc1_a = nn.Parameter(fc1_a)  # [dim, in], shared
        self.fc1_b = nn.Parameter(torch.zeros(experts_count, dim, self.ffn_features, device=device, dtype=dtype))
        self.fc2_a = nn.Parameter(fc2_a.transpose(1, 2).contiguous())  # [experts, ffn, dim]
        self.fc2_b = nn.Parameter(torch.zeros(dim, self.in_features, device=device, dtype=dtype))  # shared
        for param in (self.fc1_b, self.fc2_a):
            # Per-expert factors reduce over the expert-data-parallel group, like the expert weights.
            setattr(param, "allreduce", not self.expert_parallel)

    def forward(
        self, projection: Literal["fc1", "fc2"], x: torch.Tensor, group_layout: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """The update of ``projection``'s output for its input rows ``x`` (``fc1_delta`` or ``fc2_delta``)."""
        return {"fc1": self.fc1_delta, "fc2": self.fc2_delta}[projection](x, group_layout)

    def fc1_delta(self, x: torch.Tensor, group_layout: Optional[torch.Tensor]) -> torch.Tensor:
        """The fc1 update of rows ``x`` [rows, in] grouped by ``group_layout``: [rows, ffn] ([0, experts*ffn] if none)."""
        low_rank = torch.matmul(x, self.fc1_a.t())
        if group_layout is None:
            # No rows: the experts' fused [in, experts*ffn] width, keeping the factors in the graph.
            return torch.matmul(low_rank, self.fc1_b.transpose(0, 1).reshape(self.dim, -1)) * self.scale
        return _grouped_projection(low_rank, self.fc1_b, group_layout, self.gemm_backend) * self.scale

    def fc2_delta(self, h: torch.Tensor, group_layout: Optional[torch.Tensor]) -> torch.Tensor:
        """The fc2 update of rows ``h`` [rows, ffn] ([0, experts*ffn] if none) grouped by ``group_layout``: [rows, in]."""
        if group_layout is None:
            low_rank = torch.matmul(h, self.fc2_a.reshape(-1, self.dim))
        else:
            low_rank = _grouped_projection(h, self.fc2_a, group_layout, self.gemm_backend)
        return torch.matmul(low_rank, self.fc2_b) * self.scale

    def fc1_update(self) -> torch.Tensor:
        """Every local expert's fc1 weight update ``scale * A^T @ B_e``, fp32: [experts, in, ffn]."""
        return torch.einsum("ri,erf->eif", self.fc1_a.float(), self.fc1_b.float()) * self.scale

    def fc2_update(self) -> torch.Tensor:
        """Every local expert's fc2 weight update ``scale * A_e @ B``, fp32: [experts, ffn, in]."""
        return torch.einsum("efr,ri->efi", self.fc2_a.float(), self.fc2_b.float()) * self.scale

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Shared factors replicated across data parallelism; per-expert factors sharded by global expert."""
        ep_size = self.ep_group.size()
        ep_rank = self.ep_group.rank()
        prepend_axis_num = len(sharded_offsets)
        sharded_state_dict = {}
        for name in ("fc1_a", "fc2_b"):
            sharded_state_dict[f"{prefix}{name}"] = make_sharded_tensor_for_checkpoint(
                getattr(self, name), f"{prefix}{name}", sharded_offsets
            )
        for name in ("fc1_b", "fc2_a"):
            sharded_state_dict[f"{prefix}{name}"] = ShardedTensor.from_rank_offsets(
                f"{prefix}{name}",
                getattr(self, name),
                *sharded_offsets,
                (prepend_axis_num, ep_rank, ep_size),
                replica_id=(0, 0, self.dp_group.rank()),
                prepend_axis_num=prepend_axis_num,
            )
        return sharded_state_dict


class LoRAGroupedExperts(AdapterWrapper):
    """``GroupedExperts`` whose fc1 and fc2 outputs carry the per-expert LoRA updates of its adapter."""

    def forward(self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor, *args: Any):
        """Same contract as ``GroupedExperts.forward``."""
        if not self._adapter_enabled:
            return self.to_wrap(permuted_local_hidden_states, tokens_per_expert, *args)
        # Through __call__ on both modules, so that their forward pre-hooks fire: DDP with overlap_param_gather waits
        # there for a module's parameter all-gather, and a bound method of the adapter would skip that wait.
        return self.to_wrap(
            permuted_local_hidden_states,
            tokens_per_expert,
            *args,
            fc1_delta=partial(self.adapter, "fc1"),
            fc2_delta=partial(self.adapter, "fc2"),
        )

    updated_weights = ("weight1", "weight2")

    def weight_update(self, name: str) -> torch.Tensor:
        """The adapter's per-expert updates, in ``weight1`` / ``weight2``'s storage layout.

        ``GroupedExperts.forward`` reads ``weight1`` as [experts, in, ffn] and ``weight2`` as [experts, ffn, in], so
        each update is reshaped back from that view.
        """
        update = {"weight1": self.adapter.fc1_update, "weight2": self.adapter.fc2_update}[name]()
        return update.reshape(getattr(self.to_wrap, name).shape)
