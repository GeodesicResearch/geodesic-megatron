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
"""A small Latent-MoE Nemotron-H for GPU tests: every module kind of Nemotron 3 Super at toy widths.

Mamba, attention and MoE layers (routed experts on ``GroupedExperts`` with a latent projection, plus a shared expert),
with dropout, the auxiliary loss and the router bias update off, so that a forward is repeatable.
"""

VOCAB, SEQ = 256, 64
NUM_EXPERTS = 8
#: The Mamba mixer's d_inner (mamba_num_heads * mamba_head_dim).
D_INNER = 4 * 64


def small_nemotron_h_provider(hybrid_layer_pattern: str, **overrides):
    """The finalized ``NemotronHModelProvider`` for ``hybrid_layer_pattern`` (``|`` splits pipeline stages).

    ``overrides`` set any other provider field (dtype, parallel sizes, dispatcher, ...).
    """
    from megatron.bridge.models.nemotronh.nemotron_h_provider import NemotronHModelProvider

    fields = dict(
        hybrid_layer_pattern=hybrid_layer_pattern,
        num_layers=len(hybrid_layer_pattern.replace("|", "")),
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        mamba_num_heads=4,
        mamba_head_dim=64,
        mamba_num_groups=2,
        mamba_state_dim=16,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=2,
        moe_ffn_hidden_size=64,
        moe_latent_size=32,
        moe_shared_expert_intermediate_size=128,
        moe_shared_expert_overlap=False,
        moe_experts_impl="torch_grouped",
        moe_aux_loss_coeff=0.0,
        moe_router_bias_update_rate=0.0,
        mtp_num_layers=None,
        vocab_size=VOCAB,
        seq_length=SEQ,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    )
    fields.update(overrides)
    provider = NemotronHModelProvider(**fields)
    provider.finalize()
    return provider
