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

"""Small real Megatron-Core language models for GPU unit tests.

Both models are built from Megatron-Core's own layer specs on the current CUDA device, over whatever model-parallel
state the caller has set up, with dropout off so that a forward is repeatable. Every other ``TransformerConfig``
field, the layer count and sizes included, is the caller's.
"""

import torch


def _config(**config_fields):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(hidden_dropout=0.0, attention_dropout=0.0, use_cpu_initialization=True, **config_fields)


def gpt_model(*, vocab_size: int, max_sequence_length: int, pre_process: bool, **config_fields) -> torch.nn.Module:
    """A GPTModel of Transformer Engine layers, MoE layers when the config has experts."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel

    config = _config(**config_fields)
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=pre_process,
    )
    return model.cuda()


def hybrid_model(
    hybrid_layer_pattern: str, *, vocab_size: int, max_sequence_length: int, pre_process: bool, **config_fields
) -> torch.nn.Module:
    """A HybridModel with Megatron-Core's hybrid stack spec and the given layer pattern."""
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    model = HybridModel(
        config=_config(**config_fields),
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        hybrid_layer_pattern=hybrid_layer_pattern,
        pre_process=pre_process,
    )
    return model.cuda()
