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
"""A small Latent-MoE Nemotron-H for GPU tests, as a Megatron provider and as an HF release: every module kind of
Nemotron 3 Super at toy widths.

Mamba, attention and MoE layers (routed experts on ``GroupedExperts`` with a latent projection, plus a shared expert),
with dropout, the auxiliary loss and the router bias update off, so that a forward is repeatable.
"""

from pathlib import Path


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


def write_small_nemotron_h_release(path: Path) -> Path:
    """Write an HF NemotronH release of the small model's architecture to ``path`` and return it.

    The weights are random, the tokenizer is word-level over the vocabulary, and the keys follow the released
    checkpoints' layout (one key per expert).
    """
    import torch
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer, models
    from transformers import NemotronHConfig, NemotronHForCausalLM, PreTrainedTokenizerFast

    config = NemotronHConfig(
        vocab_size=VOCAB,
        hidden_size=128,
        layers_block_type=["mamba", "attention", "moe"],
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        mamba_num_heads=4,
        mamba_head_dim=64,
        mamba_n_groups=2,
        ssm_state_size=16,
        n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        moe_latent_size=32,
        n_shared_experts=1,
        moe_shared_expert_intermediate_size=128,
        moe_shared_expert_overlap=False,
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    # save_original_format writes the release checkpoint layout (one key per expert), which the bridge maps.
    NemotronHForCausalLM(config).to(torch.bfloat16).save_pretrained(path, save_original_format=True)
    # The releases name the embedding backbone.embeddings.weight; transformers writes backbone.embedding.weight.
    shard = path / "model.safetensors"
    tensors = load_file(shard)
    tensors["backbone.embeddings.weight"] = tensors.pop("backbone.embedding.weight")
    save_file(tensors, shard, metadata={"format": "pt"})
    vocabulary = {f"t{i}": i for i in range(VOCAB)}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="t0"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0").save_pretrained(path)
    return path
