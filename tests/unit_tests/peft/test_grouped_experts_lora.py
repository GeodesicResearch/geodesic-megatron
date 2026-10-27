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
"""Per-expert LoRA on GroupedExperts: forward, gradients, merge, checkpoint mapping and targeting."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.unit_tests.one_rank_nccl_world import one_rank_model_parallel_state


E, LATENT, FFN, HIDDEN, RANK, ALPHA = 8, 16, 24, 32, 4, 8

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (grouped-GEMM kernels)"),
    pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason="needs torch._grouped_mm"),
]


@pytest.fixture(scope="module")
def pg_collection():
    """Real world-1 mcore parallel state with the expert groups GroupedExperts reads."""
    from megatron.core import parallel_state

    with one_rank_model_parallel_state(seed=1234, expert_model_parallel_size=1):
        yield SimpleNamespace(
            ep=parallel_state.get_expert_model_parallel_group(),
            expt_tp=parallel_state.get_expert_tensor_parallel_group(),
            expt_dp=parallel_state.get_expert_data_parallel_group(),
        )


def _experts(pg_collection):
    from megatron.core.activations import squared_relu
    from megatron.core.transformer.transformer_config import TransformerConfig

    from megatron.bridge.models.mamba.grouped_experts import GroupedExperts

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=E,
        moe_ffn_hidden_size=FFN,
        moe_latent_size=LATENT,
        activation_func=squared_relu,
        add_bias_linear=False,
        gated_linear_unit=False,
        params_dtype=torch.float32,
        perform_initialization=True,
        moe_router_topk=2,
    )
    return GroupedExperts(E, config, pg_collection=pg_collection, gemm_backend="torch_grouped").cuda()


def _adapted(pg_collection):
    """GroupedExperts under CanonicalLoRA's per-expert targets, with B factors made non-zero."""
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = _experts(pg_collection)

    layer = CanonicalLoRA(
        target_modules=["experts_linear_fc1", "experts_linear_fc2"],
        dim=RANK,
        alpha=ALPHA,
        lora_A_init_method="kaiming",
    )(Layer(), training=True)
    with torch.no_grad():
        layer.experts.adapter.fc1_b.normal_(std=0.3)
        layer.experts.adapter.fc2_b.normal_(std=0.3)
    return layer.experts


def _inputs(requires_grad=False):
    tokens_per_expert = torch.tensor([5, 0, 3, 7, 1, 2, 4, 6], dtype=torch.long)
    n = int(tokens_per_expert.sum())
    x = torch.randn(n, LATENT, device="cuda", requires_grad=requires_grad)
    probs = torch.rand(n, device="cuda")
    return x, tokens_per_expert, probs


def _reference(wrapper, x, tokens_per_expert, probs):
    """Per-expert dense reference: (W1_e + s A^T B_e) then relu^2 * p then (W2_e + s A_e B)."""
    base, adapter = wrapper.to_wrap, wrapper.adapter
    scale = ALPHA / RANK
    w1 = base.weight1.view(E, LATENT, FFN)
    w2 = base.weight2.view(E, FFN, LATENT)
    outs, start = [], 0
    for e, n in enumerate(tokens_per_expert.tolist()):
        xe, pe = x[start : start + n], probs[start : start + n]
        w1e = w1[e] + scale * adapter.fc1_a.t() @ adapter.fc1_b[e]
        w2e = w2[e] + scale * adapter.fc2_a[e] @ adapter.fc2_b
        h = torch.relu(xe @ w1e) ** 2 * pe.unsqueeze(-1)
        outs.append(h @ w2e)
        start += n
    return torch.cat(outs)


def test_forward_matches_per_expert_reference(pg_collection):
    wrapper = _adapted(pg_collection)
    x, tpe, probs = _inputs()
    out, bias = wrapper(x, tpe, probs)
    assert bias is None
    # torch._grouped_mm runs fp32 inputs at TF32 precision.
    torch.testing.assert_close(out, _reference(wrapper, x, tpe, probs), rtol=2e-2, atol=1e-2)


def test_only_adapter_factors_train_and_all_receive_gradients(pg_collection):
    wrapper = _adapted(pg_collection)
    x, tpe, probs = _inputs()
    out, _ = wrapper(x, tpe, probs)
    out.square().sum().backward()
    assert not wrapper.to_wrap.weight1.requires_grad and not wrapper.to_wrap.weight2.requires_grad
    for name in ("fc1_a", "fc1_b", "fc2_a", "fc2_b"):
        grad = getattr(wrapper.adapter, name).grad
        assert grad is not None and grad.abs().sum() > 0, name
    # An expert with no rows contributes no gradient to its own factors.
    assert wrapper.adapter.fc1_b.grad[1].abs().sum() == 0


def test_zero_token_forward_keeps_adapter_in_graph(pg_collection):
    wrapper = _adapted(pg_collection)
    x = torch.randn(0, LATENT, device="cuda", requires_grad=True)
    out, _ = wrapper(x, torch.zeros(E, dtype=torch.long), torch.rand(0, device="cuda"))
    assert out.shape == (0, LATENT)
    out.sum().backward()
    assert wrapper.adapter.fc1_a.grad is not None and wrapper.adapter.fc2_b.grad is not None


def test_init_matches_tinker_layout(pg_collection):
    wrapper = _adapted(pg_collection)
    adapter = wrapper.adapter
    # Shared A [dim, in]: bound 1/sqrt(in). Per-expert A [experts, dim, ffn]: torch's 3D fan-in gives 1/sqrt(dim*ffn).
    assert adapter.fc1_a.abs().max() <= 1 / math.sqrt(LATENT)
    assert adapter.fc1_a.abs().max() > 0.5 / math.sqrt(LATENT)
    assert adapter.fc2_a.abs().max() <= 1 / math.sqrt(RANK * FFN)
    assert adapter.fc2_a.abs().max() > 0.5 / math.sqrt(RANK * FFN)
    fresh = _experts(pg_collection)
    from megatron.bridge.peft.grouped_experts_lora import GroupedExpertsLoRAAdapter

    untouched = GroupedExpertsLoRAAdapter(fresh, RANK, ALPHA, "kaiming", "zero")
    assert untouched.fc1_b.abs().sum() == 0 and untouched.fc2_b.abs().sum() == 0


def test_merge_reproduces_adapted_forward(pg_collection):
    from megatron.bridge.peft.lora import LoRAMerge

    wrapper = _adapted(pg_collection)
    x, tpe, probs = _inputs()
    with torch.no_grad():
        adapted, _ = wrapper(x, tpe, probs)
        LoRAMerge().transform(wrapper)
        merged, _ = wrapper.to_wrap(x, tpe, probs)
    torch.testing.assert_close(merged, adapted, rtol=2e-2, atol=1e-2)


def test_sharded_state_dict_keys_and_shapes(pg_collection):
    wrapper = _adapted(pg_collection)
    sd = wrapper.sharded_state_dict(prefix="decoder.layers.1.mlp.experts.")
    p = "decoder.layers.1.mlp.experts.adapter."
    assert tuple(sd[f"{p}fc1_a"].global_shape) == (RANK, LATENT)
    assert tuple(sd[f"{p}fc2_b"].global_shape) == (RANK, LATENT)
    assert tuple(sd[f"{p}fc1_b"].global_shape) == (E, RANK, FFN)
    assert tuple(sd[f"{p}fc2_a"].global_shape) == (E, FFN, RANK)
    # The base experts keep their canonical keys.
    assert sd["decoder.layers.1.mlp.experts.weight1"].key == "decoder.layers.1.mlp.experts.experts.linear_fc1.weight"


def test_partial_expert_target_raises(pg_collection):
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = _experts(pg_collection)

    with pytest.raises(ValueError, match="partial"):
        CanonicalLoRA(target_modules=["experts_linear_fc1"], lora_A_init_method="kaiming")(Layer())


def test_non_grouped_experts_module_raises():
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.Linear(4, 4)

    lora = CanonicalLoRA(target_modules=["experts_linear_fc1", "experts_linear_fc2"], lora_A_init_method="kaiming")
    with pytest.raises(NotImplementedError, match="GroupedExperts"):
        lora.transform(Layer().experts, name="experts", prefix="decoder.layers.0.mlp")
