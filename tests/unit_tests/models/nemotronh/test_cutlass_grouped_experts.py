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
"""Numerical + checkpoint-mapping tests for CutlassGroupedExperts.

Runs the REAL module on a real GPU with real single-process process groups and the real
CUTLASS grouped-GEMM kernels, compared against a pure-torch per-expert reference (the
semantic definition of the experts computation). GPU + nv-grouped-gemm are genuinely
required — the whole point of the module is the fused CUDA kernel — so the tests skip
cleanly where either is absent (e.g. CPU-only CI).
"""

import os
from types import SimpleNamespace

import pytest
import torch

try:
    import grouped_gemm  # noqa: F401

    HAVE_GG = True
except ImportError:
    HAVE_GG = False

requires_gpu_and_gg = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_GG),
    reason="needs a GPU and the nv-grouped-gemm package (CUTLASS grouped GEMM)",
)

E, LATENT, FFN, HIDDEN = 8, 16, 24, 32


@pytest.fixture(scope="module")
def pg_collection():
    """Real world-1 mcore parallel state: expert groups + the expert-parallel RNG tracker."""
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.cuda.set_device(0)
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29777")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    yield SimpleNamespace(
        ep=parallel_state.get_expert_model_parallel_group(),
        expt_tp=parallel_state.get_expert_tensor_parallel_group(),
        expt_dp=parallel_state.get_expert_data_parallel_group(),
    )
    parallel_state.destroy_model_parallel()


def _config():
    from megatron.core.activations import squared_relu
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=E,
        moe_ffn_hidden_size=FFN,
        moe_latent_size=LATENT,
        activation_func=squared_relu,
        add_bias_linear=False,
        gated_linear_unit=False,
        expert_model_parallel_size=1,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        perform_initialization=True,
        moe_router_topk=2,
    )


def _reference(x, w1_fused, w2_fused, tokens_per_expert, probs):
    """Per-expert pure-torch reference: x_e @ W1_e -> relu(.)^2 * p -> @ W2_e.

    Uses the same fused-weight reinterpretation as the module and its checkpoint factory
    (weight1 storage is logically [E, in, ffn] flattened; the declared 2D shape only carries
    the TP partition axis).
    """
    w1 = w1_fused.view(E, LATENT, FFN)  # [E, in, ffn]
    w2 = w2_fused.view(E, FFN, LATENT)
    outs, start = [], 0
    for e, n in enumerate(tokens_per_expert.tolist()):
        xe = x[start : start + n]
        pe = probs[start : start + n]
        h = xe @ w1[e]
        h = torch.nn.functional.relu(h) ** 2 * pe.unsqueeze(-1)
        outs.append((h.to(x.dtype) @ w2[e]))
        start += n
    return torch.cat(outs) if outs else x.new_zeros((0, LATENT))


@requires_gpu_and_gg
class TestCutlassGroupedExperts:
    def _build(self, pg_collection):
        from megatron.bridge.models.nemotronh.cutlass_grouped_experts import CutlassGroupedExperts

        torch.manual_seed(1234)
        torch.cuda.set_device(0)
        return CutlassGroupedExperts(E, _config(), pg_collection=pg_collection).cuda()

    def _inputs(self, requires_grad=False):
        # ragged sizes including a zero-token expert — the dropless shape profile
        tokens_per_expert = torch.tensor([5, 0, 3, 7, 1, 2, 4, 6], dtype=torch.long)
        n = int(tokens_per_expert.sum())
        x = torch.randn(n, LATENT, dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
        probs = torch.rand(n, dtype=torch.bfloat16, device="cuda")
        return x, tokens_per_expert, probs

    def test_forward_matches_per_expert_reference(self, pg_collection):
        m = self._build(pg_collection)
        x, tpe, probs = self._inputs()
        out, bias = m(x, tpe, probs)
        assert bias is None
        ref = _reference(x, m.weight1.detach(), m.weight2.detach(), tpe, probs)
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)

    def test_backward_grads_match_reference(self, pg_collection):
        m = self._build(pg_collection)
        x, tpe, probs = self._inputs(requires_grad=True)
        out, _ = m(x, tpe, probs)
        g = torch.randn_like(out)
        out.backward(g)

        w1 = m.weight1.detach().clone().requires_grad_(True)
        w2 = m.weight2.detach().clone().requires_grad_(True)
        x_ref = x.detach().clone().requires_grad_(True)
        ref = _reference(x_ref, w1, w2, tpe, probs)
        ref.backward(g)

        torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(m.weight1.grad.float(), w1.grad.float(), rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(m.weight2.grad.float(), w2.grad.float(), rtol=3e-2, atol=3e-2)

    def test_zero_token_forward_keeps_graph(self, pg_collection):
        m = self._build(pg_collection)
        tpe = torch.zeros(E, dtype=torch.long)
        x = torch.randn(0, LATENT, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        probs = torch.rand(0, dtype=torch.bfloat16, device="cuda")
        out, _ = m(x, tpe, probs)
        assert out.shape == (0, LATENT)
        out.sum().backward()  # params must stay in the autograd graph

    def test_sharded_state_dict_canonical_keys_and_shapes(self, pg_collection):
        m = self._build(pg_collection)
        sd = m.sharded_state_dict(prefix="decoder.layers.1.mlp.experts.")
        f1 = sd["decoder.layers.1.mlp.experts.weight1"]
        f2 = sd["decoder.layers.1.mlp.experts.weight2"]
        # canonical keys match the TEGroupedMLP/SequentialMLP layout the base ckpt uses
        assert f1.key == "decoder.layers.1.mlp.experts.experts.linear_fc1.weight"
        assert f2.key == "decoder.layers.1.mlp.experts.experts.linear_fc2.weight"
        # building the factory yields [num_experts(sharded axis), out, in] per-expert tensors
        st1 = f1.build_fn(f1.key, f1.data, f1.replica_id, None)
        st2 = f2.build_fn(f2.key, f2.data, f2.replica_id, None)
        assert tuple(st1.global_shape) == (E, FFN, LATENT)
        assert tuple(st2.global_shape) == (E, LATENT, FFN)
        # round-trip: merge(build) reproduces the fused weights
        merged1 = f1.merge_fn(st1.data)
        merged2 = f2.merge_fn(st2.data)
        torch.testing.assert_close(merged1, m.weight1.detach())
        torch.testing.assert_close(merged2, m.weight2.detach())
        # extra_state compatibility stubs exist per global expert
        assert sum("_extra_state" in k for k in sd) == 2 * E
