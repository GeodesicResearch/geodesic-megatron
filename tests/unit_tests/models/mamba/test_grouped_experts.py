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
"""Numerical + checkpoint-mapping tests for GroupedExperts.

Runs the REAL module on a real GPU with real single-process process groups and the real
grouped-GEMM kernels, compared against a pure-torch per-expert reference (the semantic
definition of the experts computation). Both GEMM backends are pinned separately: a
backend is a kernel swap under a shared contract, so each one has to be shown to compute
the same thing rather than inheriting the other's numerics.

A GPU is genuinely required — the whole point of the module is the fused CUDA kernel — and
so is the backend's own dependency (``torch._grouped_mm`` / the ``grouped_gemm`` package).
Those are environment boundaries, not test conveniences, so each backend skips with a
reason naming exactly what is missing.
"""

import os
from types import SimpleNamespace

import pytest
import torch


try:
    import grouped_gemm  # noqa: F401

    HAVE_GROUPED_GEMM = True
except ImportError:
    HAVE_GROUPED_GEMM = False

HAVE_TORCH_GROUPED_MM = hasattr(torch, "_grouped_mm")

# Literals rather than the module constants so collection stays import-light (no
# megatron.core at import time); test_backend_ids_match_module_constants pins them
# to the module so the parametrisation cannot drift.
BACKENDS = ("torch_grouped", "cublas_grouped")

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (fused grouped-GEMM kernels)")

E, LATENT, FFN, HIDDEN = 8, 16, 24, 32


def _require_backend(gemm_backend):
    """Skip when the backend's kernel provider is absent from this environment."""
    if gemm_backend == "cublas_grouped" and not HAVE_GROUPED_GEMM:
        pytest.skip("gemm_backend='cublas_grouped' needs the nv-grouped-gemm package (module 'grouped_gemm')")
    if gemm_backend == "torch_grouped" and not HAVE_TORCH_GROUPED_MM:
        pytest.skip(f"gemm_backend='torch_grouped' needs torch._grouped_mm, absent from torch {torch.__version__}")


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


def _config(fused_weighted_act=False):
    from megatron.core.activations import squared_relu
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        use_fused_weighted_squared_relu=fused_weighted_act,
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


def _provider():
    # Deliberately the NemotronH SUBCLASS, not MambaModelProvider: that is the realistic
    # configuration path. (It does NOT additionally pin where `moe_experts_impl` is declared:
    # these tests set it as an instance attribute, so they behave identically whichever class
    # in the MRO owns the field.)
    from megatron.bridge.models.nemotronh.nemotron_h_provider import NemotronHModelProvider

    return NemotronHModelProvider(num_layers=2, hidden_size=HIDDEN, num_attention_heads=4, num_moe_experts=E)


def _experts_slot(spec):
    """The MoESubmodules.experts entry the swap is supposed to rewrite."""
    return spec.submodules.moe_layer.submodules.mlp.keywords["submodules"].experts


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


def test_backend_ids_match_module_constants():
    """The parametrisation must cover every backend the module actually offers."""
    from megatron.bridge.models.mamba.grouped_experts import (
        CUBLAS_GROUPED,
        DEPRECATED_BACKEND_ALIASES,
        GEMM_BACKENDS,
        TORCH_GROUPED,
    )

    assert (TORCH_GROUPED, CUBLAS_GROUPED) == BACKENDS
    assert tuple(GEMM_BACKENDS) == BACKENDS
    assert DEPRECATED_BACKEND_ALIASES == {"cutlass_grouped": CUBLAS_GROUPED}


@requires_gpu
@pytest.mark.parametrize("gemm_backend", BACKENDS)
class TestGroupedExperts:
    """Every backend gets the same numerical and checkpoint pinning — none inherits the other's."""

    def _build(self, pg_collection, gemm_backend, fused_weighted_act=False):
        from megatron.bridge.models.mamba.grouped_experts import GroupedExperts

        _require_backend(gemm_backend)
        torch.manual_seed(1234)
        torch.cuda.set_device(0)
        return GroupedExperts(
            E, _config(fused_weighted_act), pg_collection=pg_collection, gemm_backend=gemm_backend
        ).cuda()

    def _inputs(self, requires_grad=False):
        # ragged sizes including a zero-token expert — the dropless shape profile
        tokens_per_expert = torch.tensor([5, 0, 3, 7, 1, 2, 4, 6], dtype=torch.long)
        n = int(tokens_per_expert.sum())
        x = torch.randn(n, LATENT, dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)
        probs = torch.rand(n, dtype=torch.bfloat16, device="cuda")
        return x, tokens_per_expert, probs

    @pytest.mark.parametrize("fused_weighted_act", [False, True])
    def test_forward_matches_per_expert_reference(self, pg_collection, gemm_backend, fused_weighted_act):
        # True = the branch the real config runs (m6c crashed on 1-D probs here)
        m = self._build(pg_collection, gemm_backend, fused_weighted_act)
        x, tpe, probs = self._inputs()
        out, bias = m(x, tpe, probs)
        assert bias is None
        ref = _reference(x, m.weight1.detach(), m.weight2.detach(), tpe, probs)
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)

    def test_backward_grads_match_reference(self, pg_collection, gemm_backend):
        m = self._build(pg_collection, gemm_backend)
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

    def test_zero_token_forward_keeps_graph(self, pg_collection, gemm_backend):
        m = self._build(pg_collection, gemm_backend)
        tpe = torch.zeros(E, dtype=torch.long)
        x = torch.randn(0, LATENT, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        probs = torch.rand(0, dtype=torch.bfloat16, device="cuda")
        out, _ = m(x, tpe, probs)
        assert out.shape == (0, LATENT)
        out.sum().backward()  # params must stay in the autograd graph

    def test_sharded_state_dict_canonical_keys_and_shapes(self, pg_collection, gemm_backend):
        # Run per backend because the module promises the checkpoint is backend-independent:
        # a warm-start must not care which kernel wrote it.
        m = self._build(pg_collection, gemm_backend)
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


@requires_gpu
class TestTorchGroupedBackend:
    """The shipped default gets its own pinning: it is what every measured run now uses."""

    def _build(self, pg_collection):
        from megatron.bridge.models.mamba.grouped_experts import GroupedExperts

        _require_backend("torch_grouped")
        torch.manual_seed(1234)
        torch.cuda.set_device(0)
        return GroupedExperts(E, _config(), pg_collection=pg_collection, gemm_backend="torch_grouped").cuda()

    def test_matches_per_expert_reference_exactly(self, pg_collection):
        """torch._grouped_mm reduces each group the same way the per-expert loop does.

        The standalone probe measured max |diff| 0.0 at champion shapes, i.e. the -16.2%
        end-to-end win costs nothing numerically. Assert that rather than a tolerance, so a
        future torch changing the grouped kernel's reduction order surfaces here instead of
        as unexplained loss drift.
        """
        m = self._build(pg_collection)
        tokens_per_expert = torch.tensor([5, 0, 3, 7, 1, 2, 4, 6], dtype=torch.long)
        n = int(tokens_per_expert.sum())
        x = torch.randn(n, LATENT, dtype=torch.bfloat16, device="cuda")
        probs = torch.rand(n, dtype=torch.bfloat16, device="cuda")

        out, _ = m(x, tokens_per_expert, probs)
        ref = _reference(x, m.weight1.detach(), m.weight2.detach(), tokens_per_expert, probs)
        assert torch.equal(out, ref), f"max |diff| = {(out.float() - ref.float()).abs().max().item()}"

    def test_autograd_produces_finite_grads(self, pg_collection):
        m = self._build(pg_collection)
        tokens_per_expert = torch.tensor([5, 0, 3, 7, 1, 2, 4, 6], dtype=torch.long)
        n = int(tokens_per_expert.sum())
        x = torch.randn(n, LATENT, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        probs = torch.rand(n, dtype=torch.bfloat16, device="cuda")

        out, _ = m(x, tokens_per_expert, probs)
        out.backward(torch.randn_like(out))

        for name, grad in (("x", x.grad), ("weight1", m.weight1.grad), ("weight2", m.weight2.grad)):
            assert grad is not None, f"{name} received no gradient"
            assert torch.isfinite(grad.float()).all(), f"{name} gradient has non-finite entries"


@requires_gpu
class TestBackendSelection:
    """gemm_backend is required and validated — which kernel runs is never implicit."""

    def test_missing_gemm_backend_raises(self, pg_collection):
        from megatron.bridge.models.mamba.grouped_experts import GroupedExperts

        with pytest.raises(TypeError, match="gemm_backend"):
            GroupedExperts(E, _config(), pg_collection=pg_collection)

    # 'cutlass_grouped' is included deliberately: the deprecated alias is resolved by the
    # provider, so the module itself must still reject it rather than guess.
    @pytest.mark.parametrize("bad_backend", ["nonsense", "cutlass_grouped"])
    def test_invalid_gemm_backend_raises(self, pg_collection, bad_backend):
        from megatron.bridge.models.mamba.grouped_experts import GroupedExperts

        with pytest.raises(ValueError, match="gemm_backend"):
            GroupedExperts(E, _config(), pg_collection=pg_collection, gemm_backend=bad_backend)

    def test_cublas_backend_without_grouped_gemm_raises(self, pg_collection, monkeypatch):
        """The message an operator hits first on a stack without nv-grouped-gemm.

        The absent dependency is faked by clearing the module handle the real import
        populates. That is the genuine boundary: the package is installed in this image, and
        a test that skipped wherever it happens to be present would never exercise the
        message on the only stacks where it fires.
        """
        from megatron.bridge.models.mamba import grouped_experts as ge

        monkeypatch.setattr(ge, "grouped_gemm", None)
        with pytest.raises(ImportError, match="nv-grouped-gemm"):
            ge.GroupedExperts(E, _config(), pg_collection=pg_collection, gemm_backend="cublas_grouped")

    def test_torch_backend_without_grouped_mm_raises(self, pg_collection, monkeypatch):
        """Same, for the shipped default on a torch predating ``_grouped_mm``."""
        from megatron.bridge.models.mamba import grouped_experts as ge

        monkeypatch.delattr(torch, "_grouped_mm", raising=False)
        with pytest.raises(ImportError, match="_grouped_mm"):
            ge.GroupedExperts(E, _config(), pg_collection=pg_collection, gemm_backend="torch_grouped")

    # The module is BF16/FP32-only and non-gated by construction; each of these guards exists
    # because the corresponding config would otherwise produce silently wrong numerics rather
    # than an error, so every one is pinned.
    @pytest.mark.parametrize(
        "field, value, match",
        [
            ("gated_linear_unit", True, "non-gated"),
            ("add_bias_linear", True, "biases"),
            ("delay_wgrad_compute", True, "delayed wgrad"),
            ("fp8", "hybrid", "BF16/FP32-only"),
        ],
    )
    @pytest.mark.parametrize("gemm_backend", BACKENDS)
    def test_unsupported_config_raises(self, pg_collection, gemm_backend, field, value, match):
        from megatron.bridge.models.mamba.grouped_experts import GroupedExperts

        _require_backend(gemm_backend)
        config = _config()
        setattr(config, field, value)
        with pytest.raises(ValueError, match=match):
            GroupedExperts(E, config, pg_collection=pg_collection, gemm_backend=gemm_backend)


class TestSwapSpec:
    """swap_moe_experts_to_grouped is the only channel the backend choice can travel."""

    def test_rejects_invalid_backend(self):
        from megatron.bridge.models.mamba.grouped_experts import swap_moe_experts_to_grouped

        provider = _provider()
        with pytest.raises(ValueError, match="gemm_backend"):
            swap_moe_experts_to_grouped(provider.mamba_stack_spec(provider), gemm_backend="nonsense")

    @pytest.mark.parametrize("gemm_backend", BACKENDS)
    def test_binds_backend_into_experts_slot(self, gemm_backend):
        from megatron.bridge.models.mamba.grouped_experts import GroupedExperts, swap_moe_experts_to_grouped

        provider = _provider()
        original = provider.mamba_stack_spec(provider)
        # Captured BEFORE the swap: object identity is what actually proves the deepcopy
        # happened. Comparing the slot against the CLASS instead would be vacuous — the slot
        # is always a functools.partial, never a bare class, so `is not GroupedExperts` holds
        # even when the caller's spec has been rewritten in place.
        original_slot = _experts_slot(original)

        swapped = swap_moe_experts_to_grouped(original, gemm_backend=gemm_backend)

        experts = _experts_slot(swapped)
        assert experts.func is GroupedExperts
        # mcore constructs the experts positionally, so the backend can only reach the
        # module as a bound keyword — check it is actually there, not merely accepted.
        assert experts.keywords == {"gemm_backend": gemm_backend}
        # the swap must not mutate the caller's spec
        assert _experts_slot(original) is original_slot, "swap mutated the caller's spec in place"
        assert original_slot.func is not GroupedExperts


class TestProviderWiring:
    """moe_experts_impl must take effect even when set AFTER construction (YAML merge path)."""

    def test_mtp_layers_with_grouped_impl_raises(self):
        """The swap does not reach an MTP block's nested MoE spec — refuse the combination."""
        provider = _provider()
        provider.moe_experts_impl = "torch_grouped"
        provider.mtp_num_layers = 1
        with pytest.raises(NotImplementedError, match="MTP"):
            provider._apply_moe_experts_impl()

    @pytest.mark.parametrize("gemm_backend", BACKENDS)
    def test_post_merge_field_swaps_experts_in_resolved_spec(self, gemm_backend):
        from megatron.bridge.models.mamba.grouped_experts import GroupedExperts

        provider = _provider()
        provider.moe_experts_impl = gemm_backend  # simulates the OmegaConf post-init merge
        provider._apply_moe_experts_impl()
        experts = _experts_slot(provider.mamba_stack_spec(provider))
        assert experts.func is GroupedExperts
        assert experts.keywords == {"gemm_backend": gemm_backend}

    def test_deprecated_alias_resolves_to_cublas_and_warns(self, caplog):
        """Existing `cutlass_grouped` configs keep working, loudly — they named a kernel sm_90 never ran."""
        from megatron.bridge.models.mamba.grouped_experts import CUBLAS_GROUPED, GroupedExperts

        provider = _provider()
        provider.moe_experts_impl = "cutlass_grouped"
        with caplog.at_level("WARNING"):
            provider._apply_moe_experts_impl()
        assert "DEPRECATED" in caplog.text

        experts = _experts_slot(provider.mamba_stack_spec(provider))
        assert experts.func is GroupedExperts
        assert experts.keywords == {"gemm_backend": CUBLAS_GROUPED}

    def test_default_leaves_spec_untouched(self):
        from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider

        provider = MambaModelProvider(num_layers=2, hidden_size=HIDDEN, num_attention_heads=4, num_moe_experts=E)
        before = provider.mamba_stack_spec
        provider._apply_moe_experts_impl()
        assert provider.mamba_stack_spec is before

    def test_unknown_impl_raises(self):
        from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider

        provider = MambaModelProvider(num_layers=2, hidden_size=HIDDEN, num_attention_heads=4, num_moe_experts=E)
        provider.moe_experts_impl = "nonsense"
        with pytest.raises(ValueError, match="moe_experts_impl"):
            provider._apply_moe_experts_impl()
