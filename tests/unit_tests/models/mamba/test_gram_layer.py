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
"""Numerical pinning for the GRAM auxiliary module and its spec swap.

The central claim of GRAM's model surgery is a NEGATIVE one: a gated-off aux module
changes nothing. Every consumer that does not drive the gate — evaluation, HF export,
coherence testing, and the whole retain-corpus half of training — depends on it, and a
violation would show up as unexplained quality drift rather than as an error. So the
comparison here is ``torch.equal`` against a separately-built vanilla ``MoELayer`` from
the same seed, not a tolerance: "numerically indistinguishable" is exactly the claim, and
bf16 tolerances would hide a real (small) contribution.

Real GPU, real TE linears, real router and dispatcher: the aux module is spliced into the
MoE layer's forward, so a mocked layer would test the splice against nothing.

``stack_spec`` (in the shared helpers) deep-copies. ``provider.mambastack_spec(provider)``
returns a PROCESS-GLOBAL ``ModuleSpec`` — the same object on every call, from every
provider instance — so a test that edited one in place would silently rewrite the spec
every later test builds from. (That global is also why both spec swaps deepcopy; see
``test_swap_does_not_mutate_the_caller_spec``.)
"""

from functools import partial

import pytest
import torch

from tests.unit_tests.gr_test_utils import (
    AUX_FFN,
    HIDDEN,
    SHARED_FFN,
    build_moe_layer,
    gram_spec,
    init_model_parallel,
    moe_builder,
    moe_config,
    stack_spec,
)


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (real TE MoE layer)")

SEQ, BATCH = 6, 2


@pytest.fixture(scope="module")
def moe_parallel_state():
    """Real world-1 mcore parallel state; MoELayer picks the default pg_collection up from it."""
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    init_model_parallel(expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    yield
    parallel_state.destroy_model_parallel()


def _render(builder):
    """A comparable rendering of a spec builder (functools.partial has no __eq__)."""
    return (builder.func, builder.args, builder.keywords) if isinstance(builder, partial) else builder


def _pair(config=None, aux_ffn=AUX_FFN):
    """A vanilla MoELayer and a GRAM-swapped one, seeded identically."""
    config = config if config is not None else moe_config()
    return build_moe_layer(moe_builder(stack_spec()), config), build_moe_layer(moe_builder(gram_spec(aux_ffn)), config)


def _input(seed=7, requires_grad=False):
    torch.manual_seed(seed)
    return torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)


def _opaque_shared_expert_builder(*, config, pg_collection, gate, name=None):
    """A working shared-expert builder that carries no ``submodules`` keyword to inspect."""
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    submodules = moe_builder(stack_spec()).keywords["submodules"].shared_experts.keywords["submodules"]
    return SharedExpertMLP(config=config, submodules=submodules, gate=gate, pg_collection=pg_collection, name=name)


def _core_params(layer):
    """Every parameter that is NOT the aux module, by name.

    Matches on the ``gr_aux.`` path segment rather than the optimizer's ``.gr_aux.``
    fragment: these layers are built standalone, so the aux parameters have no
    ``decoder.layers.N.mlp.`` prefix in front of them the way they do inside a model.
    """
    return {name: p for name, p in layer.named_parameters() if not name.startswith("gr_aux.")}


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestGatedOffIsIdentity:
    """Gate 0 (the default, and every non-driven consumer) must be bit-for-bit the core model."""

    def test_shared_parameters_are_identical_across_the_pair(self):
        """Guards the comparison itself: if the swap perturbed initialisation order, the
        output equalities below would be testing two different models."""
        vanilla, gram = _pair()
        core = _core_params(gram)
        assert set(core) == set(dict(vanilla.named_parameters()))
        for name, p in vanilla.named_parameters():
            assert torch.equal(p, core[name]), f"{name} differs between the vanilla and GRAM layers"

    def test_default_gate_is_zero(self):
        _, gram = _pair()
        assert gram.gr_gate.item() == 0.0
        assert gram.gr_gate.shape == ()
        assert gram.gr_gate.dtype == torch.bfloat16

    def test_output_is_bitwise_equal_at_gate_zero(self):
        vanilla, gram = _pair()
        x = _input()
        out_vanilla, bias_vanilla = vanilla(x)
        out_gram, bias_gram = gram(x)
        assert torch.equal(out_vanilla, out_gram), (
            f"max |diff| = {(out_vanilla.float() - out_gram.float()).abs().max().item()}"
        )
        assert bias_vanilla is None and bias_gram is None

    def test_output_is_bitwise_equal_at_gate_one_with_zero_init_fc2(self):
        """The warm-start posture: gate open on the very first forget iteration, aux weights
        still at their zero init. A run must start from the core model's exact behaviour."""
        vanilla, gram = _pair()
        gram.gr_gate.fill_(1.0)
        x = _input()
        assert torch.equal(vanilla(x)[0], gram(x)[0])

    def test_fc2_is_zero_initialised(self):
        _, gram = _pair()
        assert torch.all(gram.gr_aux.linear_fc2.weight == 0)
        assert not torch.all(gram.gr_aux.linear_fc1.weight == 0), "fc1 must be normally initialised"

    def test_aux_module_has_the_configured_width(self):
        _, gram = _pair(aux_ffn=AUX_FFN)
        assert tuple(gram.gr_aux.linear_fc1.weight.shape) == (AUX_FFN, HIDDEN)
        assert tuple(gram.gr_aux.linear_fc2.weight.shape) == (HIDDEN, AUX_FFN)

    def test_aux_width_does_not_leak_into_the_shared_expert(self):
        """GRAMAuxMLP deep-copies the config before narrowing ffn_hidden_size; a shared
        mutation would silently resize the real shared expert."""
        config = moe_config()
        _, gram = _pair(config=config)
        assert config.ffn_hidden_size != AUX_FFN
        assert tuple(gram.shared_experts.linear_fc1.weight.shape) == (SHARED_FFN, HIDDEN)

    def test_gate_is_not_in_the_state_dict(self):
        """Non-persistent by design: checkpoints carry aux weights, never a routing gate."""
        _, gram = _pair()
        assert "gr_gate" not in gram.state_dict()
        assert "gr_gate" in dict(gram.named_buffers())

    def test_state_dict_adds_only_the_aux_submodule(self):
        """A warm start from a base checkpoint must be missing exactly these keys and no
        others — which is why the launch guards demand a missing-key-tolerant strictness."""
        vanilla, gram = _pair()
        added = set(gram.state_dict()) - set(vanilla.state_dict())
        assert added == {
            "gr_aux.linear_fc1.weight",
            "gr_aux.linear_fc2.weight",
            "gr_aux.linear_fc1._extra_state",
            "gr_aux.linear_fc2._extra_state",
        }
        assert set(vanilla.state_dict()) - set(gram.state_dict()) == set()


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestGatedOnAddsTheAuxOutput:
    """Gate 1 with trained weights must add exactly ``aux(h)``, unscaled — the export merge
    into the shared expert is only exact for coefficient 1.0 on the same input."""

    def _trained_pair(self, scale=0.05):
        vanilla, gram = _pair()
        with torch.no_grad():
            gram.gr_aux.linear_fc2.weight.normal_(0.0, scale)
        gram.gr_gate.fill_(1.0)
        return vanilla, gram

    def test_output_equals_core_plus_aux_exactly(self):
        vanilla, gram = self._trained_pair()
        x = _input()
        out_core, _ = vanilla(x)
        out_gram, _ = gram(x)
        aux = gram.gr_aux(x)
        assert not torch.equal(out_core, out_gram), "aux contribution vanished — the test proves nothing"
        assert torch.equal(out_gram, out_core + aux)

    def test_aux_module_computes_the_reference_mlp(self):
        """The aux is genuinely ``W2 . squared_relu(W1 h)`` at aux width — not, say, gated,
        or fed the post-MoE activations. Tolerance here because the reference uses plain
        torch matmuls against TE's kernels."""
        _, gram = self._trained_pair()
        x = _input()
        w1 = gram.gr_aux.linear_fc1.weight.float()
        w2 = gram.gr_aux.linear_fc2.weight.float()
        reference = (torch.relu(x.float() @ w1.T) ** 2) @ w2.T
        torch.testing.assert_close(gram.gr_aux(x).float(), reference, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("gate", [0.0, 1.0])
    def test_gate_scales_the_contribution(self, gate):
        vanilla, gram = self._trained_pair()
        gram.gr_gate.fill_(gate)
        x = _input()
        out_core, _ = vanilla(x)
        assert torch.equal(gram(x)[0], out_core + gate * gram.gr_aux(x))

    def test_gate_can_be_toggled_between_forwards(self):
        """The callback flips this buffer in place every iteration."""
        vanilla, gram = self._trained_pair()
        x = _input()
        out_core, _ = vanilla(x)
        gram.gr_gate.fill_(0.0)
        assert torch.equal(gram(x)[0], out_core)
        gram.gr_gate.fill_(1.0)
        assert not torch.equal(gram(x)[0], out_core)
        gram.gr_gate.fill_(0.0)
        assert torch.equal(gram(x)[0], out_core)


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestGradients:
    """Gate 0 must still produce gradients (DDP bucket completion) and they must be exact zero."""

    def _backward(self, layer, x, grad_seed=11):
        out, _ = layer(x)
        torch.manual_seed(grad_seed)
        out.backward(torch.randn_like(out))
        return out

    def test_aux_params_receive_exact_zero_grads_at_gate_zero(self):
        """Megatron's DDP requires every parameter to produce a gradient each microbatch;
        a None grad here stalls the bucket under overlap_grad_reduce."""
        _, gram = _pair()
        self._backward(gram, _input(requires_grad=True))
        for name, p in gram.named_parameters():
            if ".gr_aux." in name:
                assert p.grad is not None, f"{name} received no gradient at gate 0"
                assert torch.all(p.grad == 0), f"{name} received a non-zero gradient at gate 0"

    def test_core_grads_are_bitwise_equal_to_the_vanilla_layer(self):
        vanilla, gram = _pair()
        x_vanilla = _input(requires_grad=True)
        x_gram = _input(requires_grad=True)
        self._backward(vanilla, x_vanilla)
        self._backward(gram, x_gram)

        core = _core_params(gram)
        for name, p in vanilla.named_parameters():
            assert (p.grad is None) == (core[name].grad is None), f"{name} grad presence differs"
            if p.grad is not None:
                assert torch.equal(p.grad, core[name].grad), f"{name} gradient differs"
        assert torch.equal(x_vanilla.grad, x_gram.grad), "input gradient differs"

    def test_aux_params_receive_non_zero_grads_at_gate_one(self):
        """The complement: gating is what zeroes them, not a detached graph."""
        _, gram = _pair()
        with torch.no_grad():
            gram.gr_aux.linear_fc2.weight.normal_(0.0, 0.05)
        gram.gr_gate.fill_(1.0)
        self._backward(gram, _input(requires_grad=True))
        for name in ("gr_aux.linear_fc1.weight", "gr_aux.linear_fc2.weight"):
            grad = dict(gram.named_parameters())[name].grad
            assert grad is not None and not torch.all(grad == 0), f"{name} got no usable gradient at gate 1"

    def test_gate_buffer_is_not_a_parameter(self):
        _, gram = _pair()
        assert gram.gr_gate.requires_grad is False
        assert "gr_gate" not in dict(gram.named_parameters())


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestConstructionRefusals:
    """Every shape assumption the export merge rests on is enforced at construction."""

    def test_split_execution_forward_raises(self):
        """intermediate_tensors is the overlap_moe_expert_parallel_comm / CUDA-graph path,
        which the aux add does not implement — refuse rather than silently drop the addend."""
        _, gram = _pair()
        with pytest.raises(NotImplementedError, match="split-execution forward"):
            gram(_input(), intermediate_tensors=object())

    @pytest.mark.parametrize("field, match", [("gated_linear_unit", "non-gated"), ("add_bias_linear", "bias-free")])
    def test_unsupported_activation_shape_raises(self, field, match):
        from megatron.core.transformer.mlp import MLPSubmodules

        from megatron.bridge.models.mamba.gram_layer import GRAMAuxMLP

        submodules = moe_builder(stack_spec()).keywords["submodules"].shared_experts.keywords["submodules"]
        assert isinstance(submodules, MLPSubmodules)
        with pytest.raises(ValueError, match=match):
            GRAMAuxMLP(config=moe_config(**{field: True}), submodules=submodules, aux_ffn_hidden_size=AUX_FFN)

    def test_layer_without_shared_experts_raises(self):
        """The aux module mirrors the shared expert and both export postures merge into it."""
        with pytest.raises(ValueError, match="requires a shared-expert MoE layer"):
            build_moe_layer(moe_builder(gram_spec()), moe_config(moe_shared_expert_intermediate_size=None))

    def test_shared_experts_builder_without_submodules_raises(self):
        """A shared-expert builder that carries no ``submodules`` keyword gives the aux
        module nothing to mirror; the message must say so rather than AttributeError.

        The stand-in is a plain function rather than a partial without the keyword: MoELayer
        itself calls the builder before GRAMMoELayer's check runs, so a builder missing
        ``submodules`` outright dies inside MoELayer. What this pins is the case the check
        actually catches — a builder that constructs fine but hides its submodules.
        """
        spec = gram_spec()
        moe_builder(spec).keywords["submodules"].shared_experts = _opaque_shared_expert_builder
        with pytest.raises(ValueError, match="could not extract MLPSubmodules"):
            build_moe_layer(moe_builder(spec), moe_config())


class TestSwapSpec:
    """The spec transform mirrors swap_moe_experts_to_grouped, including its deepcopy."""

    def test_swaps_only_the_moe_layer_class(self):
        from megatron.bridge.models.mamba.gram_layer import GRAMMoELayer

        original_submodules = moe_builder(stack_spec()).keywords["submodules"]
        swapped = moe_builder(gram_spec())

        assert swapped.func is GRAMMoELayer
        assert swapped.keywords["gr_aux_ffn_hidden_size"] == AUX_FFN
        assert set(swapped.keywords) == {"submodules", "gr_aux_ffn_hidden_size"}
        # router / experts / shared experts travel through unchanged
        swapped_submodules = swapped.keywords["submodules"]
        assert swapped_submodules.router is original_submodules.router
        assert _render(swapped_submodules.experts) == _render(original_submodules.experts)
        assert _render(swapped_submodules.shared_experts) == _render(original_submodules.shared_experts)

    def test_swap_does_not_mutate_the_caller_spec(self):
        """``mamba_stack_spec`` hands out one process-global object; an in-place swap would
        rewrite the spec for every model built afterwards in the process."""
        from megatron.core.transformer.moe.moe_layer import MoELayer

        from megatron.bridge.models.mamba.gram_layer import GRAMMoELayer, swap_moe_layer_to_gram

        original = stack_spec()
        original_builder = moe_builder(original)
        swap_moe_layer_to_gram(original, aux_ffn_hidden_size=AUX_FFN)
        assert moe_builder(original) is original_builder
        assert original_builder.func is MoELayer
        assert original_builder.func is not GRAMMoELayer

    def test_double_apply_raises(self):
        from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram

        with pytest.raises(ValueError, match="already applied"):
            swap_moe_layer_to_gram(gram_spec(), aux_ffn_hidden_size=AUX_FFN)

    @pytest.mark.parametrize("aux_ffn", [0, -1, -512])
    def test_non_positive_aux_width_raises(self, aux_ffn):
        from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram

        with pytest.raises(ValueError, match="aux_ffn_hidden_size must be positive"):
            swap_moe_layer_to_gram(stack_spec(), aux_ffn_hidden_size=aux_ffn)

    def test_non_moe_mlp_builder_raises(self):
        """Guards against applying the swap to an already-rewritten spec, where the silent
        alternative is a GRAMMoELayer wrapping the wrong class."""
        from megatron.core.transformer.mlp import MLP

        from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram

        spec = stack_spec()
        spec.submodules.moe_layer.submodules.mlp = partial(MLP)
        with pytest.raises(ValueError, match="Expected moe_layer.submodules.mlp to be partial"):
            swap_moe_layer_to_gram(spec, aux_ffn_hidden_size=AUX_FFN)


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestComposesWithGroupedExperts:
    """Both swaps edit the same mlp partial — one its func, one its submodules — so the
    order must not matter. Every real GR run applies both."""

    @pytest.fixture(autouse=True)
    def _require_torch_grouped(self):
        if not hasattr(torch, "_grouped_mm"):
            pytest.skip(f"gemm_backend='torch_grouped' needs torch._grouped_mm, absent from torch {torch.__version__}")

    def _composed(self, gram_first):
        from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram
        from megatron.bridge.models.mamba.grouped_experts import swap_moe_experts_to_grouped

        spec = stack_spec()
        order = (
            (swap_moe_layer_to_gram, swap_moe_experts_to_grouped)
            if gram_first
            else (swap_moe_experts_to_grouped, swap_moe_layer_to_gram)
        )
        for transform in order:
            if transform is swap_moe_layer_to_gram:
                spec = transform(spec, aux_ffn_hidden_size=AUX_FFN)
            else:
                spec = transform(spec, gemm_backend="torch_grouped")
        return spec

    @pytest.mark.parametrize("gram_first", [True, False])
    def test_both_orders_build_the_same_module_types(self, gram_first):
        from megatron.bridge.models.mamba.gram_layer import GRAMMoELayer
        from megatron.bridge.models.mamba.grouped_experts import GroupedExperts

        builder = moe_builder(self._composed(gram_first))
        assert builder.func is GRAMMoELayer
        assert builder.keywords["submodules"].experts.func is GroupedExperts
        layer = build_moe_layer(builder, moe_config())
        assert isinstance(layer.experts, GroupedExperts)
        assert hasattr(layer, "gr_aux")

    @pytest.mark.parametrize("gram_first", [True, False])
    def test_composed_layer_runs_and_is_identity_at_gate_zero(self, gram_first):
        from megatron.bridge.models.mamba.grouped_experts import swap_moe_experts_to_grouped

        config = moe_config()
        grouped_only = build_moe_layer(
            moe_builder(swap_moe_experts_to_grouped(stack_spec(), gemm_backend="torch_grouped")), config
        )
        composed = build_moe_layer(moe_builder(self._composed(gram_first)), config)
        x = _input()
        assert torch.equal(grouped_only(x)[0], composed(x)[0])
